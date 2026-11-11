"""Splitting a stiff system into one-way-coupled blocks, from its sparsity pattern.

A Jacobian pattern read as a graph -- row ``i`` depends on column ``j`` --
has strongly connected components, and their condensation is a DAG. A
component that no later component feeds back into is a *driver*: its
equations can be integrated on their own, and everything downstream can read
it as a known function of time. The thermal history under an
Einstein-Boltzmann hierarchy is the case this was built for (five states that
every photon multipole reads and that read nothing back), but nothing here
knows that: the split is found from the pattern alone.

``Hierarchy.from_sparsity`` (or ``split=True`` on :func:`modax.rodas5P.solve`)
collects the components that read only
components already collected into one driver block, as long as it stays
small, and takes the rest as the main block. The two are integrated by the
hierarchy kernel of :mod:`modax._hierarchy` as a two-block
:class:`~modax._hierarchy.Hierarchy`: the driver publishes its states, the
main block reads them as a quintic Hermite interpolant of ``t``, each on
steps of its own, with the driver's error handed down. Both block right-hand
sides are generated from the user's single ``ode_fn``: the driver's returns
its rows with the main states set to zero (dead code after inlining), the
main's returns its rows with the driver states taken from the interpolant.
The history, the hook's state and the error weights all keep the user's
order.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from modax._codegen import compile_device_source
from modax._hierarchy import quintic_read_lines
from modax._numba_common import as_cuda_device

# --- the pattern's blocks ---------------------------------------------------------------


@dataclass(frozen=True)
class BlockSplit:
    """Which state indices (the user's) form the driver block and the main block."""

    driver: tuple[int, ...]
    main: tuple[int, ...]
    levels: tuple[
        tuple[int, ...], ...
    ]  # the driver's components, in the order they joined

    @property
    def n_drv(self) -> int:
        return len(self.driver)

    @property
    def n_main(self) -> int:
        return len(self.main)


def components_in_order(
    pattern: tuple[tuple[int, ...], ...],
) -> list[tuple[tuple[int, ...], frozenset[int]]]:
    """The pattern's strongly connected components in topological order, each with the states it reads.

    A pair ``(members, reads)`` per component: ``members`` its state indices,
    ``reads`` the indices outside it that its rows depend on. Nothing reads a
    component listed after it.
    """
    import networkx as nx

    n = len(pattern)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for row, cols in enumerate(pattern):
        for col in cols:
            if col != row:
                graph.add_edge(col, row)  # row depends on col
    condensation = nx.condensation(graph)
    ordered = []
    for node in nx.topological_sort(condensation):
        members = tuple(sorted(condensation.nodes[node]["members"]))
        reads = frozenset(c for i in members for c in pattern[i]) - frozenset(members)
        ordered.append((members, reads))
    return ordered


def analyse_blocks(
    pattern: tuple[tuple[int, ...], ...], max_driver: int
) -> BlockSplit | None:
    """The driver and main blocks of a pattern, or ``None`` when it does not split.

    The strongly connected components are taken in topological order, and
    each joins the driver when everything it reads is already there and the
    driver stays within ``max_driver`` states, short of the whole system; the
    rest is the main block. So
    the driver is closed under "depends on", whatever the rows the main block
    does not feed (an optical depth that only integrates the driver's opacity
    joins it too), and a superset pattern, which can only merge components,
    is safe: at worst nothing splits.
    """
    components = components_in_order(pattern)
    driver: set[int] = set()
    levels: list[tuple[int, ...]] = []
    for members, reads in components:
        size = len(driver) + len(members)
        if reads <= driver and size <= max_driver and size < len(pattern):
            driver.update(members)
            levels.append(members)
    if not driver:
        return None
    main = sorted(set(range(len(pattern))) - driver)
    return BlockSplit(tuple(sorted(driver)), tuple(main), tuple(levels))


@dataclass(frozen=True, eq=False)
class SplitHierarchy:
    """A system split along its sparsity pattern, from :meth:`Hierarchy.from_sparsity <modax.Hierarchy.from_sparsity>`.

    :func:`modax.solve` integrates it as two blocks: the *driver*, the leading
    states nothing else feeds back into, on steps of its own, and the *main*
    block, which reads the driver as a quintic Hermite interpolant of ``t``.
    Both right-hand sides are generated from the one ``ode_fn``, and the
    state keeps the user's order: ``driver`` and ``main`` are indices into it.
    """

    ode_fn: object
    n: int
    sparsity: tuple[tuple[int, ...], ...]
    max_driver: int
    split: BlockSplit

    @property
    def n_vars(self) -> int:
        return self.n

    @property
    def driver(self) -> tuple[int, ...]:
        return self.split.driver

    @property
    def main(self) -> tuple[int, ...]:
        return self.split.main

    def __hash__(self) -> int:
        return hash((id(self.ode_fn), self.sparsity, self.max_driver))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SplitHierarchy)
            and other.ode_fn is self.ode_fn
            and other.sparsity == self.sparsity
            and other.max_driver == self.max_driver
        )


def restrict_pattern(pattern, indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """The pattern's rows ``indices`` on columns ``indices``, renumbered."""
    position = {idx: k for k, idx in enumerate(indices)}
    return tuple(
        tuple(sorted(position[c] for c in pattern[i] if c in position)) for i in indices
    )


# --- the generated right-hand sides -------------------------------------------------


def generate_driver_rhs(ode_fn, n_vars: int, split: BlockSplit):
    """``g(theta, t, p) -> tuple`` of the driver rows; the main states are zero and dead."""
    pos = {idx: k for k, idx in enumerate(split.driver)}
    y = ", ".join(f"th[{pos[i]}]" if i in pos else "0.0" for i in range(n_vars))
    lines = [
        "def driver_rhs(th, t, p):",
        f"    y = ({y},)",
        "    f = ode_fn(y, t, p)",
        "    return (" + ", ".join(f"f[{i}]" for i in split.driver) + ",)",
    ]
    # compiled on its own (numba copies an inlined callee's IR at every call
    # site), except that numba-cuda-mlir cannot return a one-tuple from a called
    # device function
    return compile_device_source(
        "driver_rhs",
        lines,
        {"ode_fn": as_cuda_device(ode_fn)},
        inline="never" if split.n_drv > 1 else "always",
    )


def generate_main_rhs(ode_fn, n_vars: int, split: BlockSplit, hslot: int):
    """``f(y_main, t, p) -> tuple`` of the main rows, the driver states from the quintic Hermite block of ``p``.

    ``hslot`` is where the driver's block starts in the local parameter row
    (:func:`modax._hierarchy.hermite_slot`); reading it here rather than in a
    generated wrapper keeps the user's right-hand side inlined once.
    """
    dpos = {idx: k for k, idx in enumerate(split.driver)}
    mpos = {idx: k for k, idx in enumerate(split.main)}
    lines = [
        "def main_rhs(ym, t, p):",
        *quintic_read_lines(hslot, [f"d{j}" for j in range(split.n_drv)], ""),
    ]
    y = ", ".join(
        f"d{dpos[i]}" if i in dpos else f"ym[{mpos[i]}]" for i in range(n_vars)
    )
    lines += [
        f"    y = ({y},)",
        "    f = ode_fn(y, t, p)",
        "    return (" + ", ".join(f"f[{i}]" for i in split.main) + ",)",
    ]
    return compile_device_source(
        "main_rhs",
        lines,
        {"ode_fn": as_cuda_device(ode_fn)},
        inline="never" if split.n_main > 1 else "always",
    )


# --- the kernel -------------------------------------------------------------------------


@functools.cache
def make_split_kernel(ode_fn, n_vars: int, n_params: int, options, split: BlockSplit):
    """The hierarchy kernel over the pattern's driver and main blocks, in the user's layout.

    ``options`` is the :class:`modax.rodas5P.KernelOptions`. Returns
    ``(kernel, trajectories_per_block)``.
    """
    from modax._hierarchy import Block, Hierarchy, make_hierarchy_kernel

    pattern = options.sparsity
    driver = Block(
        generate_driver_rhs(ode_fn, n_vars, split), split.n_drv, name="driver"
    )
    # the driver's Hermite block is the first after the user's parameters
    main_rhs = generate_main_rhs(ode_fn, n_vars, split, hslot=n_params)
    main = Block(
        main_rhs, split.n_main, reads=(driver,), name="main", reads_from_params=True
    )
    patterns = None
    if pattern is not None:
        patterns = (
            restrict_pattern(pattern, split.driver),
            restrict_pattern(pattern, split.main),
        )
    return make_hierarchy_kernel(
        Hierarchy(main),
        n_params,
        options,
        layout=(split.driver, split.main),
        patterns=patterns,
    )
