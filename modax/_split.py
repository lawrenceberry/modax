"""Split a stiff system into one-way-coupled blocks using its sparsity pattern.

Read as a graph, with row ``i`` depending on column ``j``, a Jacobian pattern
has strongly connected components whose condensation is a DAG. A component
that nothing later feeds back into is a *driver*. Its equations integrate on
their own, and everything downstream can read it as a known function of
time. The thermal history under an Einstein-Boltzmann hierarchy motivated
this, but the split comes from the pattern alone.

``Hierarchy.from_sparsity`` gathers the components that read only components
already gathered into one driver block, while it stays small, and makes the
rest the main block. The hierarchy kernel of :mod:`modax._hierarchy`
integrates them as a two-block :class:`~modax._hierarchy.Hierarchy`, each on
its own steps. The driver publishes its states, the main block reads them as
a quintic Hermite interpolant of ``t``, and the driver's error passes down. Both right-hand
sides come from the user's single ``ode_fn``. The driver's sets the main
states to zero (dead code after inlining), and the main's takes the driver
states from the interpolant. The history, the hook's state and the error
weights all keep the user's order.
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
    """The user's state indices that form the driver block and the main block."""

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
    """The pattern's strongly connected components in topological order, with the states each reads.

    Returns a pair ``(members, reads)`` per component. ``members`` holds its
    state indices and ``reads`` the indices outside it that its rows depend on.
    No component reads one listed after it.
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

    Walks the strongly connected components in topological order. A component
    joins the driver when everything it reads is already there and the driver
    stays within ``max_driver`` states and smaller than the whole system. The
    rest forms the main block. The driver is therefore closed under "depends
    on", and picks up rows the main block does not feed (an optical depth that
    only integrates the driver's opacity). A superset pattern can only merge
    components, so at worst nothing splits.
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

    :func:`modax.solve` integrates it as two blocks. The *driver* holds the
    leading states that nothing else feeds back into and takes its own steps.
    The *main* block reads the driver as a quintic Hermite interpolant of ``t``.
    Both right-hand sides come from the one ``ode_fn``, and the state keeps the
    user's order, with ``driver`` and ``main`` as indices into it.
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
    """``g(theta, t, p) -> tuple`` of the driver rows, with the main states zero and dead."""
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
    """``f(y_main, t, p) -> tuple`` of the main rows, reading the driver states from the quintic Hermite block of ``p``.

    ``hslot`` is where the driver's block starts in the local parameter row
    (:func:`modax._hierarchy.hermite_slot`). Reading it here rather than in a
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

    ``options`` is the :class:`~modax._options.KernelOptions`. Returns
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
