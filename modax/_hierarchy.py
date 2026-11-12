"""A stiff system as a hierarchy of one-way-coupled blocks, each on its own steps.

A :class:`Block` holds states with a right-hand side of their own. It may read
*quantities* (any functions of the state) of the blocks upstream of it and
publish quantities of its own for the blocks downstream. The blocks form a
directed acyclic graph, so nothing feeds back. The motivating case is the
thermal history under an Einstein-Boltzmann hierarchy (five states that every
photon multipole reads through four quantities), but nothing here knows that.

Splitting what one kernel could integrate whole pays twice:

* **The matrix.** A state every downstream row reads is a full column of the
  joint Jacobian, and a sparse factorisation fills it in. Split, each block
  factorises its own matrix, and a reader's Jacobian has no upstream columns.
* **The steps.** Each block takes the steps its own dynamics allow, under its
  own error norm and controller. A smooth upstream block takes long steps, and
  its right-hand side and published quantities cost one evaluation per
  *upstream* step, not one per stage of every downstream step.

How the blocks talk. Every block runs ahead of its readers and keeps a ring of
*nodes*: at each accepted step's end, its state, rate and second derivative
(``y'' = J y' + df/dt``), plus the values and first two time derivatives of
the quantities it publishes. Between nodes its dense output is the quintic
Hermite interpolant of that data, ``O(H^6)`` over a step ``H``. That is one
order beyond Rodas5P's local error, so a reader loses no order. Before
stepping over ``[t, t + h]`` a reader evaluates the dense output of every block
it reads at both ends and writes the six numbers per quantity into its
thread-local parameter row. Its generated right-hand side evaluates the
quintic there, so its whole step sees the upstream as a polynomial in ``t``.
The ring bounds how far a block runs ahead (:data:`NODES`, fewer with
sensitivities).

Methods. Each block steps with its own method (:mod:`modax._methods`):
Rodas5P, Tsit5, or ``Auto``, which switches between the two on a stiffness
estimate. The nodes look the same whatever the method. A Tsit5 block's rate at
a node is its last stage (first same as last) and its second derivative is one
forward sweep along ``(y', 1)``. Under ``Auto`` the warp, not the lane, picks
each step's method: Rodas5P when any lane about to step wants it.

The warp also decides when a block advances. When any lane's reader needs more
of an upstream block, every lane with room in its ring steps that block too.
Left alone, the lanes would reach their upstream steps at different moments
and the warp would run the upstream code at nearly every downstream step,
which is what a lockstep solve pays.

Error control. Each block accepts its steps by its own weighted RMS norm over
its own rows. A reader also seeds one extra direction of its Jacobian sweeps
with the upstream's error estimate, reads off the rate ``df/dq . eps`` at which
that error enters its rows, and turns it into the error over the upstream
step, damped where a row restores itself faster. A shorter downstream step
cannot reduce this *inherited* error, so it goes to the upstream block's
controller, which shrinks its next step until what it hands down is as
accurate as its own work. Every block's local error, own and inherited, then
meets ``rtol`` and ``atol``.

Forward sensitivities. A block carries the parameter and initial-state
directions of its own states and of its upstream blocks'. Each direction's
parameter seed is the unit vector plus the Hermite coefficients of the
upstream quantities' sensitivities, so ``df/dq S_q(t)`` comes out of the same
forward sweep as ``J_y S_y + J_p``. First- and second-order sweeps of the
quantity function give a published quantity's sensitivity and its rate. Its
second derivative would need a third-order sweep, so it comes instead from the
quintic through the sensitivity's value and rate at both nodes and the step's
midpoint (``O(H^6)`` all the same).
"""

from __future__ import annotations

import functools
import math
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from modax import tsit5 as T
from modax._codegen import compile_device_source
from modax._kernel_cache import enable_content_cache
from modax._methods import (
    TSIT5_STABILITY_SIZE,
    Auto,
    Method,
    Ordering,
    Rodas5P,
    Tsit5,
    method_parts,
    option_or,
)
from modax._numba_common import SAFETY, PIDGains, as_cuda_device
from modax._sensitivity import (
    SensitivitySpec,
    make_quantity_second_tangent,
    make_quantity_tangent,
    make_second_tangent,
    make_tangent,
)
from modax._sparse_direct import sparse_direct_solver_for
from modax._sparsity import (
    Pattern,
    Sparsity,
    colour_sparsity,
    dense_jacobian,
    normalize_sparsity,
)

if TYPE_CHECKING:
    from modax._split import SplitHierarchy

# --- the user's description -----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Block:
    """One block of a hierarchy: its states, their right-hand side, and what it reads and publishes.

    Args:
        rhs: ``rhs(y, t, p) -> tuple`` of the block's ``n`` rates, or
            ``rhs(y, t, p, q)`` when the block reads other blocks. ``q`` is then
            the flat tuple of their quantities, block after block in the order
            of ``reads``.
        n: The number of states. Must be positive.
        reads: The blocks this one reads. A block that two others read still
            integrates once.
        outputs: ``outputs(y, t, p[, q]) -> tuple`` of ``n_outputs`` quantities,
            what the blocks reading this one see. ``None`` publishes the states
            themselves.
        n_outputs: The length of the tuple ``outputs`` returns. Required with
            ``outputs``, and ``n`` (or ``None``) without it.
        sparsity: The pattern of ``rhs`` with respect to the block's own states,
            dense when ``None``.
        name: A label for the block in error messages.
        method: How the block steps: :class:`~modax.Rodas5P`,
            :class:`~modax.Tsit5` or :class:`~modax.Auto`. ``None`` takes the
            solve's.

    Raises:
        ValueError: ``n`` is not positive, or ``n_outputs`` does not match
            ``outputs``.
        TypeError: ``method`` is not a Rodas5P, Tsit5 or Auto.
    """

    rhs: Callable[..., tuple[float, ...]]
    n: int
    reads: tuple[Block, ...] = ()
    outputs: Callable[..., tuple[float, ...]] | None = None
    n_outputs: int | None = None
    sparsity: Sparsity | None = None
    name: str = ""
    # How the block steps (:mod:`modax._methods`): ``Rodas5P()``, ``Tsit5()`` or
    # ``Auto(Tsit5(), Rodas5P())``; ``None`` takes the solve's method.
    method: Method | None = None
    # Internal: ``rhs(y, t, p)`` reads its upstream's Hermite blocks from the
    # local parameter row itself (laid out as :func:`hermite_slot`), so the
    # kernel calls it without a generated wrapper. The pattern split generates
    # such a right-hand side straight from the user's ``ode_fn``.
    reads_from_params: bool = False

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("a block needs at least one state")
        if self.outputs is None and self.n_outputs not in (None, self.n):
            raise ValueError("a block publishing its states has n_outputs = n")
        if self.outputs is not None and not self.n_outputs:
            raise ValueError("a block with an outputs function needs n_outputs")
        if self.method is not None and not isinstance(
            self.method, (Rodas5P, Tsit5, Auto)
        ):
            raise TypeError(
                f"a block's method is Rodas5P, Tsit5 or Auto, got {self.method!r}"
            )
        object.__setattr__(self, "reads", tuple(self.reads))
        if self.sparsity is not None:
            object.__setattr__(
                self, "sparsity", normalize_sparsity(self.sparsity, self.n)
            )

    @property
    def n_quantities(self) -> int:
        return self.n if self.outputs is None else int(self.n_outputs)  # ty: ignore[invalid-argument-type]


@dataclass(frozen=True, eq=False)
class Hierarchy:
    """The blocks a :class:`Block` tree reaches, upstream first, and their place in the state.

    Depth-first search from ``root`` orders the blocks, each after the blocks it
    reads, in the order of their ``reads``. The flat state the solve takes and
    returns holds the blocks' states in that order unless ``layout`` gives
    another. ``offsets`` gives where each of ``blocks`` starts in it.

    Args:
        root: The block nothing reads, the last to step.
        layout: The same blocks in the order their states appear in the flat
            state. ``None`` keeps the search order.

    Raises:
        ValueError: The blocks read each other in a cycle, or ``layout`` does
            not list exactly the blocks ``root`` reaches.
    """

    root: Block
    layout: tuple[Block, ...] | None = None
    blocks: tuple[Block, ...] = field(init=False)
    offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        order: list[Block] = []
        on_stack: set[int] = set()

        def visit(block: Block) -> None:
            if block in order:
                return
            if id(block) in on_stack:
                raise ValueError("the blocks read each other in a cycle")
            on_stack.add(id(block))
            for upstream in block.reads:
                visit(upstream)
            on_stack.discard(id(block))
            order.append(block)

        visit(self.root)
        object.__setattr__(self, "blocks", tuple(order))
        layout = tuple(order) if self.layout is None else tuple(self.layout)
        if sorted(map(id, layout)) != sorted(map(id, order)):
            raise ValueError("layout must list exactly the blocks the root reaches")
        object.__setattr__(self, "layout", layout)
        start, total = {}, 0
        for block in layout:
            start[id(block)] = total
            total += block.n
        object.__setattr__(self, "offsets", tuple(start[id(block)] for block in order))

    @property
    def n_vars(self) -> int:
        return sum(block.n for block in self.blocks)

    @staticmethod
    def from_sparsity(
        ode_fn: Callable[..., tuple[float, ...]],
        n: int,
        sparsity: Sparsity,
        max_driver: int = 16,
    ) -> SplitHierarchy:
        """Split one system into a driver block and a main block along its sparsity pattern.

        Read as a graph, the pattern has strongly connected components in
        topological order. Those that read only components already taken join the
        *driver* while it holds at most ``max_driver`` states. The rest is the
        *main* block, which reads the driver. Pass the result to
        :func:`modax.solve` in place of ``ode_fn``. The driver steps ahead of the
        main block, which reads it as a quintic Hermite interpolant of ``t`` and
        hands the driver's error back to it. Both blocks step with the solve's
        ``method``. The state, the history and the error weights keep ``ode_fn``'s
        order.

        Args:
            ode_fn: The whole system's right-hand side, ``(y, t, p) -> tuple``.
            n: Its number of states.
            sparsity: Its Jacobian's pattern, as :func:`modax.solve` takes it.
            max_driver: The most states the driver may hold.

        Returns:
            A ``SplitHierarchy``, whose ``driver`` and ``main`` give the
            states each block holds.

        Raises:
            ValueError: The pattern has no one-way-coupled leading block within
                ``max_driver`` states.
        """
        from modax._split import SplitHierarchy, analyse_blocks

        pattern = normalize_sparsity(sparsity, n)
        split = analyse_blocks(pattern, max_driver)
        if split is None:
            raise ValueError(
                "the pattern has no leading block of at most "
                f"{max_driver} states that nothing else feeds back into"
            )
        return SplitHierarchy(ode_fn, n, pattern, max_driver, split)

    def index(self, block: Block) -> int:
        return self.blocks.index(block)

    def __hash__(self) -> int:
        return id(self.root)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hierarchy) and other.root is self.root


@functools.cache
def _hierarchy_of(root: Block) -> Hierarchy:
    return Hierarchy(root)


def as_hierarchy(ode_fn: object) -> Hierarchy | None:
    """``ode_fn`` as a hierarchy when it is a :class:`Block` or one, else ``None``."""
    if isinstance(ode_fn, Hierarchy):
        return ode_fn
    if isinstance(ode_fn, Block):
        return _hierarchy_of(ode_fn)
    return None


# --- the plan: per-block layout and bookkeeping ------------------------------------

NODES = 12
"""Dense-output nodes a block keeps. A reader's step must end at or before the
last of them, so this bounds how far a block runs ahead and the local memory
it takes. With sensitivities on, a block keeps :data:`NODES_SENS`.
"""

NODES_SENS = 4

QUINTIC_LINES = [
    "    s2 = s * s",
    "    s3 = s2 * s",
    "    s4 = s3 * s",
    "    s5 = s4 * s",
    "    b0 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5",
    "    b1 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5",
    "    b2 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5)",
    "    b3 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5",
    "    b4 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5",
    "    b5 = 0.5 * (s3 - 2.0 * s4 + s5)",
]


def _midpoint_curvatures() -> tuple[np.ndarray, np.ndarray]:
    """The end curvatures of the quintic through value and slope at 0, 1/2 and 1.

    Returns ``(c_left, c_right)`` with ``P''(0) = c_left . d`` and ``P''(1) =
    c_right . d`` for ``d = (P(0), P'(0), P(1/2), P'(1/2), P(1), P'(1))`` on the
    unit interval. In ``t``, scale the slopes by ``H`` and the curvatures by
    ``1/H^2``.
    """
    powers = np.arange(6)
    rows = []
    for s in (0.0, 0.5, 1.0):
        rows.append(s**powers)
        rows.append(np.where(powers > 0, powers * s ** np.maximum(powers - 1, 0), 0.0))
    coefficients = np.linalg.inv(np.array(rows))  # d -> monomial coefficients c

    def second(s: float) -> np.ndarray:
        return np.where(
            powers > 1, powers * (powers - 1) * s ** np.maximum(powers - 2, 0), 0.0
        )

    return second(0.0) @ coefficients, second(1.0) @ coefficients


_CURV_LEFT, _CURV_RIGHT = _midpoint_curvatures()


@dataclass(frozen=True)
class _BlockPlan:
    """Everything the kernel builder knows about one block."""

    b: int
    block: Block
    idx: tuple[int, ...]  # the user's indices of its states
    reads: tuple[int, ...]
    readers: tuple[int, ...]
    dirs: tuple[int, ...]  # the user's direction indices it carries
    pos: tuple[int, ...]  # user direction -> its own, or -1
    hslot: int  # where its Hermite block starts in the local parameter row, -1 if nothing reads it
    pattern: Pattern | None

    @property
    def n(self) -> int:
        return self.block.n

    @property
    def m(self) -> int:
        """Quantities it publishes (0 when nothing reads it)."""
        return self.block.n_quantities if self.readers else 0

    @property
    def identity(self) -> bool:
        return self.block.outputs is None

    @property
    def ns(self) -> int:
        return len(self.dirs)

    @property
    def size(self) -> int:
        return self.n * (1 + self.ns)

    @property
    def mq(self) -> int:
        return self.m * (1 + self.ns)

    @property
    def sink(self) -> bool:
        return not self.readers


@dataclass(frozen=True)
class _Plan:
    blocks: tuple[_BlockPlan, ...]
    n_vars: int
    n_params: int
    p_local: int  # the local parameter row's length
    n_sens: int
    dir_param_col: tuple[int, ...]

    def hoff(self, b: int, w: int) -> int:
        """Where upstream ``w``'s coefficients start in reader ``b``'s hseed rows."""
        off = 0
        for u in self.blocks[b].reads:
            if u == w:
                return off
            off += 6 * self.blocks[u].m
        raise KeyError(w)

    def hcols(self, b: int) -> int:
        return sum(6 * self.blocks[u].m for u in self.blocks[b].reads)


@dataclass(frozen=True)
class _Method:
    """A block's method resolved against the solve: which steps it takes, with what options."""

    kind: str  # "rodas", "tsit5" or "auto"
    lu_fp32: bool  # the Rodas5P factorisation in single precision
    rodas_gains: PIDGains
    tsit_gains: PIDGains
    auto: Auto | None = None
    ordering: Ordering = "amd"  # the sparse LU's fill-reducing permutation

    @property
    def implicit(self) -> bool:
        return self.kind in ("rodas", "auto")

    @property
    def explicit(self) -> bool:
        return self.kind in ("tsit5", "auto")

    @property
    def switch(self) -> Auto:
        """The ``Auto`` options of a switching block."""
        assert self.auto is not None, "only an Auto block switches"
        return self.auto


def resolve_method(bp: _BlockPlan, options) -> _Method:
    """``bp``'s method: its own, else the solve's ``default_method``, else Rodas5P, with ``None`` options taken from the solve."""
    method = bp.block.method or getattr(options, "default_method", None) or Rodas5P()
    rodas, tsit = method_parts(method)
    kind = (
        "auto"
        if isinstance(method, Auto)
        else ("rodas" if rodas is not None else "tsit5")
    )
    name = bp.block.name or f"block {bp.b}"
    if kind != "rodas" and bp.ns > 0:
        raise NotImplementedError(
            f"{name}: forward sensitivities through an explicit ({type(method).__name__}) block are not implemented"
        )
    # Only the block's own method may ask an upstream block for fp32: the
    # solve's precision applies to the final block alone.
    own, _ = method_parts(bp.block.method)
    lu = option_or(own, "lu_precision", None)
    if bp.sink:
        lu_fp32 = (lu or options.lu_precision) == "fp32"
    else:
        if lu == "fp32":
            raise ValueError(
                f"{name}: a block others read factorises in double precision (its node Jacobian is reused); "
                "lu_precision='fp32' applies to the final block only"
            )
        lu_fp32 = False
    return _Method(
        kind=kind,
        lu_fp32=lu_fp32,
        rodas_gains=option_or(rodas, "gains", options.gains),
        tsit_gains=option_or(tsit, "gains", options.gains),
        auto=method if isinstance(method, Auto) else None,
        ordering=option_or(rodas, "ordering", options.ordering),
    )


def _table(seq) -> np.ndarray:
    """``seq`` as an int32 index table, ``(-1,)`` when empty."""
    return np.asarray(seq or (-1,), dtype=np.int32)


def hermite_slot(hslot: int, j: int) -> int:
    """First of the six Hermite coefficients of quantity ``j`` in the Hermite block at ``hslot``."""
    return hslot + 2 + 6 * j


def plan_hierarchy(
    hierarchy: Hierarchy,
    n_params: int,
    spec: SensitivitySpec | None,
    layout: tuple[tuple[int, ...], ...] | None = None,
    patterns: tuple[Pattern | None, ...] | None = None,
) -> _Plan:
    """Lay the blocks out: indices, directions, Hermite slots.

    ``layout`` overrides where each block's states sit in the user's state (the
    pattern split keeps the user's order). ``patterns`` overrides the blocks'
    own sparsity.
    """
    blocks = hierarchy.blocks
    n_vars = hierarchy.n_vars
    index = {id(block): b for b, block in enumerate(blocks)}
    reads = tuple(tuple(index[id(u)] for u in block.reads) for block in blocks)
    readers: list[list[int]] = [[] for _ in blocks]
    for b, ups in enumerate(reads):
        for u in ups:
            readers[u].append(b)
    if layout is None:
        layout = tuple(
            tuple(range(off, off + block.n))
            for off, block in zip(hierarchy.offsets, blocks)
        )
    if patterns is None:
        # __post_init__ has normalised each block's sparsity to a Pattern.
        patterns = tuple(cast(Pattern | None, block.sparsity) for block in blocks)
    # directions: the user's initial-state block then the parameter block
    n_sens = spec.n_sens if spec is not None else 0
    n_y0 = spec.n_y0_dirs if spec is not None else 0
    param_cols = (
        tuple(spec.param_seed_columns) if (spec is not None and spec.wrt_params) else ()
    )
    dir_param_col = tuple([-1] * n_y0 + list(param_cols))
    owner = {}  # user state -> block
    for b, idx in enumerate(layout):
        for i in idx:
            owner[i] = b
    upstream_closure: list[set[int]] = []
    for b in range(len(blocks)):
        closure = {b}
        for u in reads[b]:
            closure |= upstream_closure[u]
        upstream_closure.append(closure)
    # Hermite slots, block after block among those something reads
    hslots, p_local = [], n_params
    for b, block in enumerate(blocks):
        if readers[b]:
            hslots.append(p_local)
            p_local += 2 + 6 * block.n_quantities
        else:
            hslots.append(-1)
    plans = []
    for b, block in enumerate(blocks):
        dirs = [
            k for k in range(n_sens) if k >= n_y0 or owner[k] in upstream_closure[b]
        ]
        pos = [-1] * n_sens
        for kk, k in enumerate(dirs):
            pos[k] = kk
        plans.append(
            _BlockPlan(
                b=b,
                block=block,
                idx=tuple(layout[b]),
                reads=reads[b],
                readers=tuple(readers[b]),
                dirs=tuple(dirs),
                pos=tuple(pos),
                hslot=hslots[b],
                pattern=patterns[b],
            )
        )
    return _Plan(tuple(plans), n_vars, n_params, p_local, n_sens, dir_param_col)


# --- generated wrappers: the upstream quantities as polynomials of t ------------------


def quintic_read_lines(hslot: int, names: list[str], suffix: str) -> list[str]:
    """Lines evaluating the quantities of the Hermite block at ``hslot`` of ``p`` into ``names``."""
    hh, hh2 = f"hh{suffix}", f"hh2{suffix}"
    lines = [
        f"    {hh} = p[{hslot + 1}]",
        f"    s = (t - p[{hslot}]) / {hh}",
        *QUINTIC_LINES,
        f"    {hh2} = {hh} * {hh}",
    ]
    for j, name in enumerate(names):
        c = hermite_slot(hslot, j)
        lines.append(
            f"    {name} = b0 * p[{c}] + {hh} * b1 * p[{c + 1}] + {hh2} * b2 * p[{c + 2}]"
            f" + b3 * p[{c + 3}] + {hh} * b4 * p[{c + 4}] + {hh2} * b5 * p[{c + 5}]"
        )
    return lines


def _quintic_reads(plan: _Plan, bp: _BlockPlan) -> tuple[list[str], str]:
    """Lines evaluating every upstream quantity from ``p``, and the tuple expression ``q``."""
    lines: list[str] = []
    names: list[str] = []
    for u in bp.reads:
        up = plan.blocks[u]
        values = [f"q_{u}_{j}" for j in range(up.m)]
        lines += quintic_read_lines(up.hslot, values, f"_{u}")
        names += values
    return lines, "(" + ", ".join(names) + ",)"


def _wrap(
    plan: _Plan, bp: _BlockPlan, name: str, user: str, fn, width: int
) -> MLIRDispatcher:
    """``{name}_b(y, t, p)``: ``fn`` with the block's upstream read from ``p``."""
    reads, q = _quintic_reads(plan, bp)
    lines = [
        f"def {name}_{bp.b}(y, t, p):",
        *reads,
        f"    q = {q}",
        f"    return {user}(y, t, p, q)",
    ]
    # numba-cuda-mlir cannot return a one-tuple from a called device function
    return compile_device_source(
        f"{name}_{bp.b}",
        lines,
        {user: as_cuda_device(fn)},
        inline="never" if width > 1 else "always",
    )


def _wrap_rhs(plan: _Plan, bp: _BlockPlan) -> MLIRDispatcher:
    """``rhs_b(y, t, p)``: the block's right-hand side with its upstream read from ``p``."""
    if not bp.reads or bp.block.reads_from_params:
        return as_cuda_device(bp.block.rhs)
    return _wrap(plan, bp, "rhs", "user_rhs", bp.block.rhs, bp.n)


def _wrap_outputs(plan: _Plan, bp: _BlockPlan) -> MLIRDispatcher | None:
    """``out_b(y, t, p)``: the block's published quantities with its upstream read from ``p``."""
    if bp.identity or bp.m == 0:
        return None
    if not bp.reads:
        return as_cuda_device(bp.block.outputs)
    return _wrap(plan, bp, "out", "user_outputs", bp.block.outputs, bp.m)


# --- shared device functions ------------------------------------------------------


@cuda.jit(device=True)
def quintic(c0, c1, c2, c3, c4, c5, s, h):
    """The quintic Hermite interpolant of ``(c0, c1, c2)`` at 0 and ``(c3, c4, c5)`` at ``h``, at ``s = (t - t0)/h``."""
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    return (
        (1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5) * c0
        + h * (s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5) * c1
        + h * h * 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5) * c2
        + (10.0 * s3 - 15.0 * s4 + 6.0 * s5) * c3
        + h * (-4.0 * s3 + 7.0 * s4 - 3.0 * s5) * c4
        + h * h * 0.5 * (s3 - 2.0 * s4 + s5) * c5
    )


@cuda.jit(device=True)
def quintic_d1(c0, c1, c2, c3, c4, c5, s, h):
    """The interpolant's first derivative in ``t`` at ``s``."""
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    return (
        (-30.0 * s2 + 60.0 * s3 - 30.0 * s4) * c0
        + h * (1.0 - 18.0 * s2 + 32.0 * s3 - 15.0 * s4) * c1
        + h * h * 0.5 * (2.0 * s - 9.0 * s2 + 12.0 * s3 - 5.0 * s4) * c2
        + (30.0 * s2 - 60.0 * s3 + 30.0 * s4) * c3
        + h * (-12.0 * s2 + 28.0 * s3 - 15.0 * s4) * c4
        + h * h * 0.5 * (3.0 * s2 - 8.0 * s3 + 5.0 * s4) * c5
    ) / h


@cuda.jit(device=True)
def quintic_d2(c0, c1, c2, c3, c4, c5, s, h):
    """The interpolant's second derivative in ``t`` at ``s``."""
    s2 = s * s
    s3 = s2 * s
    return (
        (-60.0 * s + 180.0 * s2 - 120.0 * s3) * c0
        + h * (-36.0 * s + 96.0 * s2 - 60.0 * s3) * c1
        + h * h * 0.5 * (2.0 - 18.0 * s + 36.0 * s2 - 20.0 * s3) * c2
        + (60.0 * s - 180.0 * s2 + 120.0 * s3) * c3
        + h * (-24.0 * s + 84.0 * s2 - 60.0 * s3) * c4
        + h * h * 0.5 * (6.0 * s - 24.0 * s2 + 20.0 * s3) * c5
    ) / (h * h)


@functools.cache
def dense_eval_fn(width: int) -> MLIRDispatcher:
    """A block's dense output at ``t_eval``: every row's value, rate and second derivative.

    It evaluates the quintic between the last node at or before ``t_eval`` and
    the next, built from the value and rate at both nodes, the curvature at the
    piece's start (``node_d2r``) and at its end (``node_d2l``). At a node it
    reproduces the node's own data.
    """

    @cuda.jit(device=True)
    def dense_eval(
        node_t,
        node_v,
        node_d1,
        node_d2r,
        node_d2l,
        n_nodes,
        t_eval,
        out_v,
        out_d1,
        out_d2,
    ):
        q = 0
        for r in range(n_nodes - 1):
            if node_t[r] <= t_eval:
                q = r
        h = node_t[q + 1] - node_t[q]
        s = (t_eval - node_t[q]) / h
        for j in range(width):
            c0 = node_v[q, j]
            c1 = node_d1[q, j]
            c2 = node_d2r[q, j]
            c3 = node_v[q + 1, j]
            c4 = node_d1[q + 1, j]
            c5 = node_d2l[q + 1, j]
            out_v[j] = quintic(c0, c1, c2, c3, c4, c5, s, h)
            out_d1[j] = quintic_d1(c0, c1, c2, c3, c4, c5, s, h)
            out_d2[j] = quintic_d2(c0, c1, c2, c3, c4, c5, s, h)

    return dense_eval


@functools.cache
def midpoint_eval_fn(width: int) -> MLIRDispatcher:
    """The quintic Hermite of ``(v0, d0, c0)`` and ``(v1, d1, c1)`` over ``h`` at its midpoint: values and rates."""

    @cuda.jit(device=True)
    def midpoint_eval(v0, d0, c0, v1, d1, c1, h, out_v, out_d1):
        for j in range(width):
            out_v[j] = quintic(v0[j], d0[j], c0[j], v1[j], d1[j], c1[j], 0.5, h)
            out_d1[j] = quintic_d1(v0[j], d0[j], c0[j], v1[j], d1[j], c1[j], 0.5, h)

    return midpoint_eval


# --- per-block device functions ------------------------------------------------------


def _rows(coeff_rows):
    return tuple(tuple(float(c) for c in row) for row in coeff_rows)


def _dense_or_pattern(bp: _BlockPlan):
    """The block's sparsity pattern, or every column of every row when it has none."""
    n = bp.n
    if bp.pattern is not None:
        return bp.pattern
    return tuple(tuple(range(n)) for _ in range(n))


def _jacobian_times(bp: _BlockPlan, structure) -> MLIRDispatcher:
    """``out = dT - (-J) g`` from the block's negated Jacobian in its compressed slots."""
    pattern = _dense_or_pattern(bp)
    lines = ["def jacobian_times(neg_a, g, dT, out):"]
    for r in range(bp.n):
        terms = " + ".join(
            f"neg_a[{structure.slot(r, c)}] * g[{c}]" for c in pattern[r]
        )
        lines.append(f"    out[{r}] = dT[{r}]" + (f" - ({terms})" if terms else ""))
    return compile_device_source("jacobian_times", lines, {})


def _jacobian_inf_norm(bp: _BlockPlan, structure) -> MLIRDispatcher:
    """``||J||_inf`` from the block's negated Jacobian in its slots: its largest absolute row sum."""
    pattern = _dense_or_pattern(bp)
    lines = ["def jacobian_inf_norm(neg_a):", "    largest = 0.0"]
    for r in range(bp.n):
        if not pattern[r]:
            continue
        terms = " + ".join(
            f"math.fabs(np.float64(neg_a[{structure.slot(r, c)}]))" for c in pattern[r]
        )
        lines += [
            f"    row = {terms}",
            "    if row > largest:",
            "        largest = row",
        ]
    lines.append("    return largest")
    return compile_device_source("jacobian_inf_norm", lines, {"math": math, "np": np})


def _explicit_functions(plan: _Plan, bp: _BlockPlan, write, tangent) -> dict[str, Any]:
    """The device functions of a block's Tsit5 steps (no directions: ``bp.ns == 0``).

    ``k`` is ``(7, n)``. Row 0 holds ``f(t, y)`` on entry (first same as last:
    the previous step's row 6, or the node's rate), rows 1..6 the stages, and
    row 6 the rate at the candidate ``u``. As in Rodas5P, the step is one loop
    over the stages that calls the right-hand side in one place.
    """
    b, n = bp.b, bp.n
    P_LOCAL = plan.p_local
    n_reads = len(bp.reads)
    rows = _rows(
        (
            (T.A21,),
            (T.A31, T.A32),
            (T.A41, T.A42, T.A43),
            (T.A51, T.A52, T.A53, T.A54),
            (T.A61, T.A62, T.A63, T.A64, T.A65),
            (T.B1, T.B2, T.B3, T.B4, T.B5, T.B6),
        )
    )
    c_t = np.asarray((T.C2, T.C3, T.C4, T.C5, T.C6, T.C7))

    def make_combine(s):
        row = rows[s - 1]

        @cuda.jit(device=True)
        def combine(u, y, k, h):
            for j in range(n):
                acc = row[0] * k[0, j]
                for r in range(1, s):
                    acc += row[r] * k[r, j]
                u[j] = y[j] + h * acc

        return combine

    c1, c2, c3, c4, c5, c6 = (make_combine(s) for s in range(1, 7))

    @cuda.jit(device=True)
    def combine(s, u, y, k, h):
        if s == 1:
            c1(u, y, k, h)
        elif s == 2:
            c2(u, y, k, h)
        elif s == 3:
            c3(u, y, k, h)
        elif s == 4:
            c4(u, y, k, h)
        elif s == 5:
            c5(u, y, k, h)
        else:
            c6(u, y, k, h)

    @cuda.jit(device=True, inline="never")
    def tstep(y, u, g6, k, fbuf, p_local, t, h, pbuf, hseed, zero):
        """One Tsit5 step from ``(t, y)`` with ``k[0] = f(t, y)``: the candidate in ``u``, its rate in ``k[6]``."""
        cs = cuda.const.array_like(c_t)
        for s in range(1, 7):
            combine(s, u, y, k, h)
            if s == 5:
                for j in range(n):
                    g6[j] = u[j]  # the sixth stage's state, at t + h like the candidate
            write(u, t + cs[s - 1] * h, p_local, fbuf, pbuf, hseed, zero)
            for j in range(n):
                k[s, j] = fbuf[j]

    E = (T.E1, T.E2, T.E3, T.E4, T.E5, T.E6, T.E7)

    @cuda.jit(device=True)
    def terr(k, h, err):
        """The embedded error estimate, ``h sum_i e_i k_i``."""
        for j in range(n):
            err[j] = h * (
                E[0] * k[0, j]
                + E[1] * k[1, j]
                + E[2] * k[2, j]
                + E[3] * k[3, j]
                + E[4] * k[4, j]
                + E[5] * k[5, j]
                + E[6] * k[6, j]
            )

    @cuda.jit(device=True)
    def teig(u, g6, k):
        """The largest eigenvalue magnitude, ``max_j |k7_j - k6_j| / |u_j - g6_j|`` (Hairer and Wanner II, p. 22).

        A row whose rate differs while its state matches to the last bit counts as
        (practically) infinitely stiff, as OrdinaryDiffEq.jl's division makes it.
        At a stability-limited step the two states can agree to rounding while the
        stiff rows' rates do not.
        """
        largest = 0.0
        for j in range(n):
            # branch-free (the LLVM 7 translation rejects what numba makes of the branches):
            # 0 where neither moved, huge where only the rate did
            ratio = math.fabs(k[6, j] - k[5, j]) / max(
                math.fabs(u[j] - g6[j]), 1.0e-300
            )
            if ratio > largest:
                largest = ratio
        return largest

    @cuda.jit(device=True)
    def trate(u, g6, k, jdiag):
        """Each row's restoring rate for damping the inherited error, ``-|k7 - k6| / |u - g6|``.

        This is the row's own share of the stiffness estimate, free from the two
        stages at the step's end. A Tsit5 block has no Jacobian diagonal, and
        without damping a stiff row overstates its inherited error by ``h |J_rr|``,
        which starves the upstream block of step size.
        """
        for j in range(n):
            jdiag[j] = -math.fabs(k[6, j] - k[5, j]) / max(
                math.fabs(u[j] - g6[j]), 1.0e-300
            )

    @cuda.jit(device=True)
    def tcurv(y, g, t, p_local, s2, zero):
        """``y'' = J y' + df/dt`` at a node, one forward sweep along ``(y', 1)``."""
        column = cuda.local.array(n, types.float64)
        tangent(column, y, t, p_local, g[0:n], 1.0, zero[0:P_LOCAL])
        for r in range(n):
            s2[r] = column[r]

    if n_reads > 0:

        @cuda.jit(device=True)
        def tcoup(y, t, p_local, p_seed, coup, jdiag, pbuf, zero):
            """The rates ``df/dq . eps`` at which each upstream's error enters, one sweep each."""
            column = cuda.local.array(n, types.float64)
            for e in range(n_reads):
                for c in range(P_LOCAL):
                    pbuf[c] = p_seed[e, c]
                tangent(column, y, t, p_local, zero[0:n], 0.0, pbuf)
                for r in range(n):
                    coup[e, r] = column[r]

    else:

        @cuda.jit(device=True)
        def tcoup(y, t, p_local, p_seed, coup, jdiag, pbuf, zero):
            pass

    @cuda.jit(device=True)
    def tdense(y, k, h, theta, out):
        """Tsit5's continuous extension at ``t + theta h``."""
        b1 = (
            -1.0530884977290216
            * theta
            * (theta - 1.3299890189751412)
            * (theta * theta - 1.4364028541716351 * theta + 0.7139816917074209)
        )
        b2 = (
            0.1017
            * theta
            * theta
            * (theta * theta - 2.1966568338249754 * theta + 1.2949852507374631)
        )
        b3 = (
            2.490627285651252793
            * theta
            * theta
            * (theta * theta - 2.38535645472061657 * theta + 1.57803468208092486)
        )
        b4 = (
            -16.54810288924490272
            * (theta - 1.21712927295533244)
            * (theta - 0.61620406037800089)
            * theta
            * theta
        )
        b5 = (
            47.37952196281928122
            * (theta - 1.203071208372362603)
            * (theta - 0.658047292653547382)
            * theta
            * theta
        )
        b6 = (
            -34.87065786149660974
            * (theta - 1.2)
            * (theta - 0.666666666666666667)
            * theta
            * theta
        )
        b7 = 2.5 * (theta - 1.0) * (theta - 0.6) * theta * theta
        for j in range(n):
            out[j] = y[j] + h * (
                b1 * k[0, j]
                + b2 * k[1, j]
                + b3 * k[2, j]
                + b4 * k[3, j]
                + b5 * k[4, j]
                + b6 * k[5, j]
                + b7 * k[6, j]
            )

    return {
        f"tstep_{b}": tstep,
        f"terr_{b}": terr,
        f"teig_{b}": teig,
        f"trate_{b}": trate,
        f"tcurv_{b}": tcurv,
        f"tcoup_{b}": tcoup,
        f"tdense_{b}": tdense,
    }


def _block_functions(
    plan: _Plan, bp: _BlockPlan, options, R, meth: _Method
) -> dict[str, Any]:
    """The device functions of one block, named with its index, for the kernel's namespace."""
    b, n, size, ns, m, mq = bp.b, bp.n, bp.size, bp.ns, bp.m, bp.mq
    P_LOCAL = plan.p_local
    n_reads = len(bp.reads)
    lu_dtype = np.float32 if meth.lu_fp32 else np.float64
    rhs = _wrap_rhs(plan, bp)
    out_fn = _wrap_outputs(plan, bp)
    tangent = make_tangent(rhs, n, P_LOCAL)
    # None where the block has no use for them; the device code calls them
    # only in branches it compiles for blocks that do, which ty cannot see.
    second: Any = make_second_tangent(rhs, n, P_LOCAL) if ns > 0 else None
    rhs_device = as_cuda_device(rhs)
    out_tangent: Any = (
        make_quantity_tangent(out_fn, m, n, P_LOCAL) if out_fn is not None else None
    )
    out_second: Any = (
        make_quantity_second_tangent(out_fn, m, n, P_LOCAL)
        if (out_fn is not None)
        else None
    )

    # -- seeds: a unit parameter vector plus the upstream sensitivities' Hermite coefficients
    dirs_t = _table(bp.dirs)
    dpc_t = _table(plan.dir_param_col)
    copies = tuple(
        (plan.hoff(b, u), hermite_slot(plan.blocks[u].hslot, 0), 6 * plan.blocks[u].m)
        for u in bp.reads
    )

    if copies:
        copy_t = np.asarray(copies, dtype=np.int32)

        @cuda.jit(device=True)
        def seed(kk, pbuf, hseed):
            for c in range(P_LOCAL):
                pbuf[c] = 0.0
            k = cuda.const.array_like(dirs_t)[kk]
            col = cuda.const.array_like(dpc_t)[k]
            if col >= 0:
                pbuf[col] = 1.0
            cp = cuda.const.array_like(copy_t)
            for e in range(n_reads):
                for c in range(cp[e, 2]):
                    pbuf[cp[e, 1] + c] = hseed[kk, cp[e, 0] + c]

    else:

        @cuda.jit(device=True)
        def seed(kk, pbuf, hseed):
            for c in range(P_LOCAL):
                pbuf[c] = 0.0
            k = cuda.const.array_like(dirs_t)[kk]
            col = cuda.const.array_like(dpc_t)[k]
            if col >= 0:
                pbuf[col] = 1.0

    # -- joint right-hand side [f, J S + J_p] and the sensitivity rows' df/dt --------------
    # Without directions the sensitivity loops are left out rather than run
    # zero times: numba types a loop body it never enters, and typing the
    # tangent call builds a derivative of that arity.
    if ns > 0:

        @cuda.jit(device=True)
        def write(z, t, p_local, out, pbuf, hseed, zero):
            values = rhs_device(z, t, p_local)
            for j in range(n):
                out[j] = values[j]
            tan = cuda.local.array(n, types.float64)
            for kk in range(ns):
                base = n + kk * n
                seed(kk, pbuf, hseed)
                tangent(tan, z, t, p_local, z[base : base + n], 0.0, pbuf)
                for r in range(n):
                    out[base + r] = tan[r]

        @cuda.jit(device=True)
        def sens_dt(z, t, p_local, dT, pbuf, hseed, zero):
            column = cuda.local.array(n, types.float64)
            for kk in range(ns):
                base = n + kk * n
                seed(kk, pbuf, hseed)
                second(
                    column,
                    z,
                    t,
                    p_local,
                    z[base : base + n],
                    0.0,
                    pbuf,
                    zero[0:n],
                    1.0,
                    zero[0:P_LOCAL],
                    zero[0:n],
                    0.0,
                    zero[0:P_LOCAL],
                )
                for r in range(n):
                    dT[base + r] = column[r]

    else:

        @cuda.jit(device=True)
        def write(z, t, p_local, out, pbuf, hseed, zero):
            values = rhs_device(z, t, p_local)
            for j in range(n):
                out[j] = values[j]

        @cuda.jit(device=True)
        def sens_dt(z, t, p_local, dT, pbuf, hseed, zero):
            pass

    def add_outputs(functions: dict[str, Any]) -> dict[str, Any]:
        """Add the published quantities' functions at a node to ``functions``."""
        # -- the published quantities at a node ---------------------------------------------
        if out_fn is not None:
            out_device = as_cuda_device(out_fn)
            curv_left = np.asarray(_CURV_LEFT)
            curv_right = np.asarray(_CURV_RIGHT)

            @cuda.jit(device=True)
            def outputs_primal(th, g, s2, t, p_local, Q, Q1, Q2, zero):
                """The quantities and their first and second time derivatives at a node."""
                values = out_device(th, t, p_local)
                for j in range(m):
                    Q[j] = values[j]
                column = cuda.local.array(m, types.float64)
                out_tangent(column, th, t, p_local, g[0:n], 1.0, zero[0:P_LOCAL])
                for j in range(m):
                    Q1[j] = column[j]
                # D2 o[(g, 1, 0), (g, 1, 0)] + D o[(y'', 0, 0)]
                out_second(
                    column,
                    th,
                    t,
                    p_local,
                    g[0:n],
                    1.0,
                    zero[0:P_LOCAL],
                    g[0:n],
                    1.0,
                    zero[0:P_LOCAL],
                    s2[0:n],
                    0.0,
                    zero[0:P_LOCAL],
                )
                for j in range(m):
                    Q2[j] = column[j]

            @cuda.jit(device=True)
            def outputs_error(th, t, p_local, err_y, p_seed, errq, zero):
                """The quantities' error ``D o[(eps_y, 0, eps_upstream)]`` from the block's error and its upstream's."""
                column = cuda.local.array(m, types.float64)
                out_tangent(column, th, t, p_local, err_y, 0.0, p_seed)
                for j in range(m):
                    errq[j] = column[j]

            if ns > 0:

                @cuda.jit(device=True)
                def outputs_sens(th, g, t, p_local, Q, Q1, pbuf, hseed, zero):
                    """Every direction's quantity sensitivity and its rate at ``(th, g, t)``."""
                    column = cuda.local.array(m, types.float64)
                    for kk in range(ns):
                        base = n + kk * n
                        qb = m + kk * m
                        seed(kk, pbuf, hseed)
                        out_tangent(
                            column, th, t, p_local, th[base : base + n], 0.0, pbuf
                        )
                        for j in range(m):
                            Q[qb + j] = column[j]
                        # D2 o[(S_k, 0, e_k), (g, 1, 0)] + D o[(S_k', 0, 0)]
                        out_second(
                            column,
                            th,
                            t,
                            p_local,
                            th[base : base + n],
                            0.0,
                            pbuf,
                            g[0:n],
                            1.0,
                            zero[0:P_LOCAL],
                            g[base : base + n],
                            0.0,
                            zero[0:P_LOCAL],
                        )
                        for j in range(m):
                            Q1[qb + j] = column[j]

                @cuda.jit(device=True)
                def outputs_curvature(nQ, nQ1, qa, Qm, Q1m, Qb, Q1b, h, nQ2, nQ2l, qb):
                    """The sensitivities' curvatures at the ends of the piece from node ``qa`` to node ``qb``.

                    They come from the quintic through the values and rates at the piece's
                    start (node ``qa``), midpoint and end (``Qb``, ``Q1b``). The start's
                    curvature replaces node ``qa``'s right-hand one, and the end's becomes node
                    ``qb``'s left-hand one.
                    """
                    cl = cuda.const.array_like(curv_left)
                    cr = cuda.const.array_like(curv_right)
                    for j in range(m, mq):
                        d0 = nQ[qa, j]
                        d1 = h * nQ1[qa, j]
                        d2 = Qm[j]
                        d3 = h * Q1m[j]
                        d4 = Qb[j]
                        d5 = h * Q1b[j]
                        nQ2[qa, j] = (
                            cl[0] * d0
                            + cl[1] * d1
                            + cl[2] * d2
                            + cl[3] * d3
                            + cl[4] * d4
                            + cl[5] * d5
                        ) / (h * h)
                        nQ2l[qb, j] = (
                            cr[0] * d0
                            + cr[1] * d1
                            + cr[2] * d2
                            + cr[3] * d3
                            + cr[4] * d4
                            + cr[5] * d5
                        ) / (h * h)

                functions[f"outputs_sens_{b}"] = outputs_sens
                functions[f"outputs_curvature_{b}"] = outputs_curvature

            functions[f"outputs_primal_{b}"] = outputs_primal
            functions[f"outputs_error_{b}"] = outputs_error
        return functions

    explicit = _explicit_functions(plan, bp, write, tangent) if meth.explicit else {}
    if meth.kind == "tsit5":
        functions = {
            f"write_{b}": write,
            f"seed_{b}": seed,
            f"sens_dt_{b}": sens_dt,
            f"size_{b}": size,
            f"lu_size_{b}": 1,
            f"ipiv_{b}": 1,
            **explicit,
        }
        return functions if bp.sink else add_outputs(functions)

    # -- the matrix ---------------------------------------------------------------------
    if bp.pattern is None or (not bp.sink and n <= options.split_max_driver):
        structure = (
            colour_sparsity(bp.pattern) if bp.pattern is not None else dense_jacobian(n)
        )
        factorize, solve_local = R.dense_lu_solver(n)
        ipiv_size = n
        store = None
        if bp.pattern is not None:
            # coloured from the pattern onto the dense row-major slots; the
            # slots no colour owns are cleared before each write
            store = np.full(n * structure.n_colours, -1, dtype=np.int32)
            for row, cols in enumerate(bp.pattern):
                for col in cols:
                    store[row * structure.n_colours + structure.colour[col]] = (
                        row * n + col
                    )
        lu_size = n * n
        diagonal = tuple(r * n + r for r in range(n))
        needs_clear = store is not None

        class _Dense:
            def slot(self, r, c):
                return r * n + c

        slots = _Dense()
    else:
        solver = sparse_direct_solver_for(bp.pattern, meth.ordering, "never")
        structure = solver.compressed
        factorize, solve_local = solver.factorize_local, solver.solve_local
        ipiv_size = solver.ipiv_size
        store = structure.store_slots()
        lu_size = structure.size
        diagonal = structure.diagonal
        needs_clear = store is not None and lu_size > int((store >= 0).sum())
        slots = structure
    diag_t = np.asarray(diagonal, dtype=np.int32)

    if needs_clear:

        @cuda.jit(device=True)
        def clear(lu):
            for i in range(lu_size):
                lu[i] = lu_dtype(0.0)

    else:

        @cuda.jit(device=True)
        def clear(lu):
            pass

    jac = R._make_literal_seed_jacobian(
        n_vars=n,
        n_params=P_LOCAL,
        n_colours=structure.n_colours,
        store_table=store,
        namespace=dict(
            cuda=cuda,
            float64=types.float64,
            tangent_of=tangent,
            colour_seeds=structure.seed_table(P_LOCAL),
            clear_matrix=clear,
            lu_dtype=lu_dtype,
        ),
        inline="never",
        batch=structure.n_colours + 1 + n_reads,
        coupling=n_reads,
    )

    AUTO = meth.kind == "auto"
    if AUTO:
        jinf = _jacobian_inf_norm(bp, slots)
    else:
        # numba types the branch that never runs: a stub keeps the norm out of the build

        @cuda.jit(device=True)
        def jinf(neg_a):
            return 0.0

    # -- one Rodas5P step ---------------------------------------------------------------
    c_rows = _rows(
        (
            (),
            (R.C21,),
            (R.C31, R.C32),
            (R.C41, R.C42, R.C43),
            (R.C51, R.C52, R.C53, R.C54),
            (R.C61, R.C62, R.C63, R.C64, R.C65),
            (R.C71, R.C72, R.C73, R.C74, R.C75, R.C76),
            (R.C81, R.C82, R.C83, R.C84, R.C85, R.C86, R.C87),
        )
    )
    d_row = (R.D1, R.D2, R.D3, R.D4, R.D5, 0.0, 0.0, 0.0)
    a_rows = _rows(
        (
            (R.A21,),
            (R.A31, R.A32),
            (R.A41, R.A42, R.A43),
            (R.A51, R.A52, R.A53, R.A54),
            (R.A61, R.A62, R.A63, R.A64, R.A65),
        )
    )
    T_AFTER = np.asarray((R.C2, R.C3, R.C4, R.C5, 1.0, 1.0, 1.0, 0.0))
    GAMMA = R.GAMMA

    if ns == 0:

        @cuda.jit(device=True)
        def stage_solve(
            lu, ipiv, rhs_buf, k_stages, s, work, y, p_local, t, pbuf, hseed, zero
        ):
            solve_local(lu, ipiv, rhs_buf)
            for j in range(n):
                k_stages[s, j] = np.float64(rhs_buf[j])

    else:

        @cuda.jit(device=True)
        def stage_solve(
            lu, ipiv, rhs_buf, k_stages, s, work, y, p_local, t, pbuf, hseed, zero
        ):
            solve_local(lu, ipiv, rhs_buf)
            for j in range(n):
                k_stages[s, j] = np.float64(rhs_buf[j])
            coupling = cuda.local.array(n, types.float64)
            k_state = cuda.local.array(n, types.float64)
            for j in range(n):
                k_state[j] = np.float64(k_stages[s, j])
            for kk in range(ns):
                base = n + kk * n
                seed(kk, pbuf, hseed)
                second(
                    coupling,
                    y,
                    t,
                    p_local,
                    y[base : base + n],
                    0.0,
                    pbuf,
                    k_state,
                    0.0,
                    zero[0:P_LOCAL],
                    zero[0:n],
                    0.0,
                    zero[0:P_LOCAL],
                )
                for j in range(n):
                    rhs_buf[j] = lu_dtype(work[base + j] + coupling[j])
                solve_local(lu, ipiv, rhs_buf)
                for j in range(n):
                    k_stages[s, base + j] = np.float64(rhs_buf[j])

    # The state rows go to the solver's buffer, the sensitivity rows stay in
    # ``work`` until their own solve -- as two loops, never as a branch
    # choosing between the two arrays, which the compiler turns into a select
    # of memref descriptors that the LLVM 7 translation cannot type. Every
    # coefficient is a literal and every loop has a constant trip count, so
    # each stage lowers to straight-line arithmetic; the stage number only
    # picks the function.
    def make_stage(s):
        c_row = c_rows[s]
        d = d_row[s]
        if s == 0:

            @cuda.jit(device=True)
            def stage(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            ):
                for j in range(n):
                    rhs_buf[j] = lu_dtype(work[j] + dt_use * d * dT[j])
                for j in range(n, size):
                    work[j] = work[j] + dt_use * d * dT[j]
                stage_solve(
                    lu,
                    ipiv,
                    rhs_buf,
                    k_stages,
                    0,
                    work,
                    y,
                    p_local,
                    t,
                    pbuf,
                    hseed,
                    zero,
                )

        elif d != 0.0:

            @cuda.jit(device=True)
            def stage(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            ):
                for j in range(n):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    rhs_buf[j] = lu_dtype(work[j] + dt_use * d * dT[j] + acc * inv_dt)
                for j in range(n, size):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    work[j] = work[j] + dt_use * d * dT[j] + acc * inv_dt
                stage_solve(
                    lu,
                    ipiv,
                    rhs_buf,
                    k_stages,
                    s,
                    work,
                    y,
                    p_local,
                    t,
                    pbuf,
                    hseed,
                    zero,
                )

        else:

            @cuda.jit(device=True)
            def stage(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            ):
                for j in range(n):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    rhs_buf[j] = lu_dtype(work[j] + acc * inv_dt)
                for j in range(n, size):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    work[j] = work[j] + acc * inv_dt
                stage_solve(
                    lu,
                    ipiv,
                    rhs_buf,
                    k_stages,
                    s,
                    work,
                    y,
                    p_local,
                    t,
                    pbuf,
                    hseed,
                    zero,
                )

        return stage

    def make_advance(s):
        a_row = a_rows[s]

        @cuda.jit(device=True)
        def advance(u, y, k_stages):
            for j in range(size):
                acc = a_row[0] * k_stages[0, j]
                for k in range(1, s + 1):
                    acc += a_row[k] * k_stages[k, j]
                u[j] = y[j] + acc

        return advance

    s0, s1, s2, s3, s4, s5, s6, s7 = (make_stage(s) for s in range(8))
    a0, a1, a2, a3, a4 = (make_advance(s) for s in range(5))

    # The step is one loop over the stages with the block's right-hand side
    # called at one place in it: unrolled into eight call sites, numba copied
    # the inlined right-hand side eight times over at every level of the call
    # tree. The stage number picks the literal stage function.
    @cuda.jit(device=True, inline="never")
    def stage(
        s,
        lu,
        ipiv,
        rhs_buf,
        k_stages,
        work,
        dT,
        y,
        p_local,
        t,
        dt_use,
        inv_dt,
        pbuf,
        hseed,
        zero,
    ):
        if s == 0:
            s0(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        elif s == 1:
            s1(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        elif s == 2:
            s2(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        elif s == 3:
            s3(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        elif s == 4:
            s4(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        elif s == 5:
            s5(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        elif s == 6:
            s6(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
        else:
            s7(
                lu,
                ipiv,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )

    @cuda.jit(device=True)
    def advance(s, u, y, k_stages):
        """``u`` after stage ``s``: ``y + sum_k a[s, k] k_k`` for the first five, ``u + k_s`` after."""
        if s == 0:
            a0(u, y, k_stages)
        elif s == 1:
            a1(u, y, k_stages)
        elif s == 2:
            a2(u, y, k_stages)
        elif s == 3:
            a3(u, y, k_stages)
        elif s == 4:
            a4(u, y, k_stages)
        else:
            for j in range(size):
                u[j] += k_stages[s, j]

    if bp.sink:
        # The sink's Jacobian is written at every attempt (no node to keep it
        # at); the coupling directions give the error it inherits.
        if n_reads > 0:

            @cuda.jit(device=True, inline="never")
            def step(
                y,
                u,
                work,
                dT,
                k_stages,
                lu,
                ipiv,
                rhs_buf,
                p_local,
                t,
                dt_use,
                pbuf,
                hseed,
                zero,
                p_seed,
                coup,
                jdiag,
                jn,
            ):
                """One Rodas5P step from ``y``, leaving ``u + k_stages[7]`` as the candidate."""
                inv_dt = 1.0 / dt_use
                dtgamma_inv = inv_dt / GAMMA
                diag = cuda.const.array_like(diag_t)
                jac(y, t, p_local, lu, dT, p_seed, coup)
                if AUTO:
                    jn[0] = jinf(
                        lu
                    )  # ||J||_inf, the stiffness estimate, before the step's 1/(h gamma)
                sens_dt(y, t, p_local, dT, pbuf, hseed, zero)
                for d_ in range(n):
                    jdiag[d_] = -np.float64(
                        lu[diag[d_]]
                    )  # the Jacobian's diagonal, before the step's 1/(h gamma)
                    lu[diag[d_]] += lu_dtype(dtgamma_inv)
                factorize(lu, ipiv)
                t_after = cuda.const.array_like(T_AFTER)
                for j in range(size):
                    u[j] = y[j]
                write(u, t, p_local, work, pbuf, hseed, zero)
                for s in range(8):
                    stage(
                        s,
                        lu,
                        ipiv,
                        rhs_buf,
                        k_stages,
                        work,
                        dT,
                        y,
                        p_local,
                        t,
                        dt_use,
                        inv_dt,
                        pbuf,
                        hseed,
                        zero,
                    )
                    if s < 7:
                        advance(s, u, y, k_stages)
                        write(
                            u, t + t_after[s] * dt_use, p_local, work, pbuf, hseed, zero
                        )

        else:

            @cuda.jit(device=True, inline="never")
            def step(
                y,
                u,
                work,
                dT,
                k_stages,
                lu,
                ipiv,
                rhs_buf,
                p_local,
                t,
                dt_use,
                pbuf,
                hseed,
                zero,
                p_seed,
                coup,
                jdiag,
                jn,
            ):
                inv_dt = 1.0 / dt_use
                dtgamma_inv = inv_dt / GAMMA
                diag = cuda.const.array_like(diag_t)
                jac(y, t, p_local, lu, dT)
                if AUTO:
                    jn[0] = jinf(lu)
                sens_dt(y, t, p_local, dT, pbuf, hseed, zero)
                for d_ in range(n):
                    lu[diag[d_]] += lu_dtype(dtgamma_inv)
                factorize(lu, ipiv)
                t_after = cuda.const.array_like(T_AFTER)
                for j in range(size):
                    u[j] = y[j]
                write(u, t, p_local, work, pbuf, hseed, zero)
                for s in range(8):
                    stage(
                        s,
                        lu,
                        ipiv,
                        rhs_buf,
                        k_stages,
                        work,
                        dT,
                        y,
                        p_local,
                        t,
                        dt_use,
                        inv_dt,
                        pbuf,
                        hseed,
                        zero,
                    )
                    if s < 7:
                        advance(s, u, y, k_stages)
                        write(
                            u, t + t_after[s] * dt_use, p_local, work, pbuf, hseed, zero
                        )

        return {
            f"step_{b}": step,
            f"write_{b}": write,
            f"seed_{b}": seed,
            f"size_{b}": size,
            f"lu_size_{b}": lu_size,
            f"ipiv_{b}": ipiv_size,
            **explicit,
        }

    # -- a block with nodes: FSAL step, second derivatives, published quantities -----------
    @cuda.jit(device=True, inline="never")
    def step(
        th, u, work, dT, kd, lu, ipiv, rhs_buf, p_local, t, dt_use, pbuf, hseed, zero
    ):
        """One Rodas5P step from ``th`` against the Jacobian already in ``lu`` and ``[g, S']`` in ``work``."""
        inv_dt = 1.0 / dt_use
        dtgamma_inv = inv_dt / GAMMA
        diag = cuda.const.array_like(diag_t)
        for d_ in range(n):
            lu[diag[d_]] += dtgamma_inv
        factorize(lu, ipiv)
        t_after = cuda.const.array_like(T_AFTER)
        for s in range(8):
            stage(
                s,
                lu,
                ipiv,
                rhs_buf,
                kd,
                work,
                dT,
                th,
                p_local,
                t,
                dt_use,
                inv_dt,
                pbuf,
                hseed,
                zero,
            )
            if s < 7:
                advance(s, u, th, kd)
                write(u, t + t_after[s] * dt_use, p_local, work, pbuf, hseed, zero)

    jacobian_times = _jacobian_times(bp, slots)

    if ns > 0:

        @cuda.jit(device=True)
        def second_derivatives(th, g, neg_a, dT, t, p_local, s2, pbuf, hseed, zero):
            """``y'' = J g + dg/dt`` for the state and, by the second-order sweep, ``S''`` for every direction."""
            jacobian_times(neg_a, g, dT, s2)
            column = cuda.local.array(n, types.float64)
            for kk in range(ns):
                base = n + kk * n
                seed(kk, pbuf, hseed)
                # D2 f[(S_k, 0, e_k), (g, 1, 0)] + D f[(S_k', 0, 0)]
                second(
                    column,
                    th,
                    t,
                    p_local,
                    th[base : base + n],
                    0.0,
                    pbuf,
                    g[0:n],
                    1.0,
                    zero[0:P_LOCAL],
                    g[base : base + n],
                    0.0,
                    zero[0:P_LOCAL],
                )
                for r in range(n):
                    s2[base + r] = column[r]

    else:

        @cuda.jit(device=True)
        def second_derivatives(th, g, neg_a, dT, t, p_local, s2, pbuf, hseed, zero):
            jacobian_times(neg_a, g, dT, s2)

    # the node's Jacobian, with the coupling directions when the block reads others
    if n_reads > 0:

        @cuda.jit(device=True)
        def node_jacobian(th, t, p_local, neg_a, dT, p_seed, coup, jdiag):
            jac(th, t, p_local, neg_a, dT, p_seed, coup)
            diag = cuda.const.array_like(diag_t)
            for d_ in range(n):
                jdiag[d_] = -neg_a[diag[d_]]

    else:

        @cuda.jit(device=True)
        def node_jacobian(th, t, p_local, neg_a, dT, p_seed, coup, jdiag):
            jac(th, t, p_local, neg_a, dT)

    functions = {
        f"step_{b}": step,
        f"write_{b}": write,
        f"sens_dt_{b}": sens_dt,
        f"second_{b}": second_derivatives,
        f"node_jac_{b}": node_jacobian,
        f"seed_{b}": seed,
        f"size_{b}": size,
        f"lu_size_{b}": lu_size,
        f"ipiv_{b}": ipiv_size,
        f"jinf_{b}": jinf,
        **explicit,
    }

    return add_outputs(functions)


def _hermite_writer(plan: _Plan, u: int, b: int) -> MLIRDispatcher:
    """Write upstream ``u``'s Hermite block for reader ``b``'s step into ``p_local`` and ``hseed_b``."""
    up, rp = plan.blocks[u], plan.blocks[b]
    m, hslot, hoff = up.m, up.hslot, plan.hoff(b, u)
    dirs_t = _table(rp.dirs)
    upos_t = _table(up.pos)
    ns = rp.ns

    @cuda.jit(device=True)
    def hermite(p_local, hseed, t, h, qa, qa1, qa2, qb, qb1, qb2):
        p_local[hslot] = t
        p_local[hslot + 1] = h
        for j in range(m):
            c = hslot + 2 + 6 * j
            p_local[c] = qa[j]
            p_local[c + 1] = qa1[j]
            p_local[c + 2] = qa2[j]
            p_local[c + 3] = qb[j]
            p_local[c + 4] = qb1[j]
            p_local[c + 5] = qb2[j]
        for kk in range(ns):
            k = cuda.const.array_like(dirs_t)[kk]
            ku = cuda.const.array_like(upos_t)[k]
            for j in range(m):
                c = hoff + 6 * j
                if ku >= 0:
                    base = m + ku * m + j
                    hseed[kk, c] = qa[base]
                    hseed[kk, c + 1] = qa1[base]
                    hseed[kk, c + 2] = qa2[base]
                    hseed[kk, c + 3] = qb[base]
                    hseed[kk, c + 4] = qb1[base]
                    hseed[kk, c + 5] = qb2[base]
                else:
                    for e in range(6):
                        hseed[kk, c + e] = 0.0

    return hermite


# --- the kernel, generated block by block ---------------------------------------------


class _Source:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, indent: int, text: str = "") -> None:
        for line in text.split("\n"):
            self.lines.append(("    " * indent + line) if line else "")


def _emit_kernel(
    plan: _Plan, options, n_nodes: tuple[int, ...], methods: tuple[_Method, ...]
) -> str:
    """The kernel's source: every block's buffers, the advance loops, the sink's step and the saves."""
    from modax import rodas5P as R

    S = plan.blocks[-1]
    n_vars, n_params, P = plan.n_vars, plan.n_params, plan.p_local
    n_sens = plan.n_sens
    HAS_HOOK = options.save_hook is not None
    SAVE_HISTORY = bool(options.save_history)
    src = _Source()
    w = src

    def block_name(b: int) -> str:
        return plan.blocks[b].block.name or f"block {b}"

    w(
        0,
        "def kernel(y0, times_all, params, dt0, rtol, atol, max_steps, time_rows, weights, hist, accepted_out, rejected_out, loop_out, hook_out):",
    )
    w(1, "i = cuda.blockIdx.x * TPB + cuda.threadIdx.x")
    w(
        1,
        "# the lanes of this warp that hold a trajectory: every one of them votes below",
    )
    w(1, "lane_mask = cuda.ballot_sync(0xFFFFFFFF, i < y0.shape[0])")
    w(1, "if i >= y0.shape[0]:")
    w(2, "return")
    w(1, "# one save grid shared by the ensemble, or one row per trajectory")
    w(1, "n_save = times_all.shape[0] // time_rows")
    w(1, "t_row = (i if time_rows > 1 else 0) * n_save")
    w(1, "times = times_all[t_row : t_row + n_save]")
    w(1, "tf = times[n_save - 1]")
    w(1, "dt_init = dt0 if dt0 > 0.0 else (tf - times[0]) * 1e-6")
    w(1, f"p_local = cuda.local.array({P}, np.float64)")
    w(1, f"pbuf = cuda.local.array({P}, np.float64)")
    w(
        1,
        f"zero = cuda.local.array({max(P, max(bp.size for bp in plan.blocks))}, np.float64)",
    )
    w(
        1,
        f"y_save = cuda.local.array({n_vars * (1 + n_sens) if HAS_HOOK else 1}, np.float64)",
    )
    w(1, f"for c in range({max(P, max(bp.size for bp in plan.blocks))}):")
    w(2, "zero[c] = 0.0")
    w(1, f"for c in range({P}):")
    w(2, "p_local[c] = 0.0")
    w(1, f"for c in range({n_params}):")
    w(2, "p_local[c] = params[i, c]")
    for bp in plan.blocks:
        if bp.hslot >= 0:
            w(1, f"p_local[{bp.hslot + 1}] = 1.0")
    # -- buffers --
    for bp in plan.blocks:
        b, size, n, mq, m = bp.b, bp.size, bp.n, bp.mq, bp.m
        nr = len(bp.reads)
        w(
            1,
            f"# {block_name(b)}: {n} states, {bp.ns} directions"
            + (f", {m} quantities" if m else ""),
        )
        w(
            1,
            f"hseed_{b} = cuda.local.array(({max(1, bp.ns)}, {max(1, plan.hcols(b))}), np.float64)",
        )
        w(1, f"for kk in range({max(1, bp.ns)}):")
        w(2, f"for c in range({max(1, plan.hcols(b))}):")
        w(3, f"hseed_{b}[kk, c] = 0.0")
        w(1, f"pseed_{b} = cuda.local.array(({max(1, nr)}, {P}), np.float64)")
        w(1, f"coup_{b} = cuda.local.array(({max(1, nr)}, {n}), np.float64)")
        w(1, f"jdiag_{b} = cuda.local.array({n}, np.float64)")
        w(1, f"for e in range({max(1, nr)}):")
        w(2, f"for c in range({P}):")
        w(3, f"pseed_{b}[e, c] = 0.0")
        meth = methods[b]
        if meth.explicit:
            w(1, f"tk_{b} = cuda.local.array((7, {n}), np.float64)")
            for name in ("tg6", "tfb", "te"):
                w(1, f"{name}_{b} = cuda.local.array({n}, np.float64)")
        if bp.sink:
            if meth.implicit:
                lu_t = "types.float32" if meth.lu_fp32 else "types.float64"
                w(1, f"lu_{b} = cuda.local.array(lu_size_{b}, {lu_t})")
                w(1, f"rhsb_{b} = cuda.local.array({n}, {lu_t})")
                w(1, f"ipiv_{b} = cuda.local.array(ipiv_size_{b}, np.int32)")
                w(1, f"k_{b} = cuda.local.array((8, {size}), {lu_t})")
                w(1, f"jn_{b} = cuda.local.array(1, np.float64)")
                for name in ("work", "dT", "h3"):
                    w(1, f"{name}_{b} = cuda.local.array({size}, np.float64)")
            for name in ("y", "u"):
                w(1, f"{name}_{b} = cuda.local.array({size}, np.float64)")
            continue
        N = n_nodes[b]
        if meth.implicit:
            w(1, f"lu_{b} = cuda.local.array(lu_size_{b}, np.float64)")
            w(1, f"nega_{b} = cuda.local.array(lu_size_{b}, np.float64)")
            w(1, f"nega1_{b} = cuda.local.array(lu_size_{b}, np.float64)")
            w(1, f"rhsb_{b} = cuda.local.array({n}, np.float64)")
            w(1, f"ipiv_{b} = cuda.local.array(ipiv_size_{b}, np.int32)")
            w(1, f"k_{b} = cuda.local.array((8, {size}), np.float64)")
            for name in ("work", "dT0", "dT1"):
                w(1, f"{name}_{b} = cuda.local.array({size}, np.float64)")
        for name in (
            "th",
            "th1",
            "u",
            "g0",
            "g1",
            "s20",
            "s21",
            "ya",
            "ga",
            "s2a",
            "yb",
            "gb",
            "s2b",
        ):
            w(1, f"{name}_{b} = cuda.local.array({size}, np.float64)")
        w(1, f"nt_{b} = cuda.local.array({N}, np.float64)")
        for name in ("ny", "ng", "ns2"):
            w(1, f"{name}_{b} = cuda.local.array(({N}, {size}), np.float64)")
        w(1, f"nerr_{b} = cuda.local.array(({N}, {n}), np.float64)")
        w(1, f"erry_{b} = cuda.local.array({n}, np.float64)")
        if bp.identity:
            # the published quantities are the states: the same arrays
            w(1, f"nQ_{b} = ny_{b}")
            w(1, f"nQ1_{b} = ng_{b}")
            w(1, f"nQ2_{b} = ns2_{b}")
            w(1, f"nQ2l_{b} = ns2_{b}")
            w(1, f"nerrq_{b} = nerr_{b}")
        else:
            for name in ("nQ", "nQ1", "nQ2"):
                w(1, f"{name}_{b} = cuda.local.array(({N}, {mq}), np.float64)")
            if bp.ns > 0:
                w(1, f"nQ2l_{b} = cuda.local.array(({N}, {mq}), np.float64)")
                for name in ("Qm", "Q1m"):
                    w(1, f"{name}_{b} = cuda.local.array({mq}, np.float64)")
            else:
                w(1, f"nQ2l_{b} = nQ2_{b}")
            w(1, f"nerrq_{b} = cuda.local.array(({N}, {m}), np.float64)")
        for name in ("qa", "qa1", "qa2", "qb", "qb1", "qb2"):
            w(1, f"{name}_{b} = cuda.local.array({mq}, np.float64)")

    # -- the initial state, from the user's layout --
    w(1, "# the initial state, from the user's layout")
    for bp in plan.blocks:
        b = bp.b
        y = f"y_{b}" if bp.sink else f"th_{b}"
        w(1, f"for r in range({bp.n}):")
        w(2, f"{y}[r] = y0[i, idx_{b}[r]]")
        if bp.ns:
            w(1, f"for kk in range({bp.ns}):")
            w(2, f"k = dirs_{b}[kk]")
            w(2, f"for r in range({bp.n}):")
            w(
                3,
                f"{y}[{bp.n} + kk * {bp.n} + r] = y0[i, {n_vars} + k * {n_vars} + idx_{b}[r]]",
            )
    w(1, "# every block's error weights, read once into thread-local memory")
    for bp in plan.blocks:
        b = bp.b
        w(1, f"wl_{b} = cuda.local.array({bp.size}, np.float64)")
        w(1, f"for r in range({bp.n}):")
        w(2, f"wl_{b}[r] = weights[i, idx_{b}[r]]")
        if bp.ns:
            w(1, f"for kk in range({bp.ns}):")
            w(2, f"k = dirs_{b}[kk]")
            w(2, f"for r in range({bp.n}):")
            w(
                3,
                f"wl_{b}[{bp.n} + kk * {bp.n} + r] = weights[i, {n_vars} + k * {n_vars} + idx_{b}[r]]",
            )
    if SAVE_HISTORY:
        w(1, f"for j in range({n_vars * (1 + n_sens)}):")
        w(2, "hist[i, 0, j] = y0[i, j]")
    w(1, "t = times[0]")
    w(1, "save_idx = 1")
    w(1, "n_steps = 0")
    w(1, "accepted = 0")
    w(1, "rejected = 0")
    for bp in plan.blocks:
        b = bp.b
        w(1, f"dt_{b} = dt_init")
        w(1, f"ep1_{b} = 1.0")
        w(1, f"ep2_{b} = 1.0")
        w(1, f"lep1_{b} = 0.0")
        meth = methods[b]
        if meth.kind == "auto":
            first = bool(meth.switch.stiff_first)
            w(
                1,
                f"# {block_name(b)} switches between Tsit5 and Rodas5P: the lane's wish, the warp's choice",
            )
            w(1, f"want_{b} = {first}")
            w(1, f"mode_{b} = {first}")
            w(1, f"cnt_{b} = 0")
            w(1, f"eig_{b} = 0.0")
            w(1, f"jok_{b} = {first}")
        if meth.explicit and bp.sink:
            w(1, f"tok_{b} = False  # whether tk_{b}[0] holds f(t, y)")
        if not bp.sink:
            w(1, f"inh_{b} = 0.0")
            w(1, f"nn_{b} = 1")
            w(1, f"nt_{b}[0] = t")

    # -- the first node of every block with nodes, upstream first --
    def hermite_from_single_node(b: int, indent: int) -> None:
        """Every upstream's node 0 as a degenerate Hermite block (both ends the node)."""
        bp = plan.blocks[b]
        for u in bp.reads:
            w(indent, f"for j in range({plan.blocks[u].mq}):")
            w(indent + 1, f"qa_{u}[j] = nQ_{u}[0, j]")
            w(indent + 1, f"qa1_{u}[j] = nQ1_{u}[0, j]")
            w(indent + 1, f"qa2_{u}[j] = nQ2_{u}[0, j]")
            w(
                indent,
                f"hermite_{u}_{b}(p_local, hseed_{b}, t, 1.0, qa_{u}, qa1_{u}, qa2_{u}, qa_{u}, qa1_{u}, qa2_{u})",
            )

    for bp in plan.blocks:
        if bp.sink:
            continue
        b = bp.b
        w(1, f"# {block_name(b)}: its first node")
        hermite_from_single_node(b, 1)
        meth = methods[b]
        if meth.kind == "rodas" or (meth.kind == "auto" and meth.switch.stiff_first):
            w(
                1,
                f"node_jac_{b}(th_{b}, t, p_local, nega1_{b}, dT1_{b}, pseed_{b}, coup_{b}, jdiag_{b})",
            )
            w(1, f"sens_dt_{b}(th_{b}, t, p_local, dT1_{b}, pbuf, hseed_{b}, zero)")
            w(1, f"write_{b}(th_{b}, t, p_local, g1_{b}, pbuf, hseed_{b}, zero)")
            w(
                1,
                f"second_{b}(th_{b}, g1_{b}, nega1_{b}, dT1_{b}, t, p_local, s21_{b}, pbuf, hseed_{b}, zero)",
            )
        else:
            w(1, f"write_{b}(th_{b}, t, p_local, g1_{b}, pbuf, hseed_{b}, zero)")
            w(1, f"tcurv_{b}(th_{b}, g1_{b}, t, p_local, s21_{b}, zero)")
        w(1, f"for j in range({bp.size}):")
        w(2, f"th1_{b}[j] = th_{b}[j]")
        w(2, f"ny_{b}[0, j] = th_{b}[j]")
        w(2, f"ng_{b}[0, j] = g1_{b}[j]")
        w(2, f"ns2_{b}[0, j] = s21_{b}[j]")
        w(1, f"for j in range({bp.n}):")
        w(2, f"nerr_{b}[0, j] = 0.0")
        if not bp.identity:
            w(
                1,
                f"outputs_primal_{b}(th_{b}, g1_{b}, s21_{b}, t, p_local, qa_{b}, qa1_{b}, qa2_{b}, zero)",
            )
            if bp.ns > 0:
                w(
                    1,
                    f"outputs_sens_{b}(th_{b}, g1_{b}, t, p_local, qa_{b}, qa1_{b}, pbuf, hseed_{b}, zero)",
                )
            w(1, f"for j in range({bp.mq}):")
            w(2, f"nQ_{b}[0, j] = qa_{b}[j]")
            w(2, f"nQ1_{b}[0, j] = qa1_{b}[j]")
            w(
                2, f"nQ2_{b}[0, j] = qa2_{b}[j]"
            )  # the sensitivity rows' is not known yet; nothing reads a curvature at s = 0
            if bp.ns > 0:
                w(2, f"nQ2l_{b}[0, j] = qa2_{b}[j]")
            w(1, f"for j in range({bp.m}):")
            w(2, f"nerrq_{b}[0, j] = 0.0")
    if HAS_HOOK:
        w(1, f"for j in range({max(1, options.hook_size)}):")
        w(2, "hook_out[i, j] = 0.0")
        w(1, f"for j in range({n_vars * (1 + n_sens)}):")
        w(2, "y_save[j] = y0[i, j]")
        w(1, "save_hook(0, y_save, times[0], p_local, hook_out[i])")

    # -- a reader's Hermite blocks over [t_a, t_a + h], and the upstream error over it --
    def read_upstream(b: int, t_a: str, h: str, indent: int) -> None:
        bp = plan.blocks[b]
        for e, u in enumerate(bp.reads):
            up = plan.blocks[u]
            w(
                indent,
                f"dense_q_{u}(nt_{u}, nQ_{u}, nQ1_{u}, nQ2_{u}, nQ2l_{u}, nn_{u}, {t_a}, qa_{u}, qa1_{u}, qa2_{u})",
            )
            w(
                indent,
                f"dense_q_{u}(nt_{u}, nQ_{u}, nQ1_{u}, nQ2_{u}, nQ2l_{u}, nn_{u}, {t_a} + {h}, qb_{u}, qb1_{u}, qb2_{u})",
            )
            w(
                indent,
                f"hermite_{u}_{b}(p_local, hseed_{b}, {t_a}, {h}, qa_{u}, qa1_{u}, qa2_{u}, qb_{u}, qb1_{u}, qb2_{u})",
            )
            w(
                indent,
                f"# {block_name(u)}'s error over this step: its largest estimate over the pieces the step lies in",
            )
            w(indent, f"for j in range({up.m}):")
            w(indent + 1, "err_j = 0.0")
            w(indent + 1, f"for q in range(1, nn_{u}):")
            w(
                indent + 2,
                f"if nt_{u}[q] > {t_a} + 1e-12 * max(1.0, math.fabs({t_a})) and nt_{u}[q - 1] < {t_a} + {h}:",
            )
            # (the element is read once into a local: an element re-read inside the
            # comparison's branch comes out as bitcode libNVVM cannot parse)
            w(indent + 3, f"e_q = nerrq_{u}[q, j]")
            w(indent + 3, "if math.fabs(e_q) > math.fabs(err_j):")
            w(indent + 4, "err_j = e_q")
            w(indent + 1, f"c = {hermite_slot(up.hslot, 0)} + 6 * j")
            w(indent + 1, f"pseed_{b}[{e}, c] = err_j")
            w(indent + 1, f"pseed_{b}[{e}, c + 3] = err_j")

    # -- the error a reader inherits from each upstream, from the coupling rates --
    def inherit(b: int, y: str, t_a: str, indent: int) -> None:
        bp = plan.blocks[b]
        for e, u in enumerate(bp.reads):
            w(
                indent,
                f"# the error {block_name(b)} inherits from {block_name(u)} over the piece this step lies in:",
            )
            w(
                indent,
                "# H df/dq . eps against the rows' tolerance, damped where a row restores itself",
            )
            w(indent, "# faster than the piece; it goes to the upstream's controller")
            w(indent, "q_in = 1")
            w(
                indent,
                f"while q_in < nn_{u} - 1 and nt_{u}[q_in] <= {t_a} + 1e-12 * max(1.0, math.fabs({t_a})):",
            )
            w(indent + 1, "q_in += 1")
            w(indent, f"h_cover = nt_{u}[q_in] - nt_{u}[q_in - 1]")
            w(indent, f"for r in range({bp.n}):")
            w(indent + 1, f"scale = atol + rtol * math.fabs({y}[r])")
            w(indent + 1, f"damping = 1.0 + h_cover * GAMMA * max(0.0, -jdiag_{b}[r])")
            w(
                indent + 1,
                "# the largest ratio, divided only when a row may beat the running maximum",
            )
            w(indent + 1, f"num = wl_{b}[r] * h_cover * math.fabs(coup_{b}[{e}, r])")
            w(indent + 1, "den = damping * scale")
            w(indent + 1, f"if num > inh_{u} * den:")
            w(indent + 2, "e_inh = num / den")
            w(indent + 2, f"if e_inh > inh_{u}:")
            w(indent + 3, f"inh_{u} = e_inh")

    def error_norm(b: int, y_old: str, y_new_expr, k7_expr, indent: int) -> None:
        """``err_norm`` of block ``b`` over its rows and directions."""
        bp = plan.blocks[b]
        n_error = (
            bp.n * (1 + bp.ns)
            if (options.spec is None or options.spec.error_control)
            else bp.n
        )
        w(indent, "err_local = 0.0")
        w(indent, f"for r in range({bp.n}):")
        w(indent + 1, f"y_new = {y_new_expr('r')}")
        w(
            indent + 1,
            f"scale = atol + rtol * max(math.fabs({y_old}[r]), math.fabs(y_new))",
        )
        w(indent + 1, f"e = wl_{b}[r] * {k7_expr('r')} / scale")
        w(indent + 1, "err_local += e * e")
        if bp.ns:
            w(indent, f"for kk in range({bp.ns}):")
            w(indent + 1, f"k = dirs_{b}[kk]")
            w(indent + 1, f"for r in range({bp.n}):")
            w(indent + 2, f"jj = {bp.n} + kk * {bp.n} + r")
            w(indent + 2, f"y_new = {y_new_expr('jj')}")
            w(
                indent + 2,
                f"scale = atol + rtol * max(math.fabs({y_old}[jj]), math.fabs(y_new))",
            )
            w(indent + 2, f"e = wl_{b}[jj] * {k7_expr('jj')} / scale")
            w(indent + 2, "err_local += e * e")
        w(indent, f"err_norm = math.sqrt(err_local / {n_error})")

    def controller(
        b: int, h: str, indent: int, inherited: bool, kind: str = "r"
    ) -> None:
        """The step proposal. An accepted step leaves ``min(safe_err, 1)`` in the controller's history.

        The inherited error counts like the block's own in the proposal, and in the
        history up to 1. A history entry above 1 (an accepted step whose readers
        inherited more than the tolerance) stops a rejection's proposal from
        contracting, since ``SAFETY err^E1 ep1^E2`` with ``err > 1`` needs
        ``ep1 <= 1``, and the controller could then reject a step at the same size for ever.
        Keeping only the block's own error in the history cost 20 percent on
        DISCO-EB (smaller steps after every step the inherited error limited).
        """
        w(indent, "if math.isnan(err_norm) or err_norm > 1e18:")
        w(indent + 1, "own_err = 1e18")
        w(indent, "elif err_norm == 0.0:")
        w(indent + 1, "own_err = 1e-18")
        w(indent, "else:")
        w(indent + 1, "own_err = err_norm")
        w(indent, "safe_err = own_err")
        if inherited:
            w(
                indent,
                "# the error its readers inherited over its last steps counts like its own",
            )
            w(indent, f"if inh_{b} > safe_err:")
            w(indent + 1, f"safe_err = inh_{b}")
        meth = methods[b]
        gains, order = (
            (meth.rodas_gains, R.ORDER) if kind == "r" else (meth.tsit_gains, T.ORDER)
        )
        e3 = gains.exponents(order)[2]
        c = f"{b}{kind}"
        w(indent, "lse = math.log(safe_err)")
        if e3 == 0.0:
            # err^E1 ep1^E2 as one exp of the logs, the history's log kept from its own step:
            # a log and an exp a step where two double-precision pows cost twice that
            w(indent, f"factor = SAFETY * math.exp(E1_{c} * lse + E2_{c} * lep1_{b})")
        else:
            w(
                indent,
                f"factor = SAFETY * safe_err**E1_{c} * ep1_{b}**E2_{c} * ep2_{b}**E3_{c}",
            )
        w(indent, f"if factor < FMIN_{c}:")
        w(indent + 1, f"factor = FMIN_{c}")
        w(indent, f"elif factor > FMAX_{c}:")
        w(indent + 1, f"factor = FMAX_{c}")
        w(indent, f"dt_new = {h} * factor")

    def by_mode(b: int, indent: int, emit) -> None:
        """``emit(kind, indent)`` for the block's method. Under Auto it emits both, and the warp's mode picks one."""
        kind = methods[b].kind
        if kind == "rodas":
            emit("r", indent)
        elif kind == "tsit5":
            emit("t", indent)
        else:
            # two ifs, not if/else: an else touching the same arrays as its if
            # becomes a select of memref descriptors the LLVM 7 translation rejects
            w(indent, f"if mode_{b}:")
            emit("r", indent + 1)
            w(indent, f"if not mode_{b}:")
            emit("t", indent + 1)

    def auto_vote(b: int, ready: str, indent: int) -> None:
        """The warp's method for the next step: Rodas5P if any lane about to step wants it.

        It uses separate ifs throughout, with no else and no conditional
        expressions, because the LLVM 7 translation rejects what numba makes of
        them here.
        """
        a = methods[b].switch
        w(indent, f"vote_{b} = False")
        w(indent, f"if cuda.any_sync(lane_mask, ({ready}) and want_{b}):")
        w(indent + 1, f"vote_{b} = True")
        w(indent, f"if vote_{b} and not mode_{b}:")
        w(indent + 1, f"dt_{b} = dt_{b} * {float(a.dtfac)!r}")
        w(indent, f"if mode_{b} and not vote_{b}:")
        w(indent + 1, f"dt_{b} = dt_{b} / {float(a.dtfac)!r}")
        w(indent, f"mode_{b} = vote_{b}")

    def auto_update(b: int, indent: int) -> None:
        """Count stiff and non-stiff steps in a row and turn the lane's wish over when a count passes its limit.

        Runs after every attempt, accepted or not, as OrdinaryDiffEq.jl's
        ``choose_algorithm!`` runs in each step's footer.
        """
        a = methods[b].switch
        w(
            indent,
            "# stiff: an explicit step of the next size would leave Tsit5's stability region",
        )
        w(indent, f"stiffness = eig_{b} * dt_{b} / {TSIT5_STABILITY_SIZE!r}")
        w(indent, f"stol = {float(a.nonstifftol)!r}")
        w(indent, f"if mode_{b}:")
        w(indent + 1, f"stol = {float(a.stifftol)!r}")
        w(indent, "if stiffness > stol:")
        w(indent + 1, f"if cnt_{b} < 0:")
        w(indent + 2, f"cnt_{b} = 0")
        w(indent + 1, f"cnt_{b} += 1")
        w(indent, "if not (stiffness > stol):")
        w(indent + 1, f"if cnt_{b} > 0:")
        w(indent + 2, f"cnt_{b} = 0")
        w(indent + 1, f"cnt_{b} -= 1")
        w(indent, f"if cnt_{b} > {int(a.maxstiffstep)}:")
        w(indent + 1, f"want_{b} = True")
        w(indent, f"if cnt_{b} < -{int(a.maxnonstiffstep)}:")
        w(indent + 1, f"want_{b} = False")

    def attempt_upstream(b: int, kind: str, indent: int) -> None:
        """One step from ``th_b`` over ``h_b``: the candidate in ``th1_b``, ``err_norm``, ``acc_b`` and ``dt_new``."""
        bp = plan.blocks[b]
        if kind == "r":
            w(indent, f"for c in range(lu_size_{b}):")
            w(indent + 1, f"lu_{b}[c] = nega_{b}[c]")
            w(indent, f"for j in range({bp.size}):")
            w(indent + 1, f"work_{b}[j] = g0_{b}[j]")
            w(
                indent,
                f"step_{b}(th_{b}, u_{b}, work_{b}, dT0_{b}, k_{b}, lu_{b}, ipiv_{b}, rhsb_{b}, p_local, t0_{b}, h_{b}, pbuf, hseed_{b}, zero)",
            )
            w(indent, f"for j in range({bp.size}):")
            w(indent + 1, f"th1_{b}[j] = u_{b}[j] + k_{b}[7, j]")
            error_norm(
                b,
                f"th_{b}",
                lambda j: f"th1_{b}[{j}]",
                lambda j: f"k_{b}[7, {j}]",
                indent,
            )
        else:
            w(indent, f"for j in range({bp.n}):")
            w(
                indent + 1,
                f"tk_{b}[0, j] = g0_{b}[j]  # first same as last: the node's rate",
            )
            w(
                indent,
                f"tstep_{b}(th_{b}, th1_{b}, tg6_{b}, tk_{b}, tfb_{b}, p_local, t0_{b}, h_{b}, pbuf, hseed_{b}, zero)",
            )
            w(indent, f"terr_{b}(tk_{b}, h_{b}, te_{b})")
            error_norm(
                b,
                f"th_{b}",
                lambda j: f"th1_{b}[{j}]",
                lambda j: f"te_{b}[{j}]",
                indent,
            )
        w(indent, f"acc_{b} = err_norm <= 1.0 and not math.isnan(err_norm)")
        controller(b, f"h_{b}", indent, inherited=True, kind=kind)

    def node_upstream(b: int, kind: str, indent: int) -> None:
        """At an accepted step's end ``(t1_b, th1_b)``: the rate, the second derivative, the coupling, the error."""
        bp = plan.blocks[b]
        auto = methods[b].kind == "auto"
        if kind == "r":
            w(
                indent,
                f"node_jac_{b}(th1_{b}, t1_{b}, p_local, nega1_{b}, dT1_{b}, pseed_{b}, coup_{b}, jdiag_{b})",
            )
            w(
                indent,
                f"sens_dt_{b}(th1_{b}, t1_{b}, p_local, dT1_{b}, pbuf, hseed_{b}, zero)",
            )
            w(
                indent,
                f"write_{b}(th1_{b}, t1_{b}, p_local, g1_{b}, pbuf, hseed_{b}, zero)",
            )
            w(
                indent,
                f"second_{b}(th1_{b}, g1_{b}, nega1_{b}, dT1_{b}, t1_{b}, p_local, s21_{b}, pbuf, hseed_{b}, zero)",
            )
            w(indent, f"for j in range({bp.n}):")
            w(indent + 1, f"erry_{b}[j] = k_{b}[7, j]")
            if auto:
                w(indent, f"eig_{b} = jinf_{b}(nega1_{b})")
                w(indent, f"jok_{b} = True")
        else:
            w(indent, f"for j in range({bp.n}):")
            w(indent + 1, f"g1_{b}[j] = tk_{b}[6, j]")
            w(indent + 1, f"erry_{b}[j] = te_{b}[j]")
            w(indent, f"tcurv_{b}(th1_{b}, g1_{b}, t1_{b}, p_local, s21_{b}, zero)")
            w(
                indent,
                f"tcoup_{b}(th1_{b}, t1_{b}, p_local, pseed_{b}, coup_{b}, jdiag_{b}, pbuf, zero)",
            )
            w(indent, f"trate_{b}(th1_{b}, tg6_{b}, tk_{b}, jdiag_{b})")
            if auto:
                w(indent, f"eig_{b} = teig_{b}(th1_{b}, tg6_{b}, tk_{b})")
                w(indent, f"jok_{b} = False")

    # -- advance block b until its nodes reach `target`, stepping its upstream first --
    def advance(b: int, target: str, active: str, indent: int) -> None:
        bp = plan.blocks[b]
        N = n_nodes[b]
        w(
            indent,
            f"# {block_name(b)} runs ahead on steps of its own until its dense output reaches {target}",
        )
        w(
            indent,
            f"need_{b} = {active} and nt_{b}[nn_{b} - 1] < {target} and nt_{b}[nn_{b} - 1] < tf and nn_{b} < {N} and n_steps < max_steps",
        )
        w(indent, f"while cuda.any_sync(lane_mask, need_{b}):")
        i1 = indent + 1
        w(
            i1,
            "# every lane with room steps with the warp, needed or not, so the lanes stay together",
        )
        w(
            i1,
            f"go_{b} = {active} and nt_{b}[nn_{b} - 1] < tf and nn_{b} < {N} and n_steps < max_steps",
        )
        w(i1, f"acc_{b} = False")
        w(i1, f"bail_{b} = False")
        for u in bp.reads:
            advance(u, f"nt_{b}[nn_{b} - 1] + dt_{b}", f"go_{b}", i1)
        meth = methods[b]
        ready = " and ".join([f"go_{b}"] + [f"nn_{u} >= 2" for u in bp.reads])
        w(i1, f"rdy_{b} = {ready}")
        if meth.kind == "auto":
            auto_vote(b, f"rdy_{b}", i1)
        w(i1, f"if rdy_{b}:")
        i2 = i1 + 1
        w(i2, f"t0_{b} = nt_{b}[nn_{b} - 1]")
        if meth.kind == "auto":
            w(i2, f"if mode_{b} and not jok_{b}:")
            w(
                i2 + 1,
                "# switched to Rodas5P: the node's Jacobian, which Tsit5 steps never formed",
            )
            w(
                i2 + 1,
                f"node_jac_{b}(th1_{b}, t0_{b}, p_local, nega1_{b}, dT1_{b}, pseed_{b}, coup_{b}, jdiag_{b})",
            )
            w(i2 + 1, f"jok_{b} = True")
        w(i2, f"for j in range({bp.size}):")
        w(i2 + 1, f"th_{b}[j] = th1_{b}[j]")
        w(i2 + 1, f"g0_{b}[j] = g1_{b}[j]")
        w(i2 + 1, f"s20_{b}[j] = s21_{b}[j]")
        if meth.implicit:
            ii = i2
            if meth.kind == "auto":
                w(i2, f"if mode_{b}:")
                ii = i2 + 1
            w(ii, f"for j in range({bp.size}):")
            w(ii + 1, f"dT0_{b}[j] = dT1_{b}[j]")
            w(ii, f"for c in range(lu_size_{b}):")
            w(ii + 1, f"nega_{b}[c] = nega1_{b}[c]")
        w(i2, f"h_{b} = dt_{b}")
        w(i2, f"while not acc_{b} and not bail_{b} and n_steps < max_steps:")
        i3 = i2 + 1
        w(i3, f"h_{b} = dt_{b}")
        w(i3, f"clamped_{b} = False")
        w(i3, f"if h_{b} > tf - t0_{b}:")
        w(i3 + 1, f"h_{b} = tf - t0_{b}")
        for u in bp.reads:
            w(i3, f"if h_{b} > nt_{u}[nn_{u} - 1] - t0_{b}:")
            w(i3 + 1, f"h_{b} = nt_{u}[nn_{u} - 1] - t0_{b}")
            w(i3 + 1, f"clamped_{b} = True")
        w(i3, f"if h_{b} < 1e-30:")
        w(i3 + 1, f"h_{b} = 1e-30")
        read_upstream(b, f"t0_{b}", f"h_{b}", i3)
        by_mode(b, i3, lambda kind, ind: attempt_upstream(b, kind, ind))
        if bp.reads:
            w(
                i3,
                "# a step cut short by an upstream's dense output does not lower the proposal",
            )
            w(i3, f"if acc_{b} and clamped_{b} and dt_new < dt_{b}:")
            w(i3 + 1, f"dt_new = dt_{b}")
        w(i3, f"dt_{b} = dt_new")
        w(i3, "n_steps += 1")
        if meth.kind == "auto":
            w(
                i3,
                "# a rejected attempt counts towards the switch too; an explicit lane that now",
            )
            w(
                i3,
                "# wants the implicit method leaves the retry loop so the warp can vote again",
            )
            w(i3, f"if not acc_{b}:")
            w(i3 + 1, f"if not mode_{b}:")
            w(i3 + 2, f"eig_{b} = teig_{b}(th1_{b}, tg6_{b}, tk_{b})")
            auto_update(b, i3 + 1)
            w(i3 + 1, f"if want_{b} and not mode_{b}:")
            w(i3 + 2, f"bail_{b} = True")
        w(i3, f"if acc_{b}:")
        w(i3 + 1, f"ep2_{b} = ep1_{b}")
        w(i3 + 1, f"ep1_{b} = min(safe_err, 1.0)")
        w(i3 + 1, f"lep1_{b} = min(lse, 0.0)")
        w(i2, f"if acc_{b}:")
        i3 = i2 + 1
        w(i3, f"t1_{b} = t0_{b} + h_{b}")
        by_mode(b, i3, lambda kind, ind: node_upstream(b, kind, ind))
        w(i3, f"nt_{b}[nn_{b}] = t1_{b}")
        w(i3, f"for j in range({bp.size}):")
        w(i3 + 1, f"ny_{b}[nn_{b}, j] = th1_{b}[j]")
        w(i3 + 1, f"ng_{b}[nn_{b}, j] = g1_{b}[j]")
        w(i3 + 1, f"ns2_{b}[nn_{b}, j] = s21_{b}[j]")
        w(i3, f"for j in range({bp.n}):")
        w(i3 + 1, f"nerr_{b}[nn_{b}, j] = erry_{b}[j]")
        if not bp.identity:
            w(
                i3,
                f"outputs_primal_{b}(th1_{b}, g1_{b}, s21_{b}, t1_{b}, p_local, qa_{b}, qa1_{b}, qa2_{b}, zero)",
            )
            if bp.ns > 0:
                w(
                    i3,
                    f"outputs_sens_{b}(th1_{b}, g1_{b}, t1_{b}, p_local, qa_{b}, qa1_{b}, pbuf, hseed_{b}, zero)",
                )
                w(
                    i3,
                    "# the sensitivities' curvatures: from the quintic through their values and rates at the",
                )
                w(
                    i3,
                    "# step's start, midpoint and end (a third-order sweep would give them directly)",
                )
                w(
                    i3,
                    f"midpoint_{b}(th_{b}, g0_{b}, s20_{b}, th1_{b}, g1_{b}, s21_{b}, h_{b}, ya_{b}, ga_{b})",
                )
                w(
                    i3,
                    f"outputs_sens_{b}(ya_{b}, ga_{b}, t0_{b} + 0.5 * h_{b}, p_local, Qm_{b}, Q1m_{b}, pbuf, hseed_{b}, zero)",
                )
                w(
                    i3,
                    f"outputs_curvature_{b}(nQ_{b}, nQ1_{b}, nn_{b} - 1, Qm_{b}, Q1m_{b}, qa_{b}, qa1_{b}, h_{b}, nQ2_{b}, nQ2l_{b}, nn_{b})",
                )
                w(i3, f"for j in range({bp.m}):")
                w(i3 + 1, f"nQ2l_{b}[nn_{b}, j] = qa2_{b}[j]")
            # the quantities' error from the block's own and its upstream's
            w(i3, f"for c in range({P}):")
            w(i3 + 1, "pbuf[c] = 0.0")
            for e in range(len(bp.reads)):
                w(i3, f"for c in range({n_params}, {P}):")
                w(i3 + 1, f"pbuf[c] += pseed_{b}[{e}, c]")
            w(
                i3,
                f"outputs_error_{b}(th1_{b}, t1_{b}, p_local, erry_{b}, pbuf, qb_{b}, zero)",
            )
            w(i3, f"for j in range({bp.mq}):")
            w(i3 + 1, f"nQ_{b}[nn_{b}, j] = qa_{b}[j]")
            w(i3 + 1, f"nQ1_{b}[nn_{b}, j] = qa1_{b}[j]")
            w(i3 + 1, f"nQ2_{b}[nn_{b}, j] = qa2_{b}[j]")
            w(i3, f"for j in range({bp.m}):")
            w(i3 + 1, f"nerrq_{b}[nn_{b}, j] = qb_{b}[j]")
        w(i3, f"nn_{b} += 1")
        w(i3, f"inh_{b} = 0.0")
        inherit(b, f"th1_{b}", f"t0_{b}", i3)
        if meth.kind == "auto":
            auto_update(b, i3)
        w(
            i1,
            f"need_{b} = (acc_{b} or bail_{b}) and nt_{b}[nn_{b} - 1] < {target} and nt_{b}[nn_{b} - 1] < tf and nn_{b} < {N} and n_steps < max_steps",
        )

    # -- the main loop: the sink's steps --
    s = S.b
    w(
        1,
        "# Every lane of the warp stays in the loop until all are done, so that the warp can vote.",
    )
    w(1, "while True:")
    w(2, "running = save_idx < n_save and t < tf and n_steps < max_steps")
    w(2, "if not cuda.any_sync(lane_mask, running):")
    w(3, "break")
    for u in S.reads:
        advance(u, f"t + dt_{s}", "running", 2)
    sm = methods[s]
    ready = " and ".join(["running"] + [f"nn_{u} >= 2" for u in S.reads])
    w(2, f"srdy = {ready}")
    if sm.kind == "auto":
        auto_vote(s, "srdy", 2)
    w(2, "if not srdy:")
    w(
        3,
        "continue  # this lane is done, or out of steps before its upstream could move",
    )
    w(2, f"dt_use = dt_{s}")
    w(2, "clamped = False")
    for u in S.reads:
        w(2, f"if dt_use > nt_{u}[nn_{u} - 1] - t:")
        w(3, f"dt_use = nt_{u}[nn_{u} - 1] - t")
        w(3, "clamped = True")
    w(2, "if dt_use < 1e-30:")
    w(3, "dt_use = 1e-30")
    w(2, "t_end = t + dt_use")
    read_upstream(s, "t", "dt_use", 2)

    def attempt_sink(kind: str, ind: int) -> None:
        if kind == "r":
            w(
                ind,
                f"step_{s}(y_{s}, u_{s}, work_{s}, dT_{s}, k_{s}, lu_{s}, ipiv_{s}, rhsb_{s}, p_local, t, dt_use, pbuf, hseed_{s}, zero, pseed_{s}, coup_{s}, jdiag_{s}, jn_{s})",
            )
            inherit(s, f"y_{s}", "t", ind)
            error_norm(
                s,
                f"y_{s}",
                lambda j: f"u_{s}[{j}] + k_{s}[7, {j}]",
                lambda j: f"k_{s}[7, {j}]",
                ind,
            )
        else:
            w(ind, f"if not tok_{s}:")
            w(ind + 1, f"write_{s}(y_{s}, t, p_local, tfb_{s}, pbuf, hseed_{s}, zero)")
            w(ind + 1, f"for j in range({S.n}):")
            w(ind + 2, f"tk_{s}[0, j] = tfb_{s}[j]")
            w(ind + 1, f"tok_{s} = True")
            w(
                ind,
                f"tcoup_{s}(y_{s}, t, p_local, pseed_{s}, coup_{s}, jdiag_{s}, pbuf, zero)",
            )
            w(
                ind,
                f"tstep_{s}(y_{s}, u_{s}, tg6_{s}, tk_{s}, tfb_{s}, p_local, t, dt_use, pbuf, hseed_{s}, zero)",
            )
            w(ind, f"terr_{s}(tk_{s}, dt_use, te_{s})")
            w(ind, f"trate_{s}(u_{s}, tg6_{s}, tk_{s}, jdiag_{s})")
            inherit(s, f"y_{s}", "t", ind)
            error_norm(
                s, f"y_{s}", lambda j: f"u_{s}[{j}]", lambda j: f"te_{s}[{j}]", ind
            )
        w(ind, "accept = err_norm <= 1.0 and not math.isnan(err_norm)")
        controller(s, "dt_use", ind, inherited=False, kind=kind)

    by_mode(s, 2, attempt_sink)
    w(2, "if accept:")
    w(3, f"ep2_{s} = ep1_{s}")
    w(3, f"ep1_{s} = own_err")
    w(3, f"lep1_{s} = lse")
    w(2, "# a step cut short by the dense output's end does not lower the proposal")
    w(2, f"if accept and clamped and dt_new < dt_{s}:")
    w(3, f"dt_new = dt_{s}")
    w(2, f"dt_{s} = dt_new")
    w(2, "if accept:")
    w(3, "t_old = t")

    def store_sink(
        value: str,
        ind: int,
        row: str = "save_idx",
        history: bool = SAVE_HISTORY,
        hook: bool = HAS_HOOK,
    ) -> None:
        w(ind, f"kk = jj // {S.n}")
        w(ind, f"r = jj - kk * {S.n}")
        w(
            ind,
            f"col = idx_{s}[r] if kk == 0 else {n_vars} + dirs_{s}[kk - 1] * {n_vars} + idx_{s}[r]",
        )
        if history:
            w(ind, f"hist[i, {row}, col] = {value}")
        if hook:
            w(ind, f"y_save[col] = {value}")

    def extension_coefficients(kind: str, ind: int) -> None:
        if kind != "r":
            return
        w(
            ind,
            "# the continuous extension's three stage combinations, once a step and only when it saves;",
        )
        w(ind, "# work and dT are free until the next step rewrites them")
        w(
            ind,
            "if save_idx < n_save and times[save_idx] <= t_end + 1e-12 * max(1.0, math.fabs(times[save_idx])):",
        )
        w(ind + 1, f"for jj in range({S.size}):")
        w(
            ind + 2,
            f"work_{s}[jj] = (25.948786856663858 * k_{s}[0, jj] - 2.5579724845846235 * k_{s}[1, jj]"
            f" + 10.433815404888879 * k_{s}[2, jj] - 2.3679251022685204 * k_{s}[3, jj]"
            f" + 0.524948541321073 * k_{s}[4, jj] + 1.1241088310450404 * k_{s}[5, jj]"
            f" + 0.4272876194431874 * k_{s}[6, jj] - 0.17202221070155493 * k_{s}[7, jj])",
        )
        w(
            ind + 2,
            f"dT_{s}[jj] = (-9.91568850695171 * k_{s}[0, jj] - 0.9689944594115154 * k_{s}[1, jj]"
            f" + 3.0438037242978453 * k_{s}[2, jj] - 24.495224566215796 * k_{s}[3, jj]"
            f" + 20.176138334709044 * k_{s}[4, jj] + 15.98066361424651 * k_{s}[5, jj]"
            f" - 6.789040303419874 * k_{s}[6, jj] - 6.710236069923372 * k_{s}[7, jj])",
        )
        w(
            ind + 2,
            f"h3_{s}[jj] = (11.419903575922262 * k_{s}[0, jj] + 2.8879645146136994 * k_{s}[1, jj]"
            f" + 72.92137995996029 * k_{s}[2, jj] + 80.12511834622643 * k_{s}[3, jj]"
            f" - 52.072871366152654 * k_{s}[4, jj] - 59.78993625266729 * k_{s}[5, jj]"
            f" - 0.15582684282751913 * k_{s}[6, jj] + 4.883087185713722 * k_{s}[7, jj])",
        )

    def sink_save(kind: str, ind: int) -> None:
        if kind == "r":
            w(ind, "# the sink: the Rosenbrock continuous extension")
            w(ind, f"for jj in range({S.size}):")
            w(ind + 1, f"y_new = u_{s}[jj] + k_{s}[7, jj]")
            w(
                ind + 1,
                f"val = theta1 * y_{s}[jj] + theta * (y_new + theta1 * (work_{s}[jj] + theta * (dT_{s}[jj] + theta * h3_{s}[jj])))",
            )
            store_sink("val", ind + 1)
        else:
            w(ind, "# the sink: Tsit5's continuous extension")
            w(ind, f"tdense_{s}(y_{s}, tk_{s}, dt_use, theta, tfb_{s})")
            w(ind, f"for jj in range({S.size}):")
            store_sink(f"tfb_{s}[jj]", ind + 1)

    def advance_sink(kind: str, ind: int) -> None:
        auto = sm.kind == "auto"
        if kind == "r":
            if auto:
                w(ind, f"eig_{s} = jn_{s}[0]")
                w(ind, f"tok_{s} = False")
            w(ind, f"for jj in range({S.size}):")
            w(ind + 1, f"y_{s}[jj] = u_{s}[jj] + k_{s}[7, jj]")
        else:
            if auto:
                w(ind, f"eig_{s} = teig_{s}(u_{s}, tg6_{s}, tk_{s})")
            w(ind, f"for jj in range({S.size}):")
            w(ind + 1, f"y_{s}[jj] = u_{s}[jj]")
            w(ind, f"for jj in range({S.n}):")
            w(ind + 1, f"tk_{s}[0, jj] = tk_{s}[6, jj]  # first same as last")

    if sm.kind == "rodas":
        extension_coefficients("r", 3)
    elif sm.kind == "auto":
        w(3, f"if mode_{s}:")
        extension_coefficients("r", 4)
    w(
        3,
        "while save_idx < n_save and times[save_idx] <= t_end + 1e-12 * max(1.0, math.fabs(times[save_idx])):",
    )
    w(4, "theta = (times[save_idx] - t_old) / dt_use")
    w(4, "theta1 = 1.0 - theta")
    by_mode(s, 4, sink_save)
    for bp in plan.blocks:
        if bp.sink:
            continue
        b = bp.b
        w(4, f"# {block_name(b)}: its dense output, state and sensitivities")
        w(
            4,
            f"dense_y_{b}(nt_{b}, ny_{b}, ng_{b}, ns2_{b}, ns2_{b}, nn_{b}, times[save_idx], ya_{b}, ga_{b}, s2a_{b})",
        )
        emit_block_save(w, plan, bp, f"ya_{b}", 4, SAVE_HISTORY, HAS_HOOK)
    if HAS_HOOK:
        w(4, "save_hook(save_idx, y_save, times[save_idx], p_local, hook_out[i])")
    w(4, "save_idx += 1")
    by_mode(s, 3, advance_sink)
    w(3, "t += dt_use")
    w(3, "accepted += 1")
    if sm.kind == "auto":
        auto_update(s, 3)
    w(3, "# drop the nodes the sink has passed")
    for bp in plan.blocks:
        if bp.sink:
            continue
        b = bp.b
        w(3, f"while nn_{b} > 2 and nt_{b}[1] <= t:")
        w(4, f"for q in range(nn_{b} - 1):")
        w(5, f"nt_{b}[q] = nt_{b}[q + 1]")
        w(5, f"for j in range({bp.size}):")
        w(6, f"ny_{b}[q, j] = ny_{b}[q + 1, j]")
        w(6, f"ng_{b}[q, j] = ng_{b}[q + 1, j]")
        w(6, f"ns2_{b}[q, j] = ns2_{b}[q + 1, j]")
        w(5, f"for j in range({bp.n}):")
        w(6, f"nerr_{b}[q, j] = nerr_{b}[q + 1, j]")
        if not bp.identity:
            w(5, f"for j in range({bp.mq}):")
            w(6, f"nQ_{b}[q, j] = nQ_{b}[q + 1, j]")
            w(6, f"nQ1_{b}[q, j] = nQ1_{b}[q + 1, j]")
            w(6, f"nQ2_{b}[q, j] = nQ2_{b}[q + 1, j]")
            if bp.ns > 0:
                w(6, f"nQ2l_{b}[q, j] = nQ2l_{b}[q + 1, j]")
            w(5, f"for j in range({bp.m}):")
            w(6, f"nerrq_{b}[q, j] = nerrq_{b}[q + 1, j]")
        w(4, f"nn_{b} -= 1")
    w(2, "else:")
    w(3, "rejected += 1")
    if sm.kind == "auto":
        w(
            3,
            "# a rejected attempt counts towards the switch too (its estimate is the attempt's own)",
        )
        w(3, f"if mode_{s}:")
        w(4, f"eig_{s} = jn_{s}[0]")
        w(3, f"if not mode_{s}:")
        w(4, f"eig_{s} = teig_{s}(u_{s}, tg6_{s}, tk_{s})")
        auto_update(s, 3)
    w(2, "n_steps += 1")

    # -- after the loop: the blocks' states where the sink stopped --
    for bp in plan.blocks:
        if bp.sink:
            continue
        b = bp.b
        w(1, f"if nn_{b} > 1:")
        w(
            2,
            f"dense_y_{b}(nt_{b}, ny_{b}, ng_{b}, ns2_{b}, ns2_{b}, nn_{b}, t, ya_{b}, ga_{b}, s2a_{b})",
        )
        w(1, "else:")
        w(2, f"for j in range({bp.size}):")
        w(3, f"ya_{b}[j] = th1_{b}[j]")
    w(1, "# save times past this trajectory's own end hold its final state")
    w(1, "while save_idx < n_save:")
    w(2, f"for jj in range({S.size}):")
    store_sink(f"y_{s}[jj]", 3)
    for bp in plan.blocks:
        if not bp.sink:
            emit_block_save(w, plan, bp, f"ya_{bp.b}", 2, SAVE_HISTORY, HAS_HOOK)
    if HAS_HOOK:
        w(2, "save_hook(save_idx, y_save, times[save_idx], p_local, hook_out[i])")
    w(2, "save_idx += 1")
    if not SAVE_HISTORY:
        w(1, f"for jj in range({S.size}):")
        store_sink(f"y_{s}[jj]", 2, row="0", history=True, hook=False)
        for bp in plan.blocks:
            if not bp.sink:
                emit_block_save(w, plan, bp, f"ya_{bp.b}", 1, True, False, row="0")
    w(1, "accepted_out[i] = accepted")
    w(1, "rejected_out[i] = rejected")
    w(1, "loop_out[i] = n_steps")
    return "\n".join(src.lines) + "\n"


def emit_block_save(
    w,
    plan: _Plan,
    bp: _BlockPlan,
    values: str,
    indent: int,
    save_history: bool,
    has_hook: bool,
    row: str = "save_idx",
) -> None:
    """Write a block's dense-output values (state and every user direction) into the history row and the hook's state."""
    b, n, n_vars = bp.b, bp.n, plan.n_vars
    w(indent, f"for r in range({n}):")
    if save_history:
        w(indent + 1, f"hist[i, {row}, idx_{b}[r]] = {values}[r]")
    if has_hook:
        w(indent + 1, f"y_save[idx_{b}[r]] = {values}[r]")
    if plan.n_sens:
        w(indent, f"for k in range({plan.n_sens}):")
        w(indent + 1, f"kk = pos_{b}[k]")
        w(indent + 1, f"for r in range({n}):")
        w(indent + 2, f"val = {values}[{n} + kk * {n} + r] if kk >= 0 else 0.0")
        if save_history:
            w(indent + 2, f"hist[i, {row}, {n_vars} + k * {n_vars} + idx_{b}[r]] = val")
        if has_hook:
            w(indent + 2, f"y_save[{n_vars} + k * {n_vars} + idx_{b}[r]] = val")


@functools.cache
def make_hierarchy_kernel(
    hierarchy: Hierarchy,
    n_params: int,
    options,
    layout: tuple[tuple[int, ...], ...] | None = None,
    patterns: tuple[Pattern | None, ...] | None = None,
):
    """Rodas5P over a hierarchy of blocks, one trajectory per thread.

    ``options`` is the :class:`modax.rodas5P.KernelOptions`. Returns
    ``(kernel, trajectories_per_block)``. The kernel takes the standard argument
    list and writes the history in the user's layout.
    """
    from modax import rodas5P as R

    spec: SensitivitySpec | None = options.spec
    plan = plan_hierarchy(hierarchy, n_params, spec, layout, patterns)
    tpb = int(options.trajectories_per_block)
    # numba's SSA pass recurses once per basic block, and this kernel has many.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    namespace: dict[str, Any] = dict(
        cuda=cuda,
        np=np,
        types=types,
        math=math,
        SAFETY=SAFETY,
        GAMMA=R.GAMMA,
        TPB=tpb,
    )
    methods = tuple(resolve_method(bp, options) for bp in plan.blocks)
    for bp, meth in zip(plan.blocks, methods):
        # each block's controller: its method's error order, gains and step-ratio bounds
        for kind, gains, order, fmin, fmax in (
            ("r", meth.rodas_gains, R.ORDER, R.FACTOR_MIN, R.FACTOR_MAX),
            ("t", meth.tsit_gains, T.ORDER, T.FACTOR_MIN, T.FACTOR_MAX),
        ):
            c = f"{bp.b}{kind}"
            namespace[f"E1_{c}"], namespace[f"E2_{c}"], namespace[f"E3_{c}"] = (
                gains.exponents(order)
            )
            namespace[f"FMIN_{c}"], namespace[f"FMAX_{c}"] = fmin, fmax
    if options.save_hook is not None:
        namespace["save_hook"] = as_cuda_device(options.save_hook)
    n_nodes = tuple(
        NODES if (spec is None or bp.ns == 0) else NODES_SENS for bp in plan.blocks
    )
    for bp in plan.blocks:
        b = bp.b
        namespace.update(_block_functions(plan, bp, options, R, methods[b]))
        if not bp.sink:
            namespace[f"dense_y_{b}"] = dense_eval_fn(bp.size)
            namespace[f"dense_q_{b}"] = dense_eval_fn(bp.mq) if bp.m else None
            namespace[f"midpoint_{b}"] = midpoint_eval_fn(bp.size)
        namespace[f"idx_{b}"] = np.asarray(bp.idx, dtype=np.int32)
        namespace[f"dirs_{b}"] = _table(bp.dirs)
        namespace[f"pos_{b}"] = _table(bp.pos)
        namespace[f"ipiv_size_{b}"] = namespace.pop(f"ipiv_{b}", 1)
        for u in bp.reads:
            namespace[f"hermite_{u}_{b}"] = _hermite_writer(plan, u, b)
    # const arrays are read through cuda.const.array_like(name) in the source
    source = _emit_kernel(plan, options, n_nodes, methods)
    source = _const_reads(
        source,
        [f"idx_{b}" for b in range(len(plan.blocks))]
        + [f"dirs_{b}" for b in range(len(plan.blocks))]
        + [f"pos_{b}" for b in range(len(plan.blocks))],
    )
    if os.environ.get("MODAX_HIERARCHY_DUMP"):
        print(source)
    jit_options: dict[str, Any] = (
        {}
        if options.max_registers is None
        else {"max_registers": options.max_registers}
    )
    if os.environ.get("MODAX_SPLIT_DUMP_MLIR"):
        jit_options["dump_mlir"] = True
    kernel = _compile_kernel(source, namespace, jit_options)
    if options.cache:
        enable_content_cache(kernel)
    return kernel, tpb


def _const_reads(source: str, names: list[str]) -> str:
    """Bind the index tables as constant-memory arrays at the kernel's start."""
    lines = source.split("\n")
    head, body = lines[:1], lines[1:]
    binds = [f"    {name} = cuda.const.array_like(TABLE_{name})" for name in names]
    return "\n".join(head + binds + body)


def _compile_kernel(
    source: str, namespace: dict[str, Any], jit_options: dict[str, Any]
):
    import hashlib
    import linecache

    for name in list(namespace):
        if name.startswith(("idx_", "dirs_", "pos_")):
            namespace[f"TABLE_{name}"] = namespace.pop(name)
    filename = (
        f"<modax generated kernel {hashlib.sha256(source.encode()).hexdigest()[:12]}>"
    )
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict[str, Any] = {}
    exec(compile(source, filename, "exec"), namespace, scope)  # noqa: S102
    kernel = scope["kernel"]
    kernel._generated_source = source
    return cuda.jit(**jit_options)(kernel)
