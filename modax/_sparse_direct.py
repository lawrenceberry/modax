"""A compiled sparse direct solver for the Rodas5P iteration matrix.

``dense_lu_solver`` factorises ``M = I/(h*gamma) - J`` as if it were dense,
which costs ``n_vars ** 3 / 3`` operations however few nonzeros ``M`` has. A
hand-written solver such as DISCO-EB's Schur block-LU does far better, but
only for the one structure it targets. This module compiles a direct solver
for whatever *sparsity pattern* the kernel receives, assuming no structure.

The host runs the analysis once, at kernel build time:

1. **Order.** SuiteSparse's AMD takes the symmetrised pattern and returns a
   fill-reducing permutation (see
   :py:obj:`fill_reducing_order <modax._sparse_direct.fill_reducing_order>`
   for why AMD and not COLAMD).
2. **Factor symbolically.** A pattern-only Gaussian elimination
   (:py:obj:`fill_pattern <modax._sparse_direct.fill_pattern>`) gives the
   exact pattern of ``L + U``, fill-in included, with no numbers and no
   device.
3. **Lay out.** That pattern becomes one CSR image of ``L + U``, and its
   ``(row, col) -> slot`` map becomes the
   :py:obj:`CompressedJacobian <modax._sparsity.CompressedJacobian>` the AD
   sweeps write into. The buffer holds exactly ``nnz(L + U)`` elements.
4. **Compile.** The factorisation and the two triangular solves become
   ``cuda.jit(device=True)`` functions with the same
   ``factorize_local(lu, ipiv)`` / ``solve_local(lu, ipiv, rhs)`` shape as
   :py:obj:`modax.rodas5P.dense_lu_solver`, so the kernel has no branch.
   Small structures get straight-line code with every slot a literal. Larger
   ones fall back to loops over index tables in constant memory.

Every thread runs the same pattern, so neither form causes divergence.

**Why straight-line code matters.** Table-driven, a sparse routine spends a
broadcast load on each value's index before it can load the value. A large
ensemble hides that latency, but DISCO-EB's single-cosmology case is 128
trajectories (four warps on a 46-SM device), and nothing hides it. Literal
indices remove the dependency without costing registers, since the kernel
keeps both the matrix and the right-hand side in local memory anyway.
Measured on DISCO-EB at N128: 528 ms table-driven, 419 ms with the solves
unrolled, 398 ms with the factorisation unrolled too, against 509 ms for the
hand-written Schur solver it replaced.

**No pivoting.** The compiler fixes the pattern, and every thread shares
it, so rows cannot swap based on the numbers. The *symmetric*
permutation keeps ``M``'s diagonal on the diagonal, where ``I/(h*gamma)``
puts a pivot that grows as the step shrinks. Rodas5P also satisfies the
Rosenbrock-*W* order conditions to second order, so an approximate
factorisation costs order and step size, not convergence. A pivot of exactly
zero leaves an infinity, the error norm goes to NaN, the controller rejects
the step, and the retry has a larger ``1/(h*gamma)`` on that diagonal.
``dense_lu_solver`` treats a singular column the same way.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import ClassVar, TypeAlias

import numpy as np
import numpy.typing as npt
from numba_cuda_mlir import cuda
from numba_cuda_mlir.descriptor import MLIRDispatcher

from modax._codegen import compile_device_source
from modax._methods import Ordering
from modax._sparsity import (
    CompressedJacobian,
    Pattern,
    Sparsity,
    colour_sparsity,
    normalize_sparsity,
)

IndexTable: TypeAlias = npt.NDArray[np.int32]
"""A table of slots or offsets, ``int32`` because that is what the device reads."""

# AMD, or nothing at all. CHOLMOD's orderings were reachable here while
# scikit-sparse was a dependency, and measuring them is what retired it: over
# arrow, ring and bordered-block patterns "colamd", "nesdis" and "best" each
# returned AMD's own fill to the entry -- on a symmetric analysis CHOLMOD runs
# AMD for "colamd" anyway -- and "metis" was strictly worse wherever it
# differed, 660 nonzeros against 484 on the Einstein-Boltzmann-like case, losing
# the perfect elimination order AMD finds there. Nested dissection wins on
# meshes far larger than the tens-to-few-hundred variables this solver is for,
# and it was the only thing a package with no wheels was buying.
ORDERINGS: tuple[Ordering, ...] = ("amd", "natural")


def fill_reducing_order(
    pattern: Pattern, ordering: Ordering = "amd"
) -> tuple[int, ...]:
    """A permutation of the variables that keeps ``L + U`` small.

    ``ordering="amd"`` runs AMD on the symmetrised pattern ``S + S.T``, and
    ``"natural"`` returns the identity. :data:`ORDERINGS` lists both.

    AMD rather than COLAMD because COLAMD bounds fill under whatever row
    permutation partial pivoting later picks, and there is no pivoting here.
    COLAMD's one-sided permutation also moves the diagonal, which this
    factorisation needs where ``I/(h*gamma)`` puts it. AMD approximately
    minimises the Cholesky fill of ``S + S.T``, the standard bound on an
    unpivoted LU of ``S``, with a symmetric permutation ``P S P.T`` that keeps
    the diagonal in place.

    On DISCO-EB's Einstein-Boltzmann Jacobian AMD finds a *perfect* elimination
    order, with ``nnz(L + U) == nnz(J)``. It matches the hand-written Schur
    solver's order: peel each free-streaming hierarchy from its truncated end
    inwards, then eliminate the dense core last.

    AMD comes from cvxopt, which ships it in a manylinux wheel.

    Raises:
        ValueError: If ``ordering`` is not in :data:`ORDERINGS`.
        ImportError: If ``ordering="amd"`` and cvxopt is not installed.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; expected one of {ORDERINGS}")
    n = len(pattern)
    if ordering == "natural":
        return tuple(range(n))
    try:
        from cvxopt import amd, spmatrix  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the sparse direct solver orders its variables with SuiteSparse's "
            "AMD, which reaches it through cvxopt. Install it, or pass "
            "ordering='natural' to skip the ordering entirely."
        ) from exc

    # ``amd.order`` reads the *lower triangle* of a symmetric matrix and wants
    # the pattern alone, so the symmetrised pattern goes in with a unit on every
    # entry. Nothing has to be made positive definite for it, which CHOLMOD's
    # route did need: that reached the permutation through an actual
    # factorisation, and the numbers had to survive it.
    entries = {(r, c) for r, cs in enumerate(pattern) for c in cs}
    entries |= {(c, r) for r, c in entries}
    entries |= {(i, i) for i in range(n)}
    lower = sorted((r, c) for r, c in entries if r >= c)
    matrix = spmatrix(1.0, [r for r, _ in lower], [c for _, c in lower], (n, n))
    return tuple(int(i) for i in amd.order(matrix))


def fill_pattern(pattern: Pattern) -> npt.NDArray[np.bool_]:
    """The exact ``L + U`` pattern of an unpivoted LU, as a boolean matrix.

    Runs Gaussian elimination on the pattern alone. Eliminating column ``k``
    makes every row below it inherit row ``k``'s entries right of the diagonal.
    The result is exactly what the numeric factorisation touches. Factorising a
    sample matrix instead could drop an entry that cancels for those numbers,
    leaving the buffer a slot short with no way for the kernel to notice.

    The elimination is dense, one numpy row OR per update, so it takes
    ``O(n ** 3)`` time and ``O(n ** 2)`` bytes at worst. A 200-variable pattern
    analyses in single-digit milliseconds, so a sparse symbolic factorisation
    would gain nothing.
    """
    n = len(pattern)
    filled = np.zeros((n, n), dtype=bool)
    for row, cols in enumerate(pattern):
        filled[row, list(cols)] = True
    # I/(h*gamma) lands on every diagonal whether or not J has anything there,
    # so the diagonal is part of the pattern being factorised.
    np.fill_diagonal(filled, True)
    for k in range(n):
        below = np.flatnonzero(filled[k + 1 :, k]) + k + 1
        if below.size:
            filled[below, k + 1 :] |= filled[k, k + 1 :]
    return filled


@dataclass(frozen=True)
class SparseLULayout:
    """One CSR image of ``L + U``, in elimination order.

    CSR because all three routines read by rows: the up-looking factorisation,
    the forward substitution and the back substitution.

    ``L`` (strictly left of the diagonal) and ``U`` (from the diagonal
    rightwards) share the image, because the factorisation runs in place and
    ``L``'s unit diagonal needs no storage. ``row_ptr`` and ``diag_ptr`` bracket
    the two halves of each row.

    Indices are *elimination* indices: row ``i`` here is variable ``order[i]``
    of the caller's system. :attr:`row_origin` and :attr:`col_origin` translate
    back, so the device code never permutes a vector. It visits rows in
    elimination order and reads and writes the right-hand side at the caller's
    indices.
    """

    n_vars: int
    order: tuple[int, ...]
    row_ptr: IndexTable
    col_ind: IndexTable
    diag_ptr: IndexTable

    @property
    def nnz(self) -> int:
        """Elements in one trajectory's ``L + U``, the whole footprint."""
        return int(self.row_ptr[-1])

    @property
    def row_origin(self) -> IndexTable:
        """Caller's index of each elimination row."""
        return np.asarray(self.order, dtype=np.int32)

    @property
    def col_origin(self) -> IndexTable:
        """Caller's index of the column each slot belongs to."""
        return self.row_origin[self.col_ind]

    def _elimination_slots(self) -> npt.NDArray[np.int64]:
        """``(n_vars, n_vars)`` of slots by elimination index, ``-1`` elsewhere."""
        table = np.full((self.n_vars, self.n_vars), -1, dtype=np.int64)
        rows = np.repeat(np.arange(self.n_vars), np.diff(self.row_ptr))
        table[rows, self.col_ind] = np.arange(self.nnz)
        return table

    def slot_table(self) -> npt.NDArray[np.int64]:
        """``(n_vars, n_vars)`` of slots, ``-1`` where the structure has nothing."""
        table = np.empty((self.n_vars, self.n_vars), dtype=np.int64)
        table[np.ix_(self.order, self.order)] = self._elimination_slots()
        return table

    def factorization_ops(self) -> tuple[IndexTable, IndexTable, IndexTable]:
        """The rank-one updates, addressed off the multiplier that drives them.

        Up-looking LU visits row ``i`` and, for each entry ``(i, k)`` left of the
        diagonal in increasing ``k``, forms the multiplier ``L[i, k]`` and subtracts
        ``L[i, k] * U[k, k:]`` from row ``i``. With the pattern fixed, the slot each
        subtraction writes never changes either, so this resolves on the host the search a
        runtime sparse solver spends its time on.

        The sources need no table. They are row ``k``'s upper entries, which CSR
        stores contiguously, so the inner loop runs over ``pivot_of[p] + 1 ..
        upper_end[p]``. That leaves one index load per multiply-add rather than
        three.

        Returns:
            ``(dst_start, destination, upper_end)``, indexed by the multiplier's
            slot ``p``. One multiplier's destinations are contiguous and in source
            order, so the inner loop walks ``destination`` from ``dst_start[p]``
            alongside the sources.
        """
        n = self.n_vars
        dst_start = np.zeros(self.nnz, dtype=np.int64)
        upper_end = np.asarray(self.row_ptr[1:], dtype=np.int64)[
            np.asarray(self.col_ind, dtype=np.int64)
        ]
        dst: list[int] = []
        table = self._elimination_slots()
        for i in range(n):
            for p in range(int(self.row_ptr[i]), int(self.diag_ptr[i])):
                k = int(self.col_ind[p])
                dst_start[p] = len(dst)
                for q in range(int(self.diag_ptr[k]) + 1, int(self.row_ptr[k + 1])):
                    dst.append(int(table[i, int(self.col_ind[q])]))
        return (
            dst_start.astype(np.int32),
            np.asarray(dst, dtype=np.int32),
            upper_end.astype(np.int32),
        )

    def flop_counts(self) -> tuple[int, int, int]:
        """``(multipliers, updates, substitutions)``.

        The first two count one factorisation, the third one solve (both
        substitutions). They decide between straight-line and looped code, so they
        come from the structure rather than from building the code.
        """
        lower = self.diag_ptr - self.row_ptr[:-1]
        upper = self.row_ptr[1:] - self.diag_ptr - 1
        multipliers = int(lower.sum())
        updates = int(upper[self.col_ind[_lower_mask(self)]].sum())
        return multipliers, updates, multipliers + int(upper.sum())


def _lower_mask(layout: SparseLULayout) -> npt.NDArray[np.bool_]:
    """True at the slots strictly left of their row's diagonal."""
    slot = np.arange(layout.nnz)
    row = np.repeat(np.arange(layout.n_vars), np.diff(layout.row_ptr))
    return slot < layout.diag_ptr[row]


def analyse(pattern: Pattern, ordering: Ordering = "amd") -> SparseLULayout:
    """Order, factorise symbolically, and lay out ``L + U`` in CSR.

    ``pattern`` is the normalised form, one tuple of column indices per row, as
    :py:obj:`modax._sparsity.normalize_sparsity` returns it.
    """
    n_vars = len(pattern)
    order = fill_reducing_order(pattern, ordering)
    # The elimination happens in the permuted matrix, so the pattern is
    # relabelled *before* the symbolic pass rather than after it: a permutation
    # of the fill is not the fill of the permutation, and the whole point of the
    # ordering is that it changes what fills in.
    forward = np.asarray(order, dtype=np.int64)
    inverse = np.empty(n_vars, dtype=np.int64)
    inverse[forward] = np.arange(n_vars)
    filled = fill_pattern(
        tuple(
            tuple(sorted(int(inverse[c]) for c in pattern[int(forward[i])]))
            for i in range(n_vars)
        )
    )
    row_ptr = np.zeros(n_vars + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(filled.sum(axis=1))
    col_ind = np.nonzero(filled)[1].astype(np.int32)
    diag_ptr = (row_ptr[:-1] + np.tril(filled, -1).sum(axis=1)).astype(np.int32)
    return SparseLULayout(
        n_vars=n_vars,
        order=tuple(int(i) for i in order),
        row_ptr=row_ptr,
        col_ind=col_ind,
        diag_ptr=diag_ptr,
    )


def compressed_jacobian(pattern: Pattern, layout: SparseLULayout) -> CompressedJacobian:
    """The colour-compressed layout whose slots are ``layout``'s CSR slots.

    The colouring comes from the caller's pattern as usual, so the sweeps still
    cost ``n_colours + 1``. Each ``(row, colour)`` writes to the
    factorisation's slot, so the AD deposits ``-J`` straight into the buffer the
    factorisation works in. No sweep writes the fill-in slots, which is what the
    kernel's ``needs_clear`` handles.

    Because the fill has its own slots, the pattern only needs the true
    nonzeros and can colour as tightly as it really is. A solver that owned its
    own buffer had to declare its fill-in too, which cost DISCO-EB 12 colours
    rather than 11.

    Raises:
        ValueError: If a pattern entry is missing from ``layout``, which means
            the two came from different patterns.
    """
    compressed = colour_sparsity(pattern)
    table = layout.slot_table()
    packed = np.full(layout.n_vars * compressed.n_colours, -1, dtype=np.int64)
    for row, cols in enumerate(pattern):
        for col in cols:
            slot = int(table[row, col])
            if slot < 0:
                raise ValueError(
                    f"entry ({row}, {col}) is in the pattern but not in the "
                    "factorised structure, which cannot happen unless the two "
                    "were built from different patterns"
                )
            packed[row * compressed.n_colours + compressed.colour[col]] = slot
    return replace(
        compressed,
        packed=tuple(int(s) for s in packed),
        packed_diagonal=tuple(int(table[i, i]) for i in range(layout.n_vars)),
        n_slots=layout.nnz,
    )


@dataclass(frozen=True, repr=False)
class SparseDirectSolver:
    """A Rodas5P linear solver compiled for one sparsity pattern.

    ``rodas5P.solve`` builds this from its ``sparsity``, so callers never hold
    one. :attr:`compressed` is both the layout the solver defines and the one
    the kernel fills, so the Enzyme sweeps and the factorisation agree on where
    each entry lives.

    It never pivots, so :attr:`ipiv_size` is 1. The kernel still allocates the
    array and the device functions still take it, to match the dense solver's
    shape.
    """

    layout: SparseLULayout
    compressed: CompressedJacobian
    factorize_local: MLIRDispatcher
    solve_local: MLIRDispatcher
    ipiv_size: ClassVar[int] = 1

    @property
    def nnz(self) -> int:
        return self.layout.nnz

    def __repr__(self) -> str:
        multipliers, updates, substitutions = self.layout.flop_counts()
        return (
            f"<SparseDirectSolver n_vars={self.layout.n_vars} nnz={self.nnz} "
            f"factorise={multipliers + updates} solve={substitutions} "
            f"colours={self.compressed.n_colours}>"
        )


@functools.cache
def sparse_direct_solver_for(
    pattern: Pattern, ordering: Ordering = "amd", inline: str = "always"
) -> SparseDirectSolver:
    """Compile a direct sparse solver for a *normalised* pattern.

    The kernel builder calls this, having already normalised ``sparsity``. It
    caches the result on the pattern, so a repeat call returns the same compiled
    device functions and the kernel's own cache hits.
    """
    layout = analyse(pattern, ordering)
    return SparseDirectSolver(
        layout,
        compressed_jacobian(pattern, layout),
        _make_factorize(layout, inline),
        _make_solve(layout, inline),
    )


def sparse_direct_solver(
    sparsity: Sparsity, n_vars: int, *, ordering: Ordering = "amd"
) -> SparseDirectSolver:
    """Compile a direct sparse solver for ``sparsity``.

    ``sparsity`` takes the same forms as in ``solve``: an ``(n_vars, n_vars)``
    mask, a scipy sparse matrix, or an ``(nnz, 2)`` index array. It must be a
    superset of the true nonzeros of ``J``. It need *not* include the fill-in,
    which this computes.
    """
    return sparse_direct_solver_for(normalize_sparsity(sparsity, n_vars), ordering)


# --- the compiled device functions -------------------------------------------
# The index tables live in constant memory, where every thread of a warp reads
# the same address at the same time and the read broadcasts. Each device function
# binds *one* packed table: numba emits a separate copy of a closed-over array
# per reference site, and the 64 KiB constant window is shared with the colour
# seed table and everything the caller's right-hand side closes over.


# Above these many multiply-adds the routines stay table-driven. Straight-line
# code is faster wherever it fits -- it spends no broadcast load on an index the
# analysis already knows -- but it is instruction cache, and the solves are
# emitted eight times over, once per Rosenbrock stage. The factorisation appears
# once a step, so it may be the larger of the two.
MAX_UNROLLED_SUBSTITUTIONS = 2048
MAX_UNROLLED_UPDATES = 8192


def _make_factorize(layout: SparseLULayout, inline: str = "always") -> MLIRDispatcher:
    """Up-looking LU of ``M``, in place, with no pivoting.

    Row ``i`` finishes before row ``i + 1`` starts, so row ``k`` already holds
    its final ``U`` when the multiplier ``L[i, k]`` forms. Within a row, entries
    left of the diagonal go in increasing column order, the order their own
    updates arrive in.

    The diagonal ends up holding ``1 / U[i, i]``. That costs one divide here
    and saves ``n_vars`` divides in each of the eight back substitutions per
    step.

    Small structures get this unrolled, every slot a literal, for the reasons
    ``_make_solve`` gives. Table-driven, the inner loop runs over row ``k``'s
    contiguous upper entries, so only the destination costs a broadcast load
    per multiply-add.

    A zero pivot produces an infinity rather than an error. The step
    controller rejects the step, and the smaller retry puts a larger
    ``1/(h*gamma)`` on this very diagonal.
    """
    n = layout.n_vars
    dst_start, destination, upper_end = layout.factorization_ops()
    # Slot of the pivot each multiplier divides by: diag_ptr[col_ind[p]],
    # resolved here so the factorisation never touches col_ind.
    pivot_of = layout.diag_ptr[layout.col_ind]

    if destination.size <= MAX_UNROLLED_UPDATES:
        return _unrolled_factorize(
            layout, dst_start, destination, pivot_of, upper_end, inline
        )

    tables, (ROW, DIAG, PIVOT, START, END, DST) = _pack(
        layout.row_ptr, layout.diag_ptr, pivot_of, dst_start, upper_end, destination
    )

    @cuda.jit(device=True)
    def sparse_direct_factorize(lu, ipiv):
        t = cuda.const.array_like(tables)
        for i in range(n):
            pivot = t[DIAG + i]
            for p in range(t[ROW + i], pivot):
                above = t[PIVOT + p]
                factor = lu[p] * lu[above]
                lu[p] = factor
                q = DST + t[START + p]
                for s in range(above + 1, t[END + p]):
                    lu[t[q]] -= factor * lu[s]
                    q += 1
            lu[pivot] = 1.0 / lu[pivot]

    return sparse_direct_factorize


def _unrolled_factorize(
    layout: SparseLULayout,
    dst_start: IndexTable,
    destination: IndexTable,
    pivot_of: IndexTable,
    upper_end: IndexTable,
    inline: str = "always",
) -> MLIRDispatcher:
    """Generate the factorisation with every slot spelled out as a literal.

    Each multiplier gets its own name, so the emitted code carries only the real
    dependencies and the compiler can schedule across them.
    """
    n = layout.n_vars
    row_ptr, diag_ptr = layout.row_ptr, layout.diag_ptr
    lines = ["def sparse_direct_factorize(lu, ipiv):"]
    for i in range(n):
        for p in range(int(row_ptr[i]), int(diag_ptr[i])):
            above = int(pivot_of[p])
            lines.append(f"    f{p} = lu[{p}] * lu[{above}]")
            lines.append(f"    lu[{p}] = f{p}")
            q = int(dst_start[p])
            for s in range(above + 1, int(upper_end[p])):
                lines.append(f"    lu[{int(destination[q])}] -= f{p} * lu[{s}]")
                q += 1
        pivot = int(diag_ptr[i])
        lines.append(f"    lu[{pivot}] = 1.0 / lu[{pivot}]")
    return compile_device_source("sparse_direct_factorize", lines, inline=inline)


def _make_solve(layout: SparseLULayout, inline: str = "always") -> MLIRDispatcher:
    """The two triangular solves: ``M x = rhs`` in place, sparse both ways.

    Each sweep touches only the nonzeros, ``nnz(L) + nnz(U)`` multiply-adds
    against a dense solve's ``n_vars ** 2``. Rows go in elimination order and
    each reads and writes ``rhs`` at the caller's index, so the permutation
    needs no gather, scatter or second vector. ``L`` has a unit diagonal and no
    storage, and the factorisation inverted ``U``'s diagonal, so the divide is a
    multiply.

    Small structures get the sweeps **unrolled**, every slot a literal. This
    routine runs eight times per step and dominates the solver's cost.
    Table-driven, each multiply-add loads its column index before it can load
    the value, and each row loads two more for its bounds. A full device hides
    that latency, but a small ensemble such as DISCO-EB's 128 trajectories
    (four warps) cannot.

    Unrolling cannot promote ``lu`` or ``rhs`` out of local memory, since the
    kernel indexes both with loop variables elsewhere. It trades index loads
    for instruction count and nothing else.
    """
    n = layout.n_vars
    row_ptr, diag_ptr = layout.row_ptr, layout.diag_ptr
    origin, col_origin = layout.row_origin, layout.col_origin
    _, _, substitutions = layout.flop_counts()

    if substitutions <= MAX_UNROLLED_SUBSTITUTIONS:
        return _unrolled_solve(layout, inline)

    tables, (ROW, DIAG, ROW_ORIG, COL_ORIG) = _pack(
        row_ptr, diag_ptr, origin, col_origin
    )

    @cuda.jit(device=True)
    def sparse_direct_solve(lu, ipiv, rhs):
        t = cuda.const.array_like(tables)
        for i in range(n):
            row = t[ROW_ORIG + i]
            acc = rhs[row]
            for p in range(t[ROW + i], t[DIAG + i]):
                acc -= lu[p] * rhs[t[COL_ORIG + p]]
            rhs[row] = acc
        for i in range(n - 1, -1, -1):
            row = t[ROW_ORIG + i]
            pivot = t[DIAG + i]
            acc = rhs[row]
            for p in range(pivot + 1, t[ROW + i + 1]):
                acc -= lu[p] * rhs[t[COL_ORIG + p]]
            rhs[row] = acc * lu[pivot]

    return sparse_direct_solve


def _unrolled_solve(layout: SparseLULayout, inline: str = "always") -> MLIRDispatcher:
    """Generate the two sweeps with every slot spelled out as a literal."""
    n = layout.n_vars
    row_ptr, diag_ptr = layout.row_ptr, layout.diag_ptr
    origin, col_origin = layout.row_origin, layout.col_origin

    def terms(span: range) -> str:
        return "".join(
            f" - lu[{p}] * rhs[{col_origin[p]}]" for p in range(span.start, span.stop)
        )

    lines = ["def sparse_direct_solve(lu, ipiv, rhs):"]
    for i in range(n):
        row = int(origin[i])
        lower = range(int(row_ptr[i]), int(diag_ptr[i]))
        lines.append(f"    rhs[{row}] = rhs[{row}]{terms(lower)}")
    for i in range(n - 1, -1, -1):
        row, pivot = int(origin[i]), int(diag_ptr[i])
        upper = range(pivot + 1, int(row_ptr[i + 1]))
        lines.append(f"    rhs[{row}] = (rhs[{row}]{terms(upper)}) * lu[{pivot}]")
    return compile_device_source("sparse_direct_solve", lines, inline=inline)


def _pack(*arrays: npt.NDArray[np.integer]) -> tuple[IndexTable, tuple[int, ...]]:
    """One int32 array of everything, plus each piece's offset into it.

    Every entry must be **non-negative**. numba-cuda-mlir (0.5.1) promotes a
    negative int32 read from a constant array as unsigned once it
    enters index arithmetic: ``t[i] + 4`` with ``t[i] == -1`` gives
    ``2 ** 32 + 3``, and the bad address goes unnoticed. Every table here is an
    offset, so non-negative is the natural form anyway.

    Raises:
        ValueError: If any entry is negative.
    """
    offsets = np.cumsum([0] + [len(a) for a in arrays[:-1]])
    packed = np.concatenate([np.asarray(a, dtype=np.int32) for a in arrays])
    if packed.size and packed.min() < 0:
        raise ValueError("index tables must hold non-negative offsets only")
    return packed, tuple(int(o) for o in offsets)
