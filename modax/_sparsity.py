"""Turn a Jacobian sparsity pattern into the fewest forward sweeps.

Forward-mode AD returns ``J v`` for a direction ``v``, not ``J``, so seeding
unit vectors one at a time costs ``n_vars`` sweeps. Two columns that share
no row are *structurally orthogonal*. Their contributions to ``J v`` never
land on the same component, so seeding ``v = e_i + e_j`` returns both
columns intact in one sweep, and the pattern says which component belongs to
which column.

Grouping the columns into as few such sets as possible is vertex colouring of
the column intersection graph ``S^T S``. NetworkX's greedy colouring reaches
the bound that matters (the largest set of mutually overlapping columns) on
the patterns these solvers see.

The caller's pattern must be a *superset* of the true nonzeros. A superset
only costs sweeps, but a subset silently corrupts entries where two columns
of a group overlap after all. It need not cover the factorisation's fill-in,
which gets its own slots from :py:obj:`modax._sparse_direct` and which no
column of ``J`` writes.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike
from scipy.sparse import coo_array, sparray, spmatrix

Pattern: TypeAlias = tuple[tuple[int, ...], ...]
"""A normalised sparsity pattern: the sorted column indices of each row."""

Sparsity: TypeAlias = ArrayLike | spmatrix | sparray
"""What ``solve`` accepts as ``sparsity``: an ``(n_vars, n_vars)`` mask, a
scipy sparse matrix, or an ``(nnz, 2)`` array of ``(row, col)`` pairs.
"""

# The strategies NetworkX offers that are worth trying. They are cheap -- this
# runs once per kernel build, on the host -- so the best of them is taken
# rather than one being picked in advance.
_STRATEGIES = (
    "largest_first",
    "smallest_last",
    "independent_set",
    "saturation_largest_first",
    "DSATUR",
)


@dataclass(frozen=True)
class CompressedJacobian:
    """A column-compressed Jacobian layout, and the seeds that fill it.

    Entry ``(r, c)`` of the Jacobian lives at slot ``r * n_colours + colour[c]``.
    Rows stay dense and the colouring compresses the columns, so every row of every colour has
    a slot. The sweeps write slots whose colour group has no nonzero in that row
    but nothing reads them, which keeps each write a straight run over a column
    block.

    With no pattern, colouring gives ``n_colours == n_vars`` and this is exactly
    the dense row-major matrix. The kernel's dense path is therefore not a
    special case.

    :py:obj:`modax._sparse_direct` replaces the grid with ``packed``, the slot of
    each entry in its CSR image of ``L + U``, so the sweeps write ``-J`` straight
    into the buffer the factorisation works in.
    """

    n_vars: int
    n_colours: int
    colour: tuple[int, ...]
    # Set by `modax._sparse_direct.compressed_jacobian`: the slot each
    # ``(row, colour)`` grid position collapses to, ``-1`` where the group has
    # nothing in that row. ``None`` leaves the layout on the grid, where a slot
    # exists for every position.
    packed: tuple[int, ...] | None = None
    # The slot of every diagonal entry, declared by the pattern or not.
    # ``I/(h*gamma)`` lands on every diagonal whether or not ``J`` has anything
    # there, so a packed layout keeps room for each; one the pattern leaves out
    # is written by no sweep, which is why a packed layout with any of those is
    # cleared before the sweeps.
    packed_diagonal: tuple[int, ...] | None = None
    # Slots the buffer holds beyond the ones some entry claims: the direct
    # sparse factorisation needs room for its fill-in, which belongs to no
    # column of ``J`` and so appears in no slot table. ``None`` means the
    # claimed slots are the whole buffer.
    n_slots: int | None = None

    @property
    def size(self) -> int:
        """Elements in one trajectory's matrix."""
        if self.n_slots is not None:
            return self.n_slots
        return self.n_vars * self.n_colours

    @property
    def diagonal(self) -> tuple[int, ...]:
        """Slot of each ``(i, i)`` entry, for the ``1/(h*gamma)`` term."""
        return tuple(self.slot(i, i) for i in range(self.n_vars))

    def slot(self, row: int, col: int) -> int:
        """Where entry ``(row, col)`` lives.

        Raises:
            ValueError: If the layout packs its slots and the pattern has no such entry.
        """
        if self.packed is not None and row == col:
            # The factorisation's own slot, whether or not the pattern declares
            # the diagonal. The grid cell ``(row, colour[row])`` cannot stand in
            # for it: when ``J`` has no ``(row, row)``, another column of that
            # colour may have an entry in the row and own the cell, and the
            # ``1/(h*gamma)`` would be added to that entry instead.
            assert self.packed_diagonal is not None
            return self.packed_diagonal[row]
        grid = row * self.n_colours + self.colour[col]
        if self.packed is None:
            return grid
        slot = self.packed[grid]
        if slot < 0:
            raise ValueError(
                f"entry ({row}, {col}) is not in the pattern, so a packed "
                "layout has no slot for it; declare it in the sparsity pattern"
            )
        return slot

    @property
    def has_unwritten_slots(self) -> bool:
        """Whether no colour sweep writes some slot of the buffer.

        The grid writes every slot, and a packed layout has one slot per entry of
        ``J``. Only a packed layout with unclaimed slots (the factorisation's
        fill-in, or a diagonal the pattern left out) needs clearing before the
        sweeps run.
        """
        if self.packed is None:
            return False
        return self.size > sum(1 for slot in self.packed if slot >= 0)

    def store_slots(self) -> npt.NDArray[np.int32] | None:
        """Where each ``(row, colour)`` sweep value goes, or ``None`` for a run.

        On the grid, a colour group's values form a contiguous block that the
        kernel writes straight down. In a packed layout each value goes to its own
        slot, and the kernel skips positions no entry claims.
        """
        if self.packed is None:
            return None
        return np.asarray(self.packed, dtype=np.int32)

    def seed_table(self, min_width: int = 0) -> npt.NDArray[np.float64]:
        """``(n_colours + 1, max(n_vars, min_width))`` of tangent directions.

        Row ``g`` is the indicator of colour group ``g``, so one sweep seeded with it
        yields that whole group. The final row is all zeros, for sweeps that vary
        ``t`` or a parameter rather than the state. It also serves as the null
        *parameter* direction, and there may be more parameters than states, hence
        the widening to ``min_width``.
        """
        width = max(self.n_vars, min_width)
        seeds = np.zeros((self.n_colours + 1, width), dtype=np.float64)
        for c, g in enumerate(self.colour):
            seeds[g, c] = 1.0
        return seeds


def dense_jacobian(n_vars: int) -> CompressedJacobian:
    """Every column its own colour: the row-major dense matrix."""
    return CompressedJacobian(
        n_vars=n_vars, n_colours=n_vars, colour=tuple(range(n_vars))
    )


def pattern_mask(pattern: Pattern) -> np.ndarray:
    """A normalised pattern as the ``(n, n)`` boolean mask a solve takes."""
    mask = np.zeros((len(pattern), len(pattern)), dtype=bool)
    for row, cols in enumerate(pattern):
        mask[row, list(cols)] = True
    return mask


def normalize_sparsity(sparsity: Sparsity, n_vars: int) -> Pattern:
    """Accept a dense mask, a scipy sparse matrix, or ``(row, col)`` pairs.

    Returns a hashable tuple of row tuples so the colouring can cache on it.
    A kernel is rebuilt per pattern, not per call.

    Raises:
        ValueError: If ``sparsity`` has none of those shapes, or an entry lies
            outside the ``n_vars`` by ``n_vars`` Jacobian.
    """
    pairs: Iterable[tuple[int, int]]
    if isinstance(sparsity, (spmatrix, sparray)):
        coo = coo_array(sparsity)
        pairs = zip(coo.row.tolist(), coo.col.tolist())
    else:
        arr = np.asarray(sparsity)
        if arr.ndim == 2 and arr.shape == (n_vars, n_vars):
            rows, cols = np.nonzero(arr)
            pairs = zip(rows.tolist(), cols.tolist())
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pairs = ((int(r), int(c)) for r, c in arr.tolist())
        else:
            raise ValueError(
                "sparsity must be an (n_vars, n_vars) mask, a scipy sparse "
                f"matrix, or an (nnz, 2) array of (row, col); got shape "
                f"{arr.shape} for n_vars={n_vars}"
            )
    by_row: list[set[int]] = [set() for _ in range(n_vars)]
    for r, c in pairs:
        if not (0 <= r < n_vars and 0 <= c < n_vars):
            raise ValueError(
                f"sparsity entry ({r}, {c}) lies outside an {n_vars}x{n_vars} Jacobian"
            )
        by_row[r].add(c)
    return tuple(tuple(sorted(cs)) for cs in by_row)


@functools.cache
def colour_sparsity(pattern: Pattern) -> CompressedJacobian:
    """Colour a pattern's column intersection graph, keeping the fewest colours.

    Tries several NetworkX greedy strategies and keeps the best.

    Raises:
        ValueError: If the colouring is not structurally orthogonal.
    """
    import networkx as nx

    n_vars = len(pattern)
    # Columns conflict when some row holds both. Building the adjacency from the
    # rows directly is cheaper than forming S^T S for the patterns seen here,
    # where a row has a handful of entries out of hundreds of columns.
    graph = nx.Graph()
    graph.add_nodes_from(range(n_vars))
    for cols in pattern:
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                graph.add_edge(cols[a], cols[b])

    best: dict[int, int] | None = None
    for strategy in _STRATEGIES:
        colouring = nx.coloring.greedy_color(graph, strategy=strategy)
        if best is None or max(colouring.values()) < max(best.values()):
            best = colouring
    assert best is not None

    colour = tuple(int(best[c]) for c in range(n_vars))
    compressed = CompressedJacobian(
        n_vars=n_vars, n_colours=max(colour) + 1, colour=colour
    )
    _check_orthogonal(pattern, compressed)
    return compressed


def _check_orthogonal(pattern: Pattern, compressed: CompressedJacobian) -> None:
    """Check that no row holds two columns of the same colour.

    Such a pair would share a slot and silently overwrite each other, the one
    way this scheme can go wrong. One pass over the pattern per kernel build
    rules it out.

    Raises:
        ValueError: If a row holds two columns of one colour.
    """
    for row, cols in enumerate(pattern):
        seen: dict[int, int] = {}
        for c in cols:
            g = compressed.colour[c]
            if g in seen:
                raise ValueError(
                    f"columns {seen[g]} and {c} share colour {g} but both have "
                    f"an entry in row {row}; the colouring is not structurally "
                    "orthogonal for this pattern"
                )
            seen[g] = c
