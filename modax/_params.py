"""``params`` as a pytree: its leaves, their columns, and the row they make.

The kernel reads one ``(N, n_params)`` row per trajectory, and ``ode_fn`` sees
it as one flat tuple. A solve may take ``params`` as any pytree of arrays
instead, and its leaves are concatenated, in ``jax.tree.leaves`` order, into
that row. What the structure buys is differentiation: JAX tracks symbolic
zeros per leaf, never per element, so a leaf that does not depend on what is
being differentiated reaches the sensitivity rule as a ``SymbolicZero``, and
the rule integrates only the columns of the leaves that do.

A leaf is ``()`` (one shared column), ``(k,)`` (``k`` shared columns) or
``(N, k)`` (``k`` columns, one row per trajectory). A Python list is one
array-like leaf, as :func:`jax.numpy.asarray` reads it, so a nested list of
rows keeps meaning ``(N, k)``.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp

Params: TypeAlias = Any
"""An array-like or a pytree of them (tuples, dicts, registered nodes)."""


def param_leaves(params: Params) -> tuple[jax.Array, ...]:
    """The leaves of ``params`` as float64 arrays, each 1-D or 2-D.

    A scalar leaf becomes one column, ``(1,)``.

    Raises:
        ValueError: If ``params`` has no leaves, or a leaf has more than two
            dimensions.
    """
    leaves = jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, list))
    if not leaves:
        raise ValueError("params has no leaves; pass at least one array")
    out = []
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=jnp.float64)
        if arr.ndim == 0:
            arr = arr[None]
        if arr.ndim > 2:
            raise ValueError(
                f"each params leaf must be (), (k,) or (N, k); got shape {arr.shape}"
            )
        out.append(arr)
    return tuple(out)


def combine_params(leaves: tuple[jax.Array, ...]) -> jax.Array:
    """Concatenate leaves into the ``(n_params,)`` or ``(N, n_params)`` row.

    A single leaf is returned as it is. Shared leaves are broadcast to the
    ensemble when any leaf is per-trajectory.

    Raises:
        ValueError: If two per-trajectory leaves disagree on ``N``.
    """
    if len(leaves) == 1:
        return leaves[0]
    rows = {leaf.shape[0] for leaf in leaves if leaf.ndim == 2}
    if len(rows) > 1:
        raise ValueError(f"per-trajectory params leaves disagree on N: {sorted(rows)}")
    if not rows:
        return jnp.concatenate(leaves)
    (n,) = rows
    return jnp.concatenate(
        [jnp.broadcast_to(leaf, (n, leaf.shape[-1])) for leaf in leaves], axis=1
    )


def as_param_row(params: Params) -> jax.Array:
    """``params`` as the one array the kernel reads."""
    return combine_params(param_leaves(params))


def param_width(params: Params) -> int:
    """``n_params``: the columns ``params`` contributes to the row."""
    return sum(leaf.shape[-1] for leaf in param_leaves(params))
