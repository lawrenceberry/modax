"""``modax.solve``: one entry point, the method chosen up front."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

from modax._methods import Auto, Method, Rodas5P, Tsit5


def _gain_kwargs(gains) -> dict[str, float]:
    if gains is None:
        return {}
    return {"pcoeff": gains.pcoeff, "icoeff": gains.icoeff, "dcoeff": gains.dcoeff}


def solve(ode_fn, y0, t_span, params, *, method: Method | None = None, **kwargs: Any):
    """Solve ``ode_fn`` -- a plain right-hand side, a :class:`~modax.Block` or a :class:`~modax.Hierarchy` -- with ``method``.

    The one entry point for hierarchies and for :class:`~modax.Auto`:
    :func:`modax.rodas5P.solve` and :func:`modax.tsit5.solve` take plain
    right-hand sides only. ``method`` is :class:`~modax.Rodas5P` (the
    default), :class:`~modax.Tsit5` or :class:`~modax.Auto`. For a hierarchy
    it is how the blocks that name no method of their own step; each block's
    ``method`` overrides it. A plain system with ``Auto`` is solved as a
    hierarchy of one block. The keyword
    arguments are :func:`modax.rodas5P.solve`'s (:func:`modax.tsit5.solve`'s
    for a plain system with ``Tsit5``): tolerances, error weights, the save
    hook, the controller gains the methods do not set themselves.
    """
    from modax import rodas5P, tsit5
    from modax._hierarchy import Block, as_hierarchy

    method = Rodas5P() if method is None else method
    if not isinstance(method, (Rodas5P, Tsit5, Auto)):
        raise TypeError(f"method must be Rodas5P, Tsit5 or Auto, got {method!r}")
    hierarchy = as_hierarchy(ode_fn)
    if hierarchy is not None:
        return rodas5P._solve(
            cast(Any, hierarchy), y0, t_span, params, method=method, **kwargs
        )
    if isinstance(method, Rodas5P):
        kw = {**kwargs, **_gain_kwargs(method.gains)}
        if method.lu_precision is not None:
            kw["lu_precision"] = method.lu_precision
        return rodas5P.solve(ode_fn, y0, t_span, params, **kw)  # ty: ignore[no-matching-overload]
    if isinstance(method, Tsit5):
        return tsit5.solve(  # ty: ignore[no-matching-overload]
            ode_fn, y0, t_span, params, **{**kwargs, **_gain_kwargs(method.gains)}
        )
    # Auto on a plain system: one block, which is the final block of its hierarchy
    sparsity = kwargs.pop("sparsity", None)
    block = Block(ode_fn, int(jnp.shape(y0)[-1]), sparsity=sparsity, method=method)
    return rodas5P._solve(cast(Any, block), y0, t_span, params, method=method, **kwargs)
