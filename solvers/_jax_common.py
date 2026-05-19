"""Shared scaffolding for pure-JAX adaptive ensemble solvers."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


def normalize_y0_params(y0, params):
    """Broadcast ``y0`` / ``params`` to a consistent ``(N, …)`` ensemble layout.

    Accepts either 1-D (``(n_vars,)`` / ``(n_params,)``) or 2-D
    (``(N, n_vars)`` / ``(N, n_params)``) inputs and returns 2-D arrays with a
    common leading axis. Used by both pure-JAX and numba-cuda solvers so they
    share the same calling convention.
    """
    y0_in = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (n, n_vars))
        params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (n, n_vars))
    else:
        n = y0_in.shape[0]
        n_vars = y0_in.shape[1]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )
    return y0_arr, params_arr, n, n_vars


def normalize_inputs(y0, t_span, params, first_step, batch_size):
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)

    n_save = times.shape[0]
    dt0 = jnp.float64(
        first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
    )
    bs = n if batch_size is None else batch_size
    n_chunks = (n + bs - 1) // bs
    return y0_arr, times, params_arr, n, n_vars, n_save, dt0, bs, n_chunks


def initial_history(y_init, n_save: int, n_vars: int):
    return jnp.zeros((n_save, n_vars), dtype=jnp.float64).at[0, :].set(y_init)


def error_norm(y, y_new, err_est, rtol, atol, error_weights):
    scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    scaled_error = (err_est / scale) * error_weights
    denom = jnp.maximum(jnp.sum(error_weights != 0), 1)
    return jnp.sqrt(jnp.sum(scaled_error**2) / denom)


def build_error_weights(error_weights, n: int, n_vars: int):
    """Broadcast a user ``error_weights`` argument to ``(n, n_vars)``.

    ``None`` yields all-ones (every component weighted equally); a 1-D array of
    length ``n_vars`` is broadcast across trajectories; a 2-D ``(n, n_vars)``
    array is used as-is. This per-trajectory ``error_weights`` row is consumed
    by the default :func:`error_norm`; custom ``error_norm_fn`` callables
    receive the same array and may interpret it however they like.
    """
    if error_weights is None:
        return jnp.ones((n, n_vars), dtype=jnp.float64)
    weights = jnp.asarray(error_weights, dtype=jnp.float64)
    if weights.ndim == 1:
        return jnp.broadcast_to(weights, (n, n_vars))
    return weights


def step_size_factor(
    err_norm,
    *,
    failed=False,
    exponent: float,
    safety: float,
    factor_min: float,
    factor_max: float,
):
    safe_err = jnp.where(
        failed | jnp.isnan(err_norm) | (err_norm > 1e18),
        1e18,
        jnp.where(err_norm == 0.0, 1e-18, err_norm),
    )
    return jnp.clip(safety * safe_err**exponent, factor_min, factor_max)


def build_batch_stats(trajectory_stats, *, n: int, n_chunks: int, batch_size: int):
    accepted_steps = trajectory_stats["accepted_steps"].reshape(n)
    rejected_steps = trajectory_stats["rejected_steps"].reshape(n)
    n_padded = n_chunks * batch_size
    pad_count = n_padded - n
    loop_steps_padded = jnp.pad(trajectory_stats["loop_steps"], (0, pad_count))
    loop_steps = loop_steps_padded.reshape(n_chunks, batch_size)
    valid_batches = (jnp.arange(n_padded) < n).reshape(n_chunks, batch_size)
    batch_loop_iterations = jnp.max(
        jnp.where(valid_batches, loop_steps, jnp.int32(0)), axis=1
    )
    valid_lanes = jnp.sum(valid_batches.astype(jnp.int32), axis=1)
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "batch_loop_iterations": batch_loop_iterations,
        "valid_lanes": valid_lanes,
    }


def _broadcast_for_vmap(arg, is_batched: bool, axis_size: int, name: str):
    arr = jnp.asarray(arg)
    if is_batched:
        if arr.ndim != 2:
            raise NotImplementedError(
                f"vmap over an already-ensembled {name} is not supported; "
                "call the solver with batched y0/params directly instead."
            )
        if arr.shape[0] != axis_size:
            raise ValueError(
                f"batched {name} has leading axis {arr.shape[0]}, expected {axis_size}"
            )
        return arr
    if arr.ndim != 1:
        raise NotImplementedError(
            f"vmap with unbatched ensemble-shaped {name} is not supported; "
            "call the solver with batched y0/params directly instead."
        )
    return jnp.broadcast_to(arr, (axis_size,) + arr.shape)


def _jax_stats_postprocess(stats, axis_size):
    """Default stats reshape for JAX-solver ``build_batch_stats`` output.

    ``batch_loop_iterations`` and ``valid_lanes`` are chunk-level in the JAX
    solvers (shape ``(n_chunks,)``), so under vmap we re-synthesise per-
    trajectory equivalents to match the public ``(axis_size, 1)`` shape.
    """
    accepted = stats["accepted_steps"]
    rejected = stats["rejected_steps"]
    loop_steps = accepted + rejected
    stats_out = {
        "accepted_steps": accepted[:, None],
        "rejected_steps": rejected[:, None],
        "batch_loop_iterations": loop_steps[:, None],
        "valid_lanes": jnp.ones((axis_size, 1), dtype=stats["valid_lanes"].dtype),
    }
    stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
    return stats_out, stats_batched


def per_trajectory_stats_postprocess(stats, axis_size):
    """Stats reshape when every key already has shape ``(axis_size,)``.

    Used by the numba-cuda solvers whose kernels emit per-trajectory counters
    for every stats field.
    """
    del axis_size
    stats_out = jax.tree_util.tree_map(lambda x: x[:, None], stats)
    stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
    return stats_out, stats_batched


def make_custom_vmap_solver(
    solve_impl: Callable,
    *,
    return_stats: bool,
    stats_postprocess: Callable | None = None,
):
    """Wrap a solver implementation so outer ``jax.vmap`` becomes one ensemble call.

    ``solve_impl`` must accept ``(y0, t_span, params)`` and return the normal
    public solver result for those arrays.  The custom batching rule supports
    vmapping scalar solves over ``y0`` and/or ``params`` and lowers that vmap to
    a single native ensemble solve with a leading trajectory axis.

    ``stats_postprocess`` is used to reshape the stats pytree after the
    ensemble solve. Defaults to the JAX-solver layout
    (chunk-level ``batch_loop_iterations`` / ``valid_lanes`` synthesised into
    per-trajectory equivalents). Pass ``per_trajectory_stats_postprocess`` for
    solvers whose stats are already per-trajectory.
    """

    if stats_postprocess is None:
        stats_postprocess = _jax_stats_postprocess

    @custom_vmap
    def _solve(y0, t_span, params):
        return solve_impl(y0, t_span, params)

    @_solve.def_vmap
    def _solve_vmap(axis_size, in_batched, y0, t_span, params):
        y0_batched, t_span_batched, params_batched = in_batched
        if t_span_batched:
            t_span_arr = jnp.asarray(t_span)
            if t_span_arr.ndim != 2:
                raise NotImplementedError(
                    "vmap over nested t_span values is not supported; use a shared "
                    "t_span and vmap over y0 and/or params, or call the solver directly."
                )
            # JAX can mark closed-over constant save times as batched inside
            # a larger vmapped function.  Treat that as a shared time grid.
            t_span = t_span_arr[0]

        y0_arr = _broadcast_for_vmap(y0, y0_batched, axis_size, "y0")
        params_arr = _broadcast_for_vmap(params, params_batched, axis_size, "params")
        result = solve_impl(y0_arr, t_span, params_arr)

        if not return_stats:
            return result[:, None, :, :], True

        sol, stats = result
        stats_out, stats_batched = stats_postprocess(stats, axis_size)
        return (sol[:, None, :, :], stats_out), (True, stats_batched)

    return _solve


def solve_adaptive_ensemble(
    *,
    params_arr,
    y0_arr,
    times,
    dt0,
    batch_size: int,
    n_chunks: int,
    rtol,
    atol,
    max_steps: int,
    return_stats: bool,
    step_factory: Callable,
    error_exponent: float,
    safety: float,
    factor_min: float,
    factor_max: float,
    error_weights_arr=None,
    error_norm_fn: Callable | None = None,
):
    n = y0_arr.shape[0]
    n_vars = y0_arr.shape[1]
    n_save = times.shape[0]
    tf = times[-1]

    # ``error_weights`` are supplied per-trajectory to ``error_norm_fn``, which
    # by default is the per-component weighted :func:`error_norm`.
    if error_norm_fn is None:
        error_norm_fn = error_norm
    if error_weights_arr is None:
        error_weights_arr = jnp.ones((n, n_vars), dtype=jnp.float64)

    def _solve_one(params_one, y0_one, error_weights_one):
        y_init = y0_one.copy()
        hist_init = initial_history(y_init, n_save, n_vars)
        step_one, extra_init, update_extra = step_factory(params_one)

        def cond_fn(state):
            t, _, _, _, save_idx, n_steps, _, _, _ = state
            return (save_idx < n_save) & (t < tf) & (n_steps < max_steps)

        def body_fn(state):
            (
                t,
                y,
                dt,
                hist,
                save_idx,
                n_steps,
                accepted_steps,
                rejected_steps,
                extra,
            ) = state
            next_target = times[save_idx]
            dt_use = jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30)

            y_new, err_est, failed, extra_candidate = step_one(y, t, dt_use, extra)
            err_norm = error_norm_fn(y, y_new, err_est, rtol, atol, error_weights_one)

            accept = (err_norm <= 1.0) & ~jnp.isnan(err_norm) & ~failed
            t_new = jnp.where(accept, t + dt_use, t)
            y_out = jnp.where(accept, y_new, y)

            reached = accept & (
                jnp.abs(t_new - next_target)
                <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
            )
            slot_mask = jax.nn.one_hot(save_idx, n_save, dtype=jnp.bool_) & reached
            hist_new = jnp.where(slot_mask[:, None], y_out[None, :], hist)
            save_idx_new = save_idx + reached.astype(jnp.int32)

            factor = step_size_factor(
                err_norm,
                failed=failed,
                exponent=error_exponent,
                safety=safety,
                factor_min=factor_min,
                factor_max=factor_max,
            )
            dt_new = dt_use * factor
            rejected = ~accept
            extra_new = update_extra(extra, extra_candidate, accept)

            return (
                t_new,
                y_out,
                dt_new,
                hist_new,
                save_idx_new,
                n_steps + 1,
                accepted_steps + accept.astype(jnp.int32),
                rejected_steps + rejected.astype(jnp.int32),
                extra_new,
            )

        init = (
            times[0],
            y_init,
            dt0,
            hist_init,
            jnp.int32(1),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            extra_init,
        )
        (
            _,
            _,
            _,
            hist_final,
            _,
            loop_steps,
            accepted_steps,
            rejected_steps,
            _,
        ) = jax.lax.while_loop(cond_fn, body_fn, init)
        stats = {
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "loop_steps": loop_steps,
        }
        return hist_final, stats

    results, trajectory_stats = jax.lax.map(
        lambda xs: _solve_one(*xs),
        (params_arr, y0_arr, error_weights_arr),
        batch_size=batch_size,
    )
    if not return_stats:
        return results
    return results, build_batch_stats(
        trajectory_stats, n=n, n_chunks=n_chunks, batch_size=batch_size
    )
