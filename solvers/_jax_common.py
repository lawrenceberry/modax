"""Shared scaffolding for pure-JAX adaptive ensemble solvers."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def normalize_inputs(y0, t_span, params, first_step, batch_size):
    y0_in = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)
    times = jnp.asarray(t_span, dtype=jnp.float64)

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

    n_save = times.shape[0]
    dt0 = jnp.float64(
        first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
    )
    bs = n if batch_size is None else batch_size
    n_chunks = (n + bs - 1) // bs
    return y0_arr, times, params_arr, n, n_vars, n_save, dt0, bs, n_chunks


def initial_history(y_init, n_save: int, n_vars: int):
    return jnp.zeros((n_save, n_vars), dtype=jnp.float64).at[0, :].set(y_init)


def error_norm(y, y_new, err_est, rtol, atol):
    scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    return jnp.sqrt(jnp.mean((err_est / scale) ** 2))


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
):
    n = y0_arr.shape[0]
    n_vars = y0_arr.shape[1]
    n_save = times.shape[0]
    tf = times[-1]

    def _solve_one(params_one, y0_one):
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
            err_norm = error_norm(y, y_new, err_est, rtol, atol)

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
        (params_arr, y0_arr),
        batch_size=batch_size,
    )
    if not return_stats:
        return results
    return results, build_batch_stats(
        trajectory_stats, n=n, n_chunks=n_chunks, batch_size=batch_size
    )
