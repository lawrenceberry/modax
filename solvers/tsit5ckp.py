"""Generic Tsit5 custom kernel using JAX Pallas lowered through Triton."""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas.triton as pl_triton
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from solvers import _tsit5ck_common as ck

_BLOCK_SIZE = 32


class _PallasMatrix:
    """Tiny matrix facade that lowers y[:, j] to component-vector access."""

    def __init__(self, columns):
        self._columns = columns

    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise NotImplementedError("Pallas RHS matrices support y[:, j] indexing")
        rows, col = idx
        if rows != slice(None):
            raise NotImplementedError("Pallas RHS matrices support y[:, j] indexing")
        if not isinstance(col, int):
            raise NotImplementedError("Pallas RHS matrices support integer columns")
        return self._columns[col]


def _tuple_add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _tuple_scale(s, a):
    return tuple(s * x for x in a)


def _tuple_axpy(base, h, *terms):
    out = []
    for j, yj in enumerate(base):
        acc = yj
        for coeff, k in terms:
            acc = acc + h * coeff * k[j]
        out.append(acc)
    return tuple(out)


@functools.partial(
    jax.jit,
    static_argnames=(
        "ode_fn",
        "n_save",
        "n_vars",
        "n_vars_work",
        "n_params",
        "max_steps",
        "block_size",
    ),
)
def _solve_kernel(
    ode_fn,
    y0,
    times,
    params,
    n_actual,
    dt0,
    rtol,
    atol,
    *,
    n_save,
    n_vars,
    n_vars_work,
    n_params,
    max_steps,
    block_size,
):
    n_padded = y0.shape[0]
    n_blocks = n_padded // block_size

    def kernel(
        y0_ref,
        times_ref,
        params_ref,
        n_actual_ref,
        dt0_ref,
        rtol_ref,
        atol_ref,
        hist_ref,
        acc_ref,
        rej_ref,
        loop_ref,
    ):
        block_start = pl.program_id(0) * block_size
        offsets = block_start + jnp.arange(block_size)
        valid = offsets < n_actual_ref[()]

        y = tuple(y0_ref[offsets, j] for j in range(n_vars_work))
        p = tuple(params_ref[offsets, j] for j in range(n_params))
        for j in range(n_vars_work):
            hist_ref[offsets, 0, j] = y[j]

        t0 = times_ref[0]
        tf = times_ref[n_save - 1]
        t = jnp.where(valid, t0, tf)
        dt = jnp.full((block_size,), dt0_ref[()], dtype=jnp.float64)
        save_idx = jnp.full((block_size,), 1, dtype=jnp.int32)
        n_loop = jnp.int32(0)
        accepted_steps = jnp.zeros((block_size,), dtype=jnp.int32)
        rejected_steps = jnp.zeros((block_size,), dtype=jnp.int32)
        k_fsal = tuple(
            jnp.zeros((block_size,), dtype=jnp.float64) for _ in range(n_vars_work)
        )
        has_fsal = jnp.zeros((block_size,), dtype=jnp.bool_)

        def rhs(y_state, t_state):
            return ode_fn(_PallasMatrix(y_state), t_state, _PallasMatrix(p))

        def cond_fn(state):
            t, _dt, save_idx, n_loop, *_ = state
            unfinished_t = jnp.where(valid & (save_idx < n_save), t, tf)
            return (jnp.min(unfinished_t) < tf) & (n_loop < max_steps)

        def body_fn(state):
            (
                t,
                dt,
                save_idx,
                n_loop,
                y,
                k_fsal,
                has_fsal,
                accepted_steps,
                rejected_steps,
            ) = state
            active = valid & (save_idx < n_save) & (t < tf)
            safe_save_idx = jnp.minimum(save_idx, n_save - 1)
            next_target = times_ref[safe_save_idx]
            dt_use = jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30)

            fresh = rhs(y, t)
            k1 = tuple(
                jnp.where(has_fsal, kfs, fr)
                for kfs, fr in zip(k_fsal, fresh, strict=True)
            )
            k2_y = _tuple_axpy(y, dt_use, (ck.A21, k1))
            k2 = rhs(k2_y, t + ck.C2 * dt_use)

            k3_y = _tuple_axpy(y, dt_use, (ck.A31, k1), (ck.A32, k2))
            k3 = rhs(k3_y, t + ck.C3 * dt_use)

            k4_y = _tuple_axpy(y, dt_use, (ck.A41, k1), (ck.A42, k2), (ck.A43, k3))
            k4 = rhs(k4_y, t + ck.C4 * dt_use)

            k5_y = _tuple_axpy(
                y, dt_use, (ck.A51, k1), (ck.A52, k2), (ck.A53, k3), (ck.A54, k4)
            )
            k5 = rhs(k5_y, t + ck.C5 * dt_use)

            k6_y = _tuple_axpy(
                y,
                dt_use,
                (ck.A61, k1),
                (ck.A62, k2),
                (ck.A63, k3),
                (ck.A64, k4),
                (ck.A65, k5),
            )
            k6 = rhs(k6_y, t + ck.C6 * dt_use)

            y_new = _tuple_axpy(
                y,
                dt_use,
                (ck.B1, k1),
                (ck.B2, k2),
                (ck.B3, k3),
                (ck.B4, k4),
                (ck.B5, k5),
                (ck.B6, k6),
            )
            k7 = rhs(y_new, t + ck.C7 * dt_use)

            rtol_value = rtol_ref[()]
            atol_value = atol_ref[()]
            err_sum = jnp.zeros((block_size,), dtype=jnp.float64)
            y_out = []
            for j in range(n_vars_work):
                err_est = dt_use * (
                    ck.E1 * k1[j]
                    + ck.E2 * k2[j]
                    + ck.E3 * k3[j]
                    + ck.E4 * k4[j]
                    + ck.E5 * k5[j]
                    + ck.E6 * k6[j]
                    + ck.E7 * k7[j]
                )
                if j < n_vars:
                    scale = atol_value + rtol_value * jnp.maximum(
                        jnp.abs(y[j]), jnp.abs(y_new[j])
                    )
                    ratio = err_est / scale
                    err_sum = err_sum + ratio * ratio
            err_norm = jnp.sqrt(err_sum / n_vars)
            accept = active & (err_norm <= 1.0) & ~jnp.isnan(err_norm)
            rejected = active & ~accept

            t_new = jnp.where(accept, t + dt_use, t)
            for j in range(n_vars_work):
                y_out.append(jnp.where(accept, y_new[j], y[j]))
            y_out = tuple(y_out)

            reached = accept & (
                jnp.abs(t_new - next_target)
                <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
            )
            for j in range(n_vars_work):
                old_hist = hist_ref[offsets, safe_save_idx, j]
                hist_ref[offsets, safe_save_idx, j] = jnp.where(
                    reached, y_out[j], old_hist
                )

            safe_err = jnp.where(
                jnp.isnan(err_norm) | (err_norm > 1e18),
                1e18,
                jnp.where(err_norm == 0.0, 1e-18, err_norm),
            )
            factor = jnp.clip(
                ck.SAFETY * safe_err ** (-1.0 / 5.0), ck.FACTOR_MIN, ck.FACTOR_MAX
            )
            k_fsal_new = tuple(
                jnp.where(accept, k7j, jnp.where(active, 0.0, kfs))
                for k7j, kfs in zip(k7, k_fsal, strict=True)
            )
            return (
                t_new,
                jnp.where(active, dt_use * factor, dt),
                save_idx + reached.astype(jnp.int32),
                n_loop + 1,
                y_out,
                k_fsal_new,
                jnp.where(active, accept, has_fsal),
                accepted_steps + accept.astype(jnp.int32),
                rejected_steps + rejected.astype(jnp.int32),
            )

        final = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                t,
                dt,
                save_idx,
                n_loop,
                y,
                k_fsal,
                has_fsal,
                accepted_steps,
                rejected_steps,
            ),
        )
        acc_ref[offsets] = final[-2]
        rej_ref[offsets] = final[-1]
        loop_ref[offsets] = jnp.full((block_size,), final[3], dtype=jnp.int32)

    out_shape = (
        jax.ShapeDtypeStruct((n_padded, n_save, n_vars_work), jnp.float64),
        jax.ShapeDtypeStruct((n_padded,), jnp.int32),
        jax.ShapeDtypeStruct((n_padded,), jnp.int32),
        jax.ShapeDtypeStruct((n_padded,), jnp.int32),
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(n_blocks,),
        compiler_params=pl_triton.CompilerParams(num_warps=1),
        name="tsit5ckp_generic_components",
    )(y0, times, params, n_actual, dt0, rtol, atol)


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
):
    del batch_size
    y0_arr, times, params_arr, dt0 = ck.normalize_inputs(y0, t_span, params, first_step)
    n_actual = y0_arr.shape[0]
    n_vars = y0_arr.shape[1]
    n_params = params_arr.shape[1]
    n_vars_work = 1 << (n_vars - 1).bit_length()
    pad_count = (-n_actual) % _BLOCK_SIZE
    if pad_count:
        y0_arr = np.pad(y0_arr, ((0, pad_count), (0, 0)))
        params_arr = np.pad(params_arr, ((0, pad_count), (0, 0)))
    if n_vars_work != n_vars:
        y0_arr = np.pad(y0_arr, ((0, 0), (0, n_vars_work - n_vars)))

    solution, accepted_steps, rejected_steps, loop_steps = _solve_kernel(
        ode_fn,
        jnp.asarray(y0_arr, dtype=jnp.float64),
        jnp.asarray(times, dtype=jnp.float64),
        jnp.asarray(params_arr, dtype=jnp.float64),
        jnp.asarray(n_actual, dtype=jnp.int32),
        jnp.asarray(dt0, dtype=jnp.float64),
        jnp.asarray(rtol, dtype=jnp.float64),
        jnp.asarray(atol, dtype=jnp.float64),
        n_save=times.shape[0],
        n_vars=n_vars,
        n_vars_work=n_vars_work,
        n_params=n_params,
        max_steps=int(max_steps),
        block_size=_BLOCK_SIZE,
    )
    solution = solution[:n_actual, :, :n_vars]
    accepted_steps = accepted_steps[:n_actual]
    rejected_steps = rejected_steps[:n_actual]
    loop_steps = loop_steps[:n_actual]
    if not return_stats:
        return solution
    stats = {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": jnp.ones_like(loop_steps),
    }
    return solution, stats
