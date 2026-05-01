"""Generic Tsit5 custom kernel using NVIDIA Warp."""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from solvers import _tsit5ck_common as ck


@functools.cache
def _make_kernel(ode_fn, n_vars: int):
    @wp.kernel
    def kernel(
        y0: wp.array2d(dtype=wp.float64),
        times: wp.array(dtype=wp.float64),
        params: wp.array2d(dtype=wp.float64),
        dt0: wp.float64,
        rtol: wp.float64,
        atol: wp.float64,
        max_steps: wp.int32,
        hist: wp.array3d(dtype=wp.float64),
        accepted_out: wp.array(dtype=wp.int32),
        rejected_out: wp.array(dtype=wp.int32),
        loop_out: wp.array(dtype=wp.int32),
        y: wp.array2d(dtype=wp.float64),
        u: wp.array2d(dtype=wp.float64),
        k1: wp.array2d(dtype=wp.float64),
        k2: wp.array2d(dtype=wp.float64),
        k3: wp.array2d(dtype=wp.float64),
        k4: wp.array2d(dtype=wp.float64),
        k5: wp.array2d(dtype=wp.float64),
        k6: wp.array2d(dtype=wp.float64),
        k7: wp.array2d(dtype=wp.float64),
    ):
        i = wp.tid()

        j = wp.int32(0)
        while j < n_vars:
            y[i, j] = y0[i, j]
            hist[i, 0, j] = y0[i, j]
            k7[i, j] = wp.float64(0.0)
            j += wp.int32(1)

        n_save = times.shape[0]
        t = times[0]
        tf = times[n_save - 1]
        dt = dt0
        save_idx = wp.int32(1)
        n_steps = wp.int32(0)
        accepted_steps = wp.int32(0)
        rejected_steps = wp.int32(0)
        has_fsal = wp.int32(0)

        while save_idx < n_save and t < tf and n_steps < max_steps:
            next_target = times[save_idx]
            dt_use = dt
            if dt_use > next_target - t:
                dt_use = next_target - t
            if dt_use < wp.float64(1.0e-30):
                dt_use = wp.float64(1.0e-30)

            if has_fsal != wp.int32(0):
                j = wp.int32(0)
                while j < n_vars:
                    k1[i, j] = k7[i, j]
                    j += wp.int32(1)
            else:
                ode_fn(y, t, params, k1, i)

            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + dt_use * (wp.float64(ck.A21) * k1[i, j])
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C2) * dt_use, params, k2, i)

            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + dt_use * (
                    wp.float64(ck.A31) * k1[i, j] + wp.float64(ck.A32) * k2[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C3) * dt_use, params, k3, i)

            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + dt_use * (
                    wp.float64(ck.A41) * k1[i, j]
                    + wp.float64(ck.A42) * k2[i, j]
                    + wp.float64(ck.A43) * k3[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C4) * dt_use, params, k4, i)

            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + dt_use * (
                    wp.float64(ck.A51) * k1[i, j]
                    + wp.float64(ck.A52) * k2[i, j]
                    + wp.float64(ck.A53) * k3[i, j]
                    + wp.float64(ck.A54) * k4[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C5) * dt_use, params, k5, i)

            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + dt_use * (
                    wp.float64(ck.A61) * k1[i, j]
                    + wp.float64(ck.A62) * k2[i, j]
                    + wp.float64(ck.A63) * k3[i, j]
                    + wp.float64(ck.A64) * k4[i, j]
                    + wp.float64(ck.A65) * k5[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C6) * dt_use, params, k6, i)

            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + dt_use * (
                    wp.float64(ck.B1) * k1[i, j]
                    + wp.float64(ck.B2) * k2[i, j]
                    + wp.float64(ck.B3) * k3[i, j]
                    + wp.float64(ck.B4) * k4[i, j]
                    + wp.float64(ck.B5) * k5[i, j]
                    + wp.float64(ck.B6) * k6[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C7) * dt_use, params, k7, i)

            err_sum = wp.float64(0.0)
            j = wp.int32(0)
            while j < n_vars:
                err_est = dt_use * (
                    wp.float64(ck.E1) * k1[i, j]
                    + wp.float64(ck.E2) * k2[i, j]
                    + wp.float64(ck.E3) * k3[i, j]
                    + wp.float64(ck.E4) * k4[i, j]
                    + wp.float64(ck.E5) * k5[i, j]
                    + wp.float64(ck.E6) * k6[i, j]
                    + wp.float64(ck.E7) * k7[i, j]
                )
                scale = atol + rtol * wp.max(wp.abs(y[i, j]), wp.abs(u[i, j]))
                ratio = err_est / scale
                err_sum += ratio * ratio
                j += wp.int32(1)
            err_norm = wp.sqrt(err_sum / wp.float64(n_vars))
            accept = err_norm <= wp.float64(1.0) and not wp.isnan(err_norm)

            if accept:
                t_new = t + dt_use
                j = wp.int32(0)
                while j < n_vars:
                    y[i, j] = u[i, j]
                    j += wp.int32(1)
                accepted_steps += wp.int32(1)
                has_fsal = wp.int32(1)
            else:
                t_new = t
                rejected_steps += wp.int32(1)
                j = wp.int32(0)
                while j < n_vars:
                    k7[i, j] = wp.float64(0.0)
                    j += wp.int32(1)
                has_fsal = wp.int32(0)

            reached = accept and (
                wp.abs(t_new - next_target)
                <= wp.float64(1.0e-12) * wp.max(wp.float64(1.0), wp.abs(next_target))
            )
            if reached:
                j = wp.int32(0)
                while j < n_vars:
                    hist[i, save_idx, j] = y[i, j]
                    j += wp.int32(1)
                save_idx += wp.int32(1)

            if wp.isnan(err_norm) or err_norm > wp.float64(1.0e18):
                safe_err = wp.float64(1.0e18)
            elif err_norm == wp.float64(0.0):
                safe_err = wp.float64(1.0e-18)
            else:
                safe_err = err_norm
            factor = wp.float64(ck.SAFETY) * wp.pow(
                safe_err, wp.float64(-1.0) / wp.float64(5.0)
            )
            if factor < wp.float64(ck.FACTOR_MIN):
                factor = wp.float64(ck.FACTOR_MIN)
            elif factor > wp.float64(ck.FACTOR_MAX):
                factor = wp.float64(ck.FACTOR_MAX)
            dt = dt_use * factor
            t = t_new
            n_steps += wp.int32(1)

        accepted_out[i] = accepted_steps
        rejected_out[i] = rejected_steps
        loop_out[i] = n_steps

    return kernel


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
    wp.init()
    y0_arr, times, params_arr, dt0 = ck.normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    device = "cuda"
    y0_dev = wp.array(y0_arr, dtype=wp.float64, device=device)
    times_dev = wp.array(times, dtype=wp.float64, device=device)
    params_dev = wp.array(params_arr, dtype=wp.float64, device=device)
    hist_dev = wp.empty((n, times.shape[0], n_vars), dtype=wp.float64, device=device)
    accepted_dev = wp.empty(n, dtype=wp.int32, device=device)
    rejected_dev = wp.empty(n, dtype=wp.int32, device=device)
    loop_dev = wp.empty(n, dtype=wp.int32, device=device)
    work = [wp.empty((n, n_vars), dtype=wp.float64, device=device) for _ in range(9)]

    wp.launch(
        _make_kernel(ode_fn, n_vars),
        dim=n,
        inputs=[
            y0_dev,
            times_dev,
            params_dev,
            np.float64(dt0),
            np.float64(rtol),
            np.float64(atol),
            np.int32(max_steps),
            hist_dev,
            accepted_dev,
            rejected_dev,
            loop_dev,
            *work,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    solution = hist_dev.numpy()
    if not return_stats:
        return solution
    accepted_steps = accepted_dev.numpy()
    rejected_steps = rejected_dev.numpy()
    loop_steps = loop_dev.numpy()
    return solution, ck.numpy_stats(accepted_steps, rejected_steps, loop_steps)
