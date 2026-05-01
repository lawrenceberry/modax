"""Generic Tsit5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import cuda

from solvers import _tsit5ck_common as ck


@functools.cache
def _make_kernel(ode_fn, n_vars: int):
    @cuda.jit
    def kernel(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        y,
        u,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
    ):
        i = cuda.grid(1)
        if i >= y0.shape[0]:
            return

        for j in range(n_vars):
            y[i, j] = y0[i, j]
            hist[i, 0, j] = y0[i, j]
            k7[i, j] = 0.0

        n_save = times.shape[0]
        t = times[0]
        tf = times[n_save - 1]
        dt = dt0
        save_idx = 1
        n_steps = 0
        accepted_steps = 0
        rejected_steps = 0
        has_fsal = False

        while save_idx < n_save and t < tf and n_steps < max_steps:
            next_target = times[save_idx]
            dt_use = dt
            if dt_use > next_target - t:
                dt_use = next_target - t
            if dt_use < 1e-30:
                dt_use = 1e-30

            if has_fsal:
                for j in range(n_vars):
                    k1[i, j] = k7[i, j]
            else:
                ode_fn(y, t, params, k1, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (ck.A21 * k1[i, j])
            ode_fn(u, t + ck.C2 * dt_use, params, k2, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (ck.A31 * k1[i, j] + ck.A32 * k2[i, j])
            ode_fn(u, t + ck.C3 * dt_use, params, k3, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    ck.A41 * k1[i, j] + ck.A42 * k2[i, j] + ck.A43 * k3[i, j]
                )
            ode_fn(u, t + ck.C4 * dt_use, params, k4, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    ck.A51 * k1[i, j]
                    + ck.A52 * k2[i, j]
                    + ck.A53 * k3[i, j]
                    + ck.A54 * k4[i, j]
                )
            ode_fn(u, t + ck.C5 * dt_use, params, k5, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    ck.A61 * k1[i, j]
                    + ck.A62 * k2[i, j]
                    + ck.A63 * k3[i, j]
                    + ck.A64 * k4[i, j]
                    + ck.A65 * k5[i, j]
                )
            ode_fn(u, t + ck.C6 * dt_use, params, k6, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    ck.B1 * k1[i, j]
                    + ck.B2 * k2[i, j]
                    + ck.B3 * k3[i, j]
                    + ck.B4 * k4[i, j]
                    + ck.B5 * k5[i, j]
                    + ck.B6 * k6[i, j]
                )
            ode_fn(u, t + ck.C7 * dt_use, params, k7, i)

            err_sum = 0.0
            for j in range(n_vars):
                err_est = dt_use * (
                    ck.E1 * k1[i, j]
                    + ck.E2 * k2[i, j]
                    + ck.E3 * k3[i, j]
                    + ck.E4 * k4[i, j]
                    + ck.E5 * k5[i, j]
                    + ck.E6 * k6[i, j]
                    + ck.E7 * k7[i, j]
                )
                scale = atol + rtol * max(abs(y[i, j]), abs(u[i, j]))
                ratio = err_est / scale
                err_sum += ratio * ratio
            err_norm = math.sqrt(err_sum / n_vars)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

            if accept:
                t_new = t + dt_use
                for j in range(n_vars):
                    y[i, j] = u[i, j]
                accepted_steps += 1
                has_fsal = True
            else:
                t_new = t
                rejected_steps += 1
                for j in range(n_vars):
                    k7[i, j] = 0.0
                has_fsal = False

            reached = accept and (
                abs(t_new - next_target) <= 1e-12 * max(1.0, abs(next_target))
            )
            if reached:
                for j in range(n_vars):
                    hist[i, save_idx, j] = y[i, j]
                save_idx += 1

            if math.isnan(err_norm) or err_norm > 1e18:
                safe_err = 1e18
            elif err_norm == 0.0:
                safe_err = 1e-18
            else:
                safe_err = err_norm
            factor = ck.SAFETY * safe_err ** (-1.0 / 5.0)
            if factor < ck.FACTOR_MIN:
                factor = ck.FACTOR_MIN
            elif factor > ck.FACTOR_MAX:
                factor = ck.FACTOR_MAX
            dt = dt_use * factor
            t = t_new
            n_steps += 1

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
    y0_arr, times, params_arr, dt0 = ck.normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    hist_dev = cuda.device_array((n, times.shape[0], n_vars), dtype=np.float64)
    accepted_dev = cuda.device_array(n, dtype=np.int32)
    rejected_dev = cuda.device_array(n, dtype=np.int32)
    loop_dev = cuda.device_array(n, dtype=np.int32)
    work = [cuda.device_array((n, n_vars), dtype=np.float64) for _ in range(9)]

    threads = 128
    blocks = (n + threads - 1) // threads
    _make_kernel(ode_fn, n_vars)[blocks, threads](
        cuda.to_device(y0_arr),
        cuda.to_device(times),
        cuda.to_device(params_arr),
        np.float64(dt0),
        np.float64(rtol),
        np.float64(atol),
        np.int32(max_steps),
        hist_dev,
        accepted_dev,
        rejected_dev,
        loop_dev,
        *work,
    )
    cuda.synchronize()

    solution = hist_dev.copy_to_host()
    if not return_stats:
        return solution
    accepted_steps = accepted_dev.copy_to_host()
    rejected_steps = rejected_dev.copy_to_host()
    loop_steps = loop_dev.copy_to_host()
    return solution, ck.numpy_stats(accepted_steps, rejected_steps, loop_steps)
