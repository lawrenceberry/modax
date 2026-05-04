"""Generic Tsit5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import cuda

# fmt: off
C2 = 161.0 / 1000.0
C3 = 327.0 / 1000.0
C4 = 9.0 / 10.0
C5 = 0.9800255409045097
C6 = 1.0
C7 = 1.0

A21 = 161.0 / 1000.0

A31 = -0.008480655492356989
A32 = 0.335480655492357

A41 = 2.8971530571054935
A42 = -6.359448489975075
A43 = 4.3622954328695815

A51 = 5.325864828439257
A52 = -11.748883564062828
A53 = 7.4955393428898365
A54 = -0.09249506636175525

A61 = 5.86145544294642
A62 = -12.92096931784711
A63 = 8.159367898576159
A64 = -0.071584973281401
A65 = -0.028269050394068383

A71 = 0.09646076681806523
A72 = 0.01
A73 = 0.4798896504144996
A74 = 1.379008574103742
A75 = -3.2900695154360807
A76 = 2.324710524099774

B1 = A71
B2 = A72
B3 = A73
B4 = A74
B5 = A75
B6 = A76

E1 = 0.0017800620525794302
E2 = 0.000816434459656747
E3 = -0.007880878010261985
E4 = 0.14471100717326298
E5 = -0.5823571654525553
E6 = 0.45808210592918695
E7 = -1.0 / 66.0
# fmt: on

SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 10.0


def normalize_inputs(y0, t_span, params, first_step):
    y0_in = np.asarray(y0, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    times = np.asarray(t_span, dtype=np.float64)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
        params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
    else:
        n = y0_in.shape[0]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )

    if y0_arr.ndim != 2:
        raise ValueError("custom-kernel Tsit5 expects y0 shape (N, n_vars)")
    if params_arr.ndim != 2:
        raise ValueError("custom-kernel Tsit5 expects params shape (N, n_params)")
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("t_span must be a 1-D array with at least two save times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_span must be strictly increasing")

    dt0 = (
        np.float64(first_step)
        if first_step is not None
        else np.float64((times[-1] - times[0]) * 1e-6)
    )
    return y0_arr, times, params_arr, dt0


def numpy_stats(accepted_steps, rejected_steps, loop_steps):
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": np.ones_like(loop_steps, dtype=np.int32),
    }


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
                u[i, j] = y[i, j] + dt_use * (A21 * k1[i, j])
            ode_fn(u, t + C2 * dt_use, params, k2, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (A31 * k1[i, j] + A32 * k2[i, j])
            ode_fn(u, t + C3 * dt_use, params, k3, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    A41 * k1[i, j] + A42 * k2[i, j] + A43 * k3[i, j]
                )
            ode_fn(u, t + C4 * dt_use, params, k4, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    A51 * k1[i, j] + A52 * k2[i, j] + A53 * k3[i, j] + A54 * k4[i, j]
                )
            ode_fn(u, t + C5 * dt_use, params, k5, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    A61 * k1[i, j]
                    + A62 * k2[i, j]
                    + A63 * k3[i, j]
                    + A64 * k4[i, j]
                    + A65 * k5[i, j]
                )
            ode_fn(u, t + C6 * dt_use, params, k6, i)

            for j in range(n_vars):
                u[i, j] = y[i, j] + dt_use * (
                    B1 * k1[i, j]
                    + B2 * k2[i, j]
                    + B3 * k3[i, j]
                    + B4 * k4[i, j]
                    + B5 * k5[i, j]
                    + B6 * k6[i, j]
                )
            ode_fn(u, t + C7 * dt_use, params, k7, i)

            err_sum = 0.0
            for j in range(n_vars):
                err_est = dt_use * (
                    E1 * k1[i, j]
                    + E2 * k2[i, j]
                    + E3 * k3[i, j]
                    + E4 * k4[i, j]
                    + E5 * k5[i, j]
                    + E6 * k6[i, j]
                    + E7 * k7[i, j]
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
            factor = SAFETY * safe_err ** (-1.0 / 5.0)
            if factor < FACTOR_MIN:
                factor = FACTOR_MIN
            elif factor > FACTOR_MAX:
                factor = FACTOR_MAX
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
    y0_arr, times, params_arr, dt0 = normalize_inputs(y0, t_span, params, first_step)
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
    return solution, numpy_stats(accepted_steps, rejected_steps, loop_steps)
