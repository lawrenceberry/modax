"""Generic Rodas5 custom kernel using NVIDIA Warp."""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from solvers import _rodas5ck_common as ck


@functools.cache
def _make_kernel(ode_fn, jac_fn, n_vars: int):
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
        k8: wp.array2d(dtype=wp.float64),
        jac: wp.array3d(dtype=wp.float64),
        lu_mat: wp.array3d(dtype=wp.float32),
        piv_arr: wp.array2d(dtype=wp.int32),
    ):
        i = wp.tid()

        j = wp.int32(0)
        while j < n_vars:
            y[i, j] = y0[i, j]
            hist[i, 0, j] = y0[i, j]
            j += wp.int32(1)

        n_save = times.shape[0]
        t = times[0]
        tf = times[n_save - 1]
        dt = dt0
        save_idx = wp.int32(1)
        n_steps = wp.int32(0)
        accepted_steps = wp.int32(0)
        rejected_steps = wp.int32(0)

        while save_idx < n_save and t < tf and n_steps < max_steps:
            next_target = times[save_idx]
            dt_use = dt
            if dt_use > next_target - t:
                dt_use = next_target - t
            if dt_use < wp.float64(1.0e-30):
                dt_use = wp.float64(1.0e-30)
            inv_dt = wp.float64(1.0) / dt_use
            t_end = t + dt_use

            # Compute Jacobian at (y, t)
            jac_fn(y, t, params, jac, i)

            # Build M = (1/(gamma*dt)) * I - J and factorize in-place into lu_mat
            dtgamma_inv = wp.float64(1.0) / (dt_use * wp.float64(ck.GAMMA))
            row = wp.int32(0)
            while row < n_vars:
                col = wp.int32(0)
                while col < n_vars:
                    if row == col:
                        lu_mat[i, row, col] = wp.float32(dtgamma_inv - jac[i, row, col])
                    else:
                        lu_mat[i, row, col] = wp.float32(-jac[i, row, col])
                    col += wp.int32(1)
                row += wp.int32(1)

            # LU factorization with partial pivoting (in-place on lu_mat)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = k_lu
                max_v = wp.abs(lu_mat[i, k_lu, k_lu])
                m = k_lu + wp.int32(1)
                while m < n_vars:
                    if wp.abs(lu_mat[i, m, k_lu]) > max_v:
                        max_v = wp.abs(lu_mat[i, m, k_lu])
                        piv_k = m
                    m += wp.int32(1)
                piv_arr[i, k_lu] = piv_k
                if piv_k != k_lu:
                    col = wp.int32(0)
                    while col < n_vars:
                        tmp_lu = lu_mat[i, k_lu, col]
                        lu_mat[i, k_lu, col] = lu_mat[i, piv_k, col]
                        lu_mat[i, piv_k, col] = tmp_lu
                        col += wp.int32(1)
                m = k_lu + wp.int32(1)
                while m < n_vars:
                    fac = lu_mat[i, m, k_lu] / lu_mat[i, k_lu, k_lu]
                    lu_mat[i, m, k_lu] = fac
                    col = k_lu + wp.int32(1)
                    while col < n_vars:
                        lu_mat[i, m, col] -= fac * lu_mat[i, k_lu, col]
                        col += wp.int32(1)
                    m += wp.int32(1)
                k_lu += wp.int32(1)

            # ---- Stage 1 ----
            ode_fn(y, t, params, k1, i)
            # No C correction for stage 1
            # LU solve k1 in-place (perm + fwd + bwd)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k1[i, k_lu]
                    k1[i, k_lu] = k1[i, piv_k]
                    k1[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k1[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k1[i, col])
                    col += wp.int32(1)
                k1[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k1[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k1[i, col])
                    col += wp.int32(1)
                k1[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # ---- Stage 2 ----
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + wp.float64(ck.A21) * k1[i, j]
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C2) * dt_use, params, k2, i)
            j = wp.int32(0)
            while j < n_vars:
                k2[i, j] += wp.float64(ck.C21) * k1[i, j] * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k2[i, k_lu]
                    k2[i, k_lu] = k2[i, piv_k]
                    k2[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k2[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k2[i, col])
                    col += wp.int32(1)
                k2[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k2[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k2[i, col])
                    col += wp.int32(1)
                k2[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # ---- Stage 3 ----
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + (
                    wp.float64(ck.A31) * k1[i, j] + wp.float64(ck.A32) * k2[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C3) * dt_use, params, k3, i)
            j = wp.int32(0)
            while j < n_vars:
                k3[i, j] += (
                    wp.float64(ck.C31) * k1[i, j] + wp.float64(ck.C32) * k2[i, j]
                ) * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k3[i, k_lu]
                    k3[i, k_lu] = k3[i, piv_k]
                    k3[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k3[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k3[i, col])
                    col += wp.int32(1)
                k3[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k3[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k3[i, col])
                    col += wp.int32(1)
                k3[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # ---- Stage 4 ----
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + (
                    wp.float64(ck.A41) * k1[i, j]
                    + wp.float64(ck.A42) * k2[i, j]
                    + wp.float64(ck.A43) * k3[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C4) * dt_use, params, k4, i)
            j = wp.int32(0)
            while j < n_vars:
                k4[i, j] += (
                    wp.float64(ck.C41) * k1[i, j]
                    + wp.float64(ck.C42) * k2[i, j]
                    + wp.float64(ck.C43) * k3[i, j]
                ) * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k4[i, k_lu]
                    k4[i, k_lu] = k4[i, piv_k]
                    k4[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k4[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k4[i, col])
                    col += wp.int32(1)
                k4[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k4[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k4[i, col])
                    col += wp.int32(1)
                k4[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # ---- Stage 5 ----
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + (
                    wp.float64(ck.A51) * k1[i, j]
                    + wp.float64(ck.A52) * k2[i, j]
                    + wp.float64(ck.A53) * k3[i, j]
                    + wp.float64(ck.A54) * k4[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t + wp.float64(ck.C5) * dt_use, params, k5, i)
            j = wp.int32(0)
            while j < n_vars:
                k5[i, j] += (
                    wp.float64(ck.C51) * k1[i, j]
                    + wp.float64(ck.C52) * k2[i, j]
                    + wp.float64(ck.C53) * k3[i, j]
                    + wp.float64(ck.C54) * k4[i, j]
                ) * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k5[i, k_lu]
                    k5[i, k_lu] = k5[i, piv_k]
                    k5[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k5[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k5[i, col])
                    col += wp.int32(1)
                k5[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k5[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k5[i, col])
                    col += wp.int32(1)
                k5[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # ---- Stage 6 ----
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] = y[i, j] + (
                    wp.float64(ck.A61) * k1[i, j]
                    + wp.float64(ck.A62) * k2[i, j]
                    + wp.float64(ck.A63) * k3[i, j]
                    + wp.float64(ck.A64) * k4[i, j]
                    + wp.float64(ck.A65) * k5[i, j]
                )
                j += wp.int32(1)
            ode_fn(u, t_end, params, k6, i)
            j = wp.int32(0)
            while j < n_vars:
                k6[i, j] += (
                    wp.float64(ck.C61) * k1[i, j]
                    + wp.float64(ck.C62) * k2[i, j]
                    + wp.float64(ck.C63) * k3[i, j]
                    + wp.float64(ck.C64) * k4[i, j]
                    + wp.float64(ck.C65) * k5[i, j]
                ) * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k6[i, k_lu]
                    k6[i, k_lu] = k6[i, piv_k]
                    k6[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k6[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k6[i, col])
                    col += wp.int32(1)
                k6[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k6[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k6[i, col])
                    col += wp.int32(1)
                k6[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # u = u6 + k6  (now u7)
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] += k6[i, j]
                j += wp.int32(1)

            # ---- Stage 7 ----
            ode_fn(u, t_end, params, k7, i)
            j = wp.int32(0)
            while j < n_vars:
                k7[i, j] += (
                    wp.float64(ck.C71) * k1[i, j]
                    + wp.float64(ck.C72) * k2[i, j]
                    + wp.float64(ck.C73) * k3[i, j]
                    + wp.float64(ck.C74) * k4[i, j]
                    + wp.float64(ck.C75) * k5[i, j]
                    + wp.float64(ck.C76) * k6[i, j]
                ) * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k7[i, k_lu]
                    k7[i, k_lu] = k7[i, piv_k]
                    k7[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k7[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k7[i, col])
                    col += wp.int32(1)
                k7[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k7[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k7[i, col])
                    col += wp.int32(1)
                k7[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # u = u7 + k7  (now u8)
            j = wp.int32(0)
            while j < n_vars:
                u[i, j] += k7[i, j]
                j += wp.int32(1)

            # ---- Stage 8 ----
            ode_fn(u, t_end, params, k8, i)
            j = wp.int32(0)
            while j < n_vars:
                k8[i, j] += (
                    wp.float64(ck.C81) * k1[i, j]
                    + wp.float64(ck.C82) * k2[i, j]
                    + wp.float64(ck.C83) * k3[i, j]
                    + wp.float64(ck.C84) * k4[i, j]
                    + wp.float64(ck.C85) * k5[i, j]
                    + wp.float64(ck.C86) * k6[i, j]
                    + wp.float64(ck.C87) * k7[i, j]
                ) * inv_dt
                j += wp.int32(1)
            k_lu = wp.int32(0)
            while k_lu < n_vars:
                piv_k = piv_arr[i, k_lu]
                if piv_k != k_lu:
                    tmp_k = k8[i, k_lu]
                    k8[i, k_lu] = k8[i, piv_k]
                    k8[i, piv_k] = tmp_k
                k_lu += wp.int32(1)
            k_lu = wp.int32(1)
            while k_lu < n_vars:
                rhs_k = wp.float32(k8[i, k_lu])
                col = wp.int32(0)
                while col < k_lu:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k8[i, col])
                    col += wp.int32(1)
                k8[i, k_lu] = wp.float64(rhs_k)
                k_lu += wp.int32(1)
            k_lu -= wp.int32(1)
            while k_lu >= wp.int32(0):
                rhs_k = wp.float32(k8[i, k_lu])
                col = k_lu + wp.int32(1)
                while col < n_vars:
                    rhs_k -= lu_mat[i, k_lu, col] * wp.float32(k8[i, col])
                    col += wp.int32(1)
                k8[i, k_lu] = wp.float64(rhs_k / lu_mat[i, k_lu, k_lu])
                k_lu -= wp.int32(1)

            # Error estimate = k8; y_new = u8 + k8
            err_sum = wp.float64(0.0)
            j = wp.int32(0)
            while j < n_vars:
                y_new_j = u[i, j] + k8[i, j]
                scale = atol + rtol * wp.max(wp.abs(y[i, j]), wp.abs(y_new_j))
                ratio = k8[i, j] / scale
                err_sum += ratio * ratio
                j += wp.int32(1)
            err_norm = wp.sqrt(err_sum / wp.float64(n_vars))
            accept = err_norm <= wp.float64(1.0) and not wp.isnan(err_norm)

            t_new = t
            if accept:
                t_new = t + dt_use
                j = wp.int32(0)
                while j < n_vars:
                    y[i, j] = u[i, j] + k8[i, j]
                    j += wp.int32(1)
                accepted_steps += wp.int32(1)
            else:
                rejected_steps += wp.int32(1)

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
                safe_err, wp.float64(-1.0) / wp.float64(6.0)
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
    jac_fn,
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

    flat_work = [
        wp.empty((n, n_vars), dtype=wp.float64, device=device) for _ in range(10)
    ]
    jac_dev = wp.empty((n, n_vars, n_vars), dtype=wp.float64, device=device)
    lu_dev = wp.empty((n, n_vars, n_vars), dtype=wp.float32, device=device)
    piv_dev = wp.empty((n, n_vars), dtype=wp.int32, device=device)

    wp.launch(
        _make_kernel(ode_fn, jac_fn, n_vars),
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
            *flat_work,  # y, u, k1..k8
            jac_dev,
            lu_dev,
            piv_dev,
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
