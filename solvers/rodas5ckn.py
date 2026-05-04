"""Cooperative packed-batch Rodas5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import cuda

from solvers import _rodas5ck_common as ck
from solvers._rodas5ckn_common import (
    PreparedSolve,
    as_launch_block_dim,
    block_threads_x,
    get_workspace,
    make_lu_solver,
)

_WORKSPACE_CACHE: dict[tuple[int, int, int, int], object] = {}


@functools.cache
def _make_kernel(ode_fn, jac_fn, n_vars: int):
    lu_solver = make_lu_solver(n_vars)
    batches_per_block = int(lu_solver.batches_per_block)

    block_dim = as_launch_block_dim(lu_solver.block_dim)
    block_threads = block_threads_x(block_dim)
    vec_size = batches_per_block * n_vars
    a_size = int(lu_solver.a_size())
    b_size = int(lu_solver.b_size())
    ipiv_size = int(lu_solver.ipiv_size)

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
        y_global,
        u_global,
        work_global,
        jac,
    ):
        tx = cuda.threadIdx.x
        batch = tx % batches_per_block
        lane = tx // batches_per_block
        batch_lanes = (
            block_threads + batches_per_block - 1 - batch
        ) // batches_per_block
        block_start = cuda.blockIdx.x * batches_per_block
        i = block_start + batch

        n_save = times.shape[0]
        tf = times[n_save - 1]
        v_offset = batch * n_vars
        a_offset = batch * n_vars * n_vars
        b_offset = batch * n_vars

        smem_lu = cuda.shared.array(shape=a_size, dtype=np.float32)
        smem_rhs = cuda.shared.array(shape=b_size, dtype=np.float32)
        smem_ipiv = cuda.shared.array(shape=ipiv_size, dtype=np.int32)
        smem_info = cuda.shared.array(shape=batches_per_block, dtype=np.int32)

        smem_y = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_u = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k1 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k2 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k3 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k4 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k5 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k6 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k7 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k8 = cuda.shared.array(shape=vec_size, dtype=np.float64)

        smem_err = cuda.shared.array(shape=block_threads, dtype=np.float64)
        smem_t = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt_use = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_inv_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_t_end = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_next_target = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_save_idx = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_n_steps = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accepted = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_rejected = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accept = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_reached = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_hist_idx = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_continue = cuda.shared.array(shape=1, dtype=np.int32)

        if i < y0.shape[0]:
            for j in range(lane, n_vars, batch_lanes):
                val = y0[i, j]
                smem_y[v_offset + j] = val
                y_global[i, j] = val
                hist[i, 0, j] = val
        if tx < batches_per_block:
            if i < y0.shape[0]:
                smem_t[batch] = times[0]
                smem_save_idx[batch] = 1
            else:
                smem_t[batch] = tf
                smem_save_idx[batch] = n_save
            smem_dt[batch] = dt0
            smem_n_steps[batch] = 0
            smem_accepted[batch] = 0
            smem_rejected[batch] = 0
            smem_accept[batch] = 0
            smem_reached[batch] = 0
        if tx == 0:
            smem_continue[0] = 1
        cuda.syncthreads()

        while smem_continue[0] != 0:
            active = (
                i < y0.shape[0]
                and smem_save_idx[batch] < n_save
                and smem_t[batch] < tf
                and smem_n_steps[batch] < max_steps
            )

            if lane == 0:
                if active:
                    next_target = times[smem_save_idx[batch]]
                    dt_use = smem_dt[batch]
                    if dt_use > next_target - smem_t[batch]:
                        dt_use = next_target - smem_t[batch]
                    if dt_use < 1e-30:
                        dt_use = 1e-30
                    smem_next_target[batch] = next_target
                    smem_dt_use[batch] = dt_use
                    smem_inv_dt[batch] = 1.0 / dt_use
                    smem_t_end[batch] = smem_t[batch] + dt_use
                else:
                    smem_dt_use[batch] = dt0
                    smem_inv_dt[batch] = 1.0 / dt0
                    smem_t_end[batch] = smem_t[batch]
            cuda.syncthreads()

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    y_global[i, j] = smem_y[v_offset + j]
            cuda.syncthreads()
            if lane == 0 and active:
                jac_fn(y_global, smem_t[batch], params, jac, i)
            cuda.syncthreads()

            dtgamma_inv = 1.0 / (smem_dt_use[batch] * ck.GAMMA)
            for idx_local in range(lane, n_vars * n_vars, batch_lanes):
                row = idx_local // n_vars
                col = idx_local - row * n_vars
                idx = a_offset + idx_local
                if active:
                    if row == col:
                        smem_lu[idx] = np.float32(dtgamma_inv - jac[i, row, col])
                    else:
                        smem_lu[idx] = np.float32(-jac[i, row, col])
                else:
                    smem_lu[idx] = 1.0 if row == col else 0.0
            if not active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = 0.0
            cuda.syncthreads()
            lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
            cuda.syncthreads()

            if lane == 0 and active:
                ode_fn(y_global, smem_t[batch], params, work_global, i)
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(work_global[i, j])
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k1[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = (
                        smem_y[v_offset + j] + ck.A21 * smem_k1[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if lane == 0 and active:
                ode_fn(
                    u_global,
                    smem_t[batch] + ck.C2 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + ck.C21 * smem_k1[v_offset + j] * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k2[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        ck.A31 * smem_k1[v_offset + j] + ck.A32 * smem_k2[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if lane == 0 and active:
                ode_fn(
                    u_global,
                    smem_t[batch] + ck.C3 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + (
                            ck.C31 * smem_k1[v_offset + j]
                            + ck.C32 * smem_k2[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k3[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        ck.A41 * smem_k1[v_offset + j]
                        + ck.A42 * smem_k2[v_offset + j]
                        + ck.A43 * smem_k3[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if lane == 0 and active:
                ode_fn(
                    u_global,
                    smem_t[batch] + ck.C4 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + (
                            ck.C41 * smem_k1[v_offset + j]
                            + ck.C42 * smem_k2[v_offset + j]
                            + ck.C43 * smem_k3[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k4[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        ck.A51 * smem_k1[v_offset + j]
                        + ck.A52 * smem_k2[v_offset + j]
                        + ck.A53 * smem_k3[v_offset + j]
                        + ck.A54 * smem_k4[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if lane == 0 and active:
                ode_fn(
                    u_global,
                    smem_t[batch] + ck.C5 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + (
                            ck.C51 * smem_k1[v_offset + j]
                            + ck.C52 * smem_k2[v_offset + j]
                            + ck.C53 * smem_k3[v_offset + j]
                            + ck.C54 * smem_k4[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k5[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        ck.A61 * smem_k1[v_offset + j]
                        + ck.A62 * smem_k2[v_offset + j]
                        + ck.A63 * smem_k3[v_offset + j]
                        + ck.A64 * smem_k4[v_offset + j]
                        + ck.A65 * smem_k5[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if lane == 0 and active:
                ode_fn(u_global, smem_t_end[batch], params, work_global, i)
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + (
                            ck.C61 * smem_k1[v_offset + j]
                            + ck.C62 * smem_k2[v_offset + j]
                            + ck.C63 * smem_k3[v_offset + j]
                            + ck.C64 * smem_k4[v_offset + j]
                            + ck.C65 * smem_k5[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k6[v_offset + j] = np.float64(smem_rhs[b_offset + j])
                    smem_u[v_offset + j] += smem_k6[v_offset + j]
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()

            if lane == 0 and active:
                ode_fn(u_global, smem_t_end[batch], params, work_global, i)
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + (
                            ck.C71 * smem_k1[v_offset + j]
                            + ck.C72 * smem_k2[v_offset + j]
                            + ck.C73 * smem_k3[v_offset + j]
                            + ck.C74 * smem_k4[v_offset + j]
                            + ck.C75 * smem_k5[v_offset + j]
                            + ck.C76 * smem_k6[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k7[v_offset + j] = np.float64(smem_rhs[b_offset + j])
                    smem_u[v_offset + j] += smem_k7[v_offset + j]
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()

            if lane == 0 and active:
                ode_fn(u_global, smem_t_end[batch], params, work_global, i)
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = np.float32(
                        work_global[i, j]
                        + (
                            ck.C81 * smem_k1[v_offset + j]
                            + ck.C82 * smem_k2[v_offset + j]
                            + ck.C83 * smem_k3[v_offset + j]
                            + ck.C84 * smem_k4[v_offset + j]
                            + ck.C85 * smem_k5[v_offset + j]
                            + ck.C86 * smem_k6[v_offset + j]
                            + ck.C87 * smem_k7[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k8[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            err_local = 0.0
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    y_new_j = smem_u[v_offset + j] + smem_k8[v_offset + j]
                    scale = atol + rtol * max(
                        math.fabs(smem_y[v_offset + j]), math.fabs(y_new_j)
                    )
                    ratio = smem_k8[v_offset + j] / scale
                    err_local += ratio * ratio
            smem_err[tx] = err_local
            cuda.syncthreads()

            if lane == 0:
                for other_lane in range(1, batch_lanes):
                    smem_err[tx] += smem_err[batch + other_lane * batches_per_block]

                if active:
                    err_norm = math.sqrt(smem_err[tx] / n_vars)
                    accept = err_norm <= 1.0 and not math.isnan(err_norm)
                    smem_accept[batch] = 1 if accept else 0
                    smem_reached[batch] = 0
                    smem_hist_idx[batch] = smem_save_idx[batch]

                    t_new = smem_t[batch]
                    if accept:
                        t_new = smem_t[batch] + smem_dt_use[batch]
                        smem_accepted[batch] += 1
                        reached = abs(t_new - smem_next_target[batch]) <= 1e-12 * max(
                            1.0, abs(smem_next_target[batch])
                        )
                        smem_reached[batch] = 1 if reached else 0
                        if reached:
                            smem_save_idx[batch] += 1
                    else:
                        smem_rejected[batch] += 1

                    if math.isnan(err_norm) or err_norm > 1e18:
                        safe_err = 1e18
                    elif err_norm == 0.0:
                        safe_err = 1e-18
                    else:
                        safe_err = err_norm
                    factor = ck.SAFETY * safe_err ** (-1.0 / 6.0)
                    if factor < ck.FACTOR_MIN:
                        factor = ck.FACTOR_MIN
                    elif factor > ck.FACTOR_MAX:
                        factor = ck.FACTOR_MAX
                    smem_dt[batch] = smem_dt_use[batch] * factor
                    smem_t[batch] = t_new
                    smem_n_steps[batch] += 1
                else:
                    smem_accept[batch] = 0
                    smem_reached[batch] = 0
            cuda.syncthreads()

            if smem_accept[batch] != 0:
                for j in range(lane, n_vars, batch_lanes):
                    smem_y[v_offset + j] = smem_u[v_offset + j] + smem_k8[v_offset + j]
            cuda.syncthreads()

            if smem_reached[batch] != 0:
                hist_idx = smem_hist_idx[batch]
                for j in range(lane, n_vars, batch_lanes):
                    hist[i, hist_idx, j] = smem_y[v_offset + j]
            cuda.syncthreads()

            if tx == 0:
                keep_going = 0
                for b in range(batches_per_block):
                    bi = block_start + b
                    if (
                        bi < y0.shape[0]
                        and smem_save_idx[b] < n_save
                        and smem_t[b] < tf
                        and smem_n_steps[b] < max_steps
                    ):
                        keep_going = 1
                smem_continue[0] = keep_going
            cuda.syncthreads()

        if tx < batches_per_block and i < y0.shape[0]:
            accepted_out[i] = smem_accepted[batch]
            rejected_out[i] = smem_rejected[batch]
            loop_out[i] = smem_n_steps[batch]

    return kernel, lu_solver


def prepare_solve(
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
):
    del batch_size
    y0_arr, times, params_arr, dt0 = ck.normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]

    workspace = get_workspace(_WORKSPACE_CACHE, n, n_vars, n_save, n_params)
    workspace.y0_dev.copy_to_device(y0_arr)
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)

    kernel, lu_solver = _make_kernel(ode_fn, jac_fn, n_vars)
    batches_per_block = int(lu_solver.batches_per_block)
    threads = as_launch_block_dim(lu_solver.block_dim)
    blocks = (n + batches_per_block - 1) // batches_per_block

    return PreparedSolve(
        kernel=kernel,
        lu_solver=lu_solver,
        workspace=workspace,
        dt0=np.float64(dt0),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
        max_steps=np.int32(max_steps),
        blocks=blocks,
        threads=threads,
    )


def run_prepared(prepared: PreparedSolve, *, return_stats=False, copy_solution=True):
    workspace = prepared.workspace
    prepared.kernel[prepared.blocks, prepared.threads](
        workspace.y0_dev,
        workspace.times_dev,
        workspace.params_dev,
        prepared.dt0,
        prepared.rtol,
        prepared.atol,
        prepared.max_steps,
        workspace.hist_dev,
        workspace.accepted_dev,
        workspace.rejected_dev,
        workspace.loop_dev,
        workspace.work[0],
        workspace.work[1],
        workspace.work[2],
        workspace.jac_dev,
    )
    cuda.synchronize()

    solution = (
        workspace.hist_dev.copy_to_host() if copy_solution else workspace.hist_dev
    )
    if not return_stats:
        return solution

    accepted_steps = workspace.accepted_dev.copy_to_host()
    rejected_steps = workspace.rejected_dev.copy_to_host()
    loop_steps = workspace.loop_dev.copy_to_host()
    return solution, ck.numpy_stats(accepted_steps, rejected_steps, loop_steps)


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
    prepared = prepare_solve(
        ode_fn,
        jac_fn,
        y0,
        t_span,
        params,
        batch_size=batch_size,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
    return run_prepared(prepared, return_stats=return_stats, copy_solution=True)
