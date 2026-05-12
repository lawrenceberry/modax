"""Generic KenCarp5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import gc
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import cuda
from nvmath.device import LUPivotSolver

GAMMA = 41.0 / 200.0

B_SOL0 = -872700587467.0 / 9133579230613.0
B_SOL1 = 0.0
B_SOL2 = 0.0
B_SOL3 = 22348218063261.0 / 9555858737531.0
B_SOL4 = -1143369518992.0 / 8141816002931.0
B_SOL5 = -39379526789629.0 / 19018526304540.0
B_SOL6 = 32727382324388.0 / 42900044865799.0
B_SOL7 = GAMMA

B_ERR0 = B_SOL0 - (-975461918565.0 / 9796059967033.0)
B_ERR1 = 0.0
B_ERR2 = 0.0
B_ERR3 = B_SOL3 - (78070527104295.0 / 32432590147079.0)
B_ERR4 = B_SOL4 - (-548382580838.0 / 3424219808633.0)
B_ERR5 = B_SOL5 - (-33438840321285.0 / 15594753105479.0)
B_ERR6 = B_SOL6 - (3629800801594.0 / 4656183773603.0)
B_ERR7 = B_SOL7 - (4035322873751.0 / 18575991585200.0)

C1 = 41.0 / 100.0
C2 = 2935347310677.0 / 11292855782101.0
C3 = 1426016391358.0 / 7196633302097.0
C4 = 92.0 / 100.0
C5 = 24.0 / 100.0
C6 = 3.0 / 5.0
C7 = 1.0

SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 10.0
NEWTON_MAX_ITERS = 10

_WORKSPACE_CACHE: dict[tuple[int, int, int, int], object] = {}


def clear_caches() -> None:
    _WORKSPACE_CACHE.clear()
    _make_kernel.cache_clear()
    gc.collect()


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
        raise ValueError("custom-kernel KenCarp5 expects y0 shape (N, n_vars)")
    if params_arr.ndim != 2:
        raise ValueError("custom-kernel KenCarp5 expects params shape (N, n_params)")
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


@dataclass
class Workspace:
    y0_dev: Any
    times_dev: Any
    params_dev: Any
    hist_dev: Any
    accepted_dev: Any
    rejected_dev: Any
    loop_dev: Any
    y_dev: Any
    u_dev: Any
    tmp_dev: Any
    base_dev: Any
    rhs_dev: Any
    stage_y_dev: Any
    stage_fe_dev: Any
    stage_fi_dev: Any
    jac_dev: Any


@dataclass(frozen=True)
class PreparedSolve:
    kernel: Any
    lu_solver: Any
    workspace: Workspace
    linear: np.int32
    dt0: np.float64
    rtol: np.float64
    atol: np.float64
    max_steps: np.int32
    blocks: int
    threads: Any


def make_lu_solver(
    n_vars: int,
    *,
    batches_per_block=1,
    block_dim="suggested",
):
    return LUPivotSolver(
        size=(n_vars, n_vars, 1),
        precision=np.float64,
        execution="Block",
        arrangement=("row_major", "row_major"),
        batches_per_block=batches_per_block,
        block_dim=block_dim,
    )


def as_launch_block_dim(block_dim):
    if isinstance(block_dim, int):
        return block_dim
    if isinstance(block_dim, (tuple, list)):
        return tuple(int(x) for x in block_dim)
    x = getattr(block_dim, "x", None)
    y = getattr(block_dim, "y", 1)
    z = getattr(block_dim, "z", 1)
    if x is not None:
        return (int(x), int(y), int(z))
    return 128


def block_threads_x(block_dim) -> int:
    launch = as_launch_block_dim(block_dim)
    if isinstance(launch, int):
        return launch
    return int(launch[0])


def get_workspace(
    cache: dict, n: int, n_vars: int, n_save: int, n_params: int
) -> Workspace:
    key = (n, n_vars, n_save, n_params)
    workspace = cache.get(key)
    if workspace is not None:
        return workspace

    workspace = Workspace(
        y0_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        times_dev=cuda.device_array(n_save, dtype=np.float64),
        params_dev=cuda.device_array((n, n_params), dtype=np.float64),
        hist_dev=cuda.device_array((n, n_save, n_vars), dtype=np.float64),
        accepted_dev=cuda.device_array(n, dtype=np.int32),
        rejected_dev=cuda.device_array(n, dtype=np.int32),
        loop_dev=cuda.device_array(n, dtype=np.int32),
        y_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        u_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        tmp_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        base_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        rhs_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        stage_y_dev=cuda.device_array((n, 8, n_vars), dtype=np.float64),
        stage_fe_dev=cuda.device_array((n, 8, n_vars), dtype=np.float64),
        stage_fi_dev=cuda.device_array((n, 8, n_vars), dtype=np.float64),
        jac_dev=cuda.device_array((n, n_vars, n_vars), dtype=np.float64),
    )
    cache[key] = workspace
    return workspace


@cuda.jit(device=True)
def _c(stage):
    if stage == 1:
        return C1
    if stage == 2:
        return C2
    if stage == 3:
        return C3
    if stage == 4:
        return C4
    if stage == 5:
        return C5
    if stage == 6:
        return C6
    if stage == 7:
        return C7
    return 0.0


@cuda.jit(device=True)
def _b_err(stage):
    if stage == 0:
        return B_ERR0
    if stage == 1:
        return B_ERR1
    if stage == 2:
        return B_ERR2
    if stage == 3:
        return B_ERR3
    if stage == 4:
        return B_ERR4
    if stage == 5:
        return B_ERR5
    if stage == 6:
        return B_ERR6
    return B_ERR7


@cuda.jit(device=True)
def _a_explicit(stage, prev):
    if stage == 1 and prev == 0:
        return 41.0 / 100.0
    if stage == 2:
        if prev == 0:
            return 367902744464.0 / 2072280473677.0
        if prev == 1:
            return 677623207551.0 / 8224143866563.0
    if stage == 3:
        if prev == 0:
            return 1268023523408.0 / 10340822734521.0
        if prev == 2:
            return 1029933939417.0 / 13636558850479.0
    if stage == 4:
        if prev == 0:
            return 14463281900351.0 / 6315353703477.0
        if prev == 2:
            return 66114435211212.0 / 5879490589093.0
        if prev == 3:
            return -54053170152839.0 / 4284798021562.0
    if stage == 5:
        if prev == 0:
            return 14090043504691.0 / 34967701212078.0
        if prev == 2:
            return 15191511035443.0 / 11219624916014.0
        if prev == 3:
            return -18461159152457.0 / 12425892160975.0
        if prev == 4:
            return -281667163811.0 / 9011619295870.0
    if stage == 6:
        if prev == 0:
            return 19230459214898.0 / 13134317526959.0
        if prev == 2:
            return 21275331358303.0 / 2942455364971.0
        if prev == 3:
            return -38145345988419.0 / 4862620318723.0
        if prev == 4 or prev == 5:
            return -1.0 / 8.0
    if stage == 7:
        if prev == 0:
            return -19977161125411.0 / 11928030595625.0
        if prev == 2:
            return -40795976796054.0 / 6384907823539.0
        if prev == 3:
            return 177454434618887.0 / 12078138498510.0
        if prev == 4:
            return 782672205425.0 / 8267701900261.0
        if prev == 5:
            return -69563011059811.0 / 9646580694205.0
        if prev == 6:
            return 7356628210526.0 / 4942186776405.0
    return 0.0


@cuda.jit(device=True)
def _a_implicit(stage, prev):
    if stage == 1 and prev == 1:
        return GAMMA
    if stage == 2:
        if prev == 0:
            return 41.0 / 400.0
        if prev == 1:
            return -567603406766.0 / 11931857230679.0
        if prev == 2:
            return GAMMA
    if stage == 3:
        if prev == 0:
            return 683785636431.0 / 9252920307686.0
        if prev == 2:
            return -110385047103.0 / 1367015193373.0
        if prev == 3:
            return GAMMA
    if stage == 4:
        if prev == 0:
            return 3016520224154.0 / 10081342136671.0
        if prev == 2:
            return 30586259806659.0 / 12414158314087.0
        if prev == 3:
            return -22760509404356.0 / 11113319521817.0
        if prev == 4:
            return GAMMA
    if stage == 5:
        if prev == 0:
            return 218866479029.0 / 1489978393911.0
        if prev == 2:
            return 638256894668.0 / 5436446318841.0
        if prev == 3:
            return -1179710474555.0 / 5321154724896.0
        if prev == 4:
            return -60928119172.0 / 8023461067671.0
        if prev == 5:
            return GAMMA
    if stage == 6:
        if prev == 0:
            return 1020004230633.0 / 5715676835656.0
        if prev == 2:
            return 25762820946817.0 / 25263940353407.0
        if prev == 3:
            return -2161375909145.0 / 9755907335909.0
        if prev == 4:
            return -211217309593.0 / 5846859502534.0
        if prev == 5:
            return -4269925059573.0 / 7827059040719.0
        if prev == 6:
            return GAMMA
    if stage == 7:
        if prev == 0:
            return B_SOL0
        if prev == 1:
            return B_SOL1
        if prev == 2:
            return B_SOL2
        if prev == 3:
            return B_SOL3
        if prev == 4:
            return B_SOL4
        if prev == 5:
            return B_SOL5
        if prev == 6:
            return B_SOL6
        if prev == 7:
            return GAMMA
    return 0.0


@cuda.jit(device=True)
def _predictor_coeff(stage, prev):
    if stage == 1:
        return 1.0 if prev == 0 else 0.0
    if stage == 2:
        if prev == 0:
            return 1.0 - C2 / C1
        if prev == 1:
            return C2 / C1
    if stage == 3:
        if prev == 0:
            return 1.0 - C3 / C1
        if prev == 1:
            return C3 / C1
    if stage == 4:
        if prev == 0:
            return 1.0 - C4 / C2
        if prev == 2:
            return C4 / C2
    if stage == 5:
        if prev == 0:
            return 1.0 - C5 / C4
        if prev == 4:
            return C5 / C4
    if stage == 6:
        if prev == 0:
            return 1.0 - C6 / C4
        if prev == 4:
            return C6 / C4
    if stage == 7:
        if prev == 0:
            return 1.0 - C7 / C4
        if prev == 4:
            return C7 / C4
    return 0.0


@cuda.jit(device=True)
def _clear_jac(jac, i, n_vars):
    for r in range(n_vars):
        for c in range(n_vars):
            jac[i, r, c] = 0.0


@functools.cache
def _make_kernel(explicit_ode_fn, implicit_ode_fn, implicit_jac_fn, n_vars: int):
    lu_solver = make_lu_solver(n_vars, batches_per_block=1)
    batches_per_block = int(lu_solver.batches_per_block)

    block_dim = as_launch_block_dim(lu_solver.block_dim)
    block_threads = block_threads_x(block_dim)
    a_size = int(lu_solver.a_size())
    b_size = int(lu_solver.b_size())
    ipiv_size = int(lu_solver.ipiv_size)

    @cuda.jit
    def kernel(
        y0,
        times,
        params,
        linear,
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
        tmp,
        base_vec,
        rhs,
        stage_y,
        stage_fe,
        stage_fi,
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
        a_offset = batch * n_vars * n_vars
        b_offset = batch * n_vars

        smem_lu = cuda.shared.array(shape=a_size, dtype=np.float64)
        smem_rhs = cuda.shared.array(shape=b_size, dtype=np.float64)
        smem_ipiv = cuda.shared.array(shape=ipiv_size, dtype=np.int32)
        smem_info = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_delta = cuda.shared.array(shape=block_threads, dtype=np.float64)
        smem_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_failed = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_converged = cuda.shared.array(shape=batches_per_block, dtype=np.int32)

        for j in range(n_vars):
            if i < y0.shape[0] and j % batch_lanes == lane:
                y[i, j] = y0[i, j]
                u[i, j] = y0[i, j]
                hist[i, 0, j] = y0[i, j]
        cuda.syncthreads()

        n_save = times.shape[0]
        t = times[0]
        tf = times[n_save - 1]
        dt = dt0
        save_idx = 1
        n_steps = 0
        accepted = 0
        rejected = 0

        while i < y0.shape[0] and save_idx < n_save and t < tf and n_steps < max_steps:
            next_target = times[save_idx]
            dt_use = dt
            if dt_use > next_target - t:
                dt_use = next_target - t
            if dt_use < 1e-30:
                dt_use = 1e-30

            failed = False
            for j in range(lane, n_vars, batch_lanes):
                stage_y[i, 0, j] = y[i, j]
                u[i, j] = y[i, j]
            cuda.syncthreads()
            if lane == 0:
                explicit_ode_fn(u, t, params, tmp, i)
            cuda.syncthreads()
            for j in range(lane, n_vars, batch_lanes):
                stage_fe[i, 0, j] = tmp[i, j]
                if not math.isfinite(tmp[i, j]):
                    failed = True
            cuda.syncthreads()
            if lane == 0:
                implicit_ode_fn(u, t, params, tmp, i)
            cuda.syncthreads()
            for j in range(lane, n_vars, batch_lanes):
                stage_fi[i, 0, j] = tmp[i, j]
                if not math.isfinite(tmp[i, j]):
                    failed = True

            for stage in range(1, 8):
                t_stage = t + _c(stage) * dt_use
                gamma_dt = GAMMA * dt_use
                for j in range(lane, n_vars, batch_lanes):
                    base = y[i, j]
                    pred = 0.0
                    for prev in range(stage):
                        ae = _a_explicit(stage, prev)
                        ai = _a_implicit(stage, prev)
                        pc = _predictor_coeff(stage, prev)
                        if ae != 0.0:
                            base += dt_use * ae * stage_fe[i, prev, j]
                        if ai != 0.0:
                            base += dt_use * ai * stage_fi[i, prev, j]
                        if pc != 0.0:
                            pred += pc * stage_y[i, prev, j]
                    base_vec[i, j] = base
                    rhs[i, j] = base
                    u[i, j] = pred
                cuda.syncthreads()
                if lane == 0:
                    smem_failed[batch] = 0
                cuda.syncthreads()
                if failed:
                    smem_failed[batch] = 1
                cuda.syncthreads()
                failed = smem_failed[batch] != 0

                if linear != 0:
                    if lane == 0:
                        _clear_jac(jac, i, n_vars)
                        implicit_jac_fn(u, t_stage, params, jac, i)
                    cuda.syncthreads()
                    for idx_local in range(lane, n_vars * n_vars, batch_lanes):
                        row = idx_local // n_vars
                        col = idx_local - row * n_vars
                        val = -gamma_dt * jac[i, row, col]
                        if row == col:
                            val += 1.0
                        smem_lu[a_offset + idx_local] = val
                    for j in range(lane, n_vars, batch_lanes):
                        smem_rhs[b_offset + j] = rhs[i, j]
                    cuda.syncthreads()
                    lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
                    cuda.syncthreads()
                    lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
                    cuda.syncthreads()
                    for j in range(lane, n_vars, batch_lanes):
                        val = smem_rhs[b_offset + j]
                        rhs[i, j] = val
                        stage_y[i, stage, j] = val
                        u[i, j] = val
                        if not math.isfinite(val):
                            failed = True
                    cuda.syncthreads()
                    if lane == 0:
                        implicit_ode_fn(u, t_stage, params, tmp, i)
                    cuda.syncthreads()
                    for j in range(lane, n_vars, batch_lanes):
                        stage_fi[i, stage, j] = tmp[i, j]
                        if not math.isfinite(tmp[i, j]):
                            failed = True
                else:
                    converged = False
                    it = 0
                    while (not converged) and (not failed) and it < NEWTON_MAX_ITERS:
                        if lane == 0:
                            implicit_ode_fn(u, t_stage, params, tmp, i)
                            _clear_jac(jac, i, n_vars)
                            implicit_jac_fn(u, t_stage, params, jac, i)
                        cuda.syncthreads()
                        delta_norm_acc = 0.0
                        for j in range(lane, n_vars, batch_lanes):
                            rhs[i, j] = u[i, j] - base_vec[i, j] - gamma_dt * tmp[i, j]
                            smem_rhs[b_offset + j] = rhs[i, j]
                        for idx_local in range(lane, n_vars * n_vars, batch_lanes):
                            row = idx_local // n_vars
                            col = idx_local - row * n_vars
                            val = -gamma_dt * jac[i, row, col]
                            if row == col:
                                val += 1.0
                            smem_lu[a_offset + idx_local] = val
                        cuda.syncthreads()
                        lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
                        cuda.syncthreads()
                        lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
                        cuda.syncthreads()
                        for j in range(lane, n_vars, batch_lanes):
                            delta = smem_rhs[b_offset + j]
                            rhs[i, j] = delta
                            u_new = u[i, j] - delta
                            scale = atol + rtol * max(math.fabs(u[i, j]), math.fabs(u_new))
                            ratio = delta / scale
                            delta_norm_acc += ratio * ratio
                            if not math.isfinite(u_new) or not math.isfinite(delta):
                                failed = True
                            u[i, j] = u_new
                        smem_delta[tx] = delta_norm_acc
                        if lane == 0:
                            smem_failed[batch] = 0
                        cuda.syncthreads()
                        if failed:
                            smem_failed[batch] = 1
                        cuda.syncthreads()
                        if lane == 0:
                            for other_lane in range(1, batch_lanes):
                                smem_delta[tx] += smem_delta[batch + other_lane * batches_per_block]
                            delta_norm = math.sqrt(smem_delta[tx] / n_vars)
                            if delta_norm <= 1.0 and not math.isnan(delta_norm):
                                smem_converged[batch] = 1
                            else:
                                smem_converged[batch] = 0
                            if math.isnan(delta_norm):
                                smem_failed[batch] = 1
                        cuda.syncthreads()
                        converged = smem_converged[batch] != 0
                        failed = smem_failed[batch] != 0
                        it += 1
                    if not converged:
                        failed = True
                    cuda.syncthreads()
                    if lane == 0:
                        implicit_ode_fn(u, t_stage, params, tmp, i)
                    cuda.syncthreads()
                    for j in range(lane, n_vars, batch_lanes):
                        stage_y[i, stage, j] = u[i, j]
                        stage_fi[i, stage, j] = tmp[i, j]
                        if not math.isfinite(tmp[i, j]) or not math.isfinite(u[i, j]):
                            failed = True

                cuda.syncthreads()
                if lane == 0:
                    explicit_ode_fn(u, t_stage, params, tmp, i)
                cuda.syncthreads()
                for j in range(lane, n_vars, batch_lanes):
                    stage_fe[i, stage, j] = tmp[i, j]
                    if not math.isfinite(tmp[i, j]):
                        failed = True
                cuda.syncthreads()

            err_norm_acc = 0.0
            for j in range(lane, n_vars, batch_lanes):
                y_new = stage_y[i, 7, j]
                err_est = 0.0
                for stage in range(8):
                    err_est += (
                        dt_use
                        * _b_err(stage)
                        * (stage_fe[i, stage, j] + stage_fi[i, stage, j])
                    )
                scale = atol + rtol * max(math.fabs(y[i, j]), math.fabs(y_new))
                ratio = err_est / scale
                err_norm_acc += ratio * ratio
                if not math.isfinite(y_new) or not math.isfinite(err_est):
                    failed = True
            smem_delta[tx] = err_norm_acc
            if lane == 0:
                smem_failed[batch] = 0
            cuda.syncthreads()
            if failed:
                smem_failed[batch] = 1
            cuda.syncthreads()
            if lane == 0:
                for other_lane in range(1, batch_lanes):
                    smem_delta[tx] += smem_delta[batch + other_lane * batches_per_block]
                err_norm = math.sqrt(smem_delta[tx] / n_vars)
                accept = (
                    err_norm <= 1.0
                    and (not math.isnan(err_norm))
                    and smem_failed[batch] == 0
                )
                if smem_failed[batch] != 0 or math.isnan(err_norm) or err_norm > 1e18:
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
                smem_dt[batch] = dt_use * factor
                smem_converged[batch] = 1 if accept else 0
            cuda.syncthreads()
            accept = smem_converged[batch] != 0
            dt = smem_dt[batch]

            if accept:
                t_new = t + dt_use
                t = t_new
                accepted += 1
                for j in range(lane, n_vars, batch_lanes):
                    y[i, j] = stage_y[i, 7, j]
                cuda.syncthreads()
                if math.fabs(t_new - next_target) <= 1e-12 * max(1.0, math.fabs(next_target)):
                    for j in range(lane, n_vars, batch_lanes):
                        hist[i, save_idx, j] = y[i, j]
                    cuda.syncthreads()
                    save_idx += 1
            else:
                rejected += 1

            n_steps += 1
            cuda.syncthreads()

        if i < y0.shape[0] and lane == 0:
            accepted_out[i] = accepted
            rejected_out[i] = rejected
            loop_out[i] = n_steps

    return kernel, lu_solver


def prepare_solve(
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    y0,
    t_span,
    params,
    *,
    linear: bool = False,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
):
    y0_arr, times, params_arr, dt0 = normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]

    workspace = get_workspace(_WORKSPACE_CACHE, n, n_vars, n_save, n_params)
    workspace.y0_dev.copy_to_device(y0_arr)
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)

    kernel, lu_solver = _make_kernel(
        explicit_ode_fn, implicit_ode_fn, implicit_jac_fn, n_vars
    )
    threads = as_launch_block_dim(lu_solver.block_dim)
    batches_per_block = int(lu_solver.batches_per_block)
    blocks = (n + batches_per_block - 1) // batches_per_block

    return PreparedSolve(
        kernel=kernel,
        lu_solver=lu_solver,
        workspace=workspace,
        linear=np.int32(1 if linear else 0),
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
        prepared.linear,
        prepared.dt0,
        prepared.rtol,
        prepared.atol,
        prepared.max_steps,
        workspace.hist_dev,
        workspace.accepted_dev,
        workspace.rejected_dev,
        workspace.loop_dev,
        workspace.y_dev,
        workspace.u_dev,
        workspace.tmp_dev,
        workspace.base_dev,
        workspace.rhs_dev,
        workspace.stage_y_dev,
        workspace.stage_fe_dev,
        workspace.stage_fi_dev,
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
    return solution, numpy_stats(accepted_steps, rejected_steps, loop_steps)


def solve(
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    y0,
    t_span,
    params,
    *,
    linear: bool = False,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
):
    prepared = prepare_solve(
        explicit_ode_fn,
        implicit_ode_fn,
        implicit_jac_fn,
        y0,
        t_span,
        params,
        linear=linear,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
    return run_prepared(prepared, return_stats=return_stats, copy_solution=True)
