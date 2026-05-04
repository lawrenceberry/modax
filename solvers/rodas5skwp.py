"""Warp-tiled hybrid Rodas5 custom kernel."""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from solvers import _rodas5ck_common as ck

_BLOCK_DIM = 32

wp.set_module_options({"enable_backward": False})


@wp.func
def _tile_block_size(n_vars: int):
    return wp.int32(16 if n_vars >= 16 else n_vars)


@wp.func
def _matrix_base(i: int, padded_n: int):
    return i * padded_n


@wp.func
def _build_shifted_matrix(
    jac: wp.array3d(dtype=wp.float64),
    lu_mat: wp.array2d(dtype=wp.float32),
    i: int,
    n_vars: int,
    padded_n: int,
    dt_use: wp.float64,
):
    base = _matrix_base(i, padded_n)
    dtgamma_inv = wp.float64(1.0) / (dt_use * wp.float64(ck.GAMMA))
    row = wp.int32(0)
    while row < padded_n:
        col = wp.int32(0)
        while col < padded_n:
            value = wp.float32(0.0)
            if row < n_vars and col < n_vars:
                if row == col:
                    value = wp.float32(dtgamma_inv - jac[i, row, col])
                else:
                    value = wp.float32(-jac[i, row, col])
            elif row == col:
                value = wp.float32(1.0)
            lu_mat[base + row, col] = value
            col += wp.int32(1)
        row += wp.int32(1)


@wp.func
def _rank1_update_lu(
    lu_mat: wp.array2d(dtype=wp.float32),
    i: int,
    padded_n: int,
    tile_block: int,
    lane: int,
):
    base = _matrix_base(i, padded_n)
    k = wp.int32(0)
    while k < padded_n:
        if lane == 0:
            piv_k = k
            max_v = wp.abs(lu_mat[base + k, k])
            m = k + wp.int32(1)
            while m < padded_n:
                val = wp.abs(lu_mat[base + m, k])
                if val > max_v:
                    max_v = val
                    piv_k = m
                m += wp.int32(1)
            if piv_k != k:
                col = wp.int32(0)
                while col < padded_n:
                    tmp_lu = lu_mat[base + k, col]
                    lu_mat[base + k, col] = lu_mat[base + piv_k, col]
                    lu_mat[base + piv_k, col] = tmp_lu
                    col += wp.int32(1)

            pivot = lu_mat[base + k, k]
            if wp.abs(pivot) < wp.float32(1.0e-12):
                pivot = wp.float32(1.0e-12)
                lu_mat[base + k, k] = pivot

            m = k + wp.int32(1)
            while m < padded_n:
                lu_mat[base + m, k] = lu_mat[base + m, k] / pivot
                m += wp.int32(1)

        row0 = k + wp.int32(1)
        while row0 < padded_n:
            l_tile = wp.tile_load(
                lu_mat, shape=(tile_block, 1), offset=(base + row0, k)
            )
            col0 = k + wp.int32(1)
            while col0 < padded_n:
                u_tile = wp.tile_load(
                    lu_mat, shape=(1, tile_block), offset=(base + k, col0)
                )
                a_tile = wp.tile_load(
                    lu_mat,
                    shape=(tile_block, tile_block),
                    offset=(base + row0, col0),
                    storage="shared",
                )
                wp.tile_matmul(l_tile, u_tile, a_tile, alpha=-1.0)
                wp.tile_store(lu_mat, a_tile, offset=(base + row0, col0))
                col0 += tile_block
            row0 += tile_block
        k += wp.int32(1)


@wp.func
def _write_rhs_from_stage(
    rhs: wp.array2d(dtype=wp.float32),
    stage: wp.array2d(dtype=wp.float64),
    i: int,
    n_vars: int,
    padded_n: int,
):
    base = _matrix_base(i, padded_n)
    j = wp.int32(0)
    while j < padded_n:
        value = wp.float32(0.0)
        if j < n_vars:
            value = wp.float32(stage[i, j])
        rhs[base + j, 0] = value
        j += wp.int32(1)


@wp.func
def _read_rhs_to_stage(
    rhs: wp.array2d(dtype=wp.float32),
    stage: wp.array2d(dtype=wp.float64),
    i: int,
    n_vars: int,
    padded_n: int,
):
    base = _matrix_base(i, padded_n)
    j = wp.int32(0)
    while j < n_vars:
        stage[i, j] = wp.float64(rhs[base + j, 0])
        j += wp.int32(1)


@wp.func
def _solve_rhs_tiled(
    lu_mat: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
    piv_arr: wp.array2d(dtype=wp.int32),
    i: int,
    padded_n: int,
    tile_block: int,
    lane: int,
):
    base = _matrix_base(i, padded_n)

    if lane == 0:
        k = wp.int32(0)
        while k < padded_n:
            piv_k = piv_arr[i, k]
            if piv_k != k:
                tmp_rhs = rhs[base + k, 0]
                rhs[base + k, 0] = rhs[base + piv_k, 0]
                rhs[base + piv_k, 0] = tmp_rhs
            k += wp.int32(1)

    i0 = wp.int32(0)
    while i0 < padded_n:
        rhs_tile = wp.tile_load(
            rhs, shape=(tile_block, 1), offset=(base + i0, 0), storage="shared"
        )
        j0 = wp.int32(0)
        while j0 < i0:
            l_tile = wp.tile_load(lu_mat, shape=(tile_block, tile_block), offset=(base + i0, j0))
            y_tile = wp.tile_load(rhs, shape=(tile_block, 1), offset=(base + j0, 0))
            wp.tile_matmul(l_tile, y_tile, rhs_tile, alpha=-1.0)
            j0 += tile_block
        wp.tile_store(rhs, rhs_tile, offset=(base + i0, 0))
        if lane == 0:
            r = wp.int32(0)
            while r < tile_block:
                val = rhs[base + i0 + r, 0]
                c = wp.int32(0)
                while c < r:
                    val -= lu_mat[base + i0 + r, i0 + c] * rhs[base + i0 + c, 0]
                    c += wp.int32(1)
                rhs[base + i0 + r, 0] = val
                r += wp.int32(1)
        i0 += tile_block

    i0 = padded_n - tile_block
    while i0 >= 0:
        rhs_tile = wp.tile_load(
            rhs, shape=(tile_block, 1), offset=(base + i0, 0), storage="shared"
        )
        j0 = i0 + tile_block
        while j0 < padded_n:
            u_tile = wp.tile_load(lu_mat, shape=(tile_block, tile_block), offset=(base + i0, j0))
            x_tile = wp.tile_load(rhs, shape=(tile_block, 1), offset=(base + j0, 0))
            wp.tile_matmul(u_tile, x_tile, rhs_tile, alpha=-1.0)
            j0 += tile_block
        wp.tile_store(rhs, rhs_tile, offset=(base + i0, 0))
        if lane == 0:
            r = tile_block - wp.int32(1)
            while r >= 0:
                val = rhs[base + i0 + r, 0]
                c = r + wp.int32(1)
                while c < tile_block:
                    val -= lu_mat[base + i0 + r, i0 + c] * rhs[base + i0 + c, 0]
                    c += wp.int32(1)
                rhs[base + i0 + r, 0] = val / lu_mat[base + i0 + r, i0 + r]
                r -= wp.int32(1)
        i0 -= tile_block


@functools.cache
def _make_kernel(ode_fn, jac_fn, n_vars: int):
    tile_block = 16 if n_vars >= 16 else n_vars
    padded_n = ((n_vars + tile_block - 1) // tile_block) * tile_block

    @wp.func
    def rank1_update_lu_local(
        lu_mat: wp.array2d(dtype=wp.float32),
        piv_arr: wp.array2d(dtype=wp.int32),
        i: int,
        lane: int,
    ):
        base = _matrix_base(i, padded_n)
        k = wp.int32(0)
        while k < padded_n:
            if lane == 0:
                piv_k = k
                max_v = wp.abs(lu_mat[base + k, k])
                m = k + wp.int32(1)
                while m < padded_n:
                    val = wp.abs(lu_mat[base + m, k])
                    if val > max_v:
                        max_v = val
                        piv_k = m
                    m += wp.int32(1)
                piv_arr[i, k] = piv_k
                if piv_k != k:
                    col = wp.int32(0)
                    while col < padded_n:
                        tmp_lu = lu_mat[base + k, col]
                        lu_mat[base + k, col] = lu_mat[base + piv_k, col]
                        lu_mat[base + piv_k, col] = tmp_lu
                        col += wp.int32(1)

                pivot = lu_mat[base + k, k]
                if wp.abs(pivot) < wp.float32(1.0e-12):
                    pivot = wp.float32(1.0e-12)
                    lu_mat[base + k, k] = pivot

                m = k + wp.int32(1)
                while m < padded_n:
                    lu_mat[base + m, k] = lu_mat[base + m, k] / pivot
                    m += wp.int32(1)

            row0 = k + wp.int32(1)
            while row0 < padded_n:
                l_tile = wp.tile_load(lu_mat, shape=(tile_block, 1), offset=(base + row0, k))
                col0 = k + wp.int32(1)
                while col0 < padded_n:
                    u_tile = wp.tile_load(lu_mat, shape=(1, tile_block), offset=(base + k, col0))
                    a_tile = wp.tile_load(
                        lu_mat,
                        shape=(tile_block, tile_block),
                        offset=(base + row0, col0),
                        storage="shared",
                    )
                    wp.tile_matmul(l_tile, u_tile, a_tile, alpha=-1.0)
                    wp.tile_store(lu_mat, a_tile, offset=(base + row0, col0))
                    col0 += tile_block
                row0 += tile_block
            k += wp.int32(1)

    @wp.func
    def solve_rhs_tiled_local(
        lu_mat: wp.array2d(dtype=wp.float32),
        rhs: wp.array2d(dtype=wp.float32),
        piv_arr: wp.array2d(dtype=wp.int32),
        i: int,
        lane: int,
    ):
        base = _matrix_base(i, padded_n)
        if lane == 0:
            k = wp.int32(0)
            while k < padded_n:
                piv_k = piv_arr[i, k]
                if piv_k != k:
                    tmp_rhs = rhs[base + k, 0]
                    rhs[base + k, 0] = rhs[base + piv_k, 0]
                    rhs[base + piv_k, 0] = tmp_rhs
                k += wp.int32(1)

        i0 = wp.int32(0)
        while i0 < padded_n:
            rhs_tile = wp.tile_load(
                rhs, shape=(tile_block, 1), offset=(base + i0, 0), storage="shared"
            )
            j0 = wp.int32(0)
            while j0 < i0:
                l_tile = wp.tile_load(lu_mat, shape=(tile_block, tile_block), offset=(base + i0, j0))
                y_tile = wp.tile_load(rhs, shape=(tile_block, 1), offset=(base + j0, 0))
                wp.tile_matmul(l_tile, y_tile, rhs_tile, alpha=-1.0)
                j0 += tile_block
            wp.tile_store(rhs, rhs_tile, offset=(base + i0, 0))
            if lane == 0:
                r = wp.int32(0)
                while r < tile_block:
                    val = rhs[base + i0 + r, 0]
                    c = wp.int32(0)
                    while c < r:
                        val -= lu_mat[base + i0 + r, i0 + c] * rhs[base + i0 + c, 0]
                        c += wp.int32(1)
                    rhs[base + i0 + r, 0] = val
                    r += wp.int32(1)
            i0 += tile_block

        i0 = padded_n - tile_block
        while i0 >= 0:
            rhs_tile = wp.tile_load(
                rhs, shape=(tile_block, 1), offset=(base + i0, 0), storage="shared"
            )
            j0 = i0 + tile_block
            while j0 < padded_n:
                u_tile = wp.tile_load(lu_mat, shape=(tile_block, tile_block), offset=(base + i0, j0))
                x_tile = wp.tile_load(rhs, shape=(tile_block, 1), offset=(base + j0, 0))
                wp.tile_matmul(u_tile, x_tile, rhs_tile, alpha=-1.0)
                j0 += tile_block
            wp.tile_store(rhs, rhs_tile, offset=(base + i0, 0))
            if lane == 0:
                r = tile_block - wp.int32(1)
                while r >= 0:
                    val = rhs[base + i0 + r, 0]
                    c = r + wp.int32(1)
                    while c < tile_block:
                        val -= lu_mat[base + i0 + r, i0 + c] * rhs[base + i0 + c, 0]
                        c += wp.int32(1)
                    rhs[base + i0 + r, 0] = val / lu_mat[base + i0 + r, i0 + r]
                    r -= wp.int32(1)
            i0 -= tile_block

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
        t_state: wp.array(dtype=wp.float64),
        dt_state: wp.array(dtype=wp.float64),
        save_idx_state: wp.array(dtype=wp.int32),
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
        lu_mat: wp.array2d(dtype=wp.float32),
        rhs: wp.array2d(dtype=wp.float32),
        piv_arr: wp.array2d(dtype=wp.int32),
    ):
        block_idx, lane = wp.tid()
        i = block_idx * wp.int32(_BLOCK_DIM) + lane

        n = y0.shape[0]
        n_save = times.shape[0]
        tf = times[n_save - 1]

        if i < n:
            j = wp.int32(0)
            while j < n_vars:
                y[i, j] = y0[i, j]
                hist[i, 0, j] = y0[i, j]
                j += wp.int32(1)
            t_state[i] = times[0]
            dt_state[i] = dt0
            save_idx_state[i] = wp.int32(1)
            accepted_out[i] = wp.int32(0)
            rejected_out[i] = wp.int32(0)
            loop_out[i] = wp.int32(0)

        while True:
            keep_going = wp.int32(0)
            owner = wp.int32(0)
            while owner < _BLOCK_DIM:
                io = block_idx * wp.int32(_BLOCK_DIM) + owner
                if (
                    io < n
                    and save_idx_state[io] < n_save
                    and t_state[io] < tf
                    and loop_out[io] < max_steps
                ):
                    keep_going = wp.int32(1)
                owner += wp.int32(1)
            if keep_going == 0:
                break

            owner = wp.int32(0)
            while owner < _BLOCK_DIM:
                io = block_idx * wp.int32(_BLOCK_DIM) + owner
                active = (
                    io < n
                    and save_idx_state[io] < n_save
                    and t_state[io] < tf
                    and loop_out[io] < max_steps
                )

                dt_use = wp.float64(0.0)
                inv_dt = wp.float64(0.0)
                next_target = wp.float64(0.0)
                t_end = wp.float64(0.0)

                if lane == owner and active:
                    next_target = times[save_idx_state[io]]
                    dt_use = dt_state[io]
                    if dt_use > next_target - t_state[io]:
                        dt_use = next_target - t_state[io]
                    if dt_use < wp.float64(1.0e-30):
                        dt_use = wp.float64(1.0e-30)
                    inv_dt = wp.float64(1.0) / dt_use
                    t_end = t_state[io] + dt_use

                    jac_fn(y, t_state[io], params, jac, io)
                    _build_shifted_matrix(jac, lu_mat, io, n_vars, padded_n, dt_use)

                if active:
                    rank1_update_lu_local(lu_mat, piv_arr, io, lane)

                if lane == owner and active:
                    ode_fn(y, t_state[io], params, k1, io)
                    _write_rhs_from_stage(rhs, k1, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k1, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] = y[io, j] + wp.float64(ck.A21) * k1[io, j]
                        j += wp.int32(1)
                    ode_fn(u, t_state[io] + wp.float64(ck.C2) * dt_use, params, k2, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k2[io, j] += wp.float64(ck.C21) * k1[io, j] * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k2, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k2, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] = y[io, j] + (
                            wp.float64(ck.A31) * k1[io, j]
                            + wp.float64(ck.A32) * k2[io, j]
                        )
                        j += wp.int32(1)
                    ode_fn(u, t_state[io] + wp.float64(ck.C3) * dt_use, params, k3, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k3[io, j] += (
                            wp.float64(ck.C31) * k1[io, j]
                            + wp.float64(ck.C32) * k2[io, j]
                        ) * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k3, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k3, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] = y[io, j] + (
                            wp.float64(ck.A41) * k1[io, j]
                            + wp.float64(ck.A42) * k2[io, j]
                            + wp.float64(ck.A43) * k3[io, j]
                        )
                        j += wp.int32(1)
                    ode_fn(u, t_state[io] + wp.float64(ck.C4) * dt_use, params, k4, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k4[io, j] += (
                            wp.float64(ck.C41) * k1[io, j]
                            + wp.float64(ck.C42) * k2[io, j]
                            + wp.float64(ck.C43) * k3[io, j]
                        ) * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k4, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k4, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] = y[io, j] + (
                            wp.float64(ck.A51) * k1[io, j]
                            + wp.float64(ck.A52) * k2[io, j]
                            + wp.float64(ck.A53) * k3[io, j]
                            + wp.float64(ck.A54) * k4[io, j]
                        )
                        j += wp.int32(1)
                    ode_fn(u, t_state[io] + wp.float64(ck.C5) * dt_use, params, k5, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k5[io, j] += (
                            wp.float64(ck.C51) * k1[io, j]
                            + wp.float64(ck.C52) * k2[io, j]
                            + wp.float64(ck.C53) * k3[io, j]
                            + wp.float64(ck.C54) * k4[io, j]
                        ) * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k5, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k5, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] = y[io, j] + (
                            wp.float64(ck.A61) * k1[io, j]
                            + wp.float64(ck.A62) * k2[io, j]
                            + wp.float64(ck.A63) * k3[io, j]
                            + wp.float64(ck.A64) * k4[io, j]
                            + wp.float64(ck.A65) * k5[io, j]
                        )
                        j += wp.int32(1)
                    ode_fn(u, t_end, params, k6, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k6[io, j] += (
                            wp.float64(ck.C61) * k1[io, j]
                            + wp.float64(ck.C62) * k2[io, j]
                            + wp.float64(ck.C63) * k3[io, j]
                            + wp.float64(ck.C64) * k4[io, j]
                            + wp.float64(ck.C65) * k5[io, j]
                        ) * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k6, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k6, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] += k6[io, j]
                        j += wp.int32(1)

                    ode_fn(u, t_end, params, k7, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k7[io, j] += (
                            wp.float64(ck.C71) * k1[io, j]
                            + wp.float64(ck.C72) * k2[io, j]
                            + wp.float64(ck.C73) * k3[io, j]
                            + wp.float64(ck.C74) * k4[io, j]
                            + wp.float64(ck.C75) * k5[io, j]
                            + wp.float64(ck.C76) * k6[io, j]
                        ) * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k7, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k7, io, n_vars, padded_n)

                    j = wp.int32(0)
                    while j < n_vars:
                        u[io, j] += k7[io, j]
                        j += wp.int32(1)

                    ode_fn(u, t_end, params, k8, io)
                    j = wp.int32(0)
                    while j < n_vars:
                        k8[io, j] += (
                            wp.float64(ck.C81) * k1[io, j]
                            + wp.float64(ck.C82) * k2[io, j]
                            + wp.float64(ck.C83) * k3[io, j]
                            + wp.float64(ck.C84) * k4[io, j]
                            + wp.float64(ck.C85) * k5[io, j]
                            + wp.float64(ck.C86) * k6[io, j]
                            + wp.float64(ck.C87) * k7[io, j]
                        ) * inv_dt
                        j += wp.int32(1)
                    _write_rhs_from_stage(rhs, k8, io, n_vars, padded_n)
                if active:
                    solve_rhs_tiled_local(lu_mat, rhs, piv_arr, io, lane)
                if lane == owner and active:
                    _read_rhs_to_stage(rhs, k8, io, n_vars, padded_n)

                    err_sum = wp.float64(0.0)
                    j = wp.int32(0)
                    while j < n_vars:
                        y_new_j = u[io, j] + k8[io, j]
                        scale = atol + rtol * wp.max(wp.abs(y[io, j]), wp.abs(y_new_j))
                        ratio = k8[io, j] / scale
                        err_sum += ratio * ratio
                        j += wp.int32(1)
                    err_norm = wp.sqrt(err_sum / wp.float64(n_vars))
                    accept = err_norm <= wp.float64(1.0) and not wp.isnan(err_norm)

                    t_new = t_state[io]
                    if accept:
                        t_new = t_state[io] + dt_use
                        j = wp.int32(0)
                        while j < n_vars:
                            y[io, j] = u[io, j] + k8[io, j]
                            j += wp.int32(1)
                        accepted_out[io] += wp.int32(1)
                    else:
                        rejected_out[io] += wp.int32(1)

                    reached = accept and (
                        wp.abs(t_new - next_target)
                        <= wp.float64(1.0e-12)
                        * wp.max(wp.float64(1.0), wp.abs(next_target))
                    )
                    if reached:
                        j = wp.int32(0)
                        while j < n_vars:
                            hist[io, save_idx_state[io], j] = y[io, j]
                            j += wp.int32(1)
                        save_idx_state[io] += wp.int32(1)

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
                    dt_state[io] = dt_use * factor
                    t_state[io] = t_new
                    loop_out[io] += wp.int32(1)

                owner += wp.int32(1)

    return kernel, padded_n


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

    kernel, padded_n = _make_kernel(ode_fn, jac_fn, n_vars)
    n_blocks = (n + _BLOCK_DIM - 1) // _BLOCK_DIM

    y0_dev = wp.array(y0_arr, dtype=wp.float64, device=device)
    times_dev = wp.array(times, dtype=wp.float64, device=device)
    params_dev = wp.array(params_arr, dtype=wp.float64, device=device)
    hist_dev = wp.empty((n, times.shape[0], n_vars), dtype=wp.float64, device=device)
    accepted_dev = wp.empty(n, dtype=wp.int32, device=device)
    rejected_dev = wp.empty(n, dtype=wp.int32, device=device)
    loop_dev = wp.empty(n, dtype=wp.int32, device=device)
    t_state_dev = wp.empty(n, dtype=wp.float64, device=device)
    dt_state_dev = wp.empty(n, dtype=wp.float64, device=device)
    save_idx_state_dev = wp.empty(n, dtype=wp.int32, device=device)

    flat_work = [
        wp.empty((n, n_vars), dtype=wp.float64, device=device) for _ in range(10)
    ]
    jac_dev = wp.empty((n, n_vars, n_vars), dtype=wp.float64, device=device)
    lu_dev = wp.empty((n * padded_n, padded_n), dtype=wp.float32, device=device)
    rhs_dev = wp.empty((n * padded_n, 1), dtype=wp.float32, device=device)
    piv_dev = wp.empty((n, padded_n), dtype=wp.int32, device=device)

    wp.launch_tiled(
        kernel,
        dim=[n_blocks],
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
            t_state_dev,
            dt_state_dev,
            save_idx_state_dev,
            *flat_work,
            jac_dev,
            lu_dev,
            rhs_dev,
            piv_dev,
        ],
        block_dim=_BLOCK_DIM,
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
