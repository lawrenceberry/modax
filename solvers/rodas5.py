"""Rodas5 solver — nonlinear ODE variant (ode_fn path).

Accepts an ``ode_fn(y, t, params) -> dy/dt`` whose Jacobian is recomputed at
every step via ``jax.jacfwd``.

Uses a per-trajectory ``jax.lax.while_loop``.  The ``batch_size`` parameter is
passed to ``jax.lax.map`` so that JAX vmaps groups of trajectories together.
``batch_size=N`` (default) batches all trajectories together; ``batch_size=1``
maps them one at a time.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp

# fmt: off
# Rodas5 W-transformed coefficients
_gamma = 0.19

_a21 = 2.0
_a31 = 3.040894194418781;  _a32 = 1.041747909077569
_a41 = 2.576417536461461;  _a42 = 1.622083060776640;  _a43 = -0.9089668560264532
_a51 = 2.760842080225597;  _a52 = 1.446624659844071;  _a53 = -0.3036980084553738;  _a54 = 0.2877498600325443
_a61 = -14.09640773051259; _a62 = 6.925207756232704;  _a63 = -41.47510893210728;   _a64 = 2.343771018586405;  _a65 = 24.13215229196062
_a71 = _a61;               _a72 = _a62;               _a73 = _a63;                  _a74 = _a64;              _a75 = _a65;              _a76 = 1.0
_a81 = _a61;               _a82 = _a62;               _a83 = _a63;                  _a84 = _a64;              _a85 = _a65;              _a86 = 1.0;  _a87 = 1.0

_C21 = -10.31323885133993
_C31 = -21.04823117650003; _C32 = -7.234992135176716
_C41 = 32.22751541853323;  _C42 = -4.943732386540191;  _C43 = 19.44922031041879
_C51 = -20.69865579590063; _C52 = -8.816374604402768;  _C53 = 1.260436877740897;   _C54 = -0.7495647613787146
_C61 = -46.22004352711257; _C62 = -17.49534862857472;  _C63 = -289.6389582892057;  _C64 = 93.60855400400906;  _C65 = 318.3822534212147
_C71 = 34.20013733472935;  _C72 = -14.15535402717690;  _C73 = 57.82335640988400;   _C74 = 25.83362985412365;  _C75 = 1.408950972071624;  _C76 = -6.551835421242162
_C81 = 42.57076742291101;  _C82 = -13.80770672017997;  _C83 = 93.98938432427124;   _C84 = 18.77919633714503;  _C85 = -31.58359187223370;  _C86 = -6.685968952921985;  _C87 = -5.810979938412932

# Stage time coefficients for non-autonomous ODEs: t_stage = t + c_i * dt
_c2 = 0.38
_c3 = 0.3878509998321533
_c4 = 0.4839718937873840
_c5 = 0.4570477008819580
# c6 = c7 = c8 = 1.0
# fmt: on


@functools.partial(
    jax.jit,
    static_argnames=(
        "ode_fn",
        "lu_precision",
        "batch_size",
        "max_steps",
        "return_stats",
    ),
)
def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
):
    """Rodas5 ensemble solver for nonlinear ODEs.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``dy/dt = ode_fn(y, t, params)``.
    y0 : array, shape (n_vars,) or (N, n_vars)
        Initial state. A 1-D array is broadcast to all trajectories; a 2-D
        array supplies a distinct starting point for each trajectory.
    t_span : array-like, shape (n_save,)
        Strictly-increasing 1-D array of save times (including t0).
    params : array, shape (n_params,) or (N, n_params)
        Parameters. A 1-D array is broadcast to all trajectories; a 2-D array
        supplies distinct parameters for each trajectory.
    lu_precision :
        Precision for LU factorization and LU solve: ``"fp32"`` or ``"fp64"``.
    batch_size : int or None
        Number of trajectories batched by ``jax.lax.map``. ``None`` (default)
        batches all trajectories together. Internally, ``batch_size`` makes
        ``jax.lax.map`` apply a ``vmap`` over each batch, and JAX hoists the
        per-trajectory ``while_loop`` into one loop spanning all trajectories
        in that batch.
    rtol, atol : float
        Relative and absolute error tolerances.
    first_step : float or None
        Initial step size. Defaults to ``(tf - t0) * 1e-6``.
    max_steps : int
        Maximum number of integration steps per batch.
    return_stats : bool
        If True, return ``(solution, stats)`` where ``stats`` contains raw
        per-lane step counters and per-batch loop diagnostics.

    Returns
    -------
    array, shape (N, n_save, n_vars)
        Solution at each save time for each trajectory. If ``return_stats`` is
        True, returns ``(solution, stats)``.
    """
    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64
    jac_fn = jax.jacfwd(ode_fn, argnums=0)

    y0_in = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)
    times = jnp.asarray(t_span, dtype=jnp.float64)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        N = 1
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (N, n_vars))
        params_arr = jnp.broadcast_to(params_arr, (N, params_arr.shape[0]))
    elif y0_in.ndim == 1:
        N = params_arr.shape[0]
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (N, n_vars))
    else:
        N = y0_in.shape[0]
        n_vars = y0_in.shape[1]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = jnp.broadcast_to(params_arr, (N, params_arr.shape[0]))
        elif params_arr.shape[0] != N:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )
    n_save = times.shape[0]
    tf = times[-1]

    dt0 = jnp.float64(
        first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
    )

    bs = N if batch_size is None else batch_size
    n_chunks = (N + bs - 1) // bs

    eye = jnp.eye(n_vars, dtype=lu_dtype)

    def _solve_one(params_one, y0_one):
        y_init = y0_one.copy()
        hist_init = (
            jnp.zeros((n_save, n_vars), dtype=jnp.float64).at[0, :].set(y_init)
        )
        t_init = times[0]
        dt_init = dt0
        save_idx_init = jnp.int32(1)
        accepted_steps_init = jnp.int32(0)
        rejected_steps_init = jnp.int32(0)

        def _step_one(y, t, dt):
            jac = jac_fn(y, t, params_one).astype(lu_dtype)
            dtgamma_inv = (1.0 / (dt * _gamma)).astype(lu_dtype)
            lu = jax.scipy.linalg.lu_factor(dtgamma_inv * eye - jac)
            inv_dt = 1.0 / dt

            def f_eval(u, t_stage):
                return ode_fn(u, t_stage, params_one)

            def lu_solve(rhs):
                sol = jax.scipy.linalg.lu_solve(lu, rhs.astype(lu_dtype))
                return sol.astype(jnp.float64)

            dy = f_eval(y, t)
            k1 = lu_solve(dy)

            u = y + _a21 * k1
            du = f_eval(u, t + _c2 * dt)
            k2 = lu_solve(du + _C21 * k1 * inv_dt)

            u = y + _a31 * k1 + _a32 * k2
            du = f_eval(u, t + _c3 * dt)
            k3 = lu_solve(du + (_C31 * k1 + _C32 * k2) * inv_dt)

            u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
            du = f_eval(u, t + _c4 * dt)
            k4 = lu_solve(du + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt)

            u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
            du = f_eval(u, t + _c5 * dt)
            k5 = lu_solve(du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt)

            t_end = t + dt
            u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
            du = f_eval(u, t_end)
            k6 = lu_solve(
                du
                + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt
            )

            u = u + k6
            du = f_eval(u, t_end)
            k7 = lu_solve(
                du
                + (
                    _C71 * k1
                    + _C72 * k2
                    + _C73 * k3
                    + _C74 * k4
                    + _C75 * k5
                    + _C76 * k6
                )
                * inv_dt
            )

            u = u + k7
            du = f_eval(u, t_end)
            k8 = lu_solve(
                du
                + (
                    _C81 * k1
                    + _C82 * k2
                    + _C83 * k3
                    + _C84 * k4
                    + _C85 * k5
                    + _C86 * k6
                    + _C87 * k7
                )
                * inv_dt
            )

            return u + k8, k8

        def cond_fn(state):
            t, _, _, _, save_idx, n_steps, _, _ = state
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
            ) = state
            next_target = times[save_idx]
            dt_use = jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30)

            y_new, err_est = _step_one(y, t, dt_use)

            scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
            err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2))

            accept = (err_norm <= 1.0) & ~jnp.isnan(err_norm)
            t_new = jnp.where(accept, t + dt_use, t)
            y_out = jnp.where(accept, y_new, y)

            reached = accept & (
                jnp.abs(t_new - next_target)
                <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
            )
            slot_mask = jax.nn.one_hot(save_idx, n_save, dtype=jnp.bool_) & reached
            hist_new = jnp.where(slot_mask[:, None], y_out[None, :], hist)
            save_idx_new = save_idx + reached.astype(jnp.int32)

            safe_err = jnp.where(
                jnp.isnan(err_norm) | (err_norm > 1e18),
                1e18,
                jnp.where(err_norm == 0.0, 1e-18, err_norm),
            )
            factor = jnp.clip(0.9 * safe_err ** (-1.0 / 6.0), 0.2, 6.0)
            dt_new = dt_use * factor
            rejected = ~accept
            accepted_steps_new = accepted_steps + accept.astype(jnp.int32)
            rejected_steps_new = rejected_steps + rejected.astype(jnp.int32)

            return (
                t_new,
                y_out,
                dt_new,
                hist_new,
                save_idx_new,
                n_steps + 1,
                accepted_steps_new,
                rejected_steps_new,
            )

        init = (
            t_init,
            y_init,
            dt_init,
            hist_init,
            save_idx_init,
            jnp.int32(0),
            accepted_steps_init,
            rejected_steps_init,
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
        batch_size=bs,
    )
    solution = results
    if not return_stats:
        return solution

    accepted_steps = trajectory_stats["accepted_steps"].reshape(N)
    rejected_steps = trajectory_stats["rejected_steps"].reshape(N)
    n_padded = n_chunks * bs
    pad_count = n_padded - N
    loop_steps_padded = jnp.pad(trajectory_stats["loop_steps"], (0, pad_count))
    loop_steps = loop_steps_padded.reshape(n_chunks, bs)
    valid_batches = (jnp.arange(n_padded) < N).reshape(n_chunks, bs)
    batch_loop_iterations = jnp.max(
        jnp.where(valid_batches, loop_steps, jnp.int32(0)), axis=1
    )
    valid_lanes = jnp.sum(valid_batches.astype(jnp.int32), axis=1)
    stats = {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "batch_loop_iterations": batch_loop_iterations,
        "valid_lanes": valid_lanes,
    }
    return solution, stats
