"""Tsit5 solver — nonlinear ODE variant (ode_fn path).

Accepts an ``ode_fn(y, t, params) -> dy/dt`` and applies the Tsitouras 5/4
explicit Runge-Kutta method with adaptive step sizing.

Uses a per-trajectory ``jax.lax.while_loop``.  The ``batch_size`` parameter is
passed to ``jax.lax.map`` so that JAX vmaps groups of trajectories together.
``batch_size=N`` (default) batches all trajectories together; ``batch_size=1``
maps them one at a time.
"""

import functools

import jax
import jax.numpy as jnp

# fmt: off
_C2 = 161.0 / 1000.0
_C3 = 327.0 / 1000.0
_C4 = 9.0 / 10.0
_C5 = 0.9800255409045097
_C6 = 1.0
_C7 = 1.0

_A21 = 161.0 / 1000.0

_A31 = -0.008480655492356989
_A32 = 0.335480655492357

_A41 = 2.8971530571054935
_A42 = -6.359448489975075
_A43 = 4.3622954328695815

_A51 = 5.325864828439257
_A52 = -11.748883564062828
_A53 = 7.4955393428898365
_A54 = -0.09249506636175525

_A61 = 5.86145544294642
_A62 = -12.92096931784711
_A63 = 8.159367898576159
_A64 = -0.071584973281401
_A65 = -0.028269050394068383

_A71 = 0.09646076681806523
_A72 = 0.01
_A73 = 0.4798896504144996
_A74 = 1.379008574103742
_A75 = -3.2900695154360807
_A76 = 2.324710524099774

_B1 = _A71
_B2 = _A72
_B3 = _A73
_B4 = _A74
_B5 = _A75
_B6 = _A76
_B7 = 0.0

_E1 = 0.0017800620525794302
_E2 = 0.000816434459656747
_E3 = -0.007880878010261985
_E4 = 0.14471100717326298
_E5 = -0.5823571654525553
_E6 = 0.45808210592918695
_E7 = -1.0 / 66.0
# fmt: on

_SAFETY = 0.9
_FACTOR_MIN = 0.2
_FACTOR_MAX = 10.0


@functools.partial(
    jax.jit,
    static_argnames=("ode_fn", "batch_size", "max_steps", "return_stats"),
)
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
    """Tsit5 ensemble solver for nonlinear ODEs.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
    y0 : array, shape (n_vars,) or (N, n_vars)
        Initial state. A 1-D array is broadcast to all trajectories; a 2-D
        array supplies a distinct starting point for each trajectory.
    t_span : array-like, shape (n_save,)
        Strictly-increasing 1-D array of save times (including t0).
    params : array, shape (n_params,) or (N, n_params)
        Parameters. A 1-D array is broadcast to all trajectories; a 2-D array
        supplies distinct parameters for each trajectory.
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
        If True, return ``(solution, stats)`` where ``stats`` contains
        step-count and lane-utilization diagnostics.

    Returns
    -------
    array, shape (N, n_save, n_vars)
        Solution at each save time for each trajectory. If ``return_stats`` is
        True, returns ``(solution, stats)``.
    """
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

    def _solve_one(params_one, y0_one):
        y_init = y0_one.copy()
        hist_init = (
            jnp.zeros((n_save, n_vars), dtype=jnp.float64).at[0, :].set(y_init)
        )
        t_init = times[0]
        dt_init = dt0
        save_idx_init = jnp.int32(1)
        k_fsal_init = jnp.zeros((n_vars,), dtype=jnp.float64)
        has_fsal_init = jnp.bool_(False)
        accepted_steps_init = jnp.int32(0)
        rejected_steps_init = jnp.int32(0)

        def _fresh_k1(y, t, k_fsal, has_fsal):
            return jax.lax.cond(
                has_fsal,
                lambda _: k_fsal,
                lambda _: ode_fn(y, t, params_one),
                operand=None,
            )

        def _step_one(y, t, dt, k_fsal, has_fsal):
            k1 = _fresh_k1(y, t, k_fsal, has_fsal)

            u = y + dt * (_A21 * k1)
            k2 = ode_fn(u, t + _C2 * dt, params_one)

            u = y + dt * (_A31 * k1 + _A32 * k2)
            k3 = ode_fn(u, t + _C3 * dt, params_one)

            u = y + dt * (_A41 * k1 + _A42 * k2 + _A43 * k3)
            k4 = ode_fn(u, t + _C4 * dt, params_one)

            u = y + dt * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4)
            k5 = ode_fn(u, t + _C5 * dt, params_one)

            u = y + dt * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5)
            k6 = ode_fn(u, t + _C6 * dt, params_one)

            y_new = y + dt * (
                _B1 * k1 + _B2 * k2 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6
            )
            k7 = ode_fn(y_new, t + _C7 * dt, params_one)

            err_est = dt * (
                _E1 * k1
                + _E2 * k2
                + _E3 * k3
                + _E4 * k4
                + _E5 * k5
                + _E6 * k6
                + _E7 * k7
            )
            return y_new, err_est, k7

        def cond_fn(state):
            t, _, _, _, save_idx, n_steps, _, _, _, _ = state
            return (save_idx < n_save) & (t < tf) & (n_steps < max_steps)

        def body_fn(state):
            (
                t,
                y,
                dt,
                hist,
                save_idx,
                n_steps,
                k_fsal,
                has_fsal,
                accepted_steps,
                rejected_steps,
            ) = state
            next_target = times[save_idx]
            dt_use = jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30)

            y_new, err_est, k7 = _step_one(y, t, dt_use, k_fsal, has_fsal)

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
            factor = jnp.clip(
                _SAFETY * safe_err ** (-1.0 / 5.0), _FACTOR_MIN, _FACTOR_MAX
            )
            dt_new = dt_use * factor

            k_fsal_new = jnp.where(accept, k7, jnp.zeros_like(k_fsal))
            has_fsal_new = accept
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
                k_fsal_new,
                has_fsal_new,
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
            k_fsal_init,
            has_fsal_init,
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
            _,
            _,
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
