"""Tsit5 solver — nonlinear ODE variant (ode_fn path).

Accepts an ``ode_fn(y, t, params) -> dy/dt`` and applies the Tsitouras 5/4
explicit Runge-Kutta method with adaptive step sizing.

Uses a single jax.lax.while_loop with the batch dimension inside the loop
body instead of vmap-over-while-loop.

The batch_size parameter controls how many trajectories share a while loop.
batch_size=N (default) puts all trajectories in one loop; batch_size=1
recovers the vmap-over-while-loop behaviour.
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
    static_argnames=("ode_fn", "batch_size", "max_steps"),
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
):
    """Tsit5 ensemble solver for nonlinear ODEs.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
    y0 : array, shape (n_vars,)
        Initial state shared by all trajectories.
    t_span : array-like, shape (n_save,)
        Strictly-increasing 1-D array of save times (including t0).
    params : array, shape (N, ...)
        Per-trajectory parameters.
    batch_size : int or None
        Number of trajectories per while-loop batch. ``None`` (default)
        puts all trajectories in a single loop.
    rtol, atol : float
        Relative and absolute error tolerances.
    first_step : float or None
        Initial step size. Defaults to ``(tf - t0) * 1e-6``.
    max_steps : int
        Maximum number of integration steps per batch.

    Returns
    -------
    array, shape (N, n_save, n_vars)
        Solution at each save time for each trajectory.
    """
    ode_batched = jax.vmap(ode_fn)

    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)
    times = jnp.asarray(t_span, dtype=jnp.float64)

    n_vars = y0_arr.shape[0]
    N = params_arr.shape[0]
    n_save = times.shape[0]
    tf = times[-1]

    dt0 = jnp.float64(
        first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
    )

    bs = N if batch_size is None else batch_size
    n_chunks = (N + bs - 1) // bs
    n_padded = n_chunks * bs

    if n_padded > N:
        pad_rows = jnp.broadcast_to(
            params_arr[-1:],
            (n_padded - N,) + params_arr.shape[1:],
        )
        params_padded = jnp.concatenate([params_arr, pad_rows], axis=0)
    else:
        params_padded = params_arr

    params_batches = params_padded.reshape((n_chunks, bs) + params_arr.shape[1:])

    def _solve_batch(params_batch):
        y_init = jnp.broadcast_to(y0_arr, (bs, n_vars)).copy()
        hist_init = (
            jnp.zeros((bs, n_save, n_vars), dtype=jnp.float64).at[:, 0, :].set(y_init)
        )
        t_init = jnp.full((bs,), times[0], dtype=jnp.float64)
        dt_init = jnp.full((bs,), dt0, dtype=jnp.float64)
        save_idx_init = jnp.ones((bs,), dtype=jnp.int32)
        k_fsal_init = jnp.zeros((bs, n_vars), dtype=jnp.float64)
        has_fsal_init = jnp.zeros((bs,), dtype=jnp.bool_)

        def _fresh_k1(y, t, params_batch, k_fsal, has_fsal):
            def _pick_one(y_i, t_i, p_i, k_fsal_i, has_fsal_i):
                return jax.lax.cond(
                    has_fsal_i,
                    lambda _: k_fsal_i,
                    lambda _: ode_fn(y_i, t_i, p_i),
                    operand=None,
                )

            return jax.vmap(_pick_one)(y, t, params_batch, k_fsal, has_fsal)

        def _step_batch(y, t, dt, k_fsal, has_fsal):
            dt_col = dt[:, None]

            def f_eval(u, t_stage):
                return ode_batched(u, t_stage, params_batch)

            k1 = _fresh_k1(y, t, params_batch, k_fsal, has_fsal)

            u = y + dt_col * (_A21 * k1)
            k2 = f_eval(u, t + _C2 * dt)

            u = y + dt_col * (_A31 * k1 + _A32 * k2)
            k3 = f_eval(u, t + _C3 * dt)

            u = y + dt_col * (_A41 * k1 + _A42 * k2 + _A43 * k3)
            k4 = f_eval(u, t + _C4 * dt)

            u = y + dt_col * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4)
            k5 = f_eval(u, t + _C5 * dt)

            u = y + dt_col * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5)
            k6 = f_eval(u, t + _C6 * dt)

            y_new = y + dt_col * (
                _B1 * k1 + _B2 * k2 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6
            )
            k7 = f_eval(y_new, t + _C7 * dt)

            err_est = dt_col * (
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
            t, _, _, _, save_idx, n_steps, _, _ = state
            active = save_idx < n_save
            return (jnp.min(jnp.where(active, t, tf)) < tf) & (n_steps < max_steps)

        def body_fn(state):
            t, y, dt, hist, save_idx, n_steps, k_fsal, has_fsal = state
            active = save_idx < n_save
            next_target = times[save_idx]
            dt_use = jnp.where(
                active,
                jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30),
                1e-30,
            )

            y_new, err_est, k7 = _step_batch(y, t, dt_use, k_fsal, has_fsal)

            scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
            err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2, axis=1))

            accept = active & (err_norm <= 1.0) & ~jnp.isnan(err_norm)
            t_new = jnp.where(accept, t + dt_use, t)
            y_out = jnp.where(accept[:, None], y_new, y)

            reached = accept & (
                jnp.abs(t_new - next_target)
                <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
            )
            slot_mask = (
                jax.nn.one_hot(save_idx, n_save, dtype=jnp.bool_) & reached[:, None]
            )
            hist_new = jnp.where(slot_mask[:, :, None], y_out[:, None, :], hist)
            save_idx_new = save_idx + reached.astype(jnp.int32)

            safe_err = jnp.where(
                jnp.isnan(err_norm) | (err_norm > 1e18),
                1e18,
                jnp.where(err_norm == 0.0, 1e-18, err_norm),
            )
            factor = jnp.clip(
                _SAFETY * safe_err ** (-1.0 / 5.0), _FACTOR_MIN, _FACTOR_MAX
            )
            dt_new = jnp.where(active, dt_use * factor, dt)

            k_fsal_new = jnp.where(accept[:, None], k7, jnp.zeros_like(k_fsal))
            has_fsal_new = accept

            return (
                t_new,
                y_out,
                dt_new,
                hist_new,
                save_idx_new,
                n_steps + 1,
                k_fsal_new,
                has_fsal_new,
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
        )
        _, _, _, hist_final, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return hist_final

    results = jax.vmap(_solve_batch)(params_batches)
    return results.reshape(n_padded, n_save, n_vars)[:N]
