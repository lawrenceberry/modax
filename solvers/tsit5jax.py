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

from solvers._jax_common import (
    build_error_weights,
    eval_ode_fn,
    make_custom_vmap_solver,
    normalize_inputs,
    solve_adaptive_ensemble,
)

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

_E1 = 0.0017800110522257773
_E2 = 0.0008164344596567463
_E3 = -0.007880878010261994
_E4 = 0.1447110071732629
_E5 = -0.5823571654525552
_E6 = 0.45808210592918686
_E7 = -1.0 / 66.0
# fmt: on

_SAFETY = 0.9
_FACTOR_MIN = 0.2
_FACTOR_MAX = 10.0

_A = jnp.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [_A21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [_A31, _A32, 0.0, 0.0, 0.0, 0.0, 0.0],
        [_A41, _A42, _A43, 0.0, 0.0, 0.0, 0.0],
        [_A51, _A52, _A53, _A54, 0.0, 0.0, 0.0],
        [_A61, _A62, _A63, _A64, _A65, 0.0, 0.0],
    ],
    dtype=jnp.float64,
)
_C = jnp.array([0.0, _C2, _C3, _C4, _C5, _C6], dtype=jnp.float64)
_B = jnp.array([_B1, _B2, _B3, _B4, _B5, _B6, _B7], dtype=jnp.float64)
_E = jnp.array([_E1, _E2, _E3, _E4, _E5, _E6, _E7], dtype=jnp.float64)


def _dense_coeffs(theta):
    b1 = (
        -1.0530884977290216
        * theta
        * (theta - 1.3299890189751412)
        * (theta**2 - 1.4364028541716351 * theta + 0.7139816917074209)
    )
    b2 = (
        0.1017 * theta**2 * (theta**2 - 2.1966568338249754 * theta + 1.2949852507374631)
    )
    b3 = (
        2.490627285651252793
        * theta**2
        * (theta**2 - 2.38535645472061657 * theta + 1.57803468208092486)
    )
    b4 = (
        -16.54810288924490272
        * (theta - 1.21712927295533244)
        * (theta - 0.61620406037800089)
        * theta**2
    )
    b5 = (
        47.37952196281928122
        * (theta - 1.203071208372362603)
        * (theta - 0.658047292653547382)
        * theta**2
    )
    b6 = (
        -34.87065786149660974
        * (theta - 1.2)
        * (theta - 0.666666666666666667)
        * theta**2
    )
    b7 = 2.5 * (theta - 1) * (theta - 0.6) * theta**2
    return jnp.stack([b1, b2, b3, b4, b5, b6, b7], axis=-1)


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
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    """Tsit5 ensemble solver for nonlinear ODEs."""

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _solve_impl(
            ode_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            batch_size=batch_size,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            return_stats=return_stats,
            error_weights=error_weights,
            pcoeff=pcoeff,
            icoeff=icoeff,
            dcoeff=dcoeff,
        )

    return make_custom_vmap_solver(solve_impl, return_stats=return_stats)(
        y0, t_span, params
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "ode_fn",
        "batch_size",
        "max_steps",
        "return_stats",
        "pcoeff",
        "icoeff",
        "dcoeff",
    ),
)
def _solve_impl(
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
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    """Tsit5 ensemble solver implementation.

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
    error_weights : array or None
        Per-component weights for the step-size error norm, shape ``(n_vars,)``
        or ``(N, n_vars)``. ``None`` (default) weights every component equally;
        a weight of 0 excludes that component from step-size control. Used in
        the weighted RMS step-size error norm.
    pcoeff, icoeff, dcoeff : float
        Proportional/integral/derivative gains of the PID step-size controller.
        The default ``(0, 1, 0)`` is the classic I-controller (current
        behavior). ``(0.4, 0.3, 0)`` is a typical PI controller that smooths the
        step sequence on moderately stiff problems; set ``dcoeff`` for a full
        PID. See Soderlind's digital-filter step control.

    Returns
    -------
    array, shape (N, n_save, n_vars)
        Solution at each save time for each trajectory. If ``return_stats`` is
        True, returns ``(solution, stats)``.
    """
    y0_arr, times, params_arr, n, n_vars, _, dt0, bs, n_chunks = normalize_inputs(
        y0, t_span, params, first_step, batch_size
    )
    error_weights_arr = (
        None if error_weights is None else build_error_weights(error_weights, n, n_vars)
    )

    def step_factory(params_one):
        # FSAL (First Same As Last): Tsit5 has c7 = 1, so an accepted step's
        # final stage k7 = f(t+dt, y_new) is exactly the next step's first stage
        # k1 = f(t+dt, y_new); a rejected step retries from the same (t, y) so
        # its k1 = f(t, y) is unchanged too. We therefore carry the next k1 in
        # the loop state and never recompute it inside the loop. A per-step
        # ``lax.cond`` would be defeated here because, under the ensemble vmap,
        # cond lowers to a select that executes both branches -- evaluating the
        # RHS anyway. Carrying k1 unconditionally avoids that wasted evaluation,
        # cutting the explicit RHS evaluations per step from 7 to 6.
        def extra_init(y0_one, t0):
            return eval_ode_fn(ode_fn, y0_one, t0, params_one)

        def _step_one(y, t, dt, extra):
            k1 = extra  # carried first stage: FSAL k7 (accept) or reused k1 (reject)
            k = jnp.zeros((7, y.shape[0]), dtype=y.dtype).at[0].set(dt * k1)

            def stage_body(i, k_acc):
                u = y + _A[i] @ k_acc
                ki = dt * eval_ode_fn(ode_fn, u, t + _C[i] * dt, params_one)
                return k_acc.at[i].set(ki)

            k = jax.lax.fori_loop(1, 6, stage_body, k)
            y_new = y + _B @ k
            k7 = eval_ode_fn(ode_fn, y_new, t + _C7 * dt, params_one)
            k = k.at[6].set(dt * k7)
            err_est = _E @ k
            return y_new, err_est, jnp.bool_(False), k7, k

        def update_extra(extra, k7, accept):
            # Next first stage: k7 on an accepted step (FSAL), otherwise the
            # current k1 (the retried step starts from the same t, y).
            return jnp.where(accept, k7, extra)

        def dense_eval(theta, y, y_new, dense_data):
            del y_new
            coeffs = _dense_coeffs(theta)
            return y[None, :] + coeffs @ dense_data

        return _step_one, extra_init, update_extra, dense_eval

    return solve_adaptive_ensemble(
        params_arr=params_arr,
        y0_arr=y0_arr,
        times=times,
        dt0=dt0,
        batch_size=bs,
        n_chunks=n_chunks,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        return_stats=return_stats,
        step_factory=step_factory,
        error_exponent=-1.0 / 5.0,
        safety=_SAFETY,
        factor_min=_FACTOR_MIN,
        factor_max=_FACTOR_MAX,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        error_weights_arr=error_weights_arr,
    )
