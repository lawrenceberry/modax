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
    error_norm_fn=None,
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
            error_norm_fn=error_norm_fn,
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
        "error_norm_fn",
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
    error_norm_fn=None,
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
        a weight of 0 excludes that component from step-size control. Passed
        through to a custom ``error_norm_fn``.
    error_norm_fn : callable or None
        Custom error norm with signature ``error_norm_fn(y, y_new, err_est,
        rtol, atol, error_weights) -> scalar``, where ``error_weights`` is the
        per-trajectory row. Must be a traceable
        JAX reduction returning a scalar where ``<= 1`` means "accept". ``None``
        (default) uses the built-in weighted RMS norm.
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
    error_weights_arr = build_error_weights(error_weights, n, n_vars)

    def step_factory(params_one):
        extra_init = (
            jnp.zeros((n_vars,), dtype=jnp.float64),
            jnp.bool_(False),
        )

        def _fresh_k1(y, t, k_fsal, has_fsal):
            return jax.lax.cond(
                has_fsal,
                lambda _: k_fsal,
                lambda _: ode_fn(y, t, params_one),
                operand=None,
            )

        def _step_one(y, t, dt, extra):
            k_fsal, has_fsal = extra
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
            return y_new, err_est, jnp.bool_(False), k7

        def update_extra(extra, k7, accept):
            k_fsal, _ = extra
            return jnp.where(accept, k7, jnp.zeros_like(k_fsal)), accept

        return _step_one, extra_init, update_extra

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
        error_norm_fn=error_norm_fn,
    )
