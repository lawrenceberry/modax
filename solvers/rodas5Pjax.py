"""Rodas5P solver — nonlinear ODE variant (ode_fn path).

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

from solvers._jax_common import (
    build_error_weights,
    eval_ode_fn,
    make_custom_vmap_solver,
    normalize_inputs,
    solve_adaptive_ensemble,
)

# fmt: off
# Rodas5P W-transformed coefficients (Steinebach 2023, BIT 63:27).
# Matches the Julia Rodas5P / GPURodas5P tableau. Stages 6-8 reuse the FSAL
# accumulation (u += k6, u += k7) so no a7/a8 rows are needed.
_gamma = 0.21193756319429014

_a21 = 3.0
_a31 = 2.849394379747939;  _a32 = 0.45842242204463923
_a41 = -6.954028509809101; _a42 = 2.489845061869568;   _a43 = -10.358996098473584
_a51 = 2.8029986275628964; _a52 = 0.5072464736228206;  _a53 = -0.3988312541770524; _a54 = -0.04721187230404641
_a61 = -7.502846399306121; _a62 = 2.561846144803919;   _a63 = -11.627539656261098; _a64 = -0.18268767659942256; _a65 = 0.030198172008377946

_C21 = -14.155112264123755
_C31 = -17.97296035885952; _C32 = -2.859693295451294
_C41 = 147.12150275711716; _C42 = -1.41221402718213;    _C43 = 71.68940251302358
_C51 = 165.43517024871676; _C52 = -0.4592823456491126;  _C53 = 42.90938336958603;   _C54 = -5.961986721573306
_C61 = 24.854864614690072; _C62 = -3.0009227002832186;  _C63 = 47.4931110020768;     _C64 = 5.5814197821558125;  _C65 = -0.6610691825249471
_C71 = 30.91273214028599;  _C72 = -3.1208243349937974;  _C73 = 77.79954646070892;    _C74 = 34.28646028294783;   _C75 = -19.097331116725623; _C76 = -28.087943162872662
_C81 = 37.80277123390563;  _C82 = -3.2571969029072276;  _C83 = 112.26918849496327;   _C84 = 66.9347231244047;    _C85 = -40.06618937091002;  _C86 = -54.66780262877968;  _C87 = -9.48861652309627

# Stage time coefficients for non-autonomous ODEs: t_stage = t + c_i * dt
_c2 = 0.6358126895828704
_c3 = 0.4095798393397535
_c4 = 0.9769306725060716
_c5 = 0.4288403609558664
# c6 = c7 = c8 = 1.0

# Time-derivative ("d_i") coefficients for non-autonomous ODEs. The stage RHS
# carries an additional dt*d_i*df/dt term (Hairer-Wanner II.7); without it,
# the method drops below order 5 when df/dt is nonzero. d6 = d7 = d8 = 0.
_d1 = _gamma  # = 0.21193756319429014
_d2 = -0.42387512638858027
_d3 = -0.3384627126235924
_d4 =  1.8046452872882734
_d5 =  2.325825639765069

_H_DENSE = jnp.array(
    [
        [
            25.948786856663858,
            -2.5579724845846235,
            10.433815404888879,
            -2.3679251022685204,
            0.524948541321073,
            1.1241088310450404,
            0.4272876194431874,
            -0.17202221070155493,
        ],
        [
            -9.91568850695171,
            -0.9689944594115154,
            3.0438037242978453,
            -24.495224566215796,
            20.176138334709044,
            15.98066361424651,
            -6.789040303419874,
            -6.710236069923372,
        ],
        [
            11.419903575922262,
            2.8879645146136994,
            72.92137995996029,
            80.12511834622643,
            -52.072871366152654,
            -59.78993625266729,
            -0.15582684282751913,
            4.883087185713722,
        ],
    ],
    dtype=jnp.float64,
)
# fmt: on


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
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    """Rodas5P ensemble solver for nonlinear ODEs."""

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _solve_impl(
            ode_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            lu_precision=lu_precision,
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
        "lu_precision",
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
    lu_precision: Literal["fp32", "fp64"] = "fp64",
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
    """Rodas5P ensemble solver implementation.

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
        Precision for LU factorization, triangular solves, and stage vectors:
        ``"fp32"`` or ``"fp64"``. State evolution, right-hand-side and
        Jacobian evaluations, and error control remain FP64. In FP32 mode a
        stage is retained at the precision at which the triangular solve
        produced it; widening it to FP64 cannot recover discarded precision.
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
    error_weights : array or None
        Per-component weights for the step-size error norm, shape ``(n_vars,)``
        or ``(N, n_vars)``. ``None`` (default) weights every component equally.
        A weight of 0 excludes that component from step-size control. Used in
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
    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64
    ode_eval = functools.partial(eval_ode_fn, ode_fn)
    jac_fn = jax.jacfwd(ode_eval, argnums=0)
    dT_fn = jax.jacfwd(ode_eval, argnums=1)

    y0_arr, times, params_arr, n, n_vars, _, dt0, bs, n_chunks = normalize_inputs(
        y0, t_span, params, first_step, batch_size
    )
    error_weights_arr = build_error_weights(error_weights, n, n_vars)

    eye = jnp.eye(n_vars, dtype=lu_dtype)

    def step_factory(params_one):
        def _step_one(y, t, dt, extra):
            del extra
            jac = jac_fn(y, t, params_one).astype(lu_dtype)
            dT = dT_fn(y, t, params_one)
            dtgamma_inv = (1.0 / (dt * _gamma)).astype(lu_dtype)
            lu = jax.scipy.linalg.lu_factor(dtgamma_inv * eye - jac)
            inv_dt = 1.0 / dt

            def f_eval(u, t_stage):
                return ode_eval(u, t_stage, params_one)

            def lu_solve(rhs):
                # Preserve each stage at the triangular-solve precision. In
                # FP32 mode this avoids widening an already-FP32 result only
                # to retain it for later stage combinations; FP64 mode remains
                # fully FP64. Expressions involving y/dt promote stages to
                # FP64 for state updates and error control.
                return jax.scipy.linalg.lu_solve(lu, rhs.astype(lu_dtype))

            dy = f_eval(y, t)
            k1 = lu_solve(dy + dt * _d1 * dT)

            u = y + _a21 * k1
            du = f_eval(u, t + _c2 * dt)
            k2 = lu_solve(du + dt * _d2 * dT + _C21 * k1 * inv_dt)

            u = y + _a31 * k1 + _a32 * k2
            du = f_eval(u, t + _c3 * dt)
            k3 = lu_solve(du + dt * _d3 * dT + (_C31 * k1 + _C32 * k2) * inv_dt)

            u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
            du = f_eval(u, t + _c4 * dt)
            k4 = lu_solve(
                du + dt * _d4 * dT + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt
            )

            u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
            du = f_eval(u, t + _c5 * dt)
            k5 = lu_solve(
                du
                + dt * _d5 * dT
                + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt
            )

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

            y_new = u + k8
            dense_stages = (k1, k2, k3, k4, k5, k6, k7, k8)
            dense_coeffs = []
            for row in range(3):
                accum = jnp.zeros_like(y)
                for col, stage in enumerate(dense_stages):
                    accum = accum + _H_DENSE[row, col] * stage
                dense_coeffs.append(accum)
            return y_new, k8, jnp.bool_(False), (), tuple(dense_coeffs)

        def dense_eval(theta, y, y_new, dense_data):
            h1, h2, h3 = dense_data
            theta1 = 1.0 - theta
            return (
                theta1[:, None] * y
                + theta[:, None]
                * (
                    y_new
                    + theta1[:, None]
                    * (h1 + theta[:, None] * (h2 + theta[:, None] * h3))
                )
            )

        return _step_one, (), lambda extra, candidate, accept: extra, dense_eval

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
        error_exponent=-1.0 / 6.0,
        safety=0.9,
        factor_min=0.2,
        factor_max=6.0,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        error_weights_arr=error_weights_arr,
    )
