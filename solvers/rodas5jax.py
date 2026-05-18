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

from solvers._jax_common import normalize_inputs, solve_adaptive_ensemble

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

    y0_arr, times, params_arr, _, n_vars, _, dt0, bs, n_chunks = normalize_inputs(
        y0, t_span, params, first_step, batch_size
    )

    eye = jnp.eye(n_vars, dtype=lu_dtype)

    def step_factory(params_one):
        def _step_one(y, t, dt, extra):
            del extra
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

            return u + k8, k8, jnp.bool_(False), ()

        return _step_one, (), lambda extra, candidate, accept: extra

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
    )
