"""Rodas5 solver — Rosenbrock method of order 5(4) for stiff ODEs.

W-transformed variant from Di Marzo (1993) and DISCO-EB (Hahn).
"""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402

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
# fmt: on


def solve(f, y0, t_span, *, rtol=1e-8, atol=1e-10, first_step=None, max_steps=100000):
    """Solve a stiff autonomous ODE system using the Rodas5 method.

    Args:
        f: JAX function mapping state vector y -> dy/dt.
        y0: Initial conditions (list or array).
        t_span: Tuple (t0, tf) for integration bounds.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        first_step: Initial step size (optional).
        max_steps: Maximum number of steps.

    Returns:
        Array of shape (n_components, n_time_points) with the solution.
    """
    f_jit = jax.jit(f)
    jac_fn = jax.jit(jax.jacobian(f))

    @jax.jit
    def step(y, dt):
        n = y.shape[0]
        J = jac_fn(y)
        dtgamma = dt * _gamma
        W = jnp.eye(n) / dtgamma - J
        LU_piv = jax.scipy.linalg.lu_factor(W)
        inv_dt = 1.0 / dt

        # Stage 1
        dy = f_jit(y)
        k1 = jax.scipy.linalg.lu_solve(LU_piv, dy)

        # Stage 2
        u = y + _a21 * k1
        du = f_jit(u)
        k2 = jax.scipy.linalg.lu_solve(LU_piv, du + (_C21 * k1) * inv_dt)

        # Stage 3
        u = y + _a31 * k1 + _a32 * k2
        du = f_jit(u)
        k3 = jax.scipy.linalg.lu_solve(LU_piv, du + (_C31 * k1 + _C32 * k2) * inv_dt)

        # Stage 4
        u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
        du = f_jit(u)
        k4 = jax.scipy.linalg.lu_solve(
            LU_piv, du + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt
        )

        # Stage 5
        u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
        du = f_jit(u)
        k5 = jax.scipy.linalg.lu_solve(
            LU_piv,
            du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt,
        )

        # Stage 6
        u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
        du = f_jit(u)
        k6 = jax.scipy.linalg.lu_solve(
            LU_piv,
            du + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt,
        )

        # Stage 7
        u = u + k6
        du = f_jit(u)
        k7 = jax.scipy.linalg.lu_solve(
            LU_piv,
            du
            + (_C71 * k1 + _C72 * k2 + _C73 * k3 + _C74 * k4 + _C75 * k5 + _C76 * k6)
            * inv_dt,
        )

        # Stage 8
        u = u + k7
        du = f_jit(u)
        k8 = jax.scipy.linalg.lu_solve(
            LU_piv,
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
            * inv_dt,
        )

        y_new = u + k8
        return y_new, k8

    t0, tf = t_span
    y = jnp.array(y0, dtype=jnp.float64)
    dt = first_step if first_step is not None else (tf - t0) * 1e-6
    t = t0
    ys = [y]

    for _ in range(max_steps):
        if t >= tf:
            break
        dt = min(dt, tf - t)

        y_new, err_est = step(y, jnp.float64(dt))

        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
        err_norm = float(jnp.sqrt(jnp.mean((err_est / scale) ** 2)))

        if err_norm <= 1.0 and not (err_norm != err_norm):  # accept (and not NaN)
            t += dt
            y = y_new
            ys.append(y)

        if err_norm == 0.0:
            factor = 6.0
        elif err_norm != err_norm or err_norm > 1e18:  # NaN or huge
            factor = 0.2
        else:
            factor = min(6.0, max(0.2, 0.9 * err_norm ** (-1.0 / 6.0)))
        dt = float(dt * factor)

    assert t >= tf, f"Solver did not reach tf={tf} (stopped at t={t})"
    return jnp.stack(ys).T
