"""Non-autonomous correctness tests for the Rodas5 solvers.

For a Rosenbrock-Wanner method to retain its order on a non-autonomous ODE,
each stage must include the explicit ``dt * d_i * df/dt`` correction in
addition to evaluating ``f`` at the stage time.  These tests use a linear
non-autonomous problem with a closed-form solution to verify both the JAX
and the numba/CUDA Rodas5 implementations.

Reference ODE:  dy/dt = -lambda * y + cos(omega * t),   y(0) = 0
Closed form:    y(t) = (lambda * cos(omega t) + omega * sin(omega t)
                       - lambda * exp(-lambda t)) / (lambda**2 + omega**2)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba import cuda

from solvers.rodas5jax import solve as rodas5jax_solve
from solvers.rodas5numba import solve as rodas5numba_solve

jax.config.update("jax_enable_x64", True)

LAMBDA = 10.0
OMEGA = 5.0
T_FINAL = 1.0
N_SAVE = 11


def _exact(t, lam, omega):
    return (
        lam * np.cos(omega * t) + omega * np.sin(omega * t) - lam * np.exp(-lam * t)
    ) / (lam**2 + omega**2)


def _ode_jax(y, t, p):
    lam, omega = p[0], p[1]
    return -lam * y + jnp.cos(omega * t)


@cuda.jit(device=True)
def _ode_cuda(y, t, p, dy, i):
    dy[i, 0] = -p[i, 0] * y[i, 0] + np.cos(p[i, 1] * t)


@cuda.jit(device=True)
def _jac_cuda(y, t, p, jac, i):
    jac[i, 0, 0] = -p[i, 0]


@cuda.jit(device=True)
def _time_jac_cuda(y, t, p, dT, i):
    dT[i, 0] = -p[i, 1] * np.sin(p[i, 1] * t)


def _setup():
    y0 = np.zeros((1, 1), dtype=np.float64)
    t_span = np.linspace(0.0, T_FINAL, N_SAVE, dtype=np.float64)
    params = np.array([[LAMBDA, OMEGA]], dtype=np.float64)
    exact = _exact(t_span, LAMBDA, OMEGA)
    return y0, t_span, params, exact


def test_rodas5jax_nonautonomous_matches_analytical():
    y0, t_span, params, exact = _setup()
    sol = rodas5jax_solve(
        _ode_jax,
        jnp.asarray(y0),
        jnp.asarray(t_span),
        jnp.asarray(params),
        rtol=1e-9,
        atol=1e-11,
        first_step=1e-3,
    )
    y_num = np.asarray(sol)[0, :, 0]
    err = np.max(np.abs(y_num - exact))
    assert err < 1e-7, f"max error {err:.2e} too large (with dT correction)"


def test_rodas5jax_nonautonomous_convergence():
    """At progressively tighter tolerances the error must shrink.

    Without the dT correction, the spatial accuracy of the stage RHS is
    only first order in dt, so the error plateaus regardless of how small
    rtol is.  With the fix, the error tracks the tolerance.
    """
    y0, t_span, params, exact = _setup()
    errs = []
    for rtol in (1e-4, 1e-6, 1e-8):
        sol = rodas5jax_solve(
            _ode_jax,
            jnp.asarray(y0),
            jnp.asarray(t_span),
            jnp.asarray(params),
            rtol=rtol,
            atol=rtol * 1e-2,
            first_step=1e-3,
        )
        y_num = np.asarray(sol)[0, :, 0]
        errs.append(float(np.max(np.abs(y_num - exact))))

    # Each tighter tolerance must shave at least one order of magnitude off
    # the error (Rodas5 has plenty of headroom on this smooth problem).
    assert errs[1] < errs[0] / 10, f"errors {errs} do not converge"
    assert errs[2] < errs[1] / 10, f"errors {errs} do not converge"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")
def test_rodas5numba_nonautonomous_matches_analytical():
    y0, t_span, params, exact = _setup()
    sol = rodas5numba_solve(
        _ode_cuda,
        _jac_cuda,
        y0,
        t_span,
        params,
        time_jac_fn=_time_jac_cuda,
        rtol=1e-9,
        atol=1e-11,
        first_step=1e-3,
    )
    y_num = np.asarray(sol)[0, :, 0]
    err = np.max(np.abs(y_num - exact))
    # LU is float32 in rodas5numba, so we cannot expect 1e-8 here.
    assert err < 1e-5, f"max error {err:.2e} too large (with dT correction)"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")
def test_rodas5numba_autonomous_default_unchanged():
    """Passing ``time_jac_fn=None`` (the default) must still solve an
    autonomous problem correctly: the kernel uses a zero-stub for dT and
    every ``D_i * dT`` term contributes zero.
    """
    # Re-use the same RHS with omega=0, so f = -lambda*y + cos(0) = -lambda*y + 1.
    # Equilibrium 1/lambda, exact: y(t) = 1/lambda + (1 - 1/lambda) exp(-lambda*t).
    y0 = np.ones((1, 1), dtype=np.float64)
    t_span = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    params = np.array([[LAMBDA, 0.0]], dtype=np.float64)
    exact = 1.0 / LAMBDA + (1.0 - 1.0 / LAMBDA) * np.exp(-LAMBDA * t_span)

    sol = rodas5numba_solve(
        _ode_cuda,
        _jac_cuda,
        y0,
        t_span,
        params,
        rtol=1e-7,
        atol=1e-9,
        first_step=1e-3,
    )
    y_num = np.asarray(sol)[0, :, 0]
    err = np.max(np.abs(y_num - exact))
    assert err < 1e-5, f"max error {err:.2e} too large (autonomous default)"
