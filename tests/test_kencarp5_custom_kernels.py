import jax
import jax.numpy as jnp
import numpy as np
from numba import cuda

from reference.systems.python import brusselator
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.kencarp5ckn import solve as kencarp5ckn_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = jnp.array((0.0, 0.01, 0.02), dtype=jnp.float64)
_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}


def test_kencarp5ckn_linear_matches_brusselator_baseline():
    ex_fn, im_fn, _, _ = brusselator.make_system(3)
    y0, params = brusselator.make_scenario("divergent", 3, 4)
    expected = kencarp5_solve(
        ex_fn,
        im_fn,
        jnp.asarray(y0, dtype=jnp.float64),
        _T_SPAN,
        jnp.asarray(params, dtype=jnp.float64),
        linear=True,
        **_KWARGS,
    )
    actual = kencarp5ckn_solve(
        brusselator.explicit_ode_fn_numba_cuda,
        brusselator.implicit_ode_fn_numba_cuda,
        brusselator.implicit_jac_fn_numba_cuda,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        linear=True,
        **_KWARGS,
    )
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-7, atol=1e-8)


def _explicit_zero(y, t, p):
    del t, p
    return jnp.zeros_like(y)


def _implicit_quadratic(y, t, p):
    del t
    return jnp.array([-p[0] * y[0] * y[0]], dtype=y.dtype)


@cuda.jit(device=True)
def _explicit_zero_numba_cuda(y, t, p, dy, i):
    dy[i, 0] = 0.0


@cuda.jit(device=True)
def _implicit_quadratic_numba_cuda(y, t, p, dy, i):
    dy[i, 0] = -p[i, 0] * y[i, 0] * y[i, 0]


@cuda.jit(device=True)
def _implicit_quadratic_jac_numba_cuda(y, t, p, jac, i):
    jac[i, 0, 0] = -2.0 * p[i, 0] * y[i, 0]


def test_kencarp5ckn_newton_matches_nonlinear_baseline():
    y0 = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float64)
    params = np.array([[0.5], [0.75], [1.0], [1.25]], dtype=np.float64)
    expected = kencarp5_solve(
        _explicit_zero,
        _implicit_quadratic,
        jnp.asarray(y0, dtype=jnp.float64),
        _T_SPAN,
        jnp.asarray(params, dtype=jnp.float64),
        linear=False,
        **_KWARGS,
    )
    actual = kencarp5ckn_solve(
        _explicit_zero_numba_cuda,
        _implicit_quadratic_numba_cuda,
        _implicit_quadratic_jac_numba_cuda,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        linear=False,
        **_KWARGS,
    )
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-8, atol=1e-10)
