import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from numba import cuda

from reference.systems.python import lorenz
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import solve as tsit5ckn_solve
from solvers.tsit5ckp import solve as tsit5ckp_solve
from solvers.tsit5ckw import solve as tsit5ckw_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = (0.0, 0.1)
_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}


def _decay_jax(y, t, p):
    del t
    return jnp.array([-p[0] * y[0], -p[1] * y[1]], dtype=y.dtype)


def _decay_pallas(y, t, p):
    del t
    return -p[:, 0] * y[:, 0], -p[:, 1] * y[:, 1]


@cuda.jit(device=True)
def _decay_numba_cuda(y, t, p, dy, i):
    dy[i, 0] = -p[i, 0] * y[i, 0]
    dy[i, 1] = -p[i, 1] * y[i, 1]


@wp.func
def _decay_warp(
    y: wp.array2d(dtype=wp.float64),
    t: wp.float64,
    p: wp.array2d(dtype=wp.float64),
    dy: wp.array2d(dtype=wp.float64),
    i: wp.int32,
):
    dy[i, 0] = -p[i, 0] * y[i, 0]
    dy[i, 1] = -p[i, 1] * y[i, 1]


def _baseline(y0, params):
    out = tsit5_solve(
        lorenz.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=jnp.asarray(params, dtype=jnp.float64),
        **_KWARGS,
    )
    return np.asarray(out.block_until_ready())


def test_tsit5ckp_matches_lorenz_baseline():
    y0, params = lorenz.make_scenario("divergent", 4)
    actual = tsit5ckp_solve(
        lorenz.ode_fn_pallas,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        **_KWARGS,
    )
    np.testing.assert_allclose(np.asarray(actual), _baseline(y0, params), rtol=1e-10)


def test_tsit5ckn_matches_lorenz_baseline():
    y0, params = lorenz.make_scenario("divergent", 4)
    actual = tsit5ckn_solve(
        lorenz.ode_fn_numba_cuda,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        **_KWARGS,
    )
    np.testing.assert_allclose(actual, _baseline(y0, params), rtol=1e-10)


def test_tsit5ckw_matches_lorenz_baseline_when_warp_is_installed():
    y0, params = lorenz.make_scenario("divergent", 4)
    actual = tsit5ckw_solve(
        lorenz.ode_fn_warp,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        **_KWARGS,
    )
    np.testing.assert_allclose(actual, _baseline(y0, params), rtol=1e-10)


def test_tsit5ckp_supports_generic_jax_rhs():
    y0 = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    params = jnp.array([[0.4, 0.5], [0.6, 0.7]], dtype=jnp.float64)
    expected = tsit5_solve(_decay_jax, y0, _T_SPAN, params, **_KWARGS)
    actual = tsit5ckp_solve(_decay_pallas, y0, _T_SPAN, params, **_KWARGS)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)


def test_tsit5ckn_supports_generic_device_rhs():
    y0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    params = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    expected = tsit5_solve(_decay_jax, y0, _T_SPAN, params, **_KWARGS)
    actual = tsit5ckn_solve(_decay_numba_cuda, y0, _T_SPAN, params, **_KWARGS)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-10)


def test_tsit5ckw_supports_generic_device_rhs():
    y0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    params = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    expected = tsit5_solve(_decay_jax, y0, _T_SPAN, params, **_KWARGS)
    actual = tsit5ckw_solve(_decay_warp, y0, _T_SPAN, params, **_KWARGS)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-10)
