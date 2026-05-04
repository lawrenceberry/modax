import jax
import jax.numpy as jnp
import numpy as np

from reference.systems.python import robertson
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import solve as rodas5ckn_solve
from solvers.rodas5ckp import solve as rodas5ckp_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = robertson.TIMES[:3]  # (0, 1e-6, 1e-2) — short for speed
_KWARGS = {"first_step": 1e-6, "rtol": 1e-6, "atol": 1e-8}
_CUSTOM_KERNEL_ASSERT = {"rtol": 1e-8, "atol": 3e-8}


def _baseline(y0, params):
    out = rodas5_solve(
        robertson.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=jnp.asarray(params, dtype=jnp.float64),
        **_KWARGS,
    )
    return np.asarray(out.block_until_ready())


def test_rodas5ckp_matches_robertson_baseline():
    y0, params = robertson.make_scenario("divergent", 4)
    actual = rodas5ckp_solve(
        robertson.ode_fn_pallas,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        **_KWARGS,
    )
    np.testing.assert_allclose(
        np.asarray(actual), _baseline(y0, params), **_CUSTOM_KERNEL_ASSERT
    )


def test_rodas5ckn_matches_robertson_baseline():
    y0, params = robertson.make_scenario("divergent", 4)
    actual = rodas5ckn_solve(
        robertson.ode_fn_numba_cuda,
        robertson.jac_fn_numba_cuda,
        y0=y0,
        t_span=_T_SPAN,
        params=params,
        **_KWARGS,
    )
    np.testing.assert_allclose(actual, _baseline(y0, params), **_CUSTOM_KERNEL_ASSERT)
