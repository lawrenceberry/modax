import jax
import jax.numpy as jnp
import numpy as np

from solvers.rodas5 import solve as rodas5_solve
from solvers.tsit5 import solve as tsit5_solve

jax.config.update("jax_enable_x64", True)


def _decay(y, t, p):
    del t
    return -p[0] * y


def test_tsit5_broadcasts_parameter_vector_for_batched_initial_conditions():
    y0 = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float64)
    t_span = jnp.array([0.0, 0.1], dtype=jnp.float64)
    shared_params = jnp.array([0.5], dtype=jnp.float64)
    full_params = jnp.broadcast_to(shared_params, (y0.shape[0], shared_params.shape[0]))

    shared = tsit5_solve(
        _decay,
        y0,
        t_span,
        shared_params,
        first_step=0.01,
    ).block_until_ready()
    full = tsit5_solve(
        _decay,
        y0,
        t_span,
        full_params,
        first_step=0.01,
    ).block_until_ready()

    np.testing.assert_allclose(np.asarray(shared), np.asarray(full))


def test_rodas5_broadcasts_parameter_vector_for_batched_initial_conditions():
    y0 = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float64)
    t_span = jnp.array([0.0, 0.1], dtype=jnp.float64)
    shared_params = jnp.array([0.5], dtype=jnp.float64)
    full_params = jnp.broadcast_to(shared_params, (y0.shape[0], shared_params.shape[0]))

    shared = rodas5_solve(
        _decay,
        y0,
        t_span,
        shared_params,
        first_step=0.01,
    ).block_until_ready()
    full = rodas5_solve(
        _decay,
        y0,
        t_span,
        full_params,
        first_step=0.01,
    ).block_until_ready()

    np.testing.assert_allclose(np.asarray(shared), np.asarray(full))
