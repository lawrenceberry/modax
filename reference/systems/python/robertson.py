"""Robertson stiff chemical kinetics system for solver benchmarks and tests."""

import jax.numpy as jnp
import numpy as np

N_VARS = 3
Y0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
TIMES = jnp.array((0.0, 1e-6, 1e-2, 1e2, 1e5), dtype=jnp.float64)


def ode_fn(y, t, p):
    del t
    return jnp.array(
        [
            -p[0] * y[0] + p[1] * y[1] * y[2],
            p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
            p[2] * y[1] ** 2,
        ]
    )


def explicit_ode_fn(y, t, p):
    del t
    return jnp.array(
        [
            -p[0] * y[0],
            p[0] * y[0],
            0.0,
        ]
    )


def implicit_ode_fn(y, t, p):
    del t
    return jnp.array(
        [
            p[1] * y[1] * y[2],
            -p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
            p[2] * y[1] ** 2,
        ]
    )


def make_params(size: int, seed: int = 42) -> jnp.ndarray:
    """Return Robertson rate constants with +/-10% uniform perturbation."""
    rng = np.random.default_rng(seed)
    base = np.array([0.04, 1e4, 3e7])
    return jnp.array(
        base * (1.0 + 0.1 * (2.0 * rng.random((size, 3)) - 1.0)),
        dtype=jnp.float64,
    )
