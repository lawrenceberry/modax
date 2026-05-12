"""Semi-discretized one-dimensional heat equation systems."""

import jax.numpy as jnp
import numpy as np

TIMES = jnp.array((0.0, 0.025, 0.05, 0.075, 0.1), dtype=jnp.float64)


def make_system(n_vars):
    """Construct a semi-discretized 1D heat equation with n_vars interior points."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    dx = 1.0 / (n_vars + 1)
    inv_dx2 = 1.0 / dx**2
    M_np = (
        np.diag(-2.0 * inv_dx2 * np.ones(n_vars))
        + np.diag(inv_dx2 * np.ones(n_vars - 1), 1)
        + np.diag(inv_dx2 * np.ones(n_vars - 1), -1)
    )
    M = jnp.array(M_np, dtype=jnp.float64)

    x = np.arange(1, n_vars + 1) * dx
    y0 = jnp.array(np.sin(np.pi * x), dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        return p[0] * (M @ y)

    def explicit_ode_fn(y, t, p):
        del y, t, p
        return jnp.zeros_like(y0)

    def implicit_ode_fn(y, t, p):
        return ode_fn(y, t, p)

    return {
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "y0": y0,
    }


def make_params(size, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def exact_solution(n_vars, times, params_batch):
    """Exact discrete heat-equation solution for the first sine eigenmode."""
    dx = 1.0 / (n_vars + 1)
    x = np.arange(1, n_vars + 1) * dx
    y0 = np.sin(np.pi * x)
    lambda_1 = (2.0 / dx**2) * (np.cos(np.pi * dx) - 1.0)
    alpha = np.asarray(params_batch)[:, 0]
    times_np = np.asarray(times)
    decay = np.exp(lambda_1 * alpha[:, None] * times_np[None, :])
    return y0[None, None, :] * decay[:, :, None]
