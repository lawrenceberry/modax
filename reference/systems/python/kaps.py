"""Coupled Kaps singular-perturbation systems.

Kaps' problem — controllable stiffness with a known exact solution

Each equation pair (y1_i, y2_i) is an independent copy of the original Kaps system
with a different stiffness parameter epsilon_i. The stiffness ratio of pair i is
approximately 1 / epsilon_i, so epsilon_min controls the maximum stiffness across
the system. The initial conditions lie exactly on the slow manifold, so there is
no fast initial transient, only the stable fast modes that implicit solvers must
handle without being forced into tiny steps.

The analytical solution is independent of epsilon_i for all pairs:
    y1_i(t) = exp(-2t),  y2_i(t) = exp(-t)

This allows precise validation at arbitrary stiffness without a reference solver.
"""

import jax.numpy as jnp
import numpy as np

TIMES = jnp.array((0.0, 0.5, 1.0, 2.0), dtype=jnp.float64)


def make_system(n_pairs, epsilon_min):
    """Construct n_pairs independent Kaps singular-perturbation equation pairs."""
    n_vars = 2 * n_pairs
    epsilon = jnp.array(
        [10.0 ** (np.log10(epsilon_min) * i / (n_pairs - 1)) for i in range(n_pairs)],
        dtype=jnp.float64,
    )
    y0 = jnp.array([1.0, 1.0] * n_pairs, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        s = p[0]
        y1 = y[0::2]
        y2 = y[1::2]
        dy1 = s * (-(1.0 / epsilon + 2.0) * y1 + (1.0 / epsilon) * y2**2)
        dy2 = s * (y1 - y2 - y2**2)
        return jnp.stack([dy1, dy2], axis=1).ravel()

    def explicit_ode_fn(y, t, p):
        del t
        s = p[0]
        y1 = y[0::2]
        y2 = y[1::2]
        dy1 = s * (-2.0 * y1)
        dy2 = s * (y1 - y2 - y2**2)
        return jnp.stack([dy1, dy2], axis=1).ravel()

    def implicit_ode_fn(y, t, p):
        del t
        s = p[0]
        y1 = y[0::2]
        y2 = y[1::2]
        dy1 = s * (-(1.0 / epsilon) * (y1 - y2**2))
        dy2 = jnp.zeros_like(y2)
        return jnp.stack([dy1, dy2], axis=1).ravel()

    return {
        "n_pairs": n_pairs,
        "epsilon_min": epsilon_min,
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


def exact_solution(t_span, params, n_pairs):
    """Exact solution for n_pairs Kaps equation pairs."""
    t = np.asarray(t_span, dtype=np.float64)
    s = np.asarray(params)[:, 0]
    y1 = np.exp(-2.0 * np.outer(s, t))
    y2 = np.exp(-1.0 * np.outer(s, t))
    pair = np.stack([y1, y2], axis=-1)
    return np.tile(pair, (1, 1, n_pairs))
