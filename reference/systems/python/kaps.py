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
from numba import cuda

TIMES = jnp.array((0.0, 0.5, 1.0, 2.0), dtype=jnp.float64)


def _rhs_pair(y1, y2, eps, scale):
    return (
        scale * (-(1.0 / eps + 2.0) * y1 + (1.0 / eps) * y2 * y2),
        scale * (y1 - y2 - y2 * y2),
    )


def _explicit_rhs_pair(y1, y2, scale):
    return (
        scale * (-2.0 * y1),
        scale * (y1 - y2 - y2 * y2),
    )


def _implicit_rhs_pair(y1, y2, eps, scale):
    return (
        scale * (-(1.0 / eps) * (y1 - y2 * y2)),
        y2 * 0.0,
    )


_rhs_pair_cuda = cuda.jit(device=True)(_rhs_pair)
_explicit_rhs_pair_cuda = cuda.jit(device=True)(_explicit_rhs_pair)
_implicit_rhs_pair_cuda = cuda.jit(device=True)(_implicit_rhs_pair)


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
        dy1, dy2 = _rhs_pair(y[0::2], y[1::2], epsilon, p[0])
        return jnp.stack([dy1, dy2], axis=1).ravel()

    def explicit_ode_fn(y, t, p):
        del t
        dy1, dy2 = _explicit_rhs_pair(y[0::2], y[1::2], p[0])
        return jnp.stack([dy1, dy2], axis=1).ravel()

    def implicit_ode_fn(y, t, p):
        del t
        dy1, dy2 = _implicit_rhs_pair(y[0::2], y[1::2], epsilon, p[0])
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


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    """Hardcoded for n_pairs=2, epsilon=[1.0, 0.01]."""
    scale = p[i, 0]
    dy[i, 0], dy[i, 1] = _rhs_pair_cuda(y[i, 0], y[i, 1], 1.0, scale)
    dy[i, 2], dy[i, 3] = _rhs_pair_cuda(y[i, 2], y[i, 3], 0.01, scale)


@cuda.jit(device=True)
def explicit_ode_fn_numba_cuda(y, t, p, dy, i):
    """Hardcoded for n_pairs=2."""
    scale = p[i, 0]
    for pair in range(2):
        base = 2 * pair
        dy[i, base], dy[i, base + 1] = _explicit_rhs_pair_cuda(
            y[i, base], y[i, base + 1], scale
        )


@cuda.jit(device=True)
def implicit_ode_fn_numba_cuda(y, t, p, dy, i):
    """Hardcoded for n_pairs=2, epsilon=[1.0, 0.01]."""
    scale = p[i, 0]
    dy[i, 0], dy[i, 1] = _implicit_rhs_pair_cuda(y[i, 0], y[i, 1], 1.0, scale)
    dy[i, 2], dy[i, 3] = _implicit_rhs_pair_cuda(y[i, 2], y[i, 3], 0.01, scale)


@cuda.jit(device=True)
def jac_fn_numba_cuda(y, t, p, jac, i):
    """Hardcoded for n_pairs=2, epsilon=[1.0, 0.01]."""
    for r in range(4):
        for c in range(4):
            jac[i, r, c] = 0.0
    scale = p[i, 0]
    eps0 = 1.0
    eps1 = 0.01
    jac[i, 0, 0] = scale * (-(1.0 / eps0 + 2.0))
    jac[i, 0, 1] = scale * (2.0 / eps0 * y[i, 1])
    jac[i, 1, 0] = scale
    jac[i, 1, 1] = scale * (-1.0 - 2.0 * y[i, 1])
    jac[i, 2, 2] = scale * (-(1.0 / eps1 + 2.0))
    jac[i, 2, 3] = scale * (2.0 / eps1 * y[i, 3])
    jac[i, 3, 2] = scale
    jac[i, 3, 3] = scale * (-1.0 - 2.0 * y[i, 3])


@cuda.jit(device=True)
def implicit_jac_fn_numba_cuda(y, t, p, jac, i):
    """Hardcoded for n_pairs=2, epsilon=[1.0, 0.01]."""
    for r in range(4):
        for c in range(4):
            jac[i, r, c] = 0.0
    scale = p[i, 0]
    jac[i, 0, 0] = -scale
    jac[i, 0, 1] = scale * 2.0 * y[i, 1]
    jac[i, 2, 2] = -100.0 * scale
    jac[i, 2, 3] = 200.0 * scale * y[i, 3]


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
