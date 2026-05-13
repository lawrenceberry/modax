"""Stiff diffusively coupled van der Pol oscillator ring lattice."""

import jax.numpy as jnp
import numpy as np
from numba import cuda

N_OSC = 35
N_VARS = 2 * N_OSC
N_PARAMS = 1
MU = 100.0
D = 10.0
OMEGA = 1.0
TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
Y0 = jnp.array([2.0, 0.0] * N_OSC, dtype=jnp.float64)
PARAMS = jnp.array([1.0], dtype=jnp.float64)


def _rhs_osc(x, v, x_left, x_right, scale, mu, d, omega):
    return (
        v,
        scale * mu * (1.0 - x * x) * v
        - omega * omega * x
        + d * (x_left - 2.0 * x + x_right),
    )


_rhs_osc_cuda = cuda.jit(device=True)(_rhs_osc)


def ode_fn(y, t, p):
    del t
    x = y[0::2]
    v = y[1::2]
    dx, dv = _rhs_osc(x, v, jnp.roll(x, 1), jnp.roll(x, -1), p[0], MU, D, OMEGA)
    return jnp.stack([dx, dv], axis=1).ravel()


def make_system(n_osc: int, *, mu: float = MU, d: float = D, omega: float = OMEGA):
    """Return (ode_fn, y0) for a ring of n_osc coupled van der Pol oscillators.

    Defaults reproduce the stiff baseline (mu=100, d=10, omega=1). Pass
    ``mu=1.0`` for the non-stiff variant used by explicit-method benchmarks.
    """
    y0 = jnp.array([2.0, 0.0] * n_osc, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        x = y[0::2]
        v = y[1::2]
        dx, dv = _rhs_osc(x, v, jnp.roll(x, 1), jnp.roll(x, -1), p[0], mu, d, omega)
        return jnp.stack([dx, dv], axis=1).ravel()

    return ode_fn, y0


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    """Hardcoded for n_osc=2, mu=1.0, d=10.0, omega=1.0."""
    scale = p[i, 0]
    # n_osc=2 ring: each oscillator's only neighbor is the other one
    dy[i, 0], dy[i, 1] = _rhs_osc_cuda(
        y[i, 0], y[i, 1], y[i, 2], y[i, 2], scale, 1.0, 10.0, 1.0
    )
    dy[i, 2], dy[i, 3] = _rhs_osc_cuda(
        y[i, 2], y[i, 3], y[i, 0], y[i, 0], scale, 1.0, 10.0, 1.0
    )


@cuda.jit(device=True)
def jac_fn_numba_cuda(y, t, p, jac, i):
    """Hardcoded for n_osc=2, mu=1.0, d=10.0, omega=1.0."""
    for r in range(4):
        for c in range(4):
            jac[i, r, c] = 0.0
    scale = p[i, 0]
    x0 = y[i, 0]
    v0 = y[i, 1]
    x1 = y[i, 2]
    v1 = y[i, 3]
    jac[i, 0, 1] = 1.0
    jac[i, 1, 0] = scale * (-2.0 * x0 * v0) - 21.0
    jac[i, 1, 1] = scale * (1.0 - x0 * x0)
    jac[i, 1, 2] = 20.0
    jac[i, 2, 3] = 1.0
    jac[i, 3, 0] = 20.0
    jac[i, 3, 2] = scale * (-2.0 * x1 * v1) - 21.0
    jac[i, 3, 3] = scale * (1.0 - x1 * x1)


def make_params(size: int, seed: int = 42) -> np.ndarray:
    """Return damping-scale parameters with ±20% uniform perturbation."""
    rng = np.random.default_rng(seed)
    return np.array(1.0 + 0.2 * (2.0 * rng.random((size, 1)) - 1.0), dtype=np.float64)


def make_scenario(
    n_osc: int,
    size: int,
    seed: int = 42,
    *,
    divergence: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial conditions and parameters for the coupled VDP lattice.

    ``divergence`` controls how far each trajectory is moved away from the
    synchronized baseline. ``0.0`` gives the identical state, ``1.0`` gives
    the original divergent initial-condition distribution, and larger values
    increase velocity spread and damping-scale variation.
    """
    n_vars = 2 * n_osc
    if not np.isfinite(divergence) or divergence < 0.0:
        raise ValueError("divergence must be finite and non-negative")

    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(0.25, 3.0, size=(size, n_osc))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n_osc))
    target_x = amplitudes * signs
    position_divergence = min(divergence, 1.0)
    x = 2.0 + position_divergence * (target_x - 2.0)
    v = rng.normal(0.0, 2.0 * divergence, size=(size, n_osc))
    y0 = np.empty((size, n_vars), dtype=np.float64)
    y0[:, 0::2] = x
    y0[:, 1::2] = v

    base_params = make_params(size, seed)
    param_center = max(divergence, 1.0)
    params = np.maximum(param_center + divergence * (base_params - 1.0), 1e-6).astype(
        np.float64
    )
    return y0, params


def make_initial_conditions(kind: str, size: int, seed: int = 42) -> np.ndarray:
    """Return baseline or broadly varied initial states.

    State ordering is ``(x0, v0, x1, v1, ..., x34, v34)``.
    """
    if kind == "identical":
        return np.broadcast_to(np.asarray(Y0, dtype=np.float64), (size, N_VARS)).copy()
    if kind != "ic_large":
        raise ValueError(f"unknown initial-condition kind: {kind}")

    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(0.25, 3.0, size=(size, N_OSC))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(size, N_OSC))
    x = amplitudes * signs
    v = rng.normal(0.0, 2.0, size=(size, N_OSC))
    y0 = np.empty((size, N_VARS), dtype=np.float64)
    y0[:, 0::2] = x
    y0[:, 1::2] = v
    return y0
