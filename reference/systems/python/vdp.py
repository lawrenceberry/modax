"""Stiff diffusively coupled van der Pol oscillator ring lattice."""

import jax.numpy as jnp
import numpy as np

N_OSC = 35
N_VARS = 2 * N_OSC
N_PARAMS = 1
MU = 100.0
D = 10.0
OMEGA = 1.0
TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
Y0 = jnp.array([2.0, 0.0] * N_OSC, dtype=jnp.float64)
PARAMS = jnp.array([1.0], dtype=jnp.float64)


def ode_fn(y, t, p):
    del t
    scale = p[0]
    x = y[0::2]
    v = y[1::2]
    laplacian = jnp.roll(x, -1) - 2.0 * x + jnp.roll(x, 1)
    dx = v
    dv = scale * MU * (1.0 - x * x) * v - OMEGA**2 * x + D * laplacian
    return jnp.stack([dx, dv], axis=1).ravel()


def make_system(n_osc: int, *, mu: float = MU, d: float = D, omega: float = OMEGA):
    """Return (ode_fn, y0) for a ring of n_osc coupled van der Pol oscillators.

    Defaults reproduce the stiff baseline (mu=100, d=10, omega=1). Pass
    ``mu=1.0`` for the non-stiff variant used by explicit-method benchmarks.
    """
    y0 = jnp.array([2.0, 0.0] * n_osc, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        scale = p[0]
        x = y[0::2]
        v = y[1::2]
        laplacian = jnp.roll(x, -1) - 2.0 * x + jnp.roll(x, 1)
        dx = v
        dv = scale * mu * (1.0 - x * x) * v - (omega**2) * x + d * laplacian
        return jnp.stack([dx, dv], axis=1).ravel()

    return ode_fn, y0


def make_params(size: int, seed: int = 42) -> np.ndarray:
    """Return damping-scale parameters with ±20% uniform perturbation."""
    rng = np.random.default_rng(seed)
    return np.array(1.0 + 0.2 * (2.0 * rng.random((size, 1)) - 1.0), dtype=np.float64)


SCENARIOS = ("identical", "divergent")


def make_scenario(
    scenario: str,
    n_osc: int,
    size: int,
    seed: int = 42,
    *,
    divergence: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial conditions and parameters for a named lattice scenario.

    For ``"divergent"``, ``divergence`` controls how far each trajectory is
    moved away from the synchronized baseline. ``0.0`` gives the identical
    state, ``1.0`` gives the original divergent initial-condition distribution,
    and larger values increase velocity spread and damping-scale variation.
    """
    n_vars = 2 * n_osc
    if not np.isfinite(divergence) or divergence < 0.0:
        raise ValueError("divergence must be finite and non-negative")

    if scenario == "identical":
        y0 = np.broadcast_to(
            np.array([2.0, 0.0] * n_osc, dtype=np.float64), (size, n_vars)
        ).copy()
        return y0, np.ones((size, N_PARAMS), dtype=np.float64)
    if scenario != "divergent":
        raise ValueError(f"unknown scenario: {scenario}")
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
    params = np.maximum(
        param_center + divergence * (base_params - 1.0), 1e-6
    ).astype(np.float64)
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
