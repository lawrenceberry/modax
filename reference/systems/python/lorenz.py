"""Lorenz chaotic system (3D) for solver benchmarks and tests.

Summary:
    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz

with σ = 10, β = 8/3 fixed and ρ as the ensemble parameter (centred at 28,
the standard chaotic regime; chaos onset at ρ ≈ 24.74).

The Lorenz attractor is a strange attractor: trajectories orbit it forever but
never repeat.  The maximum Lyapunov exponent λ ≈ 0.9 means that two initially
close trajectories diverge on a timescale of ~1/λ ≈ 1 time unit, so there is
no meaningful point-wise reference solution after t ≈ 5.

Instead the tests verify attractor confinement: a solver that accumulates too
much integration error leaves the attractor manifold and diverges to infinity,
while a correct solver keeps all trajectories within the known attractor
extent (|x|, |y| < 40, 0 ≤ z < 65 for ρ ≈ 28) even after long integration.
These bounds are the signature of a solver that stays on the manifold.

Parameter perturbations are ±5% around ρ = 28.  All values land in [26.6, 29.4],
well above the chaos onset, so every ensemble member is fully chaotic.

Stiffness properties:
The Lorenz system is nonstiff for ρ ≈ 28, so explicit methods are appropriate.
"""

import jax.numpy as jnp
import numpy as np

N_VARS = 3
Y0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)


def ode_fn(y, t, p):
    del t
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = p[0]
    return jnp.array(
        [
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    )


def explicit_ode_fn(y, t, p):
    return ode_fn(y, t, p)


def implicit_ode_fn(y, t, p):
    del y, t, p
    return jnp.zeros(N_VARS, dtype=jnp.float64)


def make_params(size: int, seed: int = 42) -> jnp.ndarray:
    """ρ values centred at 28 with ±5% uniform perturbation (all chaotic)."""
    rng = np.random.default_rng(seed)
    return jnp.array(
        28.0 * (1.0 + 0.05 * (2.0 * rng.random((size, 1)) - 1.0)),
        dtype=jnp.float64,
    )
