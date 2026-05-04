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

"""

import jax.numpy as jnp
import numpy as np
from numba import cuda


def _zeros_like(x):
    return x * 0.0


TIMES = (0.0, 5.0)

Y0 = jnp.array([1000.0, -500.0, 500.0], dtype=np.float64)
N_VARS = 3

PARAMS = jnp.array([0.5], dtype=np.float64)
N_PARAMS = 1


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


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = p[i, 0]
    dy[i, 0] = sigma * (y[i, 1] - y[i, 0])
    dy[i, 1] = y[i, 0] * (rho - y[i, 2]) - y[i, 1]
    dy[i, 2] = y[i, 0] * y[i, 1] - beta * y[i, 2]


def make_params(size: int, seed: int = 42) -> jnp.ndarray:
    """ρ values centred at 0.5 with ±5% uniform perturbation (all chaotic)."""
    rng = np.random.default_rng(seed)
    return jnp.array(
        0.5 * (1.0 + 0.05 * (2.0 * rng.random((size, 1)) - 1.0)),
        dtype=jnp.float64,
    )


def make_initial_conditions(size: int, seed: int = 42) -> jnp.ndarray:
    """The Lorenz system is run with rho=0.5, which places it below the pitchfork
    bifurcation at rho=1. In this regime the origin is the unique globally stable
    fixed point, so every trajectory decays exponentially toward (0, 0, 0) with
    slowest eigenvalue lambda ~ -0.47.

    Step count is therefore strictly monotone in distance from the origin: a
    trajectory that starts far away must cross a large region of phase space with
    significant ODE velocities before the state becomes small enough for Tsit5 to
    take large steps.

    The hard base IC, _HARD_Y0 = [1000, -500, 500], is fixed analytically as a
    point far from the origin. The identical scenario places every trajectory
    there, so all share the same (maximum) step count and the batch shows zero
    divergence. The ic_large scenario scales each trajectory as _HARD_Y0 * (1-t)
    for t ~ U(0, 1), moving it radially toward the origin. Because distance from
    the origin is the sole driver of difficulty, this construction guarantees that
    every ic_large trajectory is strictly easier than the identical base, and the
    step counts span a continuous range from near-maximum (t -> 0) down to
    near-zero (t -> 1).
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(size, 1))
    return Y0 * (1.0 - t)


SCENARIOS = ("identical", "divergent")


def make_scenario(
    scenario: str, size: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "identical":
        return (
            np.broadcast_to(np.asarray(Y0), (size, N_VARS)).copy(),
            np.broadcast_to(np.asarray(PARAMS), (size, N_PARAMS)).copy(),
        )
    return (
        np.asarray(make_initial_conditions(size, seed)),
        np.asarray(make_params(size, seed)),
    )
