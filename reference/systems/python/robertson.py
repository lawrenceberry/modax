"""Robertson stiff chemical kinetics system for solver benchmarks and tests."""

import jax.numpy as jnp
import numpy as np
from numba import cuda

TIMES = jnp.array((0.0, 1e-6, 1e-2, 1e2, 1e5), dtype=jnp.float64)

ALPHA = 0.99
EPS = 0.1
Y0 = jnp.array([(1 - EPS) * ALPHA, EPS, (1 - EPS) * (1 - ALPHA)], dtype=np.float64)
N_VARS = 3

PARAMS = jnp.array([0.04, 1e4, 3e7], dtype=jnp.float64)
N_PARAMS = 3


def ode_fn(y, t, p):
    del t
    return jnp.array(
        [
            -p[0] * y[0] + p[1] * y[1] * y[2],
            p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
            p[2] * y[1] ** 2,
        ]
    )


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    dy[i, 0] = -p[i, 0] * y[i, 0] + p[i, 1] * y[i, 1] * y[i, 2]
    dy[i, 1] = p[i, 0] * y[i, 0] - p[i, 1] * y[i, 1] * y[i, 2] - p[i, 2] * y[i, 1] ** 2
    dy[i, 2] = p[i, 2] * y[i, 1] ** 2


@cuda.jit(device=True)
def jac_fn_numba_cuda(y, t, p, jac, i):
    jac[i, 0, 0] = -p[i, 0]
    jac[i, 0, 1] = p[i, 1] * y[i, 2]
    jac[i, 0, 2] = p[i, 1] * y[i, 1]
    jac[i, 1, 0] = p[i, 0]
    jac[i, 1, 1] = -p[i, 1] * y[i, 2] - 2.0 * p[i, 2] * y[i, 1]
    jac[i, 1, 2] = -p[i, 1] * y[i, 1]
    jac[i, 2, 0] = 0.0
    jac[i, 2, 1] = 2.0 * p[i, 2] * y[i, 1]
    jac[i, 2, 2] = 0.0


def make_params(size: int, seed: int = 42) -> jnp.ndarray:
    """Return Robertson rate constants with +/-10% uniform perturbation."""
    rng = np.random.default_rng(seed)
    return jnp.array(
        PARAMS * (1.0 + 0.1 * (2.0 * rng.random((size, N_PARAMS)) - 1.0)),
        dtype=jnp.float64,
    )


def make_initial_conditions(size: int, seed: int = 42) -> jnp.ndarray:
    """ICs are parameterised by (alpha, epsilon):

        y(0) = [(1-eps)*alpha,  eps,  (1-eps)*(1-alpha)]

    where eps controls how much of the intermediate species y2 is present at
    t=0 and alpha distributes the remaining mass between fuel (y1) and product
    (y3).

      * eps > 0 forces the solver to resolve the fastest reaction timescale
        immediately: dy3/dt = k3*eps^2 is large at t=0, so the first step must
        be tiny and the Newton iteration may reject steps before settling onto
        the slow manifold.  The standard [1,0,0] IC has eps=0 and a long
        induction period where only the slow k1=0.04 rate is active.

      * alpha controls how long the stiff phase lasts: high alpha means plenty
        of y1 fuel to keep the reactions running, extending the hard region of
        the trajectory.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.0, ALPHA, size)
    y0 = np.column_stack(
        [
            (1 - EPS) * alpha,
            np.full(size, EPS),
            (1 - EPS) * (1 - alpha),
        ]
    )
    return y0


SCENARIOS = ("identical", "divergent")


def make_scenario(
    scenario: str, size: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "identical":
        return (
            np.broadcast_to(np.asarray(Y0), (size, N_VARS)).copy(),
            np.broadcast_to(np.asarray(PARAMS), (size, N_PARAMS)).copy(),
        )
    return make_initial_conditions(size, seed), np.asarray(make_params(size, seed))
