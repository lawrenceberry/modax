"""Semi-discretized one-dimensional heat equation systems.

Heat equation — stiffness from grid refinement

M is tridiagonal with uniform entries -2 / dx^2 on the diagonal and 1 / dx^2 on
the off-diagonals. Eigenvalues range from about -pi^2 to -4 / dx^2, so the
stiffness ratio grows like O(N^2) with grid refinement.

PDE:
    du/dt = alpha d^2u/dx^2,  x in (0, 1),  u(0, t) = u(1, t) = 0

Grid:
    N interior points, dx = 1 / (N + 1), x_i = (i + 1) * dx

The initial condition y0_i = sin(pi x_i) is the first discrete eigenmode of M,
so the exact solution is y_i(t) = y0_i * exp(lambda_1 * alpha * t), where
lambda_1 = (2 / dx^2) * (cos(pi * dx) - 1). This lets tests validate against
the analytic answer rather than a second numerical solver.

The matrix-vector product has a cancellation hazard in fp32: each row computes a
second difference whose individual terms are much larger than the result for a
smooth sine mode. The benchmarks therefore keep matrix-vector evaluation in
fp64 and vary only LU precision.
"""

import jax.numpy as jnp
import numpy as np
from numba import cuda

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


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    """Hardcoded for n=4 (inv_dx2=25)."""
    scale = p[i, 0] * 25.0
    dy[i, 0] = scale * (-2.0 * y[i, 0] + y[i, 1])
    dy[i, 1] = scale * (y[i, 0] - 2.0 * y[i, 1] + y[i, 2])
    dy[i, 2] = scale * (y[i, 1] - 2.0 * y[i, 2] + y[i, 3])
    dy[i, 3] = scale * (y[i, 2] - 2.0 * y[i, 3])


@cuda.jit(device=True)
def jac_fn_numba_cuda(y, t, p, jac, i):
    """Hardcoded for n=4 (inv_dx2=25)."""
    for r in range(4):
        for c in range(4):
            jac[i, r, c] = 0.0
    scale = p[i, 0] * 25.0
    jac[i, 0, 0] = -2.0 * scale
    jac[i, 0, 1] = scale
    jac[i, 1, 0] = scale
    jac[i, 1, 1] = -2.0 * scale
    jac[i, 1, 2] = scale
    jac[i, 2, 1] = scale
    jac[i, 2, 2] = -2.0 * scale
    jac[i, 2, 3] = scale
    jac[i, 3, 2] = scale
    jac[i, 3, 3] = -2.0 * scale


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
