"""Bateman radioactive decay chain systems.

Physical background
-------------------
The Bateman equations describe a sequential radioactive decay chain
A_1 -> A_2 -> ... -> A_n, where each species A_i decays into A_{i+1} with a
first-order rate constant lambda_i. The final species A_n is stable. Starting
from a pure parent population, the parent feeds the chain, each intermediate is
produced by its predecessor and destroyed by its own decay, and the stable end
product accumulates irreversibly.

ODE system
----------
    dN_1/dt = -lambda_1 N_1
    dN_i/dt = lambda_{i-1} N_{i-1} - lambda_i N_i
    dN_n/dt = lambda_{n-1} N_{n-1}

The coefficient matrix M is lower bidiagonal: diagonal entries -lambda_i and
subdiagonal entries +lambda_i. A global rate-scale factor p[0] multiplies M as
the ensemble parameter. Total population is conserved for all t.

Analytical solution
-------------------
Because M is constant the exact solution is N(t) = expm(p[0] * M * t) * N0.
This module evaluates it through an eigendecomposition of M so tests can compare
against a closed-form linear solution.

Stiffness character
-------------------
The Jacobian equals p[0] * M identically, independent of t and y. The stiffness
ratio is fixed at lambda_max / lambda_min throughout integration, and the
``stiffness`` parameter sets this ratio directly.
"""

import jax.numpy as jnp
import numpy as np
from numba import cuda

TIMES = jnp.array((0.0, 0.2, 0.5, 1.0, 2.0), dtype=jnp.float64)


def make_system(n_vars, stiffness):
    """Construct an n_vars Bateman decay-chain system."""
    n_radioactive = n_vars - 1
    lambdas = np.array(
        [
            10.0 ** (np.log10(stiffness) * i / (n_radioactive - 1))
            for i in range(n_radioactive)
        ],
        dtype=np.float64,
    )

    M_np = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_radioactive):
        M_np[i, i] = -lambdas[i]
        M_np[i + 1, i] = lambdas[i]

    M = jnp.array(M_np, dtype=jnp.float64)
    y0 = jnp.array([1.0] + [0.0] * n_radioactive, dtype=jnp.float64)

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
        "stiffness": stiffness,
        "ode_fn": ode_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "y0": y0,
        "M_np": M_np,
    }


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    """Hardcoded for n=4, stiffness=100 (decay rates 1, 10, 100)."""
    scale = p[i, 0]
    dy[i, 0] = scale * (-y[i, 0])
    dy[i, 1] = scale * (y[i, 0] - 10.0 * y[i, 1])
    dy[i, 2] = scale * (10.0 * y[i, 1] - 100.0 * y[i, 2])
    dy[i, 3] = scale * (100.0 * y[i, 2])


@cuda.jit(device=True)
def jac_fn_numba_cuda(y, t, p, jac, i):
    """Hardcoded for n=4, stiffness=100 (decay rates 1, 10, 100)."""
    for r in range(4):
        for c in range(4):
            jac[i, r, c] = 0.0
    scale = p[i, 0]
    jac[i, 0, 0] = -scale
    jac[i, 1, 0] = scale
    jac[i, 1, 1] = -10.0 * scale
    jac[i, 2, 1] = 10.0 * scale
    jac[i, 2, 2] = -100.0 * scale
    jac[i, 3, 2] = 100.0 * scale


def make_params(size, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def exact_solution(M_np, y0_np, t_span, params):
    """Exact solution via eigendecomposition."""
    d, L = np.linalg.eig(M_np)
    d = np.real(d)
    L = np.real(L)
    w = np.linalg.solve(L, y0_np)

    t_arr = np.asarray(t_span, dtype=np.float64)
    s_arr = np.asarray(params, dtype=np.float64)[:, 0]
    alpha = np.outer(s_arr, t_arr)
    exp_vals = np.exp(alpha[:, :, None] * d)
    return (exp_vals * w) @ L.T
