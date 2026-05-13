"""Discretized Brusselator — multi-dimensional reaction-diffusion benchmark.

A multi-dimensional **Discretized Brusselator** is a classic reaction-diffusion
system used to study pattern formation (Turing structures) and to benchmark
numerical solvers for stiff partial differential equations (PDEs).

In its continuous form, the Brusselator describes the evolution of two chemical
concentrations, ``u(x, t)`` and ``v(x, t)``, across a spatial domain.

1. The governing PDE
--------------------
The system in ``d`` dimensions is::

    du/dt = D_u * laplacian(u) + A + u^2 * v - (B + 1) * u
    dv/dt = D_v * laplacian(v) + B * u - u^2 * v

Where:

- ``laplacian`` is the Laplacian operator (sum of second partial derivatives).
- ``D_u``, ``D_v`` are diffusion coefficients.
- ``A``, ``B`` are constants representing external supply / reaction rates.
- ``A + u^2 * v - (B+1) * u`` is the reaction term (the source of nonlinearity).

2. Spatial discretization (the ODE system)
------------------------------------------
To solve this numerically the spatial domain is discretized into a grid. For
a 2-D domain of size ``N x N`` with spacings ``dx`` and ``dy``, ``u`` and ``v``
are represented as vectors ``u, v ∈ R^(N^2)``.

The Laplacian is replaced by a discrete operator, typically the standard
5-point stencil in 2-D::

    laplacian(u)[i,j] ≈ (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx^2
                       + (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy^2

When applied across the entire grid this becomes a matrix-vector product
``L * u`` where ``L`` is a sparse, high-dimensional matrix (Kronecker sums
of 1-D second-difference matrices).

3. The additive structure
-------------------------
This discretization yields a system of ``2 * N^d`` coupled ODEs. In an
**IMEX (Implicit-Explicit)** context the system is split into two parts::

    d/dt [u; v] = [D_u * L * u; D_v * L * v]       <- f_stiff (linear, diffusion)
                + [A + u^2 ⊙ v - (B+1) u;
                   B u - u^2 ⊙ v]                  <- f_non_stiff (nonlinear, reaction)

- **Stiff part (`f_stiff`)**: the diffusion terms are linear but become
  extremely stiff as grid resolution increases — eigenvalues of ``L`` scale
  as ``O(1 / dx^2)``. Handled **implicitly** to avoid tiny time steps.
- **Non-stiff part (`f_non_stiff`)**: the reaction terms are nonlinear but
  have slower timescales than the high-frequency diffusion modes. Handled
  **explicitly**.

4. Why it is a benchmark favourite
----------------------------------
- **Sparse / matrix-free testing.** In high dimensions (2-D / 3-D) the
  Jacobian of ``f_stiff`` is too large to store explicitly, forcing
  matrix-free methods and iterative linear solvers (GMRES, CG) within the
  implicit step.
- **Pattern formation.** When ``B > 1 + A^2 + D_u/D_v`` (roughly) the system
  becomes unstable and forms Turing patterns — spots or stripes. This gives
  a visual sanity check on whether the numerical scheme preserves physical
  stability.
- **Boundary conditions.** It is a clean testbed for periodic BCs (which let
  you use FFTs for the Laplacian) or Neumann BCs (zero-flux at the edges).

Implementation note
-------------------
This module realises the **1-D periodic ring** case of the above system as
a JAX-compatible benchmark for the project's IMEX solvers. Specifics:

- Single spatial dimension, ring topology (periodic BCs implemented with
  ``jnp.roll`` on the grid).
- Grid size ``n_grid`` is user-tunable; the ODE dimension is ``2 * n_grid``.
- A single diffusion coefficient ``alpha`` is used for both species
  (``D_u = D_v = alpha``); domain length ``length`` defaults to 1, so
  ``dx = length / n_grid``.
- State layout is interleaved: ``y[2i] = u_i``, ``y[2i+1] = v_i`` for
  ``i in [0, n_grid)``. ``jnp.roll(u, ±1)`` realises the periodic 3-point
  stencil ``u_{i+1} - 2*u_i + u_{i-1}`` directly on the ``u`` slice.
- The Laplacian Jacobian is linear and constant, so kencarp5 can run it
  with ``linear=True`` (one LU per stage, no Newton iteration).
- ``params[0]`` is a per-trajectory reaction-rate **scale** that multiplies
  both ``A`` and ``B`` (and only acts on the explicit reaction half).
  Used by ``make_scenario(..., divergence=...)`` to spread step counts across
  the ensemble.
- The split is exact: ``ode_fn(y, t, p) == explicit_ode_fn(y, t, p) +
  implicit_ode_fn(y, t, p)`` everywhere, so any non-split solver can use
  ``ode_fn`` directly.

Pattern-formation regime: at the default ``A = 1.0``, ``B = 3.0`` the
homogeneous fixed point ``u* = A``, ``v* = B / A`` is unstable (Hopf), so
the cosine perturbation imposed by ``Y0`` evolves into a non-trivial
trajectory that exercises both halves of the IMEX split.
"""

import jax.numpy as jnp
import numpy as np
from numba import cuda

N_GRID = 32
N_VARS = 2 * N_GRID
N_PARAMS = 1
A = 1.0
B = 3.0
ALPHA = 0.02
L = 1.0
TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
PARAMS = jnp.array([1.0], dtype=jnp.float64)


def _equilibrium(n_grid: int, a: float = A, b: float = B):
    """Single-mode perturbation around the homogeneous fixed point (u=A, v=B/A)."""
    i = np.arange(n_grid)
    u = a * (1.0 + 0.1 * np.cos(2.0 * np.pi * i / n_grid))
    v = np.full(n_grid, b / a, dtype=np.float64)
    y = np.empty(2 * n_grid, dtype=np.float64)
    y[0::2] = u
    y[1::2] = v
    return y


Y0 = jnp.asarray(_equilibrium(N_GRID), dtype=jnp.float64)


def _explicit_rhs_cell(u, v, a_eff, b_eff):
    u2v = u * u * v
    return (
        a_eff + u2v - (b_eff + 1.0) * u,
        b_eff * u - u2v,
    )


def _implicit_rhs_cell(u, v, u_left, u_right, v_left, v_right, diff_coeff):
    return (
        diff_coeff * (u_left - 2.0 * u + u_right),
        diff_coeff * (v_left - 2.0 * v + v_right),
    )


def _rhs_cell(u, v, u_left, u_right, v_left, v_right, a_eff, b_eff, diff_coeff):
    u2v = u * u * v
    return (
        a_eff + u2v - (b_eff + 1.0) * u + diff_coeff * (u_left - 2.0 * u + u_right),
        b_eff * u - u2v + diff_coeff * (v_left - 2.0 * v + v_right),
    )


_explicit_rhs_cell_cuda = cuda.jit(device=True)(_explicit_rhs_cell)
_implicit_rhs_cell_cuda = cuda.jit(device=True)(_implicit_rhs_cell)
_rhs_cell_cuda = cuda.jit(device=True)(_rhs_cell)


def make_system(
    n_grid: int,
    *,
    a: float = A,
    b: float = B,
    alpha: float = ALPHA,
    length: float = L,
):
    """Return ``(explicit_ode_fn, implicit_ode_fn, ode_fn, y0)`` closed over ``n_grid``.

    ``p[0]`` is a reaction-rate scale that multiplies ``a`` and ``b``; it acts
    only on the explicit (reaction) half. ``divergent`` scenarios perturb it.
    """
    dx = length / n_grid
    diff_coeff = alpha / (dx * dx)
    y0 = jnp.asarray(_equilibrium(n_grid, a, b), dtype=jnp.float64)

    def explicit_ode_fn(y, t, p):
        del t
        u = y[0::2]
        v = y[1::2]
        du, dv = _explicit_rhs_cell(u, v, p[0] * a, p[0] * b)
        return jnp.stack([du, dv], axis=1).ravel()

    def implicit_ode_fn(y, t, p):
        del t, p
        u = y[0::2]
        v = y[1::2]
        du, dv = _implicit_rhs_cell(
            u,
            v,
            jnp.roll(u, 1),
            jnp.roll(u, -1),
            jnp.roll(v, 1),
            jnp.roll(v, -1),
            diff_coeff,
        )
        return jnp.stack([du, dv], axis=1).ravel()

    def ode_fn(y, t, p):
        del t
        u = y[0::2]
        v = y[1::2]
        du, dv = _rhs_cell(
            u,
            v,
            jnp.roll(u, 1),
            jnp.roll(u, -1),
            jnp.roll(v, 1),
            jnp.roll(v, -1),
            p[0] * a,
            p[0] * b,
            diff_coeff,
        )
        return jnp.stack([du, dv], axis=1).ravel()

    return explicit_ode_fn, implicit_ode_fn, ode_fn, y0


_DEFAULT_EXPLICIT, _DEFAULT_IMPLICIT, _DEFAULT_ODE, _ = make_system(N_GRID)


def explicit_ode_fn(y, t, p):
    return _DEFAULT_EXPLICIT(y, t, p)


def implicit_ode_fn(y, t, p):
    return _DEFAULT_IMPLICIT(y, t, p)


def ode_fn(y, t, p):
    return _DEFAULT_ODE(y, t, p)


@cuda.jit(device=True)
def ode_fn_numba_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    a_eff = scale * A
    b_eff = scale * B
    n_grid = y.shape[1] // 2
    dx = L / n_grid
    diff_coeff = ALPHA / (dx * dx)
    for g in range(n_grid):
        left = g - 1
        if left < 0:
            left = n_grid - 1
        right = g + 1
        if right >= n_grid:
            right = 0
        u = y[i, 2 * g]
        v = y[i, 2 * g + 1]
        dy[i, 2 * g], dy[i, 2 * g + 1] = _rhs_cell_cuda(
            u,
            v,
            y[i, 2 * left],
            y[i, 2 * right],
            y[i, 2 * left + 1],
            y[i, 2 * right + 1],
            a_eff,
            b_eff,
            diff_coeff,
        )


@cuda.jit(device=True)
def explicit_ode_fn_numba_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    a_eff = scale * A
    b_eff = scale * B
    n_grid = y.shape[1] // 2
    for g in range(n_grid):
        u = y[i, 2 * g]
        v = y[i, 2 * g + 1]
        dy[i, 2 * g], dy[i, 2 * g + 1] = _explicit_rhs_cell_cuda(u, v, a_eff, b_eff)


@cuda.jit(device=True)
def implicit_ode_fn_numba_cuda(y, t, p, dy, i):
    n_grid = y.shape[1] // 2
    dx = L / n_grid
    diff_coeff = ALPHA / (dx * dx)
    for g in range(n_grid):
        left = g - 1
        if left < 0:
            left = n_grid - 1
        right = g + 1
        if right >= n_grid:
            right = 0
        u = y[i, 2 * g]
        v = y[i, 2 * g + 1]
        dy[i, 2 * g], dy[i, 2 * g + 1] = _implicit_rhs_cell_cuda(
            u,
            v,
            y[i, 2 * left],
            y[i, 2 * right],
            y[i, 2 * left + 1],
            y[i, 2 * right + 1],
            diff_coeff,
        )


@cuda.jit(device=True)
def implicit_jac_fn_numba_cuda(y, t, p, jac, i):
    n_vars = jac.shape[1]
    n_grid = n_vars // 2
    dx = L / n_grid
    diff_coeff = ALPHA / (dx * dx)
    for r in range(n_vars):
        for c in range(n_vars):
            jac[i, r, c] = 0.0
    for g in range(n_grid):
        left = g - 1
        if left < 0:
            left = n_grid - 1
        right = g + 1
        if right >= n_grid:
            right = 0
        u_row = 2 * g
        v_row = 2 * g + 1
        jac[i, u_row, 2 * g] += -2.0 * diff_coeff
        jac[i, u_row, 2 * left] += diff_coeff
        jac[i, u_row, 2 * right] += diff_coeff
        jac[i, v_row, 2 * g + 1] += -2.0 * diff_coeff
        jac[i, v_row, 2 * left + 1] += diff_coeff
        jac[i, v_row, 2 * right + 1] += diff_coeff


@cuda.jit(device=True)
def jac_fn_numba_cuda(y, t, p, jac, i):
    n_vars = jac.shape[1]
    n_grid = n_vars // 2
    for r in range(n_vars):
        for c in range(n_vars):
            jac[i, r, c] = 0.0
    scale = p[i, 0]
    b_eff = scale * B
    dx = L / n_grid
    diff_coeff = ALPHA / (dx * dx)
    for g in range(n_grid):
        left = g - 1
        if left < 0:
            left = n_grid - 1
        right = g + 1
        if right >= n_grid:
            right = 0
        u_idx = 2 * g
        v_idx = u_idx + 1
        u = y[i, u_idx]
        v = y[i, v_idx]
        jac[i, u_idx, u_idx] = 2.0 * u * v - (b_eff + 1.0) - 2.0 * diff_coeff
        jac[i, u_idx, v_idx] = u * u
        jac[i, v_idx, u_idx] = b_eff - 2.0 * u * v
        jac[i, v_idx, v_idx] = -u * u - 2.0 * diff_coeff
        jac[i, u_idx, 2 * left] += diff_coeff
        jac[i, u_idx, 2 * right] += diff_coeff
        jac[i, v_idx, 2 * left + 1] += diff_coeff
        jac[i, v_idx, 2 * right + 1] += diff_coeff


def make_params(size: int, seed: int = 42) -> np.ndarray:
    """Return reaction-scale parameters with +/-20% uniform perturbation."""
    rng = np.random.default_rng(seed)
    return np.array(1.0 + 0.2 * (2.0 * rng.random((size, 1)) - 1.0), dtype=np.float64)


def make_scenario(
    n_grid: int,
    size: int,
    seed: int = 42,
    *,
    divergence: float = 1.0,
    a: float = A,
    b: float = B,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y0_batch, params_batch)`` for the Brusselator system.

    ``divergence=0`` puts every trajectory at the cosine-perturbed equilibrium
    with scale=1. Larger values perturb both the IC and the reaction-rate
    parameter; ``divergence=1`` gives the default spread.
    """
    if not np.isfinite(divergence) or divergence < 0.0:
        raise ValueError("divergence must be finite and non-negative")
    n_vars = 2 * n_grid
    base = _equilibrium(n_grid, a, b)

    rng = np.random.default_rng(seed)
    # Random additive perturbations to u and v fields. Scale of perturbation
    # grows with ``divergence`` so step counts spread across trajectories.
    u_pert = rng.normal(0.0, 0.5 * divergence, size=(size, n_grid))
    v_pert = rng.normal(0.0, 0.5 * divergence, size=(size, n_grid))
    y0 = np.empty((size, n_vars), dtype=np.float64)
    y0[:, 0::2] = base[0::2] + u_pert
    y0[:, 1::2] = base[1::2] + v_pert
    # Keep u, v strictly positive (Brusselator state physical bounds).
    y0 = np.maximum(y0, 1e-6)

    base_params = make_params(size, seed)
    params = np.maximum(1.0 + divergence * (base_params - 1.0), 1e-6).astype(np.float64)
    return y0, params
