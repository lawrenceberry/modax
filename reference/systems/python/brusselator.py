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

from reference.systems.python._tuple_codegen import make_matrix_callback, make_tuple_callback

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

    explicit_values = []
    implicit_values = []
    ode_values = []
    n_vars = 2 * n_grid
    implicit_jac_rows = [["0.0" for _ in range(n_vars)] for _ in range(n_vars)]
    jac_rows = [["0.0" for _ in range(n_vars)] for _ in range(n_vars)]
    for g in range(n_grid):
        left = (g - 1) % n_grid
        right = (g + 1) % n_grid
        u = 2 * g
        v = u + 1
        u_left = 2 * left
        u_right = 2 * right
        v_left = u_left + 1
        v_right = u_right + 1
        u2v = f"y[{u}] * y[{u}] * y[{v}]"
        exp_u = f"p[0] * {a:.17g} + {u2v} - (p[0] * {b:.17g} + 1.0) * y[{u}]"
        exp_v = f"p[0] * {b:.17g} * y[{u}] - {u2v}"
        imp_u = f"{diff_coeff:.17g} * (y[{u_left}] - 2.0 * y[{u}] + y[{u_right}])"
        imp_v = f"{diff_coeff:.17g} * (y[{v_left}] - 2.0 * y[{v}] + y[{v_right}])"
        explicit_values.extend([exp_u, exp_v])
        implicit_values.extend([imp_u, imp_v])
        ode_values.extend([f"({exp_u}) + ({imp_u})", f"({exp_v}) + ({imp_v})"])

        for row, self_col, left_col, right_col in (
            (u, u, u_left, u_right),
            (v, v, v_left, v_right),
        ):
            coeffs: dict[int, list[str]] = {self_col: [f"-2.0 * {diff_coeff:.17g}"]}
            coeffs.setdefault(left_col, []).append(f"{diff_coeff:.17g}")
            coeffs.setdefault(right_col, []).append(f"{diff_coeff:.17g}")
            for col, terms in coeffs.items():
                implicit_jac_rows[row][col] = " + ".join(terms)
                jac_rows[row][col] = " + ".join(terms)

        jac_rows[u][u] = f"({jac_rows[u][u]}) + 2.0 * y[{u}] * y[{v}] - (p[0] * {b:.17g} + 1.0)"
        jac_rows[u][v] = f"({jac_rows[u][v]}) + y[{u}] * y[{u}]"
        jac_rows[v][u] = f"({jac_rows[v][u]}) + p[0] * {b:.17g} - 2.0 * y[{u}] * y[{v}]"
        jac_rows[v][v] = f"({jac_rows[v][v]}) - y[{u}] * y[{u}]"

    explicit_ode_fn = make_tuple_callback("explicit_ode_fn", [], explicit_values)
    implicit_ode_fn = make_tuple_callback("implicit_ode_fn", [], implicit_values)
    ode_fn = make_tuple_callback("ode_fn", [], ode_values)
    implicit_jac_fn = make_matrix_callback("implicit_jac_fn", [], implicit_jac_rows)
    jac_fn = make_matrix_callback("jac_fn", [], jac_rows)

    return explicit_ode_fn, implicit_ode_fn, ode_fn, y0, implicit_jac_fn, jac_fn


(
    explicit_ode_fn,
    implicit_ode_fn,
    ode_fn,
    _,
    implicit_jac_fn,
    jac_fn,
) = make_system(N_GRID)


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
