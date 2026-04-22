"""
Tests on the Lorenz chaotic system.

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
import pytest

from solvers.rodas5 import solve as rodas5_solve
from solvers.tsit5 import solve as tsit5_solve
from tests.reference_solvers.python.diffrax_kvaerno5 import (
    make_solver as make_kvaerno5_solver,
)
from tests.reference_solvers.python.diffrax_tsit5 import (
    make_solver as make_diffrax_tsit5_solver,
)
from tests.reference_solvers.python.julia_common import (
    JULIA_ENSEMBLE_BACKENDS,
    benchmark_julia_solver,
    julia_backend_id,
)
from tests.reference_solvers.python.julia_rodas5 import (
    make_solver as make_julia_rodas5_solver,
)
from tests.reference_solvers.python.julia_tsit5 import (
    make_solver as make_julia_tsit5_solver,
)

_T_SPAN = (0.0, 5.0)
_T_SPAN = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=jnp.float64)
_ENSEMBLE_SIZES = [10, 100]

# Conservative attractor bounds for ρ ∈ [26.6, 29.4]
_X_MAX = 40.0
_Y_MAX = 40.0
_Z_MIN = -1.0  # z stays non-negative; small slack for numerical noise
_Z_MAX = 65.0


def _make_lorenz_system():
    """Construct the Lorenz system (3D) with σ=10, β=8/3, ρ=p[0]."""
    y0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)

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
        return jnp.zeros_like(y0)

    return {
        "n_vars": 3,
        "ode_fn": ode_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "y0": y0,
    }


def _make_params_batch(size, seed):
    """ρ values centred at 28 with ±5% uniform perturbation (all chaotic)."""
    rng = np.random.default_rng(seed)
    return jnp.array(
        28.0 * (1.0 + 0.05 * (2.0 * rng.random((size, 1)) - 1.0)),
        dtype=jnp.float64,
    )


def _assert_on_attractor(states):
    """Assert that states lie within the known Lorenz attractor bounds for ρ ≈ 28.

    Accepts any shape (..., 3) — works for a single snapshot (N, 3) or the
    full trajectory array (N, n_times, 3) without a loop.
    """
    x, y, z = states[..., 0], states[..., 1], states[..., 2]
    assert np.all(np.abs(x) < _X_MAX), (
        f"x left attractor: max |x| = {np.abs(x).max():.2f}"
    )
    assert np.all(np.abs(y) < _Y_MAX), (
        f"y left attractor: max |y| = {np.abs(y).max():.2f}"
    )
    assert np.all(z > _Z_MIN), f"z below attractor: min z = {z.min():.2f}"
    assert np.all(z < _Z_MAX), f"z above attractor: max z = {z.max():.2f}"


def _run_julia_lorenz(benchmark, solver_factory, ensemble_size, ensemble_backend):
    system = _make_lorenz_system()
    params = _make_params_batch(ensemble_size, seed=42)
    solve = solver_factory(
        "lorenz",
        system_config={},
        ensemble_backend=ensemble_backend,
    )
    t_span = jnp.array(list(_T_SPAN), dtype=jnp.float64)
    return benchmark_julia_solver(
        benchmark,
        solve,
        y0=system["y0"],
        t_span=t_span,
        params=params,
        first_step=1e-4,
        rtol=1e-6,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_tsit5(benchmark, ensemble_size):
    """Tsit5 nonlinear ensemble benchmark on the Lorenz system."""
    system = _make_lorenz_system()
    params = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: tsit5_solve(
            system["ode_fn"],
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results))


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5(benchmark, ensemble_size, lu_precision):
    """Rodas5 nonlinear ensemble benchmark on the Lorenz system."""
    system = _make_lorenz_system()
    params = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_solve(
            system["ode_fn"],
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            lu_precision=lu_precision,
            first_step=1e-4,
            rtol=1e-10,
            atol=1e-12,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results))


# ---------------------------------------------------------------------------
# Reference solver timings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_tsit5(benchmark, ensemble_size):
    """Diffrax Tsit5 benchmark with attractor-confinement validation."""
    system = _make_lorenz_system()
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_diffrax_tsit5_solver(system["ode_fn"])
    t_span = jnp.array(list(_T_SPAN), dtype=jnp.float64)
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=t_span,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results[:, -1, :]))


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, ensemble_size):
    """Diffrax Kvaerno5 benchmark with attractor-confinement validation."""
    system = _make_lorenz_system()
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_kvaerno5_solver(system["ode_fn"])
    t_span = jnp.array(list(_T_SPAN), dtype=jnp.float64)
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=t_span,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results[:, -1, :]))


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5(benchmark, ensemble_size, ensemble_backend):
    """Julia Tsit5 benchmark with attractor-confinement validation."""
    results_np = _run_julia_lorenz(
        benchmark, make_julia_tsit5_solver, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_T_SPAN), 3)
    assert np.all(np.isfinite(results_np))
    _assert_on_attractor(results_np[:, -1, :])


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark with attractor-confinement validation."""
    results_np = _run_julia_lorenz(
        benchmark, make_julia_rodas5_solver, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_T_SPAN), 3)
    assert np.all(np.isfinite(results_np))
    _assert_on_attractor(results_np[:, -1, :])
