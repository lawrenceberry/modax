"""Tests on the Lorenz chaotic system."""

import jax.numpy as jnp
import numpy as np
import pytest

from solvers.rodas5 import solve as rodas5_solve
from solvers.tsit5 import solve as tsit5_solve
from reference.solvers.python.diffrax_kvaerno5 import (
    make_solver as make_kvaerno5_solver,
)
from reference.solvers.python.diffrax_tsit5 import (
    make_solver as make_diffrax_tsit5_solver,
)
from reference.solvers.python.julia_common import (
    JULIA_ENSEMBLE_BACKENDS,
    benchmark_julia_solver,
    julia_backend_id,
)
from reference.solvers.python.julia_rodas5 import (
    make_solver as make_julia_rodas5_solver,
)
from reference.solvers.python.julia_tsit5 import (
    make_solver as make_julia_tsit5_solver,
)
from reference.systems.python import lorenz

_T_SPAN = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=jnp.float64)
_ENSEMBLE_SIZES = [10, 100]

# Conservative attractor bounds for ρ ∈ [26.6, 29.4]
_X_MAX = 40.0
_Y_MAX = 40.0
_Z_MIN = -1.0  # z stays non-negative; small slack for numerical noise
_Z_MAX = 65.0


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
    params = lorenz.make_params(ensemble_size, seed=42)
    solve = solver_factory(
        "lorenz",
        system_config={},
        ensemble_backend=ensemble_backend,
    )
    return benchmark_julia_solver(
        benchmark,
        solve,
        y0=lorenz.Y0,
        t_span=_T_SPAN,
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
    params = lorenz.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: tsit5_solve(
            lorenz.ode_fn,
            y0=lorenz.Y0,
            t_span=_T_SPAN,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), lorenz.N_VARS)
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results))


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5(benchmark, ensemble_size, lu_precision):
    """Rodas5 nonlinear ensemble benchmark on the Lorenz system."""
    params = lorenz.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_solve(
            lorenz.ode_fn,
            y0=lorenz.Y0,
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

    assert results.shape == (ensemble_size, len(_T_SPAN), lorenz.N_VARS)
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results))


# ---------------------------------------------------------------------------
# Reference solver timings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_tsit5(benchmark, ensemble_size):
    """Diffrax Tsit5 benchmark with attractor-confinement validation."""
    params = lorenz.make_params(ensemble_size, seed=42)
    solve = make_diffrax_tsit5_solver(lorenz.ode_fn)
    results = benchmark.pedantic(
        lambda: solve(
            y0=lorenz.Y0,
            t_span=_T_SPAN,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), lorenz.N_VARS)
    assert np.all(np.isfinite(results))
    _assert_on_attractor(np.asarray(results[:, -1, :]))


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, ensemble_size):
    """Diffrax Kvaerno5 benchmark with attractor-confinement validation."""
    params = lorenz.make_params(ensemble_size, seed=42)
    solve = make_kvaerno5_solver(lorenz.ode_fn)
    results = benchmark.pedantic(
        lambda: solve(
            y0=lorenz.Y0,
            t_span=_T_SPAN,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), lorenz.N_VARS)
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
    assert results_np.shape == (ensemble_size, len(_T_SPAN), lorenz.N_VARS)
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
    assert results_np.shape == (ensemble_size, len(_T_SPAN), lorenz.N_VARS)
    assert np.all(np.isfinite(results_np))
    _assert_on_attractor(results_np[:, -1, :])
