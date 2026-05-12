"""Solver benchmarks and correctness checks for the Kaps reference system."""

import numpy as np
import pytest

from reference.solvers.python.diffrax_kencarp5 import solve as diffrax_kencarp5_solve
from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_common import (
    JULIA_ENSEMBLE_BACKENDS,
    benchmark_julia_solver,
    julia_backend_id,
    maybe_mark_large_ensemble_sizes,
)
from reference.solvers.python.julia_kencarp5 import solve as julia_kencarp5_solve
from reference.solvers.python.julia_kvaerno5 import solve as julia_kvaerno5_solve
from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve
from reference.systems.python import kaps
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.rodas5 import solve as rodas5_solve

_TIMES = kaps.TIMES
_N_PAIRS = [15, 25, 35]  # equation pairs → 30D, 50D, 70D
_EPSILON_MIN = [1e-2, 1e-4, 1e-6]  # smallest ε: stiffness ratio ≈ 1/ε_min
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]


@pytest.fixture
def kaps_system(request):
    """Kaps system parameterized by (n_pairs, epsilon_min)."""
    n_pairs, epsilon_min = request.param
    return kaps.make_system(n_pairs, epsilon_min)


def _run_julia_kaps(benchmark, solver, kaps_system, ensemble_size, ensemble_backend):
    system = kaps_system
    params = kaps.make_params(ensemble_size, seed=42)
    results_np = benchmark_julia_solver(
        benchmark,
        solver,
        "kaps",
        y0=system["y0"],
        t_span=_TIMES,
        params=params,
        system_config={
            "n_pairs": system["n_pairs"],
            "epsilon_min": system["epsilon_min"],
        },
        ensemble_backend=ensemble_backend,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )
    return system, results_np, params


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5(benchmark, kaps_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with exact-solution validation."""
    system = kaps_system
    params = kaps.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_solve(
            system["ode_fn"],
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            lu_precision=lu_precision,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_kencarp5(benchmark, kaps_system, ensemble_size):
    """KenCarp5 nonlinear benchmark with exact-solution validation."""
    system = kaps_system
    params = kaps.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: kencarp5_solve(
            system["explicit_ode_fn"],
            system["implicit_ode_fn"],
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, kaps_system, ensemble_size):
    """Diffrax KenCarp5 benchmark on coupled Kaps singular-perturbation systems."""
    system = kaps_system
    params = kaps.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: diffrax_kencarp5_solve(
            system["explicit_ode_fn"],
            system["implicit_ode_fn"],
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Reference solver timings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, kaps_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark on coupled Kaps singular-perturbation systems."""
    system = kaps_system
    params = kaps.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: diffrax_kvaerno5_solve(
            system["ode_fn"],
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark,
        julia_kencarp5_solve,
        kaps_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark,
        julia_rodas5_solve,
        kaps_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark,
        julia_kvaerno5_solve,
        kaps_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = kaps.exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)
