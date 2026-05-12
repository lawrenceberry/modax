"""Solver benchmarks and correctness checks for the heat reference system."""

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
from reference.systems.python import heat
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.rodas5 import solve as rodas5_solve

_TIMES = heat.TIMES
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]


def _dim_id(n_vars):
    return f"{n_vars}d"


def _run_julia_heat(benchmark, solver, heat_system, ensemble_size, ensemble_backend):
    system = heat_system
    params = heat.make_params(ensemble_size, seed=42)
    results_np = benchmark_julia_solver(
        benchmark,
        solver,
        "heat",
        y0=system["y0"],
        t_span=_TIMES,
        params=params,
        system_config={"n_vars": system["n_vars"]},
        ensemble_backend=ensemble_backend,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )
    return system, results_np, params


@pytest.fixture
def heat_system(request):
    """Configurable heat equation system parameterized by grid dimension."""
    return heat.make_system(request.param)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5(benchmark, heat_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with exact-solution validation."""
    system = heat_system
    params = heat.make_params(ensemble_size, seed=42)
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
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_kencarp5(benchmark, heat_system, ensemble_size):
    """KenCarp5 nonlinear benchmark with exact-solution validation."""
    system = heat_system
    params = heat.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: kencarp5_solve(
            system["explicit_ode_fn"],
            system["implicit_ode_fn"],
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            linear=True,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Reference solver timings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, heat_system, ensemble_size):
    """Diffrax KenCarp5 benchmark with exact-solution validation."""
    system = heat_system
    params = heat.make_params(ensemble_size, seed=42)
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
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size",
    [
        pytest.param(n, marks=pytest.mark.slow) if n >= 10000 else n
        for n in _ENSEMBLE_SIZES
    ],
)
def test_diffrax_kvaerno5(benchmark, heat_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark with exact-solution validation."""
    system = heat_system
    params = heat.make_params(ensemble_size, seed=42)
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
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark,
        julia_kencarp5_solve,
        heat_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark,
        julia_rodas5_solve,
        heat_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark,
        julia_kvaerno5_solve,
        heat_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = heat.exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)
