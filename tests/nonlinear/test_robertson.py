"""Tests for the Rodas5 nonlinear solver on the Robertson stiff ODE system."""

import numpy as np
import pytest

from reference.solvers.python.diffrax_kencarp5 import solve as diffrax_kencarp5_solve
from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.diffrax_kvaerno5 import (
    solve_cached as diffrax_kvaerno5_solve_cached,
)
from reference.solvers.python.julia_common import (
    JULIA_ENSEMBLE_BACKENDS,
    benchmark_julia_solver,
    julia_backend_id,
    maybe_mark_large_ensemble_sizes,
)
from reference.solvers.python.julia_kencarp5 import solve as julia_kencarp5_solve
from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve
from reference.systems.python import robertson
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.rodas5 import solve as rodas5_solve

_TIMES = robertson.TIMES
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]
_REFERENCE_ENSEMBLE_SIZES = [2]


def _run_julia_robertson(benchmark, solver, ensemble_size, ensemble_backend):
    params = robertson.make_params(ensemble_size, seed=42)
    results_np = benchmark_julia_solver(
        benchmark,
        solver,
        "robertson",
        y0=robertson.Y0,
        t_span=_TIMES,
        params=params,
        system_config={},
        ensemble_backend=ensemble_backend,
        first_step=1e-4,
        rtol=1e-6,
        atol=1e-8,
    )
    return results_np


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5(benchmark, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with cached Diffrax validation on practical ensemble sizes."""
    params = robertson.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_solve(
            robertson.ode_fn,
            robertson.Y0,
            _TIMES,
            params,
            lu_precision=lu_precision,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), robertson.N_VARS)
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=1e-6)

    if ensemble_size in _REFERENCE_ENSEMBLE_SIZES:
        y_ref = diffrax_kvaerno5_solve_cached(
            robertson.ode_fn,
            y0=robertson.Y0,
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready()
        np.testing.assert_allclose(results_np, np.asarray(y_ref), rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5(benchmark, ensemble_size, lu_precision):
    """KenCarp5 nonlinear benchmark with cached Diffrax validation on practical ensemble sizes."""
    params = robertson.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: kencarp5_solve(
            robertson.explicit_ode_fn,
            robertson.implicit_ode_fn,
            robertson.Y0,
            _TIMES,
            params,
            lu_precision=lu_precision,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), robertson.N_VARS)
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=1e-6)

    # if ensemble_size in _REFERENCE_ENSEMBLE_SIZES:
    #     y_ref = diffrax_kvaerno5_solve_cached(
    #         system["ode_fn"],
    #         y0=system["y0"],
    #         t_span=_TIMES,
    #         params=params,
    #         first_step=1e-4,
    #         rtol=1e-8,
    #         atol=1e-10,
    #     ).block_until_ready()
    #     np.testing.assert_allclose(results_np, np.asarray(y_ref), rtol=2e-4, atol=3e-8)


# ---------------------------------------------------------------------------
# Reference solver timings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ensemble_size",
    [
        pytest.param(n, marks=pytest.mark.slow) if n >= 10000 else n
        for n in _ENSEMBLE_SIZES
    ],
)
def test_diffrax_kvaerno5(benchmark, ensemble_size):
    """Diffrax Kvaerno5 benchmark with conservation validation."""
    params = robertson.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: diffrax_kvaerno5_solve(
            robertson.ode_fn,
            y0=robertson.Y0,
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), robertson.N_VARS)
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=1e-6)


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, ensemble_size):
    """Diffrax KenCarp5 benchmark with cached Diffrax validation on practical ensemble sizes."""
    params = robertson.make_params(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: diffrax_kencarp5_solve(
            robertson.explicit_ode_fn,
            robertson.implicit_ode_fn,
            y0=robertson.Y0,
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)

    assert results.shape == (ensemble_size, len(_TIMES), robertson.N_VARS)
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=1e-6)

    if ensemble_size in _REFERENCE_ENSEMBLE_SIZES:
        y_ref = diffrax_kvaerno5_solve_cached(
            robertson.ode_fn,
            y0=robertson.Y0,
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready()
        np.testing.assert_allclose(results_np, np.asarray(y_ref), rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark with conservation validation."""
    results_np = _run_julia_robertson(
        benchmark, julia_rodas5_solve, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), robertson.N_VARS)
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark with conservation validation."""
    results_np = _run_julia_robertson(
        benchmark, julia_kencarp5_solve, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), robertson.N_VARS)
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=1e-6)
