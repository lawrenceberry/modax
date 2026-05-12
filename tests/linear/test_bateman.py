"""Tests for Rodas5 linear and nonlinear solvers on Bateman radioactive decay chains.

Physical background
-------------------
The Bateman equations (H. Bateman, 1910) describe a sequential radioactive decay
chain A₁ → A₂ → … → Aₙ, where each species Aᵢ decays into Aᵢ₊₁ with first-order
rate constant λᵢ.  The final species Aₙ is stable (no further decay).  Starting
from a pure parent population — N₁(0) = 1, all others zero — the system evolves as
the parent feeds the chain, each intermediate species is produced by its predecessor
and destroyed by its own decay, and the stable end product accumulates irreversibly.

ODE system
----------
    dN₁/dt =                  −λ₁ N₁
    dNᵢ/dt = λᵢ₋₁ Nᵢ₋₁     − λᵢ Nᵢ    (i = 2, …, n − 1)
    dNₙ/dt = λₙ₋₁ Nₙ₋₁                  (stable end product)

The coefficient matrix M is lower bidiagonal: diagonal entries −λᵢ, subdiagonal
entries +λᵢ.  A global rate-scale factor p[0] ≈ 1 ± 10% multiplies M as the
ensemble parameter, so the ensemble explores slightly different reaction speeds.

Total population is conserved: Σᵢ Nᵢ(t) = 1 for all t.

Analytical solution
-------------------
Because M is constant the exact solution is N(t) = expm(p[0] · M · t) · N₀, where
expm denotes the matrix exponential.  This is equivalent to the classic Bateman
formula (a sum of exponentials), but computed via scipy.linalg.expm to avoid the
catastrophic cancellation that affects the direct formula for long chains with
widely separated decay constants.  This exact solution is used for correctness
validation in the tests below.

Stiffness character — CONSTANT
-------------------------------
The Jacobian of the Bateman system equals p[0] · M identically — it is independent
of both t and the state y.  The stiffness ratio is therefore fixed at λ_max / λ_min
throughout the entire integration; it does not grow or shrink with time.  The
``stiffness`` parameter sets this ratio directly.

An explicit solver requires Δt < 2 / λ_max to remain stable; a step of that size
over an integration window of O(1 / λ_min) requires O(stiffness) steps.  An
implicit L-stable solver such as Rodas5 can take O(1) steps regardless of stiffness.

What this tests
---------------
- Rodas5 linear solver (jac_fn path): validates that the constant-Jacobian fast path
  produces accurate results across three chain lengths and three stiffness ratios.
- Rodas5 nonlinear solver (ode_fn path): validates the JVP-based Jacobian on a
  problem with a known exact solution.
- Diffrax Kvaerno5: provides a timing baseline for comparison.
"""

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
from reference.systems.python import bateman
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.rodas5 import solve as rodas5_solve

_TIMES = bateman.TIMES
_SYSTEM_DIMS = [30, 50, 70]  # chain lengths (species count including stable)
_STIFFNESS_RATIOS = [1e2, 1e4, 1e6]  # λ_max / λ_min
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]


@pytest.fixture
def bateman_system(request):
    """Bateman decay-chain system parameterized by (n_vars, stiffness)."""
    n_vars, stiffness = request.param
    return bateman.make_system(n_vars, stiffness)


def _system_id(p):
    return f"{p[0]}vars-stiff1e{int(round(np.log10(p[1])))}"


def _run_julia_bateman(
    benchmark, solver, bateman_system, ensemble_size, ensemble_backend
):
    system = bateman_system
    params = bateman.make_params(ensemble_size, seed=42)
    results_np = benchmark_julia_solver(
        benchmark,
        solver,
        "bateman",
        y0=system["y0"],
        t_span=_TIMES,
        params=params,
        system_config={
            "n_vars": system["n_vars"],
            "stiffness": system["stiffness"],
        },
        ensemble_backend=ensemble_backend,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )
    return system, results_np, params


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5(benchmark, bateman_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with conservation and exact-solution validation."""
    system = bateman_system
    params = bateman.make_params(ensemble_size, seed=42)
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_kencarp5(benchmark, bateman_system, ensemble_size):
    """KenCarp5 nonlinear benchmark with conservation and exact-solution validation."""
    system = bateman_system
    params = bateman.make_params(ensemble_size, seed=42)
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

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, bateman_system, ensemble_size):
    """Diffrax KenCarp5 benchmark on Bateman decay chains."""
    system = bateman_system
    params = bateman.make_params(ensemble_size, seed=42)
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Reference solver timings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, bateman_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark on Bateman decay chains."""
    system = bateman_system
    params = bateman.make_params(ensemble_size, seed=42)
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark,
        julia_kencarp5_solve,
        bateman_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark,
        julia_rodas5_solve,
        bateman_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark,
        julia_kvaerno5_solve,
        bateman_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = bateman.exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)
