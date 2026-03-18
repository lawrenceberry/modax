"""Benchmarks for 72D CMB-inspired Boltzmann ODE system.

72-dimensional ODE with time-varying stiffness mimicking CMB Einstein-Boltzmann
equations. The system consists of coupled damped oscillators with sinusoidal
solutions at varying frequencies:

- y[0]: clock variable (dy/dt = 1, so y[0] = t)
- y[1..36]: 18 non-stiff oscillator pairs (constant damping 0.01)
- y[37..70]: 17 stiff oscillator pairs (damping = p[0] * clock, grows with time)
- y[71]: single stiff exponential decay mode (rate = p[1] * clock)

At t=0 all modes are non-stiff. As clock advances, the stiff modes acquire
large damping (stiffness ratio ~10,000 by t=10), mimicking the tight-coupling
regime in CMB Boltzmann solvers.
"""

import json
import math
import shutil
import subprocess

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.rodas5 import solve_ensemble as rodas5_solve_ensemble
from solvers.rodas5_custom_kernel_v2 import make_solver as make_rodas5_v2_solver

_JULIA_SCRIPT = "benchmarks/boltzmann_72d_julia.jl"
_HAS_JULIA = shutil.which("julia") is not None

_N_VARS = 72
_N_NONSTIFF_PAIRS = 18
_N_STIFF_PAIRS = 17
_NONSTIFF_DAMPING = 0.01
_T_SPAN = (0.0, 10.0)
_BASE_PARAMS = jnp.array([10.0, 50.0])


def _run_julia(n=2, rtol=1e-6, atol=1e-8, timeout=600):
    """Run the Julia GPU ensemble 72D benchmark and return parsed JSON result."""
    cmd = ["julia", _JULIA_SCRIPT, str(n), str(rtol), str(atol)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Julia failed:\n{result.stderr}")
    return json.loads(result.stdout.strip())


def _boltzmann_72d_ode(y, p):
    """72D CMB-inspired ODE with time-varying stiffness.

    Works with both tuples (Pallas kernel) and JAX arrays (vmap solver).
    Uses only element-wise ops and Python-level indexing.
    """
    clock = y[0]
    ones = clock * 0.0 + 1.0

    dy = [None] * _N_VARS
    dy[0] = ones  # clock

    # 18 non-stiff oscillator pairs (k=0..17)
    for k in range(_N_NONSTIFF_PAIRS):
        i1, i2 = 2 * k + 1, 2 * k + 2
        omega = 2.0 * math.pi * (k + 1)
        dy[i1] = -_NONSTIFF_DAMPING * y[i1] + omega * y[i2]
        dy[i2] = -omega * y[i1] - _NONSTIFF_DAMPING * y[i2]

    # 17 stiff oscillator pairs (k=18..34)
    for k in range(_N_NONSTIFF_PAIRS, _N_NONSTIFF_PAIRS + _N_STIFF_PAIRS):
        i1, i2 = 2 * k + 1, 2 * k + 2
        damping = p[0] * clock
        omega = 2.0 * math.pi * (k + 1)
        dy[i1] = -damping * y[i1] + omega * y[i2]
        dy[i2] = -omega * y[i1] - damping * y[i2]

    # Single stiff decay mode
    dy[71] = -p[1] * clock * y[71]

    return tuple(dy)


@jax.jit
def _boltzmann_72d_array(y, params):
    """Array-returning wrapper for vmap-based solvers."""
    return jnp.array(_boltzmann_72d_ode(y, params))


# Initial conditions: clock=0, first-in-pair=1, second-in-pair=0, decay=1
_y0_list = [0.0] * _N_VARS
for _k in range(35):
    _y0_list[2 * _k + 1] = 1.0
_y0_list[71] = 1.0
_Y0 = jnp.array(_y0_list)


@pytest.fixture
def params_batch_72d(request):
    """72D parameter sets with +/-10% uniform perturbation."""
    N = request.param
    rng = np.random.default_rng(42)
    base = np.array([10.0, 50.0])
    return jnp.array(base * (1.0 + 0.1 * (2 * rng.random((N, 2)) - 1)))


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------


def test_rodas5_v2_72d_correctness(benchmark):
    """Single-trajectory correctness check against analytical solution.

    Non-stiff oscillators have constant damping 0.01, so at t=10:
      y_{2k+1}(10) = exp(-0.1) * cos(omega_k * 10) = exp(-0.1) (since omega*10 = 20*pi*(k+1))
      y_{2k+2}(10) = 0
    Stiff modes are damped to ~0. Clock = 10.
    """
    solve = make_rodas5_v2_solver(_boltzmann_72d_ode)
    params_batch = _BASE_PARAMS[None, :]
    y0_batch = _Y0[None, :]
    results = benchmark.pedantic(
        lambda: solve(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    y = results[0]
    # Clock should equal tf
    np.testing.assert_allclose(float(y[0]), 10.0, atol=1e-4)
    # Non-stiff first components ~ exp(-0.1), second components ~ 0
    exp_val = np.exp(-0.1)
    for k in range(_N_NONSTIFF_PAIRS):
        np.testing.assert_allclose(float(y[2 * k + 1]), exp_val, rtol=1e-3)
        np.testing.assert_allclose(float(y[2 * k + 2]), 0.0, atol=1e-3)
    # Stiff components ~ 0
    for k in range(_N_NONSTIFF_PAIRS, _N_NONSTIFF_PAIRS + _N_STIFF_PAIRS):
        np.testing.assert_allclose(float(y[2 * k + 1]), 0.0, atol=1e-3)
        np.testing.assert_allclose(float(y[2 * k + 2]), 0.0, atol=1e-3)
    # Decay mode ~ 0
    np.testing.assert_allclose(float(y[71]), 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Ensemble benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params_batch_72d", [2, 100, 1000, 10000, 100_000], indirect=True
)
def test_rodas5_72d_ensemble_N(benchmark, params_batch_72d):
    """Rodas5 vmap ensemble (rodas5.py solve_ensemble)."""
    results = benchmark.pedantic(
        lambda: rodas5_solve_ensemble(
            _boltzmann_72d_array,
            y0=_y0_list,
            t_span=_T_SPAN,
            params_batch=params_batch_72d,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    assert results.shape == (params_batch_72d.shape[0], _N_VARS)
    np.testing.assert_allclose(results[:, 0], 10.0, atol=1e-3)


@pytest.mark.skipif(not _HAS_JULIA, reason="Julia not installed")
@pytest.mark.parametrize("N", [2, 100, 1000, 10000, 100_000])
def test_julia_72d_gpu_ensemble_N(benchmark, N):
    """Julia GPURosenbrock23 + EnsembleGPUKernel (CUDA, Float64)."""
    data = benchmark.pedantic(
        lambda: _run_julia(N, rtol=1e-6, atol=1e-8),
        warmup_rounds=0,
        rounds=1,
    )

    from pytest_benchmark.stats import Stats

    stats = Stats()
    stats.update(data["adaptive_dt"]["min_time_ms"] / 1000)
    benchmark.stats.stats = stats


@pytest.mark.parametrize(
    "params_batch_72d", [2, 100, 1000, 10000, 100_000], indirect=True
)
def test_rodas5_v2_72d_pallas_ensemble_N(benchmark, params_batch_72d):
    """Rodas5 v2 Pallas custom kernel ensemble."""
    solve = make_rodas5_v2_solver(_boltzmann_72d_ode)
    y0_batch = jnp.broadcast_to(_Y0, (params_batch_72d.shape[0], _N_VARS))
    results = benchmark.pedantic(
        lambda: solve(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch_72d,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    assert results.shape == (params_batch_72d.shape[0], _N_VARS)
    np.testing.assert_allclose(results[:, 0], 10.0, atol=1e-3)
