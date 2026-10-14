"""Robertson divergence-CV benchmark for Rodas5P solvers.

Runs the Robertson system with 30,000 trajectories while sweeping the
``make_scenario(..., divergence=...)`` knob. For each solver and divergence
value, the benchmark records solve time and the actual distribution of accepted
plus rejected Rodas5P steps. gradsolve Rosenbrock23, its fused NVIDIA Warp
kernel, steps each trajectory on its own thread, as modax does, but caps every
trajectory at 50000 attempts and returns NaN for one that needs more, which the
reference raises on, so its points at large divergence are recorded as
failures. Like modax it is also run with the ensemble pre-sorted by modax's
attempted steps; its own step counts differ, being a second-order method, but
rank the trajectories much the same way.

Usage:
    uv run python benchmarks/stiff_robertson_divergence/main.py
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Sets JAX's GPU memory policy before the first import that touches the card:
# the `reference.systems.python` modules build device arrays at import time,
# which initialises the backend and fixes that policy for the whole process.
import benchmarks.benchmark_common  # noqa: E402,F401
from benchmarks._divergence import DivergenceBenchmark, DivergenceCase, main
from benchmarks.benchmark_common import (  # noqa: E402
    GRADSOLVE_COLOR,
    JULIA_COLOR,
    MODAX_COLOR,
)
from modax.rodas5P import solve as rodas5P_solve
from reference.solvers.python.gradsolve_engines import solve as gradsolve_solve
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import robertson

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 30_000
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

BENCHMARK = DivergenceBenchmark(
    script_dir=Path(__file__).resolve().parent,
    system="Robertson",
    title="Robertson Rodas5P divergence",
    solve=rodas5P_solve,
    ode_fn=robertson.ode_fn,
    t_span=robertson.TIMES,
    dim=robertson.N_VARS,
    n_traj=_N_TRAJ,
    divergences=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    solver_kwargs=_SOLVER_KWARGS,
    make_data=lambda divergence: robertson.make_scenario(
        _N_TRAJ, seed=42, divergence=divergence
    ),
    julia_solve=julia_rodas5P_solve,
    julia_system="robertson",
    legend_loc="upper center",
    legend_ncol=2,
    legend_bbox_to_anchor=(0.5, -0.12),
    cases=(
        DivergenceCase(key="modax rodas5P fp32", color=MODAX_COLOR, marker="s"),
        DivergenceCase(
            key="modax rodas5P fp32 (sorted)",
            color=MODAX_COLOR,
            marker="P",
            sort_by_steps=True,
        ),
        # DivergenceCase(
        #     key="diffrax kvaerno5",
        #     color=DIFFRAX_COLOR,
        #     marker="^",
        #     mode="timing",
        #     solve_fn=diffrax_kvaerno5_solve,
        #     kwargs={**_SOLVER_KWARGS, "max_steps": 1_000_000},
        #     max_divergence=1.5,
        # ),
        # DivergenceCase(
        #     key="julia rodas5P EnsembleGPUArray",
        #     color=JULIA_COLOR,
        #     marker="D",
        #     mode="julia",
        #     ensemble_backend="EnsembleGPUArray",
        # ),
        DivergenceCase(
            key="gradsolve rosenbrock23",
            color=GRADSOLVE_COLOR,
            marker="o",
            mode="timing",
            solve_fn=gradsolve_solve,
            kwargs={**_SOLVER_KWARGS, "engine": "warp_rosenbrock"},
            jit=False,
        ),
        DivergenceCase(
            key="gradsolve rosenbrock23 (sorted)",
            color=GRADSOLVE_COLOR,
            marker="P",
            mode="timing",
            solve_fn=gradsolve_solve,
            kwargs={**_SOLVER_KWARGS, "engine": "warp_rosenbrock"},
            jit=False,
            sort_by_steps=True,
        ),
        DivergenceCase(
            key="julia rodas5P EnsembleGPUKernel",
            color=JULIA_COLOR,
            marker="v",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            max_divergence=2.5,
        ),
        DivergenceCase(
            key="julia rodas5P EnsembleGPUKernel (sorted)",
            color=JULIA_COLOR,
            marker="X",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            sort_by_steps=True,
            max_divergence=2.5,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
