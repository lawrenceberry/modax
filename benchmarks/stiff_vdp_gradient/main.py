"""Gradient benchmark on the coupled VDP lattice: ``value_and_grad`` against dimension.

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 identical trajectories, and times the value *and* the
gradient of the summed final state with respect to every trajectory's
parameter -- the lattice has one, the damping scale -- rather than the solve
alone. Five solvers:

- modax Rodas5P under ``jax.value_and_grad``, which integrates the forward
  sensitivities jointly with the state in one kernel launch; once handed the
  lattice's Jacobian sparsity pattern and once without it.
- Diffrax Kvaerno5 under ``jax.value_and_grad``, reverse mode through the
  solver with its default ``RecursiveCheckpointAdjoint``.
- gradsolve Rosenbrock23's record-and-replay adjoint, up to the 12 state
  variables gradsolve launches the kernel for: its fused NVIDIA Warp kernel
  records the step sizes it accepts, and ``jax.value_and_grad`` differentiates
  a fixed-length ``lax.scan`` replay of them. It is timed twice: the record
  together with the replay, which is what a gradient at new parameters costs,
  since the recorded mesh is only valid near the parameters it was recorded
  at; and the replay alone, the cost "once a record exists" that gradsolve's
  paper reports.
- Julia Rodas5P on both DiffEqGPU ensemble backends, integrating the
  ``vdp_sens`` system: the lattice augmented with its hand-derived forward
  sensitivity equations, since DiffEqGPU differentiates neither backend. Its
  time is the augmented solve; reading the gradient off the sensitivities at
  the final time is not counted.

A point that fails, or does not compile and solve within the case timeout, is
recorded as such and omitted from the plot. Outputs a CSV and a log-log plot
named after the GPU.

Usage:
    uv run python benchmarks/stiff_vdp_gradient/main.py
"""

import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Sets JAX's GPU memory policy before the first import that touches the card:
# the `reference.systems.python` modules build device arrays at import time,
# which initialises the backend and fixes that policy for the whole process.
import benchmarks.benchmark_common  # noqa: E402,F401
from benchmarks._sweep import (
    DIMENSION,
    IDENTICAL_DIVERGENCE,
    Problem,
    SweepBenchmark,
    SweepCase,
    main,
)
from benchmarks.benchmark_common import (  # noqa: E402
    DIFFRAX_COLOR,
    GRADSOLVE_COLOR,
    JULIA_COLOR,
    MODAX_COLOR,
)
from modax.rodas5P import solve as rodas5P_solve
from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.gradsolve_engines import (
    ROSENBROCK23_MAX_DIM,
    make_value_and_grad,
)
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import vdp

jax.config.update("jax_enable_x64", True)

_ENSEMBLE_SIZE = 1000
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}


def _problem(dim: int) -> Problem:
    n_osc = dim // 2
    ode_fn, _ = vdp.make_system(n_osc)
    y0, params = vdp.make_scenario(
        n_osc, _ENSEMBLE_SIZE, divergence=IDENTICAL_DIVERGENCE
    )
    # The Julia system carries the sensitivities as state, starting from zero.
    julia_y0 = np.concatenate([y0, np.zeros_like(y0)], axis=1)
    return Problem(
        ode_fn,
        y0,
        params,
        julia_system_config={"n_osc": n_osc},
        sparsity=vdp.make_sparsity(n_osc),
        julia_y0=julia_y0,
    )


def _gradsolve(engine: str, *, record_each_call: bool):
    """A ``SweepCase.make_run`` timing gradsolve's record-and-replay gradient."""

    def make_run(problem: Problem, t_span, **kwargs):
        return make_value_and_grad(
            problem.ode_fn,
            problem.y0,
            t_span,
            problem.params,
            engine=engine,
            record_each_call=record_each_call,
            **kwargs,
        )

    return make_run


def _skip(case: SweepCase, dim: int) -> str | None:
    if case.key.startswith("gradsolve rosenbrock23") and dim > ROSENBROCK23_MAX_DIM:
        return (
            "gradsolve launches its Rosenbrock23 kernel only up to "
            f"{ROSENBROCK23_MAX_DIM} state variables"
        )
    return None


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title="Rodas5P value and gradient — coupled VDP lattice",
    axis=DIMENSION,
    values=(2, 4, 6, 8, 10, 12, 16, 32, 64, 96, 128),
    t_span=vdp.TIMES,
    make_problem=_problem,
    julia_solve=julia_rodas5P_solve,
    julia_system="vdp_sens",
    skip=_skip,
    legend_loc="upper center",
    legend_ncol=2,
    legend_bbox_to_anchor=(0.5, -0.2),
    cases=(
        SweepCase(
            key="modax rodas5P fp32 (sparse)",
            color=MODAX_COLOR,
            marker="P",
            linestyle="--",
            mode="jax_grad",
            solve_fn=rodas5P_solve,
            kwargs=_SOLVER_KWARGS,
            sparse=True,
        ),
        SweepCase(
            key="modax rodas5P fp32 (dense)",
            color=MODAX_COLOR,
            marker="X",
            mode="jax_grad",
            solve_fn=rodas5P_solve,
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="diffrax kvaerno5",
            color=DIFFRAX_COLOR,
            marker="s",
            mode="jax_grad",
            solve_fn=diffrax_kvaerno5_solve,
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="gradsolve rosenbrock23",
            color=GRADSOLVE_COLOR,
            marker="o",
            mode="jax_grad",
            make_run=_gradsolve("warp_rosenbrock", record_each_call=True),
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="gradsolve rosenbrock23 (replay only)",
            color=GRADSOLVE_COLOR,
            marker="o",
            linestyle=":",
            mode="jax_grad",
            make_run=_gradsolve("warp_rosenbrock", record_each_call=False),
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="julia rodas5P EnsembleGPUArray",
            color=JULIA_COLOR,
            marker="^",
            mode="julia",
            ensemble_backend="EnsembleGPUArray",
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="julia rodas5P EnsembleGPUKernel",
            color=JULIA_COLOR,
            marker="v",
            linestyle="--",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            kwargs=_SOLVER_KWARGS,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
