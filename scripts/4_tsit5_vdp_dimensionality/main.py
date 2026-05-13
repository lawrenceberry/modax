"""Dimensionality scaling benchmark on the non-stiff coupled VDP lattice (Tsit5).

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 trajectories and records solve time for the local JAX
Tsit5 solver, the numba-cuda Tsit5 kernel, Diffrax Tsit5, and Julia Tsit5 with
both DiffEqGPU ensemble backends. EnsembleGPUKernel failures (expected for
large dimensions) are stored as null and omitted from the plot. Runs both
"identical" and "divergent" scenarios; outputs a CSV and log-log plot per
scenario, named after the GPU and scenario.

Uses the non-stiff coupled VDP variant (mu = 1.0) so that explicit Tsit5
remains an appropriate solver.

Usage:
    uv run python scripts/4_tsit5_vdp_dimensionality/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.systems.python import vdp
from scripts.benchmark_common import (
    BenchmarkCase,
    collect_timed_timing,
    drop_none_rows,
    format_cached_timing,
    get_gpu_name,
    gpu_slug,
    julia_solve_time_ms,
    load_cache,
    save_cache,
    time_blocked_ms,
    timing_value_or_none,
)
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import clear_caches as tsit5ckn_clear_caches
from solvers.tsit5ckn import prepare_solve as tsit5ckn_prepare_solve
from solvers.tsit5ckn import run_prepared as tsit5ckn_run_prepared

jax.config.update("jax_enable_x64", True)

_MU_NONSTIFF = 1.0
_D = 10.0
_OMEGA = 1.0

_T_SPAN = vdp.TIMES
_ENSEMBLE_SIZE = 1000
_N_RUNS = 10

_DIMENSIONS = (2, 4, 6, 8, 10, 12, 16, 32, 64, 128)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_SCENARIOS = (
    ("identical", 0.0),
    ("divergent", 1.0),
)


@dataclass(frozen=True)
class Case(BenchmarkCase):
    solve_fn: Callable[..., Any] | None = None
    mode: str = "jax"
    t_span: Any = None
    kwargs: dict[str, Any] | None = None
    system_name: str | None = None
    ensemble_backend: str | None = None

    @property
    def is_julia(self) -> bool:
        return self.system_name is not None


CASES: tuple[Case, ...] = (
    Case(
        key="local_tsit5",
        label="my tsit5",
        color="#2b7be0",
        marker="o",
        solve_fn=tsit5_solve,
        mode="jax",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
    ),
    Case(
        key="tsit5ckn",
        label="numba-cuda tsit5",
        color="#f0a202",
        marker="P",
        mode="custom",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
    ),
    Case(
        key="diffrax_tsit5",
        label="diffrax tsit5",
        color="#2ba84a",
        marker="s",
        solve_fn=diffrax_tsit5_solve,
        mode="jax",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
    ),
    Case(
        key="julia_tsit5_EnsembleGPUArray",
        label="julia tsit5 array",
        color="#9b59b6",
        marker="^",
        solve_fn=julia_tsit5_solve,
        mode="julia",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        system_name="vdp",
        ensemble_backend="EnsembleGPUArray",
    ),
    Case(
        key="julia_tsit5_EnsembleGPUKernel",
        label="julia tsit5 kernel",
        color="#9b59b6",
        marker="v",
        linestyle="--",
        solve_fn=julia_tsit5_solve,
        mode="julia",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        system_name="vdp",
        ensemble_backend="EnsembleGPUKernel",
    ),
)


@cuda.jit(device=True)
def ode_fn_vdp_numba(y, t, p, dy, i):
    n_osc = int(p[i, 0])
    scale = p[i, 1]
    mu = 1.0
    diffusion = 10.0
    for k in range(n_osc):
        kp1 = (k + 1) % n_osc
        km1 = (k + n_osc - 1) % n_osc
        xk = y[i, 2 * k]
        vk = y[i, 2 * k + 1]
        lap = y[i, 2 * kp1] - 2.0 * xk + y[i, 2 * km1]
        dy[i, 2 * k] = vk
        dy[i, 2 * k + 1] = scale * mu * (1.0 - xk * xk) * vk - xk + diffusion * lap


def _vdp_system_config(n_osc: int) -> dict[str, float | int]:
    return {"n_osc": n_osc, "mu": _MU_NONSTIFF, "d": _D, "omega": _OMEGA}


def time_case(case: Case, dim: int, *, divergence: float) -> float:
    n_osc = dim // 2
    y0_batch, params = vdp.make_scenario(
        n_osc, _ENSEMBLE_SIZE, divergence=divergence
    )
    kwargs = case.kwargs or {}

    if case.is_julia:
        return julia_solve_time_ms(
            case.solve_fn,
            case.system_name,
            y0_batch,
            case.t_span,
            params,
            system_config=_vdp_system_config(n_osc),
            ensemble_backend=case.ensemble_backend,
            **kwargs,
        )

    if case.mode == "custom":
        custom_params = np.column_stack(
            [
                np.full(_ENSEMBLE_SIZE, float(n_osc), dtype=np.float64),
                params[:, 0],
            ]
        )
        prepared = tsit5ckn_prepare_solve(
            ode_fn_vdp_numba,
            y0=y0_batch,
            t_span=case.t_span,
            params=custom_params,
            **kwargs,
        )

        def run_custom():
            return tsit5ckn_run_prepared(prepared, copy_solution=False)

        try:
            return time_blocked_ms(run_custom, _N_RUNS)
        finally:
            # Release per-dimension workspaces and compiled kernels before the
            # next dimension allocates its own.
            tsit5ckn_clear_caches()

    assert case.solve_fn is not None
    ode_fn, _ = vdp.make_system(n_osc, mu=_MU_NONSTIFF)
    y0_j = jnp.asarray(y0_batch)
    p_j = jnp.asarray(params)

    def run():
        return case.solve_fn(
            ode_fn,
            y0=y0_j,
            t_span=case.t_span,
            params=p_j,
            **kwargs,
        )

    return time_blocked_ms(run, _N_RUNS)


def collect_timing(case: Case, dim: int, divergence: float):
    return collect_timed_timing(
        case.label,
        f"dim={dim:>4}",
        lambda: time_case(case, dim, divergence=divergence),
        label_width=28,
    )


_Row = tuple[str, str, int, float | None]


def run_benchmarks(
    cases: Sequence[Case], gpu_name: str, cache: dict, scenario: str, divergence: float
) -> list[_Row]:
    scenario_cache = cache.setdefault(gpu_name, {}).setdefault(scenario, {})
    rows: list[_Row] = []
    for case in cases:
        print(f"\n{case.label}:")
        solver_cache = scenario_cache.setdefault(case.key, {})
        for dim in _DIMENSIONS:
            dim_key = str(dim)
            if dim_key in solver_cache:
                ms = solver_cache[dim_key]
                ms_text = format_cached_timing(ms)
                print(f"  {case.label:<28} dim={dim:>4} ... (cached) {ms_text}")
            else:
                ms = collect_timing(case, dim, divergence)
                solver_cache[dim_key] = ms
                save_cache(_CACHE_PATH, cache)
            rows.append((case.key, case.label, dim, timing_value_or_none(ms)))
        print()
    return rows


def save_csv(rows: list[_Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "dim", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[_Row],
    cases: Sequence[Case],
    gpu_name: str,
    output_path: Path,
    scenario: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in cases:
        dims, times_ms = drop_none_rows(rows, case.key)
        if not dims:
            continue
        ax.plot(
            dims,
            times_ms,
            marker=case.marker,
            color=case.color,
            linestyle=case.linestyle,
            label=case.label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ODE dimension")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title(
        f"Tsit5 dimensionality — {scenario} — coupled VDP lattice "
        f"(μ={_MU_NONSTIFF}) — {gpu_name}"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_DIMENSIONS)
    ax.set_xticklabels([str(d) for d in _DIMENSIONS], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def _scenario_output_paths(gpu_name: str, scenario: str) -> tuple[Path, Path]:
    slug = gpu_slug(gpu_name)
    return (
        _SCRIPT_DIR / f"results-{slug}-{scenario}.csv",
        _SCRIPT_DIR / f"plot-{slug}-{scenario}.png",
    )


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)

    for scenario, divergence in _SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===\n")
        rows = run_benchmarks(CASES, gpu_name, cache, scenario, divergence)
        csv_path, plot_path = _scenario_output_paths(gpu_name, scenario)
        save_csv(rows, csv_path)
        plot(rows, CASES, gpu_name, plot_path, scenario)


if __name__ == "__main__":
    main()
