"""Dimensionality scaling benchmark for KenCarp5 on the Brusselator system.

Sweeps ODE dimension from 4 to 128 (n_grid = 2 to 64, dim = 2*n_grid) on a
log scale with a fixed ensemble of 1000 trajectories and records solve time
for the local KenCarp5 solver (both ``linear=True`` and ``linear=False``),
Diffrax KenCarp5, and Julia KenCarp5 on ``EnsembleGPUArray``. DiffEqGPU has
no ``GPUKenCarp5`` so the kernel backend is skipped. Failed solves (e.g. at
extreme dims) are stored as null and omitted from the plot. Runs both
``identical`` and ``divergent`` scenarios.

Usage:
    uv run python scripts/14_kencarp5_brusselator_dimensionality/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_kencarp5 import solve as diffrax_kencarp5_solve
from reference.solvers.python.julia_kencarp5 import solve as julia_kencarp5_solve
from reference.systems.python import brusselator
from scripts.benchmark_common import (
    get_gpu_name,
    gpu_slug,
    load_cache,
    save_cache,
    time_blocked,
)
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.kencarp5ckn import solve as kencarp5ckn_solve
from solvers.rodas5 import solve as rodas5_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = brusselator.TIMES
_ENSEMBLE_SIZE = 1000
_N_RUNS = 5
_JULIA_BACKENDS = ("EnsembleGPUArray",)

# Total ODE dimension (= 2 * n_grid). Matches the script 9 VDP sweep shape.
_DIMENSIONS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

# (key, label, color, marker, linear)
_LOCAL_SOLVER_DEFS = [
    ("local_kencarp5_linear", "my kencarp5 linear=True", "#2b7be0", "o", True),
    ("local_kencarp5_newton", "my kencarp5 linear=False", "#e02b2b", "D", False),
]
_CUSTOM_KERNEL_SOLVER_DEFS = [
    ("kencarp5ckn_linear", "numba-cuda kencarp5 linear=True", "#f0a202", "P", True),
    ("kencarp5ckn_newton", "numba-cuda kencarp5 linear=False", "#d35400", "X", False),
]
_RODAS_SOLVER_DEFS = [
    ("local_rodas5_fp64_lu", "my rodas5 fp64 LU", "#00a6a6", "v", "fp64"),
]
# (key, label, color, marker, solve_fn)
_JAX_SOLVER_DEFS = [
    ("diffrax_kencarp5", "diffrax kencarp5", "#2ba84a", "s", diffrax_kencarp5_solve),
]
# (key, label, solve_fn, color)
# DiffEqGPU 3.13 has no SplitODEProblem support on EnsembleGPUArray, so the
# Julia row runs KenCarp5 on the combined (non-split) RHS — fully implicit,
# not IMEX. See the comment block at the top of
# reference/solvers/julia/run_solver.jl for the underlying bug.
_JULIA_SOLVER_DEFS = [
    (
        "julia_kencarp5",
        "julia kencarp5 (fully-implicit)",
        julia_kencarp5_solve,
        "#9b59b6",
    ),
]
_JULIA_BACKEND_STYLES = {
    "EnsembleGPUArray": ("array", "^", "-"),
}


@dataclass(frozen=True)
class SolverSpec:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str
    timing_fn: Callable[[int], float]


def time_local_kencarp5(dim: int, *, linear: bool, scenario: str) -> float:
    n_grid = dim // 2
    ex_fn, im_fn, _, _ = brusselator.make_system(n_grid)
    y0_batch, params = brusselator.make_scenario(scenario, n_grid, _ENSEMBLE_SIZE)
    y0_j = jnp.asarray(y0_batch)
    p_j = jnp.asarray(params)

    def run():
        return kencarp5_solve(
            ex_fn, im_fn, y0_j, _T_SPAN, p_j, linear=linear, **_SOLVER_KWARGS
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_diffrax_kencarp5(dim: int, *, scenario: str) -> float:
    n_grid = dim // 2
    ex_fn, im_fn, _, _ = brusselator.make_system(n_grid)
    y0_batch, params = brusselator.make_scenario(scenario, n_grid, _ENSEMBLE_SIZE)
    y0_j = jnp.asarray(y0_batch)
    p_j = jnp.asarray(params)

    def run():
        return diffrax_kencarp5_solve(
            ex_fn, im_fn, y0_j, _T_SPAN, p_j, **_SOLVER_KWARGS
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_custom_kernel_kencarp5(dim: int, *, linear: bool, scenario: str) -> float:
    n_grid = dim // 2
    y0_batch, params = brusselator.make_scenario(scenario, n_grid, _ENSEMBLE_SIZE)

    def run():
        return kencarp5ckn_solve(
            brusselator.explicit_ode_fn_numba_cuda,
            brusselator.implicit_ode_fn_numba_cuda,
            brusselator.implicit_jac_fn_numba_cuda,
            y0=y0_batch,
            t_span=_T_SPAN,
            params=params,
            linear=linear,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_local_rodas5(dim: int, *, scenario: str, lu_precision: str) -> float:
    n_grid = dim // 2
    _, _, ode_fn, _ = brusselator.make_system(n_grid)
    y0_batch, params = brusselator.make_scenario(scenario, n_grid, _ENSEMBLE_SIZE)
    y0_j = jnp.asarray(y0_batch)
    p_j = jnp.asarray(params)

    def run():
        return rodas5_solve(
            ode_fn,
            y0_j,
            _T_SPAN,
            p_j,
            lu_precision=lu_precision,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_julia_kencarp5(dim: int, *, ensemble_backend: str, scenario: str) -> float:
    n_grid = dim // 2
    y0_batch, params = brusselator.make_scenario(scenario, n_grid, _ENSEMBLE_SIZE)
    result = julia_kencarp5_solve._julia_solve_with_timing(
        "brusselator",
        y0_batch,
        _T_SPAN,
        params,
        system_config={"n_grid": n_grid},
        ensemble_backend=ensemble_backend,
        **_SOLVER_KWARGS,
    )
    return result.solve_time_s * 1000


def make_solver_specs(scenario: str) -> list[SolverSpec]:
    specs: list[SolverSpec] = []
    for key, label, color, marker, linear in _LOCAL_SOLVER_DEFS:
        specs.append(
            SolverSpec(
                key=key,
                label=label,
                color=color,
                marker=marker,
                linestyle="-",
                timing_fn=(
                    lambda dim, linear=linear, sc=scenario: time_local_kencarp5(
                        dim, linear=linear, scenario=sc
                    )
                ),
            )
        )
    for key, label, color, marker, linear in _CUSTOM_KERNEL_SOLVER_DEFS:
        specs.append(
            SolverSpec(
                key=key,
                label=label,
                color=color,
                marker=marker,
                linestyle="-",
                timing_fn=(
                    lambda dim, linear=linear, sc=scenario: time_custom_kernel_kencarp5(
                        dim, linear=linear, scenario=sc
                    )
                ),
            )
        )
    for key, label, color, marker, lu_precision in _RODAS_SOLVER_DEFS:
        specs.append(
            SolverSpec(
                key=key,
                label=label,
                color=color,
                marker=marker,
                linestyle="--",
                timing_fn=(
                    lambda dim, precision=lu_precision, sc=scenario: time_local_rodas5(
                        dim, scenario=sc, lu_precision=precision
                    )
                ),
            )
        )
    for key, label, color, marker, _fn in _JAX_SOLVER_DEFS:
        specs.append(
            SolverSpec(
                key=key,
                label=label,
                color=color,
                marker=marker,
                linestyle="-",
                timing_fn=(
                    lambda dim, sc=scenario: time_diffrax_kencarp5(dim, scenario=sc)
                ),
            )
        )
    for solver_key, label, _solve, color in _JULIA_SOLVER_DEFS:
        for backend in _JULIA_BACKENDS:
            backend_id, marker, linestyle = _JULIA_BACKEND_STYLES[backend]
            specs.append(
                SolverSpec(
                    key=f"{solver_key}_{backend}",
                    label=f"{label} {backend_id}",
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    timing_fn=(
                        lambda dim, b=backend, sc=scenario: time_julia_kencarp5(
                            dim, ensemble_backend=b, scenario=sc
                        )
                    ),
                )
            )
    return specs


def collect_timing(spec: SolverSpec, dim: int) -> float | None:
    print(f"  {spec.label:<28} dim={dim:>4} ...", end=" ", flush=True)
    try:
        ms = spec.timing_fn(dim)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print(f"{ms:.1f} ms")
    return ms


_Row = tuple[str, str, int, float | None]


def drop_none(rows: list[_Row], solver_key: str) -> tuple[list[int], list[float]]:
    pairs = [
        (dim, ms) for key, _, dim, ms in rows if key == solver_key and ms is not None
    ]
    if not pairs:
        return [], []
    dims, times = zip(*pairs)
    return list(dims), list(times)


def run_benchmarks(
    specs: list[SolverSpec], gpu_name: str, cache: dict, scenario: str
) -> list[_Row]:
    scenario_cache = cache.setdefault(gpu_name, {}).setdefault(scenario, {})
    rows: list[_Row] = []
    for spec in specs:
        print(f"\n{spec.label}:")
        solver_cache = scenario_cache.setdefault(spec.key, {})
        for dim in _DIMENSIONS:
            dim_key = str(dim)
            if dim_key in solver_cache:
                ms = solver_cache[dim_key]
                ms_text = f"{ms:.1f} ms" if ms is not None else "FAILED"
                print(f"  {spec.label:<28} dim={dim:>4} ... (cached) {ms_text}")
            else:
                ms = collect_timing(spec, dim)
                solver_cache[dim_key] = ms
                save_cache(_CACHE_PATH, cache)
            rows.append((spec.key, spec.label, dim, ms))
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
    specs: list[SolverSpec],
    gpu_name: str,
    output_path: Path,
    scenario: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in specs:
        dims, times_ms = drop_none(rows, spec.key)
        if not dims:
            continue
        ax.plot(
            dims,
            times_ms,
            marker=spec.marker,
            color=spec.color,
            linestyle=spec.linestyle,
            label=spec.label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ODE dimension")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title(
        f"KenCarp5 dimensionality — {scenario} — Brusselator — {gpu_name}"
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
    for scenario in brusselator.SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===\n")
        specs = make_solver_specs(scenario)
        rows = run_benchmarks(specs, gpu_name, cache, scenario)
        csv_path, plot_path = _scenario_output_paths(gpu_name, scenario)
        save_csv(rows, csv_path)
        plot(rows, specs, gpu_name, plot_path, scenario)


if __name__ == "__main__":
    main()
