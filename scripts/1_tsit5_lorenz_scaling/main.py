"""Solver scaling benchmark on the Lorenz system.

Sweeps ensemble size from 1 to 100k on a log scale and records solve time for
the local Tsit5 solver, Diffrax Tsit5, and Julia Tsit5 with both
DiffEqGPU ensemble backends. Outputs a CSV and a log-log plot per scenario
named after the GPU.

Usage:
    uv run python scripts/1_tsit5_scaling/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.systems.python import lorenz
from scripts.benchmark_common import (
    collect_timed_timing,
    format_cached_timing,
    get_gpu_name,
    gpu_slug,
    load_cache,
    save_cache,
    time_blocked,
    timing_value_or_none,
)
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import solve as tsit5ckn_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = (0.0, 5.0)
_ENSEMBLE_SIZES = (3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
_N_RUNS = 10
_JULIA_BACKENDS = ("EnsembleGPUArray", "EnsembleGPUKernel")

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

# (key, label, color, marker, solve_fn)
_LOCAL_JAX_SOLVER_DEFS = [
    ("local_tsit5", "my tsit5", "#2b7be0", "o", tsit5_solve),
]
_CUSTOM_KERNEL_SOLVER_DEFS = [
    ("tsit5ckn", "numba-cuda tsit5", "#f0a202", "P", tsit5ckn_solve),
]
_REFERENCE_JAX_SOLVER_DEFS = [
    ("diffrax_tsit5", "diffrax tsit5", "#2ba84a", "s", diffrax_tsit5_solve),
]
# (key, label, solve_fn, color)
_JULIA_SOLVER_DEFS = [
    ("julia_tsit5", "julia tsit5", julia_tsit5_solve, "#9b59b6"),
]
# backend -> (label_suffix, marker, linestyle)
_JULIA_BACKEND_STYLES = {
    "EnsembleGPUArray": ("array", "^", "-"),
    "EnsembleGPUKernel": ("kernel", "v", "--"),
}


@dataclass(frozen=True)
class SolverSpec:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str
    timing_fn: Callable[[object, object], float]


def time_local_jax_solver(solve_fn, y0, params) -> float:
    def run():
        return solve_fn(
            lorenz.ode_fn,
            y0=y0,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_custom_kernel_solver(solve_fn, y0, params) -> float:
    def run():
        return solve_fn(
            lorenz.ode_fn_numba_cuda,
            y0=np.asarray(y0),
            t_span=_T_SPAN,
            params=np.asarray(params),
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_reference_jax_solver(solve_fn, _y0, params) -> float:
    def run():
        return solve_fn(
            lorenz.ode_fn,
            y0=lorenz.Y0,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_julia_solver(solve, _y0, params, *, ensemble_backend: str) -> float:
    """Return Julia's internal solve time in ms, excluding subprocess overhead."""
    result = solve._julia_solve_with_timing(
        "lorenz",
        lorenz.Y0,
        _T_SPAN,
        np.asarray(params),
        ensemble_backend=ensemble_backend,
        **_SOLVER_KWARGS,
    )
    return result.solve_time_s * 1000


def make_solver_specs() -> list[SolverSpec]:
    specs = [
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda y0, params, fn=fn: time_local_jax_solver(fn, y0, params),
        )
        for key, label, color, marker, fn in _LOCAL_JAX_SOLVER_DEFS
    ]
    specs += [
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda y0, params, fn=fn: time_custom_kernel_solver(
                fn, y0, params
            ),
        )
        for key, label, color, marker, fn in _CUSTOM_KERNEL_SOLVER_DEFS
    ]
    specs += [
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda y0, params, fn=fn: time_reference_jax_solver(
                fn, y0, params
            ),
        )
        for key, label, color, marker, fn in _REFERENCE_JAX_SOLVER_DEFS
    ]
    for solver_key, label, solve, color in _JULIA_SOLVER_DEFS:
        for backend in _JULIA_BACKENDS:
            backend_id, marker, linestyle = _JULIA_BACKEND_STYLES[backend]
            specs.append(
                SolverSpec(
                    key=f"{solver_key}_{backend}",
                    label=f"{label} {backend_id}",
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    timing_fn=lambda y0, params, s=solve, b=backend: time_julia_solver(
                        s, y0, params, ensemble_backend=b
                    ),
                )
            )
    return specs


def collect_timing(spec: SolverSpec, size: int, y0, params):
    return collect_timed_timing(
        spec.label,
        f"n={size:>7}",
        lambda: spec.timing_fn(y0, params),
        label_width=20,
    )


_Row = tuple[str, str, int, float | None]


def drop_none(
    rows: list[_Row],
    solver_key: str,
) -> tuple[list[int], list[float]]:
    pairs = [
        (size, ms) for key, _, size, ms in rows if key == solver_key and ms is not None
    ]
    if not pairs:
        return [], []
    sizes, times = zip(*pairs)
    return list(sizes), list(times)


def run_benchmarks(
    specs: list[SolverSpec], gpu_name: str, cache: dict
) -> dict[str, list[_Row]]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows_by_scenario: dict[str, list[_Row]] = {}
    for scenario in lorenz.SCENARIOS:
        print(f"\n=== {scenario} ===\n")
        rows: list[_Row] = []
        for spec in specs:
            print(f"{spec.label}:")
            solver_cache = gpu_cache.setdefault(f"{scenario}_{spec.key}", {})
            for size in _ENSEMBLE_SIZES:
                size_key = str(size)
                if size_key in solver_cache:
                    ms = solver_cache[size_key]
                    ms_text = format_cached_timing(ms)
                    print(f"  {spec.label:<20} n={size:>7} ... (cached) {ms_text}")
                else:
                    y0, params = lorenz.make_scenario(scenario, size)
                    ms = collect_timing(spec, size, y0, params)
                    solver_cache[size_key] = ms
                    save_cache(_CACHE_PATH, cache)
                rows.append((spec.key, spec.label, size, timing_value_or_none(ms)))
            print()
        rows_by_scenario[scenario] = rows
    return rows_by_scenario


def save_csv(rows: list[_Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "ensemble_size", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[_Row],
    specs: list[SolverSpec],
    gpu_name: str,
    scenario: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in specs:
        sizes, times_ms = drop_none(rows, spec.key)
        if not sizes:
            continue
        ax.plot(
            sizes,
            times_ms,
            marker=spec.marker,
            color=spec.color,
            linestyle=spec.linestyle,
            label=spec.label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title(f"Tsit5 scaling — Lorenz ({scenario}) — {gpu_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_ENSEMBLE_SIZES)
    ax.set_xticklabels([str(n) for n in _ENSEMBLE_SIZES], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)
    specs = make_solver_specs()
    rows_by_scenario = run_benchmarks(specs, gpu_name, cache)

    slug = gpu_slug(gpu_name)
    for scenario, rows in rows_by_scenario.items():
        csv_path = _SCRIPT_DIR / f"results-{slug}-{scenario}.csv"
        plot_path = _SCRIPT_DIR / f"plot-{slug}-{scenario}.png"
        save_csv(rows, csv_path)
        plot(rows, specs, gpu_name, scenario, plot_path)


if __name__ == "__main__":
    main()
