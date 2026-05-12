"""KenCarp5 ensemble-size scaling benchmark on the Brusselator system.

Fixed grid size (n_grid=32, dim=64). Sweeps ensemble size from 3 to 100k on
a log scale and records solve time for the local KenCarp5 solver (both the
``linear=True`` single-LU-per-stage path and the ``linear=False`` Newton
path), Diffrax KenCarp5, and Julia KenCarp5 on ``EnsembleGPUArray``. Julia
provides no ``GPUKenCarp5`` so the kernel backend is skipped. Writes one CSV
and a log-log plot per scenario, named after the GPU.

Usage:
    uv run python scripts/11_kencarp5_brusselator_scaling/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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

jax.config.update("jax_enable_x64", True)

_N_GRID = 32
_T_SPAN = brusselator.TIMES
_ENSEMBLE_SIZES = (3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)
_N_RUNS = 1
_JULIA_BACKENDS = ("EnsembleGPUArray",)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

# (key, label, color, marker, linear)
_LOCAL_SOLVER_DEFS = [
    ("local_kencarp5_linear", "my kencarp5 linear=True", "#2b7be0", "o", True),
    ("local_kencarp5_newton", "my kencarp5 linear=False", "#e02b2b", "D", False),
]
# (key, label, color, marker, solve_fn)
_JAX_SOLVER_DEFS = [
    ("diffrax_kencarp5", "diffrax kencarp5", "#2ba84a", "s", diffrax_kencarp5_solve),
]
# (key, label, solve_fn, color)
_JULIA_SOLVER_DEFS = [
    # DiffEqGPU 3.13 has no SplitODEProblem support on EnsembleGPUArray
    # (see comment block at top of reference/solvers/julia/run_solver.jl),
    # so this row is Julia KenCarp5 treating the combined RHS as fully
    # implicit — *not* IMEX. Slower than my IMEX path; labelled accordingly.
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
    timing_fn: Callable[[object, object], float]


def time_local_kencarp5(ex_fn, im_fn, y0, params, *, linear: bool) -> float:
    y0_j = jnp.asarray(y0)
    p_j = jnp.asarray(params)

    def run():
        return kencarp5_solve(
            ex_fn, im_fn, y0_j, _T_SPAN, p_j, linear=linear, **_SOLVER_KWARGS
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_diffrax_kencarp5(ex_fn, im_fn, y0, params) -> float:
    y0_j = jnp.asarray(y0)
    p_j = jnp.asarray(params)

    def run():
        return diffrax_kencarp5_solve(
            ex_fn, im_fn, y0_j, _T_SPAN, p_j, **_SOLVER_KWARGS
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_julia_kencarp5(y0, params, *, ensemble_backend: str) -> float:
    result = julia_kencarp5_solve._julia_solve_with_timing(
        "brusselator",
        np.asarray(y0),
        _T_SPAN,
        np.asarray(params),
        system_config={"n_grid": _N_GRID},
        ensemble_backend=ensemble_backend,
        **_SOLVER_KWARGS,
    )
    return result.solve_time_s * 1000


def make_solver_specs(ex_fn, im_fn) -> list[SolverSpec]:
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
                    lambda y0, params, linear=linear: time_local_kencarp5(
                        ex_fn, im_fn, y0, params, linear=linear
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
                    lambda y0, params: time_diffrax_kencarp5(ex_fn, im_fn, y0, params)
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
                        lambda y0, params, b=backend: time_julia_kencarp5(
                            y0, params, ensemble_backend=b
                        )
                    ),
                )
            )
    return specs


def collect_timing(spec: SolverSpec, size: int, y0, params) -> float | None:
    print(f"  {spec.label:<28} n={size:>7} ...", end=" ", flush=True)
    try:
        ms = spec.timing_fn(y0, params)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print(f"{ms:.1f} ms")
    return ms


_Row = tuple[str, str, int, float | None]


def drop_none(rows: list[_Row], solver_key: str) -> tuple[list[int], list[float]]:
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
    for scenario in brusselator.SCENARIOS:
        print(f"\n=== {scenario} ===\n")
        rows: list[_Row] = []
        for spec in specs:
            print(f"{spec.label}:")
            solver_cache = gpu_cache.setdefault(f"{scenario}_{spec.key}", {})
            for size in _ENSEMBLE_SIZES:
                size_key = str(size)
                if size_key in solver_cache:
                    ms = solver_cache[size_key]
                    ms_text = f"{ms:.1f} ms" if ms is not None else "FAILED"
                    print(f"  {spec.label:<28} n={size:>7} ... (cached) {ms_text}")
                else:
                    y0, params = brusselator.make_scenario(scenario, _N_GRID, size)
                    ms = collect_timing(spec, size, y0, params)
                    solver_cache[size_key] = ms
                    save_cache(_CACHE_PATH, cache)
                rows.append((spec.key, spec.label, size, ms))
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
    ax.set_title(
        f"KenCarp5 scaling — Brusselator (n_grid={_N_GRID}, {scenario}) — {gpu_name}"
    )
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

    ex_fn, im_fn, _ode_fn, _y0 = brusselator.make_system(_N_GRID)
    cache = load_cache(_CACHE_PATH)
    specs = make_solver_specs(ex_fn, im_fn)
    rows_by_scenario = run_benchmarks(specs, gpu_name, cache)

    slug = gpu_slug(gpu_name)
    for scenario, rows in rows_by_scenario.items():
        csv_path = _SCRIPT_DIR / f"results-{slug}-{scenario}.csv"
        plot_path = _SCRIPT_DIR / f"plot-{slug}-{scenario}.png"
        save_csv(rows, csv_path)
        plot(rows, specs, gpu_name, scenario, plot_path)


if __name__ == "__main__":
    main()
