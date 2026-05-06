"""Lorenz divergence-CV benchmark for Tsit5 solvers.

Runs the Lorenz system with 400,000 trajectories while sweeping the
``make_scenario(..., divergence=...)`` knob. For each solver and divergence
value, the benchmark records solve time and the actual distribution of accepted
plus rejected Tsit5 steps.

Usage:
    uv run python scripts/2_5_tsit5_lorenz_divergence/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.systems.python import lorenz
from scripts.benchmark_common import (
    get_gpu_name,
    load_cache,
    output_paths,
    save_cache,
    time_blocked,
)
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import prepare_solve as tsit5ckn_prepare_solve
from solvers.tsit5ckn import run_prepared as tsit5ckn_run_prepared

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 100_000
_N_RUNS = 1
_DIM = lorenz.N_VARS
_DIVERGENCES = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_CSV_FIELDS = (
    "gpu",
    "solver_key",
    "solver",
    "divergence",
    "dim",
    "ensemble_size",
    "solve_time_ms",
    "mean_steps",
    "step_std",
    "step_cv",
    "normalized_solve_time_ms_per_step",
    "step_variance",
    "min_steps",
    "max_steps",
    "rejected_steps_mean",
)


@dataclass(frozen=True)
class SolverSpec:
    key: str
    label: str
    color: str
    marker: str
    mode: str
    ensemble_backend: str | None = None
    sort_by_steps: bool = False


_SOLVERS = (
    SolverSpec("tsit5", "JAX Tsit5", "#2b7be0", "o", "stats"),
    SolverSpec("tsit5ckn", "numba-cuda Tsit5", "#f0a202", "s", "stats"),
    SolverSpec(
        "tsit5ckn_sorted",
        "numba-cuda Tsit5 sorted",
        "#f0a202",
        "P",
        "stats",
        sort_by_steps=True,
    ),
    # SolverSpec("diffrax_tsit5", "Diffrax Tsit5", "#2ba84a", "^", "timing"),
    # SolverSpec(
    #     "julia_tsit5_EnsembleGPUArray",
    #     "Julia Tsit5 GPUArray",
    #     "#9b59b6",
    #     "D",
    #     "julia",
    #     "EnsembleGPUArray",
    # ),
    SolverSpec(
        "julia_tsit5_EnsembleGPUKernel",
        "Julia Tsit5 GPUKernel",
        "#d35400",
        "v",
        "julia",
        "EnsembleGPUKernel",
    ),
    SolverSpec(
        "julia_tsit5_EnsembleGPUKernel_sorted",
        "Julia Tsit5 GPUKernel sorted",
        "#d35400",
        "X",
        "julia",
        "EnsembleGPUKernel",
        True,
    ),
)


def make_data(divergence: float) -> tuple[np.ndarray, np.ndarray]:
    return lorenz.make_scenario(
        "divergent",
        _N_TRAJ,
        seed=42,
        divergence=divergence,
    )


def solve_with_stats(solver: SolverSpec, y0: np.ndarray, params: np.ndarray):
    if solver.key.startswith("tsit5ckn"):
        prepared = tsit5ckn_prepare_solve(
            lorenz.ode_fn_numba_cuda,
            y0=y0,
            t_span=lorenz.TIMES,
            params=params,
            **_SOLVER_KWARGS,
        )
        return tsit5ckn_run_prepared(
            prepared,
            return_stats=True,
            copy_solution=False,
        )

    return tsit5_solve(
        lorenz.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=lorenz.TIMES,
        params=jnp.asarray(params, dtype=jnp.float64),
        return_stats=True,
        **_SOLVER_KWARGS,
    )


def solve_timing_only(solver: SolverSpec, y0: np.ndarray, params: np.ndarray):
    if solver.key == "diffrax_tsit5":
        return diffrax_tsit5_solve(
            lorenz.ode_fn,
            y0=jnp.asarray(y0, dtype=jnp.float64),
            t_span=lorenz.TIMES,
            params=jnp.asarray(params, dtype=jnp.float64),
            **_SOLVER_KWARGS,
        )
    raise ValueError(f"unsupported timing-only solver: {solver.key}")


def time_solve(
    solver: SolverSpec, y0: np.ndarray, params: np.ndarray
) -> tuple[float, dict | None]:
    if solver.mode == "julia":
        result = julia_tsit5_solve._julia_solve_with_timing(
            "lorenz",
            y0,
            lorenz.TIMES,
            params,
            ensemble_backend=solver.ensemble_backend,
            **_SOLVER_KWARGS,
        )
        return result.solve_time_s * 1000, None
    if solver.mode == "timing":
        ms, _ = time_blocked(lambda: solve_timing_only(solver, y0, params), _N_RUNS)
        return ms, None
    if solver.key.startswith("tsit5ckn"):
        prepared = tsit5ckn_prepare_solve(
            lorenz.ode_fn_numba_cuda,
            y0=y0,
            t_span=lorenz.TIMES,
            params=params,
            **_SOLVER_KWARGS,
        )
        ms, result = time_blocked(
            lambda: tsit5ckn_run_prepared(
                prepared,
                return_stats=True,
                copy_solution=False,
            ),
            _N_RUNS,
        )
        _, stats = result
        return ms, stats

    ms, result = time_blocked(lambda: solve_with_stats(solver, y0, params), _N_RUNS)
    _, stats = result
    return ms, stats


def summarize_stats(stats: dict) -> dict[str, float | int]:
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    total_steps = accepted_steps + rejected_steps
    mean_steps = float(np.mean(total_steps))
    ddof = 1 if total_steps.size > 1 else 0
    step_variance = float(np.var(total_steps, ddof=ddof))
    step_std = float(np.sqrt(step_variance))
    step_cv = step_std / mean_steps if mean_steps else 0.0
    return {
        "mean_steps": mean_steps,
        "step_std": step_std,
        "step_cv": float(step_cv),
        "step_variance": step_variance,
        "min_steps": int(np.min(total_steps)),
        "max_steps": int(np.max(total_steps)),
        "rejected_steps_mean": float(np.mean(rejected_steps)),
    }


def sort_by_attempted_steps(
    y0: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    _, stats = solve_with_stats(_SOLVERS[0], y0, params)
    jax.block_until_ready(stats)
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    attempts = accepted_steps + rejected_steps
    order = np.argsort(-attempts, kind="stable")
    return y0[order], params[order], summarize_stats(stats)


def stats_from_row(row: dict) -> dict[str, float | int]:
    return {
        "mean_steps": row["mean_steps"],
        "step_std": row["step_std"],
        "step_cv": row["step_cv"],
        "step_variance": row["step_variance"],
        "min_steps": row["min_steps"],
        "max_steps": row["max_steps"],
        "rejected_steps_mean": row["rejected_steps_mean"],
    }


def format_row(row: dict) -> str:
    return (
        f"{row['solve_time_ms']:.1f} ms, "
        f"mean_steps={row['mean_steps']:.1f}, "
        f"step_cv={row['step_cv']:.3f}, "
        f"norm={row['normalized_solve_time_ms_per_step']:.4f} ms/step"
    )


def is_complete_row(value) -> bool:
    return isinstance(value, dict) and all(field in value for field in _CSV_FIELDS)


def collect_row(
    gpu_name: str,
    solver: SolverSpec,
    divergence: float,
    stats_summary: dict | None = None,
) -> dict | None:
    print(
        f"  {solver.label:<16} divergence={divergence:>4.2f} ...",
        end=" ",
        flush=True,
    )
    y0, params = make_data(divergence)
    if solver.sort_by_steps:
        y0, params, sorted_stats_summary = sort_by_attempted_steps(y0, params)
        if stats_summary is None:
            stats_summary = sorted_stats_summary
    try:
        ms, stats = time_solve(solver, y0, params)
    except Exception as exc:
        print(f"FAILED ({exc})", flush=True)
        return None

    if stats is not None:
        stats_summary = summarize_stats(stats)
    elif stats_summary is None:
        _, stats = solve_with_stats(_SOLVERS[0], y0, params)
        jax.block_until_ready(stats)
        stats_summary = summarize_stats(stats)

    normalized = (
        ms / stats_summary["mean_steps"] if stats_summary["mean_steps"] else 0.0
    )
    row = {
        "gpu": gpu_name,
        "solver_key": solver.key,
        "solver": solver.label,
        "divergence": float(divergence),
        "dim": _DIM,
        "ensemble_size": _N_TRAJ,
        "solve_time_ms": float(ms),
        **stats_summary,
        "normalized_solve_time_ms_per_step": float(normalized),
    }
    print(format_row(row), flush=True)
    return row


def run_benchmarks(gpu_name: str, cache: dict) -> list[dict]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[dict] = []
    for solver in _SOLVERS:
        print(f"\n{solver.label}:")
        solver_cache = gpu_cache.setdefault(solver.key, {})
        for divergence in _DIVERGENCES:
            divergence_key = f"{divergence:.6g}"
            cached = solver_cache.get(divergence_key)
            if is_complete_row(cached):
                row = cached
                print(
                    f"  {solver.label:<16} divergence={divergence:>4.2f} ... "
                    f"(cached) {format_row(row)}",
                    flush=True,
                )
            else:
                local_row = gpu_cache.get("tsit5", {}).get(divergence_key)
                stats_summary = (
                    stats_from_row(local_row) if is_complete_row(local_row) else None
                )
                row = collect_row(gpu_name, solver, divergence, stats_summary)
                solver_cache[divergence_key] = row
                save_cache(_CACHE_PATH, cache)
            if row is not None:
                rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


def rows_for_solver(rows: list[dict], solver_key: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["solver_key"] == solver_key),
        key=lambda row: row["divergence"],
    )


def plot(rows: list[dict], gpu_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for solver in _SOLVERS:
        solver_rows = rows_for_solver(rows, solver.key)
        if not solver_rows:
            continue
        xs = [row["step_cv"] for row in solver_rows]
        ys = [row["normalized_solve_time_ms_per_step"] for row in solver_rows]
        ax.scatter(
            xs,
            ys,
            color=solver.color,
            marker=solver.marker,
            s=42,
            label=solver.label,
        )

    ax.set_xlabel("Normalized standard deviation of attempted steps")
    ax.set_ylabel("Solve time / mean attempted steps (ms)")
    ax.set_yscale("log")
    ax.set_title(f"Lorenz Tsit5 divergence — {_N_TRAJ} trajectories — {gpu_name}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"System: {_DIM}D Lorenz, {_N_TRAJ} trajectories\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)
    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, csv_path)
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
