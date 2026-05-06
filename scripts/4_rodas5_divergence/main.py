"""Robertson divergence-CV benchmark for Rodas5 solvers.

Runs the Robertson system with 400,000 trajectories while sweeping the
``make_scenario(..., divergence=...)`` knob. For each solver and divergence
value, the benchmark records solve time and the actual distribution of accepted
plus rejected Rodas5 steps.

Usage:
    uv run python scripts/4_5_rodas5_robertson_divergence/main.py
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

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve
from reference.systems.python import robertson
from scripts.benchmark_common import (
    get_gpu_name,
    load_cache,
    output_paths,
    save_cache,
    time_blocked,
)
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import prepare_solve as rodas5ckn_prepare_solve
from solvers.rodas5ckn import run_prepared as rodas5ckn_run_prepared
from solvers.rodas5ckn import solve as rodas5ckn_solve

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 30_000
_N_RUNS = 1
_DIM = robertson.N_VARS
_T_SPAN = robertson.TIMES
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
_DIFFRAX_SOLVER_KWARGS = {**_SOLVER_KWARGS, "max_steps": 1_000_000}
_LOCAL_SOLVER_KWARGS = {
    "first_step": 1e-4,
    "rtol": 1e-6,
    "atol": 1e-8,
    "lu_precision": "fp32",
}

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
    max_divergence: float | None = None


_SOLVERS = (
    SolverSpec("rodas5_fp32_lu", "JAX Rodas5 fp32 LU", "#2b7be0", "o", "stats"),
    SolverSpec("rodas5ckn", "numba-cuda Rodas5", "#f0a202", "s", "stats"),
    SolverSpec(
        "rodas5ckn_sorted",
        "numba-cuda Rodas5 sorted",
        "#f0a202",
        "P",
        "stats",
        sort_by_steps=True,
    ),
    # SolverSpec(
    #     "diffrax_kvaerno5",
    #     "Diffrax Kvaerno5",
    #     "#2ba84a",
    #     "^",
    #     "timing",
    #     max_divergence=1.5,
    # ),
    # SolverSpec(
    #     "julia_rodas5_EnsembleGPUArray",
    #     "Julia Rodas5 GPUArray",
    #     "#9b59b6",
    #     "D",
    #     "julia",
    #     "EnsembleGPUArray",
    # ),
    SolverSpec(
        "julia_rodas5_EnsembleGPUKernel",
        "Julia Rodas5 GPUKernel",
        "#d35400",
        "v",
        "julia",
        "EnsembleGPUKernel",
    ),
    SolverSpec(
        "julia_rodas5_EnsembleGPUKernel_sorted",
        "Julia Rodas5 GPUKernel sorted",
        "#d35400",
        "X",
        "julia",
        "EnsembleGPUKernel",
        True,
        None,
    ),
)


def make_data(divergence: float) -> tuple[np.ndarray, np.ndarray]:
    return robertson.make_scenario(
        "divergent",
        _N_TRAJ,
        seed=42,
        divergence=divergence,
    )


def solve_with_stats(solver: SolverSpec, y0: np.ndarray, params: np.ndarray):
    if solver.key.startswith("rodas5ckn"):
        return rodas5ckn_solve(
            robertson.ode_fn_numba_cuda,
            robertson.jac_fn_numba_cuda,
            y0=y0,
            t_span=_T_SPAN,
            params=params,
            return_stats=True,
            **_SOLVER_KWARGS,
        )

    return rodas5_solve(
        robertson.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=jnp.asarray(params, dtype=jnp.float64),
        return_stats=True,
        **_LOCAL_SOLVER_KWARGS,
    )


def solve_timing_only(solver: SolverSpec, y0: np.ndarray, params: np.ndarray):
    if solver.key == "diffrax_kvaerno5":
        return diffrax_kvaerno5_solve(
            robertson.ode_fn,
            y0=jnp.asarray(y0, dtype=jnp.float64),
            t_span=_T_SPAN,
            params=jnp.asarray(params, dtype=jnp.float64),
            **_DIFFRAX_SOLVER_KWARGS,
        )
    raise ValueError(f"unsupported timing-only solver: {solver.key}")


def time_solve(
    solver: SolverSpec, y0: np.ndarray, params: np.ndarray
) -> tuple[float, dict | None]:
    if solver.mode == "julia":
        result = julia_rodas5_solve._julia_solve_with_timing(
            "robertson",
            y0,
            _T_SPAN,
            params,
            ensemble_backend=solver.ensemble_backend,
            **_SOLVER_KWARGS,
        )
        return result.solve_time_s * 1000, None
    if solver.mode == "timing":
        ms, _ = time_blocked(lambda: solve_timing_only(solver, y0, params), _N_RUNS)
        return ms, None
    if solver.key.startswith("rodas5ckn"):
        prepared = rodas5ckn_prepare_solve(
            robertson.ode_fn_numba_cuda,
            robertson.jac_fn_numba_cuda,
            y0=y0,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        )
        ms, result = time_blocked(
            lambda: rodas5ckn_run_prepared(
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
        f"  {solver.label:<20} divergence={divergence:>4.2f} ...",
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
            if solver.max_divergence is not None and divergence > solver.max_divergence:
                print(
                    f"  {solver.label:<20} divergence={divergence:>4.2f} ... "
                    f"SKIPPED (divergence > {solver.max_divergence:g})",
                    flush=True,
                )
                solver_cache.setdefault(divergence_key, None)
                continue
            cached = solver_cache.get(divergence_key)
            if is_complete_row(cached):
                row = cached
                print(
                    f"  {solver.label:<20} divergence={divergence:>4.2f} ... "
                    f"(cached) {format_row(row)}",
                    flush=True,
                )
            else:
                local_row = gpu_cache.get("rodas5_fp32_lu", {}).get(divergence_key)
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
    ax.set_title(f"Robertson Rodas5 divergence — {_N_TRAJ} trajectories — {gpu_name}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"System: {_DIM}D Robertson, {_N_TRAJ} trajectories\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)
    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, csv_path)
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
