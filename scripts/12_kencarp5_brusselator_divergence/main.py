"""Brusselator divergence-CV benchmark for KenCarp5 solvers.

Fixed n_grid=32 (dim=64) coupled Brusselator with 1000 trajectories while
sweeping the ``make_scenario(..., divergence=...)`` knob. For each solver and
divergence value, the benchmark records solve time and the actual distribution
of accepted plus rejected KenCarp5 steps (for the local JAX solver, which is
the only one that exposes per-lane step counters).

Usage:
    uv run python scripts/12_kencarp5_brusselator_divergence/main.py
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

from reference.solvers.python.diffrax_kencarp5 import solve as diffrax_kencarp5_solve
from reference.solvers.python.julia_kencarp5 import solve as julia_kencarp5_solve
from reference.systems.python import brusselator
from scripts.benchmark_common import (
    TIMEOUT_ERROR,
    BenchmarkTimeoutError,
    get_gpu_name,
    is_timeout,
    load_cache,
    output_paths,
    save_cache,
    time_blocked,
    timed_solve,
    timeout_cache_entry,
)
from solvers.kencarp5 import solve as kencarp5_solve
from solvers.kencarp5ckn import solve as kencarp5ckn_solve
from solvers.rodas5 import solve as rodas5_solve

jax.config.update("jax_enable_x64", True)

_N_GRID = 32
_DIM = 2 * _N_GRID
_ENSEMBLE_SIZE = 1000
_N_RUNS = 5
_T_SPAN = brusselator.TIMES
_DIVERGENCES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_EX_FN, _IM_FN, _ODE_FN, _ = brusselator.make_system(_N_GRID)

_CSV_FIELDS = (
    "gpu",
    "solver_key",
    "solver",
    "divergence",
    "dim",
    "n_grid",
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
class Case:
    key: str
    color: str
    marker: str
    mode: str  # "stats" | "timing" | "julia"
    linear: bool | None = None
    ensemble_backend: str | None = None

    @property
    def label(self) -> str:
        return self.key


CASES = (
    Case(
        "local_kencarp5_linear",
        "#2b7be0",
        "o",
        "stats",
        linear=True,
    ),
    Case(
        "local_kencarp5_newton",
        "#e02b2b",
        "D",
        "stats",
        linear=False,
    ),
    Case(
        "diffrax_kencarp5",
        "#2ba84a",
        "s",
        "timing",
    ),
    Case(
        "kencarp5ckn_linear",
        "#f0a202",
        "P",
        "kencarp5ckn",
        linear=True,
    ),
    Case(
        "kencarp5ckn_newton",
        "#d35400",
        "X",
        "kencarp5ckn",
        linear=False,
    ),
    Case(
        "local_rodas5_fp64_lu",
        "#00a6a6",
        "v",
        "timing",
    ),
    # DiffEqGPU 3.13 has no SplitODEProblem support on EnsembleGPUArray,
    # so this Julia row is fully-implicit (no IMEX) — see the comment
    # block at top of reference/solvers/julia/run_solver.jl.
    Case(
        "julia_kencarp5_EnsembleGPUArray",
        "#9b59b6",
        "^",
        "julia",
        ensemble_backend="EnsembleGPUArray",
    ),
)


def make_data(divergence: float) -> tuple[np.ndarray, np.ndarray]:
    return brusselator.make_scenario(
        _N_GRID,
        _ENSEMBLE_SIZE,
        seed=42,
        divergence=divergence,
    )


def solve_with_stats(linear: bool, y0: np.ndarray, params: np.ndarray):
    return kencarp5_solve(
        _EX_FN,
        _IM_FN,
        jnp.asarray(y0, dtype=jnp.float64),
        _T_SPAN,
        jnp.asarray(params),
        linear=linear,
        return_stats=True,
        **_SOLVER_KWARGS,
    )


def solve_timing_only(solver: Case, y0: np.ndarray, params: np.ndarray):
    if solver.key == "diffrax_kencarp5":
        return diffrax_kencarp5_solve(
            _EX_FN,
            _IM_FN,
            jnp.asarray(y0, dtype=jnp.float64),
            _T_SPAN,
            jnp.asarray(params),
            **_SOLVER_KWARGS,
        )
    if solver.key == "local_rodas5_fp64_lu":
        return rodas5_solve(
            _ODE_FN,
            jnp.asarray(y0, dtype=jnp.float64),
            _T_SPAN,
            jnp.asarray(params),
            lu_precision="fp64",
            **_SOLVER_KWARGS,
        )
    raise ValueError(f"unsupported timing-only solver: {solver.key}")


def time_solve(
    solver: Case, y0: np.ndarray, params: np.ndarray
) -> tuple[float, dict | None]:
    if solver.mode == "julia":
        result = julia_kencarp5_solve._julia_solve_with_timing(
            "brusselator",
            y0,
            _T_SPAN,
            params,
            system_config={"n_grid": _N_GRID},
            ensemble_backend=solver.ensemble_backend,
            **_SOLVER_KWARGS,
        )
        return result.solve_time_s * 1000, None
    if solver.mode == "timing":
        ms, _ = time_blocked(lambda: solve_timing_only(solver, y0, params), _N_RUNS)
        return ms, None
    if solver.mode == "kencarp5ckn":
        assert solver.linear is not None
        ms, result = time_blocked(
            lambda: kencarp5ckn_solve(
                brusselator.explicit_ode_fn_numba_cuda,
                brusselator.implicit_ode_fn_numba_cuda,
                brusselator.implicit_jac_fn_numba_cuda,
                y0=y0,
                t_span=_T_SPAN,
                params=params,
                linear=solver.linear,
                return_stats=True,
                **_SOLVER_KWARGS,
            ),
            _N_RUNS,
        )
        _, stats = result
        return ms, stats

    assert solver.linear is not None
    ms, result = time_blocked(
        lambda: solve_with_stats(solver.linear, y0, params), _N_RUNS
    )
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
    solver: Case,
    divergence: float,
    stats_summary: dict | None = None,
) -> dict | None:
    print(
        f"  {solver.label:<28} divergence={divergence:>4.2f} ...",
        end=" ",
        flush=True,
    )

    def run():
        y0, params = make_data(divergence)
        ms, stats = time_solve(solver, y0, params)

        local_stats_summary = stats_summary
        if stats is not None:
            local_stats_summary = summarize_stats(stats)
        elif local_stats_summary is None:
            _, stats = solve_with_stats(True, y0, params)
            jax.block_until_ready(stats)
            local_stats_summary = summarize_stats(stats)

        normalized = (
            ms / local_stats_summary["mean_steps"]
            if local_stats_summary["mean_steps"]
            else 0.0
        )
        return {
            "gpu": gpu_name,
            "solver_key": solver.key,
            "solver": solver.label,
            "divergence": float(divergence),
            "dim": _DIM,
            "n_grid": _N_GRID,
            "ensemble_size": _ENSEMBLE_SIZE,
            "solve_time_ms": float(ms),
            **local_stats_summary,
            "normalized_solve_time_ms_per_step": float(normalized),
        }

    try:
        row = timed_solve(run)
    except BenchmarkTimeoutError:
        print(TIMEOUT_ERROR, flush=True)
        return timeout_cache_entry()
    except Exception as exc:
        print(f"FAILED ({exc})", flush=True)
        return None
    print(format_row(row), flush=True)
    return row


def _jax_warmup() -> None:
    y0, params = make_data(_DIVERGENCES[0])
    for _ in range(2):
        result = solve_with_stats(True, y0, params)
        jax.block_until_ready(result)


def run_benchmarks(gpu_name: str, cache: dict) -> list[dict]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[dict] = []
    for solver in CASES:
        print(f"\n{solver.label}:")
        solver_cache = gpu_cache.setdefault(solver.key, {})
        for divergence in _DIVERGENCES:
            divergence_key = f"{divergence:.6g}"
            cached = solver_cache.get(divergence_key)
            if is_complete_row(cached) or is_timeout(cached):
                row = cached
                text = TIMEOUT_ERROR if is_timeout(row) else format_row(row)
                print(
                    f"  {solver.label:<28} divergence={divergence:>4.2f} ... "
                    f"(cached) {text}",
                    flush=True,
                )
            else:
                local_row = gpu_cache.get("local_kencarp5_linear", {}).get(
                    divergence_key
                )
                stats_summary = (
                    stats_from_row(local_row) if is_complete_row(local_row) else None
                )
                row = collect_row(gpu_name, solver, divergence, stats_summary)
                solver_cache[divergence_key] = row
                save_cache(_CACHE_PATH, cache)
            if row is not None and not is_timeout(row):
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
    for solver in CASES:
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
    ax.set_title(
        f"{_DIM}D Brusselator divergence — {_ENSEMBLE_SIZE} trajectories — {gpu_name}"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"System: {_DIM}D Brusselator, {_ENSEMBLE_SIZE} trajectories\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)
    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, csv_path)
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
