"""High-dimensional VDP divergence benchmark for Rodas5 solvers.

Runs the 64D coupled VDP lattice (n_osc = 32) with 1000 trajectories while
sweeping the ``make_scenario(..., divergence=...)`` knob. For each solver and
divergence value, the benchmark records solve time and the actual distribution
of accepted plus rejected Rodas5 steps.

Usage:
    uv run python scripts/6_rodas5_high_dimension_divergence/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.systems.python import coupled_vdp_lattice
from scripts.benchmark_common import (
    get_gpu_name,
    load_cache,
    output_paths,
    save_cache,
    time_blocked,
)
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import solve as rodas5ckn_solve

jax.config.update("jax_enable_x64", True)

_N_OSC = 32
_DIM = 2 * _N_OSC
_ENSEMBLE_SIZE = 1000
_N_RUNS = 10
_T_SPAN = coupled_vdp_lattice.TIMES
_DIVERGENCES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_CSV_FIELDS = (
    "gpu",
    "solver_key",
    "solver",
    "divergence",
    "dim",
    "n_osc",
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


_SOLVERS = (
    SolverSpec("rodas5_fp32_lu", "JAX Rodas5 fp32 LU", "#2b7be0", "o"),
    SolverSpec("rodas5ckn", "numba-cuda Rodas5", "#f0a202", "s"),
)


@cuda.jit(device=True)
def ode_fn_vdp_numba(y, t, p, dy, i):
    n_osc = int(p[i, 0])
    scale = p[i, 1]
    mu = 100.0
    diffusion = 10.0
    for k in range(n_osc):
        kp1 = (k + 1) % n_osc
        km1 = (k + n_osc - 1) % n_osc
        xk = y[i, 2 * k]
        vk = y[i, 2 * k + 1]
        lap = y[i, 2 * kp1] - 2.0 * xk + y[i, 2 * km1]
        dy[i, 2 * k] = vk
        dy[i, 2 * k + 1] = scale * mu * (1.0 - xk * xk) * vk - xk + diffusion * lap


@cuda.jit(device=True)
def jac_fn_vdp_numba(y, t, p, jac, i):
    n_osc = int(p[i, 0])
    n_vars = 2 * n_osc
    scale = p[i, 1]
    mu = 100.0
    diffusion = 10.0
    for r in range(n_vars):
        for c in range(n_vars):
            jac[i, r, c] = 0.0
    for k in range(n_osc):
        kp1 = (k + 1) % n_osc
        km1 = (k + n_osc - 1) % n_osc
        xk = y[i, 2 * k]
        vk = y[i, 2 * k + 1]
        jac[i, 2 * k, 2 * k + 1] = 1.0
        jac[i, 2 * k + 1, 2 * k] = scale * mu * (-2.0 * xk) * vk - 1.0 - 2.0 * diffusion
        jac[i, 2 * k + 1, 2 * k + 1] = scale * mu * (1.0 - xk * xk)
        if kp1 == km1:
            jac[i, 2 * k + 1, 2 * kp1] = 2.0 * diffusion
        else:
            jac[i, 2 * k + 1, 2 * kp1] = diffusion
            jac[i, 2 * k + 1, 2 * km1] = diffusion


def make_data(divergence: float) -> tuple[np.ndarray, np.ndarray]:
    return coupled_vdp_lattice.make_scenario(
        "divergent",
        _N_OSC,
        _ENSEMBLE_SIZE,
        seed=42,
        divergence=divergence,
    )


def solve_with_stats(solver: SolverSpec, y0: np.ndarray, params: np.ndarray):
    if solver.key == "rodas5ckn":
        ckn_params = np.column_stack(
            [
                np.full(_ENSEMBLE_SIZE, float(_N_OSC), dtype=np.float64),
                params[:, 0],
            ]
        )
        return rodas5ckn_solve(
            ode_fn_vdp_numba,
            jac_fn_vdp_numba,
            y0=y0,
            t_span=_T_SPAN,
            params=ckn_params,
            return_stats=True,
            **_SOLVER_KWARGS,
        )

    ode_fn, _ = coupled_vdp_lattice.make_system(_N_OSC)
    return rodas5_solve(
        ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=jnp.asarray(params),
        lu_precision="fp32",
        return_stats=True,
        **_SOLVER_KWARGS,
    )


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


def format_row(row: dict) -> str:
    return (
        f"{row['solve_time_ms']:.1f} ms, "
        f"mean_steps={row['mean_steps']:.1f}, "
        f"step_cv={row['step_cv']:.3f}, "
        f"norm={row['normalized_solve_time_ms_per_step']:.4f} ms/step"
    )


def is_complete_row(value) -> bool:
    return isinstance(value, dict) and all(field in value for field in _CSV_FIELDS)


def collect_row(gpu_name: str, solver: SolverSpec, divergence: float) -> dict | None:
    print(
        f"  {solver.label:<20} divergence={divergence:>4.2f} ...",
        end=" ",
        flush=True,
    )
    y0, params = make_data(divergence)
    try:
        ms, result = time_blocked(lambda: solve_with_stats(solver, y0, params), _N_RUNS)
    except Exception as exc:
        print(f"FAILED ({exc})", flush=True)
        return None

    _, stats = result
    stats_summary = summarize_stats(stats)
    normalized = ms / stats_summary["mean_steps"] if stats_summary["mean_steps"] else 0.0
    row = {
        "gpu": gpu_name,
        "solver_key": solver.key,
        "solver": solver.label,
        "divergence": float(divergence),
        "dim": _DIM,
        "n_osc": _N_OSC,
        "ensemble_size": _ENSEMBLE_SIZE,
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
                    f"  {solver.label:<20} divergence={divergence:>4.2f} ... "
                    f"(cached) {format_row(row)}",
                    flush=True,
                )
            else:
                row = collect_row(gpu_name, solver, divergence)
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
    ax.set_title(f"64D coupled VDP divergence — 1000 trajectories — {gpu_name}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"System: {_DIM}D coupled VDP lattice, {_ENSEMBLE_SIZE} trajectories\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)
    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, csv_path)
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
