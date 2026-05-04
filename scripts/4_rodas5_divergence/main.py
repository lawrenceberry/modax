"""SIMT-divergence benchmark: Rodas5 on Robertson, varying batch_size.

This benchmark measures wall time and wasted lane work when Robertson
trajectories with increasingly varied initial conditions share a Rodas5
while-loop batch.

Initial condition design
------------------------

The identical scenario pins (alpha=0.9, eps=0.1) for every trajectory,
giving the hardest representative IC.

The divergent scenario draws alpha ~ U(0, 0.9) with eps=0.1 fixed.  This
creates a wide spread of step counts (low alpha is easy because there is
little fuel left to sustain the reactions; high alpha approaches the
identical hardest case) while guaranteeing no trajectory is harder than
identical.

Usage:
    uv run python scripts/4_rodas5_divergence/main.py
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

from reference.systems.python import robertson
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

_N_TRAJ = 400_000
_T_SPAN = robertson.TIMES
_N_RUNS = 1
_BATCH_SIZES = (10_000, 25_000, 50_000, 100_000, 200_000, 400_000)
_SOLVER_KWARGS = {
    "first_step": 1e-4,
    "rtol": 1e-6,
    "atol": 1e-8,
}
_LOCAL_SOLVER_KWARGS = {
    "first_step": 1e-4,
    "rtol": 1e-6,
    "atol": 1e-8,
    "lu_precision": "fp32",
}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"


@dataclass(frozen=True)
class Scenario:
    key: str
    color: str


@dataclass(frozen=True)
class Grouping:
    key: str
    label: str
    linestyle: str
    marker: str


@dataclass(frozen=True)
class SolverSpec:
    key: str
    label: str
    solve_fn: object
    color: str


_SCENARIOS = (
    Scenario("identical", "#2b7be0"),
    Scenario("divergent", "#e02b2b"),
)

_GROUPINGS = (
    Grouping("random", "random", "-", "o"),
    Grouping("sorted", "sorted", "--", "s"),
)

_SOLVERS = (
    SolverSpec("rodas5", "my rodas5", rodas5_solve, "#2b7be0"),
    SolverSpec("rodas5ckn", "numba-cuda rodas5", rodas5ckn_solve, "#f0a202"),
)

_CSV_FIELDS = (
    "gpu",
    "solver_key",
    "solver",
    "case_key",
    "batch_size",
    "solve_time_ms",
    "total_lane_iterations",
    "wasted_lane_iterations",
    "min_batch_loop_iterations",
    "max_batch_loop_iterations",
    "wasted_lane_iteration_ratio",
)


def summarize_stats(stats: dict) -> dict[str, float | int]:
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    batch_loop_iterations = np.asarray(jax.device_get(stats["batch_loop_iterations"]))
    valid_lanes = np.asarray(jax.device_get(stats["valid_lanes"]))
    active_lane_iterations = int(np.sum(accepted_steps + rejected_steps))
    total_lane_iterations = int(np.sum(batch_loop_iterations * valid_lanes))
    wasted_lane_iterations = total_lane_iterations - active_lane_iterations
    wasted_lane_iteration_ratio = (
        wasted_lane_iterations / total_lane_iterations
        if total_lane_iterations > 0
        else 0.0
    )
    return {
        "total_lane_iterations": total_lane_iterations,
        "wasted_lane_iterations": int(wasted_lane_iterations),
        "min_batch_loop_iterations": int(np.min(batch_loop_iterations)),
        "max_batch_loop_iterations": int(np.max(batch_loop_iterations)),
        "wasted_lane_iteration_ratio": float(wasted_lane_iteration_ratio),
    }


def format_stats(row: dict) -> str:
    return (
        f"{row['solve_time_ms']:.1f} ms, "
        f"lanes={row['total_lane_iterations']}, "
        f"wasted_lanes={row['wasted_lane_iterations']}, "
        f"min_batch_steps={row['min_batch_loop_iterations']}, "
        f"max_batch_steps={row['max_batch_loop_iterations']}, "
        f"wasted={row['wasted_lane_iteration_ratio']:.3f}"
    )


def solve_with_stats(
    solver: SolverSpec, y0: np.ndarray, params: np.ndarray, batch_size: int | None
):
    if solver.key == "rodas5ckn":
        return solver.solve_fn(
            robertson.ode_fn_numba_cuda,
            robertson.jac_fn_numba_cuda,
            y0=y0,
            t_span=_T_SPAN,
            params=params,
            batch_size=batch_size,
            return_stats=True,
            **_SOLVER_KWARGS,
        )
    return solver.solve_fn(
        robertson.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=params,
        batch_size=batch_size,
        return_stats=True,
        **_LOCAL_SOLVER_KWARGS,
    )


def time_solve_with_stats(
    solver: SolverSpec, y0: np.ndarray, params: np.ndarray, batch_size: int | None
) -> tuple[float, dict]:
    ms, result = time_blocked(
        lambda: solve_with_stats(solver, y0, params, batch_size), _N_RUNS
    )
    _, stats = result
    return ms, summarize_stats(stats)


def active_attempt_order(y0: np.ndarray, params: np.ndarray) -> np.ndarray:
    _, stats = solve_with_stats(_SOLVERS[0], y0, params, batch_size=None)
    jax.block_until_ready(stats)
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    attempts = accepted_steps + rejected_steps
    return np.argsort(attempts, kind="stable")


def order_scenario_data(
    y0: np.ndarray, params: np.ndarray, scenario: Scenario, grouping: Grouping
) -> tuple[np.ndarray, np.ndarray]:
    if grouping.key == "sorted":
        order = active_attempt_order(y0, params)
        return y0[order], params[order]
    seed = sum(ord(c) for c in f"{scenario.key}:{grouping.key}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y0.shape[0])
    return y0[perm], params[perm]


def is_complete_row(value) -> bool:
    return isinstance(value, dict) and all(field in value for field in _CSV_FIELDS)


def iter_cases():
    for scenario in _SCENARIOS:
        for grouping in _GROUPINGS:
            if scenario.key == "identical" and grouping.key == "sorted":
                continue
            yield scenario, grouping


def collect_row(
    gpu_name: str,
    solver: SolverSpec,
    scenario: Scenario,
    grouping: Grouping,
    y0: np.ndarray,
    params: np.ndarray,
    batch_size: int,
) -> dict | None:
    print(
        f"  {solver.label:<18} {scenario.key:<12} {grouping.label:<6} "
        f"batch_size={batch_size:>6} ...",
        end=" ",
        flush=True,
    )
    try:
        ms, stats = time_solve_with_stats(solver, y0, params, batch_size)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    row = {
        "gpu": gpu_name,
        "solver_key": solver.key,
        "solver": solver.label,
        "case_key": f"{scenario.key}_{grouping.key}",
        "batch_size": int(batch_size),
        "solve_time_ms": float(ms),
        **stats,
    }
    print(format_stats(row), flush=True)
    return row


def run_benchmarks(gpu_name: str, cache: dict) -> list[dict]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[dict] = []
    for scenario, grouping in iter_cases():
        base_y0, base_params = robertson.make_scenario(scenario.key, _N_TRAJ)
        case_key = f"{scenario.key}_{grouping.key}"
        y0, params = order_scenario_data(base_y0, base_params, scenario, grouping)
        print(f"{scenario.key} / {grouping.label}:")
        for solver in _SOLVERS:
            solver_cache = gpu_cache.setdefault(f"{case_key}_{solver.key}", {})
            for bs in _BATCH_SIZES:
                bs_key = str(int(bs))
                cached = solver_cache.get(bs_key)
                if is_complete_row(cached):
                    row = cached
                    print(
                        f"  {solver.label:<18} {scenario.key:<12} "
                        f"{grouping.label:<6} batch_size={bs:>6} ... (cached) "
                        f"{format_stats(row)}"
                    )
                else:
                    row = collect_row(
                        gpu_name, solver, scenario, grouping, y0, params, int(bs)
                    )
                    solver_cache[bs_key] = row
                    save_cache(_CACHE_PATH, cache)
                if row is not None:
                    rows.append(row)
            print()
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


def rows_for_case(
    rows: list[dict], solver_key: str, case_key: str
) -> tuple[list[int], list[float], list[float]]:
    case_rows = sorted(
        (
            row
            for row in rows
            if row["solver_key"] == solver_key and row["case_key"] == case_key
        ),
        key=lambda row: row["batch_size"],
    )
    return (
        [row["batch_size"] for row in case_rows],
        [row["solve_time_ms"] for row in case_rows],
        [row["wasted_lane_iteration_ratio"] for row in case_rows],
    )


def plot(rows: list[dict], gpu_name: str, output_path: Path) -> None:
    fig, ax_time = plt.subplots(figsize=(11, 6))
    ax_waste = ax_time.twinx()

    time_handles = []
    waste_handles = []
    for solver in _SOLVERS:
        for scenario, grouping in iter_cases():
            case_key = f"{scenario.key}_{grouping.key}"
            xs, times_ms, wasted = rows_for_case(rows, solver.key, case_key)
            if not xs:
                continue
            label = f"{solver.label} / {scenario.key} / {grouping.label}"
            (time_line,) = ax_time.plot(
                xs,
                times_ms,
                color=solver.color,
                linestyle=grouping.linestyle,
                marker=grouping.marker,
                label=f"time: {label}",
            )
            (waste_line,) = ax_waste.plot(
                xs,
                wasted,
                color=solver.color,
                linestyle=":",
                marker=grouping.marker,
                alpha=0.85,
                label=f"waste: {label}",
            )
            time_handles.append(time_line)
            waste_handles.append(waste_line)

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Batch size")
    ax_time.set_ylabel("Solve time (ms)")
    ax_waste.set_ylabel("Wasted lane-iteration ratio")
    ax_waste.set_ylim(0.0, 1.0)
    ax_time.set_title(
        f"Rodas5 divergence - Robertson y0 + params variation - {gpu_name}"
    )
    ax_time.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_time.set_xticks(_BATCH_SIZES)
    ax_time.set_xticklabels([str(bs) for bs in _BATCH_SIZES], rotation=45, ha="right")
    handles = time_handles + waste_handles
    labels = [handle.get_label() for handle in handles]
    ax_time.legend(handles, labels, loc="upper left", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)

    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, csv_path)
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
