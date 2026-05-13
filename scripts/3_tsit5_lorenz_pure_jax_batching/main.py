"""Warp-divergence benchmark: Tsit5 on Lorenz, varying batch_size.

This benchmark measures both wall time and the lane work wasted when
trajectories with different adaptive step counts share a while-loop batch.
Timing alone can stay fairly flat when fixed GPU/JAX overhead dominates, so
the wasted lane-iteration ratio is recorded as the direct mechanism signal.

Initial condition design
------------------------
The Lorenz system is run with rho=0.5, which places it below the pitchfork
bifurcation at rho=1. In this regime the origin is the unique globally stable
fixed point, so every trajectory decays exponentially toward (0, 0, 0) with
slowest eigenvalue lambda ~ -0.47.

Step count is therefore strictly monotone in distance from the origin: a
trajectory that starts far away must cross a large region of phase space with
significant ODE velocities before the state becomes small enough for Tsit5 to
take large steps.

The hard base IC, lorenz.Y0 = [1000, -500, 500], is fixed analytically as a
point far from the origin. ``divergence=0`` places every trajectory there, so
all share the same (maximum) step count and the batch shows zero divergence.
The ic_large scenario scales each trajectory as ``Y0 * (1 - t)`` for
``t ~ U(0, 1)``, moving it radially toward the origin. Because distance from
the origin is the sole driver of difficulty, this construction guarantees that
every ic_large trajectory is strictly easier than the identical base, and the
step counts span a continuous range from near-maximum (t -> 0) down to
near-zero (t -> 1).

Usage:
    uv run python scripts/2b_tsit5_jax_divergence/main.py
"""

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.systems.python import lorenz
from scripts.benchmark_common import (
    TIMEOUT_ERROR,
    BenchmarkTimeoutError,
    get_gpu_name,
    gpu_slug,
    is_timeout,
    load_cache,
    save_cache,
    timed_solve,
    timeout_cache_entry,
)
from solvers.tsit5 import solve as tsit5_solve

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 400_000
_T_SPAN = (0.0, 5.0)
_N_RUNS = 5
_BATCH_SIZES = (5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000)
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_RHO = 0.5
_PARAMS = jnp.array([_RHO], dtype=jnp.float64)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class Grouping:
    key: str
    label: str
    linestyle: str
    marker: str


_SCENARIOS = (
    Scenario("ic_identical", "identical y0", "#2b7be0"),
    Scenario("ic_large", "large dy0", "#e02b2b"),
)

_GROUPINGS = (
    Grouping("random", "random", "-", "o"),
    Grouping("sorted", "sorted", "--", "s"),
)

_CSV_FIELDS = (
    "gpu",
    "case_key",
    "batch_size",
    "solve_time_ms",
    "total_lane_iterations",
    "wasted_lane_iterations",
    "min_batch_loop_iterations",
    "max_batch_loop_iterations",
    "wasted_lane_iteration_ratio",
)


def make_initial_conditions(
    scenario: Scenario, size: int, seed: int = 42
) -> np.ndarray:
    divergence = 0.0 if scenario.key == "ic_identical" else 1.0
    return np.asarray(
        lorenz.make_initial_conditions(size, seed=seed, divergence=divergence)
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


def solve_with_stats(y0: np.ndarray, batch_size: int | None):
    return tsit5_solve(
        lorenz.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=_PARAMS,
        batch_size=batch_size,
        return_stats=True,
        **_SOLVER_KWARGS,
    )


def time_solve_with_stats(y0: np.ndarray, batch_size: int | None) -> tuple[float, dict]:
    result = solve_with_stats(y0, batch_size)
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        result = solve_with_stats(y0, batch_size)
        jax.block_until_ready(result)
    ms = (time.perf_counter() - t0) / _N_RUNS * 1000
    _, stats = result
    return ms, summarize_stats(stats)


def active_attempt_order(y0: np.ndarray) -> np.ndarray:
    _, stats = solve_with_stats(y0, batch_size=None)
    jax.block_until_ready(stats)
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    attempts = accepted_steps + rejected_steps
    return np.argsort(attempts, kind="stable")


def order_initial_conditions(
    y0: np.ndarray, scenario: Scenario, grouping: Grouping
) -> np.ndarray:
    if grouping.key == "sorted":
        return y0[active_attempt_order(y0)]
    seed = sum(ord(c) for c in f"{scenario.key}:{grouping.key}")
    rng = np.random.default_rng(seed)
    return y0[rng.permutation(y0.shape[0])]


def is_complete_row(value) -> bool:
    return isinstance(value, dict) and all(field in value for field in _CSV_FIELDS)


def iter_cases():
    for scenario in _SCENARIOS:
        for grouping in _GROUPINGS:
            if scenario.key == "ic_identical" and grouping.key == "sorted":
                continue
            yield scenario, grouping


def collect_row(
    gpu_name: str,
    scenario: Scenario,
    grouping: Grouping,
    y0: np.ndarray,
    batch_size: int,
) -> dict | None:
    print(
        f"  {scenario.label:<18} {grouping.label:<6} batch_size={batch_size:>6} ...",
        end=" ",
        flush=True,
    )
    try:
        ms, stats = timed_solve(lambda: time_solve_with_stats(y0, batch_size))
    except BenchmarkTimeoutError:
        print(TIMEOUT_ERROR)
        return timeout_cache_entry()
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    row = {
        "gpu": gpu_name,
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
        base_y0 = make_initial_conditions(scenario, _N_TRAJ)
        case_key = f"{scenario.key}_{grouping.key}"
        case_cache = gpu_cache.setdefault(case_key, {})
        print(f"{scenario.label} / {grouping.label}:")
        try:
            y0 = timed_solve(
                lambda: order_initial_conditions(base_y0, scenario, grouping)
            )
        except BenchmarkTimeoutError:
            print(f"  ordering ... {TIMEOUT_ERROR}")
            for bs in _BATCH_SIZES:
                case_cache.setdefault(str(int(bs)), timeout_cache_entry())
            save_cache(_CACHE_PATH, cache)
            print()
            continue
        for bs in _BATCH_SIZES:
            bs_key = str(int(bs))
            cached = case_cache.get(bs_key)
            if is_complete_row(cached) or is_timeout(cached):
                row = cached
                text = TIMEOUT_ERROR if is_timeout(row) else format_stats(row)
                print(
                    f"  {scenario.label:<18} {grouping.label:<6} "
                    f"batch_size={bs:>6} ... (cached) "
                    f"{text}"
                )
            else:
                row = collect_row(gpu_name, scenario, grouping, y0, int(bs))
                case_cache[bs_key] = row
                save_cache(_CACHE_PATH, cache)
            if row is not None and not is_timeout(row):
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
    rows: list[dict], case_key: str
) -> tuple[list[int], list[float], list[float]]:
    case_rows = sorted(
        (row for row in rows if row["case_key"] == case_key),
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
    for scenario, grouping in iter_cases():
        case_key = f"{scenario.key}_{grouping.key}"
        xs, times_ms, wasted = rows_for_case(rows, case_key)
        if not xs:
            continue
        label = f"{scenario.label} / {grouping.label}"
        (time_line,) = ax_time.plot(
            xs,
            times_ms,
            color=scenario.color,
            linestyle=grouping.linestyle,
            marker=grouping.marker,
            label=f"time: {label}",
        )
        (waste_line,) = ax_waste.plot(
            xs,
            wasted,
            color=scenario.color,
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
    ax_time.set_title(f"Tsit5 divergence — Lorenz (rho={_RHO}) — {gpu_name}")
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
    slug = gpu_slug(gpu_name)
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)

    csv_path = _SCRIPT_DIR / f"results-{slug}.csv"
    save_csv(rows, csv_path)

    plot_path = _SCRIPT_DIR / f"plot-{slug}.png"
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
