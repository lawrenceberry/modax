"""Warp-divergence benchmark: KenCarp5 on Brusselator, varying batch_size.

Measures wall time and the lane work wasted when trajectories with different
adaptive step counts share a while-loop batch. The local KenCarp5 solver
exposes per-trajectory ``accepted_steps`` / ``rejected_steps`` and per-batch
``batch_loop_iterations`` via ``return_stats=True``, which we use to compute
the total and wasted lane-iterations.

Sweeps two ``linear=`` settings (``True`` / ``False``) so the warp-divergence
shape of both implicit-stage strategies can be compared.

Usage:
    uv run python scripts/13_kencarp5_brusselator_pure_jax_batching/main.py
"""

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.systems.python import brusselator
from scripts.benchmark_common import get_gpu_name, gpu_slug
from solvers.kencarp5 import solve as kencarp5_solve

jax.config.update("jax_enable_x64", True)

_N_GRID = 32
_DIM = 2 * _N_GRID
_N_TRAJ = 4096
_T_SPAN = brusselator.TIMES
_N_RUNS = 3
_BATCH_SIZES = (64, 256, 1024, 4096)
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_EX_FN, _IM_FN, _ODE_FN, _ = brusselator.make_system(_N_GRID)


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


@dataclass(frozen=True)
class LinearSetting:
    key: str
    label: str
    linear: bool


_SCENARIOS = (
    Scenario("identical", "identical y0", "#2b7be0"),
    Scenario("divergent", "divergent y0", "#e02b2b"),
)
_GROUPINGS = (
    Grouping("random", "random", "-", "o"),
    Grouping("sorted", "sorted", "--", "s"),
)
_LINEAR_SETTINGS = (
    LinearSetting("lin", "linear=True", True),
    LinearSetting("nl", "linear=False", False),
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


def load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def make_initial_data(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Per-trajectory ICs + reaction-rate params for this scenario."""
    return brusselator.make_scenario(
        scenario.key,
        _N_GRID,
        _N_TRAJ,
        seed=42,
        divergence=1.0,
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
    y0: np.ndarray, params: np.ndarray, batch_size: int | None, linear: bool
):
    return kencarp5_solve(
        _EX_FN,
        _IM_FN,
        jnp.asarray(y0, dtype=jnp.float64),
        _T_SPAN,
        jnp.asarray(params),
        batch_size=batch_size,
        linear=linear,
        return_stats=True,
        **_SOLVER_KWARGS,
    )


def time_solve_with_stats(
    y0: np.ndarray, params: np.ndarray, batch_size: int | None, linear: bool
) -> tuple[float, dict]:
    result = solve_with_stats(y0, params, batch_size, linear)
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        result = solve_with_stats(y0, params, batch_size, linear)
        jax.block_until_ready(result)
    ms = (time.perf_counter() - t0) / _N_RUNS * 1000
    _, stats = result
    return ms, summarize_stats(stats)


def active_attempt_order(y0: np.ndarray, params: np.ndarray) -> np.ndarray:
    _, stats = solve_with_stats(y0, params, batch_size=None, linear=True)
    jax.block_until_ready(stats)
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    attempts = accepted_steps + rejected_steps
    return np.argsort(attempts, kind="stable")


def order_data(
    y0: np.ndarray,
    params: np.ndarray,
    scenario: Scenario,
    grouping: Grouping,
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
            for linear in _LINEAR_SETTINGS:
                yield scenario, grouping, linear


def case_key(scenario: Scenario, grouping: Grouping, linear: LinearSetting) -> str:
    return f"{scenario.key}_{grouping.key}_{linear.key}"


def collect_row(
    gpu_name: str,
    scenario: Scenario,
    grouping: Grouping,
    linear: LinearSetting,
    y0: np.ndarray,
    params: np.ndarray,
    batch_size: int,
) -> dict | None:
    print(
        f"  {scenario.label:<14} {grouping.label:<6} {linear.label:<14} "
        f"batch_size={batch_size:>6} ...",
        end=" ",
        flush=True,
    )
    try:
        ms, stats = time_solve_with_stats(y0, params, batch_size, linear.linear)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    row = {
        "gpu": gpu_name,
        "case_key": case_key(scenario, grouping, linear),
        "batch_size": int(batch_size),
        "solve_time_ms": float(ms),
        **stats,
    }
    print(format_stats(row), flush=True)
    return row


def run_benchmarks(gpu_name: str, cache: dict) -> list[dict]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[dict] = []
    last_scenario_grouping: tuple[str, str] | None = None
    base_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ordered_data: tuple[np.ndarray, np.ndarray] | None = None
    for scenario, grouping, linear in iter_cases():
        if scenario.key not in base_data:
            base_data[scenario.key] = make_initial_data(scenario)
        cur = (scenario.key, grouping.key)
        if cur != last_scenario_grouping:
            ordered_data = order_data(
                *base_data[scenario.key], scenario, grouping
            )
            last_scenario_grouping = cur
            print(f"{scenario.label} / {grouping.label}:")

        assert ordered_data is not None
        y0, params = ordered_data
        ckey = case_key(scenario, grouping, linear)
        case_cache = gpu_cache.setdefault(ckey, {})
        for bs in _BATCH_SIZES:
            bs_key = str(int(bs))
            cached = case_cache.get(bs_key)
            if is_complete_row(cached):
                row = cached
                print(
                    f"  {scenario.label:<14} {grouping.label:<6} "
                    f"{linear.label:<14} batch_size={bs:>6} ... (cached) "
                    f"{format_stats(row)}"
                )
            else:
                row = collect_row(
                    gpu_name, scenario, grouping, linear, y0, params, int(bs)
                )
                case_cache[bs_key] = row
                save_cache(cache)
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


def rows_for_case(rows: list[dict], ckey: str):
    case_rows = sorted(
        (row for row in rows if row["case_key"] == ckey),
        key=lambda row: row["batch_size"],
    )
    return (
        [row["batch_size"] for row in case_rows],
        [row["solve_time_ms"] for row in case_rows],
        [row["wasted_lane_iteration_ratio"] for row in case_rows],
    )


def plot(rows: list[dict], gpu_name: str, output_path: Path) -> None:
    fig, ax_time = plt.subplots(figsize=(12, 6))
    ax_waste = ax_time.twinx()

    time_handles = []
    waste_handles = []
    for scenario, grouping, linear in iter_cases():
        ckey = case_key(scenario, grouping, linear)
        xs, times_ms, wasted = rows_for_case(rows, ckey)
        if not xs:
            continue
        label = f"{scenario.label} / {grouping.label} / {linear.label}"
        # Use solid lines for linear=True, dashed for linear=False; scenarios
        # encoded in color, groupings in marker.
        linestyle = "-" if linear.key == "lin" else "--"
        (time_line,) = ax_time.plot(
            xs,
            times_ms,
            color=scenario.color,
            linestyle=linestyle,
            marker=grouping.marker,
            alpha=0.9,
            label=f"time: {label}",
        )
        (waste_line,) = ax_waste.plot(
            xs,
            wasted,
            color=scenario.color,
            linestyle=":",
            marker=grouping.marker,
            alpha=0.5,
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
        f"KenCarp5 batching — Brusselator (n_grid={_N_GRID}, {_N_TRAJ} traj) — "
        f"{gpu_name}"
    )
    ax_time.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_time.set_xticks(_BATCH_SIZES)
    ax_time.set_xticklabels([str(bs) for bs in _BATCH_SIZES], rotation=45, ha="right")
    handles = time_handles + waste_handles
    labels = [handle.get_label() for handle in handles]
    ax_time.legend(handles, labels, loc="upper left", fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    slug = gpu_slug(gpu_name)
    print(f"GPU: {gpu_name}\n")

    cache = load_cache()
    rows = run_benchmarks(gpu_name, cache)

    csv_path = _SCRIPT_DIR / f"results-{slug}.csv"
    save_csv(rows, csv_path)

    plot_path = _SCRIPT_DIR / f"plot-{slug}.png"
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
