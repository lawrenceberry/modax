"""Tsit5 scaling benchmark on the Lorenz system.

Sweeps ensemble size from 1 to 100k on a log scale and records solve time.
Outputs a CSV and a log-log plot named after the GPU.

Usage:
    uv run python scripts/1_tsit5_scaling/main.py
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solvers.tsit5 import solve as tsit5_solve
from reference.systems.python import lorenz

_T_SPAN = (0.0, 5.0)
_ENSEMBLE_SIZES = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
_N_RUNS = 10

_SCRIPT_DIR = Path(__file__).resolve().parent


def get_gpu_name() -> str:
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        if out:
            return out
    except Exception:
        pass
    try:
        devices = jax.devices("gpu")
        if devices:
            return devices[0].device_kind
    except Exception:
        pass
    return "unknown_GPU"


def gpu_slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def time_solve(params) -> float:
    def run():
        return tsit5_solve(
            lorenz.ode_fn,
            y0=lorenz.Y0,
            t_span=_T_SPAN,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready()

    run()  # warmup / JIT compile
    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        run()
    return (time.perf_counter() - t0) / _N_RUNS * 1000


def main():
    gpu_name = get_gpu_name()
    slug = gpu_slug(gpu_name)
    print(f"GPU: {gpu_name}\n")

    rows: list[tuple[int, float]] = []
    for size in _ENSEMBLE_SIZES:
        params = lorenz.make_params(size)
        print(f"  n={size:>7} ...", end=" ", flush=True)
        ms = time_solve(params)
        print(f"{ms:.1f} ms")
        rows.append((size, ms))

    # CSV
    csv_path = _SCRIPT_DIR / f"results-{slug}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ensemble_size", "solve_time_ms"])
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    # Plot
    sizes = [r[0] for r in rows]
    times_ms = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, times_ms, marker="o", color="#2b7be0", label="tsit5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title(f"Tsit5 scaling — Lorenz — {gpu_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    plot_path = _SCRIPT_DIR / f"plot-{slug}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
