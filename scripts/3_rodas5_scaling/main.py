"""Solver scaling benchmark on the Robertson stiff system.

Sweeps ensemble size from 1 to 100k on a log scale and records solve time for
the local Rodas5 solver with fp32/fp64 LU precision, Diffrax Kvaerno5, and
Julia Rodas5 with both DiffEqGPU ensemble backends. Outputs a CSV and a
log-log plot named after the GPU.

Usage:
    uv run python scripts/3_rodas5_scaling/main.py
"""

import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve
from reference.systems.python import robertson
from solvers.rodas5 import solve as rodas5_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = robertson.TIMES
_ENSEMBLE_SIZES = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
_N_RUNS = 10
_JULIA_BACKENDS = ("EnsembleGPUArray", "EnsembleGPUKernel")

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

# (key, label, color, marker, lu_precision)
_LOCAL_SOLVER_DEFS = [
    ("local_rodas5_fp32_lu", "my rodas5 fp32 LU", "#2b7be0", "o", "fp32"),
    ("local_rodas5_fp64_lu", "my rodas5 fp64 LU", "#e02b2b", "D", "fp64"),
]
# (key, label, color, marker, solve_fn)
_JAX_SOLVER_DEFS = [
    ("diffrax_kvaerno5", "diffrax kvaerno5", "#2ba84a", "s", diffrax_kvaerno5_solve),
]
# (key, label, solve_fn, color)
_JULIA_SOLVER_DEFS = [
    ("julia_rodas5", "julia rodas5", julia_rodas5_solve, "#9b59b6"),
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
    timing_fn: Callable[[object], float]


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


def load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def time_local_rodas5(params, *, lu_precision: str) -> float:
    def run():
        return rodas5_solve(
            robertson.ode_fn,
            robertson.Y0,
            _T_SPAN,
            params,
            lu_precision=lu_precision,
            **_SOLVER_KWARGS,
        ).block_until_ready()

    run()  # warmup / JIT compile
    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        run()
    return (time.perf_counter() - t0) / _N_RUNS * 1000


def time_jax_solver(solve_fn, params) -> float:
    def run():
        return solve_fn(
            robertson.ode_fn,
            y0=robertson.Y0,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        ).block_until_ready()

    run()  # warmup / JIT compile
    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        run()
    return (time.perf_counter() - t0) / _N_RUNS * 1000


def time_julia_solver(solve, params, *, ensemble_backend: str) -> float:
    """Return Julia's internal solve time in ms, excluding subprocess overhead."""
    result = solve._julia_solve_with_timing(
        "robertson",
        robertson.Y0,
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
            timing_fn=lambda params, precision=lu_precision: time_local_rodas5(
                params, lu_precision=precision
            ),
        )
        for key, label, color, marker, lu_precision in _LOCAL_SOLVER_DEFS
    ]
    specs.extend(
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda params, fn=fn: time_jax_solver(fn, params),
        )
        for key, label, color, marker, fn in _JAX_SOLVER_DEFS
    )
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
                    timing_fn=lambda params, s=solve, b=backend: time_julia_solver(
                        s, params, ensemble_backend=b
                    ),
                )
            )
    return specs


def collect_timing(spec: SolverSpec, size: int, params) -> float | None:
    print(f"  {spec.label:<24} n={size:>7} ...", end=" ", flush=True)
    try:
        ms = spec.timing_fn(params)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print(f"{ms:.1f} ms")
    return ms


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


def run_benchmarks(specs: list[SolverSpec], gpu_name: str, cache: dict) -> list[_Row]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[_Row] = []
    for spec in specs:
        print(f"{spec.label}:")
        solver_cache = gpu_cache.setdefault(spec.key, {})
        for size in _ENSEMBLE_SIZES:
            size_key = str(size)
            if size_key in solver_cache:
                ms = solver_cache[size_key]
                ms_text = f"{ms:.1f} ms" if ms is not None else "FAILED"
                print(f"  {spec.label:<24} n={size:>7} ... (cached) {ms_text}")
            else:
                params = robertson.make_params(size)
                ms = collect_timing(spec, size, params)
                solver_cache[size_key] = ms
                save_cache(cache)
            rows.append((spec.key, spec.label, size, ms))
        print()
    return rows


def save_csv(rows: list[_Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "ensemble_size", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[_Row], specs: list[SolverSpec], gpu_name: str, output_path: Path
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
    ax.set_title(f"Rodas5 scaling - Robertson - {gpu_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_ENSEMBLE_SIZES)
    ax.set_xticklabels([str(n) for n in _ENSEMBLE_SIZES], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    slug = gpu_slug(gpu_name)
    print(f"GPU: {gpu_name}\n")

    cache = load_cache()
    specs = make_solver_specs()
    rows = run_benchmarks(specs, gpu_name, cache)

    csv_path = _SCRIPT_DIR / f"results-{slug}.csv"
    save_csv(rows, csv_path)

    plot_path = _SCRIPT_DIR / f"plot-{slug}.png"
    plot(rows, specs, gpu_name, plot_path)


if __name__ == "__main__":
    main()
