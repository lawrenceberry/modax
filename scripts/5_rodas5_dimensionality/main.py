"""Dimensionality scaling benchmark on the coupled VDP lattice system.

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 trajectories and records solve time for the local Rodas5
solver with fp32/fp64 LU precision, Diffrax Kvaerno5, and Julia Rodas5 with
both DiffEqGPU ensemble backends. EnsembleGPUKernel failures (expected for
large dimensions) are stored as null and omitted from the plot. Outputs a CSV
and a log-log plot named after the GPU.

Usage:
    uv run python scripts/5_rodas5_dimensionality/main.py
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

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve
from reference.systems.python import coupled_vdp_lattice
from scripts.benchmark_common import (
    get_gpu_name,
    gpu_slug,
    load_cache,
    output_paths,
    save_cache,
    time_blocked,
)
from solvers.rodas5 import solve as rodas5_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = coupled_vdp_lattice.TIMES
_ENSEMBLE_SIZE = 1000
_N_RUNS = 10
_JULIA_BACKENDS = ("EnsembleGPUArray", "EnsembleGPUKernel")

# Powers of 2 from 2 to 128 (n_osc = dim // 2: 1, 2, 4, 8, 16, 32, 64)
_DIMENSIONS = (2, 4, 6, 8, 10, 12, 16, 32, 64, 128)

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
    timing_fn: Callable[[int], float]


def time_local_rodas5(dim: int, *, lu_precision: str) -> float:
    n_osc = dim // 2
    ode_fn, y0 = coupled_vdp_lattice.make_system(n_osc)
    y0_batch = jnp.broadcast_to(y0, (_ENSEMBLE_SIZE, dim))
    params = jnp.ones((_ENSEMBLE_SIZE, 1), dtype=jnp.float64)

    def run():
        return rodas5_solve(
            ode_fn,
            y0_batch,
            _T_SPAN,
            params,
            lu_precision=lu_precision,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_diffrax(dim: int, solve_fn) -> float:
    n_osc = dim // 2
    ode_fn, y0 = coupled_vdp_lattice.make_system(n_osc)
    params = jnp.ones((_ENSEMBLE_SIZE, 1), dtype=jnp.float64)

    def run():
        return solve_fn(
            ode_fn,
            y0=y0,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_julia_solver(dim: int, solve, *, ensemble_backend: str) -> float:
    n_osc = dim // 2
    _, y0 = coupled_vdp_lattice.make_system(n_osc)
    params = np.ones((_ENSEMBLE_SIZE, 1), dtype=np.float64)
    result = solve._julia_solve_with_timing(
        "coupled_vdp_lattice",
        np.asarray(y0),
        _T_SPAN,
        params,
        system_config={"n_osc": n_osc},
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
            timing_fn=lambda dim, precision=lu_precision: time_local_rodas5(
                dim, lu_precision=precision
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
            timing_fn=lambda dim, fn=fn: time_diffrax(dim, fn),
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
                    timing_fn=lambda dim, s=solve, b=backend: time_julia_solver(
                        dim, s, ensemble_backend=b
                    ),
                )
            )
    return specs


def collect_timing(spec: SolverSpec, dim: int) -> float | None:
    print(f"  {spec.label:<28} dim={dim:>4} ...", end=" ", flush=True)
    try:
        ms = spec.timing_fn(dim)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print(f"{ms:.1f} ms")
    return ms


_Row = tuple[str, str, int, float | None]


def drop_none(rows: list[_Row], solver_key: str) -> tuple[list[int], list[float]]:
    pairs = [
        (dim, ms) for key, _, dim, ms in rows if key == solver_key and ms is not None
    ]
    if not pairs:
        return [], []
    dims, times = zip(*pairs)
    return list(dims), list(times)


def run_benchmarks(specs: list[SolverSpec], gpu_name: str, cache: dict) -> list[_Row]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[_Row] = []
    for spec in specs:
        print(f"\n{spec.label}:")
        solver_cache = gpu_cache.setdefault(spec.key, {})
        for dim in _DIMENSIONS:
            dim_key = str(dim)
            if dim_key in solver_cache:
                ms = solver_cache[dim_key]
                ms_text = f"{ms:.1f} ms" if ms is not None else "FAILED"
                print(f"  {spec.label:<28} dim={dim:>4} ... (cached) {ms_text}")
            else:
                ms = collect_timing(spec, dim)
                solver_cache[dim_key] = ms
                save_cache(_CACHE_PATH, cache)
            rows.append((spec.key, spec.label, dim, ms))
        print()
    return rows


def save_csv(rows: list[_Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "dim", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[_Row],
    specs: list[SolverSpec],
    gpu_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in specs:
        dims, times_ms = drop_none(rows, spec.key)
        if not dims:
            continue
        ax.plot(
            dims,
            times_ms,
            marker=spec.marker,
            color=spec.color,
            linestyle=spec.linestyle,
            label=spec.label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ODE dimension")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title(f"Rodas5 dimensionality — coupled VDP lattice — {gpu_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_DIMENSIONS)
    ax.set_xticklabels([str(d) for d in _DIMENSIONS], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)
    specs = make_solver_specs()
    rows = run_benchmarks(specs, gpu_name, cache)

    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, csv_path)
    plot(rows, specs, gpu_name, plot_path)


if __name__ == "__main__":
    main()
