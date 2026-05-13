"""Dimensionality scaling benchmark on the coupled VDP lattice system.

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 trajectories and records solve time for the local Rodas5
solver with fp32/fp64 LU precision, Diffrax Kvaerno5, and Julia Rodas5 with
both DiffEqGPU ensemble backends. EnsembleGPUKernel failures (expected for
large dimensions) are stored as null and omitted from the plot. Runs both
"identical" and "divergent" scenarios; outputs a CSV and log-log plot per
scenario, named after the GPU and scenario.

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
from numba import cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve
from reference.systems.python import vdp
from scripts.benchmark_common import (
    collect_timed_timing,
    format_cached_timing,
    get_gpu_name,
    gpu_slug,
    load_cache,
    save_cache,
    time_blocked,
    timing_value_or_none,
)
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import solve as rodas5ckn_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = vdp.TIMES
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
_CUSTOM_KERNEL_SOLVER_DEFS = [
    ("rodas5ckn", "numba-cuda rodas5", "#f0a202", "P", rodas5ckn_solve),
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


def time_local_rodas5(dim: int, *, lu_precision: str, scenario: str) -> float:
    n_osc = dim // 2
    ode_fn, _ = vdp.make_system(n_osc)
    y0_batch, params = vdp.make_scenario(scenario, n_osc, _ENSEMBLE_SIZE)
    y0_batch = jnp.asarray(y0_batch)
    params = jnp.asarray(params)

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


def time_custom_kernel_rodas5(dim: int, solve_fn, *, scenario: str) -> float:
    n_osc = dim // 2
    y0_batch, scale_params = vdp.make_scenario(scenario, n_osc, _ENSEMBLE_SIZE)
    params = np.column_stack(
        [
            np.full(_ENSEMBLE_SIZE, float(n_osc), dtype=np.float64),
            scale_params[:, 0],
        ]
    )

    def run():
        return solve_fn(
            ode_fn_vdp_numba,
            jac_fn_vdp_numba,
            y0=y0_batch,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_diffrax(dim: int, solve_fn, *, scenario: str) -> float:
    n_osc = dim // 2
    ode_fn, _ = vdp.make_system(n_osc)
    y0_batch, params = vdp.make_scenario(scenario, n_osc, _ENSEMBLE_SIZE)
    y0_batch = jnp.asarray(y0_batch)
    params = jnp.asarray(params)

    def run():
        return solve_fn(
            ode_fn,
            y0=y0_batch,
            t_span=_T_SPAN,
            params=params,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_julia_solver(
    dim: int, solve, *, ensemble_backend: str, scenario: str
) -> float:
    n_osc = dim // 2
    y0_batch, params = vdp.make_scenario(scenario, n_osc, _ENSEMBLE_SIZE)
    result = solve._julia_solve_with_timing(
        "vdp",
        y0_batch,
        _T_SPAN,
        params,
        system_config={"n_osc": n_osc},
        ensemble_backend=ensemble_backend,
        **_SOLVER_KWARGS,
    )
    return result.solve_time_s * 1000


def make_solver_specs(scenario: str) -> list[SolverSpec]:
    specs = [
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda dim, precision=lu_precision, sc=scenario: (
                time_local_rodas5(dim, lu_precision=precision, scenario=sc)
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
            timing_fn=lambda dim, fn=fn, sc=scenario: time_custom_kernel_rodas5(
                dim, fn, scenario=sc
            ),
        )
        for key, label, color, marker, fn in _CUSTOM_KERNEL_SOLVER_DEFS
    )
    specs.extend(
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda dim, fn=fn, sc=scenario: time_diffrax(
                dim, fn, scenario=sc
            ),
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
                    timing_fn=lambda dim, s=solve, b=backend, sc=scenario: (
                        time_julia_solver(dim, s, ensemble_backend=b, scenario=sc)
                    ),
                )
            )
    return specs


def collect_timing(spec: SolverSpec, dim: int):
    return collect_timed_timing(
        spec.label,
        f"dim={dim:>4}",
        lambda: spec.timing_fn(dim),
        label_width=28,
    )


_Row = tuple[str, str, int, float | None]


def drop_none(rows: list[_Row], solver_key: str) -> tuple[list[int], list[float]]:
    pairs = [
        (dim, ms) for key, _, dim, ms in rows if key == solver_key and ms is not None
    ]
    if not pairs:
        return [], []
    dims, times = zip(*pairs)
    return list(dims), list(times)


def _migrate_cache(cache: dict) -> dict:
    for gpu, gpu_data in cache.items():
        first_val = next(iter(gpu_data.values()), None)
        if isinstance(first_val, dict) and not any(
            k in vdp.SCENARIOS for k in gpu_data
        ):
            cache[gpu] = {"identical": gpu_data}
    return cache


def run_benchmarks(
    specs: list[SolverSpec], gpu_name: str, cache: dict, scenario: str
) -> list[_Row]:
    scenario_cache = cache.setdefault(gpu_name, {}).setdefault(scenario, {})
    rows: list[_Row] = []
    for spec in specs:
        print(f"\n{spec.label}:")
        solver_cache = scenario_cache.setdefault(spec.key, {})
        for dim in _DIMENSIONS:
            dim_key = str(dim)
            if dim_key in solver_cache:
                ms = solver_cache[dim_key]
                ms_text = format_cached_timing(ms)
                print(f"  {spec.label:<28} dim={dim:>4} ... (cached) {ms_text}")
            else:
                ms = collect_timing(spec, dim)
                solver_cache[dim_key] = ms
                save_cache(_CACHE_PATH, cache)
            rows.append((spec.key, spec.label, dim, timing_value_or_none(ms)))
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
    scenario: str,
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
    ax.set_title(
        f"Rodas5 dimensionality — {scenario} — coupled VDP lattice — {gpu_name}"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_DIMENSIONS)
    ax.set_xticklabels([str(d) for d in _DIMENSIONS], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def _scenario_output_paths(gpu_name: str, scenario: str) -> tuple[Path, Path]:
    slug = gpu_slug(gpu_name)
    return (
        _SCRIPT_DIR / f"results-{slug}-{scenario}.csv",
        _SCRIPT_DIR / f"plot-{slug}-{scenario}.png",
    )


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = _migrate_cache(load_cache(_CACHE_PATH))

    for scenario in vdp.SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===\n")
        specs = make_solver_specs(scenario)
        rows = run_benchmarks(specs, gpu_name, cache, scenario)
        csv_path, plot_path = _scenario_output_paths(gpu_name, scenario)
        save_csv(rows, csv_path)
        plot(rows, specs, gpu_name, plot_path, scenario)


if __name__ == "__main__":
    main()
