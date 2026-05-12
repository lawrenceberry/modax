"""Dimensionality scaling benchmark on the non-stiff coupled VDP lattice (Tsit5).

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 trajectories and records solve time for the local JAX
Tsit5 solver, the numba-cuda Tsit5 kernel, Diffrax Tsit5, and Julia Tsit5 with
both DiffEqGPU ensemble backends. EnsembleGPUKernel failures (expected for
large dimensions) are stored as null and omitted from the plot. Runs both
"identical" and "divergent" scenarios; outputs a CSV and log-log plot per
scenario, named after the GPU and scenario.

Uses the non-stiff coupled VDP variant (mu = 1.0) so that explicit Tsit5
remains an appropriate solver.

Usage:
    uv run python scripts/4_tsit5_vdp_dimensionality/main.py
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

from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.systems.python import coupled_vdp_lattice
from scripts.benchmark_common import (
    get_gpu_name,
    gpu_slug,
    load_cache,
    save_cache,
    time_blocked,
)
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import clear_caches as tsit5ckn_clear_caches
from solvers.tsit5ckn import prepare_solve as tsit5ckn_prepare_solve
from solvers.tsit5ckn import run_prepared as tsit5ckn_run_prepared

jax.config.update("jax_enable_x64", True)

_MU_NONSTIFF = 1.0
_D = 10.0
_OMEGA = 1.0

_T_SPAN = coupled_vdp_lattice.TIMES
_ENSEMBLE_SIZE = 1000
_N_RUNS = 10
_JULIA_BACKENDS = ("EnsembleGPUArray", "EnsembleGPUKernel")

_DIMENSIONS = (2, 4, 6, 8, 10, 12, 16, 32, 64, 128)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

_LOCAL_SOLVER_DEFS = [
    ("local_tsit5", "my tsit5", "#2b7be0", "o"),
]
_JAX_SOLVER_DEFS = [
    ("diffrax_tsit5", "diffrax tsit5", "#2ba84a", "s", diffrax_tsit5_solve),
]
_CUSTOM_KERNEL_SOLVER_DEFS = [
    ("tsit5ckn", "numba-cuda tsit5", "#f0a202", "P"),
]
_JULIA_SOLVER_DEFS = [
    ("julia_tsit5", "julia tsit5", julia_tsit5_solve, "#9b59b6"),
]
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
    mu = 1.0
    diffusion = 10.0
    for k in range(n_osc):
        kp1 = (k + 1) % n_osc
        km1 = (k + n_osc - 1) % n_osc
        xk = y[i, 2 * k]
        vk = y[i, 2 * k + 1]
        lap = y[i, 2 * kp1] - 2.0 * xk + y[i, 2 * km1]
        dy[i, 2 * k] = vk
        dy[i, 2 * k + 1] = scale * mu * (1.0 - xk * xk) * vk - xk + diffusion * lap


def time_local_tsit5(dim: int, *, scenario: str) -> float:
    n_osc = dim // 2
    ode_fn, _ = coupled_vdp_lattice.make_system(n_osc, mu=_MU_NONSTIFF)
    y0_batch, params = coupled_vdp_lattice.make_scenario(
        scenario, n_osc, _ENSEMBLE_SIZE
    )
    y0_batch = jnp.asarray(y0_batch)
    params = jnp.asarray(params)

    def run():
        return tsit5_solve(
            ode_fn,
            y0_batch,
            _T_SPAN,
            params,
            **_SOLVER_KWARGS,
        )

    ms, _ = time_blocked(run, _N_RUNS)
    return ms


def time_custom_kernel_tsit5(dim: int, *, scenario: str) -> float:
    n_osc = dim // 2
    y0_batch, scale_params = coupled_vdp_lattice.make_scenario(
        scenario, n_osc, _ENSEMBLE_SIZE
    )
    params = np.column_stack(
        [
            np.full(_ENSEMBLE_SIZE, float(n_osc), dtype=np.float64),
            scale_params[:, 0],
        ]
    )
    prepared = tsit5ckn_prepare_solve(
        ode_fn_vdp_numba,
        y0=y0_batch,
        t_span=_T_SPAN,
        params=params,
        **_SOLVER_KWARGS,
    )

    def run():
        return tsit5ckn_run_prepared(prepared, copy_solution=False)

    try:
        ms, _ = time_blocked(run, _N_RUNS)
    finally:
        # Release the workspace + compiled kernel for this dim before the next
        # dim allocates its own — they're keyed on n_vars and don't otherwise
        # get freed, which causes OOM at large dimensions.
        tsit5ckn_clear_caches()
    return ms


def time_diffrax(dim: int, solve_fn, *, scenario: str) -> float:
    n_osc = dim // 2
    ode_fn, _ = coupled_vdp_lattice.make_system(n_osc, mu=_MU_NONSTIFF)
    y0_batch, params = coupled_vdp_lattice.make_scenario(
        scenario, n_osc, _ENSEMBLE_SIZE
    )
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
    y0_batch, params = coupled_vdp_lattice.make_scenario(
        scenario, n_osc, _ENSEMBLE_SIZE
    )
    result = solve._julia_solve_with_timing(
        "coupled_vdp_lattice",
        y0_batch,
        _T_SPAN,
        params,
        system_config={
            "n_osc": n_osc,
            "mu": _MU_NONSTIFF,
            "d": _D,
            "omega": _OMEGA,
        },
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
            timing_fn=lambda dim, sc=scenario: time_local_tsit5(dim, scenario=sc),
        )
        for key, label, color, marker in _LOCAL_SOLVER_DEFS
    ]
    specs.extend(
        SolverSpec(
            key=key,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            timing_fn=lambda dim, sc=scenario: time_custom_kernel_tsit5(
                dim, scenario=sc
            ),
        )
        for key, label, color, marker in _CUSTOM_KERNEL_SOLVER_DEFS
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
        f"Tsit5 dimensionality — {scenario} — coupled VDP lattice "
        f"(μ={_MU_NONSTIFF}) — {gpu_name}"
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

    cache = load_cache(_CACHE_PATH)

    for scenario in coupled_vdp_lattice.SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===\n")
        specs = make_solver_specs(scenario)
        rows = run_benchmarks(specs, gpu_name, cache, scenario)
        csv_path, plot_path = _scenario_output_paths(gpu_name, scenario)
        save_csv(rows, csv_path)
        plot(rows, specs, gpu_name, plot_path, scenario)


if __name__ == "__main__":
    main()
