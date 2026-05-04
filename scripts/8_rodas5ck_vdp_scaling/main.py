"""Benchmark custom-kernel Rodas5 solvers on the coupled VdP lattice.

Sweeps ODE dimension from 4 to 64 (n_osc = 2 to 32) and records compilation
time (first call) and solve time (mean of N_RUNS calls) for the pure JAX
Rodas5 solver plus the numba-cuda custom-kernel backend. Outputs a CSV and a
two-panel log-log plot named after the GPU.

Usage:
    uv run python scripts/8_rodas5ck_vdp_scaling/main.py
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
)
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import (
    prepare_solve as rodas5ckn_prepare_solve,
)
from solvers.rodas5ckn import (
    run_prepared as rodas5ckn_run_prepared,
)
from solvers.rodas5ckn import solve as rodas5ckn_solve

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 1_000
_N_RUNS = 10
_DIMS = (4, 8, 16, 32, 64)  # n_osc = 2, 4, 8, 16, 32
_T_SPAN = coupled_vdp_lattice.TIMES
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_COLORS = {
    "rodas5": "#7b3fb2",
    "rodas5ckn": "#2ba84a",
}

_MU = 100.0
_D = 10.0
_OMEGA = 1.0


# ---------------------------------------------------------------------------
# numba-cuda backend — module-level @cuda.jit(device=True)
# ---------------------------------------------------------------------------


@cuda.jit(device=True)
def ode_fn_vdp_numba(y, t, p, dy, i):
    n_osc = int(p[i, 0])
    scale = p[i, 1]
    MU = 100.0
    D = 10.0
    for k in range(n_osc):
        kp1 = (k + 1) % n_osc
        km1 = (k + n_osc - 1) % n_osc
        xk = y[i, 2 * k]
        vk = y[i, 2 * k + 1]
        lap = y[i, 2 * kp1] - 2.0 * xk + y[i, 2 * km1]
        dy[i, 2 * k] = vk
        dy[i, 2 * k + 1] = scale * MU * (1.0 - xk * xk) * vk - xk + D * lap


@cuda.jit(device=True)
def jac_fn_vdp_numba(y, t, p, jac, i):
    n_osc = int(p[i, 0])
    n_vars = 2 * n_osc
    scale = p[i, 1]
    MU = 100.0
    D = 10.0
    for r in range(n_vars):
        for c in range(n_vars):
            jac[i, r, c] = 0.0
    for k in range(n_osc):
        kp1 = (k + 1) % n_osc
        km1 = (k + n_osc - 1) % n_osc
        xk = y[i, 2 * k]
        vk = y[i, 2 * k + 1]
        jac[i, 2 * k, 2 * k + 1] = 1.0
        jac[i, 2 * k + 1, 2 * k] = scale * MU * (-2.0 * xk) * vk - 1.0 - 2.0 * D
        jac[i, 2 * k + 1, 2 * k + 1] = scale * MU * (1.0 - xk * xk)
        if kp1 == km1:  # n_osc == 2 edge case
            jac[i, 2 * k + 1, 2 * kp1] = 2.0 * D
        else:
            jac[i, 2 * k + 1, 2 * kp1] = D
            jac[i, 2 * k + 1, 2 * km1] = D


# ---------------------------------------------------------------------------
# Solver specs and input construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverSpec:
    key: str
    label: str
    solve_fn: Callable
    kind: str


_SOLVERS = (
    SolverSpec("rodas5", "pure JAX rodas5.py", rodas5_solve, kind="jax"),
    SolverSpec("rodas5ckn", "numba-cuda", rodas5ckn_solve, kind="custom_kernel"),
)


def make_inputs(spec: SolverSpec, dim: int):
    n_osc = dim // 2
    y0_single = np.array([2.0, 0.0] * n_osc, dtype=np.float64)
    y0_np = np.broadcast_to(y0_single, (_N_TRAJ, dim)).copy()

    if spec.kind == "jax":
        ode_fn, _ = coupled_vdp_lattice.make_system(n_osc)
        params = jnp.ones((_N_TRAJ, 1), dtype=jnp.float64)
        return ode_fn, None, jnp.asarray(y0_np), params

    params_np = np.column_stack(
        [
            np.full(_N_TRAJ, float(n_osc), dtype=np.float64),
            np.ones(_N_TRAJ, dtype=np.float64),
        ]
    )
    return ode_fn_vdp_numba, jac_fn_vdp_numba, y0_np, params_np


def run_solver(spec: SolverSpec, ode_fn, jac_fn, y0, params):
    if spec.kind == "jax":
        return spec.solve_fn(
            ode_fn, y0=y0, t_span=_T_SPAN, params=params, **_SOLVER_KWARGS
        )
    return spec.solve_fn(
        ode_fn, jac_fn, y0=y0, t_span=_T_SPAN, params=params, **_SOLVER_KWARGS
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def block_until_ready(value) -> None:
    if isinstance(value, jax.Array):
        jax.block_until_ready(value)
    elif isinstance(value, tuple):
        for item in value:
            block_until_ready(item)
    elif isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
    # numpy arrays from custom kernels are already synchronised on return


def time_with_compile(run_fn: Callable, n_runs: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    block_until_ready(run_fn())
    compile_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        block_until_ready(run_fn())
    solve_ms = (time.perf_counter() - t0) / n_runs * 1000

    return compile_ms, solve_ms


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

# (key, label, dim, compile_ms | None, solve_ms | None, error)
_Row = tuple[str, str, int, float | None, float | None, str]


def collect_timing(
    spec: SolverSpec, dim: int
) -> tuple[float | None, float | None, str]:
    print(f"  {spec.label:<14} dim={dim:>3} ...", end=" ", flush=True)
    try:
        ode_fn, jac_fn, y0, params = make_inputs(spec, dim)

        if spec.key == "rodas5ckn":
            prepared = rodas5ckn_prepare_solve(
                ode_fn,
                jac_fn,
                y0=y0,
                t_span=_T_SPAN,
                params=params,
                **_SOLVER_KWARGS,
            )

            def run():
                return rodas5ckn_run_prepared(prepared, copy_solution=False)

        else:

            def run():
                return run_solver(spec, ode_fn, jac_fn, y0, params)

        compile_ms, solve_ms = time_with_compile(run, _N_RUNS)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None, None, str(exc)
    print(f"compile={compile_ms:.0f} ms  solve={solve_ms:.3f} ms")
    return compile_ms, solve_ms, ""


def run_benchmarks(gpu_name: str, cache: dict) -> list[_Row]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[_Row] = []
    for spec in _SOLVERS:
        print(f"\n{spec.label}:")
        solver_cache = gpu_cache.setdefault(spec.key, {})
        for dim in _DIMS:
            dim_key = str(dim)
            if dim_key in solver_cache:
                entry = solver_cache[dim_key]
                compile_ms = entry.get("compile_ms")
                solve_ms = entry.get("solve_ms")
                error = entry.get("error", "")
                c_text = f"{compile_ms:.0f} ms" if compile_ms is not None else "FAILED"
                s_text = f"{solve_ms:.3f} ms" if solve_ms is not None else "FAILED"
                print(
                    f"  {spec.label:<14} dim={dim:>3} ... (cached)"
                    f"  compile={c_text}  solve={s_text}"
                )
            else:
                compile_ms, solve_ms, error = collect_timing(spec, dim)
                solver_cache[dim_key] = {
                    "compile_ms": compile_ms,
                    "solve_ms": solve_ms,
                    "error": error,
                }
                save_cache(_CACHE_PATH, cache)
            rows.append((spec.key, spec.label, dim, compile_ms, solve_ms, error))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_csv(rows: list[_Row], gpu_name: str, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "gpu",
                "solver_key",
                "solver",
                "dim",
                "compile_time_ms",
                "solve_time_ms",
                "error",
            ),
        )
        writer.writeheader()
        for key, label, dim, compile_ms, solve_ms, error in rows:
            writer.writerow(
                {
                    "gpu": gpu_name,
                    "solver_key": key,
                    "solver": label,
                    "dim": dim,
                    "compile_time_ms": compile_ms,
                    "solve_time_ms": solve_ms,
                    "error": error,
                }
            )
    print(f"Results saved to {path}")


def _filter_valid(rows: list[_Row], solver_key: str, col: int):
    pairs = [(r[2], r[col]) for r in rows if r[0] == solver_key and r[col] is not None]
    if not pairs:
        return [], []
    dims, vals = zip(*pairs)
    return list(dims), list(vals)


def plot(rows: list[_Row], gpu_name: str, output_path: Path) -> None:
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(12, 5))

    for spec in _SOLVERS:
        color = _COLORS[spec.key]
        dims_c, vals_c = _filter_valid(rows, spec.key, col=3)
        dims_s, vals_s = _filter_valid(rows, spec.key, col=4)
        kw = dict(marker="o", color=color, label=spec.label)
        if dims_c:
            ax_c.plot(dims_c, vals_c, **kw)
        if dims_s:
            ax_s.plot(dims_s, vals_s, **kw)

    for ax, title in [(ax_c, "Compilation time"), (ax_s, "Solve time")]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("ODE dimension")
        ax.set_ylabel(f"{title} (ms)")
        ax.set_title(f"{title} — VdP lattice — {gpu_name}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_xticks(_DIMS)
        ax.set_xticklabels([str(d) for d in _DIMS])
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"Scenario: coupled VdP lattice, N={_N_TRAJ:,}, dims={_DIMS}\n")

    cache = load_cache(_CACHE_PATH)
    rows = run_benchmarks(gpu_name, cache)

    csv_path, plot_path = output_paths(_SCRIPT_DIR, gpu_name)
    save_csv(rows, gpu_name, csv_path)
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
