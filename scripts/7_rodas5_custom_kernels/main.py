"""Benchmark custom-kernel Rodas5 solvers on 10,000 divergent Robertson trajectories.

Usage:
    uv run python scripts/7_rodas5_custom_kernels/main.py
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.systems.python import robertson
from scripts.benchmark_common import get_gpu_name, output_paths
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import solve as rodas5ckn_solve

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 10_000
_N_RUNS = 10
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SolverSpec:
    key: str
    label: str
    solve_fn: Callable
    ode_fn: Callable
    jac_fn: Callable | None = field(default=None)


_SOLVERS = (
    SolverSpec("rodas5", "JAX rodas5", rodas5_solve, robertson.ode_fn),
    SolverSpec(
        "rodas5ckn",
        "numba-cuda",
        rodas5ckn_solve,
        robertson.ode_fn_numba_cuda,
        robertson.jac_fn_numba_cuda,
    ),
)


def block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, tuple):
        for item in value:
            block_until_ready(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
        return
    jax.block_until_ready(value)


def run_solver(spec: SolverSpec, y0: np.ndarray, params: np.ndarray):
    if spec.key == "rodas5":
        y0_arg = jnp.asarray(y0, dtype=jnp.float64)
        params_arg = jnp.asarray(params, dtype=jnp.float64)
        return spec.solve_fn(
            spec.ode_fn,
            y0=y0_arg,
            t_span=robertson.TIMES,
            params=params_arg,
            **_SOLVER_KWARGS,
        )
    if spec.jac_fn is not None:
        return spec.solve_fn(
            spec.ode_fn,
            spec.jac_fn,
            y0=y0,
            t_span=robertson.TIMES,
            params=params,
            **_SOLVER_KWARGS,
        )
    return spec.solve_fn(
        spec.ode_fn,
        y0=y0,
        t_span=robertson.TIMES,
        params=params,
        **_SOLVER_KWARGS,
    )


def time_solver(
    spec: SolverSpec, y0: np.ndarray, params: np.ndarray
) -> tuple[float, str]:
    result = run_solver(spec, y0, params)
    block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        result = run_solver(spec, y0, params)
        block_until_ready(result)
    ms = (time.perf_counter() - t0) / _N_RUNS * 1000.0
    return ms, type(result).__name__


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"Scenario: divergent Robertson, N={_N_TRAJ}\n")

    y0, params = robertson.make_scenario("divergent", _N_TRAJ)
    rows = []
    for spec in _SOLVERS:
        print(f"{spec.label:<14} ...", end=" ", flush=True)
        try:
            ms, result_type = time_solver(spec, y0, params)
        except Exception as exc:
            print(f"FAILED ({exc})")
            rows.append(
                {
                    "gpu": gpu_name,
                    "solver_key": spec.key,
                    "solver": spec.label,
                    "solve_time_ms": None,
                    "result_type": "",
                    "error": str(exc),
                }
            )
            continue
        print(f"{ms:.3f} ms")
        rows.append(
            {
                "gpu": gpu_name,
                "solver_key": spec.key,
                "solver": spec.label,
                "solve_time_ms": ms,
                "result_type": result_type,
                "error": "",
            }
        )

    csv_path, _ = output_paths(_SCRIPT_DIR, gpu_name)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "gpu",
                "solver_key",
                "solver",
                "solve_time_ms",
                "result_type",
                "error",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    successful = [row for row in rows if row["solve_time_ms"] is not None]
    if successful:
        fastest = min(successful, key=lambda row: row["solve_time_ms"])
        print(f"Fastest: {fastest['solver']} ({fastest['solve_time_ms']:.3f} ms)")


if __name__ == "__main__":
    main()
