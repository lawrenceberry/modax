"""Benchmark custom-kernel Tsit5 solvers on 1000 divergent Lorenz trajectories.

Usage:
    uv run python scripts/6_tsit5_custom_kernels/main.py
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.systems.python import lorenz
from scripts.benchmark_common import get_gpu_name, output_paths
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import solve as tsit5ckn_solve
from solvers.tsit5ckp import solve as tsit5ckp_solve
from solvers.tsit5ckw import solve as tsit5ckw_solve

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


_SOLVERS = (
    SolverSpec("tsit5", "JAX tsit5", tsit5_solve, lorenz.ode_fn),
    SolverSpec("tsit5ckp", "Pallas/Triton", tsit5ckp_solve, lorenz.ode_fn_pallas),
    SolverSpec("tsit5ckw", "NVIDIA Warp", tsit5ckw_solve, lorenz.ode_fn_warp),
    SolverSpec("tsit5ckn", "numba-cuda", tsit5ckn_solve, lorenz.ode_fn_numba_cuda),
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
    y0_arg = jnp.asarray(y0, dtype=jnp.float64) if spec.key == "tsit5" else y0
    params_arg = (
        jnp.asarray(params, dtype=jnp.float64) if spec.key == "tsit5" else params
    )
    return spec.solve_fn(
        spec.ode_fn,
        y0=y0_arg,
        t_span=lorenz.TIMES,
        params=params_arg,
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
    print(f"Scenario: divergent Lorenz, N={_N_TRAJ}\n")

    y0, params = lorenz.make_scenario("divergent", _N_TRAJ)
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
