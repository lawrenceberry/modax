"""Benchmark rodas5 and kencarp5 on the van der Pol system across ensemble sizes.

Runs one warmup then one timed solve for each (solver, ensemble_size) combination,
then plots solve time vs ensemble size on a log-log scale.

Timings are cached in scripts/vdp_benchmark_cache.json keyed by (n_osc, mu_max)
so that re-running to regenerate the plot skips already-collected data points.

Usage:
    uv run python scripts/benchmark_vdp.py --n-osc 15 --mu 100
"""

import argparse
import json
import subprocess
import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solvers.rodas5 import solve as rodas5_solve
from reference.solvers.python.diffrax_kvaerno5 import (
    make_solver as make_kvaerno5_solver,
)
from reference.solvers.python.julia_common import _check_julia_environment
from reference.solvers.python.julia_rodas5 import (
    make_solver as make_julia_rodas5_solver,
)

_TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
_ENSEMBLE_SIZES = [2, 10, 100, 1000, 10000]
_PRECISIONS = ["fp32", "fp64"]
_JULIA_BACKENDS = ["EnsembleGPUArray"]
_JULIA_LABELS = {
    "EnsembleGPUArray": "julia rodas5 array",
}
_CACHE_PATH = Path("scripts/vdp_benchmark_cache.json")


# ---------------------------------------------------------------------------
# GPU name
# ---------------------------------------------------------------------------


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
    return "unknown GPU"


# ---------------------------------------------------------------------------
# Cache helpers  (keyed by config_key → solver_key → ensemble_size)
# ---------------------------------------------------------------------------


def load_cache() -> dict:
    if _CACHE_PATH.exists():
        cache = json.loads(_CACHE_PATH.read_text())
        if "by_gpu" not in cache:
            gpu_name = cache.get("gpu_name") or "unknown GPU"
            legacy_sections = {
                key: value for key, value in cache.items() if key != "gpu_name"
            }
            cache = {
                "gpu_name": gpu_name,
                "by_gpu": {
                    gpu_name: legacy_sections,
                },
            }
        return cache
    return {"gpu_name": None, "by_gpu": {}}


def save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def initialize_cache() -> dict:
    cache = load_cache()
    current_gpu_name = get_gpu_name()
    cache.setdefault("by_gpu", {})
    cache["by_gpu"].setdefault(current_gpu_name, {})
    cache["gpu_name"] = current_gpu_name
    save_cache(cache)
    return cache


def cache_get(cache: dict, config_key: str, solver_key: str, size: int) -> float | None:
    gpu_cache = cache["by_gpu"][cache["gpu_name"]]
    return gpu_cache.get(config_key, {}).get(solver_key, {}).get(str(size))


def cache_set(
    cache: dict, config_key: str, solver_key: str, size: int, ms: float
) -> None:
    gpu_cache = cache["by_gpu"][cache["gpu_name"]]
    gpu_cache.setdefault(config_key, {}).setdefault(solver_key, {})[str(size)] = ms
    save_cache(cache)


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------


def make_vdp_system(n_osc: int, mu_max: float):
    mu = jnp.array(
        [10.0 ** (np.log10(max(mu_max, 1)) * i / (n_osc - 1)) for i in range(n_osc)],
        dtype=jnp.float64,
    )
    omega = jnp.array(
        [10.0 ** (-0.5 + i / (n_osc - 1)) for i in range(n_osc)],
        dtype=jnp.float64,
    )
    y0 = jnp.array([2.0, 0.0] * n_osc, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        s = p[0]
        x = y[0::2]
        v = y[1::2]
        return jnp.stack([v, s * mu * (1.0 - x * x) * v - omega**2 * x], axis=1).ravel()

    def explicit_ode_fn(y, t, p):
        del t
        x = y[0::2]
        v = y[1::2]
        return jnp.stack([v, -(omega**2) * x], axis=1).ravel()

    def implicit_ode_fn(y, t, p):
        del t
        s = p[0]
        x = y[0::2]
        v = y[1::2]
        return jnp.stack(
            [jnp.zeros_like(x), s * mu * (1.0 - x * x) * v], axis=1
        ).ravel()

    return y0, ode_fn, explicit_ode_fn, implicit_ode_fn


def make_params_batch(size: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


# ---------------------------------------------------------------------------
# Timing functions
# ---------------------------------------------------------------------------


def time_rodas5(ode_fn, y0, params, lu_precision) -> float:
    def run():
        return rodas5_solve(
            ode_fn,
            y0,
            _TIMES,
            params,
            lu_precision=lu_precision,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready()

    run()  # warmup / JIT compile
    t0 = time.perf_counter()
    run()
    return (time.perf_counter() - t0) * 1000


def time_kvaerno5(solve, y0, params) -> float:
    def run():
        return solve(
            y0=y0,
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready()

    run()  # warmup / JIT compile
    t0 = time.perf_counter()
    run()
    return (time.perf_counter() - t0) * 1000


def time_julia_rodas5(solve, y0, params) -> float:
    """Return Julia's internal solve time in ms (excludes subprocess/startup overhead)."""
    result = solve._julia_solve_with_timing(
        y0=y0,
        t_span=_TIMES,
        params=np.asarray(params),
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )
    return result.solve_time_s * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collect_or_load(
    cache, config_key, solver_key, size, label, timing_fn
) -> float | None:
    cached = cache_get(cache, config_key, solver_key, size)
    if cached is not None:
        print(f"  {label}  n={size:>6}  (cached) {cached:.1f}ms")
        return cached
    print(f"  {label}  n={size:>6} ...", end=" ", flush=True)
    try:
        ms = timing_fn()
    except Exception as e:
        print(f"FAILED ({e})")
        return None
    cache_set(cache, config_key, solver_key, size, ms)
    print(f"{ms:.1f}ms")
    return ms


def drop_none(
    sizes: list[int], vals: list[float | None]
) -> tuple[list[int], list[float]]:
    pairs = [(s, v) for s, v in zip(sizes, vals) if v is not None]
    return ([s for s, _ in pairs], [v for _, v in pairs])


def main():
    parser = argparse.ArgumentParser(description="VDP ensemble benchmark")
    parser.add_argument(
        "--n-osc",
        type=int,
        default=15,
        help="Number of oscillator pairs (default: 15 → 30D)",
    )
    parser.add_argument(
        "--mu", type=float, default=100.0, help="Max stiffness mu (default: 100)"
    )
    args = parser.parse_args()

    n_osc: int = args.n_osc
    mu_max: float = args.mu
    config_key = f"{n_osc}osc_mu{mu_max:g}"

    cache = initialize_cache()
    gpu_name = cache["gpu_name"]

    print(
        f"\nVDP benchmark: {n_osc} oscillators ({2 * n_osc}D), mu_max={mu_max:g}, GPU={gpu_name}\n"
    )

    y0, ode_fn, _explicit_ode_fn, _implicit_ode_fn = make_vdp_system(n_osc, mu_max)

    # meowax rodas5
    rodas5_timings: dict[str, list[float | None]] = {}
    for precision in _PRECISIONS:
        solver_key = f"meowax_rodas5_{precision}"
        rodas5_timings[precision] = []
        for size in _ENSEMBLE_SIZES:
            params = make_params_batch(size)
            ms = collect_or_load(
                cache,
                config_key,
                solver_key,
                size,
                f"meowax rodas5 {precision}",
                lambda params=params: time_rodas5(ode_fn, y0, params, precision),
            )
            rodas5_timings[precision].append(ms)

    # diffrax kvaerno5
    kvaerno5_timings: list[float | None] = []
    kvaerno5_solve = make_kvaerno5_solver(ode_fn)
    for size in _ENSEMBLE_SIZES:
        params = make_params_batch(size)
        ms = collect_or_load(
            cache,
            config_key,
            "diffrax_kvaerno5",
            size,
            "diffrax kvaerno5",
            lambda params=params: time_kvaerno5(kvaerno5_solve, y0, params),
        )
        kvaerno5_timings.append(ms)

    # julia rodas5
    julia_timings: dict[str, list[float | None]] = {}
    julia_check = _check_julia_environment()
    if julia_check["ok"]:
        for backend in _JULIA_BACKENDS:
            julia_timings[backend] = []
            solve = make_julia_rodas5_solver(
                "vdp",
                system_config={"n_osc": n_osc, "mu_max": mu_max},
                ensemble_backend=backend,
            )
            solver_key = f"julia_rodas5_{backend}"
            label = _JULIA_LABELS[backend]
            for size in _ENSEMBLE_SIZES:
                params = make_params_batch(size)
                ms = collect_or_load(
                    cache,
                    config_key,
                    solver_key,
                    size,
                    label,
                    lambda params=params: time_julia_rodas5(solve, y0, params),
                )
                julia_timings[backend].append(ms)
    else:
        print(f"\nSkipping Julia solvers: {julia_check['reason']}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"fp32": "#e05c2b", "fp64": "#2b7be0"}
    for precision in _PRECISIONS:
        xs, ys = drop_none(_ENSEMBLE_SIZES, rodas5_timings[precision])
        ax.plot(
            xs,
            ys,
            marker="o",
            label=f"meowax rodas5 {precision}",
            color=colors[precision],
        )

    xs, ys = drop_none(_ENSEMBLE_SIZES, kvaerno5_timings)
    ax.plot(xs, ys, marker="D", label="diffrax kvaerno5", color="#2ba84a")

    julia_colors = {"EnsembleGPUArray": "#9b59b6"}
    for backend in _JULIA_BACKENDS:
        if backend in julia_timings:
            xs, ys = drop_none(_ENSEMBLE_SIZES, julia_timings[backend])
            ax.plot(
                xs,
                ys,
                marker="^",
                label=_JULIA_LABELS[backend],
                color=julia_colors[backend],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Solve time (ms)")
    ax.set_title(
        f"VDP benchmark — {n_osc} osc ({2 * n_osc}D), μ_max={mu_max:g} — {gpu_name}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_ENSEMBLE_SIZES)
    ax.set_xticklabels([str(n) for n in _ENSEMBLE_SIZES])

    out_path = f"scripts/vdp_{n_osc}osc_mu{mu_max:g}_benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
