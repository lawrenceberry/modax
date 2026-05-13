"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, TypeVar

import jax

T = TypeVar("T")

SOLVE_TIMEOUT_SECONDS = 30.0
TIMEOUT_ERROR = "exceeded timeout"
TIMEOUT_STATUS = "timeout"


class BenchmarkTimeoutError(TimeoutError):
    pass


def timeout_cache_entry() -> dict[str, str]:
    return {"status": TIMEOUT_STATUS, "error": TIMEOUT_ERROR}


def is_timeout(value) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status") == TIMEOUT_STATUS
        and value.get("error") == TIMEOUT_ERROR
    )


def format_cached_timing(value) -> str:
    if is_timeout(value):
        return TIMEOUT_ERROR
    if value is None:
        return "FAILED"
    return f"{value:.1f} ms"


def timing_value_or_none(value) -> float | None:
    if is_timeout(value) or value is None:
        return None
    return float(value)


def run_with_timeout(
    run: Callable[[], T], timeout_s: float = SOLVE_TIMEOUT_SECONDS
) -> T:
    del timeout_s
    return run()


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


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2))


def output_paths(script_dir: Path, gpu_name: str) -> tuple[Path, Path]:
    slug = gpu_slug(gpu_name)
    return script_dir / f"results-{slug}.csv", script_dir / f"plot-{slug}.png"


def time_blocked(
    run: Callable[[], T], n_runs: int, *, timeout_s: float | None = None
) -> tuple[float, T]:
    def time_once() -> tuple[float, T]:
        result = run()
        jax.block_until_ready(result)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            result = run()
            jax.block_until_ready(result)
        return (time.perf_counter() - t0) / n_runs * 1000, result

    if timeout_s is not None:
        return run_with_timeout(time_once, timeout_s)
    return time_once()


def timed_solve(run: Callable[[], T], timeout_s: float = SOLVE_TIMEOUT_SECONDS) -> T:
    return run_with_timeout(run, timeout_s)


def collect_timed_timing(
    label: str,
    descriptor: str,
    run: Callable[[], float],
    *,
    label_width: int,
) -> float | dict[str, str] | None:
    print(f"  {label:<{label_width}} {descriptor} ...", end=" ", flush=True)
    try:
        ms = timed_solve(run)
    except BenchmarkTimeoutError:
        print(TIMEOUT_ERROR)
        return timeout_cache_entry()
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print(f"{ms:.1f} ms")
    return ms
