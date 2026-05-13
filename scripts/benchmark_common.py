"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import jax

T = TypeVar("T")

SOLVE_TIMEOUT_SECONDS = 30.0
TIMEOUT_ERROR = "exceeded timeout"
TIMEOUT_STATUS = "timeout"


class BenchmarkTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True, kw_only=True)
class BenchmarkCase:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str = "-"


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
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise BenchmarkTimeoutError(TIMEOUT_ERROR) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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


def time_blocked_ms(run: Callable[[], T], n_runs: int) -> float:
    ms, _ = time_blocked(run, n_runs)
    return ms


def julia_solve_time_ms(
    solve: Any,
    system_name: str,
    y0: Any,
    t_span: Any,
    params: Any,
    **kwargs: Any,
) -> float:
    result = solve._julia_solve_with_timing(
        system_name,
        y0,
        t_span,
        params,
        **kwargs,
    )
    return result.solve_time_s * 1000


def drop_none_rows(
    rows: Sequence[tuple[str, str, int, float | None]],
    key: str,
) -> tuple[list[int], list[float]]:
    pairs = [(x, ms) for row_key, _, x, ms in rows if row_key == key and ms is not None]
    if not pairs:
        return [], []
    xs, times = zip(*pairs)
    return list(xs), list(times)


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
