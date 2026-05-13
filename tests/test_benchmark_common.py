import time

import pytest

from scripts.benchmark_common import (
    TIMEOUT_ERROR,
    BenchmarkTimeoutError,
    format_cached_timing,
    is_timeout,
    run_with_timeout,
    timeout_cache_entry,
    timing_value_or_none,
)


def _fast_value():
    return 12.5


def _too_slow():
    time.sleep(1.0)
    return 1.0


def test_run_with_timeout_returns_fast_value():
    assert run_with_timeout(_fast_value, timeout_s=1.0) == 12.5


def test_run_with_timeout_raises_for_slow_value():
    with pytest.raises(BenchmarkTimeoutError, match=TIMEOUT_ERROR):
        run_with_timeout(_too_slow, timeout_s=0.05)


def test_timeout_cache_helpers():
    entry = timeout_cache_entry()
    assert entry == {"status": "timeout", "error": TIMEOUT_ERROR}
    assert is_timeout(entry)
    assert format_cached_timing(entry) == TIMEOUT_ERROR
    assert format_cached_timing(None) == "FAILED"
    assert format_cached_timing(3.25) == "3.2 ms"
    assert timing_value_or_none(entry) is None
    assert timing_value_or_none(None) is None
    assert timing_value_or_none(3.25) == 3.25
