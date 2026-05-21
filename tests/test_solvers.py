import jax.numpy as jnp
import pytest

from solvers.kencarp5jax import solve as kencarp5_solve
from solvers.kencarp5numba import solve as kencarp5numba_solve
from solvers.rodas5Pjax import solve as rodas5P_solve
from solvers.rodas5Pnumba import solve as rodas5Pnumba_solve
from solvers.tsit5jax import solve as tsit5_solve
from solvers.tsit5numba import solve as tsit5numba_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    parametrize_system_cases,
)


@parametrize_system_cases
def test_tsit5_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: tsit5_solve(
            case.ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
def test_tsit5numba_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: tsit5numba_solve(
            case.ode_fn,
            case.y0,
            case.t_span,
            case.params,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
def test_kencarp5_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: kencarp5_solve(
            case.explicit_ode_fn,
            case.implicit_ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            linear=case.linear_implicit,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
def test_kencarp5numba_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: kencarp5numba_solve(
            case.explicit_ode_fn,
            case.implicit_ode_fn,
            case.implicit_jac_fn,
            case.y0,
            case.t_span,
            case.params,
            linear=case.linear_implicit,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
@pytest.mark.parametrize("lu_precision", ("fp32", "fp64"))
def test_rodas5P_reference_system(benchmark, case, lu_precision):
    result = benchmark_solve(
        benchmark,
        lambda: rodas5P_solve(
            case.ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            lu_precision=lu_precision,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
def test_rodas5Pnumba_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: rodas5Pnumba_solve(
            case.ode_fn,
            case.jac_fn,
            case.y0,
            case.t_span,
            case.params,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)
