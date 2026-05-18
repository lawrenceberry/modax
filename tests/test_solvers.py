import jax.numpy as jnp
import pytest

from solvers.kencarp5 import solve as kencarp5_solve
from solvers.kencarp5ckn import solve as kencarp5ckn_solve
from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5ckn import solve as rodas5ckn_solve
from solvers.tsit5 import solve as tsit5_solve
from solvers.tsit5ckn import solve as tsit5ckn_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    ckn_callbacks,
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
def test_tsit5ckn_reference_system(benchmark, case):
    ode_fn, _, _, _, _ = ckn_callbacks(case.name)
    result = benchmark_solve(
        benchmark,
        lambda: tsit5ckn_solve(
            ode_fn,
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
def test_kencarp5ckn_reference_system(benchmark, case):
    _, explicit_fn, implicit_fn, implicit_jac_fn, _ = ckn_callbacks(case.name)
    result = benchmark_solve(
        benchmark,
        lambda: kencarp5ckn_solve(
            explicit_fn,
            implicit_fn,
            implicit_jac_fn,
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
def test_rodas5_reference_system(benchmark, case, lu_precision):
    result = benchmark_solve(
        benchmark,
        lambda: rodas5_solve(
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
def test_rodas5ckn_reference_system(benchmark, case):
    ode_fn, _, _, _, jac_fn = ckn_callbacks(case.name)
    result = benchmark_solve(
        benchmark,
        lambda: rodas5ckn_solve(
            ode_fn,
            jac_fn,
            case.y0,
            case.t_span,
            case.params,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)
