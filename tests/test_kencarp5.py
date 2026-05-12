import jax.numpy as jnp

from solvers.kencarp5 import solve as kencarp5_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    parametrize_system_cases,
)


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
