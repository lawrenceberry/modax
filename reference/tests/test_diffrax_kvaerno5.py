import jax.numpy as jnp

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    parametrize_system_cases,
)


@parametrize_system_cases
def test_diffrax_kvaerno5_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: diffrax_kvaerno5_solve(
            case.ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)
