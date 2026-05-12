import jax.numpy as jnp
import pytest

from solvers.rodas5 import solve as rodas5_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    parametrize_system_cases,
)


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
