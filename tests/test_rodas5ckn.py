from solvers.rodas5ckn import solve as rodas5ckn_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    ckn_callbacks,
    parametrize_system_cases,
)


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
