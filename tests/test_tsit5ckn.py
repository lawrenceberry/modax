from solvers.tsit5ckn import solve as tsit5ckn_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    ckn_callbacks,
    parametrize_system_cases,
)


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
