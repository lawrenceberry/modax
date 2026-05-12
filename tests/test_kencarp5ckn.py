from solvers.kencarp5ckn import solve as kencarp5ckn_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    ckn_callbacks,
    parametrize_system_cases,
)


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
