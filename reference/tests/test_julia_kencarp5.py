import numpy as np

from reference.solvers.python.julia_common import benchmark_julia_solver
from reference.solvers.python.julia_kencarp5 import solve as julia_kencarp5_solve
from tests.benchmark_helpers import assert_case_output, parametrize_system_cases


@parametrize_system_cases
def test_julia_kencarp5_reference_system(benchmark, case):
    result = benchmark_julia_solver(
        benchmark,
        julia_kencarp5_solve,
        case.name,
        y0=case.y0,
        t_span=case.t_span,
        params=case.params,
        system_config=case.system_config,
        ensemble_backend="EnsembleGPUArray",
        **case.kwargs,
    )
    assert_case_output(np.asarray(result), case)
