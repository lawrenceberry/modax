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


def test_tsit5numba_shared_matches_global():
    """The shared-memory and transposed-global backends must be bit-identical."""
    import numpy as np

    from reference.systems.python import vdp

    n_osc = 4  # dim = 8: fits the shared-memory backend
    ode_fn, _, _ = vdp.make_system(n_osc, mu=1.0)
    y0, params = vdp.make_scenario(n_osc, 256, divergence=1.0)
    y0 = jnp.asarray(np.ascontiguousarray(y0))
    params = jnp.asarray(np.ascontiguousarray(params))
    t_span = jnp.asarray(vdp.TIMES)
    kw = dict(first_step=1e-4, rtol=1e-6, atol=1e-8)

    shared = tsit5numba_solve(ode_fn, y0, t_span, params, backend="shared", **kw)
    glob = tsit5numba_solve(ode_fn, y0, t_span, params, backend="global", **kw)
    assert float(jnp.max(jnp.abs(shared - glob))) == 0.0


@parametrize_system_cases
@pytest.mark.parametrize("lu_precision", ("fp32", "fp64"))
def test_kencarp5_reference_system(benchmark, case, lu_precision):
    result = benchmark_solve(
        benchmark,
        lambda: kencarp5_solve(
            case.explicit_ode_fn,
            case.implicit_ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            lu_precision=lu_precision,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
@pytest.mark.parametrize("lu_precision", ("fp32", "fp64"))
def test_kencarp5numba_reference_system(benchmark, case, lu_precision):
    result = benchmark_solve(
        benchmark,
        lambda: kencarp5numba_solve(
            case.explicit_ode_fn,
            case.implicit_ode_fn,
            case.implicit_jac_fn,
            case.y0,
            case.t_span,
            case.params,
            lu_precision=lu_precision,
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
@pytest.mark.parametrize("lu_precision", ("fp32", "fp64"))
def test_rodas5Pnumba_reference_system(benchmark, case, lu_precision):
    result = benchmark_solve(
        benchmark,
        lambda: rodas5Pnumba_solve(
            case.ode_fn,
            case.jac_fn,
            case.y0,
            case.t_span,
            case.params,
            lu_precision=lu_precision,
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)
