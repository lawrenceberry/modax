"""Julia Rodas5P reference solver via DiffEqGPU."""

from reference.solvers.python.julia_common import solve as _solve
from reference.solvers.python.julia_common import (
    solve_with_timing as _solve_with_timing,
)


def solve(
    system_name,
    y0,
    t_span,
    params,
    *,
    system_config=None,
    ensemble_backend="EnsembleGPUArray",
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
):
    """Solve an ensemble with Julia Rodas5P."""
    return _solve(
        "rodas5P",
        system_name,
        y0,
        t_span,
        params,
        system_config=system_config,
        ensemble_backend=ensemble_backend,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )


def solve_with_timing(
    system_name,
    y0,
    t_span,
    params,
    *,
    system_config=None,
    ensemble_backend="EnsembleGPUArray",
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
):
    """Solve an ensemble with Julia Rodas5P and return timing metadata."""
    return _solve_with_timing(
        "rodas5P",
        system_name,
        y0,
        t_span,
        params,
        system_config=system_config,
        ensemble_backend=ensemble_backend,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )


solve._julia_solve_with_timing = solve_with_timing
