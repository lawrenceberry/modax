import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvers.kencarp5jax import solve as kencarp5_solve
from solvers.rodas5jax import solve as rodas5_solve
from solvers.tsit5jax import solve as tsit5_solve

jax.config.update("jax_enable_x64", True)


def _explicit_decay(y, t, params):
    del t
    return -params[0] * y


def _zero_rhs(y, t, params):
    del t, params
    return jnp.zeros_like(y)


def _have_cuda() -> bool:
    try:
        from numba import cuda
    except ImportError:
        return False
    return bool(cuda.is_available())


def _build_numba_callbacks():
    from numba import cuda

    @cuda.jit(device=True)
    def decay_device(y, t, p, dy, i):
        for j in range(dy.shape[1]):
            dy[i, j] = -p[i, 0] * y[i, j]

    @cuda.jit(device=True)
    def decay_jac_device(y, t, p, jac, i):
        for r in range(jac.shape[1]):
            for c in range(jac.shape[2]):
                jac[i, r, c] = 0.0
            jac[i, r, r] = -p[i, 0]

    @cuda.jit(device=True)
    def zero_rhs_device(y, t, p, dy, i):
        for j in range(dy.shape[1]):
            dy[i, j] = 0.0

    return decay_device, decay_jac_device, zero_rhs_device


def _solver_cases():
    cases = [
        ("tsit5", tsit5_solve, (_explicit_decay,), {}),
        ("rodas5", rodas5_solve, (_explicit_decay,), {}),
        ("kencarp5", kencarp5_solve, (_zero_rhs, _explicit_decay), {"linear": True}),
    ]
    if _have_cuda():
        from solvers.kencarp5numba import solve as kencarp5numba_solve
        from solvers.rodas5numba import solve as rodas5numba_solve
        from solvers.tsit5numba import solve as tsit5numba_solve

        decay, decay_jac, zero_rhs = _build_numba_callbacks()
        cases.extend(
            [
                ("tsit5numba", tsit5numba_solve, (decay,), {}),
                ("rodas5numba", rodas5numba_solve, (decay, decay_jac), {}),
                (
                    "kencarp5numba",
                    kencarp5numba_solve,
                    (zero_rhs, decay, decay_jac),
                    {"linear": True},
                ),
            ]
        )
    return cases


_SOLVER_CASES = _solver_cases()


@pytest.mark.parametrize(
    ("name", "solve_fn", "args", "kwargs"),
    _SOLVER_CASES,
)
def test_solver_vmap_over_params_matches_native_ensemble(name, solve_fn, args, kwargs):
    del name
    y0 = jnp.array([1.0])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])
    solve_kwargs = {
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": 0.1,
        "max_steps": 256,
        **kwargs,
    }

    direct = solve_fn(*args, y0, t_span, params, **solve_kwargs)
    vmapped = jax.vmap(lambda p: solve_fn(*args, y0, t_span, p, **solve_kwargs))(params)

    np.testing.assert_allclose(vmapped[:, 0], direct, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("name", "solve_fn", "args", "kwargs"),
    _SOLVER_CASES,
)
def test_solver_vmap_over_y0_and_params_matches_native_ensemble(
    name, solve_fn, args, kwargs
):
    del name
    y0s = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])
    solve_kwargs = {
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": 0.1,
        "max_steps": 256,
        **kwargs,
    }

    direct = solve_fn(*args, y0s, t_span, params, **solve_kwargs)
    vmapped = jax.vmap(lambda y0, p: solve_fn(*args, y0, t_span, p, **solve_kwargs))(
        y0s, params
    )

    np.testing.assert_allclose(vmapped[:, 0], direct, rtol=1e-9, atol=1e-9)


def test_solver_vmap_return_stats_shapes():
    y0 = jnp.array([1.0])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])

    _, stats = jax.vmap(
        lambda p: rodas5_solve(
            _explicit_decay,
            y0,
            t_span,
            p,
            rtol=1e-5,
            atol=1e-7,
            first_step=0.1,
            max_steps=256,
            return_stats=True,
        )
    )(params)

    assert stats["accepted_steps"].shape == (params.shape[0], 1)
    assert stats["rejected_steps"].shape == (params.shape[0], 1)
    assert stats["batch_loop_iterations"].shape == (params.shape[0], 1)
    assert stats["valid_lanes"].shape == (params.shape[0], 1)
    assert bool(jnp.all(stats["valid_lanes"] == 1))


@pytest.mark.skipif(not _have_cuda(), reason="numba.cuda unavailable")
def test_numba_solver_vmap_return_stats_shapes():
    from solvers.rodas5numba import solve as rodas5numba_solve

    decay, decay_jac, _ = _build_numba_callbacks()

    y0 = jnp.array([1.0])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])

    _, stats = jax.vmap(
        lambda p: rodas5numba_solve(
            decay,
            decay_jac,
            y0,
            t_span,
            p,
            rtol=1e-5,
            atol=1e-7,
            first_step=0.1,
            max_steps=256,
            return_stats=True,
        )
    )(params)

    assert stats["accepted_steps"].shape == (params.shape[0], 1)
    assert stats["rejected_steps"].shape == (params.shape[0], 1)
    assert stats["loop_steps"].shape == (params.shape[0], 1)
    assert stats["batch_loop_iterations"].shape == (params.shape[0], 1)
    assert stats["valid_lanes"].shape == (params.shape[0], 1)
    assert bool(jnp.all(stats["valid_lanes"] == 1))
