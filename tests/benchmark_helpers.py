from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba import cuda

from reference.systems.python import (
    bateman,
    brusselator,
    heat,
    kaps,
    lorenz,
    robertson,
    vdp,
)

jax.config.update("jax_enable_x64", True)

ENSEMBLE_SIZE = 2


@dataclass(frozen=True)
class SystemCase:
    name: str
    y0: np.ndarray
    t_span: np.ndarray
    params: np.ndarray
    ode_fn: Callable
    explicit_ode_fn: Callable
    implicit_ode_fn: Callable
    system_config: dict
    kwargs: dict
    linear_implicit: bool
    expected_shape: tuple[int, int, int]
    assert_solution: Callable[[np.ndarray, "SystemCase"], None]


def zero_ode_fn(y, t, p):
    del t, p
    return jnp.zeros_like(y)


def _bateman_case():
    system = bateman.make_system(4, 1e2)
    t_span = np.asarray(bateman.TIMES[:3], dtype=np.float64)
    params = np.asarray(bateman.make_params(ENSEMBLE_SIZE), dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(system["y0"]), (ENSEMBLE_SIZE, 4)).copy()

    def assert_solution(out, case):
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=3e-5)
        exact = bateman.exact_solution(
            system["M_np"], np.asarray(system["y0"]), case.t_span, case.params
        )
        np.testing.assert_allclose(out, exact, rtol=2e-2, atol=3e-5)

    return SystemCase(
        name="bateman",
        y0=y0,
        t_span=t_span,
        params=params,
        ode_fn=system["ode_fn"],
        explicit_ode_fn=system["explicit_ode_fn"],
        implicit_ode_fn=system["implicit_ode_fn"],
        system_config={"n_vars": 4, "stiffness": 1e2},
        kwargs={"first_step": 1e-4, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=assert_solution,
    )


def _brusselator_case():
    explicit, implicit, ode, _ = brusselator.make_system(4)
    y0, params = brusselator.make_scenario("identical", 4, ENSEMBLE_SIZE)
    t_span = np.asarray(brusselator.TIMES[:3], dtype=np.float64)
    return SystemCase(
        name="brusselator",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=ode,
        explicit_ode_fn=explicit,
        implicit_ode_fn=implicit,
        system_config={"n_grid": 4},
        kwargs={"first_step": 1e-3, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 8),
        assert_solution=lambda out, case: None,
    )


def _heat_case():
    system = heat.make_system(4)
    t_span = np.asarray(heat.TIMES[:3], dtype=np.float64)
    params = np.asarray(heat.make_params(ENSEMBLE_SIZE), dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(system["y0"]), (ENSEMBLE_SIZE, 4)).copy()

    def assert_solution(out, case):
        exact = heat.exact_solution(4, case.t_span, case.params)
        np.testing.assert_allclose(out, exact, rtol=3e-2, atol=3e-5)

    return SystemCase(
        name="heat",
        y0=y0,
        t_span=t_span,
        params=params,
        ode_fn=system["ode_fn"],
        explicit_ode_fn=system["explicit_ode_fn"],
        implicit_ode_fn=system["implicit_ode_fn"],
        system_config={"n_vars": 4},
        kwargs={"first_step": 1e-4, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=assert_solution,
    )


def _kaps_case():
    system = kaps.make_system(2, 1e-2)
    t_span = np.asarray(kaps.TIMES[:3], dtype=np.float64)
    params = np.asarray(kaps.make_params(ENSEMBLE_SIZE), dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(system["y0"]), (ENSEMBLE_SIZE, 4)).copy()

    def assert_solution(out, case):
        exact = kaps.exact_solution(case.t_span, case.params, 2)
        np.testing.assert_allclose(out, exact, rtol=3e-2, atol=3e-5)

    return SystemCase(
        name="kaps",
        y0=y0,
        t_span=t_span,
        params=params,
        ode_fn=system["ode_fn"],
        explicit_ode_fn=system["explicit_ode_fn"],
        implicit_ode_fn=system["implicit_ode_fn"],
        system_config={"n_pairs": 2, "epsilon_min": 1e-2},
        kwargs={"first_step": 1e-3, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=False,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=assert_solution,
    )


def _lorenz_case():
    y0, params = lorenz.make_scenario("identical", ENSEMBLE_SIZE)
    t_span = np.asarray((0.0, 0.01, 0.02), dtype=np.float64)
    return SystemCase(
        name="lorenz",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=lorenz.ode_fn,
        explicit_ode_fn=lorenz.ode_fn,
        implicit_ode_fn=zero_ode_fn,
        system_config={},
        kwargs={"first_step": 1e-4, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 3),
        assert_solution=lambda out, case: None,
    )


def _robertson_case():
    y0, params = robertson.make_scenario("identical", ENSEMBLE_SIZE)
    t_span = np.asarray((0.0, 1e-6, 1e-4), dtype=np.float64)
    return SystemCase(
        name="robertson",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=robertson.ode_fn,
        explicit_ode_fn=robertson.ode_fn,
        implicit_ode_fn=zero_ode_fn,
        system_config={},
        kwargs={"first_step": 1e-8, "rtol": 1e-5, "atol": 1e-7, "max_steps": 10000},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 3),
        assert_solution=lambda out, case: np.testing.assert_allclose(
            out.sum(axis=-1), 1.0, atol=3e-5
        ),
    )


def _vdp_case():
    ode, _ = vdp.make_system(2, mu=1.0)
    y0, params = vdp.make_scenario("identical", 2, ENSEMBLE_SIZE)
    t_span = np.asarray(vdp.TIMES[:3], dtype=np.float64)
    return SystemCase(
        name="vdp",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=ode,
        explicit_ode_fn=ode,
        implicit_ode_fn=zero_ode_fn,
        system_config={"n_osc": 2},
        kwargs={"first_step": 1e-3, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=lambda out, case: None,
    )


SYSTEM_CASES = (
    _bateman_case(),
    _brusselator_case(),
    _heat_case(),
    _kaps_case(),
    _lorenz_case(),
    _robertson_case(),
    _vdp_case(),
)
SYSTEM_IDS = tuple(case.name for case in SYSTEM_CASES)


def parametrize_system_cases(fn):
    return pytest.mark.parametrize("case", SYSTEM_CASES, ids=SYSTEM_IDS)(fn)


def block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def benchmark_solve(benchmark, solve_once):
    return benchmark.pedantic(
        lambda: block_until_ready(solve_once()),
        warmup_rounds=1,
        rounds=1,
    )


def assert_case_output(output, case: SystemCase):
    out = np.asarray(output)
    assert out.shape == case.expected_shape
    assert np.all(np.isfinite(out))
    case.assert_solution(out, case)


def ckn_callbacks(case_name: str):
    return _CKN_CALLBACKS[case_name]


@cuda.jit(device=True)
def _zero_cuda(y, t, p, dy, i):
    for j in range(dy.shape[1]):
        dy[i, j] = 0.0


@cuda.jit(device=True)
def _zero_jac_cuda(y, t, p, jac, i):
    for row in range(jac.shape[1]):
        for col in range(jac.shape[2]):
            jac[i, row, col] = 0.0


@cuda.jit(device=True)
def _bateman_ode_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    dy[i, 0] = scale * (-y[i, 0])
    dy[i, 1] = scale * (y[i, 0] - 10.0 * y[i, 1])
    dy[i, 2] = scale * (10.0 * y[i, 1] - 100.0 * y[i, 2])
    dy[i, 3] = scale * (100.0 * y[i, 2])


@cuda.jit(device=True)
def _bateman_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    scale = p[i, 0]
    jac[i, 0, 0] = -scale
    jac[i, 1, 0] = scale
    jac[i, 1, 1] = -10.0 * scale
    jac[i, 2, 1] = 10.0 * scale
    jac[i, 2, 2] = -100.0 * scale
    jac[i, 3, 2] = 100.0 * scale


@cuda.jit(device=True)
def _heat_ode_cuda(y, t, p, dy, i):
    scale = p[i, 0] * 25.0
    dy[i, 0] = scale * (-2.0 * y[i, 0] + y[i, 1])
    dy[i, 1] = scale * (y[i, 0] - 2.0 * y[i, 1] + y[i, 2])
    dy[i, 2] = scale * (y[i, 1] - 2.0 * y[i, 2] + y[i, 3])
    dy[i, 3] = scale * (y[i, 2] - 2.0 * y[i, 3])


@cuda.jit(device=True)
def _heat_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    scale = p[i, 0] * 25.0
    jac[i, 0, 0] = -2.0 * scale
    jac[i, 0, 1] = scale
    jac[i, 1, 0] = scale
    jac[i, 1, 1] = -2.0 * scale
    jac[i, 1, 2] = scale
    jac[i, 2, 1] = scale
    jac[i, 2, 2] = -2.0 * scale
    jac[i, 2, 3] = scale
    jac[i, 3, 2] = scale
    jac[i, 3, 3] = -2.0 * scale


@cuda.jit(device=True)
def _kaps_explicit_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    for pair in range(2):
        base = 2 * pair
        y1 = y[i, base]
        y2 = y[i, base + 1]
        dy[i, base] = scale * (-2.0 * y1)
        dy[i, base + 1] = scale * (y1 - y2 - y2 * y2)


@cuda.jit(device=True)
def _kaps_implicit_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    eps0 = 1.0
    eps1 = 0.01
    y1 = y[i, 0]
    y2 = y[i, 1]
    dy[i, 0] = scale * (-(1.0 / eps0) * (y1 - y2 * y2))
    dy[i, 1] = 0.0
    y1 = y[i, 2]
    y2 = y[i, 3]
    dy[i, 2] = scale * (-(1.0 / eps1) * (y1 - y2 * y2))
    dy[i, 3] = 0.0


@cuda.jit(device=True)
def _kaps_ode_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    eps0 = 1.0
    eps1 = 0.01
    y1 = y[i, 0]
    y2 = y[i, 1]
    dy[i, 0] = scale * (-(1.0 / eps0 + 2.0) * y1 + (1.0 / eps0) * y2 * y2)
    dy[i, 1] = scale * (y1 - y2 - y2 * y2)
    y1 = y[i, 2]
    y2 = y[i, 3]
    dy[i, 2] = scale * (-(1.0 / eps1 + 2.0) * y1 + (1.0 / eps1) * y2 * y2)
    dy[i, 3] = scale * (y1 - y2 - y2 * y2)


@cuda.jit(device=True)
def _kaps_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    scale = p[i, 0]
    eps0 = 1.0
    eps1 = 0.01
    jac[i, 0, 0] = scale * (-(1.0 / eps0 + 2.0))
    jac[i, 0, 1] = scale * (2.0 / eps0 * y[i, 1])
    jac[i, 1, 0] = scale
    jac[i, 1, 1] = scale * (-1.0 - 2.0 * y[i, 1])
    jac[i, 2, 2] = scale * (-(1.0 / eps1 + 2.0))
    jac[i, 2, 3] = scale * (2.0 / eps1 * y[i, 3])
    jac[i, 3, 2] = scale
    jac[i, 3, 3] = scale * (-1.0 - 2.0 * y[i, 3])


@cuda.jit(device=True)
def _kaps_implicit_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    scale = p[i, 0]
    jac[i, 0, 0] = -scale
    jac[i, 0, 1] = scale * 2.0 * y[i, 1]
    jac[i, 2, 2] = -100.0 * scale
    jac[i, 2, 3] = 200.0 * scale * y[i, 3]


@cuda.jit(device=True)
def _lorenz_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = p[i, 0]
    jac[i, 0, 0] = -sigma
    jac[i, 0, 1] = sigma
    jac[i, 1, 0] = rho - y[i, 2]
    jac[i, 1, 1] = -1.0
    jac[i, 1, 2] = -y[i, 0]
    jac[i, 2, 0] = y[i, 1]
    jac[i, 2, 1] = y[i, 0]
    jac[i, 2, 2] = -beta


@cuda.jit(device=True)
def _vdp_ode_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    x0 = y[i, 0]
    v0 = y[i, 1]
    x1 = y[i, 2]
    v1 = y[i, 3]
    dy[i, 0] = v0
    dy[i, 1] = scale * (1.0 - x0 * x0) * v0 - x0 + 10.0 * (x1 - 2.0 * x0 + x1)
    dy[i, 2] = v1
    dy[i, 3] = scale * (1.0 - x1 * x1) * v1 - x1 + 10.0 * (x0 - 2.0 * x1 + x0)


@cuda.jit(device=True)
def _vdp_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    scale = p[i, 0]
    x0 = y[i, 0]
    v0 = y[i, 1]
    x1 = y[i, 2]
    v1 = y[i, 3]
    jac[i, 0, 1] = 1.0
    jac[i, 1, 0] = scale * (-2.0 * x0 * v0) - 21.0
    jac[i, 1, 1] = scale * (1.0 - x0 * x0)
    jac[i, 1, 2] = 20.0
    jac[i, 2, 3] = 1.0
    jac[i, 3, 0] = 20.0
    jac[i, 3, 2] = scale * (-2.0 * x1 * v1) - 21.0
    jac[i, 3, 3] = scale * (1.0 - x1 * x1)


@cuda.jit(device=True)
def _brusselator_explicit_cuda(y, t, p, dy, i):
    scale = p[i, 0]
    a_eff = scale
    b_eff = 3.0 * scale
    for grid in range(4):
        u_idx = 2 * grid
        v_idx = u_idx + 1
        u = y[i, u_idx]
        v = y[i, v_idx]
        u2v = u * u * v
        dy[i, u_idx] = a_eff + u2v - (b_eff + 1.0) * u
        dy[i, v_idx] = b_eff * u - u2v


@cuda.jit(device=True)
def _brusselator_implicit_cuda(y, t, p, dy, i):
    diff = 0.02 / (0.25 * 0.25)
    for grid in range(4):
        left = 3 if grid == 0 else grid - 1
        right = 0 if grid == 3 else grid + 1
        u_idx = 2 * grid
        v_idx = u_idx + 1
        dy[i, u_idx] = diff * (y[i, 2 * left] - 2.0 * y[i, u_idx] + y[i, 2 * right])
        dy[i, v_idx] = diff * (
            y[i, 2 * left + 1] - 2.0 * y[i, v_idx] + y[i, 2 * right + 1]
        )


@cuda.jit(device=True)
def _brusselator_ode_cuda(y, t, p, dy, i):
    _brusselator_explicit_cuda(y, t, p, dy, i)
    diff = 0.02 / (0.25 * 0.25)
    for grid in range(4):
        left = 3 if grid == 0 else grid - 1
        right = 0 if grid == 3 else grid + 1
        u_idx = 2 * grid
        v_idx = u_idx + 1
        dy[i, u_idx] += diff * (y[i, 2 * left] - 2.0 * y[i, u_idx] + y[i, 2 * right])
        dy[i, v_idx] += diff * (
            y[i, 2 * left + 1] - 2.0 * y[i, v_idx] + y[i, 2 * right + 1]
        )


@cuda.jit(device=True)
def _brusselator_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    scale = p[i, 0]
    b_eff = 3.0 * scale
    diff = 0.02 / (0.25 * 0.25)
    for grid in range(4):
        left = 3 if grid == 0 else grid - 1
        right = 0 if grid == 3 else grid + 1
        u_idx = 2 * grid
        v_idx = u_idx + 1
        u = y[i, u_idx]
        v = y[i, v_idx]
        jac[i, u_idx, u_idx] = 2.0 * u * v - (b_eff + 1.0) - 2.0 * diff
        jac[i, u_idx, v_idx] = u * u
        jac[i, v_idx, u_idx] = b_eff - 2.0 * u * v
        jac[i, v_idx, v_idx] = -u * u - 2.0 * diff
        jac[i, u_idx, 2 * left] = diff
        jac[i, u_idx, 2 * right] = diff
        jac[i, v_idx, 2 * left + 1] = diff
        jac[i, v_idx, 2 * right + 1] = diff


@cuda.jit(device=True)
def _brusselator_implicit_jac_cuda(y, t, p, jac, i):
    _zero_jac_cuda(y, t, p, jac, i)
    diff = 0.02 / (0.25 * 0.25)
    for grid in range(4):
        left = 3 if grid == 0 else grid - 1
        right = 0 if grid == 3 else grid + 1
        u_idx = 2 * grid
        v_idx = u_idx + 1
        jac[i, u_idx, u_idx] = -2.0 * diff
        jac[i, u_idx, 2 * left] = diff
        jac[i, u_idx, 2 * right] = diff
        jac[i, v_idx, v_idx] = -2.0 * diff
        jac[i, v_idx, 2 * left + 1] = diff
        jac[i, v_idx, 2 * right + 1] = diff


_CKN_CALLBACKS = {
    "bateman": (
        _bateman_ode_cuda,
        _zero_cuda,
        _bateman_ode_cuda,
        _bateman_jac_cuda,
        _bateman_jac_cuda,
    ),
    "brusselator": (
        _brusselator_ode_cuda,
        _brusselator_explicit_cuda,
        _brusselator_implicit_cuda,
        _brusselator_implicit_jac_cuda,
        _brusselator_jac_cuda,
    ),
    "heat": (
        _heat_ode_cuda,
        _zero_cuda,
        _heat_ode_cuda,
        _heat_jac_cuda,
        _heat_jac_cuda,
    ),
    "kaps": (
        _kaps_ode_cuda,
        _kaps_explicit_cuda,
        _kaps_implicit_cuda,
        _kaps_implicit_jac_cuda,
        _kaps_jac_cuda,
    ),
    "lorenz": (
        lorenz.ode_fn_numba_cuda,
        lorenz.ode_fn_numba_cuda,
        _zero_cuda,
        _zero_jac_cuda,
        _lorenz_jac_cuda,
    ),
    "robertson": (
        robertson.ode_fn_numba_cuda,
        robertson.ode_fn_numba_cuda,
        _zero_cuda,
        _zero_jac_cuda,
        robertson.jac_fn_numba_cuda,
    ),
    "vdp": (_vdp_ode_cuda, _vdp_ode_cuda, _zero_cuda, _zero_jac_cuda, _vdp_jac_cuda),
}
