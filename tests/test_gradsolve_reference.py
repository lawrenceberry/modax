"""The gradsolve reference adapter, ``reference/solvers/python/gradsolve_engines.py``.

Both gradsolve Warp kernels the benchmarks time solve the reference systems to
the same answer as the modax solvers, and the record-and-replay gradient
matches modax's forward sensitivities. Everything here launches a kernel --
gradsolve's or modax's -- so the module needs a device, and gradsolve itself,
which installs on Python 3.11 up.
"""

import numpy as np
import pytest

from modax.rodas5P import solve as rodas5P_solve
from modax.tsit5 import solve as tsit5_solve
from reference.systems.python import lorenz, robertson, vdp

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("gradsolve")
pytest.importorskip("warp")

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

N = 8
KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
I_CONTROLLER = {"pcoeff": 0.0, "icoeff": 1.0, "dcoeff": 0.0}


@pytest.fixture(scope="module")
def ge():
    from reference.solvers.python import gradsolve_engines

    return gradsolve_engines


def test_tsit5_matches_modax_on_lorenz(ge):
    y0, params = lorenz.make_scenario(N, divergence=0.5)
    ref = np.asarray(
        tsit5_solve(lorenz.ode_fn, y0, lorenz.TIMES, params, **KWARGS, **I_CONTROLLER)
    )
    out = ge.solve(lorenz.ode_fn, y0, lorenz.TIMES, params, engine="warp_ode", **KWARGS)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out[:, 0], y0)
    np.testing.assert_allclose(out[:, -1], ref[:, -1], rtol=1e-3, atol=1e-6)


def test_tsit5_matches_modax_on_the_vdp_lattice(ge):
    n_osc = 4
    ode_fn, _ = vdp.make_system(n_osc, mu=1.0)
    y0, params = vdp.make_scenario(n_osc, N, divergence=0.0)
    ref = np.asarray(
        tsit5_solve(ode_fn, y0, vdp.TIMES, params, **KWARGS, **I_CONTROLLER)
    )
    out = ge.solve(ode_fn, y0, vdp.TIMES, params, engine="warp_ode", **KWARGS)
    np.testing.assert_allclose(out[:, -1], ref[:, -1], rtol=1e-4, atol=1e-6)


def test_rosenbrock23_matches_modax_on_robertson(ge):
    y0, params = robertson.make_scenario(N, divergence=0.5)
    ref = np.asarray(
        rodas5P_solve(robertson.ode_fn, y0, robertson.TIMES, params, **KWARGS)
    )
    out = ge.solve(
        robertson.ode_fn,
        y0,
        robertson.TIMES,
        params,
        engine="warp_rosenbrock",
        **KWARGS,
    )
    assert out.shape == ref.shape
    np.testing.assert_allclose(out[:, 0], y0)
    # Rosenbrock23 is second order, so tolerance-level agreement only.
    np.testing.assert_allclose(out[:, -1], ref[:, -1], rtol=1e-3, atol=1e-7)
    # The kernel returns the final state only.
    assert np.isnan(out[:, 1:-1]).all()


def test_unknown_engine_raises(ge):
    y0, params = lorenz.make_scenario(N, divergence=0.0)
    with pytest.raises(ValueError):
        ge.solve(lorenz.ode_fn, y0, lorenz.TIMES, params, engine="cuda_tsit5")


@pytest.mark.parametrize("record_each_call", [False, True])
def test_value_and_grad_matches_modax(ge, record_each_call):
    from benchmarks.benchmark_common import jit_value_and_grad

    n_osc = 2
    ode_fn, _ = vdp.make_system(n_osc)
    y0, params = vdp.make_scenario(n_osc, N, divergence=0.0)
    value, grad = jit_value_and_grad(
        rodas5P_solve, ode_fn, vdp.TIMES, sparsity=vdp.make_sparsity(n_osc), **KWARGS
    )(y0, params)
    run = ge.make_value_and_grad(
        ode_fn,
        y0,
        vdp.TIMES,
        params,
        engine="warp_rosenbrock",
        record_each_call=record_each_call,
        **KWARGS,
    )
    value_g, grad_g = run()
    assert np.asarray(grad_g).shape == params.shape
    np.testing.assert_allclose(float(value_g), float(value), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(grad_g), np.asarray(grad), rtol=1e-4)
