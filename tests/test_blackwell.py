"""Compiling for Blackwell GPUs (compute capability 10.0 and up).

numba-cuda-mlir lowers a Blackwell target through its LLVM 20 dialect, which
the Enzyme derivatives a kernel links cannot share, so modax compiles every
kernel and every derivative for 9.0 there and lets numba-cuda-mlir link the
result for the device (``compile_target``). The first group of tests checks
that choice with the GPU query mocked and needs no GPU. The second compiles
and links real kernels for a pretended Blackwell device, so it needs some GPU
and a toolkit that knows Blackwell. The last solves on Blackwell itself and is
skipped everywhere else.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda, tools
from numba_enzyme.cuda import CUDAEnzymeError

from modax import _jax_numba_custom_call as bridge
from modax import _sensitivity

jax.config.update("jax_enable_x64", True)

PRE_BLACKWELL = [(7, 0), (8, 9), (9, 0)]
BLACKWELL = [(10, 0), (10, 3), (12, 0)]


@pytest.fixture
def device(monkeypatch):
    """Pretend the GPU has a given compute capability, without querying it."""

    monkeypatch.delenv("NUMBA_ENZYME_CUDA_CC", raising=False)
    query = Mock(return_value=(12, 0))
    monkeypatch.setattr(tools, "get_gpu_compute_capability", query)
    bridge.compile_target.cache_clear()
    yield query
    bridge.compile_target.cache_clear()


def _kernel():
    return SimpleNamespace(targetoptions={"max_registers": 168})


@pytest.mark.parametrize("capability", PRE_BLACKWELL)
def test_older_gpus_are_their_own_target(device, capability):
    device.return_value = capability
    assert bridge.compile_target() == capability
    kernel = _kernel()
    assert bridge.target_kernel(kernel) is kernel
    assert kernel.targetoptions == {"max_registers": 168}


@pytest.mark.parametrize("capability", BLACKWELL)
def test_blackwell_compiles_for_hopper(device, capability):
    device.return_value = capability
    assert bridge.compile_target() == (9, 0)
    kernel = bridge.target_kernel(_kernel())
    assert kernel.targetoptions == {"max_registers": 168, "chip": "sm_90"}


def test_the_target_is_resolved_once(device):
    bridge.compile_target()
    bridge.compile_target()
    device.assert_called_once_with(tuple)


def test_an_explicit_target_overrides_the_device(device, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CUDA_CC", "8.6")
    assert bridge.compile_target() == (8, 6)
    assert bridge.target_kernel(_kernel()).targetoptions["chip"] == "sm_86"


def test_an_explicit_target_is_validated_as_numba_enzyme_does(device, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CUDA_CC", "6.1")
    with pytest.raises(CUDAEnzymeError, match=r"7\.0\+"):
        bridge.compile_target()


def test_derivatives_are_built_for_the_kernel_target(device, monkeypatch):
    """Both tangents carry the target, so they link with the kernel."""

    jvp = Mock(side_effect=lambda fn, **options: fn)
    monkeypatch.setattr(_sensitivity, "jvp", jvp)
    _sensitivity.clear_caches()
    try:

        def rhs(y, t, p):
            return (-p[0] * y[0],)

        _sensitivity.make_tangent(rhs, 1, 1)
        _sensitivity.make_second_tangent(rhs, 1, 1)
    finally:
        _sensitivity.clear_caches()
    assert jvp.call_count == 3
    assert all(call.kwargs["cc"] == (9, 0) for call in jvp.call_args_list)


# --- compiling for a pretended Blackwell device ------------------------------
def _toolkit_knows_blackwell():
    if not cuda.is_available():
        return False
    from cuda.bindings import nvjitlink

    return nvjitlink.version() >= (12, 8)


requires_a_blackwell_toolkit = pytest.mark.skipif(
    not _toolkit_knows_blackwell(),
    reason="needs a CUDA GPU (any) and a CUDA 12.8+ toolkit",
)


@pytest.fixture
def pretend_blackwell(monkeypatch):
    """Make numba-cuda-mlir believe the GPU is sm_120 for this test only.

    Compiling a kernel initialises the driver, so a GPU is needed, but its
    linked cubin is never loaded: the GPU it was built for is not there.
    """

    monkeypatch.delenv("NUMBA_ENZYME_CUDA_CC", raising=False)
    tools.get_gpu_compute_capability(tuple)  # cache the real one first
    monkeypatch.setattr(tools, "_cached_cc", (12, 0))
    bridge.compile_target.cache_clear()
    _sensitivity.clear_caches()
    yield
    bridge.compile_target.cache_clear()
    _sensitivity.clear_caches()


def decay(y, t, p):
    return (-p[0] * y[0] + p[1] * y[1], p[0] * y[0] - p[1] * y[1])


@requires_a_blackwell_toolkit
@pytest.mark.parametrize("with_sensitivities", [False, True])
def test_rodas5p_links_for_blackwell(pretend_blackwell, with_sensitivities):
    """The kernel and its Enzyme Jacobian compile for 9.0 and link for 12.0."""

    from modax.rodas5P import HOOK_OUT_ARGTYPE, SOLVER_ARGTYPES, KernelOptions
    from modax.rodas5P import _make_kernel as make_rodas5p_kernel

    spec = (
        _sensitivity.SensitivitySpec(2, 2, wrt_y0=False, wrt_params=True)
        if with_sensitivities
        else None
    )
    kernel, _ = make_rodas5p_kernel(decay, 2, 2, KernelOptions(spec=spec))
    cres = bridge.target_kernel(kernel).compile(
        SOLVER_ARGTYPES + (HOOK_OUT_ARGTYPE,)
    ).cres
    assert kernel.targetoptions["chip"] == "sm_90"
    assert cres.metadata["cubin"]


@requires_a_blackwell_toolkit
def test_tsit5_sensitivities_link_for_blackwell(pretend_blackwell):
    """Tsit5's kernel links an Enzyme tangent only when it carries sensitivities."""

    from modax._numba_common import PIDGains
    from modax.tsit5 import SOLVER_ARGTYPES
    from modax.tsit5 import _make_kernel as make_tsit5_kernel

    spec = _sensitivity.SensitivitySpec(2, 2, wrt_y0=True, wrt_params=True)
    kernel = make_tsit5_kernel(decay, 2, PIDGains(), spec, False)
    cres = bridge.target_kernel(kernel).compile(SOLVER_ARGTYPES).cres
    assert cres.metadata["cubin"]


# --- on Blackwell itself -----------------------------------------------------
def _on_blackwell():
    return cuda.is_available() and tools.get_gpu_compute_capability(tuple) >= (10, 0)


@pytest.mark.skipif(not _on_blackwell(), reason="needs a Blackwell GPU")
@pytest.mark.parametrize("solver", ["rodas5P", "tsit5"])
def test_solves_and_differentiates_on_blackwell(solver):
    """A linear decay against its closed form, value and parameter derivative."""

    from modax.rodas5P import solve as rodas5p_solve
    from modax.tsit5 import solve as tsit5_solve

    solve = {"rodas5P": rodas5p_solve, "tsit5": tsit5_solve}[solver]

    def one(y, t, p):
        return (-p[0] * y[0],)

    t_span = jnp.linspace(0.0, 2.0, 5)
    y0 = jnp.asarray([[1.5], [0.5]])
    rates = jnp.asarray([[0.7], [2.0]])

    def final(p):
        return solve(one, y0, t_span, p, rtol=1e-10, atol=1e-12)[:, -1, 0]

    t = float(t_span[-1])
    exact = np.asarray(y0[:, 0] * jnp.exp(-rates[:, 0] * t))
    np.testing.assert_allclose(np.asarray(final(rates)), exact, rtol=1e-7)
    jacobian = np.asarray(jax.jacfwd(final)(rates))[:, :, 0]
    np.testing.assert_allclose(np.diag(jacobian), -t * exact, rtol=1e-6)
