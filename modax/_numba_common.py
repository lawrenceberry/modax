"""What the two kernels share: callback coercion, solve settings, the step
controller and the kernel signature."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias, TypedDict

import jax
import numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher
from numba_cuda_mlir.numba_cuda.types import Type

from modax._compile_once import install as _compile_callees_once
from modax._jax_numba_custom_call import ABI_ARRAY, ABI_SCALAR_F64, ABI_SCALAR_I32

# numba-cuda-mlir compiles every device callee twice without it (modax._compile_once)
_compile_callees_once()

OdeFn: TypeAlias = (
    Callable[[Sequence[float], float, Sequence[float]], tuple[float, ...]]
    | MLIRDispatcher
)
"""The right-hand side, ``(y, t, p) -> tuple`` of scalars.

:py:obj:`as_cuda_device <modax._numba_common.as_cuda_device>` compiles a plain
function with ``cuda.jit(device=True)`` and passes a compiled one through.
"""


class SolverStats(TypedDict):
    """The ``return_stats`` dict: per-trajectory ``(N,)`` step counters."""

    accepted_steps: jax.Array
    rejected_steps: jax.Array
    loop_steps: jax.Array


SolveOutput: TypeAlias = jax.Array | tuple[jax.Array, SolverStats]
"""What an unhooked solve returns: the history, with the stats when asked."""

HookedOutput: TypeAlias = (
    tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, SolverStats]
)
"""What a hooked solve returns: the history and the hook's ``(N, n_save,
n_out)`` quantities, with the stats when asked."""

SaveHook: TypeAlias = (
    Callable[[Sequence[float], float, Sequence[float]], tuple[float, ...]]
    | MLIRDispatcher
)
"""A save hook, ``(y, t, p) -> tuple`` of the quantities to keep at every save.

Shaped like the right-hand side, and compiled the same way. The length of the
tuple it returns is the number of quantities, read from its inferred type.
"""

Solver: TypeAlias = Callable[[ArrayLike, ArrayLike, ArrayLike], SolveOutput]
"""A ``(y0, t_span, params)`` solve of fixed callback and settings."""


@dataclass(frozen=True)
class SolveSettings:
    """Per-call settings both solvers pass to the kernel at launch.

    These select no compiled code. The tolerances, ``first_step`` and
    ``max_steps`` go in as scalars and the error weights as an array input.
    """

    rtol: float = 1e-8
    atol: float = 1e-10
    first_step: float | None = None
    max_steps: int = 100_000
    error_weights: ArrayLike | None = None

    @property
    def dt0(self) -> np.float64:
        """The kernel's initial step, from ``first_step``.

        The host can't derive it from the save times, which are tracers under
        ``jit`` and the ``custom_vmap`` rule. Without ``first_step`` it is 0,
        a sentinel that makes each kernel start from 1e-6 of its own
        integration window.
        """
        return np.float64(0.0 if self.first_step is None else self.first_step)


@dataclass(frozen=True)
class PIDGains:
    """Gains of the proportional-integral-derivative step controller.

    The default ``(0, 1, 0)`` is the elementary integral controller. Non-zero
    proportional and derivative gains smooth the step-size sequence and cut
    rejected steps on stiff and mildly oscillatory problems. That narrows the
    spread of step counts across an ensemble, and so the divergence between
    threads of a warp.

    Attributes:
        pcoeff: Proportional gain ``k_P``. Defaults to 0.
        icoeff: Integral gain ``k_I``. Defaults to 1.
        dcoeff: Derivative gain ``k_D``. Defaults to 0.
    """

    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0

    def exponents(self, order: int) -> tuple[float, float, float]:
        """``(e1, e2, e3)`` of Soderlind's controller for an error of ``order``.

        ``h_next = h * 0.9 * err^e1 * err_prev^e2 * err_prev2^e3``, with
        ``e1 = -(k_I + k_P + k_D) / q``, ``e2 = (k_P + 2 k_D) / q`` and
        ``e3 = -k_D / q``. The defaults give ``e1 = -1 / q`` and
        ``e2 = e3 = 0``, which is exactly the elementary controller.
        """
        q = -1.0 / order
        return (
            q * (self.icoeff + self.pcoeff + self.dcoeff),
            -q * (self.pcoeff + 2.0 * self.dcoeff),
            q * self.dcoeff,
        )


SAFETY = 0.9


def make_step_controller(
    gains: PIDGains, order: int, factor_min: float, factor_max: float
) -> MLIRDispatcher:
    """Build the device function that rescales the step after every attempt.

    ``next_step(err_norm, err_prev, err_prev2, accept)`` returns the step
    factor, clipped to ``[factor_min, factor_max]``, and the error history to
    carry forward, which advances only on an accepted step. It clamps the
    error away from zero, infinity and NaN first, so a NaN norm shrinks the
    step by ``factor_min`` instead of poisoning the history. Both kernels call
    it once per attempted step, with their own ``order`` and factor bounds.
    """
    e1, e2, e3 = gains.exponents(order)

    @cuda.jit(device=True)
    def next_step(err_norm, err_prev, err_prev2, accept):
        if math.isnan(err_norm) or err_norm > 1e18:
            safe_err = 1e18
        elif err_norm == 0.0:
            safe_err = 1e-18
        else:
            safe_err = err_norm
        factor = SAFETY * safe_err**e1 * err_prev**e2 * err_prev2**e3
        if accept:
            err_prev2 = err_prev
            err_prev = safe_err
        if factor < factor_min:
            factor = factor_min
        elif factor > factor_max:
            factor = factor_max
        return factor, err_prev, err_prev2

    return next_step


# Both kernels open with the same parameters -- the ensemble inputs, the scalar
# step-control settings, and the history and counter outputs -- and differ only
# in what follows (Rodas5P appends its save-hook output), so the leading argument types and ABI kinds
# are declared once here.
_F64_1D = types.float64[::1]
_F64_2D = types.float64[:, ::1]
_I32_1D = types.int32[::1]
SOLVER_ARGTYPES: tuple[Type, ...] = (
    _F64_2D,  # y0
    _F64_1D,  # times: n_save shared, or N * n_save, one row per trajectory
    _F64_2D,  # params
    types.float64,  # dt0
    types.float64,  # rtol
    types.float64,  # atol
    types.int32,  # max_steps
    types.int32,  # time_rows: 1 for a shared grid, else N
    _F64_2D,  # error weights
    types.float64[:, :, ::1],  # hist
    _I32_1D,  # accepted steps
    _I32_1D,  # rejected steps
    _I32_1D,  # loop steps
)
HOOK_OUT_ARGTYPE: Type = _F64_2D
"""The ``(n, hook_size)`` rows a save hook accumulates into."""
SOLVER_INPUT_KINDS: tuple[int, ...] = (
    ABI_ARRAY,
    ABI_ARRAY,
    ABI_ARRAY,
    ABI_SCALAR_F64,
    ABI_SCALAR_F64,
    ABI_SCALAR_F64,
    ABI_SCALAR_I32,
    ABI_SCALAR_I32,
    ABI_ARRAY,
)


def build_error_weights(
    error_weights: ArrayLike | None, n: int, n_vars: int
) -> npt.NDArray[np.float64]:
    """Broadcast a user ``error_weights`` argument to an ``(n, n_vars)`` array.

    ``None`` gives all ones. A 1-D array of length ``n_vars`` applies to every
    trajectory, and a 2-D ``(n, n_vars)`` array passes through as it is. The
    kernel reads it per component as the weight in each error contribution.
    """
    if error_weights is None:
        return np.ones((n, n_vars), dtype=np.float64)
    weights = np.asarray(error_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n, n_vars))
    return np.ascontiguousarray(weights, dtype=np.float64)


@functools.cache
def as_cuda_device(fn: Callable[..., object] | MLIRDispatcher) -> MLIRDispatcher:
    """Return ``fn`` as a CUDA device function, compiling it at most once.

    Applies ``cuda.jit(device=True)`` unless ``fn`` already is one.
    """
    if isinstance(fn, MLIRDispatcher):
        return fn
    return cuda.jit(device=True)(fn)


def make_cuda_local_vector_writer(fn: OdeFn, n_vars: int) -> MLIRDispatcher:
    """Build a device function that writes ``fn``'s values into an array.

    ``write_vector(y_row, t, p_row, out)`` stores all ``n_vars`` components
    of ``fn(y_row, t, p_row)`` in ``out``. Each thread owns one trajectory,
    so ``y_row`` and ``out`` are its own local arrays.
    """
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_vector(y_row, t, p_row, out):
        values = fn_device(y_row, t, p_row)
        for j in range(n_vars):
            out[j] = values[j]

    return write_vector
