"""What the two kernels share: the callback coercion, the solve settings, the
step controller and the kernel signature."""

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

from modax._jax_numba_custom_call import ABI_ARRAY, ABI_SCALAR_F64, ABI_SCALAR_I32

OdeFn: TypeAlias = (
    Callable[[Sequence[float], float, Sequence[float]], tuple[float, ...]]
    | MLIRDispatcher
)
"""The right-hand side, ``(y, t, p) -> tuple`` of scalars.

A plain function is compiled with ``cuda.jit(device=True)`` by
[`as_cuda_device`][modax._numba_common.as_cuda_device]; one already compiled
is used as it is.
"""


class SolverStats(TypedDict):
    """The ``return_stats`` dict: per-trajectory ``(N,)`` step counters."""

    accepted_steps: jax.Array
    rejected_steps: jax.Array
    loop_steps: jax.Array


SolveOutput: TypeAlias = jax.Array | tuple[jax.Array, SolverStats]
"""What an unhooked solve returns: the history, with the stats when asked."""

Solver: TypeAlias = Callable[[ArrayLike, ArrayLike, ArrayLike], SolveOutput]
"""A ``(y0, t_span, params)`` solve of fixed callback and settings."""


@dataclass(frozen=True)
class SolveSettings:
    """The per-call settings both solvers take and hand to the kernel as scalars.

    Unlike the kernel options these select no compiled code: they are launch
    arguments, and the error weights are an array input.
    """

    rtol: float = 1e-8
    atol: float = 1e-10
    first_step: float | None = None
    max_steps: int = 100_000
    error_weights: ArrayLike | None = None

    @property
    def dt0(self) -> np.float64:
        """The initial step the kernels take, from ``first_step``.

        A launch-time scalar, so it cannot be derived here from the save
        times: under ``jit`` (and under the ``custom_vmap`` rule, which traces
        unconditionally) those are tracers. With no first step pinned it is
        the non-positive sentinel, and each kernel takes 1e-6 of its own
        integration window off the ``times`` array it already reads.
        """
        return np.float64(0.0 if self.first_step is None else self.first_step)


@dataclass(frozen=True)
class PIDGains:
    """The gains of the proportional-integral-derivative step controller.

    The default ``(0, 1, 0)`` is the elementary integral controller. Non-zero
    proportional and derivative gains give smoother step-size sequences and
    fewer rejected steps on stiff and mildly oscillatory problems, which
    reduces the spread of step counts across an ensemble and with it the
    divergence between the threads of a warp.
    """

    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0

    def exponents(self, order: int) -> tuple[float, float, float]:
        """``(e1, e2, e3)`` of Soderlind's controller for an error of ``order``.

        ``h_next = h * 0.9 * err^e1 * err_prev^e2 * err_prev2^e3``, with
        ``e1 = -(k_I + k_P + k_D) / q``, ``e2 = (k_P + 2 k_D) / q`` and
        ``e3 = -k_D / q``. The defaults give ``e1 = -1 / q`` and
        ``e2 = e3 = 0``, the elementary controller exactly.
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
    """The device function that rescales the step after every attempt.

    ``next_step(err_norm, err_prev, err_prev2, accept)`` returns the factor to
    multiply the step by and the error history to carry forward, which
    advances only on an accepted step. The error is clamped away from zero,
    infinity and NaN before it is raised to the exponents, so a rejected step
    with a NaN norm shrinks the step by ``factor_min`` rather than poisoning
    the history. Both kernels call this once per attempted step; they differ
    only in the error order and in how far a step may grow or shrink at once.
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
    _F64_1D,  # times
    _F64_2D,  # params
    types.float64,  # dt0
    types.float64,  # rtol
    types.float64,  # atol
    types.int32,  # max_steps
    _F64_2D,  # error weights
    types.float64[:, :, ::1],  # hist
    _I32_1D,  # accepted steps
    _I32_1D,  # rejected steps
    _I32_1D,  # loop steps
)
HOOK_OUT_ARGTYPE: Type = _F64_2D
"""Per-trajectory rows a save hook accumulates into (``(n, hook_size)``)."""
SOLVER_INPUT_KINDS: tuple[int, ...] = (
    ABI_ARRAY,
    ABI_ARRAY,
    ABI_ARRAY,
    ABI_SCALAR_F64,
    ABI_SCALAR_F64,
    ABI_SCALAR_F64,
    ABI_SCALAR_I32,
    ABI_ARRAY,
)


def build_error_weights(
    error_weights: ArrayLike | None, n: int, n_vars: int
) -> npt.NDArray[np.float64]:
    """Broadcast a user ``error_weights`` argument to a ``(n, n_vars)`` array.

    ``None`` yields all-ones (every component weighted equally); a 1-D array of
    length ``n_vars`` is broadcast across trajectories; a 2-D ``(n, n_vars)``
    array is used as-is. This array is copied to the device and read per
    component as the ``weight`` argument of the kernel's error-contribution
    device function.
    """
    if error_weights is None:
        return np.ones((n, n_vars), dtype=np.float64)
    weights = np.asarray(error_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n, n_vars))
    return np.ascontiguousarray(weights, dtype=np.float64)


def solver_stats(
    accepted: jax.Array, rejected: jax.Array, loop_steps: jax.Array
) -> SolverStats:
    """The ``return_stats`` dict: per-trajectory ``(N,)`` step counters.

    ``loop_steps`` counts iterations of the stepping loop, which is what
    ``max_steps`` caps.
    """
    return {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "loop_steps": loop_steps,
    }


@functools.cache
def as_cuda_device(fn: Callable[..., object] | MLIRDispatcher) -> MLIRDispatcher:
    """``fn`` as a CUDA device function: ``cuda.jit(device=True)`` unless it
    already is one, cached so a callback is compiled once.
    """
    if isinstance(fn, MLIRDispatcher):
        return fn
    return cuda.jit(device=True)(fn)


def make_cuda_local_vector_writer(fn: OdeFn, n_vars: int) -> MLIRDispatcher:
    """A vector writer over one trajectory's own thread-local arrays.

    Both kernels run one trajectory per thread, so there is no stripe to share:
    the thread owning the trajectory writes the whole vector, and both
    ``y_row`` and ``out`` are its own local memory rather than rows of a global
    scratch array.
    """
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_vector(y_row, t, p_row, out):
        values = fn_device(y_row, t, p_row)
        for j in range(n_vars):
            out[j] = values[j]

    return write_vector
