"""The Tsit5 kernel: one thread per trajectory, stage vectors in shared or local memory."""

from __future__ import annotations

import functools
import gc
import math
from dataclasses import dataclass
from typing import Literal, overload

import jax
from jax.typing import ArrayLike
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from modax._jax_common import (
    differentiable_solve,
    ensemble_ffi_call,
    prepare_inputs,
)
from modax._jax_numba_custom_call import CudaLaunch, make_launch
from modax._methods import Backend
from modax._numba_common import (
    SOLVER_ARGTYPES,
    OdeFn,
    PIDGains,
    SolveOutput,
    SolverStats,
    SolveSettings,
    make_cuda_local_vector_writer,
    make_step_controller,
)
from modax._params import Params
from modax._sensitivity import (
    RawOutput,
    SensitivitySpec,
    make_augmented_local_writer,
)
from modax._sensitivity import (
    clear_caches as clear_sensitivity_caches,
)

# fmt: off
C2 = 161.0 / 1000.0
C3 = 327.0 / 1000.0
C4 = 9.0 / 10.0
C5 = 0.9800255409045097
C6 = 1.0
C7 = 1.0

A21 = 161.0 / 1000.0

A31 = -0.008480655492356989
A32 = 0.335480655492357

A41 = 2.8971530571054935
A42 = -6.359448489975075
A43 = 4.3622954328695815

A51 = 5.325864828439257
A52 = -11.748883564062828
A53 = 7.4955393428898365
A54 = -0.09249506636175525

A61 = 5.86145544294642
A62 = -12.92096931784711
A63 = 8.159367898576159
A64 = -0.071584973281401
A65 = -0.028269050394068383

A71 = 0.09646076681806523
A72 = 0.01
A73 = 0.4798896504144996
A74 = 1.379008574103742
A75 = -3.2900695154360807
A76 = 2.324710524099774

B1 = A71
B2 = A72
B3 = A73
B4 = A74
B5 = A75
B6 = A76

E1 = 0.0017800110522257773
E2 = 0.0008164344596567463
E3 = -0.007880878010261994
E4 = 0.1447110071732629
E5 = -0.5823571654525552
E6 = 0.45808210592918686
E7 = -1.0 / 66.0
# fmt: on


# The step controller: the embedded estimate is fourth order, so the error
# order the exponents divide by is 5, and a step may grow tenfold at once.
ORDER = 5
FACTOR_MIN = 0.2
FACTOR_MAX = 10.0

# One thread per trajectory in both kernels; they differ only in where the nine
# per-trajectory vectors (the state, the trial state and the seven stages) live.
#
# The thread-local kernel keeps them in the thread's own ``cuda.local`` arrays,
# which the compiler promotes to registers where the indexing is resolvable and
# otherwise places in local memory: off-chip DRAM, cached in L2, its latency
# hidden by occupancy once the device is saturated. Nothing on chip bounds its
# block size, but a block is a warp anyway, as in Rodas5P: a small ensemble
# then spreads over as many SMs as it has warps instead of sitting on one, and
# a large one is indifferent (Lorenz at 1000 trajectories, measured: 3.2 ms in
# 32-thread blocks against 6.9 ms in 128-thread blocks; 83 ms either way at
# 100000).
#
# The shared-memory kernel keeps the same nine vectors in per-block shared
# memory, laid out ``(n_vars, _SHARED_BLOCK)`` so that consecutive lanes hold
# consecutive words and a warp's access is bank-conflict-free. It costs
# ``9 * n_vars * _SHARED_BLOCK * 8`` bytes of static shared memory per block,
# which caps ``n_vars``, and by capping the blocks an SM can hold it loses to
# the thread-local kernel once the ensemble saturates the device: up to 14%
# at 100000 trajectories of 8 to 16 components. Below that the two are within
# noise of each other on an RTX 4070 SUPER -- at these sizes the local arrays
# are promoted to registers, and the on-chip copy buys nothing over them --
# so ``"auto"`` takes shared wherever the system fits and the ensemble is
# small enough, where it is never slower, and local beyond. ``backend``
# overrides the choice.
_LOCAL_BLOCK = 32
_SHARED_BLOCK = 32
_SHARED_MAX_NVARS = 16
_SHARED_MAX_ENSEMBLE = 16384


@dataclass(frozen=True)
class KernelOptions:
    """The kernel options a solve sets besides ``ode_fn`` and the shapes.

    :py:obj:`uses_shared <modax.tsit5.KernelOptions.uses_shared>` resolves
    ``backend`` against the ensemble before it builds any kernel, so the kernel
    caches key on the gains, the spec and the resolved storage, not on this
    object.
    """

    gains: PIDGains = PIDGains()
    backend: Backend = "auto"
    spec: SensitivitySpec | None = None

    def uses_shared(self, n: int, n_system: int) -> bool:
        """Return whether ``n`` systems of ``n_system`` components run the shared-memory kernel.

        Raises:
            ValueError: If ``backend`` is unknown, or is ``"shared"`` for a
                system of more than 16 components.
        """
        if self.backend == "local":
            return False
        if self.backend == "shared":
            if n_system > _SHARED_MAX_NVARS:
                raise ValueError(
                    "shared backend requires a system size <= "
                    f"{_SHARED_MAX_NVARS}; got {n_system}. A solve carrying forward "
                    "sensitivities integrates n_vars * (1 + n_sens) components, so "
                    "it may need backend='local' where the plain solve does not"
                )
            return True
        if self.backend != "auto":
            raise ValueError(
                f"backend must be 'auto', 'shared', or 'local'; got {self.backend!r}"
            )
        return n_system <= _SHARED_MAX_NVARS and n <= _SHARED_MAX_ENSEMBLE


def clear_caches() -> None:
    """Drop the compiled kernels.

    Each ``n_vars`` compiles its own kernel and the module-level caches keep it
    alive, so call this when sweeping problem sizes in one process.
    """
    _make_body.cache_clear()
    _make_kernel.cache_clear()
    _make_jax_launch.cache_clear()
    clear_sensitivity_caches()
    gc.collect()


@functools.cache
def _make_body(
    ode_fn: OdeFn, n_vars: int, gains: PIDGains, spec: SensitivitySpec | None
) -> MLIRDispatcher:
    """Build the per-trajectory Tsit5 integration loop as a CUDA device function.

    The device function takes the trajectory index ``i`` into the ensemble's
    global arrays (``y0``, ``params``, ``weights``, ``hist``, the counters) and
    nine 1-D vectors of ``n_system`` components: ``y``, ``u`` and
    ``k1``..``k7``. The kernels pass either the thread's ``cuda.local`` arrays
    or its column of per-block ``cuda.shared`` arrays. One body for both keeps
    the two backends bit-identical.
    """
    next_step = make_step_controller(gains, ORDER, FACTOR_MIN, FACTOR_MAX)
    # With a sensitivity spec the loop integrates the joint [y, S] system: the
    # writer emits [f, J_y S + J_p] and every stage vector, the error norm and
    # the dense-output write run over n_aug components instead of n_vars. The
    # integrator itself is unchanged -- forward sensitivities are just a larger
    # ODE, which is what makes them cheap to bolt onto an existing kernel.
    if spec is None:
        ode_write = make_cuda_local_vector_writer(ode_fn, n_vars)
        n_system = n_vars
    else:
        ode_write = make_augmented_local_writer(ode_fn, spec)
        n_system = spec.n_aug
    n_error = n_vars if spec is None else spec.n_error

    @cuda.jit(device=True)
    def body(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        time_rows,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        y,
        u,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        i,
    ):
        prow = params[i]
        for j in range(n_system):
            y[j] = y0[i, j]
            hist[i, 0, j] = y0[i, j]
            k7[j] = 0.0

        # one save grid shared by the ensemble, or one row per trajectory,
        # read at an offset into the whole array
        n_save = times.shape[0] // time_rows
        t_row = (i if time_rows > 1 else 0) * n_save
        t = times[t_row]
        tf = times[t_row + n_save - 1]
        # A non-positive dt0 is the "no first step given" sentinel: start from
        # 1e-6 of the integration window.
        dt = dt0 if dt0 > 0.0 else (tf - t) * 1e-6
        save_idx = 1
        n_steps = 0
        accepted_steps = 0
        rejected_steps = 0
        # k1 holds f(t, y) once evaluated. It stays valid across a rejected
        # step, whose y and t are unchanged, so only the very first step
        # evaluates it directly; an accepted step inherits k7 (FSAL).
        k1_valid = False
        err_prev = 1.0
        err_prev2 = 1.0

        while save_idx < n_save and t < tf and n_steps < max_steps:
            dt_use = dt
            if dt_use > tf - t:
                dt_use = tf - t
            if dt_use < 1e-30:
                dt_use = 1e-30

            if not k1_valid:
                ode_write(y, t, prow, k1)
                k1_valid = True

            for j in range(n_system):
                u[j] = y[j] + dt_use * (A21 * k1[j])
            ode_write(u, t + C2 * dt_use, prow, k2)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (A31 * k1[j] + A32 * k2[j])
            ode_write(u, t + C3 * dt_use, prow, k3)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (A41 * k1[j] + A42 * k2[j] + A43 * k3[j])
            ode_write(u, t + C4 * dt_use, prow, k4)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (
                    A51 * k1[j] + A52 * k2[j] + A53 * k3[j] + A54 * k4[j]
                )
            ode_write(u, t + C5 * dt_use, prow, k5)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (
                    A61 * k1[j] + A62 * k2[j] + A63 * k3[j] + A64 * k4[j] + A65 * k5[j]
                )
            ode_write(u, t + C6 * dt_use, prow, k6)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (
                    B1 * k1[j]
                    + B2 * k2[j]
                    + B3 * k3[j]
                    + B4 * k4[j]
                    + B5 * k5[j]
                    + B6 * k6[j]
                )
            ode_write(u, t + C7 * dt_use, prow, k7)

            err_sum = 0.0
            for j in range(n_system):
                err_est = dt_use * (
                    E1 * k1[j]
                    + E2 * k2[j]
                    + E3 * k3[j]
                    + E4 * k4[j]
                    + E5 * k5[j]
                    + E6 * k6[j]
                    + E7 * k7[j]
                )
                scale = atol + rtol * max(abs(y[j]), abs(u[j]))
                r = weights[i, j] * err_est / scale
                err_sum += r * r
            err_norm = math.sqrt(err_sum / n_error)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

            t_new = t
            if accept:
                t_new = t + dt_use
                while save_idx < n_save and times[
                    t_row + save_idx
                ] <= t_new + 1e-12 * max(1.0, abs(times[t_row + save_idx])):
                    theta = (times[t_row + save_idx] - t) / dt_use
                    b1 = (
                        -1.0530884977290216
                        * theta
                        * (theta - 1.3299890189751412)
                        * (
                            theta * theta
                            - 1.4364028541716351 * theta
                            + 0.7139816917074209
                        )
                    )
                    b2 = (
                        0.1017
                        * theta
                        * theta
                        * (
                            theta * theta
                            - 2.1966568338249754 * theta
                            + 1.2949852507374631
                        )
                    )
                    b3 = (
                        2.490627285651252793
                        * theta
                        * theta
                        * (
                            theta * theta
                            - 2.38535645472061657 * theta
                            + 1.57803468208092486
                        )
                    )
                    b4 = (
                        -16.54810288924490272
                        * (theta - 1.21712927295533244)
                        * (theta - 0.61620406037800089)
                        * theta
                        * theta
                    )
                    b5 = (
                        47.37952196281928122
                        * (theta - 1.203071208372362603)
                        * (theta - 0.658047292653547382)
                        * theta
                        * theta
                    )
                    b6 = (
                        -34.87065786149660974
                        * (theta - 1.2)
                        * (theta - 0.666666666666666667)
                        * theta
                        * theta
                    )
                    b7 = 2.5 * (theta - 1.0) * (theta - 0.6) * theta * theta
                    for j in range(n_system):
                        hist[i, save_idx, j] = y[j] + dt_use * (
                            b1 * k1[j]
                            + b2 * k2[j]
                            + b3 * k3[j]
                            + b4 * k4[j]
                            + b5 * k5[j]
                            + b6 * k6[j]
                            + b7 * k7[j]
                        )
                    save_idx += 1
                for j in range(n_system):
                    y[j] = u[j]
                accepted_steps += 1
                # FSAL: k7 was evaluated at (t + h, y_{n+1}), the new (t, y).
                for j in range(n_system):
                    k1[j] = k7[j]
            else:
                rejected_steps += 1

            factor, err_prev, err_prev2 = next_step(
                err_norm, err_prev, err_prev2, accept
            )
            dt = dt_use * factor
            t = t_new
            n_steps += 1

        accepted_out[i] = accepted_steps
        rejected_out[i] = rejected_steps
        loop_out[i] = n_steps

    return body


@functools.cache
def _make_kernel(
    ode_fn: OdeFn,
    n_vars: int,
    gains: PIDGains,
    spec: SensitivitySpec | None,
    shared: bool,
) -> MLIRDispatcher:
    """Build a Tsit5 kernel with one thread per trajectory.

    ``shared=False`` puts the nine per-trajectory vectors in the thread's own
    ``cuda.local`` arrays. ``shared=True`` puts them in nine ``(n_system,
    _SHARED_BLOCK)`` ``cuda.shared`` arrays and hands the body the thread's
    column of each, a strided 1-D view. Each thread touches only its own
    column, so nothing synchronises and a thread past the ensemble's end just
    returns. The launch carries no scratch either way, and ``y0``,
    ``weights`` and ``hist`` keep their ``(n, ...)`` layouts.
    """
    body = _make_body(ode_fn, n_vars, gains, spec)
    n_system = n_vars if spec is None else spec.n_aug

    if not shared:

        @cuda.jit
        def local_kernel(
            y0,
            times,
            params,
            dt0,
            rtol,
            atol,
            max_steps,
            time_rows,
            weights,
            hist,
            accepted_out,
            rejected_out,
            loop_out,
        ):
            i = cuda.grid(1)  # ty: ignore[unresolved-attribute]
            if i >= y0.shape[0]:
                return
            y = cuda.local.array(n_system, types.float64)
            u = cuda.local.array(n_system, types.float64)
            k1 = cuda.local.array(n_system, types.float64)
            k2 = cuda.local.array(n_system, types.float64)
            k3 = cuda.local.array(n_system, types.float64)
            k4 = cuda.local.array(n_system, types.float64)
            k5 = cuda.local.array(n_system, types.float64)
            k6 = cuda.local.array(n_system, types.float64)
            k7 = cuda.local.array(n_system, types.float64)
            body(
                y0,
                times,
                params,
                dt0,
                rtol,
                atol,
                max_steps,
                time_rows,
                weights,
                hist,
                accepted_out,
                rejected_out,
                loop_out,
                y,
                u,
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                k7,
                i,
            )

        return local_kernel

    shape = (n_system, _SHARED_BLOCK)

    # The column views are taken here, one device function down from the
    # kernel, and not in the kernel itself: there the shared arrays' shapes are
    # static and numba-cuda-mlir mis-types the slice (a ``memref.collapse_shape``
    # whose stride it expects static and emits dynamic, which fails
    # verification). As a device-function argument the array is a dynamic
    # memref and the slice lowers as it should. Inlining folds the layer away.
    @cuda.jit(device=True)
    def run_column(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        time_rows,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        y,
        u,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        tx,
        i,
    ):
        body(
            y0,
            times,
            params,
            dt0,
            rtol,
            atol,
            max_steps,
            time_rows,
            weights,
            hist,
            accepted_out,
            rejected_out,
            loop_out,
            y[:, tx],
            u[:, tx],
            k1[:, tx],
            k2[:, tx],
            k3[:, tx],
            k4[:, tx],
            k5[:, tx],
            k6[:, tx],
            k7[:, tx],
            i,
        )

    @cuda.jit
    def shared_kernel(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        time_rows,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
    ):
        i = cuda.grid(1)  # ty: ignore[unresolved-attribute]
        tx = cuda.threadIdx.x  # ty: ignore[unresolved-attribute]
        if i >= y0.shape[0]:
            return
        y = cuda.shared.array(shape, types.float64)
        u = cuda.shared.array(shape, types.float64)
        k1 = cuda.shared.array(shape, types.float64)
        k2 = cuda.shared.array(shape, types.float64)
        k3 = cuda.shared.array(shape, types.float64)
        k4 = cuda.shared.array(shape, types.float64)
        k5 = cuda.shared.array(shape, types.float64)
        k6 = cuda.shared.array(shape, types.float64)
        k7 = cuda.shared.array(shape, types.float64)
        run_column(
            y0,
            times,
            params,
            dt0,
            rtol,
            atol,
            max_steps,
            time_rows,
            weights,
            hist,
            accepted_out,
            rejected_out,
            loop_out,
            y,
            u,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            tx,
            i,
        )

    return shared_kernel


@functools.cache
def _make_jax_launch(
    ode_fn: OdeFn,
    n: int,
    n_vars: int,
    gains: PIDGains,
    spec: SensitivitySpec | None,
    shared: bool,
) -> CudaLaunch:
    """Compile, load and size the kernel behind one JAX-side ensemble launch."""
    kernel = _make_kernel(ode_fn, n_vars, gains, spec, shared)
    threads = _SHARED_BLOCK if shared else _LOCAL_BLOCK
    blocks = (n + threads - 1) // threads
    return make_launch(kernel, SOLVER_ARGTYPES, grid=blocks, block=threads)


@overload
def solve(
    ode_fn: OdeFn,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step: float | None = None,
    max_steps: int = 100000,
    return_stats: Literal[False] = False,
    error_weights: ArrayLike | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    backend: Backend = "auto",
    sens_error_control: bool = True,
) -> jax.Array: ...


@overload
def solve(
    ode_fn: OdeFn,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step: float | None = None,
    max_steps: int = 100000,
    return_stats: Literal[True],
    error_weights: ArrayLike | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    backend: Backend = "auto",
    sens_error_control: bool = True,
) -> tuple[jax.Array, SolverStats]: ...


@overload
def solve(
    ode_fn: OdeFn,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step: float | None = None,
    max_steps: int = 100000,
    return_stats: bool,
    error_weights: ArrayLike | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    backend: Backend = "auto",
    sens_error_control: bool = True,
) -> SolveOutput: ...


def solve(
    ode_fn: OdeFn,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step: float | None = None,
    max_steps: int = 100000,
    return_stats: bool = False,
    error_weights: ArrayLike | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    backend: Backend = "auto",
    sens_error_control: bool = True,
) -> SolveOutput:
    """Integrate an ensemble with the Tsit5 kernel, callable from JAX.

    The solve is an XLA custom call with a ``jax.custom_jvp`` rule.
    ``jax.jvp``, ``jax.jacfwd``, ``jax.grad``, ``jax.jacrev`` and
    ``jax.value_and_grad`` work with respect to ``y0`` and ``params`` by
    integrating the forward-sensitivity system with the state (see
    ``modax/_sensitivity.py``). An undifferentiated call runs the plain kernel
    and pays nothing extra.

    Args:
        ode_fn: The right-hand side, ``(y, t, p) -> tuple`` of scalars. See
            Writing ODE callbacks. A plain function is ``cuda.jit``-ed for you.
        y0: Initial state, ``(n_vars,)`` or ``(N, n_vars)``. Every trajectory
            shares a 1-D state.
        t_span: Save times, ``(n_save,)`` shared by the ensemble or
            ``(N, n_save)`` with one increasing row per trajectory. A row's
            first entry is where that trajectory starts and its last where it
            stops. Not differentiable.
        params: Parameters: an array, ``(n_params,)`` or ``(N, n_params)``,
            or a pytree of them (a tuple, a dict) whose leaves are ``()``,
            ``(k,)`` or ``(N, k)`` and concatenate, in ``jax.tree.leaves``
            order, into the row ``ode_fn`` sees. Differentiation follows the
            leaves: only the columns of the leaves JAX differentiates get a
            sensitivity, so ``jax.grad(loss)(rates, temp)`` with
            ``params=(rates, temp)`` integrates the rates' columns alone,
            when ``temp`` is a constant or wrapped in ``stop_gradient``. ``N`` comes
            from whichever of ``y0``, ``params`` and ``t_span`` has a batch
            axis. Batched ones must agree.
        rtol: Relative tolerance of the weighted RMS error norm.
        atol: Absolute tolerance of the weighted RMS error norm.
        first_step: Initial step size. ``None`` or any non-positive value
            starts the kernel from 1e-6 of its own integration window. It is a
            launch-time scalar because JAX traces ``t_span``.
        max_steps: Per-trajectory step budget.
        return_stats: Also return a dict of per-trajectory step counters,
            ``accepted_steps``, ``rejected_steps`` and ``loop_steps``, each
            ``(N,)``.
        error_weights: Per-component weights of the error norm, ``(n_vars,)``
            or ``(N, n_vars)``. A weight of 0 takes that component out of
            step-size control.
        pcoeff: Proportional gain of the PID step controller.
        icoeff: Integral gain. The default ``(0, 1, 0)`` is the classic
            I-controller.
        dcoeff: Derivative gain.
        backend: Where the state and stage vectors live. ``"shared"`` uses
            per-block shared memory and ``"local"`` the thread's own local
            memory. The two are bit-identical, and local wins by up to 14%
            once the device saturates. ``"auto"`` picks shared for at most 16
            components and at most 16384 trajectories, and local otherwise.
        sens_error_control: Whether a differentiated solve holds the
            sensitivities to ``rtol``/``atol`` too. ``True`` makes the
            gradient as accurate as the value. ``False`` drops them from the
            error norm, so the joint solve takes the plain solve's steps and
            returns its value bit for bit, but nothing then ties the
            sensitivities' accuracy to ``rtol``.


    Returns:
        The history, ``(N, n_save, n_vars)``, followed by the stats dict when
        ``return_stats=True``.
    """
    from modax._hierarchy import as_hierarchy

    if as_hierarchy(ode_fn) is not None:
        raise TypeError(
            "tsit5.solve takes a plain right-hand side; solve a Block or Hierarchy with modax.solve"
        )

    options = KernelOptions(PIDGains(pcoeff, icoeff, dcoeff), backend)
    settings = SolveSettings(rtol, atol, first_step, max_steps, error_weights)

    # no hook here, so never a HookedOutput
    return differentiable_solve(  # ty: ignore[invalid-return-type]
        functools.partial(_solve_impl, ode_fn),
        options,
        settings,
        return_stats,
        y0,
        t_span,
        params,
        sens_error_control=sens_error_control,
    )


def _solve_impl(
    ode_fn: OdeFn,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    *,
    options: KernelOptions,
    settings: SolveSettings,
    return_stats: bool,
) -> RawOutput:
    inputs = prepare_inputs(y0, t_span, params, settings, options.spec)
    shared = options.uses_shared(inputs.n, inputs.n_system)
    launch = _make_jax_launch(
        ode_fn, inputs.n, inputs.n_vars, options.gains, options.spec, shared
    )
    # No global scratch either way: the kernel keeps the state and the stage
    # vectors on chip or in thread-local memory.
    hist, accepted, rejected, loop_steps = ensemble_ffi_call(launch, inputs, settings)
    stats = SolverStats(
        accepted_steps=accepted, rejected_steps=rejected, loop_steps=loop_steps
    )
    # Tsit5's own kernel has no save hook; a hooked Tsit5 solve runs on the
    # hierarchy kernel.
    return hist, None, (stats if return_stats else None)
