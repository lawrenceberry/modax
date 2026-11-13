"""The JAX-facing glue: ensemble inputs, the FFI launch and the ``vmap`` rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.typing import ArrayLike

from modax._jax_numba_custom_call import CudaLaunch, ffi_abi_call
from modax._numba_common import (
    SOLVER_INPUT_KINDS,
    SolveSettings,
    build_error_weights,
)
from modax._params import Params, as_param_row
from modax._sensitivity import (
    RawOutput,
    SensitivitySpec,
    augmented_error_weights,
    augmented_y0,
)


def normalize_y0_params(
    y0: ArrayLike, params: ArrayLike
) -> tuple[jax.Array, jax.Array, int, int]:
    """Broadcast ``y0`` and ``params`` to a shared ``(N, ...)`` ensemble layout.

    Each takes 1-D (``(n_vars,)``, ``(n_params,)``) or 2-D (``(N, n_vars)``,
    ``(N, n_params)``) form. A 1-D input repeats across the other's ``N``.

    Returns:
        ``(y0, params, N, n_vars)``, both arrays 2-D float64.

    Raises:
        ValueError: If either input is not 1-D or 2-D, or both are 2-D with
            different leading axes.
    """
    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params, dtype=jnp.float64)

    if y0_arr.ndim not in (1, 2) or params_arr.ndim not in (1, 2):
        raise ValueError(
            "y0 must have shape (n_vars,) or (N, n_vars) and params shape "
            f"(n_params,) or (N, n_params); got y0.shape={y0_arr.shape} and "
            f"params.shape={params_arr.shape}"
        )
    if y0_arr.ndim == 2:
        n = y0_arr.shape[0]
        if params_arr.ndim == 2 and params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_arr.shape} and "
                f"params.shape={params_arr.shape}"
            )
    elif params_arr.ndim == 2:
        n = params_arr.shape[0]
    else:
        n = 1

    if y0_arr.ndim == 1:
        y0_arr = jnp.broadcast_to(y0_arr, (n, y0_arr.shape[0]))
    if params_arr.ndim == 1:
        params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
    return y0_arr, params_arr, n, y0_arr.shape[1]


@dataclass(frozen=True)
class EnsembleInputs:
    """The array inputs of one launch, in kernel order, and their extents.

    ``n_vars`` is the state dimension and ``n_system`` the number of
    components the kernel integrates. They match unless a sensitivity spec
    augments the state, and then ``y0`` and ``weights`` already hold the
    augmented arrays.
    """

    y0: jax.Array
    times: jax.Array
    params: jax.Array
    weights: jax.Array
    n_vars: int
    n_system: int
    time_rows: int = 1  # 1 for a shared save grid, N for one per trajectory

    @property
    def n(self) -> int:
        return self.y0.shape[0]

    @property
    def n_params(self) -> int:
        return self.params.shape[1]

    @property
    def n_save(self) -> int:
        return self.times.shape[0] // self.time_rows

    @property
    def arrays(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return self.y0, self.times, self.params, self.weights


def prepare_inputs(
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    settings: SolveSettings,
    spec: SensitivitySpec | None,
) -> EnsembleInputs:
    """Normalise a solve's arguments into the arrays the kernel reads.

    A shared ``t_span`` stays as it is. An ``(N, n_save)`` one, one grid per
    trajectory, flattens row by row, and also sets ``N`` when ``y0`` and
    ``params`` are both 1-D. With a ``spec`` the kernel integrates the joint ``[y, S]`` system, so this
    widens the initial state and error weights to the augmented extent.
    """
    params = as_param_row(params)
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    if times.ndim not in (1, 2):
        raise ValueError(
            f"t_span must be (n_save,) or (N, n_save), got shape {times.shape}"
        )
    time_rows = 1
    if times.ndim == 2 and times.shape[0] == 1:
        times = times[0]
    elif times.ndim == 2:
        if n == 1 and jnp.ndim(y0) == 1 and jnp.ndim(params) == 1:
            n = times.shape[0]
            y0_arr = jnp.broadcast_to(y0_arr, (n, y0_arr.shape[1]))
            params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[1]))
        elif times.shape[0] != n:
            raise ValueError(
                f"t_span has {times.shape[0]} rows, the ensemble {n} trajectories"
            )
        time_rows = n
        times = times.reshape(-1)
    weights = build_error_weights(settings.error_weights, n, n_vars)
    if spec is not None:
        y0_arr = augmented_y0(y0_arr, spec)
        weights = augmented_error_weights(weights, spec)
    return EnsembleInputs(
        y0=y0_arr,
        times=times,
        params=params_arr,
        weights=jnp.asarray(weights),
        n_vars=n_vars,
        n_system=n_vars if spec is None else spec.n_aug,
        time_rows=time_rows,
    )


def ensemble_ffi_call(
    launch: CudaLaunch,
    inputs: EnsembleInputs,
    settings: SolveSettings,
    *,
    n_save_hist: int,
    hook_size: int,
) -> tuple[jax.Array, ...]:
    """Launch a solver kernel from JAX and return its outputs.

    Args:
        n_save_hist: The history's time extent: the number of save times, or
            1 for a kernel that keeps only the final state.
        hook_size: The width of the save hook's per-trajectory output rows,
            at least 1.

    Returns:
        ``(hist, accepted, rejected, loop, hook_out)``: ``hist`` is ``(N,
        n_save_hist, n_system)``, the counters ``(N,)`` and ``hook_out``
        ``(N, hook_size)``.
    """
    n = inputs.n
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    output_specs = (
        jax.ShapeDtypeStruct((n, n_save_hist, inputs.n_system), jnp.float64),
        int_spec,
        int_spec,
        int_spec,
        jax.ShapeDtypeStruct((n, hook_size), jnp.float64),
    )
    return ffi_abi_call(
        launch,
        inputs.arrays,
        output_specs,
        input_kinds=SOLVER_INPUT_KINDS,
        scalar_f64_values=(settings.dt0, settings.rtol, settings.atol),
        scalar_i32_values=(settings.max_steps, inputs.time_rows),
    )


def _broadcast_for_vmap(
    arg: ArrayLike, is_batched: bool, axis_size: int, name: str
) -> jax.Array:
    arr = jnp.asarray(arg)
    if is_batched:
        if arr.ndim != 2:
            raise NotImplementedError(
                f"vmap over an already-ensembled {name} is not supported; "
                "call the solver with batched y0/params directly instead."
            )
        if arr.shape[0] != axis_size:
            raise ValueError(
                f"batched {name} has leading axis {arr.shape[0]}, expected {axis_size}"
            )
        return arr
    if arr.ndim != 1:
        raise NotImplementedError(
            f"vmap with unbatched ensemble-shaped {name} is not supported; "
            "call the solver with batched y0/params directly instead."
        )
    return jnp.broadcast_to(arr, (axis_size,) + arr.shape)


def make_custom_vmap_solver(
    solve_impl: Callable[[jax.Array, jax.Array, jax.Array], RawOutput],
) -> custom_vmap:
    """Wrap a solver so an outer ``jax.vmap`` becomes one ensemble launch.

    ``solve_impl(y0, t_span, params)`` returns a :data:`RawOutput`, every
    array of it with a leading trajectory axis. The batching rule handles a
    ``vmap`` of single-trajectory solves over ``y0``, ``params`` or both, and
    runs it as one ensemble solve. A batched ``t_span`` gives each trajectory
    its own save times.

    Raises:
        NotImplementedError: If the mapped ``y0``, ``params`` or ``t_span`` is
            already an ensemble.
    """

    @custom_vmap
    def _solve(y0: ArrayLike, t_span: ArrayLike, params: ArrayLike) -> RawOutput:
        return solve_impl(jnp.asarray(y0), jnp.asarray(t_span), jnp.asarray(params))

    @_solve.def_vmap
    def _solve_vmap(
        axis_size: int,
        in_batched: tuple[bool, bool, bool],
        y0: ArrayLike,
        t_span: ArrayLike,
        params: ArrayLike,
    ) -> tuple[RawOutput, object]:
        y0_batched, t_span_batched, params_batched = in_batched
        if t_span_batched:
            # one grid per trajectory, (axis_size, n_save)
            t_span = _broadcast_for_vmap(t_span, True, axis_size, "t_span")

        y0_arr = _broadcast_for_vmap(y0, y0_batched, axis_size, "y0")
        params_arr = _broadcast_for_vmap(params, params_batched, axis_size, "params")
        result = solve_impl(y0_arr, jnp.asarray(t_span), params_arr)
        # Each mapped solve is an ensemble of one, so its outputs keep a
        # trajectory axis of length 1 behind the vmap axis.
        out = jax.tree_util.tree_map(lambda x: x[:, None], result)
        return out, jax.tree_util.tree_map(lambda _: True, out)

    return _solve
