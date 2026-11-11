"""The JAX-facing glue: ensemble inputs, the FFI launch and the ``vmap`` rule."""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.typing import ArrayLike

from modax._jax_numba_custom_call import CudaLaunch, ffi_abi_call
from modax._numba_common import (
    SOLVER_INPUT_KINDS,
    SolveOutput,
    SolveSettings,
    build_error_weights,
)
from modax._sensitivity import (
    SensitivitySpec,
    augmented_error_weights,
    augmented_y0,
    make_sensitivity_solver,
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

    @property
    def n(self) -> int:
        return self.y0.shape[0]

    @property
    def n_params(self) -> int:
        return self.params.shape[1]

    @property
    def n_save(self) -> int:
        return self.times.shape[0]

    @property
    def arrays(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return self.y0, self.times, self.params, self.weights


def prepare_inputs(
    y0: ArrayLike,
    t_span: ArrayLike,
    params: ArrayLike,
    settings: SolveSettings,
    spec: SensitivitySpec | None,
) -> EnsembleInputs:
    """Normalise a solve's arguments into the arrays the kernel reads.

    With a ``spec`` the kernel integrates the joint ``[y, S]`` system, so this
    widens the initial state and error weights to the augmented extent.
    """
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    weights = build_error_weights(settings.error_weights, n, n_vars)
    if spec is not None:
        y0_arr = augmented_y0(y0_arr, spec)
        weights = augmented_error_weights(weights, spec)
    return EnsembleInputs(
        y0=y0_arr,
        times=jnp.asarray(t_span, dtype=jnp.float64),
        params=params_arr,
        weights=jnp.asarray(weights),
        n_vars=n_vars,
        n_system=n_vars if spec is None else spec.n_aug,
    )


def ensemble_ffi_call(
    launch: CudaLaunch,
    inputs: EnsembleInputs,
    settings: SolveSettings,
    *,
    n_save_hist: int | None = None,
    hook_size: int | None = None,
) -> tuple[jax.Array, ...]:
    """Launch a solver kernel from JAX and return its outputs.

    Args:
        n_save_hist: The history's time extent when it differs from the number
            of save times. A kernel that keeps only the final state passes 1.
        hook_size: Width of the save hook's per-trajectory accumulator rows,
            which the kernel takes as one more output after the counters.

    Returns:
        ``(hist, accepted, rejected, loop)``, plus ``hook_out`` when
        you give a ``hook_size``. ``hist`` is ``(N, n_hist, n_system)``, the
        counters ``(N,)`` and ``hook_out`` ``(N, hook_size)``.
    """
    n = inputs.n
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    n_hist = inputs.n_save if n_save_hist is None else n_save_hist
    output_specs = (
        jax.ShapeDtypeStruct((n, n_hist, inputs.n_system), jnp.float64),
        int_spec,
        int_spec,
        int_spec,
    )
    if hook_size is not None:
        output_specs += (jax.ShapeDtypeStruct((n, hook_size), jnp.float64),)
    return ffi_abi_call(
        launch,
        inputs.arrays,
        output_specs,
        input_kinds=SOLVER_INPUT_KINDS,
        scalar_f64_values=(settings.dt0, settings.rtol, settings.atol),
        scalar_i32_values=(settings.max_steps,),
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
    solve_impl: Callable[[jax.Array, jax.Array, jax.Array], SolveOutput],
) -> custom_vmap:
    """Wrap a solver so an outer ``jax.vmap`` becomes one ensemble launch.

    ``solve_impl(y0, t_span, params)`` returns the public solver result. The
    batching rule handles a ``vmap`` of single-trajectory solves over ``y0``,
    ``params`` or both, and runs it as one ensemble solve with a leading
    trajectory axis. A batched ``t_span`` counts as one shared time grid.

    Raises:
        NotImplementedError: If the mapped ``y0`` or ``params`` is already an
            ensemble, or a batched ``t_span`` is not 2-D.
    """

    @custom_vmap
    def _solve(y0: ArrayLike, t_span: ArrayLike, params: ArrayLike) -> SolveOutput:
        return solve_impl(jnp.asarray(y0), jnp.asarray(t_span), jnp.asarray(params))

    @_solve.def_vmap
    def _solve_vmap(
        axis_size: int,
        in_batched: tuple[bool, bool, bool],
        y0: ArrayLike,
        t_span: ArrayLike,
        params: ArrayLike,
    ) -> tuple[SolveOutput, object]:
        y0_batched, t_span_batched, params_batched = in_batched
        if t_span_batched:
            t_span_arr = jnp.asarray(t_span)
            if t_span_arr.ndim != 2:
                raise NotImplementedError(
                    "vmap over nested t_span values is not supported; use a shared "
                    "t_span and vmap over y0 and/or params, or call the solver directly."
                )
            # JAX can mark closed-over constant save times as batched inside
            # a larger vmapped function.  Treat that as a shared time grid.
            t_span = t_span_arr[0]

        y0_arr = _broadcast_for_vmap(y0, y0_batched, axis_size, "y0")
        params_arr = _broadcast_for_vmap(params, params_batched, axis_size, "params")
        result = solve_impl(y0_arr, jnp.asarray(t_span), params_arr)

        if not isinstance(result, tuple):
            return result[:, None, :, :], True

        sol, stats = result
        stats_out = jax.tree_util.tree_map(lambda x: x[:, None], stats)
        stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
        return (sol[:, None, :, :], stats_out), (True, stats_batched)

    return _solve


def differentiable_solve(
    impl: Callable[..., SolveOutput],
    options: Any,
    settings: SolveSettings,
    return_stats: bool,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: ArrayLike,
    *,
    sens_error_control: bool,
    sens_param_columns: Sequence[int] | None,
) -> SolveOutput:
    """Solve through the ``vmap`` and forward-sensitivity rules.

    ``impl(y0, t_span, params, *, options, settings, return_stats)`` is the
    plain launch, and ``options`` a dataclass with a ``spec`` field.
    """

    # The JVP rule wraps the vmap-aware solvers rather than the other way
    # round: custom_vmap's own JVP path instantiates symbolic zeros, which is
    # what tells the rule which sensitivity blocks it has to integrate.
    def solver_for(spec: SensitivitySpec | None) -> custom_vmap:
        return make_custom_vmap_solver(
            functools.partial(
                impl,
                options=replace(options, spec=spec),
                settings=settings,
                return_stats=return_stats,
            )
        )

    return make_sensitivity_solver(
        solver_for,
        jnp.shape(y0)[-1],
        jnp.shape(params)[-1],
        sens_error_control=sens_error_control,
        sens_param_columns=sens_param_columns,
    )(y0, t_span, params)
