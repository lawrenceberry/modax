"""The JAX-facing glue: ensemble inputs, the FFI launch and the ``vmap`` rule."""

from __future__ import annotations

import functools
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
    DirectionalSolver,
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
    ``params`` are both 1-D. With a ``spec`` the kernel integrates the joint
    ``[y, s_0, ...]`` system: ``y0`` arrives joint already, and this widens
    the error weights to match.
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
    if spec is not None:
        n_vars = spec.n_vars  # y0 is the joint [y, dy0_0, ...]
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


def _ensemble_rows(
    arr: jax.Array, is_batched: bool, axis_size: int, n: int, name: str
) -> jax.Array:
    """Flatten a mapped argument to ``(axis_size * n, k)``, row ``b * n + i``.

    Unmapped it is ``(k,)`` or ``(n, k)``, and mapped it has ``axis_size``
    in front of that.
    """
    base = arr.ndim - int(is_batched)
    if base not in (1, 2) or (base == 2 and arr.shape[-2] != n):
        raise ValueError(
            f"{name} must be (k,) or ({n}, k) under each vmap; got shape {arr.shape}"
        )
    k = arr.shape[-1]
    if is_batched and base == 2:
        return arr.reshape(axis_size * n, k)
    if is_batched:
        return jnp.repeat(arr, n, axis=0)
    if base == 2:
        return jnp.tile(arr, (axis_size, 1))
    return jnp.broadcast_to(arr, (axis_size * n, k))


def make_directional_solver(
    run_for: Callable[..., Callable[[jax.Array, jax.Array, jax.Array], RawOutput]],
    n_vars: int,
    n_columns: int | None = None,
    *,
    max_directions_per_thread: int | None = None,
) -> DirectionalSolver:
    """Wrap the directional launches so ``vmap`` over directions adds columns.

    ``run_for(n_dirs, copies)(y0, t_span, params)`` launches the joint system
    of a :class:`~modax._sensitivity.SensitivitySpec`: ``y0`` holds ``[y,
    dy0_0, ...]`` and ``params`` the row ``(p, dp_0, ...)``, for an
    ensemble whose trajectories ``copies`` repeats. The returned solver
    takes ``(y0, t_span, params, dy0, dparams)`` with the tangents'
    directions on the axis before last, and returns the :data:`RawOutput`
    and the tangents of the history and the quantities, ``(N, n_save,
    n_dirs, n)``.

    ``jax.jacfwd`` maps the solver over directions with the primals fixed.
    The batching rule folds the mapped axis into the direction axis and
    launches once: each trajectory stays one thread, which integrates its
    state and a column per direction. The primal output is that launch's
    and stays unmapped, as ``jacfwd`` requires. Mapped primals are other
    trajectories, and they flatten into the ensemble.

    With ``max_directions_per_thread`` a thread integrates at most that many
    columns. The directions then split into groups of one width, the last
    padded with zero directions, and each group runs in a thread of its own
    beside a copy of its trajectory, all in one launch.

    ``n_columns`` is how many inputs of one trajectory move. More directions
    than that cannot all be independent: ``jax.jacfwd`` over the
    per-trajectory parameters of an ensemble sends every trajectory a
    direction per trajectory, nearly all of them zero for it, and the rule
    raises instead of integrating them.
    """

    @functools.cache
    def solver(n_dirs: int, copies: int = 1):
        run = run_for(n_dirs, copies)

        def impl(y0, t_span, params, dy0, dparams):
            lead = jnp.broadcast_shapes(y0.shape[:-1], dy0.shape[:-2])
            start = jnp.broadcast_to(dy0, lead + dy0.shape[-2:])
            rows = jnp.broadcast_shapes(params.shape[:-1], dparams.shape[:-2])
            moved = jnp.broadcast_to(dparams, rows + dparams.shape[-2:])
            hist, quantities, stats = run(
                jnp.concatenate(
                    [
                        jnp.broadcast_to(y0, lead + y0.shape[-1:]),
                        start.reshape(lead + (-1,)),
                    ],
                    axis=-1,
                ),
                t_span,
                jnp.concatenate(
                    [
                        jnp.broadcast_to(params, rows + params.shape[-1:]),
                        moved.reshape(rows + (-1,)),
                    ],
                    axis=-1,
                ),
            )
            primal = (
                hist[..., :n_vars],
                None if quantities is None else quantities[:, :, :1],
                stats,
            )
            tangent = (
                hist[..., n_vars:].reshape(hist.shape[:-1] + (n_dirs, n_vars)),
                None if quantities is None else quantities[:, :, 1:],
            )
            return primal, tangent

        @custom_vmap
        def _solve(y0, t_span, params, dy0, dparams):
            return impl(
                jnp.asarray(y0, dtype=jnp.float64),
                jnp.asarray(t_span),
                jnp.asarray(params, dtype=jnp.float64),
                jnp.asarray(dy0, dtype=jnp.float64),
                jnp.asarray(dparams, dtype=jnp.float64),
            )

        @_solve.def_vmap
        def _solve_vmap(axis_size, in_batched, y0, t_span, params, dy0, dparams):
            y0, t_span, params, dy0, dparams = (
                jnp.asarray(a) for a in (y0, t_span, params, dy0, dparams)
            )
            if not any(in_batched[:3]):
                return _more_directions(
                    axis_size, in_batched, y0, t_span, params, dy0, dparams
                )
            # Other trajectories: flatten the mapped axis into the ensemble,
            # the tangents with their directions folded into the last axis.
            tangents = [t.reshape(t.shape[:-2] + (-1,)) for t in (dy0, dparams)]
            arrays = [y0, t_span, params, *tangents]
            bases = [a.ndim - int(b) for a, b in zip(arrays, in_batched)]
            sizes = {a.shape[-2] for a, base in zip(arrays, bases) if base == 2}
            if len(sizes) > 1:
                raise ValueError(f"the mapped solve's arguments disagree on N: {sizes}")
            n = sizes.pop() if sizes else 1
            names = ("y0", "t_span", "params", "the y0 tangent", "the params tangent")
            flat = [
                _ensemble_rows(a, b, axis_size, n, name)
                for a, b, name in zip(arrays, in_batched, names)
            ]
            if not in_batched[1] and bases[1] == 1:
                flat[1] = t_span  # one save grid for every trajectory
            flat[3] = flat[3].reshape(-1, n_dirs, dy0.shape[-1])
            flat[4] = flat[4].reshape(-1, n_dirs, dparams.shape[-1])
            # Through the wrapped solver, so an outer vmap maps this launch too.
            out = _solve(*flat)
            out = jax.tree_util.tree_map(
                lambda x: x.reshape((axis_size, n) + x.shape[1:]), out
            )
            return out, jax.tree_util.tree_map(lambda _: True, out)

        def _more_directions(axis_size, in_batched, y0, t_span, params, dy0, dparams):
            """The mapped axis as ``axis_size`` times the directions of the same trajectories."""
            total = axis_size * n_dirs

            def fold(tangent, is_batched):
                # (axis, ..., n_dirs, k) or (..., n_dirs, k) to (..., total, k)
                if not is_batched:
                    tangent = jnp.broadcast_to(tangent, (axis_size,) + tangent.shape)
                tangent = jnp.moveaxis(tangent, 0, -3)
                return tangent.reshape(tangent.shape[:-3] + (total, tangent.shape[-1]))

            dy0, dparams = fold(dy0, in_batched[3]), fold(dparams, in_batched[4])
            if n_columns is not None and total > n_columns:
                raise ValueError(
                    f"this derivative asks a modax solve for {total} directions, "
                    f"but only {n_columns} of each trajectory's inputs move, so "
                    "every trajectory would integrate a column per direction, "
                    "most of them zero. jax.jacfwd over an ensemble's "
                    "per-trajectory parameters does this: map over the "
                    "trajectories outside the derivative instead, "
                    "jax.vmap(jax.jacfwd(f)), so each jacfwd sees one "
                    "trajectory's inputs"
                )
            primal, tangent = _along(y0, t_span, params, dy0, dparams)
            # (N, n_save, total, k) back to (axis, N, n_save, n_dirs, k)
            tangent = jax.tree_util.tree_map(
                lambda x: jnp.moveaxis(
                    x.reshape(x.shape[:2] + (axis_size, n_dirs) + x.shape[3:]), 2, 0
                ),
                tangent,
            )
            return (primal, tangent), (
                jax.tree_util.tree_map(lambda _: False, primal),
                jax.tree_util.tree_map(lambda _: True, tangent),
            )

        return _solve

    def _along(y0, t_span, params, dy0, dparams):
        """The tangents along ``total`` directions, ``(..., total, k)``, in groups of at most ``max_directions_per_thread``."""
        total = dy0.shape[-2]
        groups = 1
        if max_directions_per_thread is not None:
            groups = -(-total // max_directions_per_thread)
        if groups == 1:
            return solver(total)(y0, t_span, params, dy0, dparams)
        width = -(-total // groups)
        spare = groups * width - total
        if spare:
            dy0, dparams = (
                jnp.concatenate(
                    [d, jnp.zeros(d.shape[:-2] + (spare, d.shape[-1]), d.dtype)],
                    axis=-2,
                )
                for d in (dy0, dparams)
            )
        return _replicated(y0, t_span, params, dy0, dparams, groups, width, total)

    def _replicated(y0, t_span, params, dy0, dparams, groups, width, total):
        """A thread per trajectory and group: copy ``g`` of trajectory ``i`` is row ``i * groups + g``."""
        n = max(
            [a.shape[0] for a in (y0, params) if a.ndim == 2]
            + [d.shape[0] for d in (dy0, dparams) if d.ndim == 3]
            + [t_span.shape[0] if t_span.ndim == 2 else 1]
        )

        def copied(a):
            return jnp.repeat(jnp.broadcast_to(a, (n, a.shape[-1])), groups, axis=0)

        def grouped(d):
            # (..., groups * width, k) to (n * groups, width, k)
            return jnp.broadcast_to(d, (n,) + d.shape[-2:]).reshape(
                n * groups, width, d.shape[-1]
            )

        times = copied(t_span) if t_span.ndim == 2 and t_span.shape[0] > 1 else t_span
        primal, tangent = solver(width, groups)(
            copied(y0), times, copied(params), grouped(dy0), grouped(dparams)
        )
        # Every copy integrates the primal; the first of each trajectory's
        # is its value.
        primal = jax.tree_util.tree_map(lambda x: x[::groups], primal)
        # (n * groups, n_save, width, k) back to (n, n_save, total, k)
        tangent = jax.tree_util.tree_map(
            lambda x: jnp.moveaxis(x.reshape((n, groups) + x.shape[1:]), 1, 2).reshape(
                (n, x.shape[1], groups * width) + x.shape[3:]
            )[:, :, :total],
            tangent,
        )
        return primal, tangent

    return solver(1)
