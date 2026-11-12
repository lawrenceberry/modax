"""Continuous forward sensitivity analysis for the numba-cuda ensemble solvers.

The solve reaches JAX through an XLA FFI custom call, which autodiff cannot
see into (see "Why continuous forward sensitivities" in
docs/usage/gradients.md). A ``jax.custom_jvp`` rule supplies the derivatives
instead, integrating the sensitivity system alongside the state.

With ``S(t) = dy(t)/dtheta`` for a direction block ``theta``, differentiating
``y' = f(t, y, p)`` gives the variational equation

    S' = J_y(t) S + J_p(t),    J_y = df/dy,  J_p = df/dtheta,

with ``J_p = 0`` and ``S(t0) = I`` for the initial-state block, and ``J_p =
df/dp`` and ``S(t0) = 0`` for the parameter block. Stacked under the state it
makes one joint system

    d/dt [y, S] = [f(t, y, p), J_y S + J_p]

which the kernels integrate as an ordinary ODE of ``n_aug = n_vars *
(1 + n_sens)`` variables. The rule returns the primal and the tangent
together, so ``jax.value_and_grad`` costs one joint solve, not two solves.

Layout
------
The augmented state is the state followed by one ``n_vars``-long block per
sensitivity direction::

    z[j]                        = y[j]
    z[n_vars + k * n_vars + r]  = S[r, k]

Direction ``k`` runs over the initial-state block first (``n_vars``
directions, present only when JAX differentiates ``y0``), then the parameter
block (``n_params`` directions, present only when JAX differentiates
``params``). The solve integrates only the blocks JAX asks for.

``J_y S_k + J_p_k`` comes from ``numba_enzyme.jvp`` of the user's ``ode_fn``,
seeded over the callback's *flattened* argument list ``(y_0 ... y_n-1, t,
p_0 ... p_m-1)``. Seeding ``(S_k, 0, e_k)`` returns that whole right-hand side
in one sweep, and seeding the time argument returns ``df/dt``. The caller
supplies nothing extra and no Jacobian is ever formed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher
from numba_cuda_mlir.numba_cuda.typing.templates import Signature
from numba_enzyme import jvp

from modax._jax_numba_custom_call import compile_target
from modax._numba_common import OdeFn, SolverStats, as_cuda_device
from modax._params import combine_params

Tangent = Callable[..., None]
"""An Enzyme forward sweep: ``(out, *args, *directions)``, writing ``out``."""


@dataclass(frozen=True)
class SensitivitySpec:
    """Which sensitivity blocks a joint solve carries.

    Hashable and part of the kernel cache keys, so a solve differentiated only
    with respect to ``params`` compiles and integrates a smaller system than one
    differentiated with respect to both. ``error_control`` lives here, not on the
    solve, because the error-norm denominator it picks is a compile-time constant
    of the kernel. ``param_columns`` restricts the parameter block to those
    columns.

    Raises:
        ValueError: The spec asks for neither block, or ``param_columns`` repeats a
            column or names one outside the ``n_params`` columns.
    """

    n_vars: int
    n_params: int
    wrt_y0: bool
    wrt_params: bool
    error_control: bool = True
    param_columns: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not (self.wrt_y0 or self.wrt_params):
            raise ValueError(
                "a sensitivity spec must carry at least one direction block"
            )
        if self.param_columns is not None:
            if len(set(self.param_columns)) != len(self.param_columns):
                raise ValueError("param_columns must not repeat a column")
            bad = [c for c in self.param_columns if not 0 <= c < self.n_params]
            if bad:
                raise ValueError(
                    f"param_columns {bad} lie outside the {self.n_params} "
                    "parameter columns"
                )

    def check_extents(self, n_vars: int, n_params: int, against: str) -> None:
        """Raise ValueError unless the spec is for ``n_vars`` states and ``n_params`` parameters."""
        if self.n_vars != n_vars or self.n_params != n_params:
            raise ValueError(
                f"the sensitivity spec is for {self.n_vars} variables and "
                f"{self.n_params} parameters, {against} {n_vars} and {n_params}"
            )

    @property
    def n_y0_dirs(self) -> int:
        return self.n_vars if self.wrt_y0 else 0

    @property
    def n_param_dirs(self) -> int:
        if not self.wrt_params:
            return 0
        return self.n_params if self.param_columns is None else len(self.param_columns)

    @property
    def param_seed_columns(self) -> tuple[int, ...]:
        """Which parameter each direction seeds, in direction order."""
        if self.param_columns is None:
            return tuple(range(self.n_params))
        return self.param_columns

    @property
    def n_sens(self) -> int:
        return self.n_y0_dirs + self.n_param_dirs

    @property
    def n_aug(self) -> int:
        """Size of the joint ``[y, S]`` system the kernel integrates."""
        return self.n_vars * (1 + self.n_sens)

    @property
    def n_error(self) -> int:
        """Component count the weighted-RMS error norm divides by.

        With the sensitivities out of step-size control their weights are zero, so
        dividing by ``n_vars`` rather than ``n_aug`` reproduces the plain solve's
        error norm, and its step sequence, bit for bit.
        """
        return self.n_aug if self.error_control else self.n_vars


@functools.cache
def _quantity_signature(n_outputs: int, n_vars: int, n_params: int) -> Signature:
    """The specialisation :func:`quantity_hook` differentiates its quantities in."""
    return types.UniTuple(types.float64, n_outputs)(
        types.UniTuple(types.float64, n_vars),
        types.float64,
        types.UniTuple(types.float64, n_params),
    )


@functools.cache
def quantity_count(quantities: Callable[..., Any], n_vars: int, n_params: int) -> int:
    """How many values ``quantities(y, t, p)`` returns, from its inferred type.

    numba types the function for the tuple arguments the kernel's own state
    and parameter row stand for, which is all it takes to read the length of
    the tuple it returns. Nothing is launched and no derivative is built.

    Raises:
        TypeError: If the function does not return a tuple of numbers.
    """
    args = (
        types.UniTuple(types.float64, n_vars),
        types.float64,
        types.UniTuple(types.float64, n_params),
    )
    returned = as_cuda_device(quantities).compile_device(args).signature.return_type
    if not (
        isinstance(returned, types.BaseTuple)
        and len(returned) > 0
        and all(isinstance(t, types.Number) for t in returned)
    ):
        raise TypeError(
            "a save_hook returns a non-empty tuple of numbers, one per quantity "
            f"(write a single one as (q,)); this one returns {returned}"
        )
    return len(returned)


@functools.cache
def make_tangent(ode_fn: OdeFn, n_vars: int, n_params: int) -> Tangent:
    """``D f(y, t, p)[u]`` in one forward sweep, for any direction.

    The sensitivity right-hand side ``J_y S_k + J_p_k`` is the sweep seeded with
    ``(S_k, 0, e_k)``. Building it from unit columns would cost ``n_vars + 1``
    sweeps.
    """
    return make_quantity_tangent(ode_fn, n_vars, n_vars, n_params)


@functools.cache
def make_second_tangent(ode_fn: OdeFn, n_vars: int, n_params: int) -> Tangent:
    """``D2 f(y, t, p)[u, v] + D f(y, t, p)[w]``, forward over forward.

    The joint system's off-diagonal Jacobian block holds second derivatives of
    ``ode_fn``. Seeding ``u = (S_k, 0, e_k)`` and ``v = (w, 0, 0)`` applies that
    block to ``w`` without forming it, and ``v = (0, 1, 0)`` gives the
    sensitivity block's ``dF/dt``. numba-enzyme composes ``jvp`` with itself, so
    this is the forward sweep of the forward sweep. The extra direction ``w`` is
    the inner direction's own variation, and every caller here passes zero for
    it, which leaves the plain bilinear form.
    """
    return make_quantity_second_tangent(ode_fn, n_vars, n_vars, n_params)


@functools.cache
def make_quantity_tangent(fn, n_outputs: int, n_vars: int, n_params: int) -> Tangent:
    """``D q(y, t, p)[u]`` for a quantity function of ``n_outputs`` values of ``n_vars`` states."""
    # For the kernel's own target (compile_target), so the two link on Blackwell.
    return jvp(
        as_cuda_device(fn),
        signature=_quantity_signature(n_outputs, n_vars, n_params),
        cc=compile_target(),
    )


@functools.cache
def make_quantity_second_tangent(
    fn, n_outputs: int, n_vars: int, n_params: int
) -> Tangent:
    """``D2 q[u, v] + D q[w]`` for a quantity function, forward over forward as :func:`make_second_tangent`."""
    signature = _quantity_signature(n_outputs, n_vars, n_params)
    cc = compile_target()
    return jvp(
        jvp(as_cuda_device(fn), signature=signature, cc=cc), signature=signature, cc=cc
    )


@functools.cache
def seed_table(spec: SensitivitySpec) -> npt.NDArray[np.float64]:
    """Read-only table every unit and zero direction is a window into.

    ``2 * L`` zeros with a single ``1.0`` at ``L``, so the window starting at
    ``L - k`` has its ``1.0`` at position ``k``, and any window inside ``[0, L)``
    is all zeros. A per-thread one-hot buffer could not promote to registers,
    because its store index is dynamic.
    """
    length = max(spec.n_vars, spec.n_params)
    table = np.zeros(2 * length, dtype=np.float64)
    table[length] = 1.0
    return table


@functools.cache
def make_augmented_local_writer(ode_fn: OdeFn, spec: SensitivitySpec) -> MLIRDispatcher:
    """Joint ``[f, J_y S + J_p]`` writer for a kernel's thread-local state.

    Each thread writes its own trajectory's whole augmented vector from and into
    its own local arrays, one forward sweep per direction seeded with ``(S_k, 0,
    e_k)``. Threads share nothing, so the device function needs no synchronisation
    under its callers' divergent ``if running``.
    """
    n_vars = spec.n_vars
    n_sens = spec.n_sens
    n_y0_dirs = spec.n_y0_dirs
    n_params = spec.n_params
    param_cols = spec.param_seed_columns
    length = max(n_vars, n_params)
    fn_device = as_cuda_device(ode_fn)
    tangent_of = make_tangent(ode_fn, n_vars, n_params)
    seeds = seed_table(spec)

    @cuda.jit(device=True)
    def write_vector(z_row, t, p_row, out):
        seed = cuda.const.array_like(seeds)
        values = fn_device(z_row, t, p_row)
        for j in range(n_vars):
            out[j] = values[j]

        tangent = cuda.local.array(n_vars, types.float64)
        for k in range(n_sens):
            base = n_vars + k * n_vars
            start = 0 if k < n_y0_dirs else length - param_cols[k - n_y0_dirs]
            tangent_of(
                tangent,
                z_row,
                t,
                p_row,
                z_row[base : base + n_vars],
                0.0,
                seed[start : start + n_params],
            )
            for r in range(n_vars):
                out[base + r] = tangent[r]

    return write_vector


@functools.cache
def quantity_hook(
    quantities: Callable[..., tuple[float, ...]],
    n_outputs: int,
    n_vars: int,
    n_params: int,
    sensitivity: SensitivitySpec | None = None,
) -> MLIRDispatcher:
    """A save hook storing ``quantities(y, t, p)`` at every save, and their tangents.

    The hook stores ``n_outputs * (1 + n_sens)`` numbers per save, save-major, so
    its ``hook_size`` is that times the number of saves. Without ``sensitivity``
    (``n_sens = 0``) it stores the values alone, for a plain hooked solve. With
    it, for a joint launch of ``spec``, it follows each
    save's values with their total derivative along every direction ``k`` of the
    spec, direction after direction::

        dq/d theta_k = (dq/dy) S_k + dq/dp e_k

    (no ``e_k`` term for an initial-state direction). One Enzyme forward sweep of
    ``quantities`` seeded with ``(S_k, 0, e_k)`` gives each. The derivative
    targets the kernel's own ``compile_target``, so the two link on every GPU,
    Blackwell included, and the caller needs no Enzyme of its own.

    Parameters
    ----------
    quantities : callable
        ``quantities(y, t, p) -> tuple`` of ``n_outputs`` floats, reading
        ``y[0:n_vars]`` only. A device function, or a plain function compiled as
        one.
    n_outputs : int
        Length of the tuple ``quantities`` returns.
    n_vars, n_params : int
        The problem's state and parameter-row sizes.
    sensitivity : SensitivitySpec, optional
        The spec of the joint launch the hook serves.

    Returns
    -------
    MLIRDispatcher
        ``hook(save_idx, y, t, p_row, acc)``, the ``save_hook`` of ``solve``.

    Raises
    ------
    ValueError
        ``n_outputs`` is below 1, or ``sensitivity`` is for other extents.
    """
    if n_outputs < 1:
        raise ValueError("a quantity hook stores at least one quantity")
    if sensitivity is not None:
        sensitivity.check_extents(n_vars, n_params, "the quantities for")
    fn_device = as_cuda_device(quantities)

    if sensitivity is None:

        @cuda.jit(device=True)
        def store_values(save_idx, y, t, p_row, acc):
            values = fn_device(y, t, p_row)
            base = n_outputs * save_idx
            for j in range(n_outputs):
                acc[base + j] = values[j]

        return store_values

    n_sens = sensitivity.n_sens
    n_y0_dirs = sensitivity.n_y0_dirs
    param_cols = sensitivity.param_seed_columns
    length = max(n_vars, n_params)
    width = n_outputs * (1 + n_sens)
    seeds = seed_table(sensitivity)
    tangent_of = make_quantity_tangent(quantities, n_outputs, n_vars, n_params)

    @cuda.jit(device=True)
    def store_values_and_tangents(save_idx, y, t, p_row, acc):
        seed = cuda.const.array_like(seeds)
        values = fn_device(y, t, p_row)
        base = width * save_idx
        for j in range(n_outputs):
            acc[base + j] = values[j]

        tangent = cuda.local.array(n_outputs, types.float64)
        for k in range(n_sens):
            column = n_vars + k * n_vars
            start = 0 if k < n_y0_dirs else length - param_cols[k - n_y0_dirs]
            tangent_of(
                tangent,
                y[0:n_vars],
                t,
                p_row,
                y[column : column + n_vars],
                0.0,
                seed[start : start + n_params],
            )
            out = base + n_outputs * (1 + k)
            for j in range(n_outputs):
                acc[out + j] = tangent[j]

    return store_values_and_tangents


def clear_caches() -> None:
    """Drop the cached device writers and Enzyme derivatives.

    The module-level caches otherwise hold every ``(ode_fn, spec)`` pair's
    writer for the life of the process.
    """
    make_tangent.cache_clear()
    make_second_tangent.cache_clear()
    make_quantity_tangent.cache_clear()
    make_quantity_second_tangent.cache_clear()
    seed_table.cache_clear()
    make_augmented_local_writer.cache_clear()
    quantity_hook.cache_clear()
    quantity_count.cache_clear()


def augmented_y0(y0_arr: jax.Array, spec: SensitivitySpec) -> jax.Array:
    """``(n, n_vars)`` initial state -> ``(n, n_aug)`` joint initial state.

    ``S(t0)`` is the identity for the initial-state block and zero for the
    parameter block.
    """
    n = y0_arr.shape[0]
    blocks = [y0_arr]
    if spec.wrt_y0:
        eye = jnp.eye(spec.n_vars, dtype=y0_arr.dtype).reshape(1, -1)
        blocks.append(jnp.broadcast_to(eye, (n, spec.n_vars * spec.n_vars)))
    if spec.wrt_params:
        blocks.append(jnp.zeros((n, spec.n_vars * spec.n_param_dirs), y0_arr.dtype))
    return jnp.concatenate(blocks, axis=1)


def augmented_error_weights(
    weights_arr: npt.NDArray[np.float64], spec: SensitivitySpec
) -> npt.NDArray[np.float64]:
    """``(n, n_vars)`` error weights -> ``(n, n_aug)``.

    A zero weight drops a component from the error norm, so
    ``spec.error_control=False`` makes the joint solve take the plain solve's
    steps exactly, and ``jax.value_and_grad`` returns the plain call's value bit
    for bit. It is not the default: nothing then ties the sensitivities'
    accuracy to ``rtol``, and a stiff solver taking large steps on an easy state
    can get them badly wrong.
    """
    tail = np.ones if spec.error_control else np.zeros
    return np.concatenate(
        [
            weights_arr,
            tail((weights_arr.shape[0], spec.n_aug - spec.n_vars), dtype=np.float64),
        ],
        axis=1,
    )


def split_augmented(
    hist: jax.Array, spec: SensitivitySpec
) -> tuple[jax.Array, jax.Array]:
    """``(n, n_save, n_aug)`` -> state ``(n, n_save, n_vars)`` and ``S`` ``(n, n_save, n_vars, n_sens)``."""
    n, n_save, _ = hist.shape
    state = hist[:, :, : spec.n_vars]
    sens = hist[:, :, spec.n_vars :].reshape(n, n_save, spec.n_sens, spec.n_vars)
    return state, jnp.swapaxes(sens, 2, 3)


def _match_shape(tangent: ArrayLike, axis_size: int, name: str) -> jax.Array:
    """Broadcast an input tangent the way the primal argument was broadcast.

    ``normalize_y0_params`` broadcasts a 1-D argument across the ensemble, and
    the tangent must match for the contraction to be the true directional
    derivative. ``jnp.broadcast_to`` is linear, so reverse mode transposes it to
    the sum over trajectories a shared argument should get. Raises ValueError
    for a tangent that is neither 1-D nor 2-D.
    """
    arr = jnp.asarray(tangent)
    if arr.ndim == 1:
        return jnp.broadcast_to(arr, (axis_size,) + arr.shape)
    if arr.ndim != 2:
        raise ValueError(f"{name} tangent must be 1-D or 2-D; got shape {arr.shape}")
    return arr


def _zero_tangent(x: ArrayLike) -> jax.Array | npt.NDArray[np.void]:
    """A zero tangent of the type JAX expects for ``x``.

    Integer outputs (the step counters) take the empty ``float0`` type, not a
    zero array of their own dtype.
    """
    if jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(jnp.shape(x), dtype=jax.dtypes.float0)


RawOutput = tuple[jax.Array, "jax.Array | None", "SolverStats | None"]
"""What the solvers under the JVP and vmap rules return.

``(hist, quantities, stats)``. ``hist`` is ``(N, n_save, n_system)``.
``quantities`` is a save hook's output, ``(N, n_save, 1 + n_sens, n_out)``:
each save's values and then their tangent along every direction of the
launch's spec (``n_sens = 0`` without one), or ``None`` without a hook.
``stats`` is the step counters, or ``None`` when not asked for.
"""


def make_sensitivity_solver(
    solver_for: Callable[[SensitivitySpec | None], Callable[..., RawOutput]],
    n_vars: int,
    leaf_widths: Sequence[int],
    *,
    sens_error_control: bool,
) -> Callable[..., RawOutput]:
    """Attach a forward-sensitivity JVP rule to one solver implementation.

    ``solver_for(None)`` is the plain solve, ``(y0, t_span, params)``, and
    ``solver_for(spec)`` the solver for the joint ``[y, S]`` system of ``spec``,
    with ``n_aug`` components in place of ``n_vars``. Both return a
    :data:`RawOutput` and must already handle ``vmap``. The rule sits
    *outside* ``custom_vmap`` because that JVP path traces to a jaxpr and
    instantiates every symbolic zero, hiding which blocks the rule needs to
    integrate.

    The returned function takes ``(y0, t_span, *leaves)``, the leaves of a
    ``params`` pytree (:mod:`modax._params`) with ``leaf_widths`` columns
    each. JAX's symbolic zeros are per leaf, so the leaves whose tangent is a
    ``SymbolicZero`` are the parameter columns the joint solve leaves out.

    The rule fires only under a JAX differentiation transform, so an
    undifferentiated call pays nothing. It contracts ``S``, and a save hook's
    quantity tangents, which depend only on the primals, with the input
    tangents. That contraction is linear, so JAX transposes it and
    ``jax.grad`` and ``jax.jacrev`` work off the same rule with no adjoint
    solve. Differentiating with respect to ``t_span`` raises
    NotImplementedError.
    """
    primal_solver = solver_for(None)
    widths = tuple(int(w) for w in leaf_widths)
    n_params = sum(widths)
    starts = tuple(sum(widths[:i]) for i in range(len(widths)))

    @jax.custom_jvp
    def solve_fn(y0: ArrayLike, t_span: ArrayLike, *leaves: jax.Array) -> RawOutput:
        return primal_solver(y0, t_span, combine_params(leaves))

    @functools.partial(solve_fn.defjvp, symbolic_zeros=True)
    def solve_jvp(
        primals: tuple[ArrayLike, ...],
        tangents: tuple[ArrayLike | SymbolicZero, ...],
    ) -> tuple[RawOutput, Any]:
        y0, t_span, *leaves = primals
        dy0, dt_span, *dleaves = tangents
        params = combine_params(tuple(jnp.asarray(leaf) for leaf in leaves))

        if not isinstance(dt_span, SymbolicZero):
            raise NotImplementedError(
                "differentiating a modax solve with respect to t_span is not "
                "supported; wrap the save times in jax.lax.stop_gradient, or "
                "differentiate with respect to y0 and/or params only"
            )

        live = [i for i, d in enumerate(dleaves) if not isinstance(d, SymbolicZero)]
        wrt_y0 = not isinstance(dy0, SymbolicZero)
        wrt_params = bool(live)
        if not (wrt_y0 or wrt_params):
            out = primal_solver(y0, t_span, params)
            return out, jax.tree_util.tree_map(_zero_tangent, out)

        # All leaves live is the whole row and none is no parameter block.
        # Both keep the spec, and so the kernel, the same as a plain array's.
        param_columns = (
            None
            if len(live) in (0, len(widths))
            else tuple(c for i in live for c in range(starts[i], starts[i] + widths[i]))
        )
        spec = SensitivitySpec(
            n_vars, n_params, wrt_y0, wrt_params, sens_error_control, param_columns
        )
        hist, quantities, stats = solver_for(spec)(y0, t_span, params)
        state, sens = split_augmented(hist, spec)

        # The input tangents side by side, in the spec's direction order: the
        # initial state, then the live leaves' columns in leaf order. Each
        # leaf gets a cotangent of its own under transposition.
        axis_size = state.shape[0]
        directions = []
        if wrt_y0:
            directions.append(_match_shape(cast(ArrayLike, dy0), axis_size, "y0"))
        directions += [
            _match_shape(cast(ArrayLike, dleaves[i]), axis_size, "params") for i in live
        ]
        dirs = jnp.concatenate(directions, axis=1)
        tangent = jnp.einsum("nsik,nk->nsi", sens, dirs)

        if quantities is None:
            q_out, dq_out = None, None
        else:
            q_out = quantities[:, :, :1]
            dq_out = jnp.einsum("nsko,nk->nso", quantities[:, :, 1:], dirs)[:, :, None]
        # Step counters are integers and carry no derivative, but JAX still
        # wants a tangent of the right (float0) type for every output.
        d_stats = jax.tree_util.tree_map(_zero_tangent, stats)
        return (state, q_out, stats), (tangent, dq_out, d_stats)

    return solve_fn
