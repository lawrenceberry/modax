"""Continuous forward sensitivity analysis for the numba-cuda ensemble solvers.

The solve reaches JAX through an XLA FFI custom call, which autodiff cannot
see into (see "Why continuous forward sensitivities" in
docs/usage/gradients.md). A ``jax.custom_jvp`` rule supplies the derivatives
instead, integrating the sensitivity system alongside the state.

With ``S(t) = dy(t)/dtheta``, differentiating ``y' = f(t, y, p)`` gives the
variational equation

    S' = J_y(t) S + J_p(t),    J_y = df/dy,  J_p = df/dtheta.

Forward mode asks for directional derivatives, ``S dtheta``, and never for
``S`` itself. Contracting the variational equation with the tangent
``(dy0, dp)`` closes on one column,

    s' = J_y s + J_p dp,    s(t0) = dy0,

so the rule integrates the joint system ``[y, s]`` of ``2 n_vars``
components, whatever the number of parameters, and ``jax.jacfwd`` one
column per direction, all in the trajectory's own thread. The kernel reads
each direction as data: a trajectory's parameter row is ``(p, dp_0, ...,
dp_{d-1})``, the callbacks read ``p`` from its front, and the sweep for
column ``b`` seeds the parameters with the window ``dp_b``.

A launch that takes the tangents as data cannot be transposed, so reverse
mode is not supported: it would need ``S`` itself, a column per moving
input, or an adjoint solve.

Layout
------
The augmented state is the state followed by one ``n_vars``-long block per
direction, each starting from that direction's ``dy0``::

    z[j]                        = y[j]
    z[n_vars + b * n_vars + r]  = s_b[r]

and the parameter row the state's ``n_params`` parameters followed by one
``n_params``-long direction each::

    row[c]                       = p[c]
    row[n_params + b * n_params + c] = dp_b[c]

``J_y s_b + J_p dp_b`` comes from ``numba_enzyme.jvp`` of the user's
``ode_fn``, seeded over the callback's *flattened* argument list ``(y_0 ...
y_n-1, t, p_0 ... p_m-1)`` with ``(s_b, 0, dp_b)``: the whole right-hand side
in one sweep. Seeding the time argument returns ``df/dt``. No Jacobian is
ever formed, and Enzyme reads only the first ``n_params`` entries of the row
it is handed, so the callbacks never see the directions.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax._src.interpreters.batching import BatchTracer
from jax._src.interpreters.partial_eval import JaxprTracer
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
    """The directions a joint solve carries.

    Hashable and part of the kernel cache keys. ``error_control`` lives here,
    not on the solve, because the error-norm denominator it picks is a
    compile-time constant of the kernel. The launch's ``y0`` holds the state
    and then each direction's ``dy0``, and its parameter row the parameters
    and then each direction's ``dp`` (:attr:`row_width` entries).

    Raises:
        ValueError: The spec carries no direction.
    """

    n_vars: int
    n_params: int
    n_dirs: int
    error_control: bool = True

    def __post_init__(self) -> None:
        if self.n_dirs < 1:
            raise ValueError("a sensitivity spec must carry at least one direction")

    def check_extents(self, n_vars: int, n_params: int, against: str) -> None:
        """Raise ValueError unless the spec is for ``n_vars`` states and ``n_params`` parameters."""
        if self.n_vars != n_vars or self.n_params != n_params:
            raise ValueError(
                f"the sensitivity spec is for {self.n_vars} variables and "
                f"{self.n_params} parameters, {against} {n_vars} and {n_params}"
            )

    @property
    def n_sens(self) -> int:
        return self.n_dirs

    @property
    def row_width(self) -> int:
        """The parameter row's length: the parameters, then a ``dp`` per direction."""
        return self.n_params * (1 + self.n_dirs)

    def direction_offset(self, b: int) -> int:
        """Where direction ``b``'s ``dp`` starts in the parameter row."""
        return self.n_params * (1 + b)

    @property
    def n_aug(self) -> int:
        """Size of the joint ``[y, s_0, ...]`` system the kernel integrates."""
        return self.n_vars * (1 + self.n_dirs)

    @property
    def n_error(self) -> int:
        """Component count the weighted-RMS error norm divides by.

        With the sensitivities out of step-size control their weights are zero, so
        dividing by ``n_vars`` rather than ``n_aug`` reproduces the plain solve's
        error norm, and its step sequence, bit for bit.
        """
        return self.n_aug if self.error_control else self.n_vars


def own_params(n_params: int, spec: SensitivitySpec | None) -> int:
    """The parameters the callbacks read, out of a launch's ``n_params``-wide row.

    Raises:
        ValueError: The row is not as wide as the spec's.
    """
    if spec is None:
        return n_params
    if n_params != spec.row_width:
        raise ValueError(
            f"a joint launch's parameter row has {spec.row_width} entries "
            f"({spec.n_params} parameters and {spec.n_dirs} directions), "
            f"got {n_params}"
        )
    return spec.n_params


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

    The sensitivity right-hand side ``J_y s_b + J_p dp_b`` is the sweep seeded
    with ``(s_b, 0, dp_b)``. Building it from unit columns would cost
    ``n_vars + 1`` sweeps.
    """
    return make_quantity_tangent(ode_fn, n_vars, n_vars, n_params)


@functools.cache
def make_second_tangent(ode_fn: OdeFn, n_vars: int, n_params: int) -> Tangent:
    """``D2 f(y, t, p)[u, v] + D f(y, t, p)[w]``, forward over forward.

    The joint system's off-diagonal Jacobian block holds second derivatives of
    ``ode_fn``. Seeding ``u = (s_b, 0, dp_b)`` and ``v = (w, 0, 0)`` applies that
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
    """Read-only zeros, every zero direction a window into it.

    The zero state, time and parameter seeds of the sweeps that need one. A
    per-thread buffer could not promote to registers.
    """
    length = max(spec.n_vars, spec.n_params)
    return np.zeros(length, dtype=np.float64)


@functools.cache
def make_augmented_local_writer(ode_fn: OdeFn, spec: SensitivitySpec) -> MLIRDispatcher:
    """Joint ``[f, J_y s_b + J_p dp_b]`` writer for a kernel's thread-local state.

    Each thread writes its own trajectory's whole augmented vector from and into
    its own local arrays, one forward sweep per direction seeded with ``(s_b,
    0, dp_b)``, ``dp_b`` read from the parameter row. Threads share nothing, so
    the device function needs no synchronisation under its callers' divergent
    ``if running``.
    """
    n_vars = spec.n_vars
    n_dirs = spec.n_dirs
    n_params = spec.n_params
    fn_device = as_cuda_device(ode_fn)
    tangent_of = make_tangent(ode_fn, n_vars, n_params)

    @cuda.jit(device=True)
    def write_vector(z_row, t, p_row, out):
        values = fn_device(z_row, t, p_row)
        for j in range(n_vars):
            out[j] = values[j]

        tangent = cuda.local.array(n_vars, types.float64)
        for k in range(n_dirs):
            base = n_vars + k * n_vars
            start = n_params * (1 + k)
            tangent_of(
                tangent,
                z_row,
                t,
                p_row,
                z_row[base : base + n_vars],
                0.0,
                p_row[start : start + n_params],
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
    save's values with their derivative along every direction ``b`` of the
    spec, direction after direction::

        dq_b = (dq/dy) s_b + (dq/dp) dp_b

    One Enzyme forward sweep of ``quantities`` seeded with ``(s_b, 0, dp_b)``
    gives each. The derivative
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
    width = n_outputs * (1 + n_sens)
    tangent_of = make_quantity_tangent(quantities, n_outputs, n_vars, n_params)

    @cuda.jit(device=True)
    def store_values_and_tangents(save_idx, y, t, p_row, acc):
        values = fn_device(y, t, p_row)
        base = width * save_idx
        for j in range(n_outputs):
            acc[base + j] = values[j]

        tangent = cuda.local.array(n_outputs, types.float64)
        for k in range(n_sens):
            column = n_vars + k * n_vars
            start = n_params * (1 + k)
            tangent_of(
                tangent,
                y[0:n_vars],
                t,
                p_row,
                y[column : column + n_vars],
                0.0,
                p_row[start : start + n_params],
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
    """Check that ``(n, n_aug)`` holds the joint initial state ``[y0, dy0_0, ...]``.

    Raises:
        ValueError: It is not as wide as the joint system.
    """
    if y0_arr.shape[1] != spec.n_aug:
        raise ValueError(
            f"a joint initial state has {spec.n_aug} components, got {y0_arr.shape[1]}"
        )
    return y0_arr


def augmented_error_weights(
    weights_arr: npt.NDArray[np.float64], spec: SensitivitySpec
) -> npt.NDArray[np.float64]:
    """``(n, n_vars)`` error weights -> ``(n, n_aug)``.

    A zero weight drops a component from the error norm, so
    ``spec.error_control=False`` makes the joint solve take the plain solve's
    steps exactly, and a ``jax.jvp`` returns the plain call's value bit for
    bit. It is not the default: nothing then ties the sensitivities'
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


def _reverse_mode(tangents: Sequence[Any]) -> bool:
    """Whether these tangents are reverse mode's: no value yet, to be transposed.

    ``jax.grad``, ``jax.vjp``, ``jax.jacrev`` and ``jax.linearize`` trace the
    rule with partial-evaluation tracers, possibly under ``vmap``. Under
    ``jit`` the tangents are staged values, which ``jit(jvp(f))`` and
    ``grad(jit(f))`` both give, so the second is caught only when JAX tries
    to transpose the launch.
    """
    for tangent in tangents:
        while isinstance(tangent, jax.core.Tracer):
            if isinstance(tangent, JaxprTracer):
                return True
            if not isinstance(tangent, BatchTracer):
                break
            tangent = tangent.val
    return False


_REVERSE_MODE = (
    "modax differentiates in forward mode only: use jax.jvp or jax.jacfwd. "
    "The solve integrates the derivative along each tangent direction, and "
    "reverse mode (jax.grad, jax.value_and_grad, jax.vjp, jax.jacrev, "
    "jax.linearize) has no directions to give it. For the gradient of a "
    "scalar loss, jax.jacfwd(loss) costs one column per parameter, and "
    "jax.jacfwd(lambda p: (loss(p), loss(p)), has_aux=True) returns the "
    "value from the same solve"
)


DirectionalSolver = Callable[..., tuple[RawOutput, tuple[Any, Any]]]
"""``(y0, t_span, params, dy0, dparams)`` to the :data:`RawOutput` and the
tangents of its history and quantities. The tangents hold the directions on
the axis before last, ``(..., n_dirs, n_vars)`` and ``(..., n_dirs,
n_params)``, and so do the results."""


def make_sensitivity_solver(
    primal_solver: Callable[..., RawOutput],
    directional: Callable[[int], DirectionalSolver],
    n_vars: int,
    leaf_widths: Sequence[int],
) -> Callable[..., RawOutput]:
    """Attach a forward-sensitivity JVP rule to one solver implementation.

    ``primal_solver`` is the plain solve, ``(y0, t_span, params)``, and
    ``directional(n_columns)`` the solver of the joint system along given
    directions, for a derivative that moves ``n_columns`` inputs of each
    trajectory. Both return a :data:`RawOutput` and must already handle
    ``vmap``. The rule sits *outside* ``custom_vmap`` because that JVP path
    traces to a jaxpr and instantiates every symbolic zero, hiding which
    inputs move.

    The returned function takes ``(y0, t_span, *leaves)``, the leaves of a
    ``params`` pytree (:mod:`modax._params`) with ``leaf_widths`` columns
    each. A leaf whose tangent is a ``SymbolicZero`` moves no column.

    The rule fires only under a JAX differentiation transform, so an
    undifferentiated call pays nothing. It hands the tangents to
    ``directional``, which integrates one column along them: ``jax.jvp``
    costs one column, and ``jax.jacfwd`` one column per direction through
    its ``vmap``. The launch takes the tangents as data, so JAX cannot
    transpose it, and reverse mode raises.

    Raises:
        NotImplementedError: Under reverse mode, or when differentiating with
            respect to ``t_span``.
    """
    widths = tuple(int(w) for w in leaf_widths)

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
        if not (wrt_y0 or live):
            out = primal_solver(y0, t_span, params)
            return out, jax.tree_util.tree_map(_zero_tangent, out)
        if _reverse_mode(
            [d for d in (dy0, *dleaves) if not isinstance(d, SymbolicZero)]
        ):
            raise NotImplementedError(_REVERSE_MODE)

        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        dy0_arr = jnp.asarray(dy0) if wrt_y0 else jnp.zeros_like(y0_arr)
        dparams = combine_params(
            tuple(
                jnp.zeros_like(leaf) if isinstance(d, SymbolicZero) else jnp.asarray(d)
                for leaf, d in zip(leaves, dleaves)
            )
        )
        n_columns = (n_vars if wrt_y0 else 0) + sum(widths[i] for i in live)
        # one direction: the solver takes them stacked on the axis before last
        out, (tangent, dq_out) = directional(n_columns)(
            y0_arr,
            t_span,
            params,
            dy0_arr[..., None, :],
            dparams[..., None, :],
        )
        tangent = tangent[:, :, 0]
        d_stats = jax.tree_util.tree_map(_zero_tangent, out[2])
        return out, (tangent, dq_out, d_stats)

    return solve_fn
