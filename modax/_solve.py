"""``modax.solve``: one entry point, the method chosen up front."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Sequence, cast

import jax.numpy as jnp
from jax.typing import ArrayLike

from modax._methods import Auto, Method, Rodas5P, Tsit5, method_parts, option_or
from modax._numba_common import OdeFn, PIDGains, SolveOutput
from modax._sparsity import Pattern, normalize_sparsity, pattern_mask

if TYPE_CHECKING:
    from modax._hierarchy import Block, Hierarchy
    from modax._sensitivity import SensitivitySpec
    from modax._sparsity import Sparsity
    from modax.rodas5P import ArrayRhs, HookedOutput, SaveHook


def _controller(method: Method) -> dict[str, Any]:
    """Return the solve-level method options as per-method solver keywords.

    In a hierarchy a block's method falls back to these for any option it
    leaves ``None``. An ``Auto`` contributes its stiff method's options.
    """
    rodas, tsit = method_parts(method)
    gains = option_or(rodas, "gains", option_or(tsit, "gains", PIDGains()))
    return {
        "pcoeff": gains.pcoeff,
        "icoeff": gains.icoeff,
        "dcoeff": gains.dcoeff,
        "lu_precision": option_or(rodas, "lu_precision", "fp32"),
        "ordering": option_or(rodas, "ordering", "amd"),
    }


@functools.cache
def _one_block(rhs: OdeFn, n: int, pattern: Pattern | None, method: Method):
    """Wrap a plain system as a one-block hierarchy, the same object every call.

    The hierarchy kernel caches on the block, so a fresh block would mean a
    fresh compile on every solve.
    """
    from modax._hierarchy import Block

    sparsity = None if pattern is None else pattern_mask(pattern)
    return Block(rhs, n, sparsity=sparsity, method=method)


def solve(
    ode_fn: OdeFn | Block | Hierarchy,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: ArrayLike,
    *,
    method: Method | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step: float | None = None,
    max_steps: int = 100000,
    error_weights: ArrayLike | None = None,
    return_stats: bool = False,
    sparsity: Sparsity | None = None,
    sens_error_control: bool = True,
    sens_param_columns: Sequence[int] | None = None,
    save_hook: SaveHook | None = None,
    hook_size: int = 0,
    save_history: bool = True,
    sensitivity: SensitivitySpec | None = None,
    tf_index: int | None = None,
    array_rhs: ArrayRhs | None = None,
    trajectories_per_block: int | None = None,
    max_registers: int | None = None,
    cache: bool = False,
) -> SolveOutput | HookedOutput:
    """Integrate an ensemble of ODE trajectories.

    ``ode_fn`` is a plain right-hand side, a :class:`~modax.Block`, a
    :class:`~modax.Hierarchy`, or a system split by
    :meth:`Hierarchy.from_sparsity <modax.Hierarchy.from_sparsity>`.
    ``method`` picks the stepper: :class:`~modax.Rodas5P` (the default),
    :class:`~modax.Tsit5`, or :class:`~modax.Auto`, which switches between
    them. The method object holds its own numerical options, and the
    arguments here cover the problem and the launch. In a hierarchy,
    ``method`` steps every block that names no method of its own.

    The solve is ``jit``-traceable, and an outer ``jax.vmap`` becomes one
    larger ensemble launch. ``jax.jvp``, ``jax.grad`` and friends
    differentiate it with respect to ``y0`` and ``params`` by integrating the
    forward-sensitivity system with the state. That works for Tsit5 and
    Rodas5P on plain systems and for Rodas5P blocks in a hierarchy.

    A plain Tsit5 solve runs on Tsit5's own kernel, which has no save hook,
    per-trajectory end time, kernel cache or launch tuning. Ask for any of
    those and modax solves it as a one-block Tsit5 hierarchy instead, as it
    always does for a plain ``Auto`` solve. The hierarchy kernel has all of
    them but no ``backend``.

    Args:
        ode_fn: The right-hand side ``(y, t, p) -> tuple`` of scalars, a
            :class:`~modax.Block`, a :class:`~modax.Hierarchy`, or the result
            of :meth:`Hierarchy.from_sparsity
            <modax.Hierarchy.from_sparsity>`. A plain function is
            ``cuda.jit``-ed for you. See Writing ODE callbacks.
        y0: Initial state, ``(n_vars,)`` or ``(N, n_vars)``. Every trajectory
            shares a 1-D state. For a hierarchy, the blocks' states upstream
            first (or in its ``layout``'s order).
        t_span: Save times, ``(n_save,)``, shared across the ensemble. Not
            differentiable.
        params: Parameters, ``(n_params,)`` or ``(N, n_params)``. ``N`` comes
            from whichever of ``y0`` and ``params`` has a batch axis. If both do,
            they must agree.
        method: :class:`~modax.Rodas5P` (``None`` means ``Rodas5P()``),
            :class:`~modax.Tsit5` or :class:`~modax.Auto`.
        rtol: Relative tolerance of the weighted RMS error norm.
        atol: Absolute tolerance of the weighted RMS error norm.
        first_step: The first step size. ``None`` (or a non-positive value)
            starts each trajectory from ``1e-6`` of its own integration
            window.
        max_steps: Attempted steps a trajectory may take before it stops.
        error_weights: Per-component weights of the error norm, ``(n_vars,)``.
            A zero takes a component out of step-size control.
        return_stats: Also return a dict of per-trajectory step counters.
        sparsity: The Jacobian's pattern: an ``(n_vars, n_vars)`` mask, a
            scipy sparse matrix, or an ``(nnz, 2)`` array of ``(row, col)``
            pairs. It must cover every structural nonzero. Rodas5P colours the
            Jacobian's columns (one AD sweep per colour, not per column) and
            compiles a sparse direct LU for the pattern. Tsit5 forms no
            Jacobian and ignores it. A hierarchy's blocks carry their own.
        sens_error_control: Hold the sensitivities to the tolerance too, when
            differentiated. ``False`` takes them out of step-size control, so
            the joint solve takes the plain solve's steps.
        sens_param_columns: The ``params`` columns to differentiate with
            respect to. ``None`` means all of them.
        save_hook: A device function ``hook(save_idx, y, t, p_row, acc)``
            the kernel calls at every save time with the dense-output state,
            accumulating into an ``(N, hook_size)`` output (see
            :func:`modax.rodas5P.quantity_hook`). The solve then returns
            ``(history, hook_output)`` and is not differentiable.
        hook_size: The width of the save hook's output row.
        save_history: ``False`` keeps only the final state, for a hook that
            consumes the saves itself.
        sensitivity: A sensitivity spec the solve integrates as its primal,
            so a save hook can form tangents. Tsit5 does not take it.
        tf_index: A ``params`` column holding each trajectory's own end time.
        array_rhs: An ``f(y, t, p, out)`` device function for the stage
            evaluations, so they skip the tuple form ``ode_fn`` keeps for
            differentiation. Plain Rodas5P solves only.
        trajectories_per_block: Trajectories per CUDA block. ``None`` means
            a warp.
        max_registers: A cap on the kernel's registers per thread.
        cache: Keep the compiled kernel on disk, keyed by everything that
            built it, so a later process loads it.

    Returns:
        The integration history, ``(N, n_save, n_vars)``. With
        ``return_stats``, ``(history, stats)``. With a save hook,
        ``(history, hook_output)`` or ``(history, hook_output, stats)``.

    Raises:
        TypeError: If ``method`` is not a Rodas5P, Tsit5 or Auto.
        ValueError: If ``array_rhs`` comes with anything but a plain Rodas5P
            solve, or a split system gets a ``sparsity`` or a ``y0`` of the
            wrong width.
    """
    from modax import rodas5P, tsit5
    from modax._hierarchy import as_hierarchy
    from modax._split import SplitHierarchy

    method = Rodas5P() if method is None else method
    if not isinstance(method, (Rodas5P, Tsit5, Auto)):
        raise TypeError(f"method must be Rodas5P, Tsit5 or Auto, got {method!r}")
    common: dict[str, Any] = dict(
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        error_weights=error_weights,
        return_stats=return_stats,
        sens_error_control=sens_error_control,
        sens_param_columns=sens_param_columns,
    )
    # what the hierarchy kernel and Rodas5P's have, and Tsit5's does not
    kernel: dict[str, Any] = dict(
        save_hook=save_hook,
        hook_size=hook_size,
        save_history=save_history,
        sensitivity=sensitivity,
        tf_index=tf_index,
        trajectories_per_block=trajectories_per_block,
        max_registers=max_registers,
        cache=cache,
    )
    n_vars = int(jnp.shape(y0)[-1])
    hierarchy = as_hierarchy(ode_fn)
    split = ode_fn if isinstance(ode_fn, SplitHierarchy) else None
    plain_rodas = hierarchy is None and split is None and isinstance(method, Rodas5P)
    if array_rhs is not None and not plain_rodas:
        raise ValueError("array_rhs applies to a plain system under Rodas5P only")

    if split is not None:
        if sparsity is not None:
            raise ValueError("a split system carries its own pattern; pass none")
        if split.n != n_vars:
            raise ValueError(f"the split system has {split.n} states, y0 {n_vars}")
        return rodas5P._solve(
            cast(OdeFn, split.ode_fn),
            y0,
            t_span,
            params,
            method=method,
            sparsity=pattern_mask(split.sparsity),
            split=True,
            split_max_driver=split.max_driver,
            **common,
            **kernel,
            **_controller(method),
        )

    if hierarchy is None and isinstance(method, Tsit5):
        on_hierarchy_kernel = (
            save_hook is not None
            or not save_history
            or sensitivity is not None
            or tf_index is not None
            or trajectories_per_block is not None
            or max_registers is not None
            or cache
        )
        if not on_hierarchy_kernel:
            gains = method.gains or PIDGains()
            return tsit5.solve(
                cast(OdeFn, ode_fn),
                y0,
                t_span,
                params,
                rtol=rtol,
                atol=atol,
                first_step=first_step,
                max_steps=max_steps,
                error_weights=error_weights,
                return_stats=return_stats,
                sens_error_control=sens_error_control,
                sens_param_columns=sens_param_columns,
                pcoeff=gains.pcoeff,
                icoeff=gains.icoeff,
                dcoeff=gains.dcoeff,
                backend=method.backend,
            )

    if hierarchy is None and not isinstance(method, Rodas5P):
        # Auto, or Tsit5 asking for the hierarchy kernel's options: one block,
        # the final block of its hierarchy
        pattern = None if sparsity is None else normalize_sparsity(sparsity, n_vars)
        ode_fn = _one_block(cast(OdeFn, ode_fn), n_vars, pattern, method)
        sparsity = None
    return rodas5P._solve(
        cast(Any, ode_fn),
        y0,
        t_span,
        params,
        method=method,
        sparsity=sparsity,
        array_rhs=array_rhs,
        **common,
        **kernel,
        **_controller(method),
    )
