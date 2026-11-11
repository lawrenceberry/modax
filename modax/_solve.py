"""``modax.solve``: one entry point, the method chosen up front."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Sequence, cast

import jax.numpy as jnp
from jax.typing import ArrayLike

from modax._methods import Auto, Method, Rodas5P, Tsit5
from modax._numba_common import OdeFn, PIDGains, SolveOutput
from modax._sparsity import Pattern, normalize_sparsity, pattern_mask

if TYPE_CHECKING:
    from modax._hierarchy import Block, Hierarchy
    from modax._sensitivity import SensitivitySpec
    from modax._sparsity import Sparsity
    from modax.rodas5P import ArrayRhs, HookedOutput, SaveHook


def _controller(method: Method) -> dict[str, Any]:
    """The solve-level method options as the per-method solvers' keywords.

    In a hierarchy these are what a block's own method falls back to where it
    leaves an option ``None``: an ``Auto`` contributes its stiff method's.
    """
    rodas = method.stiff if isinstance(method, Auto) else method
    rodas = rodas if isinstance(rodas, Rodas5P) else None
    tsit = method.nonstiff if isinstance(method, Auto) else method
    tsit = tsit if isinstance(tsit, Tsit5) else None
    gains = (rodas.gains if rodas is not None else None) or (
        tsit.gains if tsit is not None else None
    )
    gains = gains or PIDGains()
    return {
        "pcoeff": gains.pcoeff,
        "icoeff": gains.icoeff,
        "dcoeff": gains.dcoeff,
        "lu_precision": (rodas.lu_precision if rodas is not None else None) or "fp32",
        "ordering": (rodas.ordering if rodas is not None else None) or "amd",
    }


@functools.cache
def _one_block(rhs: OdeFn, n: int, pattern: Pattern | None, method: Method):
    """A plain system as the one block of a hierarchy, the same object every call.

    The hierarchy kernel is cached on the block, so a block built afresh on
    every solve would compile afresh on every solve.
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

    ``ode_fn`` is a plain right-hand side, a :class:`~modax.Block` or a
    :class:`~modax.Hierarchy` of blocks, or a system split along its sparsity
    pattern by :meth:`Hierarchy.from_sparsity <modax.Hierarchy.from_sparsity>`.
    How it steps is ``method``: :class:`~modax.Rodas5P` (the default),
    :class:`~modax.Tsit5`, or :class:`~modax.Auto`, which switches between
    them. The method object holds the options of that method's numerics;
    everything here is about the problem and the launch. In a hierarchy
    ``method`` is how the blocks that name no method of their own step.

    The solve is ``jit``-traceable, an outer ``jax.vmap`` becomes one larger
    ensemble launch, and ``jax.jvp``, ``jax.grad`` and their relatives
    differentiate it with respect to ``y0`` and ``params`` by integrating the
    forward-sensitivity system alongside the state (Tsit5 and Rodas5P on plain
    systems, Rodas5P blocks in a hierarchy).

    A plain system under ``Tsit5`` runs on Tsit5's own kernel, which has no
    save hook, per-trajectory end time, kernel cache or launch tuning. Ask for
    any of those and it is solved as a hierarchy of one Tsit5 block instead,
    as a plain system under ``Auto`` always is: the same method, on the
    hierarchy kernel, which has them all (and no ``backend``).

    Args:
        ode_fn: The right-hand side ``(y, t, p) -> tuple`` of scalars, a
            :class:`~modax.Block`, a :class:`~modax.Hierarchy`, or the result
            of :meth:`Hierarchy.from_sparsity
            <modax.Hierarchy.from_sparsity>`. A plain function is
            ``cuda.jit``-ed for you; see Writing ODE callbacks.
        y0: Initial state, ``(n_vars,)`` or ``(N, n_vars)``; a 1-D state is
            shared by every trajectory. A hierarchy's is the blocks' states,
            upstream first (or in its ``layout``'s order).
        t_span: Save times, ``(n_save,)``, shared across the ensemble. Not
            differentiable.
        params: Parameters, ``(n_params,)`` or ``(N, n_params)``. ``N`` comes
            from whichever of ``y0`` and ``params`` is batched; if both are,
            they must agree.
        method: :class:`~modax.Rodas5P` (``None`` means ``Rodas5P()``),
            :class:`~modax.Tsit5` or :class:`~modax.Auto`.
        rtol: Relative tolerance of the weighted RMS error norm.
        atol: Absolute tolerance of the weighted RMS error norm.
        first_step: The first step size. ``None`` (or a non-positive value)
            starts each trajectory from ``1e-6`` of its own integration
            window.
        max_steps: Attempted steps a trajectory may take before it stops.
        error_weights: Per-component weights of the error norm, ``(n_vars,)``;
            a zero takes a component out of step-size control.
        return_stats: Also return a dict of per-trajectory step counters.
        sparsity: The Jacobian's pattern: an ``(n_vars, n_vars)`` mask, a
            scipy sparse matrix, or an ``(nnz, 2)`` array of ``(row, col)``
            pairs. It must cover every structural nonzero. Rodas5P then
            colours the Jacobian's columns (one AD sweep per colour, not per
            column) and compiles a sparse direct LU for the pattern; Tsit5,
            which forms no Jacobian, ignores it. A hierarchy's blocks carry
            their own.
        sens_error_control: Hold the sensitivities to the tolerance too, when
            differentiated. ``False`` takes them out of step-size control, so
            the joint solve takes the plain solve's steps.
        sens_param_columns: The ``params`` columns to differentiate with
            respect to; ``None`` is all of them.
        save_hook: A device function ``hook(save_idx, y, t, p_row, acc)``
            called at every save time with the dense-output state,
            accumulating into an ``(N, hook_size)`` output (see
            :func:`modax.rodas5P.quantity_hook`). The solve then returns
            ``(history, hook_output)`` and is not differentiable.
        hook_size: The width of the save hook's output row.
        save_history: ``False`` keeps only the final state, for a hook that
            consumes the saves itself.
        sensitivity: A sensitivity spec integrated as the solve's primal, so
            a save hook can form tangents. Not through Tsit5.
        tf_index: A ``params`` column holding each trajectory's own end time.
        array_rhs: A plain Rodas5P solve only. An ``f(y, t, p, out)`` device
            function for the stage evaluations, which need not be the tuple
            form ``ode_fn`` takes for differentiation.
        trajectories_per_block: Trajectories per CUDA block; ``None`` is a
            warp.
        max_registers: A cap on the kernel's registers per thread.
        cache: Keep the compiled kernel on disk, keyed by everything it was
            built from, so a later process loads it.

    Returns:
        The integration history, ``(N, n_save, n_vars)``. With
        ``return_stats``, ``(history, stats)``; with a save hook,
        ``(history, hook_output)`` or ``(history, hook_output, stats)``.
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
