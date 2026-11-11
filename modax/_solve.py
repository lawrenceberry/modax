"""``modax.solve``: one entry point, the method chosen up front."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import jax.numpy as jnp
from jax.typing import ArrayLike

from modax._methods import Auto, Method, Rodas5P, Tsit5
from modax._numba_common import OdeFn, PIDGains, SolveOutput

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
    split: bool = False,
    split_max_driver: int = 16,
    trajectories_per_block: int | None = None,
    max_registers: int | None = None,
    cache: bool = False,
) -> SolveOutput | HookedOutput:
    """Integrate an ensemble of ODE trajectories.

    ``ode_fn`` is a plain right-hand side, a :class:`~modax.Block` or a
    :class:`~modax.Hierarchy` of blocks. How it steps is ``method``:
    :class:`~modax.Rodas5P` (the default), :class:`~modax.Tsit5`, or
    :class:`~modax.Auto`, which switches between them. The method object holds
    the options of that method's numerics; everything here is about the
    problem and the launch. In a hierarchy ``method`` is how the blocks that
    name no method of their own step.

    The solve is ``jit``-traceable, an outer ``jax.vmap`` becomes one larger
    ensemble launch, and ``jax.jvp``, ``jax.grad`` and their relatives
    differentiate it with respect to ``y0`` and ``params`` by integrating the
    forward-sensitivity system alongside the state (Tsit5 and Rodas5P on plain
    systems, Rodas5P blocks in a hierarchy).

    Args:
        ode_fn: The right-hand side ``(y, t, p) -> tuple`` of scalars, a
            :class:`~modax.Block` or a :class:`~modax.Hierarchy`. A plain
            function is ``cuda.jit``-ed for you; see Writing ODE callbacks.
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
            column) and compiles a sparse direct LU for the pattern. A
            hierarchy's blocks carry their own.
        sens_error_control: Hold the sensitivities to the tolerance too, when
            differentiated. ``False`` takes them out of step-size control, so
            the joint solve takes the plain solve's steps.
        sens_param_columns: The ``params`` columns to differentiate with
            respect to; ``None`` is all of them.
        save_hook: Rodas5P only. A device function ``hook(save_idx, y, t,
            p_row, acc)`` called at every save time with the dense-output
            state, accumulating into an ``(N, hook_size)`` output (see
            :func:`modax.rodas5P.quantity_hook`). The solve then returns
            ``(history, hook_output)`` and is not differentiable.
        hook_size: The width of the save hook's output row.
        save_history: ``False`` keeps only the final state, for a hook that
            consumes the saves itself.
        sensitivity: Rodas5P only. A sensitivity spec integrated as the
            solve's primal, so a save hook can form tangents.
        tf_index: Rodas5P only. A ``params`` column holding each
            trajectory's own end time.
        array_rhs: Rodas5P only. An ``f(y, t, p, out)`` device function for
            the stage evaluations, which need not be the tuple form ``ode_fn``
            takes for differentiation.
        split: Rodas5P only. Split the system along its ``sparsity``
            pattern's one-way coupling into a driver block integrated on
            steps of its own and a main block that reads it.
        split_max_driver: The most states the driver block may take.
        trajectories_per_block: Rodas5P only. Trajectories per CUDA block;
            ``None`` is a warp.
        max_registers: Rodas5P only. A cap on the kernel's registers per
            thread.
        cache: Rodas5P only. Keep the compiled kernel on disk, keyed by
            everything it was built from, so a later process loads it.

    Returns:
        The integration history, ``(N, n_save, n_vars)``. With
        ``return_stats``, ``(history, stats)``; with a save hook,
        ``(history, hook_output)`` or ``(history, hook_output, stats)``.
    """
    from modax import rodas5P, tsit5
    from modax._hierarchy import Block, as_hierarchy

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
    rodas_only: dict[str, Any] = dict(
        save_hook=save_hook,
        hook_size=hook_size,
        save_history=save_history,
        sensitivity=sensitivity,
        tf_index=tf_index,
        array_rhs=array_rhs,
        split=split,
        split_max_driver=split_max_driver,
        trajectories_per_block=trajectories_per_block,
        max_registers=max_registers,
        cache=cache,
    )

    hierarchy = as_hierarchy(ode_fn)
    if hierarchy is not None or isinstance(method, Auto):
        if hierarchy is None:
            # Auto on a plain system: one block, the final block of its hierarchy
            ode_fn = Block(
                cast(OdeFn, ode_fn),
                int(jnp.shape(y0)[-1]),
                sparsity=sparsity,
                method=method,
            )
            sparsity = None
        return rodas5P._solve(
            cast(Any, ode_fn),
            y0,
            t_span,
            params,
            method=method,
            sparsity=sparsity,
            **common,
            **rodas_only,
            **_controller(method),
        )
    rhs = cast(OdeFn, ode_fn)  # a plain right-hand side from here on
    if isinstance(method, Rodas5P):
        return rodas5P._solve(
            rhs,
            y0,
            t_span,
            params,
            sparsity=sparsity,
            **common,
            **rodas_only,
            **_controller(method),
        )

    # Tsit5 on a plain system: its kernel takes none of the Rodas5P-only options
    defaults = {
        "sparsity": None,
        "save_hook": None,
        "hook_size": 0,
        "save_history": True,
        "sensitivity": None,
        "tf_index": None,
        "array_rhs": None,
        "split": False,
        "split_max_driver": 16,
        "trajectories_per_block": None,
        "max_registers": None,
        "cache": False,
    }
    given = {**rodas_only, "sparsity": sparsity}
    unsupported = [
        k
        for k, default in defaults.items()
        if (given[k] is not None if default is None else given[k] != default)
    ]
    if unsupported:
        raise ValueError(
            f"Tsit5 does not take {', '.join(unsupported)}; use Rodas5P or Auto"
        )
    gains = method.gains or PIDGains()
    return tsit5.solve(
        rhs,
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
