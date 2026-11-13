"""``modax.solve``: the one entry point, from the arguments to the kernel launch."""

from __future__ import annotations

import functools
import gc
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.typing import ArrayLike

from modax import _rodas5P, _tsit5
from modax._hierarchy import Hierarchy, as_hierarchy, make_hierarchy_kernel
from modax._jax_common import (
    ensemble_ffi_call,
    make_custom_vmap_solver,
    prepare_inputs,
)
from modax._jax_numba_custom_call import CudaLaunch, make_launch
from modax._methods import Auto, Method, Rodas5P, Tsit5
from modax._numba_common import (
    SOLVER_ARGTYPES,
    HookedOutput,
    OdeFn,
    SaveHook,
    SolveOutput,
    SolverStats,
    SolveSettings,
)
from modax._options import DEFAULT_TRAJECTORIES_PER_BLOCK, KernelOptions
from modax._params import Params, param_leaves, param_width
from modax._sensitivity import (
    RawOutput,
    SensitivitySpec,
    make_sensitivity_solver,
    quantity_count,
    quantity_hook,
)
from modax._sensitivity import clear_caches as clear_sensitivity_caches
from modax._sparsity import Pattern, normalize_sparsity, pattern_mask
from modax._split import SplitHierarchy, make_split_kernel

if TYPE_CHECKING:
    from modax._hierarchy import Block
    from modax._sparsity import Sparsity

System = OdeFn | Hierarchy | SplitHierarchy
"""What a kernel integrates: a plain right-hand side, a hierarchy or a split system."""


def solve(
    ode_fn: OdeFn | Block | Hierarchy,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
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
    save_hook: SaveHook | None = None,
    save_history: bool = True,
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

    Args:
        ode_fn: The right-hand side ``(y, t, p) -> tuple`` of scalars, a
            :class:`~modax.Block`, a :class:`~modax.Hierarchy`, or the result
            of :meth:`Hierarchy.from_sparsity
            <modax.Hierarchy.from_sparsity>`. A plain function is
            ``cuda.jit``-ed for you. See Writing ODE callbacks.
        y0: Initial state, ``(n_vars,)`` or ``(N, n_vars)``. Every trajectory
            shares a 1-D state. For a hierarchy, the blocks' states upstream
            first (or in its ``layout``'s order).
        t_span: Save times, ``(n_save,)`` shared by the ensemble or
            ``(N, n_save)`` with one increasing row per trajectory. A row's
            first entry is where that trajectory starts and its last where it
            stops, so trajectories can start and end at times of their own.
            Not differentiable.
        params: Parameters: an array, ``(n_params,)`` or ``(N, n_params)``,
            or a pytree of them (a tuple, a dict) whose leaves are ``()``,
            ``(k,)`` or ``(N, k)`` and concatenate, in ``jax.tree.leaves``
            order, into the row ``ode_fn`` sees. Differentiation follows the
            leaves: only the columns of the leaves JAX differentiates get a
            sensitivity, so ``jax.grad(loss)(rates, temp)`` with
            ``params=(rates, temp)`` integrates the rates' columns alone,
            when ``temp`` is a constant or wrapped in ``stop_gradient``. Each
            column costs a second-order Enzyme sweep per stage. ``N`` comes
            from whichever of ``y0``, ``params`` and ``t_span`` has a batch
            axis. Batched ones must agree.
        method: :class:`~modax.Rodas5P` (``None`` means ``Rodas5P()``),
            :class:`~modax.Tsit5` or :class:`~modax.Auto`.
        rtol: Relative tolerance of the weighted RMS error norm.
        atol: Absolute tolerance of the weighted RMS error norm.
        first_step: The first step size. ``None`` (or a non-positive value)
            starts each trajectory from ``1e-6`` of its own integration
            window.
        max_steps: Attempted steps a trajectory may take before it stops.
            Save times it did not reach hold its final state.
        error_weights: Per-component weights of the error norm, ``(n_vars,)``
            or ``(N, n_vars)``. A zero takes a component out of step-size
            control.
        return_stats: Also return a dict of per-trajectory step counters,
            ``accepted_steps``, ``rejected_steps`` and ``loop_steps``, each
            ``(N,)``.
        sparsity: The Jacobian's pattern: an ``(n_vars, n_vars)`` mask, a
            scipy sparse matrix, or an ``(nnz, 2)`` array of ``(row, col)``
            pairs. It must cover every structural nonzero. Rodas5P colours the
            Jacobian's columns (one AD sweep per colour, not per column) and
            compiles a sparse direct LU for the pattern. Tsit5 forms no
            Jacobian and ignores it. A hierarchy's blocks carry their own.
        sens_error_control: Hold the sensitivities to the tolerance too, when
            differentiated. ``False`` takes them out of step-size control, so
            the joint solve takes the plain solve's steps.
        save_hook: A function ``(y, t, p) -> tuple`` of quantities to keep
            at every save time, shaped like ``ode_fn`` and compiled the same
            way. The solve returns ``(history, quantities)``, the quantities
            ``(N, n_save, n_out)`` with ``n_out`` the length of the tuple the
            hook returns. They are differentiable like the history.
        save_history: ``False`` keeps only the final state, for a hook that
            consumes the saves itself.
        trajectories_per_block: Trajectories per CUDA block, one per thread.
            ``None`` means a warp, since trajectories step adaptively and
            divergence within a warp costs nothing.
        max_registers: A cap on the kernel's registers per thread.
        cache: Keep the compiled kernel on disk, keyed by everything that
            built it, so a later process loads it.

    Returns:
        The integration history, ``(N, n_save, n_vars)``. With
        ``return_stats``, ``(history, stats)``. With a save hook,
        ``(history, quantities)`` or ``(history, quantities, stats)``.

    Raises:
        TypeError: If ``method`` is not a Rodas5P, Tsit5 or Auto, or a
            ``save_hook`` does not return a tuple of numbers.
        ValueError: If ``t_span``'s rows disagree with the ensemble, a
            hierarchy or split system gets a ``sparsity`` or a ``y0`` of the
            wrong width, or ``save_history=False`` comes without a hook.
    """
    method = Rodas5P() if method is None else method
    if not isinstance(method, (Rodas5P, Tsit5, Auto)):
        raise TypeError(f"method must be Rodas5P, Tsit5 or Auto, got {method!r}")
    if save_hook is None and not save_history:
        raise ValueError("save_history=False needs a save_hook to consume the saves")
    n_vars = int(jnp.shape(y0)[-1])
    system, pattern, max_driver = _system(ode_fn, method, n_vars, sparsity)
    if save_hook is not None:
        # Typed here, on the host, so a hook of the wrong shape fails before
        # any tracing does.
        quantity_count(save_hook, n_vars, param_width(params))
    options = KernelOptions(
        method=method,
        trajectories_per_block=(
            DEFAULT_TRAJECTORIES_PER_BLOCK
            if trajectories_per_block is None
            else int(trajectories_per_block)
        ),
        sparsity=pattern,
        max_registers=max_registers,
        save_hook=save_hook,
        save_history=bool(save_history),
        split_max_driver=max_driver,
        cache=bool(cache),
    )
    settings = SolveSettings(rtol, atol, first_step, max_steps, error_weights)

    # The JVP rule wraps the vmap-aware solvers rather than the other way
    # round: custom_vmap's own JVP path instantiates symbolic zeros, which is
    # what tells the rule which sensitivity blocks it has to integrate.
    def solver_for(spec: SensitivitySpec | None) -> custom_vmap:
        return make_custom_vmap_solver(
            functools.partial(
                _run,
                system,
                options=replace(options, spec=spec),
                settings=settings,
                return_stats=return_stats,
            )
        )

    leaves = param_leaves(params)
    hist, quantities, stats = make_sensitivity_solver(
        solver_for,
        n_vars,
        [leaf.shape[-1] for leaf in leaves],
        sens_error_control=sens_error_control,
    )(y0, t_span, *leaves)
    out: tuple[Any, ...] = (hist,)
    if quantities is not None:
        out += (quantities[:, :, 0],)  # the values alone: (N, n_save, n_out)
    if return_stats:
        out += (stats,)
    return cast(Any, out[0] if len(out) == 1 else out)


def _system(
    ode_fn: OdeFn | Block | Hierarchy | SplitHierarchy,
    method: Method,
    n_vars: int,
    sparsity: Sparsity | None,
) -> tuple[System, Pattern | None, int]:
    """What the kernel integrates, the Jacobian pattern it gets, and the split's driver bound.

    The method picks the kernel of a plain system, and nothing else does:
    Rodas5P and Tsit5 have kernels of their own, and ``Auto`` runs as a
    one-block hierarchy.
    """
    if isinstance(ode_fn, SplitHierarchy):
        if sparsity is not None:
            raise ValueError("a split system carries its own pattern; pass none")
        if ode_fn.n != n_vars:
            raise ValueError(f"the split system has {ode_fn.n} states, y0 {n_vars}")
        return ode_fn, ode_fn.sparsity, ode_fn.max_driver
    hierarchy = as_hierarchy(ode_fn)
    if hierarchy is not None:
        if hierarchy.n_vars != n_vars:
            raise ValueError(
                f"the hierarchy has {hierarchy.n_vars} states, y0 has {n_vars}"
            )
        if sparsity is not None:
            raise ValueError(
                "a hierarchy's blocks carry their own sparsity; pass none to solve"
            )
        return hierarchy, None, 16
    rhs = cast(OdeFn, ode_fn)
    if isinstance(method, Tsit5):
        return rhs, None, 16
    # Normalised here because the kernel builders cache on it, and a pattern
    # has to arrive as the same hashable value twice.
    pattern = None if sparsity is None else normalize_sparsity(sparsity, n_vars)
    if isinstance(method, Auto):
        return (
            cast(Hierarchy, as_hierarchy(_one_block(rhs, n_vars, pattern, method))),
            None,
            16,
        )
    return rhs, pattern, 16


@functools.cache
def _one_block(rhs: OdeFn, n: int, pattern: Pattern | None, method: Method) -> Block:
    """Wrap a plain system as a one-block hierarchy, the same object every call.

    The hierarchy kernel caches on the block, so a fresh block would mean a
    fresh compile on every solve.
    """
    from modax._hierarchy import Block

    sparsity = None if pattern is None else pattern_mask(pattern)
    return Block(rhs, n, sparsity=sparsity, method=method)


def _run(
    system: System,
    y0: ArrayLike,
    t_span: ArrayLike,
    params: Params,
    *,
    options: KernelOptions,
    settings: SolveSettings,
    return_stats: bool,
) -> RawOutput:
    """Launch the kernel once and return its :data:`~modax._sensitivity.RawOutput`.

    The quantities are ``(N, n_save, 1 + n_sens, n_out)``, each save's values
    and then their tangents along the spec's directions, or ``None`` without
    a hook.
    """
    inputs = prepare_inputs(y0, t_span, params, settings, options.spec)
    spec = options.spec
    n_out = n_dirs = 0
    if options.save_hook is not None:
        # The user's quantities become the kernel's hook here, where the spec
        # and the save count are known: values and tangents, save after save.
        n_out = quantity_count(options.save_hook, inputs.n_vars, inputs.n_params)
        n_dirs = 1 + (spec.n_sens if spec is not None else 0)
        options = replace(
            options,
            save_hook=quantity_hook(
                options.save_hook, n_out, inputs.n_vars, inputs.n_params, spec
            ),
            hook_size=n_out * n_dirs * inputs.n_save,
        )
    launch = _make_jax_launch(system, inputs.n, inputs.n_vars, inputs.n_params, options)
    # No global scratch: every kernel keeps a trajectory's working set in
    # registers, thread-local or shared memory.
    hist, accepted, rejected, loop_steps, hook_out = ensemble_ffi_call(
        launch,
        inputs,
        settings,
        n_save_hist=inputs.n_save if options.save_history else 1,
        hook_size=max(1, options.hook_size),
    )
    quantities = (
        None if n_out == 0 else hook_out.reshape(inputs.n, inputs.n_save, n_dirs, n_out)
    )
    stats = SolverStats(
        accepted_steps=accepted, rejected_steps=rejected, loop_steps=loop_steps
    )
    return hist, quantities, (stats if return_stats else None)


@functools.cache
def _make_jax_launch(
    system: System, n: int, n_vars: int, n_params: int, options: KernelOptions
) -> CudaLaunch:
    """Compile, load and size the kernel behind one ensemble launch."""
    if isinstance(system, SplitHierarchy):
        kernel, tpb = make_split_kernel(
            system.ode_fn, n_vars, n_params, options, system.split
        )
    elif isinstance(system, Hierarchy):
        kernel, tpb = make_hierarchy_kernel(system, n_params, options)
    elif isinstance(options.method, Tsit5):
        kernel, tpb = _tsit5.make_kernel(system, n, n_vars, options)
    else:
        kernel, tpb = _rodas5P.make_kernel(system, n_vars, n_params, options)
    return make_launch(
        kernel, SOLVER_ARGTYPES, grid=(n + tpb - 1) // tpb, block=(tpb, 1, 1)
    )


def clear_caches() -> None:
    """Drop the compiled kernels and the derivative helpers this process holds.

    Every problem size and option set compiles a kernel of its own, and the
    caches keep it alive for the life of the process, so call this when
    sweeping problem sizes in one process.
    """
    _make_jax_launch.cache_clear()
    _one_block.cache_clear()
    _tsit5._make_kernel.cache_clear()
    _rodas5P.make_kernel.cache_clear()
    make_hierarchy_kernel.cache_clear()
    make_split_kernel.cache_clear()
    clear_sensitivity_caches()
    gc.collect()
