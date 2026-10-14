"""gradsolve's NVIDIA Warp kernels behind the reference solvers' ``solve`` signature.

`gradsolve <https://github.com/ECLIPSE-AI4Science/gradsolve>`_ (Spurio Mancini
2026, arXiv:2609.02876 and arXiv:2609.28458) is a JAX library of GPU ensemble
ODE solvers. This module times its two fused Warp kernels, each running one
thread per trajectory:

``warp_ode``
    Adaptive Tsit5 (order 5). gradsolve refuses it on a GPU above
    :data:`TSIT5_MAX_DIM` state variables.
``warp_rosenbrock``
    Adaptive Rosenbrock23 (order 2, L-stable) with an in-kernel analytic
    Jacobian and Gaussian elimination; refused above
    :data:`ROSENBROCK23_MAX_DIM`. At the tolerances the benchmarks use it takes
    several times the steps of a fifth-order method.

A Warp kernel needs its right-hand side as a Warp function. gradsolve builds
one by translating the jaxpr of the ``jnp`` form of the callback
(``register_jax_field``), and for the stiff kernel the jaxpr of its
``jax.jacfwd`` too, so every callback here reaches the kernels the same way.

gradsolve returns the final state as host NumPy, and the stiff kernel has no
JAX bridge at all: it is launched with ``wp.launch`` and its results copied
back. A timing through :func:`solve` therefore includes the host-device copies
and the library's Python routing, as the torchdiffeq reference's does; the
modax and Diffrax cases leave their result on the device. Two of the solver
settings the other references take do not reach gradsolve: ``first_step`` is
accepted and ignored, because gradsolve starts every solve from one hundredth
of the integration window, and so is ``max_steps``, because each kernel carries
its own cap (4096 attempts for Tsit5, 50000 for Rosenbrock23). A trajectory
that exhausts the cap comes back as NaN; :func:`solve` raises then, so a
benchmark records the point as failed rather than timing an unfinished solve.

The kernels return the final state only, so the intermediate save times of the
``(N, n_save, n_vars)`` result are NaN.
"""

from __future__ import annotations

from typing import Any, Callable

import gradsolve
import jax
import jax.numpy as jnp
import numpy as np
from gradsolve.dispatch import NVAR_CEILING, choose_remat
from gradsolve.warp import warp_ode as _warp_ode
from gradsolve.warp import warp_replay as _warp_replay
from gradsolve.warp import warp_rosenbrock as _warp_rosenbrock

ENGINES = ("warp_ode", "warp_rosenbrock")

TSIT5_MAX_DIM: int = getattr(_warp_ode, "_GPU_VEC_DIM_CEILING", NVAR_CEILING)
"""The largest state dimension gradsolve launches its Tsit5 kernel for on a GPU."""

ROSENBROCK23_MAX_DIM: int = getattr(_warp_rosenbrock, "_GPU_MAT_DIM_CEILING", 12)
"""The largest state dimension gradsolve launches its Rosenbrock23 kernel for on a GPU."""

_DEVICE = "cuda"


class Problem:
    """A reference callback as the object gradsolve's ``Problem`` protocol wants.

    ``f_jax(t, y, p)`` is the tuple-form ``ode_fn(y, t, p)`` with its components
    stacked, which gradsolve traces and translates into a Warp function.
    """

    def __init__(
        self, name: str, ode_fn: Callable, dim: int, t0: float, t1: float, stiff: bool
    ):
        self.name = name
        self.dim = int(dim)
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.is_stiff = bool(stiff)
        self.ode_fn = ode_fn

    def f_jax(self, t, y, p):
        return jnp.stack(jnp.broadcast_arrays(*self.ode_fn(y, t, p)))


def _check_engine(engine: str) -> bool:
    """Whether ``engine`` is the stiff kernel; raises on an unknown one."""
    if engine not in ENGINES:
        raise ValueError(f"unknown gradsolve engine {engine!r}; one of {ENGINES}")
    return engine == "warp_rosenbrock"


# Problems by (name, dim, t0, t1, stiff). Each holds its callback, so an id()
# in a name cannot be reused for another callback while the entry is live.
_PROBLEMS: dict[tuple, Problem] = {}


def make_problem(ode_fn, n_vars: int, t_span, *, stiff: bool) -> Problem:
    """Wrap a reference callback for gradsolve.

    The problem is named after the callback, so that gradsolve's field
    registry -- keyed by name, first registration wins -- holds one
    translation per callback, and the same arguments give the same object
    back, so a timing loop does not build a new one per call.
    """
    t_span = np.asarray(t_span, dtype=np.float64)
    name = f"modax_{'stiff' if stiff else 'nonstiff'}_{n_vars}_{id(ode_fn):x}"
    key = (name, int(n_vars), float(t_span[0]), float(t_span[-1]), bool(stiff))
    problem = _PROBLEMS.get(key)
    if problem is None or problem.ode_fn is not ode_fn:
        problem = Problem(name, ode_fn, n_vars, t_span[0], t_span[-1], stiff)
        _PROBLEMS[key] = problem
    return problem


def _ensemble(y0, params) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 1:
        params = params[None, :]
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.ndim == 1:
        y0 = np.broadcast_to(y0, (params.shape[0], y0.shape[0]))
    return np.ascontiguousarray(y0), np.ascontiguousarray(params)


def _check_finite(y_final: np.ndarray, engine: str) -> None:
    bad = int(np.sum(~np.isfinite(y_final).all(axis=1)))
    if bad:
        raise RuntimeError(
            f"gradsolve {engine}: {bad} of {y_final.shape[0]} trajectories did not "
            "reach the end of the integration window within the engine's step cap"
        )


def _fused_solve(problem: Problem, y0, params, engine: str, *, rtol, atol):
    """One fused-kernel solve, raising if gradsolve ran anything else."""
    result = gradsolve.solve(
        problem,
        y0,
        params,
        engine=engine,
        fused=True,
        rtol=rtol,
        atol=atol,
        device=_DEVICE,
    )
    if result.route is not None and result.route.actual != engine:
        raise RuntimeError(
            f"gradsolve ran {result.route.actual!r} in place of {engine!r}: "
            f"{result.route.reason}"
        )
    _check_finite(np.asarray(result.y_final), engine)
    return result


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    engine: str,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=None,
):
    """Solve an ensemble with one gradsolve Warp kernel and return a NumPy array.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``,
        returning a tuple of components; it is traced with ``jnp`` arrays.
    y0 : array, shape [n_vars] or [N, n_vars]
        Shared initial state (broadcast to all trajectories) or per-trajectory.
    t_span : array-like, shape [n_save]
        Strictly increasing array of save times (len >= 2). The kernel
        integrates from the first to the last.
    params : array, shape [N, n_params]
        Per-trajectory parameters.
    engine : str
        ``"warp_ode"`` (Tsit5) or ``"warp_rosenbrock"`` (Rosenbrock23).
    first_step, max_steps
        Accepted for signature parity and ignored; see the module docstring.

    Returns
    -------
    ndarray, shape [N, n_save, n_vars]
        The initial and final states, with NaN at every save time between.

    Raises
    ------
    RuntimeError
        If gradsolve ran a different engine from the one asked for, or any
        trajectory did not finish; the benchmarks record either as a failure.
    """
    del first_step, max_steps
    stiff = _check_engine(engine)
    y0_arr, params_arr = _ensemble(y0, params)
    times = np.asarray(t_span, dtype=np.float64)
    problem = make_problem(ode_fn, y0_arr.shape[1], times, stiff=stiff)
    result = _fused_solve(problem, y0_arr, params_arr, engine, rtol=rtol, atol=atol)
    out = np.full((y0_arr.shape[0], len(times), y0_arr.shape[1]), np.nan)
    out[:, 0, :] = y0_arr
    out[:, -1, :] = np.asarray(result.y_final, dtype=np.float64)
    return out


# --- gradients ----------------------------------------------------------------
#
# gradsolve's gradient is its record-and-replay adjoint: the Warp kernel
# records the step sizes it accepts and ``jax.value_and_grad`` differentiates
# a fixed-length ``lax.scan`` replay of them. There are two costs to time: the
# replay alone, which is what gradsolve's paper reports for a gradient "once a
# record exists", and the record together with the replay, which is what a
# gradient at new parameters costs, since the recorded mesh is only valid near
# the parameters it was recorded at. The replay is jitted once over
# ``(params, y0, dts)``, so a record on every call feeds a fresh mesh into the
# same compiled program instead of rebuilding the closure
# ``gradsolve.grad_closure`` returns, which would retrace it.


def make_value_and_grad(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    engine: str,
    rtol=1e-8,
    atol=1e-10,
    record_each_call: bool,
    first_step=None,
    max_steps=None,
) -> Callable[[], tuple[Any, Any]]:
    """Return a zero-argument callable computing the value and gradient of the
    summed final state with respect to ``params`` through gradsolve's
    record-and-replay adjoint, the counterpart of
    ``benchmarks.benchmark_common.jit_value_and_grad``.

    With ``record_each_call`` the callable records the mesh with the Warp
    kernel and then differentiates its replay, the cost of a gradient at new
    parameters; without it the mesh is recorded here, once, and the callable
    differentiates the replay alone. The record buffer is sized from a
    forward solve at these settings -- twice the most steps any trajectory
    attempted -- in place of the library's fixed default of 50000 rows per
    trajectory for the stiff kernel, which at 1000 trajectories is a 400 MB
    buffer whose zeroing and copy would be most of what is timed.
    ``first_step`` and ``max_steps`` are accepted and ignored, as in
    :func:`solve`.
    """
    del first_step, max_steps
    stiff = _check_engine(engine)
    y0_arr, params_arr = _ensemble(y0, params)
    times = np.asarray(t_span, dtype=np.float64)
    problem = make_problem(ode_fn, y0_arr.shape[1], times, stiff=stiff)
    probe = _fused_solve(problem, y0_arr, params_arr, engine, rtol=rtol, atol=atol)
    attempted = np.asarray(probe.accepted_steps) + np.asarray(probe.rejected_steps)
    record_steps = int(2 * attempted.max()) + 16
    remat = choose_remat(problem.dim, problem.is_stiff)

    if stiff:
        record_fn = _warp_replay.record_rosenbrock

        def replay(y0_j, p, dts):
            return _warp_replay.replay_solve_rosenbrock_jax(
                problem, y0_j, p, dts, remat=remat
            )
    else:
        record_fn = _warp_replay.record

        def replay(y0_j, p, dts):
            return _warp_replay.replay_solve_jax(problem, y0_j, p, dts, remat=remat)

    def record():
        _y, _n, dts = record_fn(
            problem,
            y0_arr,
            params_arr,
            rtol=rtol,
            atol=atol,
            max_steps=record_steps,
            device=_DEVICE,
        )
        return dts

    def loss(p, y0_j, dts):
        return jnp.sum(replay(y0_j, p, dts))

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    y0_j = jnp.asarray(y0_arr)
    params_j = jnp.asarray(params_arr)

    if record_each_call:

        def run():
            return value_and_grad(params_j, y0_j, jnp.asarray(record()))
    else:
        dts_j = jnp.asarray(record())

        def run():
            return value_and_grad(params_j, y0_j, dts_j)

    return run
