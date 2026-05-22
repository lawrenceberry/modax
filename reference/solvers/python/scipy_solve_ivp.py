"""Reference CPU solver via :func:`scipy.integrate.solve_ivp`.

This mirrors the no-GPU baseline used by serial cosmology codes such as ECHO21,
which integrate each trajectory independently on the CPU with
``scipy.integrate.solve_ivp``.  It is used as the "without GPU" reference for the
worked cosmological examples.

The right-hand side functions in the examples are written with ``jax.numpy``.
To keep the comparison honest (a serial CPU solve with no GPU traffic) the RHS
is JIT-compiled on the CPU backend and all per-step arrays are kept on the host,
so no GPU dispatch occurs during the integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

_CPU = jax.devices("cpu")[0]


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    method: str = "LSODA",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step=None,
    max_steps: int | None = None,
    **_ignored,
):
    """Solve an ensemble serially on the CPU with ``scipy.solve_ivp``.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side ``ode_fn(y, t, params) -> dy/dt`` (jax.numpy).
    y0 : array, shape ``[n_vars]`` or ``[N, n_vars]``
        Shared initial state (broadcast to all trajectories) or per-trajectory.
    t_span : array-like, shape ``[n_save]``
        Strictly increasing save times (``len >= 2``).
    params : array, shape ``[N, ...]``
        Per-trajectory parameters.
    method : str
        Any ``solve_ivp`` method.  ``"LSODA"`` (auto stiff/non-stiff) matches the
        choice made by ECHO21; use ``"RK45"`` for non-stiff problems.

    Returns
    -------
    array, shape ``[N, n_save, n_vars]``
    """
    params_arr = np.asarray(params)
    n = params_arr.shape[0]
    y0_arr = np.asarray(y0, dtype=np.float64)
    if y0_arr.ndim == 1:
        y0_arr = np.broadcast_to(y0_arr, (n, y0_arr.shape[0]))
    save_times = np.asarray(t_span, dtype=np.float64)
    t0, tf = float(save_times[0]), float(save_times[-1])
    n_vars = y0_arr.shape[1]

    # Compile the RHS once on the CPU backend so each evaluation stays on host.
    with jax.default_device(_CPU):
        cpu_rhs = jax.jit(lambda y, t, p: ode_fn(y, t, p), backend="cpu")
        params_cpu = [jax.device_put(params_arr[i], _CPU) for i in range(n)]

    out = np.full((n, save_times.shape[0], n_vars), np.nan, dtype=np.float64)
    for i in range(n):
        p_i = params_cpu[i]

        def fun(t, y, p=p_i):
            return np.asarray(cpu_rhs(jnp.asarray(y), t, p), dtype=np.float64)

        sol = solve_ivp(
            fun,
            (t0, tf),
            y0_arr[i],
            method=method,
            t_eval=save_times,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
        )
        # On an early failure solve_ivp leaves ``sol.y`` as an empty list;
        # otherwise it is an ``[n_vars, n_reached]`` array.  Pad the remainder
        # (or the whole row) with NaN.
        ys = np.asarray(sol.y, dtype=np.float64)
        if ys.ndim == 2 and ys.shape[0] == n_vars:
            ys = ys.T
            out[i, : ys.shape[0]] = ys
    return out
