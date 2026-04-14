"""KenCarp5 solver via diffrax — IMEX ESDIRK method of order 5."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np


def _solve_with_diffrax(
    explicit_ode_fn,
    implicit_ode_fn,
    y0_arr,
    save_times,
    params_arr,
    *,
    rtol,
    atol,
    first_step,
    max_steps,
    backend=None,
):
    """Solve an ensemble using Diffrax KenCarp5."""
    t0 = float(save_times[0])
    tf = float(save_times[-1])
    dt0 = first_step if first_step is not None else (tf - t0) * 1e-6

    def _solve_one(p):
        term = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, y, args: explicit_ode_fn(y, t, p)),
            diffrax.ODETerm(lambda t, y, args: implicit_ode_fn(y, t, p)),
        )
        solver = diffrax.KenCarp5()
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=tf,
            dt0=dt0,
            y0=y0_arr,
            stepsize_controller=controller,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(ts=save_times),
        )
        return sol.ys

    return jax.jit(jax.vmap(_solve_one), backend=backend)(params_arr)


def make_solver(explicit_ode_fn, implicit_ode_fn):
    """Create a reusable KenCarp5 ensemble solver.

    Parameters
    ----------
    explicit_ode_fn : callable
        Explicit ODE contribution with signature
        ``explicit_ode_fn(y, t, params) -> dy/dt``.
    implicit_ode_fn : callable
        Implicit ODE contribution with signature
        ``implicit_ode_fn(y, t, params) -> dy/dt``.
    """

    def _solve(
        y0,
        t_span,
        params,
        *,
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
    ):
        """Solve an ensemble of split ODEs.

        Parameters
        ----------
        y0 : array, shape [n_vars]
            Shared initial state.
        t_span : array-like, shape [n_save]
            Strictly increasing array of save times (len >= 2).
        params : array, shape [N, ...]
            Per-trajectory parameters.

        Returns
        -------
        array, shape [N, n_save, n_vars]
        """
        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        params_arr = jnp.asarray(params)
        save_times = jnp.asarray(
            np.asarray(t_span, dtype=np.float64), dtype=jnp.float64
        )
        return _solve_with_diffrax(
            explicit_ode_fn,
            implicit_ode_fn,
            y0_arr,
            save_times,
            params_arr,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
        )

    return _solve
