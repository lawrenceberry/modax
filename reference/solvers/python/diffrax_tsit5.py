"""Tsit5 solver via diffrax — explicit RK method of order 5."""

import functools

import diffrax
import jax
import jax.numpy as jnp
import numpy as np


def make_solver(ode_fn):
    """Create a reusable Tsit5 ensemble solver.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
    """

    @functools.partial(jax.jit, static_argnames=("max_steps",))
    def _solve_impl(y0_arr, save_times, params_arr, *, rtol, atol, dt0, max_steps):
        t0 = save_times[0]
        tf = save_times[-1]

        def _solve_one(p):
            term = diffrax.ODETerm(lambda t, y, args: ode_fn(y, t, p))
            solver = diffrax.Tsit5()
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

        return jax.vmap(_solve_one)(params_arr)

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
        """Solve an ensemble of ODEs.

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
        times_np = np.asarray(t_span, dtype=np.float64)
        save_times = jnp.asarray(times_np, dtype=jnp.float64)
        dt0 = jnp.float64(
            first_step if first_step is not None else (times_np[-1] - times_np[0]) * 1e-6
        )
        return _solve_impl(
            y0_arr,
            save_times,
            params_arr,
            rtol=rtol,
            atol=atol,
            dt0=dt0,
            max_steps=max_steps,
        )

    return _solve
