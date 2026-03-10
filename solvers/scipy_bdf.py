import jax
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402


def solve(f, y0, t_span, *, method="BDF", rtol=1e-8, atol=1e-10, first_step=None):
    """Solve an ODE system defined by a JAX function.

    Args:
        f: JAX function mapping state vector y -> dy/dt.
        y0: Initial conditions (list or array).
        t_span: Tuple (t0, tf) for integration bounds.
        method: scipy solve_ivp method (default "BDF" for stiff systems).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        first_step: Initial step size (optional).

    Returns:
        Array of shape (n_components, n_time_points) with the solution.
    """
    f_jit = jax.jit(f)
    jac_fn = jax.jit(jax.jacobian(f))

    def rhs(_t, y):
        return f_jit(jnp.array(y)).tolist()

    def jac(_t, y):
        return jac_fn(jnp.array(y)).tolist()

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        jac=jac,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
    )
    assert sol.success, f"solve_ivp failed: {sol.message}"
    return jnp.array(sol.y)
