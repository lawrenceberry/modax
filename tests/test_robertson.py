import jax
import jax.numpy as jnp
import numpy as np

from solvers.scipy_bdf import solve


@jax.jit
def robertson(y):
    """Robertson equation - stiff ODE system (Appendix A.1.3, arXiv:2304.06835).

    dy1/dt = -0.04*y1 + 1e4*y2*y3
    dy2/dt =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    dy3/dt =  3e7*y2^2
    """
    y1, y2, y3 = y
    return jnp.array(
        [
            -0.04 * y1 + 1e4 * y2 * y3,
            0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
            3e7 * y2**2,
        ]
    )


def test_scipy_bdf(benchmark):
    y = benchmark.pedantic(
        lambda: solve(
            robertson, y0=[1.0, 0.0, 0.0], t_span=(0.0, 1e5), first_step=1e-4
        ),
        warmup_rounds=1,
        rounds=1,
    )

    # Conservation: y1 + y2 + y3 = 1 (the system is conservative)
    total = y[0] + y[1] + y[2]
    np.testing.assert_allclose(total, 1.0, atol=1e-6)

    # Check final state against known values
    np.testing.assert_allclose(y[0, -1], 1.786592e-02, rtol=1e-4)
    np.testing.assert_allclose(y[1, -1], 7.274753e-08, rtol=1e-4)
    np.testing.assert_allclose(y[2, -1], 9.821340e-01, rtol=1e-4)
