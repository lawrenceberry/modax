"""The split solve: a driver block integrated ahead of the main block, from the pattern alone."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import modax
from modax import Hierarchy, Rodas5P, Tsit5

jax.config.update("jax_enable_x64", True)
cuda = pytest.importorskip("numba_cuda_mlir.cuda")

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

LAM, MU = 1.0, 25.0
PATTERN = np.array([[0, 0], [1, 0], [1, 1]])
T_SPAN = np.linspace(0.0, 3.0, 7)
FP64 = Rodas5P(lu_precision="fp64")


def ode(y, t, p):
    """A forced decay driving a stiff follower: theta' = -lam theta + cos t, x' = -mu (x - theta)."""
    return (-p[0] * y[0] + math.cos(t), -p[1] * (y[1] - y[0]))


SPLIT = Hierarchy.from_sparsity(ode, 2, PATTERN)


def solve(y0, p, split, **kw):
    """The system split by its pattern, or whole with the pattern as its sparsity."""
    if split:
        return modax.solve(SPLIT, y0, T_SPAN, p, method=FP64, **kw)
    return modax.solve(ode, y0, T_SPAN, p, sparsity=PATTERN, method=FP64, **kw)


def split_blocks(pattern, n, max_driver=16):
    return Hierarchy.from_sparsity(ode, n, pattern, max_driver)


def theta_exact(t, lam, theta0):
    c = theta0 - lam / (1.0 + lam**2)
    return c * jnp.exp(-lam * t) + (lam * jnp.cos(t) + jnp.sin(t)) / (1.0 + lam**2)


def x_exact(t, lam, mu, theta0, x0):
    """x = mu e^{-mu t} int_0^t e^{mu s} theta(s) ds + x0 e^{-mu t}."""
    c = theta0 - lam / (1.0 + lam**2)
    a = lam / (1.0 + lam**2)
    b = 1.0 / (1.0 + lam**2)
    part_c = c * mu / (mu - lam) * (jnp.exp(-lam * t) - jnp.exp(-mu * t))
    # mu e^{-mu t} int e^{mu s} (a cos s + b sin s) ds
    e = jnp.exp(-mu * t)
    ic = (mu * jnp.cos(t) + jnp.sin(t) - mu * e) / (mu**2 + 1.0)
    is_ = (mu * jnp.sin(t) - jnp.cos(t) + e) / (mu**2 + 1.0)
    return part_c + mu * (a * ic + b * is_) + x0 * e


def test_the_pattern_splits():
    blocks = split_blocks(PATTERN, 2)
    assert blocks.driver == (0,) and blocks.main == (1,)
    with pytest.raises(ValueError, match="nothing else feeds back"):
        split_blocks(np.ones((2, 2)), 2)


def test_the_driver_is_closed_under_dependence():
    # 0 drives the coupled pair (1, 2); 3 integrates 0 alone and nothing reads it
    pattern = np.array([(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 0), (3, 0)])
    blocks = split_blocks(pattern, 4, max_driver=2)
    assert blocks.driver == (0, 3) and blocks.main == (1, 2)
    assert split_blocks(pattern, 4, max_driver=1).driver == (0,)
    assert split_blocks(pattern, 4, max_driver=4).driver == (
        0,
        1,
        2,
    )  # the pair fits; 3 would empty the main block
    # a sink that reads the pair stays in the main block
    pattern = np.vstack([pattern, [(3, 1)]])
    assert split_blocks(pattern, 4, max_driver=2).driver == (0,)


@pytest.mark.parametrize("split", [False, True])
def test_split_matches_the_exact_solution(split):
    y0 = jnp.array([[0.3, 0.1], [1.0, -0.5]])
    p = jnp.array([[LAM, MU], [0.5 * LAM, 2.0 * MU]])
    sol = np.asarray(solve(y0, p, split, rtol=1e-9, atol=1e-11))
    for n in range(2):
        th = np.asarray(theta_exact(T_SPAN, p[n, 0], y0[n, 0]))
        x = np.asarray(x_exact(T_SPAN, p[n, 0], p[n, 1], y0[n, 0], y0[n, 1]))
        assert np.abs(sol[n, :, 0] - th).max() < 1e-7
        assert np.abs(sol[n, :, 1] - x).max() < 1e-7


def test_split_and_plain_agree_to_the_tolerance():
    y0 = jnp.array([[0.3, 0.1]])
    p = jnp.array([[LAM, MU]])
    kw = dict(rtol=1e-8, atol=1e-10)
    plain = np.asarray(solve(y0, p, False, **kw))
    sp = np.asarray(solve(y0, p, True, **kw))
    assert np.abs(plain - sp).max() < 1e-6


def test_split_sensitivities_match_the_closed_form():
    """d(theta, x)/d(lam, mu) and d/dy0 through the split, against the exact solution's derivatives."""
    y0 = jnp.array([0.3, 0.1])
    p = jnp.array([LAM, MU])
    kw = dict(rtol=1e-9, atol=1e-11, sens_error_control=False)

    def final(p_, y0_):
        return solve(y0_, p_, True, **kw)[0, -1]

    def exact(p_, y0_):
        t = T_SPAN[-1]
        return jnp.array(
            [theta_exact(t, p_[0], y0_[0]), x_exact(t, p_[0], p_[1], y0_[0], y0_[1])]
        )

    jp = np.asarray(jax.jacfwd(final, argnums=0)(p, y0))
    jp_exact = np.asarray(jax.jacfwd(exact, argnums=0)(p, y0))
    assert np.abs(jp - jp_exact).max() < 1e-5 * (1.0 + np.abs(jp_exact).max())
    jy = np.asarray(jax.jacfwd(final, argnums=1)(p, y0))
    jy_exact = np.asarray(jax.jacfwd(exact, argnums=1)(p, y0))
    assert np.abs(jy - jy_exact).max() < 1e-5 * (1.0 + np.abs(jy_exact).max())
    # reverse mode falls out of the same rule
    g = np.asarray(jax.grad(lambda p_: final(p_, y0)[1])(p))
    assert np.abs(g - jp_exact[1]).max() < 1e-5 * (1.0 + np.abs(jp_exact).max())


def test_split_with_a_save_hook_sees_the_user_layout():
    from numba_cuda_mlir import cuda as nbcuda

    @nbcuda.jit(device=True)
    def hook(y, t, p):
        return (y[0], y[1])

    y0 = jnp.array([[0.3, 0.1]])
    p = jnp.array([[LAM, MU]])
    hist, acc = solve(y0, p, True, rtol=1e-8, atol=1e-10, save_hook=hook)
    hist = np.asarray(hist)
    assert np.allclose(np.asarray(acc), hist, atol=1e-9)


def test_the_driver_steps_on_its_own():
    """A slow driver under a fast follower takes far fewer steps than the follower, at the same accuracy."""
    y0 = jnp.array([0.3, 0.1])
    p = jnp.array(
        [0.05, MU]
    )  # a decay a hundred times slower than the follower's relaxation
    sol, stats = solve(y0, p, True, rtol=1e-8, atol=1e-10, return_stats=True)
    accepted, rejected, loop = (
        int(stats[k][0]) for k in ("accepted_steps", "rejected_steps", "loop_steps")
    )
    driver_attempts = loop - accepted - rejected
    assert 0 < driver_attempts < accepted / 2, (driver_attempts, accepted)
    expected = x_exact(T_SPAN, 0.05, MU, 0.3, 0.1)
    assert jnp.max(jnp.abs(sol[0, :, 1] - expected)) < 1e-6


def test_from_sparsity_steps_with_the_solve_method():
    y0 = jnp.array([[0.3, 0.1]])
    p = jnp.array([[LAM, MU]])
    assert (SPLIT.driver, SPLIT.main) == ((0,), (1,))
    kw = dict(rtol=1e-8, atol=1e-10)
    implicit = np.asarray(solve(y0, p, True, **kw))
    # the blocks step with the solve's method, whatever it is
    explicit = np.asarray(modax.solve(SPLIT, y0, T_SPAN, p, method=Tsit5(), **kw))
    assert np.abs(explicit - implicit).max() < 1e-6


def test_from_sparsity_refuses_a_pattern_that_does_not_split():
    with pytest.raises(ValueError, match="nothing else feeds back"):
        Hierarchy.from_sparsity(ode, 2, np.ones((2, 2)))
