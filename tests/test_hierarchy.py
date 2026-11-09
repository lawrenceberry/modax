"""A hierarchy of one-way-coupled blocks with published quantities, against the joint solve."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modax import solve
from modax._hierarchy import Block, Hierarchy

jax.config.update("jax_enable_x64", True)
cuda = pytest.importorskip("numba_cuda_mlir")

T_SPAN = np.linspace(0.0, 3.0, 7)
TOL = dict(rtol=1e-9, atol=1e-11, lu_precision="fp64")


def rhs_a(y, t, p):
    """A forced decay."""
    return (-p[0] * y[0] + math.cos(t),)


def out_a(y, t, p):
    """What the others read of it: two nonlinear quantities."""
    return (math.sin(y[0]), y[0] * t)


def rhs_b(y, t, p, q):
    """A stiff follower of the first quantity, forced by the second."""
    return (-p[1] * (y[0] - q[0]) + q[1],)


def rhs_c(y, t, p, q):
    """Reads the follower's state and the root's quantities: a diamond."""
    return (-p[2] * (y[0] - q[0]) + q[1],)


def joint(y, t, p):
    """The same three equations as one system."""
    return (
        -p[0] * y[0] + math.cos(t),
        -p[1] * (y[1] - math.sin(y[0])) + y[0] * t,
        -p[2] * (y[2] - y[1]) + math.sin(y[0]),
    )


A = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a")
B = Block(rhs_b, 1, reads=(A,), name="b")
C = Block(rhs_c, 1, reads=(B, A), name="c")
CHAIN = Hierarchy(C)

Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])


def test_the_hierarchy_orders_the_blocks_upstream_first():
    assert CHAIN.blocks == (A, B, C)
    assert CHAIN.offsets == (0, 1, 2)
    assert CHAIN.n_vars == 3


def test_a_cycle_is_refused():
    first = Block(rhs_b, 1, reads=(), name="first")
    second = Block(rhs_b, 1, reads=(first,), name="second")
    object.__setattr__(first, "reads", (second,))
    with pytest.raises(ValueError, match="cycle"):
        Hierarchy(second)


def test_the_blocks_match_the_joint_solve():
    split = np.asarray(solve(CHAIN, Y0, T_SPAN, P, **TOL))
    whole = np.asarray(solve(joint, Y0, T_SPAN, P, **TOL))
    assert np.abs(split - whole).max() < 1e-7


def test_the_sensitivities_match_the_joint_solve():
    def tangent(ode, dp):
        def at(p):
            return solve(ode, Y0, T_SPAN, p, **TOL)

        return np.asarray(jax.jvp(at, (P,), (dp,))[1])

    for dp in (
        jnp.array([[1.0, 0.0, 0.0]] * 2),
        jnp.array([[0.0, 1.0, 0.0]] * 2),
        jnp.array([[0.0, 0.0, 1.0]] * 2),
    ):
        split = tangent(CHAIN, dp)
        whole = tangent(joint, dp)
        assert np.abs(split - whole).max() < 1e-6 * max(1.0, np.abs(whole).max())


def test_every_block_steps_on_its_own():
    _, stats = solve(CHAIN, Y0, T_SPAN, P, return_stats=True, **TOL)
    _, joint_stats = solve(joint, Y0, T_SPAN, P, return_stats=True, **TOL)
    # the root's steps are counted in loop_steps on top of the sink's accepted and rejected ones
    extra = (
        np.asarray(stats["loop_steps"])
        - np.asarray(stats["accepted_steps"])
        - np.asarray(stats["rejected_steps"])
    )
    assert (extra > 0).all()
    assert (
        np.asarray(stats["accepted_steps"])
        <= np.asarray(joint_stats["accepted_steps"]) * 1.5
    ).all()
