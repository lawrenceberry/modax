"""``params`` as a pytree, and differentiation that follows its leaves.

JAX tracks symbolic zeros per leaf, so the sensitivity rule knows which
columns of the row move. The first half needs no device: it checks the
flattening, and what the rule hands a stand-in solver. The second solves the linear cascade of ``test_sensitivity``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax import Rodas5P, Tsit5, solve
from modax._params import as_param_row, param_leaves, param_width
from modax._sensitivity import make_sensitivity_solver
from tests.test_sensitivity import (
    T_SPAN,
    Y0,
    A,
    B,
    cascade,
    cascade_dparams,
    cascade_exact,
)

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")
both_solvers = pytest.mark.parametrize(
    "solve_fn, kwargs",
    [
        pytest.param(solve, {"method": Tsit5()}, id="tsit5"),
        pytest.param(solve, {"method": Rodas5P(lu_precision="fp64")}, id="rodas5P"),
    ],
)


# --- flattening (no device needed) -----------------------------------------
def test_a_plain_array_is_one_leaf_and_its_own_row():
    params = jnp.arange(6.0).reshape(3, 2)
    assert len(param_leaves(params)) == 1
    np.testing.assert_array_equal(as_param_row(params), params)


def test_leaves_concatenate_in_tree_order():
    row = as_param_row((1.0, jnp.array([2.0, 3.0]), {"b": 5.0, "a": 4.0}))
    # a dict's leaves come out in sorted key order
    np.testing.assert_array_equal(row, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert param_width((1.0, jnp.zeros(2))) == 3


def test_shared_leaves_broadcast_to_the_ensemble():
    rates = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    row = as_param_row((rates, 9.0))
    np.testing.assert_array_equal(row[:, :2], rates)
    np.testing.assert_array_equal(row[:, 2], 9.0)


def test_a_list_is_one_array_like_leaf():
    """A nested list of rows keeps meaning ``(N, k)``, as ``jnp.asarray`` reads it."""
    row = as_param_row([[1.0, 2.0], [3.0, 4.0]])
    assert row.shape == (2, 2)


def test_bad_leaves_raise():
    with pytest.raises(ValueError, match="disagree on N"):
        as_param_row((jnp.zeros((2, 1)), jnp.zeros((3, 1))))
    with pytest.raises(ValueError, match=r"\(\), \(k,\) or \(N, k\)"):
        as_param_row(jnp.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="no leaves"):
        as_param_row(())


# --- what the rule hands the directional solver (no device needed) ---------
N_VARS, N_SAVE = 2, 3


def _recording(plain: list, calls: list):
    """Stand-ins for the kernels: the plain solve notes each call, the
    directional one the columns that move and the directions it is given."""

    def primal_solver(y0, t_span, params):
        plain.append(np.asarray(params).shape)
        return jnp.zeros((jnp.shape(params)[0], N_SAVE, N_VARS)), None, None

    def directional(n_columns):
        def solve(y0, t_span, params, dy0, dparams):
            calls.append((n_columns, np.asarray(dy0), np.asarray(dparams)))
            hist = jnp.zeros((jnp.shape(params)[0], N_SAVE, N_VARS))
            # the tangents come with their directions on the axis before last
            return (hist, None, None), (hist[:, :, None] + jnp.sum(dparams), None)

        return solve

    return primal_solver, directional


def _solve_with_leaves(plain, calls, leaves, y0=None):
    widths = [leaf.shape[-1] for leaf in leaves]
    fn = make_sensitivity_solver(*_recording(plain, calls), N_VARS, widths)
    y0 = jnp.zeros((1, N_VARS)) if y0 is None else y0
    hist, _, _ = fn(y0, jnp.arange(N_SAVE * 1.0), *leaves)
    return hist


def test_only_the_differentiated_leaves_columns_move():
    rates, temp, tail = jnp.ones((1, 2)), jnp.ones((1, 1)), jnp.ones((1, 3))
    plain, calls = [], []
    jax.jvp(
        lambda r, x: _solve_with_leaves(plain, calls, (r, temp, x)),
        (rates, tail),
        (jnp.ones((1, 2)), jnp.ones((1, 3))),
    )
    ((n_columns, _, dparams),) = calls
    assert n_columns == 5
    np.testing.assert_array_equal(dparams, [[[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]])


def test_stop_gradient_takes_a_leaf_out():
    rates, temp = jnp.ones((1, 2)), jnp.ones((1, 1))
    plain, calls = [], []
    jax.jvp(
        lambda r, t: _solve_with_leaves(plain, calls, (r, jax.lax.stop_gradient(t))),
        (rates, temp),
        (jnp.ones((1, 2)), jnp.ones((1, 1))),
    )
    assert calls[-1][0] == 2


def test_differentiating_y0_alone_moves_no_parameter():
    leaves = (jnp.ones((1, 2)), jnp.ones((1, 1)))
    plain, calls = [], []
    jax.jvp(
        lambda y: _solve_with_leaves(plain, calls, leaves, y),
        (jnp.zeros((1, N_VARS)),),
        (jnp.ones((1, N_VARS)),),
    )
    ((n_columns, dy0, dparams),) = calls
    assert n_columns == N_VARS
    np.testing.assert_array_equal(dy0, np.ones((1, 1, N_VARS)))
    np.testing.assert_array_equal(dparams, np.zeros((1, 1, 3)))


def test_an_undifferentiated_solve_is_the_plain_one():
    plain, calls = [], []
    _solve_with_leaves(plain, calls, (jnp.ones((1, 2)),))
    assert plain == [(1, 2)] and calls == []


def test_forward_mode_hands_the_tangent_to_the_directional_solver():
    """One direction over every leaf, with zeros where a leaf does not move."""
    rates, temp = jnp.ones((1, 2)), jnp.ones((1, 1))
    plain, calls = [], []
    _, tangent = jax.jvp(
        lambda r: _solve_with_leaves(plain, calls, (r, temp)),
        (rates,),
        (jnp.asarray([[2.0, 3.0]]),),
    )
    assert plain == []
    ((_, dy0, dparams),) = calls
    np.testing.assert_array_equal(dy0, np.zeros((1, 1, N_VARS)))
    np.testing.assert_array_equal(dparams, [[[2.0, 3.0, 0.0]]])
    np.testing.assert_array_equal(tangent, np.full((1, N_SAVE, N_VARS), 5.0))


def test_reverse_mode_raises():
    rates, temp = jnp.ones((1, 2)), jnp.ones((1, 1))

    def loss(r):
        return _solve_with_leaves([], [], (r, temp)).sum()

    for transform in (jax.grad(loss), jax.jacrev(loss), jax.jit(jax.grad(loss))):
        with pytest.raises(NotImplementedError, match="forward mode only"):
            transform(rates)


# --- on the device ---------------------------------------------------------
@requires_cuda
@both_solvers
def test_jacfwd_over_one_leaf_matches_closed_form(solve_fn, kwargs):
    """``params=(a, b)``: differentiating ``a`` alone gives the ``a`` column."""
    y0, t_span = jnp.asarray(Y0), jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-8, atol=1e-10, **kwargs)

    def run(a):
        return solve_fn(cascade, y0, t_span, (a, B), **settings)[0]

    np.testing.assert_allclose(
        np.asarray(run(A)), cascade_exact(T_SPAN, Y0, [A, B]), rtol=1e-7
    )
    jac = np.asarray(jax.jacfwd(run)(A))
    np.testing.assert_allclose(
        jac, cascade_dparams(T_SPAN, Y0, [A, B])[..., 0], rtol=1e-6, atol=1e-9
    )


@requires_cuda
@both_solvers
def test_leaf_gradients_match_the_whole_array(solve_fn, kwargs):
    """Each leaf's derivative is its own columns of the plain array's."""
    y0, t_span = jnp.asarray(Y0), jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(params):
        return jnp.sum(solve_fn(cascade, y0, t_span, params, **settings) ** 2)

    whole = np.asarray(jax.jacfwd(loss)(jnp.array([A, B])))
    tree = jax.jacfwd(loss)({"a": jnp.array(A), "b": jnp.array(B)})
    np.testing.assert_allclose(float(tree["a"]), whole[0], rtol=1e-7)
    np.testing.assert_allclose(float(tree["b"]), whole[1], rtol=1e-7)

    only_a = jax.jacfwd(lambda a: loss((a, B)))(A)
    np.testing.assert_allclose(float(only_a), whole[0], rtol=1e-7)


@requires_cuda
@both_solvers
def test_mixed_shared_and_per_trajectory_leaves(solve_fn, kwargs):
    """A per-trajectory leaf beside a shared one, differentiated row by row."""
    t_span = jnp.asarray(T_SPAN)
    y0 = jnp.asarray(Y0)
    rates = jnp.array([[A], [0.4], [1.1]])
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(a):
        return jnp.sum(solve_fn(cascade, y0, t_span, (a, B), **settings) ** 2)

    grad = np.asarray(jax.vmap(jax.jacfwd(loss))(rates))
    assert grad.shape == (3, 1)
    for row in range(3):
        single = jax.jacfwd(loss)(rates[row])
        np.testing.assert_allclose(grad[row], np.asarray(single), rtol=1e-8)

    # jacfwd over the whole leaf would send each trajectory all three rows'
    # directions for its one moving column.
    with pytest.raises(ValueError, match="directions"):
        jax.jacfwd(loss)(rates)
