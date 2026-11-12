"""``params`` as a pytree, and differentiation that follows its leaves.

JAX tracks symbolic zeros per leaf, so the sensitivity rule integrates only
the columns of the leaves being differentiated. The first half needs no
device: it checks the flattening, and which columns the rule asks a stand-in
solver for. The second solves the linear cascade of ``test_sensitivity``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax._params import as_param_row, param_leaves, param_width
from modax._sensitivity import SensitivitySpec, make_sensitivity_solver
from modax.rodas5P import solve as rodas5P_solve
from modax.tsit5 import solve as tsit5_solve
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
        pytest.param(tsit5_solve, {}, id="tsit5"),
        pytest.param(rodas5P_solve, {"lu_precision": "fp64"}, id="rodas5P"),
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


# --- which columns the rule integrates (no device needed) ------------------
N_VARS, N_SAVE = 2, 3


def _recording_solver(specs: list[SensitivitySpec | None]):
    """A stand-in for the kernels that notes each spec and returns zeros."""

    def solver_for(spec):
        def solve(y0, t_span, params):
            specs.append(spec)
            n_system = N_VARS if spec is None else spec.n_aug
            return jnp.zeros((jnp.shape(params)[0], N_SAVE, n_system))

        return solve

    return solver_for


def _solve_with_leaves(specs, leaves):
    widths = [leaf.shape[-1] for leaf in leaves]
    fn = make_sensitivity_solver(
        _recording_solver(specs), N_VARS, widths, sens_error_control=True
    )
    return fn(jnp.zeros((1, N_VARS)), jnp.arange(N_SAVE * 1.0), *leaves)


def test_only_the_differentiated_leaves_columns_are_integrated():
    rates, temp, tail = jnp.ones((1, 2)), jnp.ones((1, 1)), jnp.ones((1, 3))

    specs = []
    jax.jacfwd(lambda t: _solve_with_leaves(specs, (rates, t, tail)))(temp)
    assert specs[-1].param_columns == (2,)
    assert specs[-1].n_aug == N_VARS * 2

    specs = []
    jax.grad(lambda r, x: _solve_with_leaves(specs, (r, temp, x)).sum(), (0, 1))(
        rates, tail
    )
    assert specs[-1].param_columns == (0, 1, 3, 4, 5)


def test_stop_gradient_takes_a_leaf_out():
    rates, temp = jnp.ones((1, 2)), jnp.ones((1, 1))
    specs = []
    jax.grad(
        lambda r, t: _solve_with_leaves(specs, (r, jax.lax.stop_gradient(t))).sum(),
        (0, 1),
    )(rates, temp)
    assert specs[-1].param_columns == (0, 1)


def test_every_leaf_differentiated_is_the_whole_row():
    """The same spec, and so the same kernel, as a plain array."""
    rates, temp = jnp.ones((1, 2)), jnp.ones((1, 1))
    specs = []
    jax.grad(lambda r, t: _solve_with_leaves(specs, (r, t)).sum(), (0, 1))(rates, temp)
    assert specs[-1].param_columns is None
    assert specs[-1].n_params == 3


def test_differentiating_y0_alone_carries_no_parameter_block():
    leaves = (jnp.ones((1, 2)), jnp.ones((1, 1)))
    specs = []
    widths = [leaf.shape[-1] for leaf in leaves]
    fn = make_sensitivity_solver(
        _recording_solver(specs), N_VARS, widths, sens_error_control=True
    )
    jax.jacfwd(lambda y: fn(y, jnp.arange(N_SAVE * 1.0), *leaves))(
        jnp.zeros((1, N_VARS))
    )
    assert specs[-1] == SensitivitySpec(N_VARS, 3, True, False, True, None)


def test_an_undifferentiated_solve_is_the_plain_one():
    specs = []
    _solve_with_leaves(specs, (jnp.ones((1, 2)),))
    assert specs == [None]


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
    """Each leaf's cotangent is its own columns of the plain array's gradient."""
    y0, t_span = jnp.asarray(Y0), jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(params):
        return jnp.sum(solve_fn(cascade, y0, t_span, params, **settings) ** 2)

    whole = np.asarray(jax.grad(loss)(jnp.array([A, B])))
    tree = jax.grad(loss)({"a": jnp.array(A), "b": jnp.array(B)})
    np.testing.assert_allclose(float(tree["a"]), whole[0], rtol=1e-7)
    np.testing.assert_allclose(float(tree["b"]), whole[1], rtol=1e-7)

    only_a = jax.grad(lambda a: loss((a, B)))(A)
    np.testing.assert_allclose(float(only_a), whole[0], rtol=1e-7)


@requires_cuda
@both_solvers
def test_mixed_shared_and_per_trajectory_leaves(solve_fn, kwargs):
    """A per-trajectory leaf beside a shared one, under grad and under vmap."""
    t_span = jnp.asarray(T_SPAN)
    y0 = jnp.asarray(Y0)
    rates = jnp.array([[A], [0.4], [1.1]])
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(a):
        return jnp.sum(solve_fn(cascade, y0, t_span, (a, B), **settings) ** 2)

    grad = np.asarray(jax.grad(loss)(rates))
    assert grad.shape == (3, 1)
    for row in range(3):
        single = jax.grad(loss)(rates[row])
        np.testing.assert_allclose(grad[row], np.asarray(single), rtol=1e-8)

    def vmapped(a):
        per_solve = jax.vmap(
            lambda r: solve_fn(cascade, y0, t_span, (r, B), **settings)
        )(a)
        return jnp.sum(per_solve**2)

    np.testing.assert_allclose(np.asarray(jax.grad(vmapped)(rates)), grad, rtol=1e-8)
