"""The save hook: quantities ``(y, t, p) -> tuple`` kept at every save time.

The kernel evaluates the hook on the same dense-output states the history
holds, so its values must match the same function applied to the saved
history afterwards. The number of quantities is read from the tuple the hook
returns. The quantities are differentiable like the history, through the
same JVP rule: a differentiated solve stores each save's values and then
their tangent along every direction, so everything here is checked against
``jax.jacfwd`` of the quantities over a plain solve's history.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax._sensitivity import SensitivitySpec, quantity_hook
from modax.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

N_SAVE = 33
SETTINGS = dict(rtol=1e-8, atol=1e-10, lu_precision="fp64")


def damped_oscillator(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])


def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])


def energy_alone(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]),)


QUANTITIES = pytest.mark.parametrize(
    "quantities, n_out", [(energy_and_position, 2), (energy_alone, 1)]
)


def _problem():
    t_span = np.linspace(0.0, 4.0, N_SAVE, dtype=np.float64)
    y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]], dtype=np.float64)
    params = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]], dtype=np.float64)
    return y0, t_span, params


def _over_history(quantities, n_out, y0, p):
    """The quantities applied to a plain solve's history, ``(N, n_save, n_out)``."""
    _, t_span, _ = _problem()
    hist = rodas5P_solve(damped_oscillator, y0, t_span, p, **SETTINGS)
    values = quantities((hist[..., 0], hist[..., 1]), None, (p[..., None, 0],))
    return jnp.stack(values[:n_out], axis=-1)


def _hooked(quantities, y0, p, **kwargs):
    """The quantities as a hooked solve returns them, ``(N, n_save, n_out)``."""
    _, t_span, _ = _problem()
    _, q = rodas5P_solve(
        damped_oscillator, y0, t_span, p, save_hook=quantities, **SETTINGS, **kwargs
    )
    return q


# --- values ------------------------------------------------------------------
@requires_cuda
@QUANTITIES
def test_the_hook_keeps_its_quantities_at_every_save(quantities, n_out):
    y0, t_span, params = _problem()
    hist, q = rodas5P_solve(
        damped_oscillator, y0, t_span, params, save_hook=quantities, **SETTINGS
    )
    assert np.asarray(hist).shape == (3, N_SAVE, 2)
    assert q.shape == (3, N_SAVE, n_out)
    expected = _over_history(quantities, n_out, jnp.asarray(y0), jnp.asarray(params))
    np.testing.assert_allclose(q, expected, rtol=1e-12, atol=1e-15)
    # The undamped trajectory conserves energy, and the kept values show it.
    np.testing.assert_allclose(q[2, :, 0], q[2, 0, 0], rtol=1e-6)


@requires_cuda
def test_save_history_false_keeps_the_final_state_and_the_quantities():
    y0, t_span, params = _problem()
    hist = np.asarray(rodas5P_solve(damped_oscillator, y0, t_span, params, **SETTINGS))
    final, q = rodas5P_solve(
        damped_oscillator,
        y0,
        t_span,
        params,
        save_hook=energy_and_position,
        save_history=False,
        **SETTINGS,
    )
    assert np.asarray(final).shape == (3, 1, 2)
    np.testing.assert_allclose(np.asarray(final)[:, 0], hist[:, -1], atol=1e-13)
    # What an in-kernel accumulation used to do, done on the kept quantities.
    np.testing.assert_allclose(
        np.trapezoid(np.asarray(q)[..., 1], t_span, axis=1),
        np.trapezoid(hist[..., 0], t_span, axis=1),
        rtol=1e-12,
    )


def test_save_history_false_needs_a_hook():
    y0, t_span, params = _problem()
    with pytest.raises(ValueError, match="save_history=False"):
        rodas5P_solve(damped_oscillator, y0, t_span, params, save_history=False)


@requires_cuda
def test_a_hook_must_return_a_tuple():
    y0, t_span, params = _problem()
    with pytest.raises(TypeError, match=r"write a single one as \(q,\)"):
        rodas5P_solve(
            damped_oscillator,
            y0,
            t_span,
            params,
            save_hook=lambda y, t, p: y[0] * y[0],
        )


# --- derivatives through the JVP rule ----------------------------------------
@requires_cuda
@QUANTITIES
def test_quantities_differentiate_like_the_history(quantities, n_out):
    """``jax.jacfwd`` of the hooked quantities, with respect to ``params`` and ``y0``."""
    y0, _, params = _problem()
    y0, params = jnp.asarray(y0), jnp.asarray(params)
    rows = np.arange(3)
    for argnum in (0, 1):
        hooked = jax.jacfwd(
            lambda a, p: _hooked(quantities, a, p, sens_error_control=False),
            argnums=argnum,
        )(y0, params)
        expected = jax.jacfwd(
            lambda a, p: _over_history(quantities, n_out, a, p), argnums=argnum
        )(y0, params)
        # Each trajectory's quantities depend on its own row only.
        np.testing.assert_allclose(
            np.asarray(hooked)[rows, :, :, rows],
            np.asarray(expected)[rows, :, :, rows],
            rtol=1e-6,
            atol=1e-9,
        )


@requires_cuda
def test_grad_through_the_quantities_alone():
    """``save_history=False``: a loss of the quantities, and its gradient."""
    y0, _, params = _problem()
    params = jnp.asarray(params)

    def loss(p):
        q = _hooked(energy_alone, y0, p, save_history=False)
        return jnp.sum(q**2)

    def reference(p):
        return jnp.sum(_over_history(energy_alone, 1, jnp.asarray(y0), p) ** 2)

    value, grad = jax.value_and_grad(loss)(params)
    np.testing.assert_allclose(float(value), float(reference(params)), rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(jax.grad(reference)(params)), rtol=1e-6
    )


@requires_cuda
def test_quantity_gradients_follow_the_params_leaves():
    """Differentiating one leaf of the parameters gives its columns' gradient."""
    y0, _, params = _problem()
    stiffness, damping = jnp.asarray(params[:, :1]), jnp.asarray(params[:, 1:])

    def loss(k, c):
        return jnp.sum(_hooked(energy_and_position, y0, (k, c)) ** 2)

    only_k = jax.grad(loss)(stiffness, damping)
    both = jax.grad(loss, (0, 1))(stiffness, damping)
    assert only_k.shape == (3, 1)
    np.testing.assert_allclose(np.asarray(only_k), np.asarray(both[0]), rtol=1e-6)


@requires_cuda
def test_a_vmapped_hooked_solve_is_one_ensemble():
    y0, t_span, params = _problem()
    batched = jax.vmap(lambda a, p: _hooked(energy_and_position, a, p))(y0, params)
    assert batched.shape == (3, 1, N_SAVE, 2)
    np.testing.assert_allclose(
        np.asarray(batched)[:, 0],
        np.asarray(_hooked(energy_and_position, y0, params)),
        rtol=1e-12,
        atol=1e-15,
    )


# --- one direction per trajectory, the layout DISCO-EB uses ------------------
@requires_cuda
@QUANTITIES
def test_jvp_along_one_leaf_carries_one_column(quantities, n_out):
    """A leaf of its own for one parameter column, differentiated by ``jax.jvp``.

    The rule then integrates that column alone (``param_columns=(0,)``), out of
    step control with ``sens_error_control=False``, and the quantities'
    tangent is that column of their Jacobian.
    """
    y0, _, params = _problem()
    stiffness, damping = jnp.asarray(params[:, :1]), jnp.asarray(params[:, 1:])

    values, tangent = jax.jvp(
        lambda k: _hooked(
            quantities,
            y0,
            (k, damping),
            save_history=False,
            sens_error_control=False,
        ),
        (stiffness,),
        (jnp.ones_like(stiffness),),
    )
    assert values.shape == tangent.shape == (3, N_SAVE, n_out)
    y0_j, p_j = jnp.asarray(y0), jnp.asarray(params)
    rows = np.arange(3)
    d_p = np.asarray(
        jax.jacfwd(lambda p: _over_history(quantities, n_out, y0_j, p))(p_j)
    )[rows, :, :, rows]
    np.testing.assert_allclose(
        values, _over_history(quantities, n_out, y0_j, p_j), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(tangent, d_p[..., 0], rtol=1e-6, atol=1e-9)


def test_quantity_hook_needs_a_spec_of_its_problem():
    with pytest.raises(ValueError, match="sensitivity spec"):
        quantity_hook(
            energy_alone, 1, 2, 2, SensitivitySpec(3, 2, wrt_y0=False, wrt_params=True)
        )
    with pytest.raises(ValueError, match="at least one"):
        quantity_hook(energy_alone, 0, 2, 2)
