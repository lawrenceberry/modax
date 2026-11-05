"""The Rodas5P save hook: a device function run inside the launch at every save.

The hook sees the same dense-output states the history holds, so an
accumulation it performs in-kernel must match the same accumulation done on the
saved history afterwards. Checked here with a trapezoid integral of the state
over the save grid, and with the state stored per save, both against the
history of a plain solve; and ``save_history=False`` must still return the
final state. A joint solve asked for by its ``SensitivitySpec`` hands the hook
the augmented state, so a hook can store a quantity and its derivative along
the sensitivities, which is what makes a hooked solve differentiable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax._sensitivity import SensitivitySpec, split_augmented
from modax.rodas5P import quantity_hook
from modax.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

N_SAVE = 33


def damped_oscillator(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])


def _problem():
    t_span = np.linspace(0.0, 4.0, N_SAVE, dtype=np.float64)
    y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]], dtype=np.float64)
    params = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]], dtype=np.float64)
    return y0, t_span, params


def _trapezoid_weights(t_span):
    w = np.zeros_like(t_span)
    w[1:] += 0.5 * np.diff(t_span)
    w[:-1] += 0.5 * np.diff(t_span)
    return w


@requires_cuda
def test_hook_accumulates_what_the_history_integrates():
    y0, t_span, params = _problem()
    weights = cuda.to_device(_trapezoid_weights(t_span))

    def integrate_state(save_idx, y, t, p_row, acc):
        # int y_0 dt, int y_1 dt, and the number of saves seen.
        acc[0] += weights[save_idx] * y[0]
        acc[1] += weights[save_idx] * y[1]
        acc[2] += 1.0

    settings = dict(rtol=1e-8, atol=1e-10, lu_precision="fp64")
    hist = np.asarray(rodas5P_solve(damped_oscillator, y0, t_span, params, **settings))
    final, acc = rodas5P_solve(
        damped_oscillator,
        y0,
        t_span,
        params,
        save_hook=integrate_state,
        hook_size=3,
        save_history=False,
        **settings,
    )
    final, acc = np.asarray(final), np.asarray(acc)

    assert final.shape == (3, 1, 2)
    np.testing.assert_allclose(final[:, 0], hist[:, -1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        acc[:, 0], np.trapezoid(hist[..., 0], t_span, axis=1), rtol=1e-12
    )
    np.testing.assert_allclose(
        acc[:, 1], np.trapezoid(hist[..., 1], t_span, axis=1), rtol=1e-12
    )
    assert np.all(acc[:, 2] == N_SAVE)


@requires_cuda
def test_hook_can_store_derived_saves_beside_the_history():
    y0, t_span, params = _problem()

    def store_energy(save_idx, y, t, p_row, acc):
        acc[save_idx] = 0.5 * (y[1] * y[1] + p_row[0] * y[0] * y[0])

    settings = dict(rtol=1e-8, atol=1e-10, lu_precision="fp64")
    hist, energy = rodas5P_solve(
        damped_oscillator,
        y0,
        t_span,
        params,
        save_hook=store_energy,
        hook_size=N_SAVE,
        **settings,
    )
    hist, energy = np.asarray(hist), np.asarray(energy)
    assert hist.shape == (3, N_SAVE, 2)
    expected = 0.5 * (hist[..., 1] ** 2 + params[:, :1] * hist[..., 0] ** 2)
    np.testing.assert_allclose(energy, expected, rtol=1e-13, atol=1e-15)
    # The undamped trajectory conserves energy; the hooked values show it.
    np.testing.assert_allclose(energy[2], energy[2, 0], rtol=1e-6)


def test_save_history_false_needs_a_hook():
    y0, t_span, params = _problem()
    with pytest.raises(ValueError, match="save_history=False"):
        rodas5P_solve(damped_oscillator, y0, t_span, params, save_history=False)


@requires_cuda
def test_hook_sees_the_joint_state_of_a_sensitivity_solve():
    """With ``sensitivity`` the hook reads ``[y, S]`` and can store a tangent.

    The energy ``E = (y_1^2 + p_0 y_0^2) / 2`` and its derivative along the
    parameter sensitivities, ``dE/dp_k = y_1 S_1k + p_0 y_0 S_0k + y_0^2
    delta_0k / 2``, stored per save, against the plain solve's JVP rule.
    """

    y0, t_span, params = _problem()
    # Out of step control, the joint solve takes the plain solve's steps.
    spec = SensitivitySpec(2, 2, wrt_y0=False, wrt_params=True, error_control=False)
    n_aug = spec.n_aug

    def store_energy_and_tangent(save_idx, y, t, p_row, acc):
        # The layout of _sensitivity: z[2 + k * 2 + r] = S[r, k].
        energy = 0.5 * (y[1] * y[1] + p_row[0] * y[0] * y[0])
        acc[3 * save_idx] = energy
        for k in range(2):
            d_energy = y[1] * y[2 + 2 * k + 1] + p_row[0] * y[0] * y[2 + 2 * k]
            if k == 0:
                d_energy += 0.5 * y[0] * y[0]
            acc[3 * save_idx + 1 + k] = d_energy

    settings = dict(rtol=1e-8, atol=1e-10, lu_precision="fp64")
    final, acc = rodas5P_solve(
        damped_oscillator,
        y0,
        t_span,
        params,
        save_hook=store_energy_and_tangent,
        hook_size=3 * N_SAVE,
        save_history=False,
        sensitivity=spec,
        **settings,
    )
    final, acc = np.asarray(final), np.asarray(acc).reshape(3, N_SAVE, 3)
    assert final.shape == (3, 1, n_aug)

    def energy_history(p):
        hist = rodas5P_solve(damped_oscillator, y0, t_span, p, **settings)
        return 0.5 * (hist[..., 1] ** 2 + p[:, None, 0] * hist[..., 0] ** 2)

    expected = np.asarray(energy_history(jnp.asarray(params)))
    d_expected = np.asarray(jax.jacfwd(energy_history)(jnp.asarray(params)))
    np.testing.assert_allclose(acc[:, :, 0], expected, rtol=1e-12, atol=1e-15)
    for k in range(2):
        np.testing.assert_allclose(
            acc[:, :, 1 + k],
            np.stack([d_expected[i, :, i, k] for i in range(3)]),
            rtol=1e-6,
            atol=1e-9,
        )

    # The augmented history itself is what the hook read.
    hist = rodas5P_solve(
        damped_oscillator, y0, t_span, params, sensitivity=spec, **settings
    )
    state, sens = split_augmented(hist, spec)
    np.testing.assert_allclose(np.asarray(state)[:, -1], final[:, 0, :2], atol=1e-13)
    assert sens.shape == (3, N_SAVE, 2, 2)


def test_sensitivity_spec_must_fit_the_problem():
    y0, t_span, params = _problem()
    with pytest.raises(ValueError, match="sensitivity spec"):
        rodas5P_solve(
            damped_oscillator,
            y0,
            t_span,
            params,
            sensitivity=SensitivitySpec(3, 2, wrt_y0=False, wrt_params=True),
        )


# --- quantity_hook: the hook built from a function of (y, t, p) --------------
def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])


def energy_alone(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]),)


def _quantity_histories(quantities, n_outputs):
    """The quantities over the plain solve's history, and their Jacobians."""

    y0, t_span, params = _problem()
    settings = dict(rtol=1e-8, atol=1e-10, lu_precision="fp64")

    def of(y0, p):
        hist = rodas5P_solve(damped_oscillator, y0, t_span, p, **settings)
        values = quantities((hist[..., 0], hist[..., 1]), None, (p[:, None, 0],))
        return jnp.stack(values[:n_outputs], axis=-1)

    y0_j, p_j = jnp.asarray(y0), jnp.asarray(params)
    d_y0, d_p = jax.jacfwd(of, argnums=(0, 1))(y0_j, p_j)
    rows = np.arange(3)
    # Each trajectory's quantities depend on its own row only.
    return (
        np.asarray(of(y0_j, p_j)),
        np.asarray(d_y0)[rows, :, :, rows],
        np.asarray(d_p)[rows, :, :, rows],
    )


@requires_cuda
@pytest.mark.parametrize(
    "quantities, n_outputs", [(energy_and_position, 2), (energy_alone, 1)]
)
def test_quantity_hook_stores_the_quantities(quantities, n_outputs):
    y0, t_span, params = _problem()
    hook = quantity_hook(quantities, n_outputs, 2, 2)
    _, acc = rodas5P_solve(
        damped_oscillator,
        y0,
        t_span,
        params,
        save_hook=hook,
        hook_size=n_outputs * N_SAVE,
        save_history=False,
        rtol=1e-8,
        atol=1e-10,
        lu_precision="fp64",
    )
    expected, _, _ = _quantity_histories(quantities, n_outputs)
    np.testing.assert_allclose(
        np.asarray(acc).reshape(3, N_SAVE, n_outputs), expected, rtol=1e-12, atol=1e-15
    )


@requires_cuda
@pytest.mark.parametrize(
    "quantities, n_outputs", [(energy_and_position, 2), (energy_alone, 1)]
)
@pytest.mark.parametrize(
    "wrt_y0, wrt_params, param_columns",
    [(False, True, None), (True, True, None), (False, True, (0,))],
)
def test_quantity_hook_stores_their_tangents(
    quantities, n_outputs, wrt_y0, wrt_params, param_columns
):
    """Values, then their tangent along each direction of the spec, per save.

    Against ``jax.jacfwd`` of the same quantities over the plain solve's
    history: the initial-state directions first, then the parameter columns.
    """

    y0, t_span, params = _problem()
    spec = SensitivitySpec(
        2,
        2,
        wrt_y0=wrt_y0,
        wrt_params=wrt_params,
        error_control=False,
        param_columns=param_columns,
    )
    hook = quantity_hook(quantities, n_outputs, 2, 2, spec)
    _, acc = rodas5P_solve(
        damped_oscillator,
        y0,
        t_span,
        params,
        save_hook=hook,
        hook_size=n_outputs * (1 + spec.n_sens) * N_SAVE,
        save_history=False,
        sensitivity=spec,
        rtol=1e-8,
        atol=1e-10,
        lu_precision="fp64",
    )
    acc = np.asarray(acc).reshape(3, N_SAVE, 1 + spec.n_sens, n_outputs)
    values, d_y0, d_p = _quantity_histories(quantities, n_outputs)
    np.testing.assert_allclose(acc[:, :, 0], values, rtol=1e-12, atol=1e-15)
    columns = [d_y0[..., r] for r in range(spec.n_y0_dirs)] + [
        d_p[..., c] for c in spec.param_seed_columns
    ]
    for k, expected in enumerate(columns):
        np.testing.assert_allclose(acc[:, :, 1 + k], expected, rtol=1e-6, atol=1e-9)


def test_quantity_hook_needs_a_spec_of_its_problem():
    with pytest.raises(ValueError, match="sensitivity spec"):
        quantity_hook(
            energy_alone,
            1,
            2,
            2,
            SensitivitySpec(3, 2, wrt_y0=False, wrt_params=True),
        )
    with pytest.raises(ValueError, match="at least one"):
        quantity_hook(energy_alone, 0, 2, 2)
