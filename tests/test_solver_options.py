"""Solve options that are neither the tableau nor the sparsity: block width,
the ``df/dt`` stage term, and per-trajectory save times.

The linear solve is not one of them any more. It is the default dense LU, or a
sparse direct solver compiled from ``sparsity`` -- there is no third choice and
nothing to pass. ``test_sparse_direct.py`` covers the sparse path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

import modax
from modax._options import DEFAULT_TRAJECTORIES_PER_BLOCK, KernelOptions

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")


# Non-autonomous: dy/dt = -lam*y + forcing*t, y(0) = 0. Without the
# dt*d_i*df/dt stage correction a Rosenbrock-W method drops below order 5 here,
# so this is what checks df/dt reaches the stages.
LAMBDA = 10.0
FORCING = 5.0


def forced_decay(y, t, p):
    return (-p[0] * y[0] + p[1] * t,)


def _exact_forced(t):
    return FORCING * (LAMBDA * t - 1.0 + np.exp(-LAMBDA * t)) / LAMBDA**2


def test_trajectories_per_block():
    """A warp by default; nothing on chip bounds an explicit request."""
    assert KernelOptions().trajectories_per_block == DEFAULT_TRAJECTORIES_PER_BLOCK
    assert DEFAULT_TRAJECTORIES_PER_BLOCK == 32
    assert KernelOptions(trajectories_per_block=64).trajectories_per_block == 64
    with pytest.raises(ValueError, match="must be positive"):
        KernelOptions(trajectories_per_block=0)
    with pytest.raises(ValueError, match="must be positive"):
        modax.solve(
            forced_decay,
            np.zeros(1),
            np.linspace(0.0, 1.0, 3),
            np.ones(2),
            trajectories_per_block=0,
        )


@requires_cuda
def test_the_time_derivative_term_reaches_the_stages():
    t_span = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    y0 = np.zeros((1, 1), dtype=np.float64)
    params = np.array([[LAMBDA, FORCING]], dtype=np.float64)
    result = np.asarray(
        modax.solve(
            forced_decay,
            y0,
            t_span,
            params,
            method=modax.Rodas5P(lu_precision="fp64"),
            rtol=1e-10,
            atol=1e-12,
            first_step=1e-4,
        )
    )[0, :, 0]
    assert np.allclose(result, _exact_forced(t_span), rtol=1e-8, atol=1e-10)


def _exact_from(t, t0):
    """The forced decay from y(t0) = 0."""
    return _exact_forced(t) - _exact_forced(t0) * np.exp(-LAMBDA * (t - t0))


# Two trajectories on grids of their own: different starts, ends and spacing.
GRIDS = np.array([np.linspace(0.0, 1.0, 11), np.linspace(0.3, 0.8, 11)])
PARAMS2 = np.array([[LAMBDA, FORCING]] * 2)


@requires_cuda
@pytest.mark.parametrize(
    "method",
    [modax.Rodas5P(lu_precision="fp64"), modax.Tsit5(), modax.Auto()],
    ids=["rodas5p", "tsit5", "auto"],
)
def test_each_trajectory_runs_on_its_own_save_times(method):
    """A row of t_span is that trajectory's start, saves and end."""
    result = np.asarray(
        modax.solve(
            forced_decay,
            np.zeros((2, 1)),
            GRIDS,
            PARAMS2,
            method=method,
            rtol=1e-10,
            atol=1e-12,
            first_step=1e-4,
        )
    )
    for row, grid in enumerate(GRIDS):
        exact = _exact_from(grid, grid[0])
        assert np.allclose(result[row, :, 0], exact, rtol=1e-7, atol=1e-9), row


@requires_cuda
def test_per_trajectory_save_times_set_the_ensemble_size():
    """With 1-D y0 and params, t_span's rows make the ensemble."""
    shared = np.asarray(
        modax.solve(
            forced_decay, np.zeros(1), GRIDS, PARAMS2[0], rtol=1e-10, atol=1e-12
        )
    )
    assert shared.shape == (2, 11, 1)
    with pytest.raises(ValueError, match="rows"):
        modax.solve(forced_decay, np.zeros((3, 1)), GRIDS, PARAMS2[0])


@requires_cuda
def test_vmap_over_t_span_gives_each_trajectory_its_grid():
    def one(grid):
        return modax.solve(
            forced_decay, np.zeros(1), grid, PARAMS2[0], rtol=1e-10, atol=1e-12
        )

    mapped = np.asarray(jax.vmap(one)(jnp.asarray(GRIDS)))[:, 0]
    direct = np.asarray(
        modax.solve(
            forced_decay, np.zeros(1), GRIDS, PARAMS2[0], rtol=1e-10, atol=1e-12
        )
    )
    np.testing.assert_array_equal(mapped, direct)
