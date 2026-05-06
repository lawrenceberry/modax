"""Minimal Julia Rodas5P kernel regression test."""

import numpy as np

from reference.solvers.python.julia_rodas5 import solve as julia_rodas5_solve


def test_julia_rodas5p_kernel_stiff_scalar_regression():
    """Validate GPURodas5P on a 1D stiff scalar ODE (u' = -p*u)."""
    y0 = np.array([10.0], dtype=np.float64)
    t_span = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float64)
    params = np.array([[1.0], [0.5]], dtype=np.float64)

    ys = np.asarray(
        julia_rodas5_solve(
            "stiff_scalar",
            y0,
            t_span,
            params,
            ensemble_backend="EnsembleGPUKernel",
            first_step=1e-3,
            rtol=1e-7,
            atol=1e-9,
        )
    )

    expected = y0[0] * np.exp(-params[:, 0][:, None] * t_span[None, :])

    assert ys.shape == (params.shape[0], t_span.shape[0], y0.shape[0])
    np.testing.assert_allclose(ys[:, :, 0], expected, rtol=1e-3, atol=1e-5)


def test_julia_rodas5p_kernel_batched_y0_regression():
    """Batched GPUKernel u0 values must stay static arrays inside Julia."""
    y0 = np.array([[10.0], [20.0]], dtype=np.float64)
    t_span = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float64)
    params = np.array([[1.0], [0.5]], dtype=np.float64)

    ys = np.asarray(
        julia_rodas5_solve(
            "stiff_scalar",
            y0,
            t_span,
            params,
            ensemble_backend="EnsembleGPUKernel",
            first_step=1e-3,
            rtol=1e-7,
            atol=1e-9,
        )
    )

    expected = y0[:, 0][:, None] * np.exp(-params[:, 0][:, None] * t_span[None, :])

    assert ys.shape == (params.shape[0], t_span.shape[0], y0.shape[1])
    np.testing.assert_allclose(ys[:, :, 0], expected, rtol=1e-3, atol=1e-5)
