"""Shared helpers for custom-kernel Tsit5 variants."""

from __future__ import annotations

import numpy as np

# fmt: off
C2 = 161.0 / 1000.0
C3 = 327.0 / 1000.0
C4 = 9.0 / 10.0
C5 = 0.9800255409045097
C6 = 1.0
C7 = 1.0

A21 = 161.0 / 1000.0

A31 = -0.008480655492356989
A32 = 0.335480655492357

A41 = 2.8971530571054935
A42 = -6.359448489975075
A43 = 4.3622954328695815

A51 = 5.325864828439257
A52 = -11.748883564062828
A53 = 7.4955393428898365
A54 = -0.09249506636175525

A61 = 5.86145544294642
A62 = -12.92096931784711
A63 = 8.159367898576159
A64 = -0.071584973281401
A65 = -0.028269050394068383

A71 = 0.09646076681806523
A72 = 0.01
A73 = 0.4798896504144996
A74 = 1.379008574103742
A75 = -3.2900695154360807
A76 = 2.324710524099774

B1 = A71
B2 = A72
B3 = A73
B4 = A74
B5 = A75
B6 = A76

E1 = 0.0017800620525794302
E2 = 0.000816434459656747
E3 = -0.007880878010261985
E4 = 0.14471100717326298
E5 = -0.5823571654525553
E6 = 0.45808210592918695
E7 = -1.0 / 66.0
# fmt: on

SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 10.0


def normalize_inputs(y0, t_span, params, first_step):
    y0_in = np.asarray(y0, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    times = np.asarray(t_span, dtype=np.float64)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
        params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
    else:
        n = y0_in.shape[0]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )

    if y0_arr.ndim != 2:
        raise ValueError("custom-kernel Tsit5 expects y0 shape (N, n_vars)")
    if params_arr.ndim != 2:
        raise ValueError("custom-kernel Tsit5 expects params shape (N, n_params)")
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("t_span must be a 1-D array with at least two save times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_span must be strictly increasing")

    dt0 = (
        np.float64(first_step)
        if first_step is not None
        else np.float64((times[-1] - times[0]) * 1e-6)
    )
    return y0_arr, times, params_arr, dt0


def numpy_stats(accepted_steps, rejected_steps, loop_steps):
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": np.ones_like(loop_steps, dtype=np.int32),
    }
