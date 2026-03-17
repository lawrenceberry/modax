"""GPURosenbrock23 solver with Pallas GPU kernel.

Implements the same GPURosenbrock23 algorithm from DiffEqGPU.jl:
2-stage Rosenbrock method (order 2) with 3rd-stage error estimator
and PI step size controller. Robertson-specific Pallas/Triton kernel
with 32 trajectories per block.

Reference: https://github.com/SciML/DiffEqGPU.jl
"""
# TODOS:
# 1. Remove Cramer's rule and use a more general jax LU solve
# 2. Instead of 3 1D blocks, use a single 2D block
# 3. Add support for non-Robertson problems i.e. general ODEs, using jax.grad or jax.jacfwd for Jacobian and a more general kernel body
# 4. Rewrite the rodas5_custom_kernel solver to match how the rosenbrock23 solver works i.e. batching 32 trajectories in one warp, masking out trajectories that are done.

import functools
import math

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax._src.pallas.triton import core as pltriton
from jax.experimental import pallas as pl

# Rosenbrock23 coefficients
_d = 1.0 / (2.0 + math.sqrt(2.0))  # ≈ 0.2928932188134524
_e32 = 6.0 + math.sqrt(2.0)  # ≈ 7.414213562373095

# PI step size controller (order=2)
_beta1 = 7.0 / 20.0  # 7 / (10 * order)
_beta2 = 2.0 / 10.0  # 2 / (5 * order)
_qmax = 10.0
_qmin = 0.2  # 1/5
_sc_gamma = 0.9  # 9/10
_qoldinit = 1.0e-4

_BLOCK = 32


# ---------------------------------------------------------------------------
# Robertson-specific Rosenbrock23 step (fully scalar, works on vectors too)
# ---------------------------------------------------------------------------


def _robertson_rb23_step(y0, y1, y2, p0, p1, p2, dt):
    """Single Rosenbrock23 step for Robertson. Returns (u0, u1, u2, e0, e1, e2)."""
    gamma = dt * _d

    # Robertson RHS
    def F(s0, s1, s2):
        return (
            -p0 * s0 + p1 * s1 * s2,
            p0 * s0 - p1 * s1 * s2 - p2 * s1 * s1,
            p2 * s1 * s1,
        )

    # Jacobian of F at (y0, y1, y2) via forward-mode AD (jax.jvp)
    ones = jnp.ones_like(y0)
    _, (j00, j10, j20) = jax.jvp(lambda s: F(s, y1, y2), (y0,), (ones,))
    _, (j01, j11, j21) = jax.jvp(lambda s: F(y0, s, y2), (y1,), (ones,))
    _, (j02, j12, j22) = jax.jvp(lambda s: F(y0, y1, s), (y2,), (ones,))

    # W = I - gamma * J
    w00 = 1.0 - gamma * j00
    w01 = -gamma * j01
    w02 = -gamma * j02
    w10 = -gamma * j10
    w11 = 1.0 - gamma * j11
    w12 = -gamma * j12
    w20 = -gamma * j20
    w21 = -gamma * j21
    w22 = 1.0 - gamma * j22

    # LU decomposition of W (Doolittle, no pivoting)
    u00 = w00
    u01 = w01
    u02 = w02
    l10 = w10 / u00
    u11 = w11 - l10 * u01
    u12 = w12 - l10 * u02
    l20 = w20 / u00
    l21 = (w21 - l20 * u01) / u11
    u22 = w22 - l20 * u02 - l21 * u12

    # Solve W @ x = v via LU: forward substitution then back substitution
    def S(v0, v1, v2):
        z0 = v0
        z1 = v1 - l10 * z0
        z2 = v2 - l20 * z0 - l21 * z1
        x2 = z2 / u22
        x1 = (z1 - u12 * x2) / u11
        x0 = (z0 - u01 * x1 - u02 * x2) / u00
        return x0, x1, x2

    # Stage 1: k1 = W⁻¹ * F₀  (dT=0 for autonomous system)
    F00, F01, F02 = F(y0, y1, y2)
    k10, k11, k12 = S(F00, F01, F02)

    # Stage 2: k2 = W⁻¹ * (F₁ - k1) + k1
    dto2 = dt / 2.0
    F10, F11, F12 = F(y0 + dto2 * k10, y1 + dto2 * k11, y2 + dto2 * k12)
    s0, s1, s2 = S(F10 - k10, F11 - k11, F12 - k12)
    k20 = s0 + k10
    k21 = s1 + k11
    k22 = s2 + k12

    # Solution: u = uprev + dt * k2
    u0 = y0 + dt * k20
    u1 = y1 + dt * k21
    u2 = y2 + dt * k22

    # Error estimation (3rd stage): k3 = W⁻¹ * (F₂ - e32*(k2-F₁) - 2*(k1-F₀))
    F20, F21, F22 = F(u0, u1, u2)
    r0 = F20 - _e32 * (k20 - F10) - 2.0 * (k10 - F00)
    r1 = F21 - _e32 * (k21 - F11) - 2.0 * (k11 - F01)
    r2 = F22 - _e32 * (k22 - F12) - 2.0 * (k12 - F02)
    k30, k31, k32 = S(r0, r1, r2)

    # Error = dt/6 * (k1 - 2*k2 + k3)
    dto6 = dt / 6.0
    e0 = dto6 * (k10 - 2.0 * k20 + k30)
    e1 = dto6 * (k11 - 2.0 * k21 + k31)
    e2 = dto6 * (k12 - 2.0 * k22 + k32)

    return u0, u1, u2, e0, e1, e2


# ---------------------------------------------------------------------------
# Public API: single solve (generic)
# ---------------------------------------------------------------------------


def solve(f, y0, t_span, *, rtol=1e-8, atol=1e-10, first_step=None, max_steps=100000):
    """Solve a stiff autonomous ODE using Rosenbrock23."""
    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    t0, tf = t_span
    dt0 = jnp.float64(first_step if first_step is not None else (tf - t0) * 1e-6)
    jac_fn = jax.jacfwd(f)

    def cond_fn(state):
        t, _, _, n, _ = state
        return (t < tf) & (n < max_steps)

    def body_fn(state):
        t, y, dt, n, qold = state
        dt = jnp.minimum(dt, tf - t)

        J = jac_fn(y)
        gamma = dt * _d
        W = jnp.eye(3) - gamma * J
        lu, piv = jax.scipy.linalg.lu_factor(W)
        S = lambda v: jax.scipy.linalg.lu_solve((lu, piv), v)

        F0 = f(y)
        k1 = S(F0)

        F1 = f(y + (dt / 2.0) * k1)
        k2 = S(F1 - k1) + k1

        u = y + dt * k2

        F2 = f(u)
        k3 = S(F2 - _e32 * (k2 - F1) - 2.0 * (k1 - F0))

        err_vec = (dt / 6.0) * (k1 - 2.0 * k2 + k3)
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(u))
        EEst = jnp.sqrt(jnp.mean((err_vec / scale) ** 2))

        accept = (EEst <= 1.0) & ~jnp.isnan(EEst)
        t_new = jnp.where(accept, t + dt, t)
        y_out = jnp.where(accept, u, y)

        # PI controller
        q11 = jnp.where(EEst == 0.0, 1e-18, EEst) ** _beta1
        q_accept = jnp.clip(q11 / (qold**_beta2) / _sc_gamma, 1.0 / _qmax, 1.0 / _qmin)
        q_reject = jnp.minimum(1.0 / _qmin, q11 / _sc_gamma)
        dtnew = jnp.where(accept, dt / q_accept, dt / q_reject)
        qold_new = jnp.where(accept, jnp.maximum(EEst, _qoldinit), qold)

        return (t_new, y_out, dtnew, n + 1, qold_new)

    _, final_y, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (jnp.float64(t0), y0_arr, dt0, jnp.int32(0), jnp.float64(_qoldinit)),
    )
    return final_y


def solve_ensemble(
    f,
    y0,
    t_span,
    params_batch,
    *,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
):
    """Solve ensemble using vmap."""

    def _solve_one(params):
        return solve(
            lambda y: f(y, params),
            y0,
            t_span,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
        )

    return jax.jit(jax.vmap(_solve_one))(params_batch)


# ---------------------------------------------------------------------------
# Pallas/Triton custom kernel (Robertson-specific, 32 trajectories per block)
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    static_argnames=("n_pad", "tf", "dt0", "r_tol", "a_tol", "y00", "y01", "y02", "ms"),
)
def _rb23_pallas_solve(
    p0_arr, p1_arr, p2_arr, *, n_pad, tf, dt0, r_tol, a_tol, y00, y01, y02, ms
):
    """JIT-compiled Pallas kernel call (cached across invocations)."""

    def kernel_body(p0_ref, p1_ref, p2_ref, y0_ref, y1_ref, y2_ref):
        p0 = p0_ref[...]
        p1 = p1_ref[...]
        p2 = p2_ref[...]

        z = p0 * 0.0
        t = z + 0.0
        s0 = z + y00
        s1 = z + y01
        s2 = z + y02
        dt_v = z + dt0
        qold = z + _qoldinit

        def cond_fn(state):
            t, _, _, _, _, _, n = state
            return (jnp.min(t) < tf) & (n < ms)

        def body_fn(state):
            t, s0, s1, s2, dt_v, qold, n = state
            active = t < tf
            dt_use = jnp.maximum(jnp.minimum(dt_v, tf - t), 1e-30)

            n0, n1, n2, e0, e1, e2 = _robertson_rb23_step(
                s0, s1, s2, p0, p1, p2, dt_use
            )

            sc0 = a_tol + r_tol * jnp.maximum(jnp.abs(s0), jnp.abs(n0))
            sc1 = a_tol + r_tol * jnp.maximum(jnp.abs(s1), jnp.abs(n1))
            sc2 = a_tol + r_tol * jnp.maximum(jnp.abs(s2), jnp.abs(n2))
            err_sq = (e0 / sc0) ** 2 + (e1 / sc1) ** 2 + (e2 / sc2) ** 2
            EEst = jnp.sqrt(err_sq / 3.0)

            accept = (EEst <= 1.0) & ~jnp.isnan(EEst)
            mask = active & accept

            t_new = jnp.where(mask, t + dt_use, t)
            o0 = jnp.where(mask, n0, s0)
            o1 = jnp.where(mask, n1, s1)
            o2 = jnp.where(mask, n2, s2)

            safe_EEst = jnp.where(
                jnp.isnan(EEst) | (EEst > 1e18),
                1e18,
                jnp.where(EEst == 0.0, 1e-18, EEst),
            )
            q11 = safe_EEst**_beta1
            q_accept = jnp.clip(
                q11 / (qold**_beta2) / _sc_gamma,
                1.0 / _qmax,
                1.0 / _qmin,
            )
            q_reject = jnp.minimum(1.0 / _qmin, q11 / _sc_gamma)

            new_dt = jnp.where(accept, dt_use / q_accept, dt_use / q_reject)
            new_dt = jnp.where(active, new_dt, dt_v)
            new_qold = jnp.where(mask, jnp.maximum(EEst, _qoldinit), qold)

            return (t_new, o0, o1, o2, new_dt, new_qold, n + 1)

        _, r0, r1, r2, _, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (t, s0, s1, s2, dt_v, qold, jnp.int32(0))
        )

        y0_ref[...] = r0
        y1_ref[...] = r1
        y2_ref[...] = r2

    bs = pl.BlockSpec((_BLOCK,), lambda i: (i,))
    return pl.pallas_call(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((n_pad,), jnp.float64),
            jax.ShapeDtypeStruct((n_pad,), jnp.float64),
            jax.ShapeDtypeStruct((n_pad,), jnp.float64),
        ],
        grid=(n_pad // _BLOCK,),
        in_specs=[bs, bs, bs],
        out_specs=[bs, bs, bs],
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
    )(p0_arr, p1_arr, p2_arr)


def solve_ensemble_pallas(
    y0,
    t_span,
    params_batch,
    *,
    rtol=1e-6,
    atol=1e-8,
    first_step=None,
    max_steps=100000,
):
    """Solve Robertson ensemble using a Pallas/Triton custom GPU kernel.

    Uses GPURosenbrock23 algorithm matching DiffEqGPU.jl: 2-stage Rosenbrock
    with 3rd-stage error estimator and PI step size controller.
    32 trajectories per block with per-trajectory masking.
    """
    N = params_batch.shape[0]
    N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK
    tf = float(t_span[1])
    dt0 = float(
        first_step if first_step is not None else (tf - float(t_span[0])) * 1e-6
    )

    p0_arr = jnp.pad(params_batch[:, 0], (0, N_pad - N))
    p1_arr = jnp.pad(params_batch[:, 1], (0, N_pad - N))
    p2_arr = jnp.pad(params_batch[:, 2], (0, N_pad - N))

    y0_out, y1_out, y2_out = _rb23_pallas_solve(
        p0_arr,
        p1_arr,
        p2_arr,
        n_pad=N_pad,
        tf=tf,
        dt0=dt0,
        r_tol=float(rtol),
        a_tol=float(atol),
        y00=float(y0[0]),
        y01=float(y0[1]),
        y02=float(y0[2]),
        ms=int(max_steps),
    )

    return jnp.stack([y0_out[:N], y1_out[:N], y2_out[:N]], axis=1)
