"""General-purpose Rosenbrock23 ensemble ODE solver with Pallas GPU kernel.

Implements the GPURosenbrock23 algorithm from DiffEqGPU.jl:
2-stage Rosenbrock method (order 2) with 3rd-stage error estimator
and PI step size controller. Pallas/Triton custom GPU kernel
with 32 trajectories per block.

Reference: https://github.com/SciML/DiffEqGPU.jl
"""
# TODOS:
# 1. Rewrite the rodas5_custom_kernel solver to match how the rosenbrock23 solver works i.e. batching 32 trajectories in one warp, masking out trajectories that are done.

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

# The number of ODE trajectories solved per block. One CUDA warp runs on one of 4 blocks within a streaming
# multi-processor (SM). It always executes 32 threads in lockstep, even if the actual hardware in the block
# doesn't have enough cores to execute them all in parallel (e.g. if there are only 2 FP64 cores, the 32
# threads will take 16 clock cycles to execute the warp). Therefore, if this value is smaller than 32, we
# don't fully utilize the available operations/s in the block, but if it's larger than 32, we are batching
# more ODEs at once, and will end up with more "warp divergence" where some ODE trajectories have finished
# before others, so operations/s go unused as some ODEs have finished and no longer have any work to do.
_BLOCK = 32


def _pad_cols_pow2(n_cols):
    """Return the next power of 2 >= n_cols."""
    return 1 << (n_cols - 1).bit_length()


# ---------------------------------------------------------------------------
# General Rosenbrock23 step (works with any ODE via tuples of scalars)
# ---------------------------------------------------------------------------


def _make_rb23_step(ode_fn, n_vars):
    """Create a Rosenbrock23 step function for a general ODE.

    All state/parameter components are separate (_BLOCK,) arrays accessed via
    Python tuple indexing, avoiding JAX array slicing (unsupported in Pallas/Triton).
    Python loops over n_vars are unrolled at JAX trace time.

    Args:
        ode_fn: ODE right-hand side (y_tuple, p_tuple) -> dy_tuple where each
            element is a scalar-like array. Must use only element-wise operations.
        n_vars: Number of state variables (used for loop unrolling).
    """

    def step(y, p, dt):
        """Compute one Rosenbrock23 step.

        Args:
            y: tuple of n_vars (_BLOCK,) arrays — current state.
            p: tuple of n_params (_BLOCK,) arrays — parameters.
            dt: (_BLOCK,) step size.

        Returns:
            (u, err) where u and err are tuples of n_vars (_BLOCK,) arrays.
        """
        gamma = dt * _d

        def F(y):
            return ode_fn(y, p)

        F0 = F(y)

        # Jacobian via forward-mode AD (one jvp per column)
        ones = y[0] * 0.0 + 1.0
        J_cols = []  # J_cols[j][i] = dF_i / dy_j
        for j in range(n_vars):

            def f_j(yj, j=j):
                y_mod = tuple(yj if k == j else y[k] for k in range(n_vars))
                return F(y_mod)

            _, col = jax.jvp(f_j, (y[j],), (ones,))
            J_cols.append(col)

        # W = I - gamma * J (nested lists of (_BLOCK,) arrays)
        W = [[None] * n_vars for _ in range(n_vars)]
        for i in range(n_vars):
            for j in range(n_vars):
                w_ij = -gamma * J_cols[j][i]
                if i == j:
                    w_ij = w_ij + 1.0
                W[i][j] = w_ij

        # LU decomposition (Doolittle, no pivoting)
        U = [[W[i][j] for j in range(n_vars)] for i in range(n_vars)]
        L = [[y[0] * 0.0 for _ in range(n_vars)] for _ in range(n_vars)]
        for j in range(n_vars):
            for i in range(j + 1, n_vars):
                L[i][j] = U[i][j] / U[j][j]
                for k in range(j, n_vars):
                    U[i][k] = U[i][k] - L[i][j] * U[j][k]

        def S(b):
            """Solve W @ x = b via LU forward/back substitution."""
            z = list(b)
            for i in range(n_vars):
                for j in range(i):
                    z[i] = z[i] - L[i][j] * z[j]
            x = list(z)
            for i in range(n_vars - 1, -1, -1):
                for j in range(i + 1, n_vars):
                    x[i] = x[i] - U[i][j] * x[j]
                x[i] = x[i] / U[i][i]
            return tuple(x)

        # Stage 1: k1 = W⁻¹ * F₀
        k1 = S(F0)

        # Stage 2: k2 = W⁻¹ * (F₁ - k1) + k1
        dto2 = dt / 2.0
        y_mid = tuple(y[i] + dto2 * k1[i] for i in range(n_vars))
        F1 = F(y_mid)
        s = S(tuple(F1[i] - k1[i] for i in range(n_vars)))
        k2 = tuple(s[i] + k1[i] for i in range(n_vars))

        # Solution: u = y + dt * k2
        u = tuple(y[i] + dt * k2[i] for i in range(n_vars))

        # Error estimation (3rd stage)
        F2 = F(u)
        rhs3 = tuple(
            F2[i] - _e32 * (k2[i] - F1[i]) - 2.0 * (k1[i] - F0[i])
            for i in range(n_vars)
        )
        k3 = S(rhs3)

        dto6 = dt / 6.0
        err = tuple(dto6 * (k1[i] - 2.0 * k2[i] + k3[i]) for i in range(n_vars))

        return u, err

    return step


# ---------------------------------------------------------------------------
# Pallas/Triton custom kernel (general ODE, 32 trajectories per block)
# ---------------------------------------------------------------------------


def make_solver(ode_fn):
    """Create a Pallas ensemble solver for the given ODE.

    Args:
        ode_fn: ODE right-hand side function with signature
            (y, p) -> dy/dt
            where y and p are tuples of scalar-like values and the return
            is a tuple of the same length as y. The function must use only
            element-wise operations so it can run inside a Pallas/Triton
            kernel. In particular, it must NOT use JAX array indexing
            (y[i] must be Python tuple indexing, not JAX lax.slice) and
            must NOT return a jnp.array — Pallas/Triton requires all
            intermediate tensors to have a power-of-2 number of elements,
            so jnp.array([a, b, c]) with 3 components would be rejected.

    Returns:
        solve_ensemble_pallas function.

    Example::

        def robertson(y, p):
            return (
                -p[0] * y[0] + p[1] * y[1] * y[2],
                 p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1]**2,
                 p[2] * y[1]**2,
            )

        solve = make_solver(robertson)
        results = solve(y0_batch, (0.0, 1e5), params_batch)
    """

    @functools.partial(
        jax.jit,
        static_argnames=(
            "n_pad",
            "p_cols",
            "y_cols",
            "n_vars",
            "n_params",
            "tf",
            "dt0",
            "r_tol",
            "a_tol",
            "ms",
        ),
    )
    def _rb23_pallas_solve(
        params_arr,
        y0_arr,
        *,
        n_pad,
        p_cols,
        y_cols,
        n_vars,
        n_params,
        tf,
        dt0,
        r_tol,
        a_tol,
        ms,
    ):
        step_fn = _make_rb23_step(ode_fn, n_vars)

        def kernel_body(params_ref, y0_ref, y_ref):
            p = tuple(params_ref.at[:, i][...] for i in range(n_params))
            y = tuple(y0_ref.at[:, i][...] for i in range(n_vars))

            z = p[0] * 0.0
            t = z + 0.0
            dt_v = z + dt0
            qold = z + _qoldinit

            def cond_fn(state):
                return (jnp.min(state[0]) < tf) & (state[-1] < ms)

            def body_fn(state):
                t = state[0]
                y = state[1 : 1 + n_vars]
                dt_v = state[1 + n_vars]
                qold = state[2 + n_vars]
                n = state[3 + n_vars]

                active = t < tf
                dt_use = jnp.maximum(jnp.minimum(dt_v, tf - t), 1e-30)

                u, err = step_fn(y, p, dt_use)

                sc = tuple(
                    a_tol + r_tol * jnp.maximum(jnp.abs(y[i]), jnp.abs(u[i]))
                    for i in range(n_vars)
                )
                err_sq = sum((err[i] / sc[i]) ** 2 for i in range(n_vars))
                EEst = jnp.sqrt(err_sq / n_vars)

                accept = (EEst <= 1.0) & ~jnp.isnan(EEst)
                mask = active & accept

                t_new = jnp.where(mask, t + dt_use, t)
                y_new = tuple(jnp.where(mask, u[i], y[i]) for i in range(n_vars))

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

                return (t_new,) + y_new + (new_dt, new_qold, n + 1)

            init = (t,) + y + (dt_v, qold, jnp.int32(0))
            final = jax.lax.while_loop(cond_fn, body_fn, init)

            result = final[1 : 1 + n_vars]
            for i in range(n_vars):
                y_ref.at[:, i][...] = result[i]

        p_bs = pl.BlockSpec((_BLOCK, p_cols), lambda i: (i, 0))
        y_bs = pl.BlockSpec((_BLOCK, y_cols), lambda i: (i, 0))
        return pl.pallas_call(
            kernel_body,
            out_shape=jax.ShapeDtypeStruct((n_pad, y_cols), jnp.float64),
            grid=(n_pad // _BLOCK,),
            in_specs=(p_bs, y_bs),
            out_specs=y_bs,
            # num_warps: specifies how many warps (groups of 32 threads) are
            # assigned to run on each block of the SM. Increasing num_warps
            # means the GPU can switch between warps to hide SRAM memory latency,
            # at the cost of higher register usage.
            # num_stages: specifies how many kernel data blocks (p_bs, y_bs)will
            # be loaded into SRAM from HBM at once. Increasing num_stages means the
            # GPU does not need to wait for new data from HBM to be loaded into SRAM
            # before running the kernel, at the cost of higher SRAM usage.
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
        )(params_arr, y0_arr)

    def solve_ensemble_pallas(
        y0_batch,
        t_span,
        params_batch,
        *,
        rtol=1e-6,
        atol=1e-8,
        first_step=None,
        max_steps=100000,
    ):
        """Solve ODE ensemble using a Pallas/Triton custom GPU kernel.

        Args:
            y0_batch: Per-trajectory initial conditions, shape (N, n_vars).
            t_span: (t0, tf) time interval.
            params_batch: Per-trajectory parameters, shape (N, n_params).
        """
        N = params_batch.shape[0]
        n_vars = y0_batch.shape[1]
        n_params = params_batch.shape[1]

        # Verify ODE function dimensions via JAX abstract evaluation
        y_trace = tuple(jax.ShapeDtypeStruct((), jnp.float64) for _ in range(n_vars))
        p_trace = tuple(jax.ShapeDtypeStruct((), jnp.float64) for _ in range(n_params))
        out_trace = jax.eval_shape(ode_fn, y_trace, p_trace)
        assert len(out_trace) == n_vars, (
            f"ODE function returns {len(out_trace)} components but y0 has {n_vars}"
        )

        N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK
        tf = float(t_span[1])
        dt0 = float(
            first_step if first_step is not None else (tf - float(t_span[0])) * 1e-6
        )

        p_cols = _pad_cols_pow2(n_params)
        y_cols = _pad_cols_pow2(n_vars)

        params_arr = jnp.pad(params_batch, ((0, N_pad - N), (0, p_cols - n_params)))
        y0_arr = jnp.pad(y0_batch, ((0, N_pad - N), (0, y_cols - n_vars)))

        y_out = _rb23_pallas_solve(
            params_arr,
            y0_arr,
            n_pad=N_pad,
            p_cols=p_cols,
            y_cols=y_cols,
            n_vars=n_vars,
            n_params=n_params,
            tf=tf,
            dt0=dt0,
            r_tol=float(rtol),
            a_tol=float(atol),
            ms=int(max_steps),
        )

        return y_out[:N, :n_vars]

    return solve_ensemble_pallas
