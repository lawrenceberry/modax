"""General-purpose Rodas5 ensemble ODE solver with Pallas GPU kernel.

Implements the Rodas5 algorithm — a Rosenbrock method of order 5(4) for stiff
ODEs — using the same general-purpose make_solver pattern as
rosenbrock23_custom_kernel.py.

The W-transformed variant is from Di Marzo (1993) and DISCO-EB (Hahn).
8 stages with adaptive step size control. Pallas/Triton custom GPU kernel
with 32 trajectories per block.

Reference: https://github.com/SciML/DiffEqGPU.jl
"""

import functools

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax._src.pallas.triton import core as pltriton
from jax.experimental import pallas as pl

# fmt: off
# Rodas5 W-transformed coefficients
_gamma = 0.19

_a21 = 2.0
_a31 = 3.040894194418781;  _a32 = 1.041747909077569
_a41 = 2.576417536461461;  _a42 = 1.622083060776640;  _a43 = -0.9089668560264532
_a51 = 2.760842080225597;  _a52 = 1.446624659844071;  _a53 = -0.3036980084553738;  _a54 = 0.2877498600325443
_a61 = -14.09640773051259; _a62 = 6.925207756232704;  _a63 = -41.47510893210728;   _a64 = 2.343771018586405;  _a65 = 24.13215229196062

_C21 = -10.31323885133993
_C31 = -21.04823117650003; _C32 = -7.234992135176716
_C41 = 32.22751541853323;  _C42 = -4.943732386540191;  _C43 = 19.44922031041879
_C51 = -20.69865579590063; _C52 = -8.816374604402768;  _C53 = 1.260436877740897;   _C54 = -0.7495647613787146
_C61 = -46.22004352711257; _C62 = -17.49534862857472;  _C63 = -289.6389582892057;  _C64 = 93.60855400400906;  _C65 = 318.3822534212147
_C71 = 34.20013733472935;  _C72 = -14.15535402717690;  _C73 = 57.82335640988400;   _C74 = 25.83362985412365;  _C75 = 1.408950972071624;  _C76 = -6.551835421242162
_C81 = 42.57076742291101;  _C82 = -13.80770672017997;  _C83 = 93.98938432427124;   _C84 = 18.77919633714503;  _C85 = -31.58359187223370;  _C86 = -6.685968952921985;  _C87 = -5.810979938412932
# fmt: on

# The number of ODE trajectories solved per block.
_BLOCK = 32


def _pad_cols_pow2(n_cols):
    """Return the next power of 2 >= n_cols."""
    return 1 << (n_cols - 1).bit_length()


# ---------------------------------------------------------------------------
# General Rodas5 step (works with any ODE via tuples of scalars)
# ---------------------------------------------------------------------------


def _make_rodas5_step(ode_fn, n_vars):
    """Create a Rodas5 step function for a general ODE.

    All state/parameter components are separate (_BLOCK,) arrays accessed via
    Python tuple indexing. Python loops over n_vars are unrolled at JAX trace
    time.

    Args:
        ode_fn: ODE right-hand side (y_tuple, p_tuple) -> dy_tuple.
        n_vars: Number of state variables.
    """

    def step(y, p, dt):
        """Compute one Rodas5 step.

        Args:
            y: tuple of n_vars (_BLOCK,) arrays — current state.
            p: tuple of n_params (_BLOCK,) arrays — parameters.
            dt: (_BLOCK,) step size.

        Returns:
            (y_new, err) where y_new and err are tuples of n_vars (_BLOCK,) arrays.
        """
        dtgamma = dt * _gamma
        inv_dt = 1.0 / dt

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

        # W = I/(dt*gamma) - J
        d = 1.0 / dtgamma
        W = [[None] * n_vars for _ in range(n_vars)]
        for i in range(n_vars):
            for j in range(n_vars):
                w_ij = -J_cols[j][i]
                if i == j:
                    w_ij = w_ij + d
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

        # Stage 1: k1 = W^{-1} * F0
        k1 = S(F0)

        # Stage 2
        u = tuple(y[i] + _a21 * k1[i] for i in range(n_vars))
        Fu = F(u)
        rhs = tuple(
            Fu[i] + _C21 * k1[i] * inv_dt for i in range(n_vars)
        )
        k2 = S(rhs)

        # Stage 3
        u = tuple(
            y[i] + _a31 * k1[i] + _a32 * k2[i] for i in range(n_vars)
        )
        Fu = F(u)
        rhs = tuple(
            Fu[i] + (_C31 * k1[i] + _C32 * k2[i]) * inv_dt
            for i in range(n_vars)
        )
        k3 = S(rhs)

        # Stage 4
        u = tuple(
            y[i] + _a41 * k1[i] + _a42 * k2[i] + _a43 * k3[i]
            for i in range(n_vars)
        )
        Fu = F(u)
        rhs = tuple(
            Fu[i] + (_C41 * k1[i] + _C42 * k2[i] + _C43 * k3[i]) * inv_dt
            for i in range(n_vars)
        )
        k4 = S(rhs)

        # Stage 5
        u = tuple(
            y[i] + _a51 * k1[i] + _a52 * k2[i] + _a53 * k3[i] + _a54 * k4[i]
            for i in range(n_vars)
        )
        Fu = F(u)
        rhs = tuple(
            Fu[i]
            + (_C51 * k1[i] + _C52 * k2[i] + _C53 * k3[i] + _C54 * k4[i])
            * inv_dt
            for i in range(n_vars)
        )
        k5 = S(rhs)

        # Stage 6
        u = tuple(
            y[i]
            + _a61 * k1[i]
            + _a62 * k2[i]
            + _a63 * k3[i]
            + _a64 * k4[i]
            + _a65 * k5[i]
            for i in range(n_vars)
        )
        Fu = F(u)
        rhs = tuple(
            Fu[i]
            + (
                _C61 * k1[i]
                + _C62 * k2[i]
                + _C63 * k3[i]
                + _C64 * k4[i]
                + _C65 * k5[i]
            )
            * inv_dt
            for i in range(n_vars)
        )
        k6 = S(rhs)

        # Stage 7: u = u_stage6 + k6
        u = tuple(u[i] + k6[i] for i in range(n_vars))
        Fu = F(u)
        rhs = tuple(
            Fu[i]
            + (
                _C71 * k1[i]
                + _C72 * k2[i]
                + _C73 * k3[i]
                + _C74 * k4[i]
                + _C75 * k5[i]
                + _C76 * k6[i]
            )
            * inv_dt
            for i in range(n_vars)
        )
        k7 = S(rhs)

        # Stage 8: u = u_stage7 + k7
        u = tuple(u[i] + k7[i] for i in range(n_vars))
        Fu = F(u)
        rhs = tuple(
            Fu[i]
            + (
                _C81 * k1[i]
                + _C82 * k2[i]
                + _C83 * k3[i]
                + _C84 * k4[i]
                + _C85 * k5[i]
                + _C86 * k6[i]
                + _C87 * k7[i]
            )
            * inv_dt
            for i in range(n_vars)
        )
        k8 = S(rhs)

        # Solution: y_new = u + k8, error estimate = k8
        y_new = tuple(u[i] + k8[i] for i in range(n_vars))

        return y_new, k8

    return step


# ---------------------------------------------------------------------------
# Pallas/Triton custom kernel (general ODE, 32 trajectories per block)
# ---------------------------------------------------------------------------


def make_solver(ode_fn):
    """Create a Pallas ensemble solver for the given ODE using Rodas5.

    Args:
        ode_fn: ODE right-hand side function with signature
            (y, p) -> dy/dt
            where y and p are tuples of scalar-like values and the return
            is a tuple of the same length as y. The function must use only
            element-wise operations so it can run inside a Pallas/Triton
            kernel.

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
    def _rodas5_pallas_solve(
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
        step_fn = _make_rodas5_step(ode_fn, n_vars)

        def kernel_body(params_ref, y0_ref, y_ref):
            p = tuple(params_ref.at[:, i][...] for i in range(n_params))
            y = tuple(y0_ref.at[:, i][...] for i in range(n_vars))

            z = p[0] * 0.0
            t = z + 0.0
            dt_v = z + dt0

            def cond_fn(state):
                return (jnp.min(state[0]) < tf) & (state[-1] < ms)

            def body_fn(state):
                t = state[0]
                y = state[1 : 1 + n_vars]
                dt_v = state[1 + n_vars]
                n = state[2 + n_vars]

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
                y_new = tuple(
                    jnp.where(mask, u[i], y[i]) for i in range(n_vars)
                )

                safe_EEst = jnp.where(
                    jnp.isnan(EEst) | (EEst > 1e18),
                    1e18,
                    jnp.where(EEst == 0.0, 1e-18, EEst),
                )
                factor = jnp.clip(0.9 * safe_EEst ** (-1.0 / 6.0), 0.2, 6.0)
                new_dt = jnp.where(active, dt_use * factor, dt_v)

                return (t_new,) + y_new + (new_dt, n + 1)

            init = (t,) + y + (dt_v, jnp.int32(0))
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
        """Solve ODE ensemble using Rodas5 in a Pallas/Triton custom GPU kernel.

        Args:
            y0_batch: Per-trajectory initial conditions, shape (N, n_vars).
            t_span: (t0, tf) time interval.
            params_batch: Per-trajectory parameters, shape (N, n_params).
        """
        N = params_batch.shape[0]
        n_vars = y0_batch.shape[1]
        n_params = params_batch.shape[1]

        # Verify ODE function dimensions via JAX abstract evaluation
        y_trace = tuple(
            jax.ShapeDtypeStruct((), jnp.float64) for _ in range(n_vars)
        )
        p_trace = tuple(
            jax.ShapeDtypeStruct((), jnp.float64) for _ in range(n_params)
        )
        out_trace = jax.eval_shape(ode_fn, y_trace, p_trace)
        assert len(out_trace) == n_vars, (
            f"ODE function returns {len(out_trace)} components but y0 has {n_vars}"
        )

        N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK
        tf = float(t_span[1])
        dt0 = float(
            first_step
            if first_step is not None
            else (tf - float(t_span[0])) * 1e-6
        )

        p_cols = _pad_cols_pow2(n_params)
        y_cols = _pad_cols_pow2(n_vars)

        params_arr = jnp.pad(
            params_batch, ((0, N_pad - N), (0, p_cols - n_params))
        )
        y0_arr = jnp.pad(y0_batch, ((0, N_pad - N), (0, y_cols - n_vars)))

        y_out = _rodas5_pallas_solve(
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
