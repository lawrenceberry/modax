"""General-purpose Rodas5 ensemble ODE solver with Pallas GPU kernel.

Implements the Rodas5 algorithm — a Rosenbrock method of order 5(4) for stiff
ODEs — using the same general-purpose make_solver pattern as
rosenbrock23_custom_kernel.py.

The W-transformed variant is from Di Marzo (1993) and DISCO-EB (Hahn).
8 stages with adaptive step size control. Pallas/Triton custom GPU kernel
with 32 trajectories per block.

The LU decomposition and forward/back substitution use jax.lax.fori_loop
with Pallas Ref scratch memory and pl.ds dynamic indexing, producing compact
compiled GPU loops instead of O(n^3) unrolled instructions.

Reference: https://github.com/SciML/DiffEqGPU.jl
"""

import functools
import itertools

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
# Rodas5 step: fori_loop LU with Pallas Ref scratch memory
# ---------------------------------------------------------------------------


def _make_rodas5_step(ode_fn, n_vars):
    """Rodas5 step with fori_loop-based LU using Pallas Ref scratch memory.

    For systems where unrolling the O(n^3) LU decomposition would generate
    too much Triton IR code (n_vars > 16). Uses fori_loop for compact
    compiled GPU loops and Pallas Refs with pl.ds for dynamic matrix access.
    """
    nv = n_vars  # Python int, promoted automatically in JAX arithmetic

    def step(y, p, dt, w_ref, x_ref):
        dtgamma = dt * _gamma
        inv_dt = 1.0 / dt
        d = 1.0 / dtgamma

        def F(y):
            return ode_fn(y, p)

        F0 = F(y)

        # Jacobian via JVP (trace-time unrolled, n_vars iterations)
        ones = y[0] * 0.0 + 1.0
        J_cols = []
        for j in range(n_vars):

            def f_j(yj, j=j):
                y_mod = tuple(yj if k == j else y[k] for k in range(n_vars))
                return F(y_mod)

            _, col = jax.jvp(f_j, (y[j],), (ones,))
            J_cols.append(col)

        # Fill W = I/(dt*gamma) - J into w_ref (static Python-int indexing)
        for i_s in range(n_vars):
            for j_s in range(n_vars):
                val = -J_cols[j_s][i_s]
                if i_s == j_s:
                    val = val + d
                w_ref.at[:, i_s * nv + j_s][...] = val

        # --- LU decomposition in-place on w_ref via fori_loop ---
        def lu_col(j, carry):
            pivot = w_ref.at[:, pl.ds(j * nv + j, 1)][...][:, 0]

            def lu_row(i, carry):
                l_ij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0] / pivot
                w_ref.at[:, pl.ds(i * nv + j, 1)][...] = l_ij[:, None]

                def lu_elem(k, carry):
                    ik = w_ref.at[:, pl.ds(i * nv + k, 1)][...][:, 0]
                    jk = w_ref.at[:, pl.ds(j * nv + k, 1)][...][:, 0]
                    w_ref.at[:, pl.ds(i * nv + k, 1)][...] = (ik - l_ij * jk)[:, None]
                    return carry

                return jax.lax.fori_loop(j + 1, nv, lu_elem, carry)

            return jax.lax.fori_loop(j + 1, nv, lu_row, carry)

        jax.lax.fori_loop(0, nv, lu_col, jnp.int32(0))

        # --- Solve W @ x = b via LU stored in w_ref ---
        def S(b_tuple):
            # Store RHS in x_ref (static Python-int indexing)
            for i_s in range(n_vars):
                x_ref.at[:, i_s][...] = b_tuple[i_s]

            # Forward substitution: z[i] -= L[i,j] * z[j] for j < i
            def fwd_row(i, carry):
                def fwd_col(j, carry):
                    xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    xj = x_ref.at[:, pl.ds(j, 1)][...][:, 0]
                    lij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                    x_ref.at[:, pl.ds(i, 1)][...] = (xi - lij * xj)[:, None]
                    return carry

                return jax.lax.fori_loop(0, i, fwd_col, carry)

            jax.lax.fori_loop(0, nv, fwd_row, jnp.int32(0))

            # Back substitution: x[i] = (z[i] - sum U[i,j]*x[j]) / U[i,i]
            def bwd_row(i_rev, carry):
                i = nv - 1 - i_rev

                def bwd_col(j_off, carry):
                    j = i + 1 + j_off
                    xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    xj = x_ref.at[:, pl.ds(j, 1)][...][:, 0]
                    uij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                    x_ref.at[:, pl.ds(i, 1)][...] = (xi - uij * xj)[:, None]
                    return carry

                jax.lax.fori_loop(0, nv - 1 - i, bwd_col, carry)
                xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                uii = w_ref.at[:, pl.ds(i * nv + i, 1)][...][:, 0]
                x_ref.at[:, pl.ds(i, 1)][...] = (xi / uii)[:, None]
                return carry

            jax.lax.fori_loop(0, nv, bwd_row, jnp.int32(0))

            # Read result (static Python-int indexing)
            return tuple(x_ref.at[:, i_s][...] for i_s in range(n_vars))

        return _rodas5_stages(F, S, y, F0, inv_dt, n_vars)

    return step


# ---------------------------------------------------------------------------
# Shared: Rodas5 8-stage computation (used by both step implementations)
# ---------------------------------------------------------------------------


def _rodas5_stages(F, S, y, F0, inv_dt, n_vars):
    """Compute the 8 Rodas5 stages given ODE function F and linear solver S."""
    k1 = S(F0)

    u = jax.tree.map(lambda yi, k1i: yi + _a21 * k1i, y, k1)
    Fu = F(u)
    k2 = S(jax.tree.map(lambda fui, k1i: fui + _C21 * k1i * inv_dt, Fu, k1))

    u = jax.tree.map(
        lambda yi, k1i, k2i: yi + _a31 * k1i + _a32 * k2i,
        y,
        k1,
        k2,
    )
    Fu = F(u)
    k3 = S(
        jax.tree.map(
            lambda fui, k1i, k2i: fui + (_C31 * k1i + _C32 * k2i) * inv_dt,
            Fu,
            k1,
            k2,
        )
    )

    u = jax.tree.map(
        lambda yi, k1i, k2i, k3i: yi + _a41 * k1i + _a42 * k2i + _a43 * k3i,
        y,
        k1,
        k2,
        k3,
    )
    Fu = F(u)
    k4 = S(
        jax.tree.map(
            lambda fui, k1i, k2i, k3i: (
                fui + (_C41 * k1i + _C42 * k2i + _C43 * k3i) * inv_dt
            ),
            Fu,
            k1,
            k2,
            k3,
        )
    )

    u = jax.tree.map(
        lambda yi, k1i, k2i, k3i, k4i: (
            yi + _a51 * k1i + _a52 * k2i + _a53 * k3i + _a54 * k4i
        ),
        y,
        k1,
        k2,
        k3,
        k4,
    )
    Fu = F(u)
    k5 = S(
        jax.tree.map(
            lambda fui, k1i, k2i, k3i, k4i: (
                fui + (_C51 * k1i + _C52 * k2i + _C53 * k3i + _C54 * k4i) * inv_dt
            ),
            Fu,
            k1,
            k2,
            k3,
            k4,
        )
    )

    u = jax.tree.map(
        lambda yi, k1i, k2i, k3i, k4i, k5i: (
            yi + _a61 * k1i + _a62 * k2i + _a63 * k3i + _a64 * k4i + _a65 * k5i
        ),
        y,
        k1,
        k2,
        k3,
        k4,
        k5,
    )
    Fu = F(u)
    k6 = S(
        jax.tree.map(
            lambda fui, k1i, k2i, k3i, k4i, k5i: (
                fui
                + (_C61 * k1i + _C62 * k2i + _C63 * k3i + _C64 * k4i + _C65 * k5i)
                * inv_dt
            ),
            Fu,
            k1,
            k2,
            k3,
            k4,
            k5,
        )
    )

    u = jax.tree.map(lambda ui, k6i: ui + k6i, u, k6)
    Fu = F(u)
    k7 = S(
        jax.tree.map(
            lambda fui, k1i, k2i, k3i, k4i, k5i, k6i: (
                fui
                + (
                    _C71 * k1i
                    + _C72 * k2i
                    + _C73 * k3i
                    + _C74 * k4i
                    + _C75 * k5i
                    + _C76 * k6i
                )
                * inv_dt
            ),
            Fu,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
        )
    )

    u = jax.tree.map(lambda ui, k7i: ui + k7i, u, k7)
    Fu = F(u)
    k8 = S(
        jax.tree.map(
            lambda fui, k1i, k2i, k3i, k4i, k5i, k6i, k7i: (
                fui
                + (
                    _C81 * k1i
                    + _C82 * k2i
                    + _C83 * k3i
                    + _C84 * k4i
                    + _C85 * k5i
                    + _C86 * k6i
                    + _C87 * k7i
                )
                * inv_dt
            ),
            Fu,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
        )
    )

    y_new = jax.tree.map(lambda ui, k8i: ui + k8i, u, k8)
    return y_new, k8


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
            "w_cols",
            "x_cols",
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
        w_cols,
        x_cols,
        n_vars,
        n_params,
        tf,
        dt0,
        r_tol,
        a_tol,
        ms,
    ):
        step_fn = _make_rodas5_step(ode_fn, n_vars)

        def kernel_body(params_ref, y0_ref, y_ref, w_ref, x_ref):
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

                u, err = step_fn(y, p, dt_use, w_ref, x_ref)

                sc = jax.tree.map(
                    lambda yi, ui: (
                        a_tol + r_tol * jnp.maximum(jnp.abs(yi), jnp.abs(ui))
                    ),
                    y,
                    u,
                )
                err_terms = jax.tree.map(
                    lambda eri, sci: (eri / sci) ** 2,
                    err,
                    sc,
                )
                err_sq = jax.tree_util.tree_reduce(
                    lambda acc, x: acc + x,
                    err_terms,
                    initializer=jnp.zeros_like(t),
                )
                EEst = jnp.sqrt(err_sq / n_vars)

                accept = (EEst <= 1.0) & ~jnp.isnan(EEst)
                mask = active & accept

                t_new = jnp.where(mask, t + dt_use, t)
                y_new = jax.tree.map(lambda ui, yi: jnp.where(mask, ui, yi), u, y)

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
        w_bs = pl.BlockSpec((_BLOCK, w_cols), lambda i: (i, 0))
        x_bs = pl.BlockSpec((_BLOCK, x_cols), lambda i: (i, 0))
        results = pl.pallas_call(
            kernel_body,
            out_shape=[
                jax.ShapeDtypeStruct((n_pad, y_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, w_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, x_cols), jnp.float64),
            ],
            grid=(n_pad // _BLOCK,),
            in_specs=(p_bs, y_bs),
            out_specs=(y_bs, w_bs, x_bs),
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
        )(params_arr, y0_arr)
        return results[0]

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
        y_trace = tuple(itertools.repeat(jax.ShapeDtypeStruct((), jnp.float64), n_vars))
        p_trace = tuple(
            itertools.repeat(jax.ShapeDtypeStruct((), jnp.float64), n_params)
        )
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
        w_cols = _pad_cols_pow2(n_vars * n_vars)
        x_cols = _pad_cols_pow2(n_vars)

        params_arr = jnp.pad(params_batch, ((0, N_pad - N), (0, p_cols - n_params)))
        y0_arr = jnp.pad(y0_batch, ((0, N_pad - N), (0, y_cols - n_vars)))

        y_out = _rodas5_pallas_solve(
            params_arr,
            y0_arr,
            n_pad=N_pad,
            p_cols=p_cols,
            y_cols=y_cols,
            w_cols=w_cols,
            x_cols=x_cols,
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
