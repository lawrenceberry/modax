"""KenCarp5 solver — nonlinear IMEX ODE variant (split ode_fn path).

Accepts split ODE functions
``explicit_ode_fn(y, t, params) -> dy/dt`` and
``implicit_ode_fn(y, t, params) -> dy/dt``.

The implicit DIRK stage equation is solved via a per-trajectory modified
Newton iteration, using ``jax.jacfwd`` to form the Jacobian of the implicit
part.  Because KenCarp5 is singly-diagonal (ESDIRK), the iteration matrix
``W = I - γΔt·J`` is the same for every implicit stage, so ``J`` is evaluated
once per step and ``W`` is factored once and reused across all stages and
Newton iterations.  ``lu_precision`` selects whether that factorization is done
in float32 or float64.

Uses a per-trajectory ``jax.lax.while_loop``.  The ``batch_size`` parameter is
passed to ``jax.lax.map`` so that JAX vmaps groups of trajectories together.
``batch_size=N`` (default) batches all trajectories together; ``batch_size=1``
maps them one at a time.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from solvers._jax_common import (
    build_error_weights,
    eval_ode_fn,
    make_custom_vmap_solver,
    normalize_inputs,
    solve_adaptive_ensemble,
)

# fmt: off
_GAMMA = 41.0 / 200.0

_B_SOL = np.array([
    -872700587467.0 / 9133579230613.0,
    0.0,
    0.0,
    22348218063261.0 / 9555858737531.0,
    -1143369518992.0 / 8141816002931.0,
    -39379526789629.0 / 19018526304540.0,
    32727382324388.0 / 42900044865799.0,
    _GAMMA,
], dtype=np.float64)
_B_EMBEDDED = np.array([
    -975461918565.0 / 9796059967033.0,
    0.0,
    0.0,
    78070527104295.0 / 32432590147079.0,
    -548382580838.0 / 3424219808633.0,
    -33438840321285.0 / 15594753105479.0,
    3629800801594.0 / 4656183773603.0,
    4035322873751.0 / 18575991585200.0,
], dtype=np.float64)
_B_ERROR = _B_SOL - _B_EMBEDDED
_C = np.array([
    0.0,
    41.0 / 100.0,
    2935347310677.0 / 11292855782101.0,
    1426016391358.0 / 7196633302097.0,
    92.0 / 100.0,
    24.0 / 100.0,
    3.0 / 5.0,
    1.0,
], dtype=np.float64)

_A_EXPLICIT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [41.0 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1268023523408.0 / 10340822734521.0, 0.0, 1029933939417.0 / 13636558850479.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [14463281900351.0 / 6315353703477.0, 0.0, 66114435211212.0 / 5879490589093.0, -54053170152839.0 / 4284798021562.0, 0.0, 0.0, 0.0, 0.0],
    [14090043504691.0 / 34967701212078.0, 0.0, 15191511035443.0 / 11219624916014.0, -18461159152457.0 / 12425892160975.0, -281667163811.0 / 9011619295870.0, 0.0, 0.0, 0.0],
    [19230459214898.0 / 13134317526959.0, 0.0, 21275331358303.0 / 2942455364971.0, -38145345988419.0 / 4862620318723.0, -1.0 / 8.0, -1.0 / 8.0, 0.0, 0.0],
    [-19977161125411.0 / 11928030595625.0, 0.0, -40795976796054.0 / 6384907823539.0, 177454434618887.0 / 12078138498510.0, 782672205425.0 / 8267701900261.0, -69563011059811.0 / 9646580694205.0, 7356628210526.0 / 4942186776405.0, 0.0],
], dtype=np.float64)

_A_IMPLICIT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, _GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [41.0 / 400.0, -567603406766.0 / 11931857230679.0, _GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0],
    [683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, _GAMMA, 0.0, 0.0, 0.0, 0.0],
    [3016520224154.0 / 10081342136671.0, 0.0, 30586259806659.0 / 12414158314087.0, -22760509404356.0 / 11113319521817.0, _GAMMA, 0.0, 0.0, 0.0],
    [218866479029.0 / 1489978393911.0, 0.0, 638256894668.0 / 5436446318841.0, -1179710474555.0 / 5321154724896.0, -60928119172.0 / 8023461067671.0, _GAMMA, 0.0, 0.0],
    [1020004230633.0 / 5715676835656.0, 0.0, 25762820946817.0 / 25263940353407.0, -2161375909145.0 / 9755907335909.0, -211217309593.0 / 5846859502534.0, -4269925059573.0 / 7827059040719.0, _GAMMA, 0.0],
    [_B_SOL[0], _B_SOL[1], _B_SOL[2], _B_SOL[3], _B_SOL[4], _B_SOL[5], _B_SOL[6], _GAMMA],
], dtype=np.float64)

_PREDICTOR = (
    np.array([1.0], dtype=np.float64),
    np.array([1.0 - _C[2] / _C[1], _C[2] / _C[1]], dtype=np.float64),
    np.array([1.0 - _C[3] / _C[1], _C[3] / _C[1], 0.0], dtype=np.float64),
    np.array([1.0 - _C[4] / _C[2], 0.0, _C[4] / _C[2], 0.0], dtype=np.float64),
    np.array([1.0 - _C[5] / _C[4], 0.0, 0.0, 0.0, _C[5] / _C[4]], dtype=np.float64),
    np.array([1.0 - _C[6] / _C[4], 0.0, 0.0, 0.0, _C[6] / _C[4], 0.0], dtype=np.float64),
    np.array([1.0], dtype=np.float64) * 0.0 + np.array([
        1.0 - _C[7] / _C[4], 0.0, 0.0, 0.0, _C[7] / _C[4], 0.0, 0.0
    ], dtype=np.float64),
)
# fmt: on

_SAFETY = 0.9
_FACTOR_MIN = 0.2
_FACTOR_MAX = 10.0
_NEWTON_MAX_ITERS = 10


def _row_partition(mask):
    """Return a stable inactive-first permutation and its inverse.

    Rows where ``mask`` is False come first; True (active) rows come last.

    Implemented via cumsum rather than ``jnp.argsort(stable=True)`` because the
    latter trips an XLA ``permutation_sort_simplifier`` bug when the mask is
    constant under ``vmap`` inside ``jax.lax.while_loop`` (the case when every
    row of the implicit Jacobian is nonzero — e.g. linear diffusion).
    """
    n = mask.shape[-1]
    mask_i = mask.astype(jnp.int32)
    cum_active = jnp.cumsum(mask_i, axis=-1) - mask_i
    cum_inactive = jnp.cumsum(1 - mask_i, axis=-1) - (1 - mask_i)
    n_active = jnp.sum(mask_i, axis=-1, dtype=jnp.int32)
    n_inactive = jnp.int32(n) - n_active
    # target[i] = position in the partitioned permutation where row i ends up.
    inv_perm = jnp.where(mask, n_inactive + cum_active, cum_inactive).astype(jnp.int32)
    indices = jnp.arange(n, dtype=jnp.int32)
    perm = jnp.zeros(n, dtype=jnp.int32).at[inv_perm].set(indices)
    return perm, inv_perm, n_active


def _permute_vector(vec, perm):
    return jnp.take_along_axis(vec, perm, axis=-1)


def _unpermute_vector(vec_perm, inv_perm):
    return jnp.take_along_axis(vec_perm, inv_perm, axis=-1)


def _permute_matrix(mat, perm):
    mat_rows = jnp.take_along_axis(mat, perm[..., :, None], axis=-2)
    return jnp.take_along_axis(mat_rows, perm[..., None, :], axis=-1)


def _make_reduced_implicit_solver(n_vars, lu_dtype):
    """Factor / solve ``(I - coeff * J) x = rhs`` with a reduced LU.

    Rows where ``mask`` is False are inactive: their implicit Jacobian rows
    are zero, so ``x_i = rhs_i`` by direct substitution.  Active rows form a
    k×k sub-system solved by LU, with coupling from inactive columns included
    in the RHS.  ``lax.switch`` is used to select the right branch at runtime
    while keeping XLA slice sizes static.

    The factor and solve phases are deliberately separated so that a single
    factorization of ``W = I - coeff * J`` can be reused across every Newton
    iteration *and* every implicit stage within a step (modified Newton).  The
    rationale for why this reuse is safe lives in ``_step_one``.

    ``lu_dtype`` (``jnp.float32`` or ``jnp.float64``) selects the precision of
    the factorization and triangular solve.  The residual is always formed and
    the convergence test always done in float64 by the caller, so ``lu_dtype``
    only affects the Newton *correction direction* -- it changes the
    convergence rate, not the solution Newton converges to.
    """

    def _factor_single(jac_perm, n_active, coeff):
        """LU-factor the active k×k block of ``I - coeff*J`` (padded to n_vars)."""
        branches = []
        for active_size in range(n_vars + 1):
            n_inactive = n_vars - active_size

            def _branch(args, *, _k=active_size, _ni=n_inactive):
                jac_perm, coeff = args
                # Pad the factors to a static n_vars shape so every switch
                # branch returns identically-shaped arrays; only the leading
                # k×k block is ever read back during the solve phase.
                lu = jnp.zeros((n_vars, n_vars), dtype=lu_dtype)
                piv = jnp.zeros(n_vars, dtype=jnp.int32)
                if _k == 0:
                    return lu, piv
                mat_aa = (
                    jnp.eye(_k, dtype=jnp.float64) - coeff * jac_perm[_ni:, _ni:]
                ).astype(lu_dtype)
                lu_k, piv_k = jax.scipy.linalg.lu_factor(mat_aa)
                lu = lu.at[:_k, :_k].set(lu_k)
                piv = piv.at[:_k].set(piv_k)
                return lu, piv

            branches.append(_branch)

        return jax.lax.switch(n_active, branches, (jac_perm, coeff))

    def _solve_single(lu, piv, jac_perm, rhs_perm, n_active, coeff):
        branches = []
        for active_size in range(n_vars + 1):
            n_inactive = n_vars - active_size

            def _branch(args, *, _k=active_size, _ni=n_inactive):
                lu, piv, jac_perm, rhs_perm, coeff = args
                if _k == 0:
                    return rhs_perm
                x_inactive = rhs_perm[:_ni]
                jac_an = jac_perm[_ni:, :_ni]
                # Form the (float64) active RHS, drop to lu_dtype for the
                # triangular solve, then return to float64.
                rhs_active = rhs_perm[_ni:] + coeff * (jac_an @ x_inactive)
                x_active = jax.scipy.linalg.lu_solve(
                    (lu[:_k, :_k], piv[:_k]), rhs_active.astype(lu_dtype)
                ).astype(jnp.float64)
                return jnp.concatenate((x_inactive, x_active), axis=0)

            branches.append(_branch)

        return jax.lax.switch(n_active, branches, (lu, piv, jac_perm, rhs_perm, coeff))

    def factor_masked(jac, mask, coeff):
        """Factor ``W = I - coeff*J`` once; reused across stages/iterations."""
        perm, inv_perm, n_active = _row_partition(mask)
        jac_perm = _permute_matrix(jac, perm)
        lu, piv = _factor_single(jac_perm, n_active, coeff)
        return (perm, inv_perm, n_active, jac_perm, lu, piv)

    def solve_factored(factored, rhs, coeff):
        """Solve ``W x = rhs`` using a factorization from ``factor_masked``."""
        perm, inv_perm, n_active, jac_perm, lu, piv = factored
        rhs_perm = _permute_vector(rhs, perm)
        x_perm = _solve_single(lu, piv, jac_perm, rhs_perm, n_active, coeff)
        return _unpermute_vector(x_perm, inv_perm)

    return factor_masked, solve_factored


def solve(
    explicit_ode_fn,
    implicit_ode_fn,
    y0,
    t_span,
    params,
    *,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    """KenCarp5 ensemble solver for split IMEX ODEs."""

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _solve_impl(
            explicit_ode_fn,
            implicit_ode_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            lu_precision=lu_precision,
            batch_size=batch_size,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            return_stats=return_stats,
            error_weights=error_weights,
            pcoeff=pcoeff,
            icoeff=icoeff,
            dcoeff=dcoeff,
        )

    return make_custom_vmap_solver(solve_impl, return_stats=return_stats)(
        y0, t_span, params
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "explicit_ode_fn",
        "implicit_ode_fn",
        "lu_precision",
        "batch_size",
        "max_steps",
        "return_stats",
        "pcoeff",
        "icoeff",
        "dcoeff",
    ),
)
def _solve_impl(
    explicit_ode_fn,
    implicit_ode_fn,
    y0,
    t_span,
    params,
    *,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
):
    """KenCarp5 ensemble solver implementation.

    Parameters
    ----------
    explicit_ode_fn : callable
        Explicit part: ``explicit_ode_fn(y, t, params) -> dy/dt``.
    implicit_ode_fn : callable
        Implicit part: ``implicit_ode_fn(y, t, params) -> dy/dt``.
    y0 : array, shape (n_vars,) or (N, n_vars)
        Initial state. A 1-D array is broadcast to all trajectories; a 2-D
        array supplies a distinct starting point for each trajectory.
    t_span : array-like, shape (n_save,)
        Strictly-increasing 1-D array of save times (including t0).
    params : array, shape (n_params,) or (N, n_params)
        Parameters. A 1-D array is broadcast to all trajectories; a 2-D array
        supplies distinct parameters for each trajectory.
    lu_precision : {"fp32", "fp64"}
        Precision of the per-step LU factorization and triangular solves.
        ``"fp64"`` (default) factors ``W = I - γΔt·J`` in double precision;
        ``"fp32"`` factors in single precision. The Newton residual and
        convergence test stay in float64 either way, so fp32 only affects the
        correction direction (convergence rate), not the converged solution --
        see ``_make_reduced_implicit_solver``.
    batch_size : int or None
        Number of trajectories batched by ``jax.lax.map``. ``None`` (default)
        batches all trajectories together. Internally, ``batch_size`` makes
        ``jax.lax.map`` apply a ``vmap`` over each batch, and JAX hoists the
        per-trajectory ``while_loop`` into one loop spanning all trajectories
        in that batch.
    rtol, atol : float
        Relative and absolute error tolerances.
    first_step : float or None
        Initial step size. Defaults to ``(tf - t0) * 1e-6``.
    max_steps : int
        Maximum number of integration steps per batch.
    return_stats : bool
        If True, return ``(solution, stats)`` where ``stats`` contains raw
        per-lane step counters and per-batch loop diagnostics.
    error_weights : array or None
        Per-component weights for the step-size error norm, shape ``(n_vars,)``
        or ``(N, n_vars)``. ``None`` (default) weights every component equally;
        a weight of 0 excludes that component from step-size control. Used in
        the weighted RMS step-size error norm.
    pcoeff, icoeff, dcoeff : float
        Proportional/integral/derivative gains of the PID step-size controller.
        The default ``(0, 1, 0)`` is the classic I-controller (current
        behavior). ``(0.4, 0.3, 0)`` is a typical PI controller that smooths the
        step sequence on moderately stiff problems; set ``dcoeff`` for a full
        PID. See Soderlind's digital-filter step control.

    Returns
    -------
    array, shape (N, n_save, n_vars)
        Solution at each save time for each trajectory. If ``return_stats`` is
        True, returns ``(solution, stats)``.
    """
    explicit_eval = functools.partial(eval_ode_fn, explicit_ode_fn)
    implicit_eval = functools.partial(eval_ode_fn, implicit_ode_fn)
    implicit_jac_fn = jax.jacfwd(implicit_eval, argnums=0)

    y0_arr, times, params_arr, n, n_vars, _, dt0, bs, n_chunks = normalize_inputs(
        y0, t_span, params, first_step, batch_size
    )
    error_weights_arr = build_error_weights(error_weights, n, n_vars)

    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64
    factor_masked, solve_factored = _make_reduced_implicit_solver(n_vars, lu_dtype)

    def step_factory(params_one):
        def f_explicit(u, t_stage):
            return explicit_eval(u, t_stage, params_one)

        def f_implicit(u, t_stage):
            return implicit_eval(u, t_stage, params_one)

        def jac_implicit(u, t_stage):
            return implicit_jac_fn(u, t_stage, params_one)

        def _newton_stage(base, t_stage, predictor, factored, mask, gamma_dt):
            # ``mask`` and ``factored`` are computed once per step (see
            # ``_step_one``) and shared by every stage.  Rows with all-zero
            # Jacobian rows are "inactive": their implicit RHS is
            # u-independent, so they are resolved by direct substitution
            # without entering Newton.
            def direct_fn(_):
                # All rows inactive: one f_impl eval, no LU, no loop.
                fi = f_implicit(predictor, t_stage)
                u = base + gamma_dt * fi
                return u, fi, jnp.bool_(False)

            def newton_fn(_):
                def cond_fn(state):
                    _, converged, failed, it = state
                    return (~(converged | failed)) & (it < _NEWTON_MAX_ITERS)

                def body_fn(state):
                    u, converged, failed, it = state
                    fi = f_implicit(u, t_stage)
                    # Modified Newton: the residual is evaluated *exactly*
                    # every iteration, but the iteration matrix
                    # W = I - γΔt·J reuses the step-start factorization (J
                    # frozen).  An inexact/frozen W only changes the
                    # convergence rate, never the fixed point Newton
                    # converges to, so the stage equation is still solved
                    # to tolerance.
                    res = u - base - gamma_dt * fi
                    # Reduced solve: inactive rows → delta_i = res_i (direct),
                    # active rows → k×k Newton step with coupling correction.
                    delta = solve_factored(factored, res, gamma_dt)
                    u_new = jnp.where(converged | failed, u, u - delta)
                    scale = atol + rtol * jnp.maximum(jnp.abs(u), jnp.abs(u_new))
                    delta_norm = jnp.sqrt(jnp.mean((delta / scale) ** 2))
                    invalid = (
                        jnp.any(~jnp.isfinite(u_new))
                        | jnp.any(~jnp.isfinite(delta))
                        | jnp.isnan(delta_norm)
                    )
                    converged_new = converged | ((delta_norm <= 1.0) & ~invalid)
                    failed_new = failed | invalid
                    return (u_new, converged_new, failed_new, it + 1)

                init = (
                    predictor,
                    jnp.bool_(False),
                    jnp.bool_(False),
                    jnp.int32(0),
                )
                u_final, converged, failed, _ = jax.lax.while_loop(
                    cond_fn, body_fn, init
                )
                fi_final = f_implicit(u_final, t_stage)
                failed = failed | ~converged | jnp.any(~jnp.isfinite(fi_final))
                return u_final, fi_final, failed

            return jax.lax.cond(jnp.any(mask), newton_fn, direct_fn, None)

        def _step_one(y, t, dt, extra):
            del extra
            gamma_dt = dt * _GAMMA

            # --- Freeze the implicit Jacobian once per step --------------------
            # KenCarp5 is an ESDIRK method: every implicit diagonal coefficient
            # equals γ (= _GAMMA).  Each implicit stage i solves
            #     u = base_i + γ·dt·f_impl(u, t_i),
            # whose Newton iteration matrix is W = I - γ·dt·J with
            # J = ∂f_impl/∂u.  Because γ and dt are constant within a step, W is
            # the *same* operator for all seven implicit stages.  We therefore
            # evaluate J once (at the step-start point) and factor W a single
            # time, then reuse that factorization
            #   * across every Newton iteration of a stage (modified Newton), and
            #   * across all seven implicit stages.
            # This is safe because the Newton residual
            #     res = u - base_i - γ·dt·f_impl(u, t_i)
            # is still evaluated exactly each iteration; a frozen iteration
            # matrix only affects how fast Newton converges, never the fixed
            # point it converges to.  (We deliberately do *not* reuse J across
            # steps: a fresh J and factorization are formed every step.)
            jac = jac_implicit(y, t)
            mask = jnp.any(jac != 0.0, axis=1)
            factored = factor_masked(jac, mask, gamma_dt)

            stage_y = []
            stage_fe = []
            stage_fi = []
            failed = jnp.any(~jnp.isfinite(jac))

            y_stage = y
            t_stage = t
            fe_stage = f_explicit(y_stage, t_stage)
            fi_stage = f_implicit(y_stage, t_stage)
            stage_y.append(y_stage)
            stage_fe.append(fe_stage)
            stage_fi.append(fi_stage)
            failed = (
                failed
                | jnp.any(~jnp.isfinite(fe_stage))
                | jnp.any(~jnp.isfinite(fi_stage))
            )

            for i in range(1, 8):
                t_stage = t + _C[i] * dt
                base = y
                for j in range(i):
                    ae = _A_EXPLICIT[i, j]
                    ai = _A_IMPLICIT[i, j]
                    if ae != 0.0:
                        base = base + dt * ae * stage_fe[j]
                    if ai != 0.0:
                        base = base + dt * ai * stage_fi[j]

                predictor = jnp.zeros_like(y)
                predictor_coeff = _PREDICTOR[i - 1]
                for j, coeff in enumerate(predictor_coeff):
                    if coeff != 0.0:
                        predictor = predictor + coeff * stage_y[j]

                y_stage, fi_stage, stage_failed = _newton_stage(
                    base, t_stage, predictor, factored, mask, gamma_dt
                )
                fe_stage = f_explicit(y_stage, t_stage)
                failed = (
                    failed
                    | stage_failed
                    | jnp.any(~jnp.isfinite(y_stage))
                    | jnp.any(~jnp.isfinite(fe_stage))
                )
                stage_y.append(y_stage)
                stage_fe.append(fe_stage)
                stage_fi.append(fi_stage)

            y_new = stage_y[-1]
            err_est = jnp.zeros_like(y)
            for i in range(8):
                total_stage = stage_fe[i] + stage_fi[i]
                err_est = err_est + dt * _B_ERROR[i] * total_stage
            failed = (
                failed | jnp.any(~jnp.isfinite(y_new)) | jnp.any(~jnp.isfinite(err_est))
            )
            return y_new, err_est, failed, ()

        return _step_one, (), lambda extra, candidate, accept: extra

    return solve_adaptive_ensemble(
        params_arr=params_arr,
        y0_arr=y0_arr,
        times=times,
        dt0=dt0,
        batch_size=bs,
        n_chunks=n_chunks,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        return_stats=return_stats,
        step_factory=step_factory,
        error_exponent=-1.0 / 5.0,
        safety=_SAFETY,
        factor_min=_FACTOR_MIN,
        factor_max=_FACTOR_MAX,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        error_weights_arr=error_weights_arr,
    )
