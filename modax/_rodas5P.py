"""The Rodas5P kernel: one thread per trajectory, every buffer thread-local."""

from __future__ import annotations

import functools
import math
from typing import (
    Any,
    Callable,
    Iterator,
)

import numpy as np
import numpy.typing as npt
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from modax._codegen import compile_device_source
from modax._kernel_cache import enable_content_cache
from modax._numba_common import (
    OdeFn,
    as_cuda_device,
    make_cuda_local_vector_writer,
    make_step_controller,
)
from modax._options import KernelOptions
from modax._sensitivity import (
    make_augmented_local_writer,
    make_second_tangent,
    make_tangent,
    seed_table,
)
from modax._sparse_direct import sparse_direct_solver_for
from modax._sparsity import dense_jacobian

# fmt: off
# Rodas5P W-transformed coefficients (Steinebach 2023, BIT 63:27).
# Matches the Julia Rodas5P / GPURodas5P tableau. Stages 6-8 reuse the FSAL
# accumulation (u += k6, u += k7) so no A7/A8 rows are needed.
GAMMA = 0.21193756319429014

A21 = 3.0
A31 = 2.849394379747939
A32 = 0.45842242204463923
A41 = -6.954028509809101
A42 = 2.489845061869568
A43 = -10.358996098473584
A51 = 2.8029986275628964
A52 = 0.5072464736228206
A53 = -0.3988312541770524
A54 = -0.04721187230404641
A61 = -7.502846399306121
A62 = 2.561846144803919
A63 = -11.627539656261098
A64 = -0.18268767659942256
A65 = 0.030198172008377946

C21 = -14.155112264123755
C31 = -17.97296035885952
C32 = -2.859693295451294
C41 = 147.12150275711716
C42 = -1.41221402718213
C43 = 71.68940251302358
C51 = 165.43517024871676
C52 = -0.4592823456491126
C53 = 42.90938336958603
C54 = -5.961986721573306
C61 = 24.854864614690072
C62 = -3.0009227002832186
C63 = 47.4931110020768
C64 = 5.5814197821558125
C65 = -0.6610691825249471
C71 = 30.91273214028599
C72 = -3.1208243349937974
C73 = 77.79954646070892
C74 = 34.28646028294783
C75 = -19.097331116725623
C76 = -28.087943162872662
C81 = 37.80277123390563
C82 = -3.2571969029072276
C83 = 112.26918849496327
C84 = 66.9347231244047
C85 = -40.06618937091002
C86 = -54.66780262877968
C87 = -9.48861652309627

C2 = 0.6358126895828704
C3 = 0.4095798393397535
C4 = 0.9769306725060716
C5 = 0.4288403609558664
# C6 = C7 = C8 = 1.0

# Time-derivative ("d_i") coefficients. The stage RHS carries an additional
# dt*d_i*df/dt term (Hairer-Wanner II.7); without it, the method drops below
# order 5 when df/dt is nonzero. d6 = d7 = d8 = 0.
D1 = GAMMA  # = 0.21193756319429014
D2 = -0.42387512638858027
D3 = -0.3384627126235924
D4 =  1.8046452872882734
D5 =  2.325825639765069
# fmt: on


# The step controller: the error order the exponents divide by, and how far a
# step may grow or shrink at once.
ORDER = 6
FACTOR_MIN = 0.2
FACTOR_MAX = 6.0


@functools.cache
def dense_lu_solver(n_vars: int) -> tuple[MLIRDispatcher, MLIRDispatcher]:
    """Build a dense LU with partial pivoting, one system per thread.

    The kernel uses it when it has no sparsity pattern: a right-looking LU over
    the thread's own row-major buffer, then both triangular solves in place. It
    returns the same ``(factorize_local, solve_local)`` pair that
    :py:obj:`sparse_direct_solver <modax._sparse_direct.sparse_direct_solver>`
    builds from a pattern, so the kernel has no branch.

    It replaced nvmath's block-collective ``LUPivotSolver``, which needed shared
    memory and a barrier around every factorisation. Matrices of tens of
    variables are too small for cooperation between threads to pay off.
    """
    n = n_vars

    @cuda.jit(device=True)
    def factorize_local(lu, ipiv):
        for i in range(n):
            # Partial pivoting: the largest remaining entry in this column.
            max_val = abs(lu[i * n + i])
            pivot = i
            for k in range(i + 1, n):
                val = abs(lu[k * n + i])
                if val > max_val:
                    max_val = val
                    pivot = k
            ipiv[i] = pivot

            if pivot != i:
                for j in range(n):
                    tmp = lu[i * n + j]
                    lu[i * n + j] = lu[pivot * n + j]
                    lu[pivot * n + j] = tmp

            # A singular column is left alone rather than guarded against: the
            # Rosenbrock-W property tolerates an approximate factorisation, and
            # the step controller rejects whatever comes out of one that is not.
            piv = lu[i * n + i]
            if piv != 0.0:
                inv_piv = 1.0 / piv
                for k in range(i + 1, n):
                    factor = lu[k * n + i] * inv_piv
                    lu[k * n + i] = factor
                    for j in range(i + 1, n):
                        lu[k * n + j] -= factor * lu[i * n + j]

    @cuda.jit(device=True)
    def solve_local(lu, ipiv, rhs):
        # Forward substitution through L, applying the pivots as they come.
        for i in range(n):
            pivot = ipiv[i]
            if pivot != i:
                tmp = rhs[i]
                rhs[i] = rhs[pivot]
                rhs[pivot] = tmp
            acc = rhs[i]
            for j in range(i):
                acc -= lu[i * n + j] * rhs[j]
            rhs[i] = acc
        # Back substitution through U.
        for i in range(n - 1, -1, -1):
            acc = rhs[i]
            for j in range(i + 1, n):
                acc -= lu[i * n + j] * rhs[j]
            rhs[i] = acc / lu[i * n + i]

    return factorize_local, solve_local


# Direction sets per jvp call. Several seeds in one call share one Enzyme entry
# function, so once nvJitLink has inlined it the primal work the sweeps have in
# common -- for an ODE whose coefficients depend on t alone, all of it -- is one
# computation for LLVM to CSE rather than one per sweep. On DISCO-EB that was
# the difference between 580 ms and 509 ms at N128, on top of the literal
# seeds. numba-enzyme accepts up to 8 sets per call (its _MAX_DIRECTIONS), so
# 12 colours take two calls.
SEED_BATCH = 8


def _make_literal_seed_jacobian(
    *,
    n_vars: int,
    n_params: int,
    n_colours: int,
    store_table: npt.NDArray[np.int32] | None,
    namespace: dict[str, object],
    inline: str = "always",
    batch: int | None = None,
    coupling: int = 0,
) -> MLIRDispatcher:
    """Generate the Jacobian writer with every colour's seed row spelled out.

    ``store_table`` maps ``(row, colour)`` to a slot, with ``-1`` for none. With
    ``None`` the layout is the plain colour grid, entry ``(row, g)`` at
    ``row * n_colours + g``. The time sweep (``df/dt`` into ``dT``) is one more
    direction of the same calls. With ``batch`` the directions go in equal
    batches of at most that many, padded with zero directions, so Enzyme lowers
    the primal for one arity only.

    ``coupling`` adds that many directions seeded at run time. The writer then
    also takes ``p_seed``, a ``(coupling, n_params)`` array of parameter seeds,
    and ``coup``, a ``(coupling, n_vars)`` array that receives
    ``df/dp . p_seed[k]`` for each. The split kernel uses this to seed an
    upstream block's error through its Hermite values and read off the error
    rate it hands down.
    """
    zero_row = n_colours
    n_directions = n_colours + 1 + coupling
    if batch is None:
        batch = SEED_BATCH
    else:
        n_batches = -(-n_directions // min(int(batch), SEED_BATCH))
        batch = -(-n_directions // n_batches)

    def slots_for(g: int) -> Iterator[tuple[int, int]]:
        for row in range(n_vars):
            if store_table is None:
                yield row, row * n_colours + g
            else:
                slot = int(store_table[row * n_colours + g])
                if slot >= 0:
                    yield row, slot

    # (state seed row, time seed, parameter seed): the colours, the time
    # direction, the coupling directions, then zero directions to fill the
    # last batch (their results are not read)
    zero_p = f"seed[{zero_row}, 0:{n_params}]"
    directions = [(g, "0.0", zero_p) for g in range(n_colours)] + [
        (zero_row, "1.0", zero_p)
    ]
    directions += [(zero_row, "0.0", f"p_seed[{k}]") for k in range(coupling)]
    directions += [(zero_row, "0.0", zero_p)] * (-len(directions) % batch)

    def stores(
        direction: tuple[int, str, str], value_of: Callable[[int], str]
    ) -> list[str]:
        g, dt_seed, p_seed = direction
        if dt_seed == "1.0":
            return [f"    dT[{row}] = {value_of(row)}" for row in range(n_vars)]
        if p_seed.startswith("p_seed"):
            k = p_seed[len("p_seed[") : -1]
            return [f"    coup[{k}, {row}] = {value_of(row)}" for row in range(n_vars)]
        if g == zero_row:
            return []  # padding
        return [
            f"    lu[{slot}] = lu_dtype(-{value_of(row)})" for row, slot in slots_for(g)
        ]

    lines = [
        "def write_negated_jacobian(y_local, t, p_row, lu, dT"
        + (", p_seed, coup):" if coupling else "):"),
        "    seed = cuda.const.array_like(colour_seeds)",
        f"    column = cuda.local.array({n_vars}, float64)",
        "    clear_matrix(lu)",
    ]
    if batch > 1:
        lines.append(f"    block = cuda.local.array(({batch}, {n_vars}), float64)")
    for start in range(0, len(directions), batch):
        group = directions[start : start + batch]
        if len(group) == 1:
            g, dt_seed, p_seed = group[0]
            lines.append(
                f"    tangent_of(column, y_local, t, p_row, seed[{g}, 0:{n_vars}], {dt_seed}, {p_seed})"
            )
            lines += stores(group[0], lambda row: f"column[{row}]")
        else:
            dirs = ", ".join(
                f"seed[{g}, 0:{n_vars}], {dt_seed}, {p_seed}"
                for g, dt_seed, p_seed in group
            )
            lines.append(f"    tangent_of(block, y_local, t, p_row, {dirs})")
            for k, direction in enumerate(group):
                lines += stores(direction, lambda row, k=k: f"block[{k}, {row}]")
    return compile_device_source(
        "write_negated_jacobian", lines, namespace, inline=inline
    )


@functools.cache
def make_kernel(
    ode_fn: OdeFn, n_vars: int, n_params: int, options: KernelOptions
) -> tuple[MLIRDispatcher, int]:
    """Build the Rodas5P kernel, one trajectory per CUDA thread.

    Each thread owns its state, ten stage vectors, ``df/dt``, step controller
    and iteration matrix as local arrays and scalars. Threads share nothing and
    nothing synchronises inside a step, so neither the state dimension nor the
    matrix bounds the block size, and each thread assembles its own Jacobian.

    The linear solve is one code path over a buffer the layout describes.
    Without a pattern that is the dense row-major matrix and
    :py:obj:`dense_lu_solver <modax._rodas5P.dense_lu_solver>`. With one it is
    the sparse factorisation's CSR image and a solver compiled for it.

    Returns:
        The kernel and its trajectories per block.
    """
    spec, sparsity, lu_precision = options.spec, options.sparsity, options.lu_precision
    # Both compile-time constants: numba prunes the hook call and the history
    # writes it does not need, so a solve without a hook compiles as before.
    HAS_HOOK = options.save_hook is not None
    SAVE_HISTORY = bool(options.save_history)
    hook_size = max(1, int(options.hook_size))
    if HAS_HOOK:
        save_hook = as_cuda_device(options.save_hook)
    else:

        @cuda.jit(device=True)
        def save_hook(save_idx, y, t, p_row, acc):
            return

    next_step = make_step_controller(options.gains, ORDER, FACTOR_MIN, FACTOR_MAX)
    # The state, right-hand side, Jacobian and error estimate are always
    # float64; lu_dtype governs only the iteration matrix and its solves.
    # Rodas5P satisfies the Rosenbrock-W order conditions to second order, so
    # the FP32 rounding in the factorisation costs an error that scales with
    # that rounding, absorbed by the step controller, rather than convergence.
    lu_dtype = np.float32 if lu_precision == "fp32" else np.float64
    # cuda.local.array wants the numba type rather than the numpy dtype.
    lu_local_dtype = types.float32 if lu_precision == "fp32" else types.float64
    if sparsity is None:
        structure = dense_jacobian(n_vars)
        # The dense LU pivots the whole state; the sparse one pivots nothing.
        factorize_local, solve_local = dense_lu_solver(n_vars)
        ipiv_per = n_vars
    else:
        solver = sparse_direct_solver_for(sparsity, options.ordering)
        structure = solver.compressed
        factorize_local, solve_local = solver.factorize_local, solver.solve_local
        ipiv_per = solver.ipiv_size
    # With a spec the kernel integrates the joint [y, S] system, so the state,
    # stage and error extents become the augmented ones. The matrix does not:
    # the joint iteration matrix is block lower triangular with the same
    # M0 = I/(h*gamma) - J on every diagonal block, so one factorisation of the
    # n_vars block serves the state and every sensitivity column.
    size = n_vars if spec is None else spec.n_aug
    n_error = n_vars if spec is None else spec.n_error
    n_sens = 0 if spec is None else spec.n_sens

    tpb = int(options.trajectories_per_block)
    n_colours = structure.n_colours

    lu_size = structure.size
    diag_table = np.asarray(structure.diagonal, dtype=np.int32)
    store_table = structure.store_slots()

    if spec is not None:
        ode_write = make_augmented_local_writer(ode_fn, spec)
    else:
        ode_write = make_cuda_local_vector_writer(ode_fn, n_vars)

    if spec is not None:
        n_y0_dirs = spec.n_y0_dirs
        param_cols = spec.param_seed_columns
        seeds = seed_table(spec)
        length = max(n_vars, n_params)
        second_tangent = make_second_tangent(ode_fn, n_vars, n_params)

        @cuda.jit(device=True)
        def write_sensitivity_time_derivative(y_local, t, p_row, dT):
            """Write ``dF/dt`` for the sensitivity rows of the joint system.

            This is ``d/dt (J_y S_k + J_p_k)`` at fixed state, a second derivative of
            ode_fn that the second-order sweep returns when seeded with the time
            direction on the outside. Without it these rows of ``dT`` hold stale local
            memory, and the method loses order when the parameter dependence varies
            in time.
            """
            seed = cuda.const.array_like(seeds)
            column = cuda.local.array(n_vars, types.float64)
            for k in range(n_sens):
                base = n_vars + k * n_vars
                start = 0 if k < n_y0_dirs else length - param_cols[k - n_y0_dirs]
                second_tangent(
                    column,
                    y_local,
                    t,
                    p_row,
                    y_local[base : base + n_vars],
                    0.0,
                    seed[start : start + n_params],
                    seed[0:n_vars],
                    1.0,
                    seed[0:n_params],
                    # The inner direction does not vary: zero here leaves the
                    # plain second-order form.
                    seed[0:n_vars],
                    0.0,
                    seed[0:n_params],
                )
                for row in range(n_vars):
                    dT[base + row] = column[row]

    else:

        @cuda.jit(device=True)
        def write_sensitivity_time_derivative(y_local, t, p_row, dT):
            pass

    # --- the negated Jacobian, straight into the thread's own buffer ---------
    # ``df/dy`` and ``df/dt`` are forward-differentiated out of ode_fn by
    # Enzyme. Each sweep is seeded with a whole colour group rather than one
    # unit vector: the columns in a group share no row, so their contributions
    # to ``J v`` never collide and one sweep yields the lot. That is
    # ``n_colours + 1`` sweeps where a column at a time costs ``n_vars + 1``,
    # and for a banded or bordered system the difference is an order of
    # magnitude. See "Derived Jacobians" in AGENTS.md.
    #
    # This leaves -J in the buffer and df/dt in dT; the kernel then adds
    # 1/(h*gamma) on the diagonal slots to finish M. Writing straight here,
    # rather than staging J through global memory and reading it back, avoids a
    # per-step round-trip of the whole matrix.
    tangent_of = make_tangent(ode_fn, n_vars, n_params)
    colour_seeds = structure.seed_table(n_params)

    # The grid writes every slot and a packed layout has one slot per entry,
    # so both come out fully defined by the sweeps. Only a layout with slots
    # nothing writes -- the factorisation's fill-in, or a packed diagonal the
    # pattern left out -- is cleared first.
    if structure.has_unwritten_slots:

        @cuda.jit(device=True)
        def clear_matrix(lu):
            for i in range(lu_size):
                lu[i] = lu_dtype(0.0)

    else:

        @cuda.jit(device=True)
        def clear_matrix(lu):
            pass

    # The colour loop is unrolled into call sites whose seed rows are
    # *literals*. The derivative links as LTO IR and nvJitLink inlines it into
    # the kernel before constant propagation, so a seed the compiler can see
    # folds: the zero components kill their tangent arithmetic and each sweep
    # collapses to its own colour group's columns. The same seed read through
    # a loop variable arrives in registers and cannot fold, which is what had
    # this kernel at the register cap with a 13 KB spill frame. Several seeds
    # go into each call (SEED_BATCH) so the sweeps' shared primal is computed
    # once, and the stores are literal too, one per entry the colour holds.
    write_negated_jacobian = _make_literal_seed_jacobian(
        n_vars=n_vars,
        n_params=n_params,
        n_colours=n_colours,
        store_table=store_table,
        namespace=dict(
            cuda=cuda,
            float64=types.float64,
            tangent_of=tangent_of,
            colour_seeds=colour_seeds,
            clear_matrix=clear_matrix,
            lu_dtype=lu_dtype,
        ),
    )

    # --- one Rosenbrock stage, state row plus any sensitivity rows -----------
    if spec is None:

        @cuda.jit(device=True)
        def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t):
            solve_local(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[s, j] = np.float64(rhs[j])

    else:

        @cuda.jit(device=True)
        def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t):
            """Solve one stage of the joint system by forward substitution.

            The joint iteration matrix is block lower triangular

                [  M0        ] [ k_y   ]   [ r_y   ]
                [ -L_k   M0  ] [ k_S_k ] = [ r_S_k ]

            so the state row solves first, then each sensitivity row against the same
            factorisation with ``L_k k_y`` added to its right-hand side. The kernel
            never forms ``L_k``. One second-order sweep of ode_fn gives the correction,
            seeded with the sensitivity column inside and the state increment outside.
            Like the Jacobian, it stays frozen at the step's base point, and only the
            outer direction changes per stage.
            """
            seed = cuda.const.array_like(seeds)
            coupling = cuda.local.array(n_vars, types.float64)
            # The stage vectors carry lu_precision, but Enzyme's endpoint is
            # built for float64 directions, so the state increment is widened
            # here rather than the stages being stored wider throughout.
            k_state = cuda.local.array(n_vars, types.float64)
            solve_local(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[s, j] = np.float64(rhs[j])
            for j in range(n_vars):
                k_state[j] = np.float64(k_stages[s, j])
            for k in range(n_sens):
                base = n_vars + k * n_vars
                start = 0 if k < n_y0_dirs else length - param_cols[k - n_y0_dirs]
                second_tangent(
                    coupling,
                    y,
                    t,
                    p_row,
                    y[base : base + n_vars],
                    0.0,
                    seed[start : start + n_params],
                    k_state,
                    0.0,
                    seed[0:n_params],
                    # The inner direction does not vary: zero here leaves
                    # the plain bilinear form.
                    seed[0:n_vars],
                    0.0,
                    seed[0:n_params],
                )
                for j in range(n_vars):
                    rhs[j] = lu_dtype(work[base + j] + coupling[j])
                solve_local(lu, ipiv, rhs)
                for j in range(n_vars):
                    k_stages[s, base + j] = np.float64(rhs[j])

    if spec is None:

        @cuda.jit(device=True)
        def put_stage_rhs(rhs_buf, work, j, value):
            rhs_buf[j] = lu_dtype(value)

    else:

        @cuda.jit(device=True)
        def put_stage_rhs(rhs_buf, work, j, value):
            # State rows go straight to the solver's buffer, which holds one
            # n_vars block per trajectory; sensitivity rows are staged until
            # their own solve, once the coupling term is known.
            if j < n_vars:
                rhs_buf[j] = lu_dtype(value)
            else:
                work[j] = value

    # --- the stages themselves, one device function per tableau row ----------
    # Every coefficient is a closure constant and every loop below has a
    # constant trip count, so each function lowers to the straight-line
    # arithmetic the row spelled out by hand would: the compiled kernels carry
    # the same instruction mix as the hand-unrolled form did. The split on
    # ``d`` keeps the three stages without a df/dt term from emitting an
    # ``x * 0.0`` the compiler may not drop.
    # Row ``s`` of the tableau's C holds ``s`` entries; stage 0 has none and
    # the rows are indexed by ``s - 1`` below.
    c_rows = (
        (C21,),
        (C31, C32),
        (C41, C42, C43),
        (C51, C52, C53, C54),
        (C61, C62, C63, C64, C65),
        (C71, C72, C73, C74, C75, C76),
        (C81, C82, C83, C84, C85, C86, C87),
    )
    d_row = (D1, D2, D3, D4, D5, 0.0, 0.0, 0.0)
    a_rows = (
        (A21,),
        (A31, A32),
        (A41, A42, A43),
        (A51, A52, A53, A54),
        (A61, A62, A63, A64, A65),
    )

    def make_stage(s: int) -> MLIRDispatcher:
        d = d_row[s]

        if s == 0:

            @cuda.jit(device=True)
            def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_row, t, dt_use, inv_dt):
                for j in range(size):
                    put_stage_rhs(rhs, work, j, work[j] + dt_use * d * dT[j])
                stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t)

        elif d != 0.0:
            c_row = c_rows[s - 1]

            @cuda.jit(device=True)
            def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_row, t, dt_use, inv_dt):
                for j in range(size):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    put_stage_rhs(
                        rhs, work, j, work[j] + dt_use * d * dT[j] + acc * inv_dt
                    )
                stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t)

        else:
            c_row = c_rows[s - 1]

            @cuda.jit(device=True)
            def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_row, t, dt_use, inv_dt):
                for j in range(size):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    put_stage_rhs(rhs, work, j, work[j] + acc * inv_dt)
                stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t)

        return stage

    def make_advance(s: int) -> MLIRDispatcher:
        a_row = a_rows[s]
        n_prev = s + 1

        @cuda.jit(device=True)
        def advance(u, y, k_stages):
            for j in range(size):
                acc = a_row[0] * k_stages[0, j]
                for k in range(1, n_prev):
                    acc += a_row[k] * k_stages[k, j]
                u[j] = y[j] + acc

        return advance

    stage_0, stage_1, stage_2, stage_3, stage_4, stage_5, stage_6, stage_7 = (
        make_stage(s) for s in range(8)
    )
    advance_0, advance_1, advance_2, advance_3, advance_4 = (
        make_advance(s) for s in range(5)
    )

    # Annotated rather than inferred: splatted into `cuda.jit`, a
    # `dict[str, int]` is matched against every keyword of numba's decorator in
    # turn and each non-`int` one is reported as a mismatch.
    jit_options: dict[str, Any] = (
        {}
        if options.max_registers is None
        else {"max_registers": options.max_registers}
    )

    @cuda.jit(**jit_options)
    def kernel(
        y0,
        times_all,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        time_rows,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        hook_out,
    ):
        # `blockIdx`/`threadIdx` exist only inside a compiled kernel;
        # numba materialises them during lowering, so a static reader
        # sees nothing on the `cuda` namespace here.
        i = cuda.blockIdx.x * tpb + cuda.threadIdx.x  # ty: ignore[unresolved-attribute]
        # Nothing in a step is collective any more, so a thread past the end of
        # the ensemble simply leaves rather than shadowing a trajectory through
        # barriers it would otherwise have to reach.
        if i >= y0.shape[0]:
            return
        diag = cuda.const.array_like(diag_table)

        # one save grid shared by the ensemble, or one row per trajectory
        n_save = times_all.shape[0] // time_rows
        t_row = (i if time_rows > 1 else 0) * n_save
        times = times_all[t_row : t_row + n_save]
        tf = times[n_save - 1]
        # A non-positive dt0 is the "no first step given" sentinel: start from
        # 1e-6 of the integration window.
        dt_init = dt0 if dt0 > 0.0 else (tf - times[0]) * 1e-6

        lu_buf = cuda.local.array(lu_size, lu_local_dtype)
        rhs_buf = cuda.local.array(n_vars, lu_local_dtype)
        ipiv_buf = cuda.local.array(ipiv_per, np.int32)

        y = cuda.local.array(size, np.float64)
        u = cuda.local.array(size, np.float64)
        work = cuda.local.array(size, np.float64)
        dT = cuda.local.array(size, np.float64)
        # The stage increments are what came out of the linear solve, so they
        # already carry only lu_precision; storing them wider keeps rounding
        # that is not there and doubles the thread's local-memory footprint,
        # and that footprint is what the local-memory traffic scales with.
        k_stages = cuda.local.array((8, size), lu_local_dtype)
        # The dense-output state a save hook is handed; a single dummy slot
        # when there is no hook, so the default kernel carries no extra local
        # memory.
        y_save = cuda.local.array(size if HAS_HOOK else 1, np.float64)

        for j in range(size):
            y[j] = y0[i, j]
        if SAVE_HISTORY:
            for j in range(size):
                hist[i, 0, j] = y[j]

        t = times[0]
        save_idx = 1
        dt = dt_init
        n_steps = 0
        accepted = 0
        rejected = 0
        err_prev = 1.0
        err_prev2 = 1.0

        p_row = params[i]

        if HAS_HOOK:
            for j in range(hook_size):
                hook_out[i, j] = 0.0
            save_hook(0, y, times[0], p_row, hook_out[i])

        # Purely this thread's loop: it steps until its own trajectory is done
        # and then falls out, with no reference to what the rest of the block is
        # doing.
        while save_idx < n_save and t < tf and n_steps < max_steps:
            dt_use = dt
            if dt_use > tf - t:
                dt_use = tf - t
            if dt_use < 1e-30:
                dt_use = 1e-30
            inv_dt = 1.0 / dt_use
            t_end = t + dt_use

            dtgamma_inv = 1.0 / (dt_use * GAMMA)
            write_negated_jacobian(y, t, p_row, lu_buf, dT)
            write_sensitivity_time_derivative(y, t, p_row, dT)
            for d in range(n_vars):
                lu_buf[diag[d]] += lu_dtype(dtgamma_inv)
            factorize_local(lu_buf, ipiv_buf)

            # The eight Rosenbrock stages. Each forms its right-hand side from
            # the tableau row, solves against the step's one factorisation, and
            # the u update between them is the next stage's argument.
            ode_write(y, t, p_row, work)
            stage_0(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_0(u, y, k_stages)
            ode_write(u, t + C2 * dt_use, p_row, work)
            stage_1(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_1(u, y, k_stages)
            ode_write(u, t + C3 * dt_use, p_row, work)
            stage_2(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_2(u, y, k_stages)
            ode_write(u, t + C4 * dt_use, p_row, work)
            stage_3(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_3(u, y, k_stages)
            ode_write(u, t + C5 * dt_use, p_row, work)
            stage_4(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_4(u, y, k_stages)
            # Stages 6-8 share the end point and accumulate into u (FSAL form).
            ode_write(u, t_end, p_row, work)
            stage_5(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            for j in range(size):
                u[j] += k_stages[5, j]
            ode_write(u, t_end, p_row, work)
            stage_6(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            for j in range(size):
                u[j] += k_stages[6, j]
            ode_write(u, t_end, p_row, work)
            stage_7(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )

            # Weighted RMS error estimate.
            err_local = 0.0
            for j in range(size):
                y_new_j = u[j] + k_stages[7, j]
                scale = atol + rtol * max(math.fabs(y[j]), math.fabs(y_new_j))
                r = weights[i, j] * k_stages[7, j] / scale
                err_local += r * r
            err_norm = math.sqrt(err_local / n_error)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

            factor, err_prev, err_prev2 = next_step(
                err_norm, err_prev, err_prev2, accept
            )
            dt = dt_use * factor

            if accept:
                t_old = t
                t_new = t_old + dt_use
                while save_idx < n_save and times[save_idx] <= t_new + 1e-12 * max(
                    1.0, math.fabs(times[save_idx])
                ):
                    theta = (times[save_idx] - t_old) / dt_use
                    theta1 = 1.0 - theta
                    for j in range(size):
                        h1 = (
                            25.948786856663858 * k_stages[0, j]
                            - 2.5579724845846235 * k_stages[1, j]
                            + 10.433815404888879 * k_stages[2, j]
                            - 2.3679251022685204 * k_stages[3, j]
                            + 0.524948541321073 * k_stages[4, j]
                            + 1.1241088310450404 * k_stages[5, j]
                            + 0.4272876194431874 * k_stages[6, j]
                            - 0.17202221070155493 * k_stages[7, j]
                        )
                        h2 = (
                            -9.91568850695171 * k_stages[0, j]
                            - 0.9689944594115154 * k_stages[1, j]
                            + 3.0438037242978453 * k_stages[2, j]
                            - 24.495224566215796 * k_stages[3, j]
                            + 20.176138334709044 * k_stages[4, j]
                            + 15.98066361424651 * k_stages[5, j]
                            - 6.789040303419874 * k_stages[6, j]
                            - 6.710236069923372 * k_stages[7, j]
                        )
                        h3 = (
                            11.419903575922262 * k_stages[0, j]
                            + 2.8879645146136994 * k_stages[1, j]
                            + 72.92137995996029 * k_stages[2, j]
                            + 80.12511834622643 * k_stages[3, j]
                            - 52.072871366152654 * k_stages[4, j]
                            - 59.78993625266729 * k_stages[5, j]
                            - 0.15582684282751913 * k_stages[6, j]
                            + 4.883087185713722 * k_stages[7, j]
                        )
                        y_new_j = u[j] + k_stages[7, j]
                        # Rosenbrock continuous extension between y_old and
                        # y_new. y[j] is still y_old here; it is advanced
                        # after this loop.
                        val = theta1 * y[j] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
                        if SAVE_HISTORY:
                            hist[i, save_idx, j] = val
                        if HAS_HOOK:
                            y_save[j] = val
                    if HAS_HOOK:
                        save_hook(save_idx, y_save, times[save_idx], p_row, hook_out[i])
                    save_idx += 1
                for j in range(size):
                    y[j] = u[j] + k_stages[7, j]
                t += dt_use
                accepted += 1
            else:
                rejected += 1
            n_steps += 1

        # Save times past this trajectory's own end time hold its final state.
        while save_idx < n_save:
            if SAVE_HISTORY:
                for j in range(size):
                    hist[i, save_idx, j] = y[j]
            if HAS_HOOK:
                save_hook(save_idx, y, times[save_idx], p_row, hook_out[i])
            save_idx += 1
        if not SAVE_HISTORY:
            # Without the history the one slot holds the final state.
            for j in range(size):
                hist[i, 0, j] = y[j]
        accepted_out[i] = accepted
        rejected_out[i] = rejected
        loop_out[i] = n_steps

    if options.cache:
        enable_content_cache(kernel)
    return kernel, tpb
