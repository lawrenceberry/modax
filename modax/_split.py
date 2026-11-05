"""Splitting a stiff system into one-way-coupled blocks, from its sparsity pattern.

A Jacobian pattern read as a graph -- row ``i`` depends on column ``j`` --
has strongly connected components, and their condensation is a DAG. A
component that no later component feeds back into is a *driver*: its
equations can be integrated on their own, and everything downstream can read
it as a known function of time. The thermal history under an
Einstein-Boltzmann hierarchy is the case this was built for (four states that
every photon multipole reads and that read nothing back), but nothing here
knows that: the split is found from the pattern alone.

With ``split=True`` the Rodas5P solve collects the components that read only
components already collected into one driver block, as long as it stays small,
takes the rest as the main block, and integrates them in order inside one
kernel:

1. The driver takes a Rodas5P step of its own over the outer step, on a dense
   matrix of its size, with a Jacobian that is always fresh (the end-point
   Jacobian of one step is the start-point Jacobian of the next).
2. Its values, first and second derivatives at both ends
   (``theta'' = J theta' + dg/dt``) are written into the thread's local copy
   of the parameter row as a **quintic Hermite** block. The interpolant is
   ``O(h^6)``, one order beyond the method's local error, so the main step
   keeps fifth order; a cubic would cost it one.
3. The main block takes its Rodas5P step with the driver states replaced, in
   a generated right-hand side, by that polynomial of ``t``. Its state
   Jacobian has no driver columns; Enzyme's time sweep differentiates through
   the polynomial, so ``df/dt`` carries ``df/dtheta theta'``.
4. One error norm over both embedded estimates accepts or rejects both.

Both block right-hand sides are generated from the user's single ``ode_fn``:
the driver's returns its rows with the main states set to zero (dead code
after inlining), the main's returns its rows with the driver states
interpolated. The history, the hook's state and the error weights all keep
the user's order, so the sensitivity rule and the ``vmap`` wrapper are
untouched.

Forward sensitivities ride on both blocks. The driver carries its own joint
system (its initial-state directions and the parameter directions); the main
block carries every direction, and for each one the parameter-direction seed
of its sweeps is the unit vector plus the Hermite coefficients of the driver's
sensitivity for that direction, so ``df/dtheta S_theta(t)`` comes out of the
same forward sweep as ``J_y S_y + J_p``. The sensitivities' second derivatives
for their quintic come from the second-order sweep the joint system already
uses.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba_cuda_mlir import cuda, types

from modax._codegen import compile_device_source
from modax._numba_common import as_cuda_device
from modax._sensitivity import SensitivitySpec, make_second_tangent, make_tangent
from modax._sparse_direct import sparse_direct_solver_for
from modax._sparsity import dense_jacobian

# --- the pattern's blocks ---------------------------------------------------------------


@dataclass(frozen=True)
class BlockSplit:
    """Which state indices (the user's) form the driver block and the main block."""

    driver: tuple[int, ...]
    main: tuple[int, ...]
    levels: tuple[tuple[int, ...], ...]  # the driver's components, in the order they joined

    @property
    def n_drv(self) -> int:
        return len(self.driver)

    @property
    def n_main(self) -> int:
        return len(self.main)


def components_in_order(pattern: tuple[tuple[int, ...], ...]) -> list[tuple[tuple[int, ...], frozenset[int]]]:
    """The pattern's strongly connected components in topological order, each with the states it reads.

    A pair ``(members, reads)`` per component: ``members`` its state indices,
    ``reads`` the indices outside it that its rows depend on. Nothing reads a
    component listed after it.
    """
    import networkx as nx

    n = len(pattern)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for row, cols in enumerate(pattern):
        for col in cols:
            if col != row:
                graph.add_edge(col, row)  # row depends on col
    condensation = nx.condensation(graph)
    ordered = []
    for node in nx.topological_sort(condensation):
        members = tuple(sorted(condensation.nodes[node]["members"]))
        reads = frozenset(c for i in members for c in pattern[i]) - frozenset(members)
        ordered.append((members, reads))
    return ordered


def analyse_blocks(
    pattern: tuple[tuple[int, ...], ...], max_driver: int
) -> BlockSplit | None:
    """The driver and main blocks of a pattern, or ``None`` when it does not split.

    The strongly connected components are taken in topological order, and
    each joins the driver when everything it reads is already there and the
    driver stays within ``max_driver`` states, short of the whole system; the
    rest is the main block. So
    the driver is closed under "depends on", whatever the rows the main block
    does not feed (an optical depth that only integrates the driver's opacity
    joins it too), and a superset pattern, which can only merge components,
    is safe: at worst nothing splits.
    """
    components = components_in_order(pattern)
    driver: set[int] = set()
    levels: list[tuple[int, ...]] = []
    for members, reads in components:
        size = len(driver) + len(members)
        if reads <= driver and size <= max_driver and size < len(pattern):
            driver.update(members)
            levels.append(members)
    if not driver:
        return None
    main = sorted(set(range(len(pattern))) - driver)
    return BlockSplit(tuple(sorted(driver)), tuple(main), tuple(levels))


def restrict_pattern(pattern, indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """The pattern's rows ``indices`` on columns ``indices``, renumbered."""
    position = {idx: k for k, idx in enumerate(indices)}
    return tuple(
        tuple(sorted(position[c] for c in pattern[i] if c in position)) for i in indices
    )


# --- the generated right-hand sides -------------------------------------------------


def hermite_slot(drv_col: int, j: int) -> int:
    """First of the six Hermite coefficients of driver state ``j`` in the local parameter row."""
    return drv_col + 2 + 6 * j


def hermite_width(n_drv: int) -> int:
    return 2 + 6 * n_drv


_QUINTIC = [
    "    s2 = s * s",
    "    s3 = s2 * s",
    "    s4 = s3 * s",
    "    s5 = s4 * s",
    "    b0 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5",
    "    b1 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5",
    "    b2 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5)",
    "    b3 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5",
    "    b4 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5",
    "    b5 = 0.5 * (s3 - 2.0 * s4 + s5)",
]


def generate_driver_rhs(ode_fn, n_vars: int, split: BlockSplit):
    """``g(theta, t, p) -> tuple`` of the driver rows; the main states are zero and dead."""
    pos = {idx: k for k, idx in enumerate(split.driver)}
    y = ", ".join(f"th[{pos[i]}]" if i in pos else "0.0" for i in range(n_vars))
    lines = [
        "def driver_rhs(th, t, p):",
        f"    y = ({y},)",
        "    f = ode_fn(y, t, p)",
        "    return (" + ", ".join(f"f[{i}]" for i in split.driver) + ",)",
    ]
    return compile_device_source("driver_rhs", lines, {"ode_fn": as_cuda_device(ode_fn)})


def generate_main_rhs(ode_fn, n_vars: int, split: BlockSplit, drv_col: int):
    """``f(y_main, t, p) -> tuple`` of the main rows, the driver states from the Hermite block of ``p``."""
    dpos = {idx: k for k, idx in enumerate(split.driver)}
    mpos = {idx: k for k, idx in enumerate(split.main)}
    lines = [
        "def main_rhs(ym, t, p):",
        f"    hh = p[{drv_col + 1}]",
        f"    s = (t - p[{drv_col}]) / hh",
        *_QUINTIC,
        "    hh2 = hh * hh",
    ]
    for j in range(split.n_drv):
        b = hermite_slot(drv_col, j)
        lines.append(
            f"    d{j} = b0 * p[{b}] + hh * b1 * p[{b + 1}] + hh2 * b2 * p[{b + 2}]"
            f" + b3 * p[{b + 3}] + hh * b4 * p[{b + 4}] + hh2 * b5 * p[{b + 5}]"
        )
    y = ", ".join(f"d{dpos[i]}" if i in dpos else f"ym[{mpos[i]}]" for i in range(n_vars))
    lines += [
        f"    y = ({y},)",
        "    f = ode_fn(y, t, p)",
        "    return (" + ", ".join(f"f[{i}]" for i in split.main) + ",)",
    ]
    return compile_device_source("main_rhs", lines, {"ode_fn": as_cuda_device(ode_fn)})


@cuda.jit(device=True)
def quintic(c0, c1, c2, c3, c4, c5, s, h):
    """The quintic Hermite interpolant of ``(c0, c1, c2)`` at 0 and ``(c3, c4, c5)`` at ``h``, at ``s = (t - t0)/h``."""
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    return (
        (1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5) * c0
        + h * (s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5) * c1
        + h * h * 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5) * c2
        + (10.0 * s3 - 15.0 * s4 + 6.0 * s5) * c3
        + h * (-4.0 * s3 + 7.0 * s4 - 3.0 * s5) * c4
        + h * h * 0.5 * (s3 - 2.0 * s4 + s5) * c5
    )


# --- the kernel -------------------------------------------------------------------------


def _rows(coeff_rows):
    return tuple(tuple(float(c) for c in row) for row in coeff_rows)


@functools.cache
def make_split_kernel(ode_fn, n_vars: int, n_params: int, options, split: BlockSplit):
    """Rodas5P with the driver block integrated ahead of the main block, one trajectory per thread.

    ``options`` is the :class:`modax.rodas5P.KernelOptions`; the tableau and
    the dense-output coefficients are the kernel's own. Returns
    ``(kernel, trajectories_per_block)``; the kernel takes the standard
    argument list and writes the history in the user's layout.
    """
    from modax import rodas5P as R

    spec: SensitivitySpec | None = options.spec
    D, M = split.driver, split.main
    n_drv, n_main = split.n_drv, split.n_main
    drv_col = n_params
    n_params_local = n_params + hermite_width(n_drv)
    tpb = int(options.trajectories_per_block)
    lu_dtype = np.float32 if options.lu_precision == "fp32" else np.float64
    lu_local_dtype = types.float32 if options.lu_precision == "fp32" else types.float64
    tf_index = options.tf_index
    HAS_HOOK = options.save_hook is not None
    SAVE_HISTORY = bool(options.save_history)
    hook_size = max(1, int(options.hook_size))
    if HAS_HOOK:
        save_hook = as_cuda_device(options.save_hook)
    else:

        @cuda.jit(device=True)
        def save_hook(save_idx, y, t, p_row, acc):
            return

    e1 = R.EXPONENT * (options.icoeff + options.pcoeff + options.dcoeff)
    e2 = -R.EXPONENT * (options.pcoeff + 2.0 * options.dcoeff)
    e3 = R.EXPONENT * options.dcoeff
    GAMMA, SAFETY, FACTOR_MIN, FACTOR_MAX = R.GAMMA, R.SAFETY, R.FACTOR_MIN, R.FACTOR_MAX
    C2, C3, C4, C5 = R.C2, R.C3, R.C4, R.C5
    c_rows = _rows(((), (R.C21,), (R.C31, R.C32), (R.C41, R.C42, R.C43), (R.C51, R.C52, R.C53, R.C54),
                    (R.C61, R.C62, R.C63, R.C64, R.C65), (R.C71, R.C72, R.C73, R.C74, R.C75, R.C76),
                    (R.C81, R.C82, R.C83, R.C84, R.C85, R.C86, R.C87)))
    d_row = (R.D1, R.D2, R.D3, R.D4, R.D5, 0.0, 0.0, 0.0)
    a_rows = _rows(((R.A21,), (R.A31, R.A32), (R.A41, R.A42, R.A43), (R.A51, R.A52, R.A53, R.A54),
                    (R.A61, R.A62, R.A63, R.A64, R.A65)))

    # -- directions -----------------------------------------------------------------
    # The user's directions: the initial-state block (one per user state, when
    # asked for) then the parameter block. The driver carries its own
    # initial-state directions and the parameter directions; a main-state
    # initial direction has no driver sensitivity at all.
    n_sens = spec.n_sens if spec is not None else 0
    n_y0_dirs = spec.n_y0_dirs if spec is not None else 0
    param_cols = tuple(spec.param_seed_columns) if (spec is not None and spec.wrt_params) else ()
    n_param_dirs = len(param_cols)
    wrt_y0 = spec is not None and spec.wrt_y0
    n_drv_y0 = n_drv if wrt_y0 else 0
    n_sens_d = n_drv_y0 + n_param_dirs
    dpos = {idx: k for k, idx in enumerate(D)}
    dir_to_drv = np.full(max(1, n_sens), -1, dtype=np.int32)  # user direction -> driver direction
    dir_param_col = np.full(max(1, n_sens), -1, dtype=np.int32)  # user direction -> parameter column
    for k in range(n_sens):
        if k < n_y0_dirs:
            if k in dpos:
                dir_to_drv[k] = dpos[k]
        else:
            dir_to_drv[k] = n_drv_y0 + (k - n_y0_dirs)
            dir_param_col[k] = param_cols[k - n_y0_dirs]
    drv_to_dir = np.full(max(1, n_sens_d), -1, dtype=np.int32)
    for k in range(n_sens):
        if dir_to_drv[k] >= 0:
            drv_to_dir[dir_to_drv[k]] = k
    error_control = spec.error_control if spec is not None else True
    size_d = n_drv * (1 + n_sens_d)
    size_m = n_main * (1 + n_sens)
    n_error = (n_vars + (n_sens * n_vars if error_control else 0))
    D_idx = np.asarray(D, dtype=np.int32)
    M_idx = np.asarray(M, dtype=np.int32)
    HSEED_ROWS = max(1, n_sens)
    HSEED_COLS = max(1, 6 * n_drv)
    N_AUG = n_vars * (1 + n_sens)
    Y_SAVE_LEN = N_AUG if options.save_hook is not None else 1
    # numba's SSA pass recurses once per basic block, and this kernel has many.
    import sys

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    # -- right-hand sides and derivatives ---------------------------------------------
    driver_rhs = generate_driver_rhs(ode_fn, n_vars, split)
    main_rhs = generate_main_rhs(ode_fn, n_vars, split, drv_col)
    drv_tangent = make_tangent(driver_rhs, n_drv, n_params_local)
    main_tangent = make_tangent(main_rhs, n_main, n_params_local)
    # numba types every call site, dead or not, and typing a jvp builds it, so
    # the second-order sweeps exist (and are referenced) only with directions.
    HAS_SENS = n_sens > 0
    drv_second = make_second_tangent(driver_rhs, n_drv, n_params_local) if HAS_SENS else None
    main_second = make_second_tangent(main_rhs, n_main, n_params_local) if HAS_SENS else None
    driver_device = as_cuda_device(driver_rhs)
    main_device = as_cuda_device(main_rhs)

    # the main block's matrix
    if options.sparsity is None:
        main_structure = dense_jacobian(n_main)
        m_factorize, m_solve = R.dense_lu_solver(n_main)
        m_ipiv = n_main
    else:
        main_solver = sparse_direct_solver_for(restrict_pattern(options.sparsity, M), options.ordering)
        main_structure = main_solver.compressed
        m_factorize, m_solve = main_solver.factorize_local, main_solver.solve_local
        m_ipiv = main_solver.ipiv_size
    m_lu_size = main_structure.size
    m_diag = np.asarray(main_structure.diagonal, dtype=np.int32)
    m_store = main_structure.store_slots()
    m_needs_clear = m_store is not None and m_lu_size > int((m_store >= 0).sum())
    d_structure = dense_jacobian(n_drv)
    d_factorize, d_solve = R.dense_lu_solver(n_drv)
    d_lu_size = n_drv * n_drv
    d_diag = np.asarray(d_structure.diagonal, dtype=np.int32)

    if m_needs_clear:

        @cuda.jit(device=True)
        def m_clear(lu):
            for i in range(m_lu_size):
                lu[i] = lu_dtype(0.0)

    else:

        @cuda.jit(device=True)
        def m_clear(lu):
            pass

    @cuda.jit(device=True)
    def d_clear(lu):
        pass

    write_main_jacobian = R._make_literal_seed_jacobian(
        n_vars=n_main, n_params=n_params_local, n_colours=main_structure.n_colours,
        store_table=m_store,
        namespace=dict(cuda=cuda, float64=types.float64, tangent_of=main_tangent,
                       colour_seeds=main_structure.seed_table(n_params_local), clear_matrix=m_clear,
                       lu_dtype=lu_dtype),
    )
    write_driver_jacobian = R._make_literal_seed_jacobian(
        n_vars=n_drv, n_params=n_params_local, n_colours=n_drv, store_table=None,
        namespace=dict(cuda=cuda, float64=types.float64, tangent_of=drv_tangent,
                       colour_seeds=d_structure.seed_table(n_params_local), clear_matrix=d_clear,
                       lu_dtype=np.float64),
    )

    # -- the seeds of the sensitivity sweeps ----------------------------------------
    # pbuf is a thread-local parameter-direction vector: a unit vector on the
    # direction's parameter column, plus, for the main block, the Hermite
    # coefficients of the driver's sensitivity in that direction (hseed), so
    # that the sweep through the polynomial yields df/dtheta S_theta(t).
    P_LOCAL = n_params_local
    ZERO_LEN = max(n_main, n_drv, n_params_local)
    dir_param_col_t = dir_param_col
    dir_to_drv_t = dir_to_drv
    drv_to_dir_t = drv_to_dir

    @cuda.jit(device=True)
    def seed_main(k, pbuf, hseed):
        for c in range(P_LOCAL):
            pbuf[c] = 0.0
        col = cuda.const.array_like(dir_param_col_t)[k]
        if col >= 0:
            pbuf[col] = 1.0
        kd = cuda.const.array_like(dir_to_drv_t)[k]
        if kd >= 0:
            for c in range(6 * n_drv):
                pbuf[drv_col + 2 + c] = hseed[k, c]

    @cuda.jit(device=True)
    def seed_driver(kd, pbuf):
        for c in range(P_LOCAL):
            pbuf[c] = 0.0
        k = cuda.const.array_like(drv_to_dir_t)[kd]
        col = cuda.const.array_like(dir_param_col_t)[k]
        if col >= 0:
            pbuf[col] = 1.0

    # -- joint right-hand sides [f, J S + J_p] ---------------------------------------
    @cuda.jit(device=True)
    def write_driver(z, t, p_local, out, pbuf, zero):
        values = driver_device(z, t, p_local)
        for j in range(n_drv):
            out[j] = values[j]
        tangent = cuda.local.array(n_drv, types.float64)
        for kd in range(n_sens_d):
            base = n_drv + kd * n_drv
            seed_driver(kd, pbuf)
            drv_tangent(tangent, z, t, p_local, z[base : base + n_drv], 0.0, pbuf)
            for r in range(n_drv):
                out[base + r] = tangent[r]

    @cuda.jit(device=True)
    def write_main(z, t, p_local, out, pbuf, hseed, zero):
        values = main_device(z, t, p_local)
        for j in range(n_main):
            out[j] = values[j]
        tangent = cuda.local.array(n_main, types.float64)
        for k in range(n_sens):
            base = n_main + k * n_main
            seed_main(k, pbuf, hseed)
            main_tangent(tangent, z, t, p_local, z[base : base + n_main], 0.0, pbuf)
            for r in range(n_main):
                out[base + r] = tangent[r]

    # dF/dt of the sensitivity rows (second-order sweep seeded with the time direction)
    if HAS_SENS:

        @cuda.jit(device=True)
        def write_driver_sens_dt(z, t, p_local, dT, pbuf, zero):
            column = cuda.local.array(n_drv, types.float64)
            for kd in range(n_sens_d):
                base = n_drv + kd * n_drv
                seed_driver(kd, pbuf)
                drv_second(column, z, t, p_local, z[base : base + n_drv], 0.0, pbuf,
                           zero[0:n_drv], 1.0, zero[0:P_LOCAL], zero[0:n_drv], 0.0, zero[0:P_LOCAL])
                for r in range(n_drv):
                    dT[base + r] = column[r]

        @cuda.jit(device=True)
        def write_main_sens_dt(z, t, p_local, dT, pbuf, hseed, zero):
            column = cuda.local.array(n_main, types.float64)
            for k in range(n_sens):
                base = n_main + k * n_main
                seed_main(k, pbuf, hseed)
                main_second(column, z, t, p_local, z[base : base + n_main], 0.0, pbuf,
                            zero[0:n_main], 1.0, zero[0:P_LOCAL], zero[0:n_main], 0.0, zero[0:P_LOCAL])
                for r in range(n_main):
                    dT[base + r] = column[r]

    else:

        @cuda.jit(device=True)
        def write_driver_sens_dt(z, t, p_local, dT, pbuf, zero):
            pass

        @cuda.jit(device=True)
        def write_main_sens_dt(z, t, p_local, dT, pbuf, hseed, zero):
            pass

    # -- one Rosenbrock stage of a block (state rows, then sensitivity rows by forward substitution)
    def make_block(n, size, n_sens_b, solve_local, lu_dt, second, seed_fn, with_hseed):
        """The stage, advance and solve device functions of one block."""

        if n_sens_b == 0:

            @cuda.jit(device=True)
            def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_local, t, pbuf, hseed, zero):
                solve_local(lu, ipiv, rhs)
                for j in range(n):
                    k_stages[s, j] = np.float64(rhs[j])

        elif with_hseed:

            @cuda.jit(device=True)
            def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_local, t, pbuf, hseed, zero):
                solve_local(lu, ipiv, rhs)
                for j in range(n):
                    k_stages[s, j] = np.float64(rhs[j])
                coupling = cuda.local.array(n, types.float64)
                k_state = cuda.local.array(n, types.float64)
                for j in range(n):
                    k_state[j] = np.float64(k_stages[s, j])
                for k in range(n_sens_b):
                    base = n + k * n
                    seed_fn(k, pbuf, hseed)
                    second(coupling, y, t, p_local, y[base : base + n], 0.0, pbuf,
                           k_state, 0.0, zero[0:P_LOCAL], zero[0:n], 0.0, zero[0:P_LOCAL])
                    for j in range(n):
                        rhs[j] = lu_dt(work[base + j] + coupling[j])
                    solve_local(lu, ipiv, rhs)
                    for j in range(n):
                        k_stages[s, base + j] = np.float64(rhs[j])

        else:

            @cuda.jit(device=True)
            def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_local, t, pbuf, hseed, zero):
                solve_local(lu, ipiv, rhs)
                for j in range(n):
                    k_stages[s, j] = np.float64(rhs[j])
                coupling = cuda.local.array(n, types.float64)
                k_state = cuda.local.array(n, types.float64)
                for j in range(n):
                    k_state[j] = np.float64(k_stages[s, j])
                for k in range(n_sens_b):
                    base = n + k * n
                    seed_fn(k, pbuf)
                    second(coupling, y, t, p_local, y[base : base + n], 0.0, pbuf,
                           k_state, 0.0, zero[0:P_LOCAL], zero[0:n], 0.0, zero[0:P_LOCAL])
                    for j in range(n):
                        rhs[j] = lu_dt(work[base + j] + coupling[j])
                    solve_local(lu, ipiv, rhs)
                    for j in range(n):
                        k_stages[s, base + j] = np.float64(rhs[j])

        # The state rows go to the solver's buffer, the sensitivity rows stay in
        # ``work`` until their own solve -- as two loops, never as a branch
        # choosing between the two arrays, which the compiler turns into a
        # select of memref descriptors that the LLVM 7 translation cannot type.
        def make_stage(s):
            c_row = c_rows[s]
            d = d_row[s]
            if s == 0:

                @cuda.jit(device=True)
                def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero):
                    for j in range(n):
                        rhs[j] = lu_dt(work[j] + dt_use * d * dT[j])
                    for j in range(n, size):
                        work[j] = work[j] + dt_use * d * dT[j]
                    stage_solve(lu, ipiv, rhs, k_stages, 0, work, y, p_local, t, pbuf, hseed, zero)

            elif d != 0.0:

                @cuda.jit(device=True)
                def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero):
                    for j in range(n):
                        acc = c_row[0] * k_stages[0, j]
                        for k in range(1, s):
                            acc += c_row[k] * k_stages[k, j]
                        rhs[j] = lu_dt(work[j] + dt_use * d * dT[j] + acc * inv_dt)
                    for j in range(n, size):
                        acc = c_row[0] * k_stages[0, j]
                        for k in range(1, s):
                            acc += c_row[k] * k_stages[k, j]
                        work[j] = work[j] + dt_use * d * dT[j] + acc * inv_dt
                    stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_local, t, pbuf, hseed, zero)

            else:

                @cuda.jit(device=True)
                def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero):
                    for j in range(n):
                        acc = c_row[0] * k_stages[0, j]
                        for k in range(1, s):
                            acc += c_row[k] * k_stages[k, j]
                        rhs[j] = lu_dt(work[j] + acc * inv_dt)
                    for j in range(n, size):
                        acc = c_row[0] * k_stages[0, j]
                        for k in range(1, s):
                            acc += c_row[k] * k_stages[k, j]
                        work[j] = work[j] + acc * inv_dt
                    stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_local, t, pbuf, hseed, zero)

            return stage

        def make_advance(s):
            a_row = a_rows[s]

            @cuda.jit(device=True)
            def advance(u, y, k_stages):
                for j in range(size):
                    acc = a_row[0] * k_stages[0, j]
                    for k in range(1, s + 1):
                        acc += a_row[k] * k_stages[k, j]
                    u[j] = y[j] + acc

            return advance

        stages = tuple(make_stage(s) for s in range(8))
        advances = tuple(make_advance(s) for s in range(5))
        return stages, advances

    (ms0, ms1, ms2, ms3, ms4, ms5, ms6, ms7), (ma0, ma1, ma2, ma3, ma4) = make_block(
        n_main, size_m, n_sens, m_solve, lu_dtype, main_second, seed_main, True
    )
    (ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7), (da0, da1, da2, da3, da4) = make_block(
        n_drv, size_d, n_sens_d, d_solve, np.float64, drv_second, seed_driver, False
    )

    @cuda.jit(device=True)
    def driver_step(th, u, work, dT, kd, lu, ipiv, rhs, p_local, t, dt_use, pbuf, zero):
        """One Rodas5P step of the driver block from ``th``; leaves ``u + kd[7]`` as the candidate."""
        inv_dt = 1.0 / dt_use
        dtgamma_inv = inv_dt / GAMMA
        for d_ in range(n_drv):
            lu[d_ * n_drv + d_] += dtgamma_inv
        d_factorize(lu, ipiv)
        # work holds [g, S'] at th (FSAL)
        ds0(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        da0(u, th, kd)
        write_driver(u, t + C2 * dt_use, p_local, work, pbuf, zero)
        ds1(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        da1(u, th, kd)
        write_driver(u, t + C3 * dt_use, p_local, work, pbuf, zero)
        ds2(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        da2(u, th, kd)
        write_driver(u, t + C4 * dt_use, p_local, work, pbuf, zero)
        ds3(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        da3(u, th, kd)
        write_driver(u, t + C5 * dt_use, p_local, work, pbuf, zero)
        ds4(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        da4(u, th, kd)
        write_driver(u, t + dt_use, p_local, work, pbuf, zero)
        ds5(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        for j in range(size_d):
            u[j] += kd[5, j]
        write_driver(u, t + dt_use, p_local, work, pbuf, zero)
        ds6(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)
        for j in range(size_d):
            u[j] += kd[6, j]
        write_driver(u, t + dt_use, p_local, work, pbuf, zero)
        ds7(lu, ipiv, rhs, kd, work, dT, th, p_local, t, dt_use, inv_dt, pbuf, zero, zero)

    @cuda.jit(device=True)
    def main_step(y, u, work, dT, k_stages, lu, ipiv, rhs, p_local, t, dt_use, pbuf, hseed, zero):
        """One Rodas5P step of the main block from ``y``; leaves ``u + k_stages[7]`` as the candidate."""
        inv_dt = 1.0 / dt_use
        dtgamma_inv = inv_dt / GAMMA
        diag = cuda.const.array_like(m_diag)
        write_main_jacobian(y, t, p_local, lu, dT)
        write_main_sens_dt(y, t, p_local, dT, pbuf, hseed, zero)
        for d_ in range(n_main):
            lu[diag[d_]] += lu_dtype(dtgamma_inv)
        m_factorize(lu, ipiv)
        write_main(y, t, p_local, work, pbuf, hseed, zero)
        ms0(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        ma0(u, y, k_stages)
        write_main(u, t + C2 * dt_use, p_local, work, pbuf, hseed, zero)
        ms1(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        ma1(u, y, k_stages)
        write_main(u, t + C3 * dt_use, p_local, work, pbuf, hseed, zero)
        ms2(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        ma2(u, y, k_stages)
        write_main(u, t + C4 * dt_use, p_local, work, pbuf, hseed, zero)
        ms3(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        ma3(u, y, k_stages)
        write_main(u, t + C5 * dt_use, p_local, work, pbuf, hseed, zero)
        ms4(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        ma4(u, y, k_stages)
        write_main(u, t + dt_use, p_local, work, pbuf, hseed, zero)
        ms5(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        for j in range(size_m):
            u[j] += k_stages[5, j]
        write_main(u, t + dt_use, p_local, work, pbuf, hseed, zero)
        ms6(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)
        for j in range(size_m):
            u[j] += k_stages[6, j]
        write_main(u, t + dt_use, p_local, work, pbuf, hseed, zero)
        ms7(lu, ipiv, rhs, k_stages, work, dT, y, p_local, t, dt_use, inv_dt, pbuf, hseed, zero)

    @cuda.jit(device=True)
    def state_second_derivative(g, neg_a, dT, s2):
        """``theta'' = J g + dg/dt`` for the driver state."""
        for r in range(n_drv):
            acc = dT[r]
            for c in range(n_drv):
                acc -= neg_a[r * n_drv + c] * g[c]
            s2[r] = acc

    if HAS_SENS and n_sens_d > 0:

        @cuda.jit(device=True)
        def second_derivatives(th, g, neg_a, dT, t, p_local, s2, pbuf, zero):
            """``theta''`` for the state and, by the second-order sweep, ``S''`` for every driver direction."""
            state_second_derivative(g, neg_a, dT, s2)
            column = cuda.local.array(n_drv, types.float64)
            for kd in range(n_sens_d):
                base = n_drv + kd * n_drv
                seed_driver(kd, pbuf)
                # D2 g[(S_k, 0, e_k), (g, 1, 0)] + D g[(S_k', 0, 0)]
                drv_second(column, th, t, p_local, th[base : base + n_drv], 0.0, pbuf,
                           g[0:n_drv], 1.0, zero[0:P_LOCAL], g[base : base + n_drv], 0.0, zero[0:P_LOCAL])
                for r in range(n_drv):
                    s2[base + r] = column[r]

    else:

        @cuda.jit(device=True)
        def second_derivatives(th, g, neg_a, dT, t, p_local, s2, pbuf, zero):
            state_second_derivative(g, neg_a, dT, s2)

    @cuda.jit(device=True)
    def write_hermite(p_local, hseed, t, dt_use, th0, g0, s20, th1, g1, s21):
        """The quintic Hermite block of the state into ``p_local``, of the sensitivities into ``hseed``."""
        p_local[drv_col] = t
        p_local[drv_col + 1] = dt_use
        for j in range(n_drv):
            b = drv_col + 2 + 6 * j
            p_local[b] = th0[j]
            p_local[b + 1] = g0[j]
            p_local[b + 2] = s20[j]
            p_local[b + 3] = th1[j]
            p_local[b + 4] = g1[j]
            p_local[b + 5] = s21[j]
        for k in range(n_sens):
            kd = cuda.const.array_like(dir_to_drv_t)[k]
            for j in range(n_drv):
                c = 6 * j
                if kd >= 0:
                    base = n_drv + kd * n_drv + j
                    hseed[k, c] = th0[base]
                    hseed[k, c + 1] = g0[base]
                    hseed[k, c + 2] = s20[base]
                    hseed[k, c + 3] = th1[base]
                    hseed[k, c + 4] = g1[base]
                    hseed[k, c + 5] = s21[base]
                else:
                    for q in range(6):
                        hseed[k, c + q] = 0.0

    jit_options: dict[str, Any] = {} if options.max_registers is None else {"max_registers": options.max_registers}
    import os as _os

    if _os.environ.get("MODAX_SPLIT_DUMP_MLIR"):
        jit_options["dump_mlir"] = True

    @cuda.jit(**jit_options)
    def kernel(y0, times, params, dt0, rtol, atol, max_steps, weights, hist, accepted_out, rejected_out, loop_out, hook_out):
        i = cuda.blockIdx.x * tpb + cuda.threadIdx.x  # ty: ignore[unresolved-attribute]
        if i >= y0.shape[0]:
            return
        Dix = cuda.const.array_like(D_idx)
        Mix = cuda.const.array_like(M_idx)
        d2dir = cuda.const.array_like(drv_to_dir_t)

        n_save = times.shape[0]
        tf = times[n_save - 1]
        dt_init = dt0 if dt0 > 0.0 else (tf - times[0]) * 1e-6

        # main block
        lu_buf = cuda.local.array(m_lu_size, lu_local_dtype)
        rhs_buf = cuda.local.array(n_main, lu_local_dtype)
        ipiv_buf = cuda.local.array(m_ipiv, np.int32)
        y = cuda.local.array(size_m, np.float64)
        u = cuda.local.array(size_m, np.float64)
        work = cuda.local.array(size_m, np.float64)
        dT = cuda.local.array(size_m, np.float64)
        k_stages = cuda.local.array((8, size_m), lu_local_dtype)
        # driver block
        th = cuda.local.array(size_d, np.float64)
        th1 = cuda.local.array(size_d, np.float64)
        ud = cuda.local.array(size_d, np.float64)
        workd = cuda.local.array(size_d, np.float64)
        g0 = cuda.local.array(size_d, np.float64)
        g1 = cuda.local.array(size_d, np.float64)
        s20 = cuda.local.array(size_d, np.float64)
        s21 = cuda.local.array(size_d, np.float64)
        dTd = cuda.local.array(size_d, np.float64)
        dTd1 = cuda.local.array(size_d, np.float64)
        kd = cuda.local.array((8, size_d), np.float64)
        neg_a = cuda.local.array(d_lu_size, np.float64)
        neg_a1 = cuda.local.array(d_lu_size, np.float64)
        lu_d = cuda.local.array(d_lu_size, np.float64)
        ipiv_d = cuda.local.array(n_drv, np.int32)
        rhs_d = cuda.local.array(n_drv, np.float64)
        # shared scratch
        p_local = cuda.local.array(P_LOCAL, np.float64)
        pbuf = cuda.local.array(P_LOCAL, np.float64)
        zero = cuda.local.array(ZERO_LEN, np.float64)
        hseed = cuda.local.array((HSEED_ROWS, HSEED_COLS), np.float64)
        y_save = cuda.local.array(Y_SAVE_LEN, np.float64)

        for c in range(ZERO_LEN):
            zero[c] = 0.0
        for c in range(P_LOCAL):
            p_local[c] = 0.0
        for c in range(n_params):
            p_local[c] = params[i, c]
        p_local[drv_col + 1] = 1.0
        for k in range(HSEED_ROWS):
            for c in range(HSEED_COLS):
                hseed[k, c] = 0.0

        # the initial state, from the user's layout
        for j in range(n_drv):
            th[j] = y0[i, Dix[j]]
        for r in range(n_main):
            y[r] = y0[i, Mix[r]]
        for kd_ in range(n_sens_d):
            k = d2dir[kd_]
            for j in range(n_drv):
                th[n_drv + kd_ * n_drv + j] = y0[i, n_vars + k * n_vars + Dix[j]]
        for k in range(n_sens):
            for r in range(n_main):
                y[n_main + k * n_main + r] = y0[i, n_vars + k * n_vars + Mix[r]]
        if SAVE_HISTORY:
            for j in range(N_AUG):
                hist[i, 0, j] = y0[i, j]

        t = times[0]
        save_idx = 1
        dt = dt_init
        n_steps = 0
        accepted = 0
        rejected = 0
        err_prev = 1.0
        err_prev2 = 1.0
        tf_local = tf
        if tf_index >= 0:
            tf_local = params[i, tf_index]

        # the driver's start-point derivatives: Jacobian, joint rate, second derivatives
        write_driver_jacobian(th, t, p_local, neg_a, dTd)
        write_driver_sens_dt(th, t, p_local, dTd, pbuf, zero)
        write_driver(th, t, p_local, g0, pbuf, zero)
        second_derivatives(th, g0, neg_a, dTd, t, p_local, s20, pbuf, zero)

        if HAS_HOOK:
            for j in range(hook_size):
                hook_out[i, j] = 0.0
            for j in range(N_AUG):
                y_save[j] = y0[i, j]
            save_hook(0, y_save, times[0], p_local, hook_out[i])

        while save_idx < n_save and t < tf_local and n_steps < max_steps:
            dt_use = dt
            if dt_use > tf_local - t:
                dt_use = tf_local - t
            if dt_use < 1e-30:
                dt_use = 1e-30
            t_end = t + dt_use

            # 1. the driver over [t, t_end], against its fresh Jacobian
            for c in range(d_lu_size):
                lu_d[c] = neg_a[c]
            for j in range(size_d):
                workd[j] = g0[j]
            driver_step(th, ud, workd, dTd, kd, lu_d, ipiv_d, rhs_d, p_local, t, dt_use, pbuf, zero)
            for j in range(size_d):
                th1[j] = ud[j] + kd[7, j]
            # its end-point derivatives (next step's start-point ones, if accepted)
            write_driver_jacobian(th1, t_end, p_local, neg_a1, dTd1)
            write_driver_sens_dt(th1, t_end, p_local, dTd1, pbuf, zero)
            write_driver(th1, t_end, p_local, g1, pbuf, zero)
            second_derivatives(th1, g1, neg_a1, dTd1, t_end, p_local, s21, pbuf, zero)
            # 2. the Hermite block the main block reads
            write_hermite(p_local, hseed, t, dt_use, th, g0, s20, th1, g1, s21)
            # 3. the main block
            main_step(y, u, work, dT, k_stages, lu_buf, ipiv_buf, rhs_buf, p_local, t, dt_use, pbuf, hseed, zero)

            # 4. one error norm over both
            err_local = 0.0
            for r in range(n_main):
                y_new = u[r] + k_stages[7, r]
                scale = atol + rtol * max(math.fabs(y[r]), math.fabs(y_new))
                e = weights[i, Mix[r]] * k_stages[7, r] / scale
                err_local += e * e
            for j in range(n_drv):
                scale = atol + rtol * max(math.fabs(th[j]), math.fabs(th1[j]))
                e = weights[i, Dix[j]] * kd[7, j] / scale
                err_local += e * e
            for k in range(n_sens):
                for r in range(n_main):
                    jj = n_main + k * n_main + r
                    y_new = u[jj] + k_stages[7, jj]
                    scale = atol + rtol * max(math.fabs(y[jj]), math.fabs(y_new))
                    e = weights[i, n_vars + k * n_vars + Mix[r]] * k_stages[7, jj] / scale
                    err_local += e * e
            for kd_ in range(n_sens_d):
                k = d2dir[kd_]
                for j in range(n_drv):
                    jj = n_drv + kd_ * n_drv + j
                    scale = atol + rtol * max(math.fabs(th[jj]), math.fabs(th1[jj]))
                    e = weights[i, n_vars + k * n_vars + Dix[j]] * kd[7, jj] / scale
                    err_local += e * e
            err_norm = math.sqrt(err_local / n_error)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

            if math.isnan(err_norm) or err_norm > 1e18:
                safe_err = 1e18
            elif err_norm == 0.0:
                safe_err = 1e-18
            else:
                safe_err = err_norm
            factor = SAFETY * safe_err**e1 * err_prev**e2 * err_prev2**e3
            if accept:
                err_prev2 = err_prev
                err_prev = safe_err
            if factor < FACTOR_MIN:
                factor = FACTOR_MIN
            elif factor > FACTOR_MAX:
                factor = FACTOR_MAX
            dt = dt_use * factor

            if accept:
                t_old = t
                while save_idx < n_save and times[save_idx] <= t_end + 1e-12 * max(1.0, math.fabs(times[save_idx])):
                    theta = (times[save_idx] - t_old) / dt_use
                    theta1 = 1.0 - theta
                    # main block: the Rosenbrock continuous extension
                    for jj in range(size_m):
                        h1 = (25.948786856663858 * k_stages[0, jj] - 2.5579724845846235 * k_stages[1, jj]
                              + 10.433815404888879 * k_stages[2, jj] - 2.3679251022685204 * k_stages[3, jj]
                              + 0.524948541321073 * k_stages[4, jj] + 1.1241088310450404 * k_stages[5, jj]
                              + 0.4272876194431874 * k_stages[6, jj] - 0.17202221070155493 * k_stages[7, jj])
                        h2 = (-9.91568850695171 * k_stages[0, jj] - 0.9689944594115154 * k_stages[1, jj]
                              + 3.0438037242978453 * k_stages[2, jj] - 24.495224566215796 * k_stages[3, jj]
                              + 20.176138334709044 * k_stages[4, jj] + 15.98066361424651 * k_stages[5, jj]
                              - 6.789040303419874 * k_stages[6, jj] - 6.710236069923372 * k_stages[7, jj])
                        h3 = (11.419903575922262 * k_stages[0, jj] + 2.8879645146136994 * k_stages[1, jj]
                              + 72.92137995996029 * k_stages[2, jj] + 80.12511834622643 * k_stages[3, jj]
                              - 52.072871366152654 * k_stages[4, jj] - 59.78993625266729 * k_stages[5, jj]
                              - 0.15582684282751913 * k_stages[6, jj] + 4.883087185713722 * k_stages[7, jj])
                        y_new = u[jj] + k_stages[7, jj]
                        val = theta1 * y[jj] + theta * (y_new + theta1 * (h1 + theta * (h2 + theta * h3)))
                        k = jj // n_main
                        r = jj - k * n_main
                        col = Mix[r] if k == 0 else n_vars + (k - 1) * n_vars + Mix[r]
                        if SAVE_HISTORY:
                            hist[i, save_idx, col] = val
                        if HAS_HOOK:
                            y_save[col] = val
                    # driver block: the quintic Hermite of the state and its sensitivities
                    for j in range(n_drv):
                        val = quintic(th[j], g0[j], s20[j], th1[j], g1[j], s21[j], theta, dt_use)
                        if SAVE_HISTORY:
                            hist[i, save_idx, Dix[j]] = val
                        if HAS_HOOK:
                            y_save[Dix[j]] = val
                    for k in range(n_sens):
                        kd_ = cuda.const.array_like(dir_to_drv_t)[k]
                        for j in range(n_drv):
                            col = n_vars + k * n_vars + Dix[j]
                            val = 0.0
                            if kd_ >= 0:
                                jj = n_drv + kd_ * n_drv + j
                                val = quintic(th[jj], g0[jj], s20[jj], th1[jj], g1[jj], s21[jj], theta, dt_use)
                            if SAVE_HISTORY:
                                hist[i, save_idx, col] = val
                            if HAS_HOOK:
                                y_save[col] = val
                    if HAS_HOOK:
                        save_hook(save_idx, y_save, times[save_idx], p_local, hook_out[i])
                    save_idx += 1
                for jj in range(size_m):
                    y[jj] = u[jj] + k_stages[7, jj]
                for j in range(size_d):
                    th[j] = th1[j]
                    g0[j] = g1[j]
                    s20[j] = s21[j]
                    dTd[j] = dTd1[j]
                for c in range(d_lu_size):
                    neg_a[c] = neg_a1[c]
                t += dt_use
                accepted += 1
            else:
                rejected += 1
            n_steps += 1

        # save times past this trajectory's own end hold its final state
        while save_idx < n_save:
            for jj in range(size_m):
                k = jj // n_main
                r = jj - k * n_main
                col = Mix[r] if k == 0 else n_vars + (k - 1) * n_vars + Mix[r]
                if SAVE_HISTORY:
                    hist[i, save_idx, col] = y[jj]
                if HAS_HOOK:
                    y_save[col] = y[jj]
            for j in range(n_drv):
                if SAVE_HISTORY:
                    hist[i, save_idx, Dix[j]] = th[j]
                if HAS_HOOK:
                    y_save[Dix[j]] = th[j]
            for k in range(n_sens):
                kd_ = cuda.const.array_like(dir_to_drv_t)[k]
                for j in range(n_drv):
                    col = n_vars + k * n_vars + Dix[j]
                    val = th[n_drv + kd_ * n_drv + j] if kd_ >= 0 else 0.0
                    if SAVE_HISTORY:
                        hist[i, save_idx, col] = val
                    if HAS_HOOK:
                        y_save[col] = val
            if HAS_HOOK:
                save_hook(save_idx, y_save, times[save_idx], p_local, hook_out[i])
            save_idx += 1
        if not SAVE_HISTORY:
            for jj in range(size_m):
                k = jj // n_main
                r = jj - k * n_main
                col = Mix[r] if k == 0 else n_vars + (k - 1) * n_vars + Mix[r]
                hist[i, 0, col] = y[jj]
            for j in range(n_drv):
                hist[i, 0, Dix[j]] = th[j]
            for k in range(n_sens):
                kd_ = cuda.const.array_like(dir_to_drv_t)[k]
                for j in range(n_drv):
                    hist[i, 0, n_vars + k * n_vars + Dix[j]] = th[n_drv + kd_ * n_drv + j] if kd_ >= 0 else 0.0
        accepted_out[i] = accepted
        rejected_out[i] = rejected
        loop_out[i] = n_steps

    return kernel, tpb
