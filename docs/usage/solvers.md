# Choosing a solver

modax has two solvers, each in its own module with the same `solve(...)` entry
point, so switching between them is a change of import:

| Method      | Type                   | Use for           | Module              |
|-------------|------------------------|-------------------|---------------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `modax.tsit5`       |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `modax.rodas5P`     |

Both run one CUDA thread per trajectory with per-trajectory adaptive stepping,
take the same callback, trace under `jax.jit`, lower an outer `jax.vmap` to one
ensemble launch, and are differentiable through
[forward sensitivities](gradients.md). What separates them is stiffness.

## Tsit5: explicit, for non-stiff problems

The Tsitouras 5(4) pair is a seven-stage, fifth-order explicit Runge-Kutta
scheme with an embedded fourth-order error estimate and a fourth-order
continuous extension that costs no additional evaluations. It is
first-same-as-last: the seventh stage of an accepted step is the first stage of
the next, so an accepted step costs six evaluations of $f$ and nothing else, no
Jacobian and no linear algebra. Its fixed seven-stage structure, with no inner
loop other than the adaptive retry, suits SIMT execution particularly well,
since the only source of divergence is the number of steps each trajectory
takes. The kernel body is a direct transcription of the tableau, every
coefficient a compile-time constant and every loop of constant trip count, so
each stage compiles to straight-line arithmetic.

The nine per-trajectory working vectors, the state, the trial state and the
seven stage derivatives, live either in on-chip shared memory or in
thread-local memory, which resides in off-chip DRAM and is cached in L2. The
shared-memory kernel lays each vector out as a $(D, 32)$ array so that
consecutive lanes hold consecutive words and a warp's access is coalesced. The
integrator is one device function over nine one-dimensional vectors, handed
either the thread's local arrays or its column of each shared array, so the two
kernels are bit-identical and `backend` chooses between them. Shared memory
costs $9 \times 32 \times D \times 8$ bytes per block, and reserving it caps the
blocks an SM can hold. At small ensemble sizes the local arrays are promoted to
registers and the two kernels time within noise of each other; once the
ensemble saturates the device the shared kernel loses by up to 14 per cent. The
`"auto"` default therefore takes shared memory only when $D \le 16$ and
$N \le 16\,384$, a transition set somewhat below what the shared-memory budget
would allow in order to keep occupancy high, and thread-local memory
otherwise. Either way the launch carries no scratch: the kernel once held the
vectors in transposed global scratch that XLA allocated, and that path went for
simplicity.

**Pros**

- *Cheapest per step.* The work per step is $O(n_\text{vars})$ arithmetic, so
  it scales roughly linearly with dimension.
- *Smallest footprint.* Nine state-sized vectors per trajectory and no matrix.
- *Cheap gradients.* A sensitivity column needs one first-order Enzyme sweep
  per stage and no second derivatives.

**Cons**

- *Stiffness makes it crawl.* An explicit method's step is bounded by
  stability rather than accuracy, so on a stiff problem it takes vast numbers
  of tiny steps, or exhausts `max_steps`, however smooth the solution is.
- *Fewer options.* There is no sparsity pattern, save hook, per-trajectory end
  time (`tf_index`) or `array_rhs`; those belong to Rodas5P.

## Rodas5P: linearly implicit, for stiff problems

For stiff systems an explicit method is forced by stability rather than
accuracy to take prohibitively small steps, and an implicit method is required.
A standard implicit Runge-Kutta method solves a non-linear system at every
stage by Newton iteration, whose iteration count is data dependent and
therefore a source of thread divergence. A Rosenbrock method avoids the
iteration altogether by linearising each stage equation about the current
state, which turns the step into a sequence of *linear* systems that share one
matrix: one factorisation and a fixed number of triangular solves per step, a
known and data-independent amount of work, so the lanes of a warp stay in
step. This is why it is two to three orders of magnitude faster than Diffrax's
fully implicit Kvaerno5 on the [Robertson benchmark](../benchmarks/robertson.md).

The method is the fifth-order, eight-stage Rodas5P of Steinebach (2023) in the
transformed form of Hairer and Wanner. With $J = \partial f/\partial\mathbf{y}$
and $f_t = \partial f/\partial t$ at $(t_n, \mathbf{y}_n)$, each stage solves

$$
\left(\frac{1}{h\gamma} I - J\right) \mathbf{k}_i = f\!\left(t_n + \alpha_i h,\ \mathbf{y}_n + \sum_{j<i} a_{ij}\mathbf{k}_j\right) + \frac{1}{h}\sum_{j<i} c_{ij}\mathbf{k}_j + d_i\, h\, f_t,
$$

and the solution advances by $\mathbf{y}_{n+1} = \mathbf{y}_n + \sum_i m_i
\mathbf{k}_i$. The eighth stage is constructed so that $\mathbf{k}_8$ is the
difference between the fifth-order solution and an embedded fourth-order one,
so the error estimate costs nothing extra, and the last three stages share the
end point and accumulate into the same argument vector. Each step evaluates
$-J$ and $f_t$ straight into the thread's matrix buffer, adds $1/(h\gamma)$ to
the diagonal, factorises in place, runs the eight stages, forms the error norm
from $\mathbf{k}_8$ and applies the controller. Both $J$ and $f_t$ are
differentiated out of `ode_fn` by Enzyme, so there is still nothing to write
beyond the right-hand side.

Rodas5P attains its full fifth order only when $J$ is exact. It is not a full
W-method, whose order conditions would hold for an arbitrary $J$, but its
coefficients satisfy the W-method conditions up to second order, so an inexact
$J$ reduces it to order 2, with order 1 for the embedded estimator. The size of
the additional error scales with the perturbation to $J$, and the step-size
controller compensates by taking smaller steps. That robustness is what lets
the [sparse solver](sparse.md) do without pivoting, and what allows the
factorisation and triangular solves to run in single precision, where the
perturbation is at the level of FP32 rounding: `lu_precision="fp32"` is the
default, and it halves the matrix buffer's footprint and uses the FP32 cores
that outnumber the FP64 ones, while `"fp64"` is there for ill-conditioned
systems. The state, right-hand side, Jacobian and error estimate stay in double
precision throughout.

**Pros**

- *Stable on stiff problems.* Steps are set by accuracy rather than stability,
  so it takes large steps where Tsit5 would take tiny ones.
- *Scales with sparsity.* A [`sparsity` pattern](sparse.md) cuts the Jacobian
  to one Enzyme sweep per colour and replaces the dense LU with a sparse direct
  solve compiled for that structure.
- *More options.* Save hooks that consume the history inside the launch,
  per-trajectory end times (`tf_index`), an `array_rhs` for the primal stage
  evaluations, and register and block-size controls.

**Cons**

- *More work per step.* Eight right-hand-side evaluations,
  $n_\text{colours} + 1$ Enzyme sweeps for the Jacobian and
  $\partial f/\partial t$, an LU factorisation and a triangular solve per
  stage. Dense, the factorisation is $O(n_\text{vars}^3)$ work and
  $O(n_\text{vars}^2)$ memory per trajectory, so without a pattern it falls
  behind quickly above a few tens of variables.
- *Slower to compile.* The Enzyme differentiation and the generated linear
  algebra make the first solve of a new system markedly slower to build.
- *Dearer gradients.* A sensitivity column also needs second derivatives
  (forward over forward) of `ode_fn`; see [Computing gradients](gradients.md).

## Which to use?

Start from the problem's stiffness. If the solution varies slowly but the
system has fast decaying modes, as in chemical kinetics, reaction networks and
strongly damped oscillators, use Rodas5P, and give it a sparsity pattern if the
Jacobian has one. If every timescale in the system is one you need to resolve
anyway, use Tsit5. When in doubt, run both on a few trajectories with
`return_stats=True`: a Tsit5 solve whose step counts dwarf Rodas5P's is stiff.
