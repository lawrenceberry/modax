# Choosing a solver

modax has two solvers, each in its own module with the same `solve(...)` entry
point, so switching between them is a change of import:

| Method      | Type                   | Use for           | Module              |
|-------------|------------------------|-------------------|---------------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `modax.tsit5`       |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `modax.rodas5P`     |

Both run one CUDA thread per trajectory with per-trajectory adaptive stepping,
take the same callback, are `jit`-traceable, lower an outer `jax.vmap` to one
ensemble launch, and are differentiable through
[forward sensitivities](gradients.md). What separates them is stiffness.

## Tsit5: explicit, for non-stiff problems

Tsitouras's fifth-order explicit Runge-Kutta pair. A step is a fixed sequence
of right-hand-side evaluations -- six per accepted step, since the last stage
of one step is the first of the next -- and nothing else: no Jacobian, no
linear algebra.

**Pros**

- **Cheapest per step.** Work per step is $O(n_\text{vars})$ arithmetic, so it
  scales roughly linearly with dimension.
- **Smallest footprint.** Nine state-sized vectors per trajectory. For systems
  of up to 16 components and ensembles of up to 16,384 trajectories they live
  in shared memory, and otherwise in thread-local memory; `backend` overrides
  the choice, and the two give bit-identical results.
- **Cheap gradients.** A sensitivity column needs one first-order Enzyme sweep
  per stage and no second derivatives.

**Cons**

- **Stiffness makes it crawl.** An explicit method's step is bounded by
  stability, not accuracy, so on a stiff problem it takes vast numbers of tiny
  steps -- or exhausts `max_steps` -- however smooth the solution is.
- **Fewer options.** No sparsity pattern, save hooks, per-trajectory end times
  (`tf_index`) or `array_rhs`; those are Rodas5P's.

## Rodas5P: linearly implicit, for stiff problems

A fifth-order Rosenbrock-W method. Each step factorises the iteration matrix
$M = I/(h\gamma) - J$ once and then solves a linear system per stage against
that one factorisation. The Jacobian $J$, and the $\partial f/\partial t$ a
non-autonomous system needs, are differentiated out of `ode_fn` by Enzyme, so
there is still nothing to write beyond the right-hand side.

**Pros**

- **Stable on stiff problems.** Steps are set by accuracy, not stability, so it
  takes large steps where Tsit5 would take tiny ones.
- **Fixed cost per step.** Unlike a fully implicit method such as Kvaerno5,
  there is no Newton iteration whose count varies from trajectory to
  trajectory, so lanes in a warp stay in step -- which is why it is two to
  three orders of magnitude faster than Diffrax's Kvaerno5 on the
  [Robertson benchmark](../benchmarks/robertson.md).
- **Tolerant of an approximate Jacobian.** As a W-method it stays convergent
  with an inexact $J$, at worst losing a little order and taking some extra
  steps. That is what lets the sparse solver do without pivoting, and the
  factorisation run in FP32: `lu_precision="fp32"`, the default, halves the
  LU's local-memory footprint at the expense of slightly lower method order,
  and `"fp64"` is there for ill-conditioned systems.
- **Scales with sparsity.** A [`sparsity` pattern](sparse.md) cuts the Jacobian
  to one Enzyme sweep per colour and replaces the dense LU with a sparse direct
  solve compiled for that structure.
- **More options.** Save hooks that consume the history inside the launch,
  per-trajectory end times (`tf_index`), an `array_rhs` for the primal stage
  evaluations, and register and block-size controls.

**Cons**

- **More work per step.** Eight right-hand-side evaluations, $n_\text{colours}
  + 1$ Enzyme sweeps for the Jacobian and $\partial f/\partial t$, an LU
  factorisation and a triangular solve per stage. Dense, the factorisation is
  $O(n_\text{vars}^3)$ work and $O(n_\text{vars}^2)$ memory per trajectory, so
  without a pattern it falls behind quickly above a few tens of variables.
- **Slower to compile.** The Enzyme differentiation and the generated linear
  algebra make the first solve of a new system markedly slower to build.
- **Dearer gradients.** A sensitivity column also needs second derivatives
  (forward over forward) of `ode_fn`; see [Gradients](gradients.md).

## Which to use

Start from the problem's stiffness. If the solution varies slowly but the
system has fast decaying modes -- chemical kinetics, reaction networks,
strongly damped oscillators -- use Rodas5P, and give it a sparsity pattern if
the Jacobian has one. If every timescale in the system is one you need to
resolve anyway, use Tsit5. When in doubt, run both on a few trajectories with
`return_stats=True`: a Tsit5 solve whose step counts dwarf Rodas5P's is stiff.
