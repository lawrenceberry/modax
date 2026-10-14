# Design

modax implements two GPU-accelerated, automatically differentiable ODE solvers
optimised for massive ensembles of low-dimensional divergent trajectories: the
explicit Tsitouras 5(4) method for non-stiff systems and the fifth-order
Rodas5P Rosenbrock-Wanner method for stiff systems. This page sets out the
architectural constraints of a GPU that shaped them, and then the design
choices those constraints led to. It follows Section 2 of the
[paper](https://arxiv.org/abs/2609.35661); the [Motivations](motivations.md)
page explains why a new library was needed at all.

## Architectural constraints

Realising the theoretical arithmetic throughput of a GPU for ensemble ODE
solving requires careful consideration of the underlying architecture. Four
constraints matter here.

### SIMT parallelism and thread divergence

To achieve high arithmetic throughput within a limited transistor and power
budget, GPUs adopt an execution model fundamentally different from that of
CPUs. Threads are scheduled in groups of 32, known as a *warp*, and the 32
lanes of a warp execute the same instruction in every clock cycle, each on its
own data held in registers. This is single-instruction, multiple-thread (SIMT)
parallelism, and it means GPU threads cannot operate entirely independently;
they work in concert with the other lanes of their warp. Fortunately the model
is ideally matched to ensemble ODE problems, since $N$ independent trajectories
naturally provide $N$ independent streams of identical arithmetic work.

SIMT execution introduces a hazard when code contains conditional branches. If
some threads in a warp take one branch while others take another, both
branches execute serially and the threads not following the active branch sit
idle. This is *thread divergence*, and it can severely reduce utilisation when
branches are unbalanced. An adaptive solver adjusts its step size by evaluating
an error estimate after each proposed step, accepting the step when the error
is within tolerance and otherwise retrying with a smaller one. When an ensemble
is solved within a single warp, differences in the number of steps the
trajectories need make the warp serialise over its slowest member. Differences
in the number of Newton iterations an implicit solver needs are a second source
of the same divergence. Minimising divergent control flow is therefore a key
design criterion.

### Memory access patterns

A GPU has memory tiers of very different bandwidths and latencies, from larger
but higher-latency off-chip global memory to smaller, lower-latency on-chip
caches and registers that are private to each *streaming multiprocessor* (SM),
the unit that schedules and executes warps. Warps are launched in *thread
blocks* that the hardware assigns to SMs, and each SM holds several resident
blocks at once. To hide the latency of global memory, an SM interleaves the
execution of its resident warps, so that data loading for one overlaps with
computation for another. The ratio of resident warps to the architectural
maximum per SM is the *occupancy*, and resident warps compete for the SM's
limited registers and shared memory: the more of either a warp reserves, the
lower the occupancy.

A low-dimensional ODE occupies little register and cache space, which allows
high occupancy. And because every solve is independent, no inter-trajectory
communication through slow global memory is required, so the temporary
variables of the solver, the stage vectors, the error estimate and the step
controller's state, can stay in the registers and on-chip memory of the SM
doing the work.

### Floating-point precision

Scientific computing has traditionally relied on 64-bit double-precision
arithmetic, for which x86 CPUs provide extensive hardware support. GPUs offer
substantially fewer FP64 than FP32 cores: half as many on an H100 and a
sixty-fourth on a gaming-oriented RTX 4090. That motivates asking how much
double-precision arithmetic a solver strictly needs for correctness, and
whether any step of the algorithm can run in single precision without
compromising the solution. Single-precision data also takes half the space in
the fastest caches, which lets more warps be resident on each SM.

### Kernel launch overhead

Each invocation of a GPU kernel incurs a fixed overhead of several
microseconds as the driver schedules it, transfers its parameters and
initialises execution contexts on the SMs. That is negligible for a kernel
doing substantial arithmetic, but the right-hand side of a small ODE is often a
handful of operations, and a solver that issues many kernel calls per step, one
per Runge-Kutta stage or one per linear algebra routine, is severely penalised.
Minimising the number of distinct kernel launches is critical.

## Design choices

### Single-step, high-order methods

Both solvers are single-step methods, rather than the multistep backward
differentiation formula (BDF) methods common on CPUs. A BDF method fits a
polynomial through the last few accepted steps and adapts both the step size
and the polynomial order. It needs fewer stage evaluations, but reading and
writing the multistep history buffer is costly on a GPU and is exactly the
bandwidth pressure to avoid in the memory-bound regime of massive ensembles,
and its dynamic order selection compounds thread divergence. The additional
stage evaluations of a single-step method are fixed, predictable and map
cleanly onto SIMT lanes.

For the same reason both methods are of high order. Compared with a lower-order
method of equivalent accuracy, a higher-order one needs more stage evaluations
per step but fewer steps. The stages are a fixed cost that every trajectory in
a warp pays identically, whereas extra steps extend the adaptive loop and raise
the risk of divergence. The two methods themselves, and where each keeps its
working vectors, are described on the [Choosing a solver](usage/solvers.md)
page.

### One trajectory per thread, one warp per block, one kernel per solve

How best to expose the parallelism across an ensemble depends on how similar
its trajectories are. When they need similar numbers of steps the
straightforward approach is to vectorise a single-trajectory solver with
`jax.vmap`, as the Diffrax baseline does: each trajectory keeps its own step
size and accept-or-reject decision, but the batched while loop advances all
$N$ in lockstep and terminates only when the slowest has converged. When the
trajectories are heterogeneous, the faster ones waste iterations waiting for
the slower.

Truly independent execution needs a custom CUDA kernel in which each trajectory
runs its own adaptive loop to completion without synchronising with the
others. XLA cannot express this, since it requires a statically known
computation graph, so the solvers are written with Numba-CUDA-MLIR, which
compiles thread-level CUDA programs from plain Python. One trajectory is
assigned to each thread of a 32-thread warp, and each warp is launched as its
own thread block, so a trajectory is bottlenecked only on the slowest member
of its own warp. The Ada architecture of the benchmark GPU provides 48 warp
slots per SM, so many warps progress concurrently; as soon as every trajectory
in a warp has completed, its solutions are read out, the block retires and the
scheduler slots the next block from the queue into the space it freed. Nothing
in a step is collective and nothing synchronises, so a finished thread simply
returns, and a thread past the end of the ensemble leaves at once. That is also
why `trajectories_per_block` may be anything at all: no on-chip resource bounds
it, since every per-trajectory buffer is the thread's own local memory.

The whole solve is compiled into one device program launched once per solve,
which removes the per-step launch overhead a vectorised solver pays. The
kernel is exposed to JAX as an XLA FFI custom call, so a modax `solve` is an
ordinary JAX primitive that traces under `jax.jit`, carries no data copies or
host synchronisation, and lowers an outer `jax.vmap` to a single ensemble
launch. The [Lorenz](benchmarks/lorenz.md) and
[Robertson](benchmarks/robertson.md) divergence benchmarks show what the
per-thread loop buys, and that pre-sorting an ensemble so that similar
trajectories share a warp removes the sensitivity to divergence almost
entirely.

### Adaptive stepping, the error norm and dense output

Both solvers embed a lower-order set of weights over the same stages, so the
difference between the two solutions estimates the local error at no extra
cost. The error norm is a weighted root-mean-square over the scaled components,

$$
\varepsilon_n = \sqrt{\frac{1}{D}\sum_{d=1}^{D} \left(\frac{w_d\, \hat{e}_{n,d}}{\mathrm{atol} + \mathrm{rtol}\,\max\!\left(|y_{n,d}|, |y_{n+1,d}|\right)}\right)^2},
$$

where $w_d$ is a per-component weight the caller may set through
`error_weights`. A weight of zero removes a component from step-size control
entirely, which matters in physical applications where the state mixes
quantities of very different magnitude and importance: without weighting the
norm can be dominated by components that are large but uninteresting, forcing
unnecessarily small steps, or under-resolve small but observationally
important ones.

Step-size selection is a proportional-integral-derivative controller,

$$
h_{n+1} = h_n \, \min\!\left(\phi_{\max},\, \max\!\left(\phi_{\min},\, 0.9\, \varepsilon_n^{\,e_1}\, \varepsilon_{n-1}^{\,e_2}\, \varepsilon_{n-2}^{\,e_3}\right)\right),
$$

with $e_1 = -(k_I + k_P + k_D)/q$, $e_2 = (k_P + 2k_D)/q$ and $e_3 = -k_D/q$,
where $(k_P, k_I, k_D)$ are the `pcoeff`, `icoeff` and `dcoeff` arguments and
$q$ is 5 for Tsit5 and 6 for Rodas5P. The default gains $(0, 1, 0)$ recover
the elementary integral controller. Non-zero proportional and derivative gains
tend to produce smoother step-size sequences and fewer rejected steps on stiff
and mildly oscillatory problems, which reduces the variance in step count
across an ensemble and with it the thread divergence. The error history
advances only on accepted steps.

Rather than forcing the step sequence to land on every output time, an
accepted step evaluates the method's continuous extension at the output times
it contains. The controller is then free to choose steps from the error
estimate alone, and a solve takes the same steps whether one or a thousand
output times are requested.

### The Jacobian comes from Enzyme

An implicit solver needs the Jacobian, and Numba compiles to LLVM and NVVM
device code that JAX cannot see, so the derivatives come from Enzyme, which
differentiates the compiled LLVM code itself. The CUDA-capable fork
[numba-enzyme-cuda](https://pypi.org/project/numba-enzyme-cuda/) differentiates
CUDA device functions with primitives such as `jvp`, emits the derivative as
NVVM link-time-optimisation IR that the CUDA linker inlines into the solver
kernel before its final optimisation pass, and composes with itself to produce
second derivatives. None of this is visible to the caller, who supplies only
$f$; there is no `jac_fn`.

The differentiation is forward mode, and it computes the directional
derivative $J\mathbf{v}$ rather than the matrix. A forward sweep seeded with a
direction over the flattened argument list $(y_1, \ldots, y_D, t, p_1, \ldots,
p_P)$ costs roughly one extra evaluation of $f$ in both compute and memory, so
$J$ and $f_t$ together cost $D + 1$ sweeps, with the time derivative obtained
by seeding $t$ instead of the state. Sweeping one direction at a time keeps the
working set at $O(D)$ and lets each column fold straight into the matrix
buffer, where a full `jacfwd` would put $D^2$ doubles in per-thread local
memory. Reverse mode could reach a whole row per sweep by slicing the
right-hand side into scalar components, and was measured to win below roughly
16 state variables, but it needs $D$ primals and $D$ Enzyme differentiations
against forward's one apiece; a cold first solve at 96 state variables took
171 s that way against 49 s this way.

Two details of how the sweeps are called carry most of the performance. The
seed rows are literals: because the derivative is inlined before constant
propagation, a seed the compiler can see folds, its zero components kill their
tangent arithmetic, and a sweep collapses to the columns it actually seeds. The
same seed read through a loop variable arrives in registers and cannot fold,
and that version of the kernel sat at the 168-register cap with a 13 KB spill
frame. And several seeds share one call, eight at a time, so the primal work
the sweeps have in common is computed once and left to LLVM to eliminate the
duplicates. On the 50-variable Einstein-Boltzmann system of DISCO-EB at 128
trajectories this was 665.9 ms with a loop over the colours, 594.7 ms with
literal seeds and 508.8 ms with eight seeds per call, against 552 ms for the
hand-written Jacobian the sweeps replaced.

### Exploiting sparsity

The limited fast memory of a GPU motivates sparse differentiation and sparse
linear algebra, both to reduce the memory footprint of each solve and improve
occupancy and to reduce the arithmetic. modax exploits a Jacobian's sparsity
pattern when it is known at compile time, and the one argument `sparsity` is
the whole interface.

The forward sweeps are optimised by graph colouring. Columns of $J$ that share
no non-zero row are structurally orthogonal: they do not contribute to the same
component of $J\mathbf{v}$, so seeding both at once returns both intact. Partitioning
the columns into as few such groups as possible is a vertex colouring of the
graph in which columns are adjacent when some row contains both, which
NetworkX's greedy colouring does well enough. The Jacobian and $f_t$ then cost
$n_\text{colours} + 1$ sweeps instead of $D + 1$, an order of magnitude fewer
for a banded system. The pattern must be a superset of the true non-zeros,
because colouring a subset silently corrupts the entries where two columns of
a group overlap after all.

The linear solve is compiled for the pattern. At kernel-build time the
variables are reordered with SuiteSparse's approximate minimum degree
algorithm, the exact non-zero pattern of $L + U$ for the permuted matrix is
obtained by symbolic Gaussian elimination, and it is laid out as one
compressed sparse row image that the factorisation and both triangular solves
read by rows. The sweeps write $-J$ directly into that buffer, the
factorisation overwrites it in place, and the solves index the right-hand side
by the original variable numbers, so nothing is permuted at run time. There is
no pivoting, since swapping rows on the numbers would make the pattern depend
on the data and differ between threads; the AMD permutation is applied
symmetrically, which keeps $1/(h\gamma)$ on the diagonal so that every pivot is
structurally present and grows as the step shrinks, and a badly conditioned
pivot is a matter for the step-size controller. Where the structure is small
enough the three routines are emitted as straight-line code with every slot a
literal, and above that as loops over index tables in constant memory. Without
a pattern every column gets its own colour and the matrix is a dense row-major
array factorised by right-looking LU with partial pivoting; the two solvers
have the same shape, so the kernel has no branch between them. [Exploiting
sparsity](usage/sparse.md) argues each of these choices in detail.

### Forward sensitivities

Gradients come from forward sensitivity analysis rather than from
differentiating the kernel. Each column $S_k$ of the sensitivity obeys the
variational equation $S_k' = J_y S_k + J_{p,k}$, a linear ODE of the same
dimension as the state, and the columns being differentiated are stacked under
the state into one joint system of $D(1 + m)$ components that the existing
kernels integrate as an ordinary ODE. The right-hand side of each sensitivity
block is $f$ differentiated in the direction $(S_k, 0, \mathbf{e}_k)$, one
forward sweep of the same Enzyme-differentiated $f$ that Rodas5P uses for its
Jacobian, and $J_y$ and $J_p$ are never formed.

For a stiff system the sensitivities inherit the stiffness of the state
exactly, so they go through the stiff solver, but not through a factorisation
of the whole joint matrix. Because $f$ does not depend on $S$, the joint
Jacobian is block lower triangular with the same $J_y$ on every diagonal block,
so the iteration matrix has the same $M_0 = I/(h\gamma) - J_y$ on every
diagonal block and each stage is a block forward substitution against a single
factorisation of $M_0$: the state row is solved first, and each sensitivity row
is then solved against the same factors with its right-hand side corrected by
the coupling term, which is a second directional derivative of $f$. The
matrix buffer stays $D$-sized and the cost of a gradient grows linearly with
the number of columns. An adjoint would scale better with that number, but
reverse mode must revisit the forward trajectory in reverse order, and storing
it is impractical for large ensembles on a memory-constrained GPU while
recomputing it backwards is unstable for the stiff, dissipative systems the
solver exists for. [Computing gradients](usage/gradients.md) carries the full
argument and the measured costs.

## How a solve reaches the GPU

There is no direct Numba launch path: every solve goes through `solve` and the
XLA custom call.

1. `solve` normalises the ensemble shapes into the arrays the kernel reads
   (`modax._jax_common.prepare_inputs`) and builds, or fetches from its
   cache, the kernel for this `ode_fn`, dimension and set of options.
2. `modax._jax_numba_custom_call` compiles a launcher for that kernel,
   registers it as an XLA FFI target and exposes it through `ffi_abi_call`.
3. `modax._jax_common.ensemble_ffi_call` makes the call. The inputs keep
   their natural `(N, ...)` layouts and there is no scratch, since every
   per-trajectory buffer is the thread's own local memory.
4. `modax._jax_common.make_custom_vmap_solver` wraps the result so that an
   outer `jax.vmap` over a single solve lowers to one ensemble launch rather
   than to a batched trace.

Because the whole thing is a `jax.ffi.ffi_call`, a solve is traceable under
`jit` and usable inside `lax.scan` and `vmap`; `examples/bbn_estimation` calls
one from a BlackJAX nested-sampling likelihood.

## Caches

Kernel construction is cached with `functools.cache` on the callback and its
options, and numba-enzyme keys its derivative cache on the primal's lowered
IR. Because the five-argument derivative call inlines, the kernel's PTX is
byte-identical across processes and the CUDA JIT cache hits.
`modax.tsit5.clear_caches()` drops Tsit5's compiled kernels and the sensitivity
helpers' caches; Rodas5P has no equivalent, so its kernels stay cached for the
life of the process.
