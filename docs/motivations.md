# Motivations

An ordinary differential equation describes the evolution of a state vector
$\mathbf{y}(t) \in \mathbb{R}^D$ according to a known function $f$,

$$
\frac{\mathrm{d}\mathbf{y}}{\mathrm{d}t} = f(t, \mathbf{y};\, \mathbf{p}), \qquad \mathbf{y}(t_0) = \mathbf{y}_0,
$$

where $\mathbf{p} \in \mathbb{R}^P$ is a set of parameters governing the
dynamics. Together with an initial condition $\mathbf{y}_0$ at time $t_0$ these
define an initial value problem, which a numerical solver approximates by
advancing the state through a sequence of discrete time steps. Such solvers
underpin a vast range of scientific and engineering applications, from
simulating the time evolution of physical systems to training neural
differential equations.

A particularly demanding class of problems requires solving not one but
$N \gg 1$ such systems simultaneously, differing only in their initial
conditions $\mathbf{y}_0^{(i)}$ or parameters $\mathbf{p}^{(i)}$. modax exists
for this class of problem, and this page sets out why it needed a library of its
own. The [paper](https://arxiv.org/abs/2609.35661) makes the same argument at
greater length, with benchmarks.

## Where massive ensembles arise

Problems of this type arise naturally across many scientific domains.

- *Bayesian parameter estimation and inverse problems.* Inferring the
  parameters $\mathbf{p}$ of a physical system from observational data requires
  evaluating the likelihood over a large ensemble of candidate parameter
  vectors. Each evaluation involves solving the ODE forward in time, so Markov
  chain Monte Carlo or nested sampling inference for an ODE-governed system may
  demand millions of solves. The [parameter estimation
  example](examples/parameter-estimation.md) fits a big bang nucleosynthesis
  network this way.
- *Monte Carlo uncertainty quantification.* Propagating uncertainty in initial
  conditions or model parameters through a dynamical system requires solving
  the ODE for many draws from the input distribution. Ensemble weather
  forecasting integrates the same atmospheric model from many slightly
  perturbed initial conditions, and assessing the risk of a collisional cascade
  in orbit means propagating thousands of independent debris trajectories. The
  [uncertainty quantification example](examples/uncertainty-quantification.md)
  propagates astrophysical priors through to the global 21 cm signal.
- *Uncoupled ODE systems.* Many physical models described by partial
  differential equations decompose into a large number of mutually independent
  low-dimensional ODEs, such as linear PDEs that decouple into ODEs in their
  diagonal basis. The Fourier modes of the translation-invariant
  Einstein-Boltzmann equations are the canonical example: codes like CAMB and
  CLASS compute the CMB power spectrum by solving hundreds to thousands of
  independent ODEs at a time. The [physically-decoupled systems
  example](examples/physically-decoupled-systems.md) computes the primordial
  power spectrum mode by mode.

In each case the individual ODEs are small, with a state dimension
$D \lesssim 200$, and mutually independent, in that the solution to trajectory
$i$ does not depend on trajectory $j$.

## Why a GPU?

Small, independent problems are a good fit for modern hardware accelerators.
A GPU offers a very large number of computational cores best suited to
embarrassingly parallel workloads that do not require much memory or
collaboration between cores. A top-of-the-range NVIDIA H100 contains tens of
thousands of cores and provides up to 67 TFLOPS of double-precision
floating-point performance, against hundreds of cores and only 1-2 TFLOPS for a
comparably priced CPU. For an ensemble of ODEs that parallelises across
thousands of independent compute streams, this is a potential
order-of-magnitude improvement in both time-to-solution and cost per solve.

Realising it is not automatic. The architectural constraints of a GPU, in
particular single-instruction multiple-thread execution, limited fast memory
and the far higher single-precision throughput, shape what an efficient solver
has to look like; the [Design](design.md) page goes through them one by one.

## Why not the existing libraries?

The prevailing model of GPU computing has been shaped by machine learning,
where a single large neural network occupies much of the device memory,
parallelism is expressed through large tensor operations, and automatic
differentiation is a central requirement. For the scientific applications
above, in which a single ODE is far too small to use a GPU on its own, the more
important axis of parallelism is across the ensemble of independent problems,
each with a small state and modest memory requirements. Efficient ensemble
evaluation is therefore the primary objective, with gradients a useful
additional capability rather than the organising principle. The
machine-learning origins of the existing GPU ODE libraries are reflected in
their designs.

### DiffEqGPU.jl

DiffEqGPU.jl is a purpose-built Julia library for GPU-accelerated ensemble ODE
solving, and it targets precisely the regime modax does. It exposes two
parallelisation strategies. The `EnsembleGPUKernel` backend compiles the entire
adaptive solve loop into a single GPU kernel in which each trajectory takes its
own independent time steps. The `EnsembleGPUArray` backend combines $N$ small
ODEs into a single large one and solves the combined system with joint time
steps and GPU-accelerated array operations. As its authors point out, the
former suits large ensembles of highly divergent trajectories better: the
joint time steps of the latter bottleneck on the slowest trajectory in each
batch, and the array operations incur separate kernel launches whose overhead
can dominate the cost of a solve for low-dimensional problems. Its one
drawback for our purposes is that it lives in Julia, outside the Python and
JAX ecosystem in which many scientific programs, such as the nested sampler of
the parameter estimation example, run.

### Diffrax

Diffrax is a comprehensive JAX library for the numerical integration of
differential equations. Built on JAX's `lax.while_loop` primitive for adaptive
time-stepping, it provides a wide range of solvers and first-class support for
automatic differentiation, and was designed primarily for neural differential
equations, with relatively many parameters and an expensive right-hand side
compared to the problems here.

Applying it to an ensemble through `jax.vmap` produces a batched solver whose
trajectories step through time independently, as in `EnsembleGPUKernel`, but
whose execution is a single global loop of vectorised array operations over the
batch, as in `EnsembleGPUArray`. The loop must keep iterating until the slowest
trajectory is solved, so it too bottlenecks on ensembles of highly divergent
trajectories, and XLA cannot express anything else: when a `lax.while_loop` is
batched under `vmap`, every trajectory in the batch iterates in lockstep, and
there is no mechanism for an individual trajectory to terminate early and
release its resources.

### torchdiffeq

torchdiffeq is a library of ODE solvers written in PyTorch, which like Diffrax
was designed with neural differential equations in mind. It implements no
adaptive implicit methods, focusing instead on explicit Runge-Kutta integrators
and reverse-mode gradients through the continuous adjoint method. It solves an
ensemble by making the state a $(D, N)$ tensor and treating it as one large
ODE, so trajectories advance in unison with a single step size dictated by
whichever trajectory demands the smallest step. The adaptive loop itself runs
in Python on the host, dispatching stage evaluations to the GPU, and each step
therefore carries a large fixed overhead: negligible for a large, expensive
system, but dominant for an ensemble of small ODEs.

While the paper was in preparation two concurrent papers presented GRADSOLVE,
a JAX library for GPU ODE ensembles with reverse-mode gradients through the
exact discrete adjoint, which we acknowledge as an independent contribution.

## What modax does differently

modax takes DiffEqGPU.jl's ensemble-kernel approach and brings it to JAX.
Each solver is a hand-written CUDA kernel in which one thread integrates one
trajectory through its own adaptive loop, so a trajectory is bottlenecked only
on the slowest member of its own warp of 32 rather than on the slowest member
of the whole ensemble, and the entire solve is one kernel launch. The kernel is
exposed to JAX as an XLA FFI custom call, which makes a solve an ordinary JAX
primitive: it traces under `jax.jit`, an outer `jax.vmap` lowers to a single
ensemble launch, and `jax.grad` and its relatives work through a forward
sensitivity rule. For stiff systems it implements a linearly implicit
Rosenbrock-Wanner method, whose fixed sequence of linear solves avoids the
data-dependent Newton iteration that costs a fully implicit method so dearly on
a GPU, and it exploits a known sparsity pattern of the Jacobian in both the
differentiation and the linear algebra. The [Design](design.md) page explains
each of these choices.
