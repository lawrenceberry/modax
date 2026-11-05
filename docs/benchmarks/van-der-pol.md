# Van der Pol

The Lorenz and Robertson systems are three-dimensional. A lattice of van der
Pol oscillators studies how the solvers scale with dimension. One oscillator
evolves as

$$
\dot y_1 = y_2, \qquad \dot y_2 = \mu (1 - y_1^2)\, y_2 - y_1,
$$

where $\mu$ sets the stiffness: stiff for $\mu \gg 1$, mildly non-stiff at
$\mu = 1$. A $D$-dimensional system couples $D/2$ oscillators diffusively in a
ring,

$$
\dot x_i = v_i, \qquad
\dot v_i = \mu (1 - x_i^2)\, v_i - x_i + d\,(x_{i-1} + x_{i+1} - 2 x_i),
$$

with indices modulo $D/2$ and coupling strength $d = 10$. Its Jacobian has a
banded, cyclic tridiagonal-block sparsity pattern that couples every
oscillator to the rest of the lattice while leaving the stiffness tunable
through $\mu$. The non-stiff lattice ($\mu = 1$) benchmarks Tsit5 and the
stiff lattice ($\mu = 100$) benchmarks Rodas5P, both over $N = 1000$
identical trajectories starting from $(x_i, v_i) = (2, 0)$, integrated over
$t \in [0, 1]$ with output every 0.25, at `rtol = 1e-6`, `atol = 1e-8`.

All benchmarks were performed on a single NVIDIA GeForce RTX 4070 SUPER, a
consumer-grade device with a similar number of CUDA cores to, and the same
architecture as, an L4-class datacentre GPU. DiffEqGPU.jl runs with
`cpu_offload = 0`, so that every trajectory is solved on the GPU. Each point is
measured in a child process of its own, and a point that does not compile and
solve within 180 s is recorded as a timeout and left off the plot. Reported
times exclude compilation: each is the mean over repeated solves after one
warm-up call. The discussion follows Section 3 of the
[paper](https://arxiv.org/abs/2609.35661).

## Non-stiff: scaling with dimension

![Non-stiff van der Pol dimensionality](plots/nonstiff_vdp_dimensionality.png)

modax Tsit5 is the fastest solver at every dimension and scales roughly
linearly with $D$, since the right-hand side is $O(D)$ arithmetic per stage and
no matrix is factorised. At low dimension and small ensembles the state and
stage vectors live in shared memory; past $D = 16$ they move to the slower but
more abundant thread-local memory, which is the kink between 16 and 32. The
switch is set below the point that actually exhausts shared memory, to keep
warp occupancy high: an SM holds several resident warps, and while one waits on
DRAM the scheduler can run another. The dip in Diffrax's timings at
$D = 16$ looks like a backend shape-specialisation effect rather than a change
in numerical work, since its step counts match the neighbouring dimensions.

```bash
uv run python benchmarks/nonstiff_vdp_dimensionality/main.py
```

## Stiff: scaling with dimension

![Stiff van der Pol dimensionality](plots/stiff_vdp_dimensionality.png)

With sparsity analysis disabled, Rodas5P must store and factorise a dense
$D \times D$ Jacobian: $O(D^2)$ memory per trajectory and $O(D^3)$ work per
step. That the FP32 factorisation performs much like FP64 suggests either that
the solver is memory-bound rather than compute-bound, or that the inexact
Jacobian costs extra steps. Exploiting sparsity is therefore what lets the
solver scale. The lattice's pattern is banded, so the number of colours, and
with it the number of forward-mode sweeps per Jacobian, is constant in $D$ and
the Jacobian's cost grows only linearly. The sparsity-aware FP32 modax solver keeps a
50–110× lead over Diffrax Kvaerno5 across the whole range, while the dense
solvers lose theirs as $D$ grows and time out at $D = 128$. DiffEqGPU.jl's
`EnsembleGPUKernel` appears to exceed register or cache limits, and does not
compile and solve within the 180 s cap above $D = 16$.

```bash
uv run python benchmarks/stiff_vdp_dimensionality/main.py
```

## Stiff: Jacobian density

![Stiff van der Pol sparsity](plots/stiff_vdp_sparsity.png)

The dimension sweep keeps nearest-neighbour coupling, so the Jacobian stays
banded and its density falls as $1/D$. Here the dimension is fixed at $D = 64$
and the coupling range -- how many places along the ring, on either side,
oscillators are coupled -- runs from nearest-neighbour to all-to-all, changing
the number of structurally non-zero entries without changing $D$; the
coupling strength is divided by the range, so the total coupling stays fixed. The dense
modax solver and its Diffrax and DiffEqGPU.jl counterparts are insensitive to
density, as expected: they always differentiate, store and factorise the full
$64 \times 64$ matrix. The sparsity-aware modax solver slows as the pattern
fills in, since each extra neighbour adds colour sweeps and fill-in to the
sparse LU, but at 25 per cent density it is still over 15× faster than the
dense solvers. This is not limited to banded systems: any pattern in which a
substantial fraction of the Jacobian is known to be zero benefits. DiffEqGPU.jl's
`EnsembleGPUKernel` does not compile and solve within the 180 s cap at
$D = 64$.

```bash
uv run python benchmarks/stiff_vdp_sparsity/main.py
```

## Stiff: value and gradient

![Stiff van der Pol gradient](plots/stiff_vdp_gradient.png)

The cost of the final state together with its gradient with respect to each
trajectory's damping scale (a parameter multiplying $\mu$), which is one sensitivity column integrated alongside the
state (see [Computing gradients](../usage/gradients.md)). modax is differentiated by
wrapping the solve in `jax.value_and_grad`, which lowers to one kernel launch
integrating the joint system. Diffrax is differentiated the same way, but in
reverse mode through its default recursive checkpointing adjoint. DiffEqGPU.jl
differentiates neither ensemble backend, so it integrates hand-derived forward
sensitivity equations instead. The sparsity-aware modax solver returns value
and gradient at two to four times the cost of the plain solve -- one extra
forward-mode sweep and one pair of triangular solves per stage for the
sensitivity column -- and is roughly two orders of magnitude faster than both
Diffrax and `EnsembleGPUArray` at every dimension.

```bash
uv run python benchmarks/stiff_vdp_gradient/main.py
```
