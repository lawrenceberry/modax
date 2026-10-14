# Lorenz

The Lorenz system is the low-dimensional, non-stiff benchmark, used for the
explicit Tsit5 solvers:

$$
\dot x = \sigma (y - x), \qquad
\dot y = x(\rho - z) - y, \qquad
\dot z = xy - \beta z,
$$

with $\sigma = 10$ and $\beta = 8/3$. Rather than the chaotic $\rho = 28$ the
benchmarks use $\rho = 0.5$, below the pitchfork bifurcation at $\rho = 1$,
where the origin is the unique globally stable fixed point and every
trajectory decays towards it. Trajectories start far from the origin at
$y_0 = (1000, -500, 500)$ and are integrated over $t \in [0, 5]$ at
`rtol = 1e-6`, `atol = 1e-8`. In this regime the number of steps a trajectory
needs is a monotone function of its initial distance from the origin, which
gives direct control over how much work each trajectory demands.

All benchmarks ran on a single NVIDIA GeForce RTX 4070 SUPER, a consumer
card with a similar number of CUDA cores to, and the same architecture as, an
L4-class datacentre GPU. DiffEqGPU.jl runs with `cpu_offload = 0`, so every
trajectory is solved on the GPU. Every point is measured in its own process,
compilation included, and one that does not compile and solve within 180 s is
left off the plot. The discussion follows Section 3 of the
[paper](https://arxiv.org/abs/2609.35661).

## Scaling with ensemble size

![Lorenz scaling](plots/nonstiff_lorenz_scaling.png)

Wall-clock solve time against ensemble size $N$, from 3 to $10^5$ identical
trajectories. At $N = 1000$ modax is 30× faster than Diffrax, consistent with
eliminating essentially all kernel launch overhead, and 2.5× faster than
DiffEqGPU.jl's `EnsembleGPUKernel`, scaling slightly more favourably as $N$
grows. A plausible reason is the kernel's lower register and cache footprint,
which allows higher warp occupancy: more trajectories execute concurrently and
hide memory latency before the scheduler has to serialise them into successive
waves.

```bash
uv run python benchmarks/nonstiff_lorenz_scaling/main.py
```

## Trajectory divergence

![Lorenz divergence](plots/nonstiff_lorenz_divergence.png)

Sensitivity to trajectory divergence at $N = 10^5$, chosen to saturate the
GPU. A divergence parameter controls the spread of initial distances from the
origin. Solve time is divided by the mean number of steps, to remove the
trivial effect of harder trajectories needing more steps, and plotted against
the standard deviation of attempted steps across the ensemble, which measures
how unevenly the work is spread. torchdiffeq uses Dopri8, since it does not
implement Tsit5.

Solve times rise with divergence, because harder trajectories force the easier
ones in their warp to iterate redundantly. `EnsembleGPUArray` is more than an
order of magnitude slower per step than every other solver throughout. When
trajectories are pre-sorted by difficulty before being assigned to warps, so
that similar trajectories share a warp, modax's sensitivity to divergence is
essentially eliminated. Random assignment can put the easiest and hardest
trajectories in the same warp, maximising wasted work. Sorting is not always
trivial, but a heuristic from physical intuition is often a good proxy.

```bash
uv run python benchmarks/nonstiff_lorenz_divergence/main.py
```
