# Robertson

The Robertson system is a prototypical stiff chemical-kinetics problem, an
autocatalytic reaction among three species, used for the implicit Rodas5P
solvers:

$$
\begin{aligned}
\dot y_1 &= -0.04\,y_1 + 10^4\,y_2 y_3, \\
\dot y_2 &= 0.04\,y_1 - 10^4\,y_2 y_3 - 3\times10^7\,y_2^2, \\
\dot y_3 &= 3\times10^7\,y_2^2.
\end{aligned}
$$

It is stiff because of the disparity between the fast kinetic rates of the
first two reactions and the slow evolution of the whole system. The classical
initial condition $y(0) = (1, 0, 0)$ has a long induction period in which only
the slow rate is active; the benchmarks instead start from
$y(0) = (0.891, 0.1, 0.009)$, putting 10 per cent of the mass in the
intermediate species so that the fastest reaction is active at once and the
solver must resolve the stiff timescale from its first step. Trajectories are
integrated over $t \in [0, 10^5]$, with output at $t = 10^{-6}, 10^{-2},
10^2$ and $10^5$, at `rtol = 1e-6`, `atol = 1e-8`.

All benchmarks ran on a single NVIDIA GeForce RTX 4070 SUPER, a consumer
card with a similar number of CUDA cores to, and the same architecture as, an
L4-class datacentre GPU. DiffEqGPU.jl runs with `cpu_offload = 0`, so every
trajectory is solved on the GPU. Each solver's points run in a child process of
their own, and a point that does not compile and solve within 180 s is recorded
as a timeout and left off the plot. Reported times exclude compilation: each is
the mean over repeated solves after one warm-up call. The discussion follows Section 3 of the
[paper](https://arxiv.org/abs/2609.35661).

## Scaling with ensemble size

![Robertson scaling](plots/stiff_robertson_scaling.png)

Wall-clock solve time against ensemble size, from 1 to $10^5$ identical
trajectories. Diffrax has no Rodas5P, so its fifth-order Kvaerno5 implicit
Runge-Kutta method stands in. At $N = 1000$ modax Rodas5P is 6× faster than
DiffEqGPU.jl's `EnsembleGPUKernel` and 700× faster than Diffrax. As a fully
implicit Runge-Kutta method, Kvaerno5 needs an iterative quasi-Newton solve at
every stage, and the variable iteration counts cause severe thread divergence;
a Rosenbrock method's fixed-cost linear solves avoid it, which is why they suit
GPUs. DiffEqGPU.jl has no $N = 1$ point: `EnsembleGPUArray` is skipped there,
since it hands a one-trajectory ensemble to its serial fallback, and
`EnsembleGPUKernel` failed at that size.

```bash
uv run python benchmarks/stiff_robertson_scaling/main.py
```

## Trajectory divergence

![Robertson divergence](plots/stiff_robertson_divergence.png)

Sensitivity to divergence at $N = 3 \times 10^4$. The divergence parameter
$\delta \in [0, 3]$ draws the initial split of mass between $y_1$ and $y_3$
uniformly once $\delta > 0$, moves the intermediate fraction $y_2(0)$
log-uniformly down from 0.1 (to about $10^{-6}$ at $\delta = 3$), and scales each rate constant by an independent factor
$10^u$ with $u \sim \mathcal{U}(-\delta, \delta)$, so at $\delta = 3$ the rates
span six orders of magnitude across the ensemble. The DiffEqGPU.jl
`EnsembleGPUKernel` cases stop at $\delta = 2.5$. As for Lorenz, divergence
substantially increases solve time, and pre-sorting trajectories by expected
difficulty, so that a warp holds trajectories needing similar numbers of
steps, again essentially eliminates modax's sensitivity to it.

```bash
uv run python benchmarks/stiff_robertson_divergence/main.py
```
