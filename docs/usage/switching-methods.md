# Switching between methods

If your problem is stiff in some regions and non-stiff in others, you can use `Auto(Tsit5(), Rodas5P())` to switch between the explicit Tsit5 method and the implicit Rodas5P method automatically. The block will step explicitly while it is not stiff and implicitly while it is stiff, similar to how OrdinaryDiffEq.jl's `AutoSwitch` works.

## Stiffness switching

After every attempted step, accepted or rejected, the block estimates its
Jacobian's largest eigenvalue magnitude. Tsit5 gets it for free from its last
two stages, $\max_j |k_{7,j} - k_{6,j}| / |y_{n+1,j} - g_{6,j}|$ (Hairer and
Wanner II, p. 22). Rodas5P uses $\|J\|_\infty$. A step is stiff when the
estimate times the next step size exceeds 0.9 of Tsit5's stability size.

More than `maxstiffstep` (10) stiff steps in a row switch the block to
Rodas5P and double the step (`dtfac`). More than `maxnonstiffstep` (3)
non-stiff steps switch it back and halve the step, including rejected attempts.
On the GPU, the entire 32 thread warp steps with Rodas5P if any lane requires it.

## Why is this helpful?

A component that couples one-way to many other components in the state produces a full column of the joint
Jacobian, and a full column fills in a sparse factorisation. Split, each
block factorises only its own matrix, and upstream quantities enter a
reader's right-hand side as known functions of time. Sub-systems can take longer steps
without being bound by the dynamics of other stiff sub-systems, reducing the number of right hand side
evaluations that are needed.

## How is this implemented?

Each block runs ahead of its readers under its own error control. After each
accepted step it stores a node: its state, rate and second derivative
($y'' = J y' + \partial_t f$), and the values and first two time derivatives
of the quantities it publishes. Between nodes, a quintic Hermite interpolant
gives $O(H^6)$ dense output over a step $H$, one order beyond either
method's local error.

Before a reader steps over $[t, t + h]$, it writes the quintic's
coefficients into its copy of the parameter row. Its stages and Jacobian
sweeps then see the upstream as a polynomial in $t$. When any lane needs
a further time step of an upstream block, every lane steps that block, keeping the warp
together.

## Error control

Each block holds its own error to `rtol` and `atol`. A reader also estimates
the error it inherits from upstream: an extra direction in its Jacobian sweep
returns $\partial f / \partial q \cdot \epsilon$, the rate at which the
upstream error $\epsilon$ enters each row. A shorter reader step cannot
reduce this error, so the reader passes it to the upstream controller, which
shortens its own steps instead. Every block's local error, own and
inherited, then stays within tolerance.

## Sensitivities

Every block carries forward sensitivities for the parameters and for the
initial states of itself and its upstream blocks. The parameter seeds include
the Hermite coefficients of the upstream sensitivities, so one forward sweep
returns $J_y S_y + J_p + \partial f / \partial q \, S_q$. A published
quantity's sensitivity and its rate come from first- and second-order sweeps.
Its second derivative comes from a quintic fit through the sensitivity's
value and rate at both nodes and the midpoint, still $O(H^6)$.
