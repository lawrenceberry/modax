# Computing gradients

Both solvers are differentiable in forward mode with respect to `y0` and
`params`:

```python
import jax
import modax

def loss(params):
    y = modax.solve(ode_fn, y0, t_span, params)
    return jnp.sum((y[:, -1, :] - observed) ** 2)

grad = jax.jacfwd(loss)(params)   # one joint solve, a column per parameter

def loss_and_value(params):
    value = loss(params)
    return value, value

grad, value = jax.jacfwd(loss_and_value, has_aux=True)(params)   # still one solve
```

`jax.jvp` and `jax.jacfwd` work inside `jit` and `vmap` as usual.
Derivatives are computed only when a differentiation transform requests
them, so an undifferentiated `solve(...)` runs the same kernel as before at
no additional cost. Reverse mode (`jax.grad`, `jax.value_and_grad`,
`jax.vjp`, `jax.jacrev`, `jax.linearize`) raises as it is not yet supported.

Sensitivity analysis is implemented in `modax/_sensitivity.py`.

## Choosing what to differentiate

Each direction you ask for adds `n_vars` components to the joint system, so
`jax.jvp` adds one column and `jax.jacfwd` one per differentiated input.
`params` can be a pytree of arrays (a tuple, a dict, any
registered node) as well as a single array. Its leaves are concatenated in
`jax.tree.leaves` order into the row `ode_fn` sees, so the callback is
unchanged. Each leaf is `()` (one shared column), `(k,)` (`k` shared
columns) or `(N, k)` (one row per trajectory), and shared leaves broadcast
beside per-trajectory ones.

```python
def loss(rates, temp):
    y = modax.solve(ode_fn, y0, t_span, (rates, temp))   # p = (*rates, temp)
    return jnp.sum((y[:, -1, :] - observed) ** 2)

jax.jacfwd(loss)(rates, temp)          # a column per rate
jax.jacfwd(lambda r: modax.solve(ode_fn, y0, t_span, (r, temp)))(rates)
```

A leaf that does not depend on what is being differentiated is a symbolic
zero, and its entries of the tangent are zero.

For an ensemble with per-trajectory parameters, map over the trajectories
*outside* the derivative:

```python
def final(p):                      # one trajectory's parameters, (n_params,)
    return modax.solve(ode_fn, y0, t_span, p)[0, -1]

jac = jax.vmap(jax.jacfwd(final))(params)   # params (N, n_params)
```

Each `jacfwd` then sees one trajectory's `n_params` inputs, and the outer
`vmap` lowers to one launch of the whole ensemble with `n_params` columns a
thread. `jax.jacfwd` over the whole `(N, n_params)` array would instead send
every trajectory all `N * n_params` directions, nearly all of them zero, since JAX does not know that trajectory `i` reads only row `i`. The solve
will raise in that case rather than exhaustively integrate the zeros.

## Forward sensitivity analysis

Requesting a derivative integrates the continuous forward-sensitivity system
alongside the state. Writing $S = \partial y/\partial\theta$ and
differentiating $y' = f(t, y, p)$ with respect to $\theta$ gives the
variational equation

$$\frac{dS}{dt} = J_y(t)\,S(t) + J_p(t), \qquad J_y = \frac{\partial f}{\partial y},\quad J_p = \frac{\partial f}{\partial \theta},$$

which can be integrated jointly with the state as one larger ODE,

$$\frac{\partial}{\partial t}\begin{bmatrix} y \\ S \end{bmatrix} = \begin{bmatrix} f(t, y, p) \\ J_y(t)\,S + J_p(t) \end{bmatrix}.$$

Because the value and the sensitivity are produced by the same integration,
one solve returns both, rather than one for the value and another for the
derivative.

### Contracting with the tangent

Integrated as written, $S$ contains a column for every parameter being differentiated. With
$\theta = (y_0, p)$ and $p \in \mathbb{R}^D$, the initial-state block
$S_{y_0} = \partial y/\partial y_0$ is $n_\text{vars} \times n_\text{vars}$
and the parameter block $S_p = \partial y/\partial p$ is
$n_\text{vars} \times D$. The joint system then has $n_\text{vars}\,(1 + n_\text{vars} + D)$ components: $n_\text{vars}^2$ for the initial conditions and $n_\text{vars} D$ for the parameters.

However the full joint system $S$ is not needed in forward mode. `jax.jvp` supplies a tangent
$\dot\theta = (\dot y_0, \dot p)$ and asks only for the change of $y(t)$ along
that direction, which is a linear combination of the columns of $S$:

$$s(t) = S_{y_0}(t)\,\dot y_0 + \sum_{k=1}^{D} \dot p_k\,S_k(t).$$

The tangent is constant in time, so differentiating $s$ differentiates each
column, and substituting the equations above gives:

$$\frac{ds}{dt} = J_y \underbrace{\left(S_{y_0}\dot y_0 + \sum_k \dot p_k S_k\right)}_{s} + \underbrace{\sum_k \dot p_k \frac{\partial f}{\partial p_k}}_{J_p\,\dot p}.$$

The initial condition contracts the same way, $s(t_0) = I\,\dot y_0 + 0 =
\dot y_0$, giving:

$$\frac{ds}{dt} = J_y(t)\,s(t) + J_p(t)\,\dot p, \qquad s(t_0) = \dot y_0.$$

The joint system $[y, s]$ therefore requires only $2\,n_\text{vars}$ components however 
many parameters are differentiated, including
the initial conditions which enter $s$ through its starting value rather
than as separate columns. The forcing $J_p\,\dot p$ is the directional
derivative of $f$ along $\dot p$, which forward-mode AD returns in one
sweep, as it does $J_y s$.

`jax.jacfwd` asks for one such $s$ per basis direction, so it integrates
$n_\text{vars}(1 + d)$ components for $d$ directions: the state and one
column per direction.

### Implicit methods

For implicit methods, `modax` integrates $[y, s]$ one Rosenbrock step at a
time. Since $f$ does not depend on $s$, the joint Jacobian is block lower
triangular,

$$A = \begin{bmatrix} J_y & 0 \\ L & J_y\end{bmatrix}, \qquad L = \frac{\partial}{\partial y}\left(J_y\,s + J_p\,\dot p\right),$$

and the iteration matrix $M = I/(h\gamma) - A$ inherits this structure with
the same block $M_0 = I/(h\gamma) - J_y$ on both diagonals. A joint step
therefore never factorises the $2n_\text{vars} \times 2n_\text{vars}$
matrix. Each stage is a block forward substitution against one
factorisation of $M_0$,

$$M_0\,k_y = r_y, \qquad M_0\,k_s = r_s + L\,k_y,$$

which solves the state row first, then the direction's row against the same
factors with its right-hand side corrected by the coupling term. $L$ is
never formed either: $L\,k_y$ is a second directional derivative of $f$
(see [below](#computing-second-derivatives)).

`jax.jacfwd` adds one $s_b$ per direction, each with its own coupling block
$L_b$ below the state and $M_0$ again on its diagonal. Each direction costs
one more substitution against the same factors, so for $d$ directions the
factorisation stays $O(n_\text{vars}^3)$ and its buffer $n_\text{vars}^2$,
where factorising the joint matrix would cost $O(n_\text{vars}^3(1+d)^3)$
and $n_\text{vars}^2(1+d)^2$.

The state and its directions advance under one step-size controller with
one accept-or-reject decision. Because of the triangular structure, solving
the state before the directions gives the exact solution of the joint linear
system rather than an approximation to it. Nothing is stored between steps:
$s$ needs $y$ at the step's own stage points, and the joint step has them in
hand.

### Implementation

The joint system is constructed by seeding an Enzyme `jvp` of `ode_fn` with the tangent $(\dot y_0, 0, \dot p)$, which returns the right-hand side of the $s$ equation:

$$D f(y, t, p)\,[s, 0, \dot p] = J_y\,s + J_p\,\dot p.$$

The zero corresponds to the time derivative, which is not needed. $\dot p$ is supplied by extending each trajectory's parameters from $p$ to $(p, \dot p_0, \dot p_1, \ldots, \dot p_{d-1}),$ one $\dot p_b$ per direction, and each direction's state is initialised to $\dot y_{0,b}$. The save
hook is seeded the same way, so the quantities it keeps carry their tangents
$\partial q/\partial y\,s + \partial q/\partial p\,\dot p$ too.

A thread's local memory grows with its columns, and the driver reserves it
for every resident thread, so a large state with many directions can
exhaust the device. `max_directions_per_thread` caps the columns one thread
integrates:

```python
y = modax.solve(ode_fn, y0, t_span, params, max_directions_per_thread=2)
```

The directions then split into groups which each run in their own thread with a copy of the primal trajectory. A saturated GPU device will ultimately do more work, but a small ensemble will finish sooner since its directional derivatives run in parallel.

## Computing second derivatives

The coupling block $L$ is a second derivative of the right-hand side, with
respect to (state, state) and (state, parameter), contracted with the
direction:

$$L = \frac{\partial}{\partial y}\left(J_y\,s + J_p\,\dot p\right) = \frac{\partial^2 f}{\partial y\,\partial y}\!\left[\cdot,\,s\right] + \frac{\partial^2 f}{\partial y\,\partial p}\!\left[\cdot,\,\dot p\right].$$

It arises because $s' = J_y(y)\,s + J_p(y)\,\dot p$ is a linear ODE whose
coefficients depend on $y$, and an implicit method has to differentiate
those coefficients. A Newton-iterated method such as BDF or an SDIRK scheme
could treat $J_y$ as a preconditioner and converge regardless of the
omission, but Rodas5P is linearly implicit: its Jacobian is part of the
formula, so an approximation to it enters the answer.

The second derivatives are computed with
[numba-enzyme](https://github.com/Qruise-ai/numba-enzyme), whose `jvp`
composes with itself. `jvp(jvp(f))` is a forward-over-forward directional
derivative, $D^2 f(x)[u,v]$. The kernel seeds $u = (s, 0, \dot p)$, the
direction itself. With $v = (k_y, 0, 0)$ it returns $L\,k_y$ directly, without
forming the matrix $L$, and with $v = (0, 1, 0)$ it returns the $\partial
F/\partial t$ of the direction's rows, the other second derivative a
Rosenbrock method requires. The same mechanism supplies the first-order
right-hand side: $J_y s + J_p\,\dot p$ is one sweep along $u$, so each
direction costs one sweep rather than a full Jacobian.

Composing derivatives is less straightforward here than in JAX, where
`jax.jvp` maps a jaxpr to a jaxpr and is therefore closed under itself. The
`jvp` of numba-enzyme maps a Python callable to a compiled device symbol, and
differentiating that symbol again would present Enzyme with an external
declaration that has no body. The fork instead records the chain of
differentiations and emits every level as a definition in one module, where a
single Enzyme pass resolves the nested markers.
