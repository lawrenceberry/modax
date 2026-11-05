# Computing gradients

Both solvers are differentiable with respect to `y0` and `params`:

```python
import jax
from modax.rodas5P import solve

def loss(params):
    y = solve(ode_fn, y0, t_span, params)
    return jnp.sum((y[:, -1, :] - observed) ** 2)

value, grad = jax.value_and_grad(loss)(params)   # one joint solve
```

`jax.jvp`, `jax.jacfwd`, `jax.grad`, `jax.jacrev` and `jax.value_and_grad` all
work, inside `jit` and `vmap` as usual. Derivatives are computed only when a
differentiation transform requests them, so an undifferentiated `solve(...)`
runs the same kernel as before at no additional cost.

Sensitivity analysis is implemented in [`modax._sensitivity`][].

## Forward sensitivity analysis

Requesting a derivative integrates the continuous forward-sensitivity system
alongside the state. Writing $S = \partial y/\partial\theta$ and
differentiating $y' = f(t, y, p)$ with respect to $\theta$ gives the
variational equation

$$\frac{dS}{dt} = J_y(t)\,S(t) + J_p(t), \qquad J_y = \frac{\partial f}{\partial y},\quad J_p = \frac{\partial f}{\partial \theta},$$

which the solver integrates jointly with the state as one larger ODE,

$$\frac{\partial}{\partial t}\begin{bmatrix} y \\ S \end{bmatrix} = \begin{bmatrix} f(t, y, p) \\ J_y(t)\,S + J_p(t) \end{bmatrix}.$$

Because the value and the sensitivity are produced by the same integration,
`jax.value_and_grad` costs one solve rather than one for the value and
another for the derivative.

For implicit methods, `modax` integrates the joint state $[y, S]$ with one Rosenbrock step at a time.
Since $f$ does not depend on $S$, the joint Jacobian is block lower
triangular,

$$A = \begin{bmatrix} J_y & 0 \\ L & J_y\end{bmatrix}, \qquad L = \frac{\partial}{\partial y}\left(J_y S + J_p\right),$$

and the iteration matrix $M = I/(h\gamma) - A$ inherits this structure with
the same block $M_0 = I/(h\gamma) - J_y$ on every diagonal. A joint step
therefore does not factorise an $n_\text{aug} \times n_\text{aug}$ matrix.
Each stage is a block forward substitution against a single factorisation of
$M_0$,

$$M_0\,k_y = r_y, \qquad M_0\,k_{S_k} = r_{S_k} + L_k\,k_y,$$

in which the state row is solved first and each sensitivity row is then solved
against the same factors with its right-hand side corrected by the coupling
term. The factorisation remains $O(n_\text{vars}^3)$ rather than
$O(n_\text{vars}^3(1+n_\text{sens})^3)$ and its buffer $n_\text{vars}^2$
rather than $n_\text{vars}^2(1+n_\text{sens})^2$. The state and the
sensitivities advance under one step-size controller with one accept-or-reject
decision, and because of the triangular structure solving the state before the
sensitivities is the exact solution of the joint linear system rather than an
approximation to it. Nothing is stored between steps: the sensitivity needs
$y$ at the step's own stage points, and the joint step has them in hand.

## Computing second derivatives

The coupling block $L$ is a second derivative of the right-hand side, with
respect to (state, state) and (state, parameter):

$$L_k = \frac{\partial^2 f}{\partial y\,\partial y}\!\left[\cdot,\,S_k\right] + \frac{\partial^2 f}{\partial y\,\partial p_k}.$$

It arises because $S' = J_y(y)S + J_p(y)$ is a linear ODE whose coefficients
depend on $y$, and an implicit method has to differentiate those coefficients.
A Newton-iterated method such as BDF or an SDIRK scheme could treat $J_y$ as a
preconditioner and converge regardless of the omission, but Rodas5P is
linearly implicit: its Jacobian is part of the formula, so an approximation to
it enters the answer.

The second derivatives are computed with
[numba-enzyme](https://github.com/Qruise-ai/numba-enzyme), whose `jvp`
composes with itself. `jvp(jvp(f))` is a forward-over-forward directional
derivative, $D^2 f(x)[u,v]$. Seeding $u = (S_k, 0, e_k)$ and $v = (k_y, 0, 0)$
returns $L_k k_y$ directly, without the matrix $L_k$ being formed, and seeding
$v = (0,1,0)$ instead returns the $\partial F/\partial t$ of the sensitivity
rows, the other second derivative a Rosenbrock method requires. The same
mechanism supplies the first-order right-hand side, since $J_y S_k + J_p$ is
itself a directional derivative and so costs one sweep per column rather than
a full Jacobian.

Composing derivatives is less straightforward here than in JAX, where
`jax.jvp` maps a jaxpr to a jaxpr and is therefore closed under itself. The
`jvp` of numba-enzyme maps a Python callable to a compiled device symbol, and
differentiating that symbol again would present Enzyme with an external
declaration that has no body. The fork instead records the chain of
differentiations and emits every level as a definition in one module, where a
single Enzyme pass resolves the nested markers.
