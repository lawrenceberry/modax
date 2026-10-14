# modax

--8<-- "README.md:overview"

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started.md)**

    Install it, and run the first solve.

-   :material-function-variant: **[Calling a solver](guide/api.md)**

    `solve(...)`, its arguments, and what the callbacks have to look like.

-   :material-grid: **[Sparse systems](guide/sparse.md)**

    One `sparsity` pattern buys both a coloured Jacobian and a compiled sparse
    direct solve.

-   :material-chart-line: **[Gradients](guide/gradients.md)**

    `jax.grad` and friends, through a continuous forward-sensitivity solve.

</div>

## Solvers (`modax/`)

| Method      | Type                   | Use for           | File           |
|-------------|------------------------|-------------------|----------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`     |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`   |

Rodas5P supports an `lu_precision` (`"fp32"`/`"fp64"`) knob: the FP32
factorisation halves shared-memory use without lowering method order, since the
Rosenbrock order conditions hold under an approximate Jacobian.
