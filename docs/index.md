# Quickstart

--8<-- "README.md:overview"

## Installation

--8<-- "README.md:install"

The CPU install is enough for the tests that exercise the typing and lowering
pipeline without a device (`tests/test_examples.py`), and for building these
docs.

## A first solve

--8<-- "README.md:quickstart"

The ensemble dimension comes from whichever of `y0` and `params` is
two-dimensional; both may be, and they must then agree. The history is
`(N, n_save, n_vars)`; see [Running a solver](usage/running.md) for what
`return_stats` and save hooks add to it.

## Where to go next

<div class="grid cards" markdown>

-   :material-scale-balance: **[Choosing a solver](usage/solvers.md)**

    Tsit5 for non-stiff problems, Rodas5P for stiff ones, and what each costs.

-   :material-function-variant: **[Running a solver](usage/running.md)**

    `solve(...)`, its arguments, and the rules the callbacks obey.

-   :material-code-braces: **[Writing ODE callbacks](usage/callbacks.md)**

    The rules a right-hand side compiled for the GPU has to follow.

-   :material-grid: **[Sparse systems](usage/sparse.md)**

    One `sparsity` pattern buys both a coloured Jacobian and a compiled sparse
    direct solve.

-   :material-chart-line: **[Gradients](usage/gradients.md)**

    `jax.grad` and friends, through a continuous forward-sensitivity solve.

</div>
