# Quickstart

--8<-- "README.md:overview"

## Installation

--8<-- "README.md:install"

The CPU install is enough to build these docs and to run the host-side part
of the test suite, which is small: most tests need a device, and
`tests/test_examples.py` compiles the examples' callbacks with
`cuda.compile_ptx`, which initialises the CUDA driver and so skips without
one.

## A first solve

--8<-- "README.md:quickstart"

The ensemble dimension comes from whichever of `y0` and `params` is
two-dimensional; both may be, and they must then agree. The history has shape
`(N, n_save, n_vars)`, and [Running a solver](usage/running.md) describes what
`return_stats` and save hooks add to it.

## Where to go next?

<div class="grid cards" markdown>

-   :material-scale-balance: **[Choosing a solver](usage/solvers.md)**

    Tsit5 for non-stiff problems, Rodas5P for stiff ones, and what each costs.

-   :material-function-variant: **[Running a solver](usage/running.md)**

    `solve(...)`, its arguments, and the rules the callbacks obey.

-   :material-grid: **[Exploiting sparsity](usage/sparse.md)**

    One `sparsity` pattern buys both a coloured Jacobian and a compiled sparse
    direct solve.

-   :material-chart-line: **[Computing gradients](usage/gradients.md)**

    `jax.grad` and friends, through a continuous forward-sensitivity solve.

-   :material-lightbulb-on-outline: **[Motivations](motivations.md)**

    Where massive ensembles arise, and why the existing GPU libraries were
    not enough.

-   :material-cube-outline: **[Design](design.md)**

    The constraints a GPU imposes and the design choices they led to.

</div>
