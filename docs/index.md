# Quickstart

--8<-- "README.md:overview"

## Solvers (`modax/`)

| Method      | Type                   | Use for           | File           |
|-------------|------------------------|-------------------|----------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`     |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`   |

Rodas5P supports an `lu_precision` (`"fp32"`/`"fp64"`) knob: the FP32
factorisation halves shared-memory use without lowering method order, since the
Rosenbrock order conditions hold under an approximate Jacobian.

## Installation

--8<-- "README.md:install"

The CPU install is enough for the tests that exercise the typing and lowering
pipeline without a device (`tests/test_examples.py`), and for building these
docs.

## A first solve

--8<-- "README.md:quickstart"

The ensemble dimension comes from whichever of `y0` and `params` is
two-dimensional; both may be, and they must then agree. The result is always
`(N, n_save, n_vars)`.

## Where to go next

<div class="grid cards" markdown>

-   :material-function-variant: **[Calling a solver](guide/api.md)**

    `solve(...)`, its arguments, and the rules the callbacks obey.

-   :material-code-braces: **[Writing ODE callbacks](guide/callbacks.md)**

    What `numba_cuda_mlir` allows, and how a right-hand side that must also run
    under JAX is written once.

-   :material-grid: **[Sparse systems](guide/sparse.md)**

    One `sparsity` pattern buys both a coloured Jacobian and a compiled sparse
    direct solve.

-   :material-chart-line: **[Gradients](guide/gradients.md)**

    `jax.grad` and friends, through a continuous forward-sensitivity solve.

</div>

## Citation

--8<-- "README.md:citation"

## License

--8<-- "README.md:license"

??? note "Full license text"

    ```text
    --8<-- "LICENSE"
    ```
