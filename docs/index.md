# Quickstart

```{include} ../README.md
:start-after: <!-- start:overview -->
:end-before: <!-- end:overview -->
```

## Installation

```{include} ../README.md
:start-after: <!-- start:install -->
:end-before: <!-- end:install -->
```

The CPU install is enough to build these docs and to run the host-side part
of the test suite, which is small: most tests need a device, and
`tests/test_examples.py` compiles the examples' callbacks with
`cuda.compile_ptx`, which initialises the CUDA driver and so skips without
one.

## Your first solve

```{include} ../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

The ensemble dimension comes from whichever of `y0` and `params` is
two-dimensional; if both are 2D, they must then agree. The returned integration history has shape
`(N, n_save, n_vars)`.

## Where to go next?

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Choosing a solver
:link: usage/solvers
:link-type: doc

Tsit5 for non-stiff problems, Rodas5P for stiff ones.
:::

:::{grid-item-card} Running a solver
:link: usage/running
:link-type: doc

`solve(...)`, its arguments, and the rules callbacks must obey.
:::

:::{grid-item-card} Exploiting sparsity
:link: usage/sparse
:link-type: doc

Speeding up solves using knowledge of which inputs affect which outputs.
:::

:::{grid-item-card} Computing gradients
:link: usage/gradients
:link-type: doc

`jax.grad` using continuous forward-sensitivity solves.
:::

::::

```{toctree}
:hidden:

Quickstart <self>
motivations
design
development
acknowledgement
```

```{toctree}
:hidden:
:caption: Usage

usage/solvers
usage/running
usage/sparse
usage/hierarchies
usage/switching-methods
usage/gradients
```

```{toctree}
:hidden:
:caption: Examples

examples/parameter-estimation
examples/uncertainty-quantification
examples/physically-uncoupled-systems
```

```{toctree}
:hidden:
:caption: Benchmarks

benchmarks/lorenz
benchmarks/robertson
benchmarks/van-der-pol
```

```{toctree}
:hidden:
:caption: API Reference

api/solve
api/methods
```
