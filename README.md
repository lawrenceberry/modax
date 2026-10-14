# modax

<!--
The coverage badge reads a gist that the `tests` workflow rewrites on every
push to main; the id below is the repository variable COVERAGE_GIST_ID.
These badges sit above the `overview` snippet marker, so the documentation
site does not repeat them.
-->
[![tests](https://github.com/lawrenceberry/modax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lawrenceberry/modax/actions/workflows/tests.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Flawrenceberry%2Fb451e14a9620fabb0d49c345f13b5414%2Fraw%2Fmodax-coverage.json)](https://github.com/lawrenceberry/modax/actions/workflows/tests.yml)
[![lint](https://github.com/lawrenceberry/modax/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/lawrenceberry/modax/actions/workflows/lint.yml)
[![PyPI](https://img.shields.io/pypi/v/modax-solvers)](https://pypi.org/project/modax-solvers/)
[![Downloads](https://img.shields.io/pypi/dm/modax-solvers)](https://pypi.org/project/modax-solvers/)
[![Python](https://img.shields.io/pypi/pyversions/modax-solvers)](https://pypi.org/project/modax-solvers/)
[![JAX](https://img.shields.io/badge/JAX-compatible-3f51b5)](https://docs.jax.dev/)

**[Documentation](https://lawrenceberry.github.io/modax/)**

<!-- --8<-- [start:overview] -->

GPU-accelerated ODE solvers for **massive ensembles** (1-100k) of low-dimensional (<200D) ODE trajectories, built on
JAX and Numba-CUDA-MLIR. Applications include: Bayesian parameter inference, uncertainty quantification and the integration of physically uncoupled systems.

Every solver is a hand-written **CUDA custom kernel** compiled by
Numba-CUDA-MLIR: one CUDA thread
per trajectory, hand-written step kernels with in-kernel LU factorisation,
exposed to JAX as an XLA FFI custom call. That binding makes each solver an
ordinary JAX primitive — `jit`-traceable, and `vmap` over a single solve lowers
to one native ensemble launch.

<!-- --8<-- [end:overview] -->

The [documentation](https://lawrenceberry.github.io/modax/) covers the solver
API, writing ODE callbacks, sparse systems, gradients, the kernel design and
the worked examples.

<!-- --8<-- [start:install] -->

## Installation

```bash
pip install "modax-solvers[cuda13]"   # or [cuda12]; Python 3.10 needs cuda12
```

There is no system library to install first: every dependency ships wheels,
the AMD ordering included. A GPU is needed to run a solve. The Enzyme-derived
Jacobians come from
[numba-enzyme-cuda](https://pypi.org/project/numba-enzyme-cuda/), the
CUDA-enabled fork of numba-enzyme, which carries its own LLVM 15 and Enzyme
binaries and provides the `numba_enzyme` import package, so upstream
`numba-enzyme` must not be installed alongside it. See
[wheels/README.md](wheels/README.md) for what is in the wheel and why.

## Development setup

The repository uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/lawrenceberry/modax.git && cd modax
uv sync                 # CPU
uv sync --extra cuda13  # or --extra cuda12, for GPU

uv run pytest tests     # most tests skip themselves without a GPU
uv run ruff format && uv run ruff check --fix
uv run ty check
uv run pre-commit install   # runs the three checks above on every commit
```

The documentation builds in an environment of its own, since syncing the docs
group into the default one replaces the project's packages:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv sync --only-group docs --no-install-project
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --no-sync mkdocs serve
```

<!-- --8<-- [end:install] -->
