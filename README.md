<h1 align="center">modax</h1>

<!--
The coverage badge reads a gist that the `tests` workflow rewrites on every
push to main; the id below is the repository variable COVERAGE_GIST_ID.
These badges sit above the `overview` snippet marker, so the documentation
site does not repeat them.
-->
<p align="center">
  <a href="https://github.com/lawrenceberry/modax/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/lawrenceberry/modax/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/lawrenceberry/modax/actions/workflows/tests.yml"><img alt="coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Flawrenceberry%2Fb451e14a9620fabb0d49c345f13b5414%2Fraw%2Fmodax-coverage.json"></a>
  <a href="https://github.com/lawrenceberry/modax/actions/workflows/lint.yml"><img alt="lint" src="https://github.com/lawrenceberry/modax/actions/workflows/lint.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/modax-solvers/"><img alt="PyPI" src="https://img.shields.io/pypi/v/modax-solvers"></a>
  <a href="https://pypi.org/project/modax-solvers/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/modax-solvers"></a>
  <a href="https://pypi.org/project/modax-solvers/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/modax-solvers"></a>
  <a href="https://docs.jax.dev/"><img alt="JAX" src="https://img.shields.io/badge/JAX-compatible-3f51b5"></a>
  <a href="https://arxiv.org/abs/2609.35661"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.35661-b31b1b.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="https://lawrenceberry.github.io/modax/">Documentation</a> •
  <a href="https://arxiv.org/abs/2609.35661">Paper</a> •
  <a href="#citation">Citation</a>
</p>

<!-- --8<-- [start:overview] -->

`modax` is a library of GPU-accelerated ODE solvers optimised for solving **massive ensembles** (1k-100k) of low-dimensional (<200D) ODE trajectories, built on JAX and Numba-CUDA-MLIR. Applications include: Bayesian parameter estimation, uncertainty quantification and the integration of physically uncoupled systems.

Every solver is compiled as a **custom CUDA kernel** using Numba-CUDA-MLIR. Each ODE trajectory is solved within its own CUDA thread and results are exposed to JAX as an XLA FFI custom call. Calls to `modax` therefore behave as ordinary JAX primitives. They are `jit`-traceable, auto-differentiable and a `vmap` over a single solve lowers to one native ensemble launch.

<!-- --8<-- [end:overview] -->

The [documentation](https://lawrenceberry.github.io/modax/) covers the solver
API and explains how to use the library. The [paper](https://arxiv.org/abs/2609.35661) details the technical implementation, example applications and performance benchmarks against other GPU-accelerated ODE libraries in Julia and Python.

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

<!-- --8<-- [end:install] -->

## Quickstart

A stiff decay chain A → B → C, solved for 1000 rate constants at once, with
the Jacobian's sparsity pattern (i.e. which inputs affect which outputs) declared:

```python
import jax.numpy as jnp
import numpy as np
from modax.rodas5P import solve


def ode_fn(y, t, p):  # tuples of scalars in and out
    return -p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1]


sparsity = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool)
params = jnp.stack([jnp.logspace(0, 4, 1000), jnp.ones(1000)], axis=1)
y0 = jnp.array([1.0, 0.0, 0.0])
t_span = jnp.linspace(0.0, 10.0, 11)

y = solve(ode_fn, y0, t_span, params, sparsity=sparsity)  # (1000, 11, 3)
```

<!-- --8<-- [start:development] -->

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

<!-- --8<-- [end:development] -->

## Citation

If you use modax in your research, please cite the
[arXiv article](https://arxiv.org/abs/2609.35661):

```bibtex
@misc{berry2026acceleratingmassiveensemblesordinary,
      title={Accelerating massive ensembles of ordinary differential equations}, 
      author={Lawrence Berry and Will Handley and Oliver Hahn and Nils Schöneberg},
      year={2026},
      eprint={2609.35661},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2609.35661}, 
}
```

## License

modax is released under the [MIT License](LICENSE).
