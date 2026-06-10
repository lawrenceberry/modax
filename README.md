# modax

GPU-accelerated ODE solvers for **massive ensembles** (1-100k) of low-dimensional (<200D) ODE trajectories, built on
JAX and Numba-CUDA. Applications include: Bayesian parameter inference, uncertainty quantification and the integration of physically uncoupled systems.

Each solver ships in two interchangeable backends behind one API:

- a **pure-JAX** backend (`*jax.py`) — `vmap`/`lax.while_loop` over the batch,
  end-to-end differentiable and `jit`-friendly;
- a **Numba-CUDA custom-kernel** backend (`*numba.py`) — one CUDA thread per
  trajectory, hand-written step kernels with in-kernel LU factorisation, exposed
  to JAX as a custom call.

## Solvers (`solvers/`)

| Method      | Type                   | Use for           | Files                                |
|-------------|------------------------|-------------------|--------------------------------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `tsit5jax.py`, `tsit5numba.py`       |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `rodas5Pjax.py`, `rodas5Pnumba.py`   |
| **KenCarp5**| ESDIRK (order 5)       | Stiff systems     | `kencarp5jax.py`, `kencarp5numba.py` |

Implicit solvers support an `lu_precision` (`"fp32"`/`"fp64"`) knob: the FP32
factorisation halves shared-memory use without lowering method order, since the
Rosenbrock/SDIRK order conditions hold under an approximate Jacobian.

## API

All backends expose a single `solve(...)` entry point that integrates an
ensemble in one call:

```python
from solvers.rodas5Pjax import solve

# ode_fn(y, t, params) -> dy/dt
y = solve(
    ode_fn,
    y0,          # (n_vars,) or (N, n_vars)     initial state(s)
    t_span,      # (n_save,) output times (shared across the ensemble)
    params,      # (n_params,) or (N, n_params) per-trajectory parameters
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100_000,
    return_stats=False,                  # also return per-step accept/reject counts
    error_weights=None,                  # optional per-component weights (0 = ignore)
    pcoeff=0.0, icoeff=1.0, dcoeff=0.0,  # PID step-controller gains
)
# y has shape (N, n_save, n_vars)
```

Calling conventions per backend:

- **JAX implicit solvers** (`rodas5Pjax`, `kencarp5jax`) recompute the Jacobian
  internally with `jax.jacfwd` — only `ode_fn` is needed.
- **Numba implicit solvers** (`rodas5Pnumba`, `kencarp5numba`) take explicit
  CUDA-device callables: `solve(ode_fn, jac_fn, y0, t_span, params, ...)`, with
  an optional `time_jac_fn` (∂f/∂t) for non-autonomous systems.
- **Tsit5** (explicit) needs no Jacobian on either backend.

Importing `solvers` enables JAX float64.

## Install & run

```bash
uv sync                 # CPU
uv sync --extra cuda13  # or --extra cuda12, for GPU

uv run pytest
uv run ruff format && uv run ruff check --fix
```

## Examples

Worked end-to-end problems live in `examples/` (each with its own README):

- `bbn_estimation/` — toy Big Bang Nucleosynthesis network with nested-sampling
  parameter estimation and a modax/Diffrax/scipy solver benchmark;
- `21cm_igm_evolution/` — toy global 21cm IGM thermal/ionisation history;
- `mukhanov_sasaki/` — Mukhanov–Sasaki mode evolution.

Scaling, dimensionality and divergence benchmarks are under `scripts/`.
