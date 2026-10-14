# Running a solver

All solvers expose a single `solve(...)` entry point that integrates an
ensemble in one call:

```python
from modax.rodas5P import solve

# ode_fn is a CUDA-device callable: (y, t, p) -> tuple
y = solve(
    ode_fn,
    y0,          # (n_vars,) or (N, n_vars)     initial state(s)
    t_span,      # (n_save,) output times (shared across the ensemble)
    params,      # (n_params,) or (N, n_params) per-trajectory parameters
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100_000,
    return_stats=False,                  # also return per-trajectory step counters
    error_weights=None,                  # optional per-component weights (0 = ignore)
    pcoeff=0.0, icoeff=1.0, dcoeff=0.0,  # PID step-controller gains
    sens_error_control=True,             # error-control the sensitivities too
    sparsity=None,                       # Jacobian pattern; see "Exploiting sparsity"
    ordering="amd",                      # its fill-reducing permutation
)
# y has shape (N, n_save, n_vars)
```

Calling conventions:

- The callbacks are compiled with `numba_cuda_mlir`, so they take and return fixed-size
  tuples of scalars rather than arrays, and use `math` rather than `numpy`/`jax.numpy`.
  Plain Python functions are jitted automatically; pre-`cuda.jit`ed ones are used as-is.
  A right-hand side that must also run under `jax` — to compare against a
  Diffrax baseline, say — need not be written twice: `examples/dual_backend.py`
  builds both forms from one body, parameterised over the names the two
  backends spell differently.
- **Rodas5P** (implicit) needs only `ode_fn`. Its Jacobian ∂f/∂y, and the ∂f/∂t
  a non-autonomous system needs to retain full order, are differentiated out of
  `ode_fn` with [numba-enzyme](https://github.com/Qruise-ai/numba-enzyme),
  which runs Enzyme over the callback's LLVM IR.
- **Tsit5** (explicit) needs no derivatives at all.

Importing `modax` enables JAX float64.

## Shapes

| Argument | Shape | Notes |
|----------|-------|-------|
| `y0` | `(n_vars,)` or `(N, n_vars)` | broadcast across the ensemble when 1-D |
| `t_span` | `(n_save,)` | output times, shared across the ensemble |
| `params` | `(n_params,)` or `(N, n_params)` | one row per trajectory |
| result | `(N, n_save, n_vars)` | |

`N` is taken from whichever input is batched; if both are, they must agree.
`return_stats=True` adds a dict of per-trajectory step counters alongside the
history.

Every argument, with its default, is documented in the
[API reference](../api/solvers.md).
