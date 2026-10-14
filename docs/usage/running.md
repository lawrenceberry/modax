# Running a solver

Both solvers expose a single `solve(...)` entry point that integrates an
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

Importing `modax` enables JAX float64. All arguments and their defaults are documented in the [API reference](../api/solvers.md).

## Initial state, integration timespan and parameters

| Argument | Description | Shape | Notes |
|----------|-------------|-------|-------|
| `y0` | the initial state $\mathbf{y}(t_0)$ of each trajectory | `(n_vars,)` or `(N, n_vars)` | broadcast across the ensemble when 1-D |
| `t_span` | the times at which the solution is saved; the first is $t_0$ and the last the end of the integration | `(n_save,)` | output times, shared across the ensemble |
| `params` | the parameters $\mathbf{p}$ passed to `ode_fn` for each trajectory | `(n_params,)` or `(N, n_params)` | one row per trajectory |
| result | the state of every trajectory at every save time | `(N, n_save, n_vars)` | |

`N` is taken from whichever input is batched; if both are, they must agree.
`return_stats=True` adds a dict of per-trajectory step counters alongside the
history.

## Writing ODE function callbacks

The right-hand side of the ODE is compiled with
[`numba_cuda_mlir`](https://github.com/NVIDIA/numba-cuda-mlir) and runs as device
code inside the solver kernel, which constrains how it is written.

```python
def ode_fn(y, t, p):
    """(y, t, p) -> tuple of the same length as y."""
    y0, y1 = y
    mu, = p
    return y1, mu * (1.0 - y0 * y0) * y1 - y0
```

A plain Python function is compiled automatically, and one already decorated
with `cuda.jit(device=True)` is used as it is. It has to follow these rules.

- *Fixed-size tuples of scalars, in and out.* Arrays are not accepted. The
  tuple's length is the dimension of the system and has to be known when the
  kernel is built.
- *`math`, not `numpy` or `jax.numpy`.* `math.exp`, `math.log`, `math.sqrt`
  and the rest lower to device intrinsics.
- *No calls to plain Python helpers.* Device code cannot call an undecorated
  Python function. Either inline the shared work or decorate the helper with
  `@cuda.jit(device=True)`.
- *Constant indices into the tuple.* A runtime index makes numba emit a
  bounds-check `cmpxchg`, and Enzyme refuses the atomic ("cannot handle
  unknown instruction") when it differentiates the callback for Rodas5P.
  Wrapping the loop around a thread-local array instead passes typing but
  yields IR that NVVM will not verify. For this reason the reference systems
  under `reference/systems/python/` are generated from one expression string
  per component, and DISCO-EB's unrolled right-hand side is generated from
  its looped one.
- *Closed-over arrays.* These are placed in CUDA constant memory, of which
  there is 64 KiB per module, and numba emits one copy per reference site.
  Pack related tables into a single array and bind it to a local variable
  before indexing it; `make_mode_ode` in `examples/mukhanov_sasaki/main.py`
  shows how.
