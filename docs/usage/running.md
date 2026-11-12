# Running a solver

`modax.solve` integrates an ensemble in one call. The method and its options
are one object, `method=`; everything else describes the problem or the
launch:

```python
import modax

# ode_fn is a CUDA-device callable: (y, t, p) -> tuple
y = modax.solve(
    ode_fn,
    y0,          # (n_vars,) or (N, n_vars)     initial state(s)
    t_span,      # (n_save,) shared save times, or (N, n_save) one row each
    params,      # (n_params,) or (N, n_params) per-trajectory parameters
    method=modax.Rodas5P(                # the default; or modax.Tsit5(), modax.Auto()
        lu_precision="fp32",             # the LU's precision
        gains=modax.PIDGains(0.0, 1.0, 0.0),  # PID step-controller gains
        ordering="amd",                  # the sparse LU's fill-reducing permutation
    ),
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100_000,
    return_stats=False,                  # also return per-trajectory step counters
    error_weights=None,                  # optional per-component weights (0 = ignore)
    sens_error_control=True,             # error-control the sensitivities too
    sparsity=None,                       # Jacobian pattern; see "Exploiting sparsity"
)
# y has shape (N, n_save, n_vars)
```

Importing `modax` enables JAX float64. All arguments and their defaults are documented in the [API reference](../api/solve.md).

## Initial state, integration timespan and parameters

| Argument | Description | Shape | Notes |
|----------|-------------|-------|-------|
| `y0` | the initial state $\mathbf{y}(t_0)$ of each trajectory | `(n_vars,)` or `(N, n_vars)` | broadcast across the ensemble when 1-D |
| `t_span` | the times at which the solution is saved; the first is $t_0$ and the last the end of the integration | `(n_save,)` or `(N, n_save)` | shared by the ensemble when 1-D, otherwise one grid per trajectory, so each can start and end at times of its own |
| `params` | the parameters $\mathbf{p}$ passed to `ode_fn` for each trajectory | `(n_params,)` or `(N, n_params)`, or a pytree of such leaves | one row per trajectory; see [choosing what to differentiate](gradients.md#choosing-what-to-differentiate) |
| result | the state of every trajectory at every save time | `(N, n_save, n_vars)` | |

`N` comes from whichever of `y0`, `params` and `t_span` is batched. Batched inputs must agree.
`return_stats=True` adds a dict of per-trajectory step counters alongside the
history.

## Keeping derived quantities

A `save_hook` is a function that computes quantities derived from the state at every save time. It takes the same form as `ode_fn`, `(y, t, p) -> tuple`, is compiled the same way, and it returns values alongside the state history. If you set `save_history=False`, the solver does not store the state history, only the derived quantities. This is a helpful memory-saving option when the state is large and you only need a few derived quantities.

```python
def energy(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]),)   # one quantity: (q,)

final, q = modax.solve(ode_fn, y0, t_span, params,
                       save_hook=energy, save_history=False)
# final: (N, 1, n_vars), q: (N, n_save, 1)
```

The quantities differentiate as normal: `jax.grad` of a loss of `q` integrates the sensitivities and stores each quantity's tangents in the kernel.

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
- *Loops are allowed.* The body may loop over the tuple with a runtime index
  or over a `cuda.local.array`, as a hierarchy of multipoles naturally is.
  numba-enzyme optimises the callback with LLVM's `O3` pipeline before Enzyme
  differentiates it for Rodas5P, which unrolls the loops and folds away the
  bounds checks whose failure path (a `cmpxchg`) Enzyme would otherwise refuse.
  Loops therefore need a trip count LLVM can see: a `range` over bounds fixed
  when the kernel is built.
- *Closed-over arrays.* These are placed in CUDA constant memory, of which
  there is 64 KiB per module, and numba emits one copy per reference site.
  Pack related tables into a single array and bind it to a local variable
  before indexing it; `make_mode_ode` in `examples/mukhanov_sasaki/main.py`
  shows how.
