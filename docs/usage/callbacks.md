# Writing ODE callbacks

The right-hand side is compiled with
[`numba_cuda_mlir`](https://github.com/NVIDIA/numba-cuda) and runs as device
code inside the solver kernel, which constrains how it is written.

```python
def ode_fn(y, t, p):
    """(y, t, p) -> tuple of the same length as y."""
    y0, y1 = y
    mu, = p
    return y1, mu * (1.0 - y0 * y0) * y1 - y0
```

A plain Python function is jitted automatically; one already decorated with
`cuda.jit(device=True)` is used as it is.

## The rules

- **Fixed-size tuples of scalars**, in and out — not arrays. The tuple's length
  is the system's dimension and has to be known when the kernel is built.
- **`math`, not `numpy` or `jax.numpy`.** `math.exp`, `math.log`, `math.sqrt`
  and the rest lower to device intrinsics.
- **Device code cannot call a plain Python helper.** Either inline the shared
  work or pre-decorate the helper with `@cuda.jit(device=True)`.
- **Index the tuple with constants only.** A runtime index into a tuple makes
  numba emit a bounds-check `cmpxchg`, and Enzyme refuses the atomic ("cannot
  handle unknown instruction") when it differentiates the callback for
  Rodas5P. Wrapping the loop around a thread-local array instead gets past
  typing but yields IR NVVM will not verify. This is why the reference systems
  under `reference/systems/python/` are generated from one expression string
  per component, and why DISCO-EB's unrolled right-hand side is generated from
  its looped one.
- **Watch what you close over.** Closed-over arrays land in CUDA *constant
  memory*, 64 KiB per module, and numba emits one copy **per reference site**.
  Pack related tables into a single array and bind it to a local before
  indexing it — `make_mode_ode` in `examples/mukhanov_sasaki/main.py` is the
  worked case.
