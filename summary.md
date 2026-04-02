# ODE Solver Optimization Summary

## Main Solver Changes

- Extended `solvers/rodas5_custom_kernel_v6.py` to support time-dependent Jacobians via `jac_fn(t)` instead of a fixed matrix input.
- Kept the stage logic matrix-specialized by materializing the Jacobian at the start of each step and storing it into `m_ref`.
- Made the time-dependent Jacobian handling genuinely per-trajectory instead of effectively per-block.
- Changed `v6` to accept `t_span` as a 1D array of save times and return `(N, T, D)` histories instead of only the final state.
- Added validation for `t_span` shape, minimum length, and monotonicity.
- Preserved the default Pallas codepath for the standard `v6` solver behavior.

## Precision / Linear Solve Work

- Simplified `v6` linear solve modes to:
  - `linear_solver="fp64"`
  - `linear_solver="fp32"`
- Kept the Rodas state/error-control path in `float64`, while allowing the LU workspace / solve path to run in `float32`.
- Tried iterative refinement for mixed-precision solves, but removed it because the residual-correction path was too slow and too memory-heavy in practice.
- Tried converting the stored Jacobian to `float32` in the FP32 path; correctness held, but runtime did not improve materially.

## Tensor Core / Blocked LU Work

- Added `solvers/rodas5_custom_kernel_v7_tc.py` as an experimental blocked-LU solver using dense batched JAX arrays.
- The `v7` path performs blocked no-pivot LU in `float32` and uses TF32-style dot operations for trailing matrix updates.
- Switched the `v7` dense matmul path from `lax.dot_general` to `lax.dot`, then to `pl.dot`-based batched calls. This improved the `50d, N=1000` runtime.
- Added a public `tc_block_lu` toggle to `v6`.
- Current stable behavior: `tc_block_lu=True` routes through the experimental blocked-LU backend, while `tc_block_lu=False` preserves the original `v6` Pallas kernel.

## Test Suite Changes

- Renamed `tests/test_30d.py` to `tests/test_nn_reactions.py`.
- Generalized the nearest-neighbor reaction tests to configurable dimension `D` via pytest parametrization.
- Added `v6` tests for:
  - time-dependent closed-form correctness
  - multi-save output correctness
  - ensemble runtime benchmarks
  - compile-time estimates
  - invalid `t_span` rejection
  - FP32-vs-FP64 agreement
- Added `v7` tests for:
  - `50d` closed-form correctness
  - `50d` ensemble benchmarks over multiple ensemble sizes
  - `50d` compile-time estimate
- Added focused `v6 tc_block_lu` tests for:
  - `50d` correctness in both `fp64` and `fp32`
  - `fp32`, `N=1000`, `50d` runtime comparison against scalar `v6`
  - compile-time smoke test

## Measured Results

Observed during development in this environment:

- `v6 fp64`, `50d`, `N=1000`: about `117.23 ms`
- `v6 fp32`, `50d`, `N=1000`: about `85.04 ms`
- `v7 tc`, `50d`, `N=1000`:
  - about `27.86 ms` before the `pl.dot` switch
  - about `24.92 ms` after the `pl.dot` switch
- `v7 tc`, `50d` compile-time estimate: about `16.58-16.67 s`

The focused `v6 tc_block_lu` runtime comparison test passed for the requested case:

- `linear_solver="fp32"`
- `D=50`
- `N=1000`
- assertion: `tc_block_lu=True` is faster than the scalar FP32 `v6` path

## Direct In-Kernel `_BLOCK=32` Blocked-LU Attempt

I also tried to port the blocked-LU path directly into the existing `_BLOCK=32` Pallas kernel in `v6`.

What was attempted:

- replacing large Python-unrolled loops with `jax.lax.fori_loop`
- keeping `pl.dot` only for per-lane 2D trailing updates
- building per-lane `L21` / `U12` tiles inside the kernel
- avoiding scatter by using ref writes and more static control flow

What blocked that approach on this Triton/Pallas stack:

- unsupported `scatter-add`
- unsupported `scatter`
- unsupported `vmap` over nontrivial ref slices
- unsupported non-power-of-two temporary shapes
- unsupported `concatenate` patterns
- unsupported `slice` / `dynamic_slice` in the tile assembly / writeback path

Current outcome:

- the direct in-kernel `_BLOCK=32` blocked-LU path is not active
- the stable public `tc_block_lu` toggle currently routes through the experimental blocked-LU backend instead

## Current Status

- `v6` remains the main solver for the Pallas kernel path.
- `v6` supports:
  - time-dependent Jacobians
  - multi-time saves
  - `fp64` / `fp32` linear solve modes
  - `tc_block_lu` as an experimental toggle
- `v7` remains in the repo as the experimental blocked-LU / Tensor Core-style reference implementation.
- The main unresolved item is a fully direct in-kernel `_BLOCK=32` blocked-LU implementation that lowers cleanly on this Triton/Pallas version.
