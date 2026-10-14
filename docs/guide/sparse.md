# Sparse systems

Rodas5P takes a `sparsity` pattern, and that one argument is the whole
interface — there is no linear solver to write or to pass:

```python
y = solve(ode_fn, y0, t_span, params,
          sparsity=pattern)   # (n_vars, n_vars) mask, scipy sparse, or (nnz, 2)
```

A pattern buys two separate things. The Jacobian is recovered in one Enzyme
sweep per *colour* of the pattern's column intersection graph rather than one
per column, since columns sharing no row can be seeded together and the pattern
says which output component belongs to which. And the iteration matrix
`M = I/(hγ) − J` gets a **direct sparse solver compiled for that exact
structure**: an in-kernel sparse LU and a pair of sparse triangular solves, one
trajectory per thread, in place of the dense LU. The pattern must be a superset
of the true nonzeros — colouring a superset only costs sweeps, colouring a
subset silently corrupts entries — but it need *not* cover the factorisation's
fill-in, which is worked out from it. With no pattern, every column gets its own
colour and the matrix is factorised densely: the same mechanism at its
uninformative end rather than a second code path.

On DISCO-EB's 50-variable Einstein-Boltzmann system this is **22% faster** than
the hand-written Schur block-LU it replaced, and it asks nothing of the caller
but the pattern.

## The choices behind it

All of the analysis happens once, on the host, when the kernel is built
(`modax/_sparse_direct.py`).

**AMD for the ordering, not COLAMD.** The obvious alternative, COLAMD, orders
the *columns* so that fill stays bounded whatever row permutation partial
pivoting later chooses. That is the right objective exactly when there will be
pivoting — and there will not be, because the pattern is compiled into the
kernel and cannot depend on the numbers. COLAMD's permutation is also one-sided,
so it moves the diagonal off the diagonal, and this factorisation needs the
diagonal precisely where `I/(hγ)` puts it. AMD instead minimises (approximately)
the fill of the Cholesky factor of `S + Sᵀ`, which is the standard bound on the
fill of an unpivoted `LU` of `S`, and it does so with a *symmetric* permutation
`P S Pᵀ` that leaves every diagonal entry on the diagonal. It is what UMFPACK
and SuperLU use in their "symmetric mode", for these reasons, and an iteration
matrix is about as close to structurally symmetric as an unsymmetric matrix
gets. It comes from SuiteSparse through [cvxopt](https://cvxopt.org), whose
wheel carries AMD itself, so no system library is involved;
`ordering="natural"` skips the ordering and those two are the whole of the
choice. CHOLMOD's other orderings were on offer while scikit-sparse was a
dependency, and measuring them is what retired it: `colamd`, `nesdis` and
`best` each returned AMD's own fill to the entry, and `metis` was strictly
worse where it differed — 660 nonzeros against 484 on the Einstein-Boltzmann
structure below, losing the perfect elimination order.

**No pivoting at all.** The pattern has to be fixed at compile time and the same
in every thread, so rows cannot be swapped on the numbers — which would also
reintroduce the warp divergence one-trajectory-per-thread is there to avoid. Two
things make that sound. The permutation is symmetric, so `M`'s diagonal stays on
the diagonal and `I/(hγ)` guarantees every pivot is structurally present and
grows without bound as the step shrinks. And Rodas5P is a Rosenbrock-**W**
method: order 5 survives an approximate factorisation, so a badly conditioned
pivot costs step-size control rather than correctness, and the controller is
what notices. A pivot that reaches exactly zero leaves an infinity, the error
norm goes to NaN, the step is rejected, and the smaller step puts a larger
`1/(hγ)` on that very diagonal.

**A symbolic factorisation for the footprint, not a trial numeric one.** The
`L + U` pattern comes from pattern-only Gaussian elimination, which is exact: it
is what the numeric factorisation will touch, no more and no less. Factorising a
sample matrix and counting cannot be — a coefficient that happens to vanish for
those particular numbers, or an exact cancellation, drops an entry another
right-hand side needs, and the buffer is then one slot short in a kernel with no
way to say so. It is also cheaper, needing neither a plausible matrix nor a
device. The implementation is bit-per-entry over the whole matrix, `O(n³/64)`
time and `O(n²)` bits, which for the tens-to-a-few-hundred variables these
solvers target analyses in milliseconds and buys nothing back from a sparse
symbolic algorithm.

**CSR, not CSC.** Every one of the three routines that reads the matrix reads it
*by rows*: the up-looking factorisation takes row `i` and subtracts multiples of
the rows above it, the forward substitution is a dot product of row `i` of `L`
with the solution so far, and the back substitution is the same over row `i` of
`U`. One row-major image serves all three. CSC would have to be transposed for
two of them, and a column-oriented factorisation would still leave the solves
wanting rows. `L` and `U` share that one image — `L` strictly left of the
diagonal, `U` from it rightwards — because the factorisation is in place and a
unit diagonal needs no storage, so the buffer is exactly `nnz(L + U)`, which is
per-thread local memory and the thing that bounds occupancy.

**The Jacobian is written straight into the factorisation's buffer.** Colouring
and storage are separate questions, and the AD's colour sweeps deposit `−J` at
the CSR slots the factorisation will read, with the fill-in slots simply cleared
beforehand. Nothing is staged through global memory and read back, and nothing is
expanded to a dense matrix in between. It also means the pattern may be declared
as tightly as it really is: a hand-written solver owning its own buffer had to
declare its fill-in in the pattern to have somewhere to put it, and paid colours
for that.

**Straight-line code where it fits.** Table-driven, a sparse routine spends a
broadcast load on the index of every value before it can issue the load of the
value itself, and that dependent pair is only free when enough other
trajectories are in flight to cover it. DISCO-EB's single-cosmology case is 128
trajectories — four warps on a 46-SM device — and nothing covers it. So below
`MAX_UNROLLED_SUBSTITUTIONS` / `MAX_UNROLLED_UPDATES` the routines are emitted
as straight-line code with every slot a literal, and above them they fall back
to loops over index tables in constant memory. Unrolling costs no registers,
since the kernel indexes both the matrix and the right-hand side with loop
variables of its own and neither can leave local memory whatever this does — it
trades index loads for instruction count and nothing else. Measured on DISCO-EB
at N128: **528 ms** table-driven, **419 ms** with the solves unrolled, **398 ms**
with the factorisation unrolled too, against **509 ms** for the hand-written
Schur solver. The two emissions are checked against each other and required to
agree bit for bit.

**What it finds on a real problem.** DISCO-EB's Einstein-Boltzmann Jacobian is a
densely coupled core bordered by tridiagonal free-streaming hierarchies. AMD
returns a *perfect* elimination order for it — zero fill, `nnz(L + U) = nnz(J)` —
and the order it finds is the hand-written Schur solver's: peel each hierarchy
from its truncated end inwards, where every variable has degree two, then
eliminate the dense core last.

## Where the pieces live

| Module | What it does |
|--------|--------------|
| [`modax._sparsity`][] | colours the pattern's column intersection graph and defines `CompressedJacobian`, the layout the Enzyme sweeps write into |
| [`modax._sparse_direct`][] | orders the pattern, factorises it symbolically, and compiles the sparse LU and triangular solves for that structure |

Both are documented under [API reference / Internals](../api/internals.md).
