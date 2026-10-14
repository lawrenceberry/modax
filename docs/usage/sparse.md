# Exploiting sparsity

Rodas5P takes a `sparsity` pattern describing which inputs of the right-hand
side affect which outputs. It is the only argument involved: there is no
linear solver to write or to select, because the kernel builds one from the
pattern itself.

```python
y = solve(ode_fn, y0, t_span, params,
          sparsity=pattern)   # (n_vars, n_vars) mask, scipy sparse, or (nnz, 2)
```

The pattern is used in two places. The Jacobian is recovered in one Enzyme
sweep per colour of the pattern's column intersection graph rather than one
per column, since columns that share no row can be seeded together and the
pattern says which output component belongs to which. And the iteration
matrix $M = I/(h\gamma) - J$ is factorised by a direct sparse solver compiled
for that exact structure, an in-kernel sparse LU and a pair of sparse
triangular solves in place of the dense ones, still one trajectory per thread.
The pattern must be a superset of the true non-zeros: colouring a superset
only costs extra sweeps, whereas colouring a subset silently corrupts the
entries where two columns of a group overlap after all. It need not include
the fill-in of the factorisation, which is computed from it. Without a pattern
every column gets its own colour and the matrix is a dense row-major array
factorised with partial pivoting; this is the same mechanism at its
uninformative end, and the dense and sparse solvers have the same shape, so
the kernel does not branch between them.

| Module | What it does |
|--------|--------------|
| [`modax._sparsity`][] | colours the pattern's column intersection graph and defines `CompressedJacobian`, the layout the Enzyme sweeps write into |
| [`modax._sparse_direct`][] | orders the pattern, factorises it symbolically, and compiles the sparse LU and triangular solves for that structure |

Both are documented under [API reference / Internals](../api/internals.md).

## Ordering

The variables are reordered to reduce fill-in using SuiteSparse's approximate
minimum degree (AMD) algorithm, which reaches the package through
[cvxopt](https://cvxopt.org); its wheel carries AMD, so no system library is
needed. `ordering="natural"` skips the reordering, and those are the two
options.

AMD was chosen over COLAMD, the other obvious candidate. COLAMD orders the
columns so that fill stays bounded whatever row permutation partial pivoting
later chooses, which is the right objective when there will be pivoting. There
is none here, because the pattern is compiled into the kernel and cannot
depend on the numbers. COLAMD's permutation is also one-sided, so it moves the
diagonal off the diagonal, whereas this factorisation needs the diagonal where
$I/(h\gamma)$ puts it. AMD instead approximately minimises the fill of the
Cholesky factor of $S + S^\top$, which is the standard bound on the fill of an
unpivoted LU of $S$, and it does so with a symmetric permutation $P S P^\top$
that leaves every diagonal entry in place. UMFPACK and SuperLU use it in their
symmetric modes for the same reasons, and an iteration matrix is about as
close to structurally symmetric as an unsymmetric matrix gets.

CHOLMOD's other orderings were available while scikit-sparse was a dependency,
and measuring them is what led to its removal: `colamd`, `nesdis` and `best`
each reproduced AMD's fill exactly, and `metis` produced more fill wherever it
differed.

## Pivoting

There is no pivoting. The pattern has to be fixed at compile time and the same
in every thread, so rows cannot be swapped on the numbers, and doing so would
in any case reintroduce the warp divergence that one trajectory per thread is
there to avoid. Two things make the omission safe. Because the permutation is
symmetric, the diagonal of $M$ stays on the diagonal, so $I/(h\gamma)$
guarantees that every pivot is structurally present and grows without bound as
the step shrinks. And Rodas5P satisfies the Rosenbrock-W order conditions to
second order, so an approximate factorisation costs order and step size rather
than convergence; a badly conditioned pivot is a matter for the step-size
controller. A pivot that reaches exactly zero leaves an infinity in the
factors, the error norm becomes NaN, the step is rejected, and the smaller step
puts a larger $1/(h\gamma)$ on that diagonal.

## Symbolic factorisation

The pattern of $L + U$ is obtained by Gaussian elimination on the pattern
alone, which is exact: it is what the numeric factorisation will touch, no
more and no less. Factorising a sample matrix and recording where the
non-zeros land cannot be, since a coefficient that happens to vanish for those
particular numbers, or an exact cancellation, drops an entry another
right-hand side would need, and the buffer is then one slot short in a kernel
with no way to report it. The symbolic pass is also cheaper, needing neither a
plausible matrix nor a device. It is implemented as a dense boolean
elimination over the whole matrix, one numpy row OR per update, so it is
$O(n^3)$ time at worst and $O(n^2)$ bytes. For the tens to a few hundred
variables these solvers target this takes milliseconds, and a sparse symbolic
algorithm would gain nothing.

## Sparse storage

The factorised pattern is laid out as a single compressed sparse row (CSR)
image of $L + U$. CSR rather than CSC because each of the three routines that
reads the matrix reads it by rows: the up-looking factorisation takes row $i$
and subtracts multiples of the rows above it, the forward substitution is a
dot product of row $i$ of $L$ with the solution so far, and the back
substitution is the same over row $i$ of $U$. One row-major image therefore
serves all three, where CSC would have to be transposed for two of them, and a
column-oriented factorisation would still leave the solves wanting rows. $L$
and $U$ share the image, $L$ strictly left of the diagonal and $U$ from the
diagonal rightwards, since the factorisation is in place and a unit diagonal
needs no storage. The buffer is thus exactly $\mathrm{nnz}(L + U)$ elements of
per-thread local memory, and that is the solver's entire footprint.

The Jacobian is written straight into this buffer. Colouring and storage are
separate questions, and the colour sweeps deposit $-J$ at the CSR slots the
factorisation will read, after the buffer has been cleared wherever it holds
slots no sweep writes, such as the fill-in. Nothing is staged through global
memory and read back, and nothing is expanded to a dense matrix in between.
One consequence is that the pattern can be declared as tightly as it really
is. A hand-written solver that owned its own buffer had to declare its fill-in
in the pattern to have somewhere to put it, and paid extra colours for doing
so.

## Compiled code

Where the structure is small enough, the factorisation and the two triangular
solves are emitted as straight-line code with every slot a literal; above
`MAX_UNROLLED_SUBSTITUTIONS` and `MAX_UNROLLED_UPDATES` they fall back to
loops over index tables in constant memory. The reason is latency. A
table-driven sparse routine spends a broadcast load on the index of every
value before it can issue the load of the value itself, and that dependent
pair of loads is only hidden when enough other trajectories are in flight to
cover it, and an ensemble of a few warps on a device of tens of SMs has
nothing to hide it with. Unrolling costs no registers,
because the kernel indexes both the matrix and the right-hand side with loop
variables of its own elsewhere, so neither can leave local memory; the
exchange is index loads for instruction count. On a small ensemble of a
50-variable system, unrolling the triangular solves cut the solve time by
about a fifth, and unrolling the factorisation as well by a quarter. The two
emissions are tested against each other and required to agree bit for bit.
