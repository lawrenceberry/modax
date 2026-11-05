# Hierarchies of one-way-coupled blocks

Some stiff systems have a block of equations that everything else reads and
that reads nothing back: a thermal history under a Boltzmann hierarchy, a
chemistry that drives a transport equation, a slowly evolving environment
under a fast population. Rodas5P can integrate such a system as a
*hierarchy* of blocks, each on steps of its own, instead of as one system.

```python
from modax import Block, Hierarchy
from modax.rodas5P import solve

def thermal(y, t, p):                      # a root block: states y, parameters p
    return (-p[0] * y[0] + math.cos(t),)

def published(y, t, p):                    # what the others read of it
    return (math.sin(y[0]), y[0] * t)

def hierarchy(y, t, p, q):                 # q: the upstream quantities, in order
    return (-p[1] * (y[0] - q[0]) + q[1],)

driver = Block(thermal, n=1, outputs=published, n_outputs=2)
main = Block(hierarchy, n=1, reads=(driver,), sparsity=pattern)
sol = solve(Hierarchy(main), y0, t_span, params)   # y0: the blocks' states, upstream first
```

A `Block` is a set of `n` states with a right-hand side of its own. It may
`read` other blocks, whose published quantities it receives as the tuple `q`
(block after block, in the order of `reads`), and it may publish quantities
of its own through `outputs` (its states when `None`). The blocks form a
directed acyclic graph: nothing feeds back. `Hierarchy(root)` collects the
blocks the root reaches, upstream first; the flat state the solve takes and
returns holds their states in that order (or in the order of `layout`, when
given). Each block carries its own `sparsity`, the pattern of its right-hand
side with respect to its own states.

The pattern split, `solve(ode_fn, ..., sparsity=pattern, split=True)`, is the
same machinery found from a single right-hand side: the strongly connected
components of the pattern that nothing downstream feeds into form a block
that publishes its states, and the rest reads them.

## Why

A state every downstream row reads is a full column of the joint Jacobian,
and a sparse factorisation of a full column fills in. Split, each block
factorises its own matrix, and a downstream block's Jacobian has no upstream
columns at all: the upstream quantities enter its right-hand side as known
functions of time. And each block takes the steps its own dynamics allow. A
smooth upstream block takes long steps where the downstream block takes many,
so its right-hand side, and the quantities it publishes, are evaluated once
per *upstream* step rather than once per stage of every downstream step.

## How

Every block runs ahead of its readers on Rodas5P steps of its own, accepted
or rejected by its own error norm and controller, keeping a ring of nodes: at
each accepted step's end its state, rate and second derivative
($y'' = J y' + \partial_t f$), and for the quantities it publishes their
values and first two time derivatives. Between nodes its dense output is the
quintic Hermite interpolant of that data, $O(H^6)$ over a step $H$, one order
beyond the method's local error, so a reader loses no order. A reader about
to step over $[t, t + h]$ reads that dense output at both ends and writes
the twelve numbers per quantity into its thread-local copy of the parameter
row; its generated right-hand side evaluates the quintic there, so the
reader's own step -- its stages, its Jacobian sweeps, its time sweep -- sees
the upstream as a polynomial of $t$. The warp decides when a block advances:
whenever any lane's reader needs more of an upstream block, every lane with
room in its ring steps that block too, so the lanes stay together.

## Error control

Each block's steps are held to `rtol` and `atol` by its own weighted RMS norm
over its own rows. The error the upstream makes is handed down as well: a
reader seeds one extra direction of its Jacobian sweeps with the upstream's
error estimate over the piece its step lies in, reads off the rate
$\partial f / \partial q \cdot \epsilon$ at which that error enters its rows,
and turns it into the error it would make over the upstream step, damped
where a row restores itself faster than that. This *inherited* error cannot
be reduced by a shorter downstream step -- the upstream's error is already
made -- so it is fed to the upstream block's controller, whose next step
shrinks until the increments it hands down are as accurate as its own. The
tolerance then means what it does for a single system: every block's local
error, its own and what it inherits, is held to the tolerance.

## Sensitivities

Forward sensitivities ride on every block. A block carries the parameter
directions and the initial-state directions of its own states and its
upstream blocks'; the parameter-direction seed of its sweeps is the unit
vector plus the Hermite coefficients of the upstream quantities'
sensitivities in that direction, so $\partial f / \partial q \, S_q(t)$ comes
out of the same forward sweep as $J_y S_y + J_p$. A published quantity's
sensitivity and its rate come from first- and second-order sweeps of the
quantity function; its second derivative, which would need a third-order
sweep, comes from the quintic through the sensitivity's value and rate at the
two nodes and at the midpoint of the step, $O(H^6)$ all the same.
