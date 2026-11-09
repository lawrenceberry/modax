# Hierarchies of one-way-coupled blocks

Some stiff systems contain a block of equations that couple one-way to many other components in the state. modax can integrate such a system as a
*hierarchy* of sub-systems, each with their own steps and if need be, their own method, instead of as one system.

```python
import modax
from modax import Block, Hierarchy

def thermal(y, t, p):                      # a root block: states y, parameters p
    return (-p[0] * y[0] + math.cos(t),)

def published(y, t, p):                    # what the others read of it
    return (math.sin(y[0]), y[0] * t)

def hierarchy(y, t, p, q):                 # q: the upstream quantities, in order
    return (-p[1] * (y[0] - q[0]) + q[1],)

driver = Block(thermal, n=1, outputs=published, n_outputs=2)
main = Block(hierarchy, n=1, reads=(driver,), sparsity=pattern)
sol = modax.solve(Hierarchy(main), y0, t_span, params)   # y0: the blocks' states, upstream first
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

Each sub-system may use a different method, and the blocks step independently:

```python
import modax
from modax import Auto, Block, Hierarchy, Rodas5P, Tsit5

background = Block(expansion, 3, outputs=rates, n_outputs=2, method=Tsit5())
thermal = Block(recombination, 4, reads=(background,), outputs=opacity,
                n_outputs=4, method=Rodas5P())
main = Block(boltzmann, 51, reads=(background, thermal), sparsity=pattern,
             method=Rodas5P(lu_precision="fp32"))

sol = modax.solve(Hierarchy(main), y0, t_span, params, rtol=1e-4, atol=1e-4)
```
