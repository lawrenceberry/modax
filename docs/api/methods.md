# Methods

How {py:func}`modax.solve` steps: pass one of these as `method=`, or give
one to a {py:class}`~modax.Block`. Each object holds the options of its
method's numerics; everything about the problem and the launch is an
argument of `solve`.

| Method | Type | Use for |
|---|---|---|
| {py:class}`~modax.Tsit5` | Explicit Runge-Kutta, order 5 | Non-stiff systems |
| {py:class}`~modax.Rodas5P` | Rosenbrock-W, order 5 | Stiff systems (the default) |
| {py:class}`~modax.Auto` | Switches between the two | Systems stiff only some of the time |

```{toctree}
:maxdepth: 1

methods/tsit5
methods/rodas5P
methods/auto
```
