# Solvers

Both solvers expose the same `solve(...)` entry point, and `modax.solve`
takes the method as an argument (a plain system, a `Block` or a `Hierarchy`).
Importing `modax` enables JAX float64.

```{eval-rst}
.. automodule:: modax._solve
```

```{eval-rst}
.. automodule:: modax._methods
```

```{eval-rst}
.. automodule:: modax.rodas5P
```

```{eval-rst}
.. automodule:: modax.tsit5
```
