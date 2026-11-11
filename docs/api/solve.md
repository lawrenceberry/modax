# solve

`modax.solve` is the one entry point. It integrates a plain right-hand side,
a {py:class}`~modax.Block` or a {py:class}`~modax.Hierarchy` of blocks. The
method is one object, `method=`: {py:class}`~modax.Tsit5`,
{py:class}`~modax.Rodas5P` (the default) or {py:class}`~modax.Auto`, which
switches between them, each carrying its own options (see {doc}`methods`). Every other argument describes the problem or the launch.
Importing `modax` enables JAX float64.

```{eval-rst}
.. autofunction:: modax.solve
```

## Step control

```{eval-rst}
.. autoclass:: modax.PIDGains
```

## Hierarchies

```{eval-rst}
.. autoclass:: modax.Block
   :members:

.. autoclass:: modax.Hierarchy
   :members:
```
