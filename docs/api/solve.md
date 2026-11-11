# solve

`modax.solve` is the one entry point. It integrates a plain right-hand side,
a {py:class}`~modax.Block` or a {py:class}`~modax.Hierarchy` of blocks with
the method you choose: {doc}`tsit5`, {doc}`rodas5P` or
{py:class}`~modax.Auto`, which switches between them. Importing `modax`
enables JAX float64.

```{eval-rst}
.. autofunction:: modax.solve
```

## Hierarchies

```{eval-rst}
.. autoclass:: modax.Block
   :members:

.. autoclass:: modax.Hierarchy
   :members:
```

## Switching methods

```{eval-rst}
.. autoclass:: modax.Auto
```
