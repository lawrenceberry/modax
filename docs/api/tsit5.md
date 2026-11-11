# Tsit5

The explicit method, for non-stiff systems. Pass `method=modax.Tsit5()` to
{py:func}`modax.solve`, or give it to a {py:class}`~modax.Block`. Tsit5
takes none of `solve`'s arguments marked "Rodas5P only".

```{eval-rst}
.. autoclass:: modax.Tsit5
```
