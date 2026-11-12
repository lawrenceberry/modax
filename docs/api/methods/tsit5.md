# Tsit5

The explicit method, for non-stiff systems. Pass `method=modax.Tsit5()` to
{py:func}`modax.solve`, or give it to a {py:class}`~modax.Block`. A plain
Tsit5 solve that asks for a save hook, `cache` or launch tuning
runs as a one-block hierarchy, still with Tsit5.

```{eval-rst}
.. autoclass:: modax.Tsit5
```
