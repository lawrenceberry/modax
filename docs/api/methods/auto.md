# Auto

Switches a system or a block between {doc}`tsit5` and {doc}`rodas5P` as it
turns stiff and back. Pass `method=modax.Auto()` to {py:func}`modax.solve`,
or give it to a {py:class}`~modax.Block`. See
[Switching methods](../../usage/switching-methods.md) for how the switch
decides.

```{eval-rst}
.. autoclass:: modax.Auto
```
