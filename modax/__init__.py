"""Solver package initialization."""

import jax

jax.config.update("jax_enable_x64", True)


def __getattr__(name: str):
    # A hierarchy of one-way-coupled blocks, each on steps of its own
    # (modax._hierarchy); imported on first use so that ``import modax`` stays
    # free of the CUDA toolchain.
    if name in ("Block", "Hierarchy"):
        from modax import _hierarchy

        return getattr(_hierarchy, name)
    # How a block steps (modax._methods), and the solve that takes one.
    if name in ("Rodas5P", "Tsit5", "Auto"):
        from modax import _methods

        return getattr(_methods, name)
    if name == "solve":
        from modax._solve import solve

        return solve
    raise AttributeError(name)
