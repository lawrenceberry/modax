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
    raise AttributeError(name)
