"""Compile generated straight-line device source into a ``cuda.jit`` function.

Two parts of the Rodas5P kernel are emitted as source rather than written: the
sparse factorisation and triangular solves, with every slot a literal
(:py:obj:`modax._sparse_direct`), and the Jacobian writer, with every colour's
seed row a literal (``modax.rodas5P``). Both are load-bearing -- see the
measurements in AGENTS.md -- and both need the same three things done to the
text they produce, which is what this module does once.
"""

from __future__ import annotations

import hashlib
import linecache
from typing import Mapping, Sequence

from numba_cuda_mlir import cuda
from numba_cuda_mlir.descriptor import MLIRDispatcher


def compile_device_source(
    name: str,
    lines: Sequence[str],
    namespace: Mapping[str, object] | None = None,
    inline: str = "always",
) -> MLIRDispatcher:
    """Exec ``lines`` defining ``name`` and return it as a device function.

    The source is registered with ``linecache`` under a filename of its own, so
    a numba typing error inside it points at the offending line rather than at
    nothing, and the text stays readable from a debugger as
    ``fn._generated_source``. ``namespace`` supplies whatever the generated body
    refers to by name. ``inline="never"`` keeps the function out of numba's
    IR-level inlining (and LLVM's): numba deep-copies an inlined callee's IR at
    every call site at every level of the call tree, which for a generated
    right-hand side of fifty states is minutes of build time, where a call
    costs the kernel nothing it can measure.
    """
    source = "\n".join(lines) + "\n"
    # named by the source's digest, so that the name is the same in every process
    filename = (
        f"<modax generated {name} {hashlib.sha256(source.encode()).hexdigest()[:12]}>"
    )
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict[str, object] = {}
    exec(compile(source, filename, "exec"), dict(namespace or {}), scope)  # noqa: S102
    generated = scope[name]
    generated._generated_source = source  # ty: ignore[unresolved-attribute]
    return cuda.jit(device=True, inline=inline)(generated)
