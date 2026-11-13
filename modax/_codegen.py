"""Compile generated straight-line device source into a ``cuda.jit`` function.

The Rodas5P kernel emits two parts as source: the sparse factorisation and
triangular solves, with every slot a literal (:py:obj:`modax._sparse_direct`),
and the Jacobian writer, with every colour's seed row a literal
(``modax._rodas5P``). Both need the same handling of the text they produce,
and this module does it in one place.
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

    The source goes into ``linecache`` under its own filename, so a numba typing
    error inside it points at the offending line. A debugger can read the text
    as ``fn._generated_source``. ``namespace`` supplies the names the generated
    body refers to.

    ``inline="never"`` keeps the function out of numba's and LLVM's inlining.
    numba deep-copies an inlined callee's IR at every call site, which costs
    minutes of build time for a fifty-state right-hand side, while the call
    costs the kernel nothing measurable.
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
