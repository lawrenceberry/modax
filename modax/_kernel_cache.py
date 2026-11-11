"""The compiled kernel on disk, keyed by what it was built from.

numba-cuda-mlir can cache a compiled kernel on disk (``cache=True``), keyed on
the kernel's bytecode and a pickle of its closure. The closure of a modax
kernel is a graph of device functions, and a pickled dispatcher carries a
uuid drawn afresh in every process, so that key never matches twice: every
process rebuilt the kernel, minutes of numba inlining for a 50-state system.

:func:`content_digest` walks the same graph and hashes what determines the
code instead -- bytecode, constants, closure cells, the globals a function
reads, the arrays and dispatchers it reaches, the primal behind a lazy
numba-enzyme derivative -- and :class:`ContentKeyedCache` puts that digest in
the key. A right-hand side edited in place, a changed pattern or option, a
new numba-cuda-mlir or numba-enzyme: all change the digest; the same build
in a new process loads in a second.
"""

from __future__ import annotations

import hashlib
import os
import types
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from numba_cuda_mlir.caching import MLIRCache, MLIRCacheImpl
from numba_cuda_mlir.numba_cuda.core.caching import CacheImpl, _CacheLocator

_SCALARS = (int, float, complex, bool, str, bytes, type(None))


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def content_digest(root) -> str:
    """A SHA-256 hex digest of everything the compiled form of ``root`` depends on."""

    hasher = hashlib.sha256()
    seen: set[int] = set()
    for name in ("numba-cuda-mlir", "numba-enzyme-cuda", "cuda-bindings"):
        hasher.update(f"{name}={_package_version(name)};".encode())

    def feed(tag: str, payload: str | bytes):
        hasher.update(tag.encode())
        hasher.update(payload if isinstance(payload, bytes) else payload.encode())
        hasher.update(b";")

    def walk(obj):
        if isinstance(obj, _SCALARS):
            feed(type(obj).__name__, repr(obj))
            return
        if isinstance(obj, np.generic):
            feed("np", repr(obj))
            return
        if id(obj) in seen:
            feed("seen", str(id(obj) in seen))  # a back edge, not the object again
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            feed("ndarray", f"{obj.dtype}{obj.shape}")
            feed("data", np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, (tuple, list, frozenset, set)):
            items = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
            feed(type(obj).__name__, str(len(obj)))
            for item in items:
                walk(item)
        elif isinstance(obj, dict):
            feed("dict", str(len(obj)))
            for key in sorted(obj, key=repr):
                walk(key)
                walk(obj[key])
        elif isinstance(obj, types.CodeType):
            feed("code", obj.co_code)
            feed("names", repr(obj.co_names))
            feed("varnames", repr(obj.co_varnames))
            walk(obj.co_consts)
        elif isinstance(obj, types.FunctionType):
            feed("function", obj.__qualname__)
            walk(obj.__code__)
            walk(obj.__defaults__)
            walk(obj.__kwdefaults__)
            if obj.__closure__:
                for cell in obj.__closure__:
                    try:
                        walk(cell.cell_contents)
                    except ValueError:
                        feed("cell", "empty")
            g = obj.__globals__
            for name in obj.__code__.co_names:
                if name in g:
                    feed("global", name)
                    walk(g[name])
            # a numba-enzyme placeholder keeps its primal as an attribute
            for name, value in sorted(vars(obj).items()):
                if not name.startswith("__"):
                    feed("attr", name)
                    walk(value)
        elif type(obj).__name__ == "MLIRDispatcher":
            feed("dispatcher", "")
            walk(obj.py_func)
            walk(obj.targetoptions)
        elif isinstance(obj, types.ModuleType):
            feed("module", obj.__name__)
        elif isinstance(obj, type):
            feed("class", f"{obj.__module__}.{obj.__qualname__}")
        elif hasattr(obj, "__dict__") and not callable(obj):
            feed("object", type(obj).__qualname__)
            walk(vars(obj))
        else:
            feed("repr", repr(obj))  # numba types, enums, slices: addressless reprs

    walk(root)
    return hasher.hexdigest()


class _GeneratedLocator(_CacheLocator):
    """Where a function generated from source (:mod:`modax._codegen`) is cached: beside this package.

    numba's locators want a source file on disk; a generated function has
    only its ``<modax generated ...>`` name. Its freshness is in the content
    digest of the key, so the source stamp is constant.
    """

    def __init__(self, py_func):
        self._name = py_func.__name__
        self._cache_path = os.environ.get("NUMBA_CACHE_DIR") or os.path.join(
            os.path.dirname(__file__), "__pycache__"
        )

    def get_cache_path(self):
        return self._cache_path

    def get_source_stamp(self):
        return 0

    def get_disambiguator(self):
        return self._name

    @classmethod
    def from_function(cls, py_func, py_file):
        return cls(py_func) if str(py_file).startswith("<modax generated") else None


class _ContentKeyedImpl(MLIRCacheImpl):
    _locator_classes = [_GeneratedLocator, *CacheImpl._locator_classes]


class ContentKeyedCache(MLIRCache):
    """numba-cuda-mlir's kernel cache with :func:`content_digest` in place of the closure pickle."""

    _impl_class = _ContentKeyedImpl

    def __init__(self, py_func, targetoptions, digest: str):
        self._digest = digest
        super().__init__(py_func, targetoptions)

    def _index_key(self, sig, codegen):
        sig_key, magic, (code_hash, _closure_hash), option_key = super()._index_key(
            sig, codegen
        )
        return (sig_key, magic, (code_hash, self._digest), option_key)


def enable_content_cache(kernel) -> None:
    """Give a kernel dispatcher the on-disk cache keyed by its content."""

    kernel._cache = ContentKeyedCache(
        kernel.py_func, kernel.targetoptions, content_digest(kernel.py_func)
    )
