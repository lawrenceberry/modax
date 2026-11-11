"""Compile each device function once, not twice, in numba-cuda-mlir.

numba-cuda-mlir (0.5.1) compiles a device function called from another twice.
Typing the caller compiles the callee to learn its return type
(``MLIRDispatcher.get_call_template``, with no ABI name), and lowering the
caller then asks for it under the mangled name it links it by
(``build_user_defined_function_call``: ``<qualname>_<id(dispatcher)>`` and
the argument types); the cached result is only reused under the same name,
so the whole function -- typing, MLIR lowering, optimisation -- is compiled
again. For a hierarchy kernel that is every step, stage, Jacobian writer,
right-hand side and factorisation, some 6 s of a 36 s build.

:func:`install` makes the typing-time compile use the name lowering will ask
for, so lowering finds it cached. The compiled code is the same function
under the same name; if a call's argument types fold differently in the two
places the names differ and the callee is compiled twice, as before. Set
``MODAX_COMPILE_CALLEES_TWICE=1`` to leave numba-cuda-mlir untouched.
"""

from __future__ import annotations

import os

_INSTALLED: list[bool] = []


def install() -> None:
    """Patch ``MLIRDispatcher.get_call_template`` once per process (idempotent)."""
    if _INSTALLED or os.environ.get("MODAX_COMPILE_CALLEES_TWICE"):
        return
    from numba_cuda_mlir import descriptor
    from numba_cuda_mlir.numba_cuda import typing
    from numba_cuda_mlir.tools import generate_mangled_name

    dispatcher_cls = descriptor.MLIRDispatcher
    if not all(
        hasattr(dispatcher_cls, name)
        for name in (
            "get_call_template",
            "_compile_as_device_callee",
            "_compile_device_callee",
        )
    ):
        return  # a numba-cuda-mlir this was not written against: leave it alone

    def lowering_dispatcher(self):
        """The dispatcher ``build_user_defined_function_call`` will compile through."""
        if self.targetoptions.get("device", False):
            return self
        # The device twin _compile_as_device_callee keeps, created with its
        # options so it is not replaced (lowering reuses the same attribute).
        opts = self.targetoptions.copy()
        opts["device"] = True
        opts["lto"] = False
        twin = getattr(self, "_device_dispatcher", None)
        if twin is None or twin.targetoptions != opts:
            twin = descriptor.MLIRDispatcher(self.py_func, targetoptions=opts)
            self._device_dispatcher = twin
        return twin

    def get_call_template(self, args, kws):
        pysig, args = self._compiler.fold_argument_types(args, kws)
        kws = {}
        if self._can_compile:
            fn = lowering_dispatcher(self)
            abi_name = generate_mangled_name(
                f"{fn.py_func.__qualname__}_{id(fn)}", tuple(args)
            )
            self._compile_as_device_callee(tuple(args), abi_name=abi_name)
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures
        )
        return call_template, pysig, args, kws

    dispatcher_cls.get_call_template = get_call_template
    _INSTALLED.append(True)
