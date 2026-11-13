"""``KernelOptions``: everything besides the callback and the shapes that selects a kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modax._methods import (
    LUPrecision,
    Method,
    Ordering,
    Rodas5P,
    method_parts,
    option_or,
)
from modax._numba_common import PIDGains

if TYPE_CHECKING:
    from modax._sensitivity import SensitivitySpec
    from modax._sparsity import Pattern

# One warp per block: the trajectories in a block step adaptively and so
# diverge, and a warp is the granularity at which that divergence is free.
DEFAULT_TRAJECTORIES_PER_BLOCK = 32


@dataclass(frozen=True)
class KernelOptions:
    """The method and the launch options of one solve.

    It is hashable and every kernel builder caches on it, so two solves with
    equal options and the same callback share one kernel. :func:`modax.solve`
    builds it once. The sensitivity rule swaps in ``spec`` and the launch
    swaps the user's quantities for the kernel's hook, and nothing else
    changes it.

    Attributes:
        method: The solve's method. A hierarchy's blocks that name no method
            of their own step with it.
        trajectories_per_block: Threads per CUDA block, one trajectory each.
        sparsity: The Jacobian's normalised pattern, or ``None``.
        max_registers: The kernel's cap on registers per thread.
        save_hook: The user's quantities ``(y, t, p) -> tuple`` before the
            launch, and the kernel's hook ``hook(save_idx, y, t, p_row, acc)``
            after (:func:`~modax._sensitivity.quantity_hook`).
        hook_size: The width of each trajectory's hook output row.
        save_history: ``False`` keeps the final state in place of the history.
        split_max_driver: The most states a split system's driver holds; a
            hierarchy's upstream block that small factorises densely.
        cache: Keep the compiled kernel on disk (:mod:`modax._kernel_cache`).
        spec: The forward-sensitivity system to integrate, or ``None``.
    """

    method: Method = Rodas5P()
    trajectories_per_block: int = DEFAULT_TRAJECTORIES_PER_BLOCK
    sparsity: Pattern | None = None
    max_registers: int | None = None
    save_hook: Any = None
    hook_size: int = 0
    save_history: bool = True
    split_max_driver: int = 16
    cache: bool = False
    spec: SensitivitySpec | None = None

    def __post_init__(self) -> None:
        if self.trajectories_per_block < 1:
            raise ValueError(
                "trajectories_per_block must be positive, "
                f"got {self.trajectories_per_block}"
            )

    # The solve-level defaults a hierarchy's blocks fall back to. An Auto
    # contributes its stiff method's options.
    @property
    def gains(self) -> PIDGains:
        rodas, tsit = method_parts(self.method)
        return option_or(rodas, "gains", option_or(tsit, "gains", PIDGains()))

    @property
    def lu_precision(self) -> LUPrecision:
        return option_or(method_parts(self.method)[0], "lu_precision", "fp32")

    @property
    def ordering(self) -> Ordering:
        return option_or(method_parts(self.method)[0], "ordering", "amd")
