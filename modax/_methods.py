"""How a system or a block steps: the method objects :func:`modax.solve` and :class:`~modax.Block` take.

A method is a small frozen dataclass of the options that belong to one
solver's numerics (the LU's precision and fill-reducing ordering, the
controller's gains, Tsit5's memory backend). What applies to the problem or
the whole launch (tolerances, error weights, the sparsity pattern, the save
hook, sensitivities) is an argument of :func:`modax.solve`. Being hashable, a
method is part of the kernel's identity like everything else a block carries.

* :class:`Rodas5P` -- linearly implicit, for stiff blocks: a Jacobian and an
  LU every step.
* :class:`Tsit5` -- explicit, for non-stiff blocks: seven right-hand sides a
  step (the last reused as the next step's first), no Jacobian, no LU.
* :class:`Auto` -- one of each, switched by a stiffness estimate the way
  OrdinaryDiffEq.jl's ``AutoSwitch`` does (:class:`Auto` has the details).

In a hierarchy, ``None`` for a block's option means the value of the method
given to :func:`modax.solve`: its ``gains``, its ``ordering``, and its
``lu_precision`` on the final block (a block others read factorises in double
precision). Left ``None`` there too, the defaults are the integral controller,
AMD and single precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from modax._numba_common import PIDGains

LUPrecision: TypeAlias = Literal["fp32", "fp64"]
"""The precision of the per-step LU factorisation and triangular solves."""
Ordering: TypeAlias = Literal["amd", "natural"]
"""The sparse LU's fill-reducing permutation: AMD, or the pattern's own order."""
Backend: TypeAlias = Literal["auto", "shared", "local"]
"""Where the Tsit5 kernel keeps the state and its stage vectors."""


@dataclass(frozen=True)
class Rodas5P:
    """Rodas5P, the linearly implicit Rosenbrock-W method of order 5, for stiff systems.

    Every step builds the Jacobian by forward-mode AD of the right-hand side
    and factorises the iteration matrix in the thread's own memory.

    Args:
        lu_precision: ``"fp32"`` (the default) or ``"fp64"``, the precision
            the iteration matrix is factorised in. Single precision halves the
            LU's footprint; as a W-method Rodas5P stays convergent with an
            inexact matrix, and the step controller absorbs the difference.
            A block others read always factorises in double.
        gains: The step controller's :class:`PIDGains`. The default is the
            integral controller ``(0, 1, 0)``.
        ordering: The fill-reducing permutation of the sparse LU when
            :func:`modax.solve` has a ``sparsity`` pattern: ``"amd"`` (the
            default) or ``"natural"``, which keeps the pattern's own order.
    """

    lu_precision: LUPrecision | None = None
    gains: PIDGains | None = None
    ordering: Ordering | None = None

    def __post_init__(self) -> None:
        if self.lu_precision not in (None, "fp32", "fp64"):
            raise ValueError(
                f"lu_precision must be 'fp32' or 'fp64', got {self.lu_precision!r}"
            )
        if self.ordering not in (None, "amd", "natural"):
            raise ValueError(
                f"ordering must be 'amd' or 'natural', got {self.ordering!r}"
            )


@dataclass(frozen=True)
class Tsit5:
    """Tsitouras's explicit Runge--Kutta 5(4) pair, first-same-as-last, for non-stiff systems.

    Seven right-hand-side evaluations a step, the last reused as the next
    step's first, and no Jacobian.

    Args:
        gains: The step controller's :class:`PIDGains`. The default is the
            integral controller ``(0, 1, 0)``.
        backend: Where a plain system's state and stage vectors live:
            ``"shared"`` (per-block shared memory), ``"local"`` (the thread's
            own memory) or ``"auto"`` (the default: shared for up to 16
            components and 16384 trajectories). The two are bit-identical. A
            block of a hierarchy always uses thread-local memory.
    """

    gains: PIDGains | None = None
    backend: Backend = "auto"

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "shared", "local"):
            raise ValueError(
                f"backend must be 'auto', 'shared' or 'local', got {self.backend!r}"
            )


TSIT5_STABILITY_SIZE = 3.5068
"""How far along the negative real axis Tsit5's stability region reaches, ``|h lambda|``."""


@dataclass(frozen=True)
class Auto:
    """Explicit while the block is not stiff, implicit while it is.

    The switch is OrdinaryDiffEq.jl's ``AutoSwitch``. After every attempted
    step, accepted or rejected (as OrdinaryDiffEq.jl's ``choose_algorithm!``
    runs in each step's footer), the block estimates the largest eigenvalue magnitude of its
    Jacobian: from an explicit step at no cost, as ``max_j |k7_j - k6_j| /
    |y_new,j - g6_j|`` over the two stages Tsit5 evaluates at the step's end
    (Hairer and Wanner II, p. 22, in the infinity norm); from an implicit step
    as ``||J||_inf``, the largest absolute row sum of the Jacobian it has
    anyway. The step is *stiff* when ``estimate * h_next`` exceeds
    ``stifftol`` (``nonstifftol`` while explicit) times Tsit5's stability
    size, 3.5068: an explicit step that long would be unstable. More than
    ``maxstiffstep`` stiff steps in a row switch to ``stiff``, with the step
    multiplied by ``dtfac``; more than ``maxnonstiffstep`` non-stiff ones in a
    row switch back, with it divided.

    On the GPU the choice is the warp's, not the lane's: a lane only votes,
    and the warp steps implicitly when any of its active lanes wants to, so
    the lanes of a warp never run both methods one after the other. The
    implicit method is always a valid step for a lane that would rather be
    explicit; the converse is not true. ``stiff_first`` starts implicit.
    """

    nonstiff: Tsit5 = Tsit5()
    stiff: Rodas5P = Rodas5P()
    maxstiffstep: int = 10
    maxnonstiffstep: int = 3
    nonstifftol: float = 0.9
    stifftol: float = 0.9
    dtfac: float = 2.0
    stiff_first: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.nonstiff, Tsit5):
            raise TypeError(
                f"Auto's non-stiff method must be Tsit5, got {type(self.nonstiff).__name__}"
            )
        if not isinstance(self.stiff, Rodas5P):
            raise TypeError(
                f"Auto's stiff method must be Rodas5P, got {type(self.stiff).__name__}"
            )
        if self.maxstiffstep < 0 or self.maxnonstiffstep < 0 or self.dtfac <= 0.0:
            raise ValueError("Auto needs non-negative step counts and a positive dtfac")


Method = Rodas5P | Tsit5 | Auto


def method_parts(method: Method) -> tuple[Rodas5P | None, Tsit5 | None]:
    """``method``'s implicit and explicit methods, ``None`` where it has none."""
    if isinstance(method, Auto):
        return method.stiff, method.nonstiff
    return (
        method if isinstance(method, Rodas5P) else None,
        method if isinstance(method, Tsit5) else None,
    )


def option_or(method: Rodas5P | Tsit5 | None, name: str, default):
    """``method``'s option ``name``, or ``default`` where there is no method or it leaves the option ``None``."""
    value = getattr(method, name) if method is not None else None
    return default if value is None else value
