"""The method objects :func:`modax.solve` and :class:`~modax.Block` take.

A method is a small frozen dataclass holding one solver's numerical options:
the LU's precision and fill-reducing ordering, the controller's gains, and
Tsit5's memory backend. Options that belong to the problem or the launch
(tolerances, error weights, the sparsity pattern, the save hook,
sensitivities) are arguments of :func:`modax.solve`. A method is hashable, so
it forms part of the kernel's identity.

* :class:`Rodas5P`: linearly implicit, for stiff blocks. A Jacobian and an LU
  every step.
* :class:`Tsit5`: explicit, for non-stiff blocks. Seven right-hand sides a
  step (the last reused as the next step's first), no Jacobian, no LU.
* :class:`Auto`: one of each, switched by a stiffness estimate like
  OrdinaryDiffEq.jl's ``AutoSwitch``.

In a hierarchy, a block option left ``None`` takes the value from the method
passed to :func:`modax.solve`: its ``gains``, its ``ordering``, and its
``lu_precision`` on the final block (a block others read factorises in double
precision). If that is ``None`` too, the defaults are the integral
controller, AMD and single precision.
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

    Every step builds the Jacobian by forward-mode AD of the right-hand side and
    factorises the iteration matrix in the thread's own memory.

    Args:
        lu_precision: ``"fp32"`` (the default) or ``"fp64"``, the precision of
            the iteration matrix's factorisation. Single precision halves the
            LU's footprint. As a W-method, Rodas5P stays convergent with an
            inexact matrix and the step controller absorbs the difference. A
            block others read always factorises in double.
        gains: The step controller's :class:`PIDGains`. The default is the
            integral controller ``(0, 1, 0)``.
        ordering: The sparse LU's fill-reducing permutation when
            :func:`modax.solve` has a ``sparsity`` pattern. ``"amd"`` (the
            default) or ``"natural"``, which keeps the pattern's own order.

    Raises:
        ValueError: If ``lu_precision`` or ``ordering`` is not one of the values
            above.
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

    Seven right-hand-side evaluations a step, the last reused as the next step's
    first, and no Jacobian.

    Args:
        gains: The step controller's :class:`PIDGains`. The default is the
            integral controller ``(0, 1, 0)``.
        backend: Where a plain system's state and stage vectors live.
            ``"shared"`` uses per-block shared memory, ``"local"`` the thread's
            own memory, and ``"auto"`` (the default) picks shared for up to 16
            components and 16384 trajectories. The two give bit-identical
            results. A block of a hierarchy always uses thread-local memory.

    Raises:
        ValueError: If ``backend`` is not one of the values above.
    """

    gains: PIDGains | None = None
    backend: Backend = "auto"

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "shared", "local"):
            raise ValueError(
                f"backend must be 'auto', 'shared' or 'local', got {self.backend!r}"
            )


TSIT5_STABILITY_SIZE = 3.5068
"""How far Tsit5's stability region reaches along the negative real axis, ``|h lambda|``."""


@dataclass(frozen=True)
class Auto:
    """Explicit while the block is not stiff, implicit while it is.

    The switch follows OrdinaryDiffEq.jl's ``AutoSwitch``. After every attempted
    step, accepted or rejected, the block estimates the largest eigenvalue
    magnitude of its Jacobian. An explicit step gets it for free as ``max_j
    |k7_j - k6_j| / |y_new,j - g6_j|`` over the two stages Tsit5 evaluates at
    the step's end (Hairer and Wanner II, p. 22, in the infinity norm). An
    implicit step uses ``||J||_inf``, the largest absolute row sum of the
    Jacobian it already has. A step counts as *stiff* when ``estimate * h_next``
    exceeds ``stifftol`` (``nonstifftol`` while explicit) times Tsit5's
    stability size, 3.5068, because an explicit step that long would be
    unstable.

    On the GPU the warp chooses, not the lane. Each lane votes, and the warp
    steps implicitly when any active lane wants to, so a warp never runs both
    methods back to back. The implicit method is always a valid step for a lane
    that would rather be explicit, but not the other way round.

    Args:
        nonstiff: The explicit method, a :class:`Tsit5`.
        stiff: The implicit method, a :class:`Rodas5P`.
        maxstiffstep: Switch to ``stiff`` after more than this many stiff steps
            in a row, multiplying the step by ``dtfac``. Defaults to 10.
        maxnonstiffstep: Switch back to ``nonstiff`` after more than this many
            non-stiff steps in a row, dividing the step by ``dtfac``. Defaults
            to 3.
        nonstifftol: The stiffness threshold while explicit. Defaults to 0.9.
        stifftol: The stiffness threshold while implicit. Defaults to 0.9.
        dtfac: The step factor applied on a switch. Defaults to 2.0.
        stiff_first: Start implicit. Defaults to ``False``.

    Raises:
        TypeError: If ``nonstiff`` is not a :class:`Tsit5` or ``stiff`` not a
            :class:`Rodas5P`.
        ValueError: If a step count is negative or ``dtfac`` is not positive.
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
    """``method``'s option ``name``, or ``default`` if there is no method or the option is ``None``."""
    value = getattr(method, name) if method is not None else None
    return default if value is None else value
