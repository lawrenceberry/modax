"""How a block of a hierarchy steps: the method objects a :class:`~modax.Block` takes.

A method is a small frozen dataclass of the options that belong to one block's
solver; what applies to the whole launch (tolerances, error weights, the save
hook, sensitivities) stays on :func:`modax.solve`. Being hashable, a method is
part of the kernel's identity like everything else a block carries.

* :class:`Rodas5P` -- linearly implicit, for stiff blocks: a Jacobian and an
  LU every step.
* :class:`Tsit5` -- explicit, for non-stiff blocks: seven right-hand sides a
  step (the last reused as the next step's first), no Jacobian, no LU.
* :class:`Auto` -- one of each, switched by a stiffness estimate the way
  OrdinaryDiffEq.jl's ``AutoSwitch`` does (:class:`Auto` has the details).

``None`` for an option means the solve's own value: ``gains=None`` the solve's
``pcoeff``/``icoeff``/``dcoeff``, ``lu_precision=None`` the solve's
``lu_precision`` on the final block and double precision upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modax._numba_common import PIDGains


@dataclass(frozen=True)
class Rodas5P:
    """Rodas5P, the linearly implicit Rosenbrock method of order 5."""

    lu_precision: Literal["fp32", "fp64"] | None = None
    gains: PIDGains | None = None

    def __post_init__(self) -> None:
        if self.lu_precision not in (None, "fp32", "fp64"):
            raise ValueError(
                f"lu_precision must be 'fp32' or 'fp64', got {self.lu_precision!r}"
            )


@dataclass(frozen=True)
class Tsit5:
    """Tsitouras's explicit Runge--Kutta 5(4) pair, first-same-as-last."""

    gains: PIDGains | None = None


TSIT5_STABILITY_SIZE = 3.5068
"""How far along the negative real axis Tsit5's stability region reaches, ``|h lambda|``."""


@dataclass(frozen=True)
class Auto:
    """Explicit while the block is not stiff, implicit while it is.

    The switch is OrdinaryDiffEq.jl's ``AutoSwitch``. After every accepted
    step the block estimates the largest eigenvalue magnitude of its
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
