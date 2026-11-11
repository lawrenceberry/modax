"""GPU-accelerated ODE solvers for massive ensembles of trajectories."""

import jax

# Before anything below traces a JAX function: the solvers work in float64.
jax.config.update("jax_enable_x64", True)

from modax._hierarchy import Block, Hierarchy  # noqa: E402
from modax._methods import Auto, Rodas5P, Tsit5  # noqa: E402
from modax._numba_common import PIDGains  # noqa: E402
from modax._solve import solve  # noqa: E402

__all__ = ["Auto", "Block", "Hierarchy", "PIDGains", "Rodas5P", "Tsit5", "solve"]
