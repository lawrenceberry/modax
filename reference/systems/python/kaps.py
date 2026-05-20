"""Coupled Kaps singular-perturbation systems.

Kaps' problem — controllable stiffness with a known exact solution

Each equation pair (y1_i, y2_i) is an independent copy of the original Kaps system
with a different stiffness parameter epsilon_i. The stiffness ratio of pair i is
approximately 1 / epsilon_i, so epsilon_min controls the maximum stiffness across
the system. The initial conditions lie exactly on the slow manifold, so there is
no fast initial transient, only the stable fast modes that implicit solvers must
handle without being forced into tiny steps.

The analytical solution is independent of epsilon_i for all pairs:
    y1_i(t) = exp(-2t),  y2_i(t) = exp(-t)

This allows precise validation at arbitrary stiffness without a reference solver.
"""

import jax.numpy as jnp
import numpy as np

from reference.systems.python._tuple_codegen import make_matrix_callback, make_tuple_callback

TIMES = jnp.array((0.0, 0.5, 1.0, 2.0), dtype=jnp.float64)


def make_system(n_pairs, epsilon_min):
    """Construct n_pairs independent Kaps singular-perturbation equation pairs."""
    n_vars = 2 * n_pairs
    epsilon = jnp.array(
        [10.0 ** (np.log10(epsilon_min) * i / (n_pairs - 1)) for i in range(n_pairs)],
        dtype=jnp.float64,
    )
    y0 = jnp.array([1.0, 1.0] * n_pairs, dtype=jnp.float64)

    ode_values = []
    explicit_values = []
    implicit_values = []
    jac_rows = [["0.0" for _ in range(n_vars)] for _ in range(n_vars)]
    implicit_jac_rows = [["0.0" for _ in range(n_vars)] for _ in range(n_vars)]
    for pair, eps in enumerate(np.asarray(epsilon)):
        base = 2 * pair
        inv_eps = f"{1.0 / float(eps):.17g}"
        ode_values.extend(
            [
                f"p[0] * (-({inv_eps} + 2.0) * y[{base}] + {inv_eps} * y[{base + 1}] * y[{base + 1}])",
                f"p[0] * (y[{base}] - y[{base + 1}] - y[{base + 1}] * y[{base + 1}])",
            ]
        )
        explicit_values.extend(
            [
                f"p[0] * (-2.0 * y[{base}])",
                f"p[0] * (y[{base}] - y[{base + 1}] - y[{base + 1}] * y[{base + 1}])",
            ]
        )
        implicit_values.extend(
            [
                f"p[0] * (-{inv_eps} * (y[{base}] - y[{base + 1}] * y[{base + 1}]))",
                "0.0",
            ]
        )
        jac_rows[base][base] = f"p[0] * (-({inv_eps} + 2.0))"
        jac_rows[base][base + 1] = f"p[0] * (2.0 * {inv_eps} * y[{base + 1}])"
        jac_rows[base + 1][base] = "p[0]"
        jac_rows[base + 1][base + 1] = f"p[0] * (-1.0 - 2.0 * y[{base + 1}])"
        implicit_jac_rows[base][base] = f"-{inv_eps} * p[0]"
        implicit_jac_rows[base][base + 1] = f"2.0 * {inv_eps} * p[0] * y[{base + 1}]"

    ode_fn = make_tuple_callback("ode_fn", [], ode_values)
    explicit_ode_fn = make_tuple_callback("explicit_ode_fn", [], explicit_values)
    implicit_ode_fn = make_tuple_callback("implicit_ode_fn", [], implicit_values)
    jac_fn = make_matrix_callback("jac_fn", [], jac_rows)
    implicit_jac_fn = make_matrix_callback("implicit_jac_fn", [], implicit_jac_rows)

    return {
        "n_pairs": n_pairs,
        "epsilon_min": epsilon_min,
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "jac_fn": jac_fn,
        "implicit_jac_fn": implicit_jac_fn,
        "y0": y0,
    }


_DEFAULT = make_system(2, 1e-2)
ode_fn = _DEFAULT["ode_fn"]
explicit_ode_fn = _DEFAULT["explicit_ode_fn"]
implicit_ode_fn = _DEFAULT["implicit_ode_fn"]
jac_fn = _DEFAULT["jac_fn"]
implicit_jac_fn = _DEFAULT["implicit_jac_fn"]


def make_params(size, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def exact_solution(t_span, params, n_pairs):
    """Exact solution for n_pairs Kaps equation pairs."""
    t = np.asarray(t_span, dtype=np.float64)
    s = np.asarray(params)[:, 0]
    y1 = np.exp(-2.0 * np.outer(s, t))
    y2 = np.exp(-1.0 * np.outer(s, t))
    pair = np.stack([y1, y2], axis=-1)
    return np.tile(pair, (1, 1, n_pairs))
