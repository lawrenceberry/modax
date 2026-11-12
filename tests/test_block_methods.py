"""Blocks of a hierarchy stepping with their own methods: Rodas5P, Tsit5 and the stiffness switch."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import modax
from modax import Auto, Block, Hierarchy, Rodas5P, Tsit5
from modax.rodas5P import solve as rodas_solve
from modax.tsit5 import solve as tsit_solve

jax.config.update("jax_enable_x64", True)
cuda = pytest.importorskip("numba_cuda_mlir.cuda")

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

T_SPAN = np.linspace(0.0, 3.0, 7)
TOL = dict(rtol=1e-9, atol=1e-11, lu_precision="fp64")  # the per-method solvers'
# modax.solve takes the LU's precision on its method
SOLVE_TOL = dict(rtol=1e-9, atol=1e-11, method=Rodas5P(lu_precision="fp64"))


def rhs_a(y, t, p):
    """A forced decay: not stiff."""
    return (-p[0] * y[0] + math.cos(t),)


def out_a(y, t, p):
    return (math.sin(y[0]), y[0] * t)


def rhs_b(y, t, p, q):
    """A stiff follower of the first quantity, forced by the second."""
    return (-p[1] * (y[0] - q[0]) + q[1],)


def rhs_c(y, t, p, q):
    """Reads the follower's state and the root's quantities: a diamond."""
    return (-p[2] * (y[0] - q[0]) + q[1],)


def joint(y, t, p):
    return (
        -p[0] * y[0] + math.cos(t),
        -p[1] * (y[1] - math.sin(y[0])) + y[0] * t,
        -p[2] * (y[2] - y[1]) + math.sin(y[0]),
    )


Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])


def chain(ma=None, mb=None, mc=None):
    a = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a", method=ma)
    b = Block(rhs_b, 1, reads=(a,), name="b", method=mb)
    c = Block(rhs_c, 1, reads=(b, a), name="c", method=mc)
    return Hierarchy(c)


@pytest.fixture(scope="module")
def whole():
    return np.asarray(rodas_solve(joint, Y0, T_SPAN, P, **TOL))


@pytest.mark.parametrize(
    "methods",
    [
        (Tsit5(), None, None),  # an explicit upstream block under implicit readers
        (None, None, Tsit5()),  # an explicit final block reading implicit ones
        (Tsit5(), Tsit5(), Tsit5()),
        (Auto(), Auto(), Auto()),
        (Tsit5(), Auto(), Rodas5P(lu_precision="fp64")),
    ],
    ids=["tsit-root", "tsit-sink", "all-tsit", "all-auto", "mixed"],
)
def test_mixed_methods_match_the_joint_solve(methods, whole):
    split = np.asarray(modax.solve(chain(*methods), Y0, T_SPAN, P, **SOLVE_TOL))
    assert np.abs(split - whole).max() < 1e-7


def test_modax_solve_takes_the_default_method(whole):
    split = np.asarray(
        modax.solve(chain(), Y0, T_SPAN, P, method=Tsit5(), rtol=1e-9, atol=1e-11)
    )
    assert np.abs(split - whole).max() < 1e-7


def test_the_plain_solvers_refuse_blocks():
    for plain in (rodas_solve, tsit_solve):
        with pytest.raises(TypeError, match="modax.solve"):
            plain(chain(), Y0, T_SPAN, P, rtol=1e-9, atol=1e-11)
        with pytest.raises(TypeError, match="modax.solve"):
            plain(Block(rhs_a, 1), Y0[:, :1], T_SPAN, P)


def stiffening(y, t, p):
    """``y' = -lambda(t) (y - cos t) - sin t``: relaxation onto cos t at a rate growing from 1 to 1e5."""
    lam = math.exp(p[0] * t)
    return (-lam * (y[0] - math.cos(t)) - math.sin(t),)


STIFF_T = np.linspace(0.0, 4.0, 9)
STIFF_P = jnp.array([[math.log(1.0e5) / 4.0]] * 32)
STIFF_Y0 = jnp.array([[1.0]] * 32)


def test_auto_switches_to_rodas_where_the_problem_turns_stiff():
    exact = np.cos(STIFF_T)
    kw = dict(rtol=1e-6, atol=1e-9, return_stats=True)
    auto, auto_stats = modax.solve(
        stiffening, STIFF_Y0, STIFF_T, STIFF_P, method=Auto(), **kw
    )
    explicit, explicit_stats = modax.solve(
        Block(stiffening, 1), STIFF_Y0, STIFF_T, STIFF_P, method=Tsit5(), **kw
    )
    implicit, implicit_stats = modax.solve(
        Block(stiffening, 1), STIFF_Y0, STIFF_T, STIFF_P, method=Rodas5P(), **kw
    )
    for hist in (auto, explicit, implicit):
        assert np.abs(np.asarray(hist)[:, :, 0] - exact).max() < 1e-4
    steps = {
        name: int(np.asarray(stats["accepted_steps"]).max())
        for name, stats in (
            ("auto", auto_stats),
            ("tsit5", explicit_stats),
            ("rodas", implicit_stats),
        )
    }
    # Tsit5 alone is held to h |lambda| < 3.5 once lambda grows; Auto is not
    assert steps["auto"] < steps["tsit5"] / 5, steps


def stiff_quantity(y, t, p):
    return (y[0],)


def follower(y, t, p, q):
    """A slow reader of the stiffening block's state."""
    return (-(y[0] - q[0]),)


def test_an_upstream_auto_block_switches_where_its_problem_turns_stiff():
    """An upstream block starting explicit counts its rejected attempts too and gets out of Tsit5.

    Rejected attempts count towards the switch (OrdinaryDiffEq.jl's
    ``choose_algorithm!`` runs in every step's footer), and a lane that comes
    to want Rodas5P inside the retry loop leaves it so its warp can vote again,
    rather than rejecting at Tsit5's stability limit until ``max_steps``.
    """
    root = Block(
        stiffening, 1, outputs=stiff_quantity, n_outputs=1, name="root", method=Auto()
    )
    sink = Block(follower, 1, reads=(root,), name="sink")
    y0 = jnp.array([[1.0, 1.0]] * 32)
    hist, stats = modax.solve(
        Hierarchy(sink, layout=(root, sink)),
        y0,
        STIFF_T,
        STIFF_P,
        rtol=1e-6,
        atol=1e-9,
        max_steps=20000,
        return_stats=True,
    )
    hist = np.asarray(hist)
    assert np.abs(hist[:, :, 0] - np.cos(STIFF_T)).max() < 1e-4
    assert np.isfinite(hist).all()
    assert int(np.asarray(stats["loop_steps"]).max()) < 2000, stats


def test_explicit_blocks_refuse_sensitivities():
    def at(p):
        return modax.solve(chain(Tsit5()), Y0, T_SPAN, p, **SOLVE_TOL)

    with pytest.raises(NotImplementedError, match="explicit"):
        jax.jvp(at, (P,), (jnp.ones_like(P),))


def test_an_upstream_block_factorises_in_double():
    with pytest.raises(ValueError, match="final block only"):
        modax.solve(chain(Rodas5P(lu_precision="fp32")), Y0, T_SPAN, P, **SOLVE_TOL)


def test_the_solves_precision_skips_upstream_blocks(whole):
    """A solve-level fp32 LU reaches the final block only."""
    method = Rodas5P(lu_precision="fp32")
    split = np.asarray(
        modax.solve(chain(), Y0, T_SPAN, P, method=method, rtol=1e-9, atol=1e-11)
    )
    assert np.abs(split - whole).max() < 1e-5


def test_a_method_is_checked():
    with pytest.raises(TypeError):
        Block(rhs_a, 1, method="tsit5")
    with pytest.raises(TypeError):
        Auto(Rodas5P(), Tsit5())


def decay(y, t, p):
    return (-p[0] * y[0], -p[1] * y[1])


DECAY_Y0 = jnp.array([[1.0, 2.0]] * 32)
DECAY_P = jnp.array([[1.0, 3.0]] * 32)


def test_the_method_carries_its_own_options():
    """``modax.solve`` passes each method's options through to its solver."""
    pattern = np.eye(2, dtype=bool)
    rodas = {
        ordering: np.asarray(
            modax.solve(
                decay,
                DECAY_Y0,
                T_SPAN,
                DECAY_P,
                method=Rodas5P(lu_precision="fp64", ordering=ordering),
                sparsity=pattern,
                rtol=1e-9,
                atol=1e-11,
            )
        )
        for ordering in ("amd", "natural")
    }
    exact = np.exp(-np.outer(T_SPAN, [1.0, 3.0])) * [1.0, 2.0]
    for hist in rodas.values():
        assert np.abs(hist - exact).max() < 1e-7
    tsit = {
        backend: np.asarray(
            modax.solve(
                decay,
                DECAY_Y0,
                T_SPAN,
                DECAY_P,
                method=Tsit5(backend=backend),
                rtol=1e-9,
                atol=1e-11,
            )
        )
        for backend in ("shared", "local")
    }
    np.testing.assert_array_equal(tsit["shared"], tsit["local"])


def test_a_plain_tsit5_solve_ignores_sparsity():
    kw = dict(method=Tsit5(), rtol=1e-9, atol=1e-11)
    plain = np.asarray(modax.solve(decay, DECAY_Y0, T_SPAN, DECAY_P, **kw))
    with_pattern = modax.solve(
        decay, DECAY_Y0, T_SPAN, DECAY_P, sparsity=np.eye(2), **kw
    )
    np.testing.assert_array_equal(plain, np.asarray(with_pattern))


def test_a_plain_tsit5_solve_takes_a_save_hook_on_the_hierarchy_kernel():
    """A hook routes a plain Tsit5 solve to a one-block hierarchy, still Tsit5."""
    weights = cuda.to_device(np.full(T_SPAN.size, 1.0))

    def total(save_idx, y, t, p_row, acc):
        acc[0] += weights[save_idx] * y[0]

    kw = dict(method=Tsit5(), rtol=1e-9, atol=1e-11)
    hist = np.asarray(modax.solve(decay, DECAY_Y0, T_SPAN, DECAY_P, **kw))
    hooked, acc = modax.solve(
        decay, DECAY_Y0, T_SPAN, DECAY_P, save_hook=total, hook_size=1, **kw
    )
    assert np.abs(np.asarray(hooked) - hist).max() < 1e-8
    np.testing.assert_allclose(
        np.asarray(acc)[:, 0], hist[:, :, 0].sum(axis=1), rtol=1e-8
    )


def test_method_options_are_checked():
    with pytest.raises(ValueError, match="ordering"):
        Rodas5P(ordering="metis")
    with pytest.raises(ValueError, match="backend"):
        Tsit5(backend="global")


def test_a_repeated_plain_auto_solve_reuses_its_block():
    """The one-block wrapper is cached, so the hierarchy kernel is compiled once."""
    from modax._solve import _one_block

    kw = dict(method=Auto(), rtol=1e-6, atol=1e-9)
    modax.solve(decay, DECAY_Y0, T_SPAN, DECAY_P, **kw)
    before = _one_block.cache_info()
    modax.solve(decay, DECAY_Y0, T_SPAN, DECAY_P, **kw)
    after = _one_block.cache_info()
    assert after.hits == before.hits + 1 and after.misses == before.misses
