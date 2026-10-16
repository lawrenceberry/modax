"""Bayesian parameter estimation from primordial light-element abundances.

A four-species Big Bang Nucleosynthesis network (n, p, D, 4He) is integrated
with the Rodas5P kernel solver, with x = Q/T as the independent variable, and
the baryon-to-photon ratio log10(eta_10) and N_eff are fitted to the observed
abundances by nested slice sampling (BlackJAX). Every likelihood
evaluation is an independent stiff universe, so the sampler's population is
one batched ensemble solve. The physics, the statistical model and the
benchmark are laid out in README.md next to this file.

Usage:
    uv run python examples/bbn_estimation/main.py [--benchmark]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmark_common import (
    MODAX_COLOR,
    configure_latex_plot_style,
    print_plot_title,
)
from examples._common import Backends, parse_args, run_benchmark
from examples.dual_backend import build_fn, build_rhs
from modax.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants (natural units, MeV)
# ---------------------------------------------------------------------------

Q = 1.293  # MeV, n-p mass difference
B_D = 2.224573  # MeV, deuterium binding energy
M_N = 938.272  # MeV, nucleon mass
M_E = 0.510999  # MeV, electron mass
M_PL = 1.2209e22  # MeV, unreduced Planck mass
ZETA3 = 1.2020569  # Riemann zeta(3)
N_EFF_SM = 3.044  # Standard-model N_eff
HBAR = 6.582119e-22  # MeV s
HBARC = 1.973270e-11  # MeV cm
N_AVOGADRO = 6.02214076e23
T9_MEV = 0.08617333  # MeV per 10^9 K

# Natural-unit neutron lifetime: tau_n [s] -> tau_n / hbar [MeV^-1]
TAU_N_MEV = 879.4 / HBAR

# Thermonuclear rates come as N_A <sigma v> in cm^3 mol^-1 s^-1; this turns
# them into <sigma v> in MeV^-2.
CM3_PER_MOL_S = HBAR / HBARC**3 / N_AVOGADRO

# Entropy degrees of freedom of the electromagnetic plasma while the e+e-
# pairs are relativistic: 2 (photons) + 7/8 * 4 (e+ and e-, two spins each).
G_S_HIGH = 5.5

# ---------------------------------------------------------------------------
# Thermal history: e+e- thermodynamics by quadrature inside the kernel
# ---------------------------------------------------------------------------
#
# Through e+e- annihilation (T ~ m_e) the plasma's energy and entropy degrees
# of freedom fall from 10.75 to 3.38 and the photons are heated relative to
# the decoupled neutrinos.  The Fermi-Dirac integrals that describe this have
# no closed form, polynomial fits in ln(m_e/T) converge too slowly to be worth
# having (1e-3 at degree 24: the integrands' poles at E = i pi sit next to
# the annihilation region), and a lookup table is not an option in the device
# callback because Enzyme cannot differentiate through a constant-memory
# array.  So the right-hand side evaluates the integrals itself, by 16-point
# Gauss-Laguerre quadrature over the dimensionless momentum u = p/T, with the
# nodes and weights (the weights carrying the e^u that undoes the Laguerre
# weight) written out as literals: a runtime index into a tuple of constants
# emits a bounds check Enzyme will not differentiate either.  Sixteen points
# reproduce the integrals to 1e-5 of their massless value across the range,
# which moves Y_P by under 1e-6.  Each point costs one sqrt and one exp in
# fp64, which a GeForce GPU executes at a 64th of its fp32 rate; this
# quadrature is about half the kernel's cost.

X_SPAN = jnp.array([Q / 10.0, Q / 0.005])  # T: 10 MeV -> 5 keV
X_SAVE = X_SPAN  # save at start and end only

# The photons' rho/T^4 and p/T^4: g = 2 bosons.
RHO_PHOTON = math.pi**2 / 15.0
P_PHOTON = math.pi**2 / 45.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
#
# The solvers want the network in two forms: ``rodas5P`` compiles its
# right-hand side with ``numba_cuda_mlir`` and differentiates it with Enzyme,
# while the Diffrax reference backend traces ``jnp`` arrays.  Each piece below
# is therefore written once, as a factory over the names the two spell
# differently, and built in both forms by ``examples/dual_backend.py``.  The
# ``.device`` member of each is a CUDA device function; ``.jax`` is the traced
# one; ``.host`` is the device arithmetic in plain Python, which is what
# ``tests/test_examples.py`` compares against ``.jax`` without needing a GPU.


def _make_hubble(*, sqrt):
    def hubble(T, N_eff, g_rho_em, g_s_em):
        """H(T) in MeV (Friedmann, radiation domination).

        The plasma contributes ``g_rho_em``; the neutrinos, decoupled with
        T_nu = T while the e+e- pairs were relativistic, are colder by the
        entropy the pairs have since released into the photons:
        T_nu / T = (g_s_em / 5.5)^(1/3), which tends to (4/11)^(1/3).
        """
        t_nu_over_t = (g_s_em / G_S_HIGH) ** (1.0 / 3.0)
        g_rho = g_rho_em + 1.75 * N_eff * t_nu_over_t**4
        return sqrt(4.0 * math.pi**3 * g_rho / 45.0) * T**2 / M_PL

    return hubble


def _make_n_baryon():
    def n_baryon(T, eta, g_s_em):
        """Baryon number density in MeV^3.

        ``eta`` is the baryon-to-photon ratio today. Baryons dilute as a^-3
        while the photons are heated by annihilation, so at temperature T
        n_b = eta n_gamma(T) g_s_em(T) / 2, which is 11/4 eta n_gamma before
        annihilation and eta n_gamma after it.
        """
        return eta * (2.0 * ZETA3 / math.pi**2) * T**3 * g_s_em / 2.0

    return n_baryon


# n <-> p weak rates: the fits of Serpico et al. (2004), Appendix C, to the
# Born rates with radiative and finite-mass corrections, as functions of
# z = m_e / T and normalised to the measured neutron lifetime. The p -> n fit
# is quoted for T >= 0.1 MeV and set to zero below; it is continued here
# instead, where its exp(-2.86 z) tail is below 1e-7 of the decay rate and
# leaves the right-hand side smooth.
WEAK_NP_COEFFS = (
    1.0, 0.15735, 4.6172, -0.40520e2, 0.13875e3, -0.59898e2, 0.66752e2,
    -0.16705e2, 3.8071, -0.39140, 0.023590, -0.83696e-4, -0.42095e-4, 0.17675e-5,
)  # fmt: skip
WEAK_PN_COEFFS = (
    0.22211e2, -0.72798e2, 0.11571e3, -0.11763e2, 0.45521e2, -3.7973, 0.41266,
    -0.026210, 0.87934e-3, -0.12016e-4,
)  # fmt: skip
Q_NP = 0.33979
Q_PN = 2.8602


def _make_weak_rate_np(*, exp):
    def weak_rate_np(T):
        """Total n -> p rate [MeV], including free neutron decay."""
        w = T / M_E  # 1 / z
        a = WEAK_NP_COEFFS
        poly = a[13]
        poly = a[12] + w * poly
        poly = a[11] + w * poly
        poly = a[10] + w * poly
        poly = a[9] + w * poly
        poly = a[8] + w * poly
        poly = a[7] + w * poly
        poly = a[6] + w * poly
        poly = a[5] + w * poly
        poly = a[4] + w * poly
        poly = a[3] + w * poly
        poly = a[2] + w * poly
        poly = a[1] + w * poly
        poly = a[0] + w * poly
        return exp(-Q_NP * w) * poly / TAU_N_MEV

    return weak_rate_np


def _make_weak_rate_pn(*, exp):
    def weak_rate_pn(T):
        """Total p -> n rate [MeV]."""
        w = T / M_E  # 1 / z
        b = WEAK_PN_COEFFS
        poly = b[9]
        poly = b[8] + w * poly
        poly = b[7] + w * poly
        poly = b[6] + w * poly
        poly = b[5] + w * poly
        poly = b[4] + w * poly
        poly = b[3] + w * poly
        poly = b[2] + w * poly
        poly = b[1] + w * poly
        poly = b[0] + w * poly
        return exp(-Q_PN / w) * w * poly / TAU_N_MEV

    return weak_rate_pn


# Thermonuclear rates: the T9 fits of Serpico et al. (2004), Appendix E, to
# the measured S-factors, each used up to the T9 at which its authors (and
# LINX's PArthENoPE network) stop trusting it and held constant above.  Above
# those temperatures deuterium is in Saha equilibrium whatever the rate.
T9_MAX_PN = 1.5
T9_MAX_DP = 4.0
T9_MAX_DD = 10.0


def _make_sigma_v_pn(*, sqrt, minimum):
    def sigma_v_pn(T):
        """<sigma v> for p + n -> D + gamma [MeV^-2]."""
        T9 = minimum(T / T9_MEV, T9_MAX_PN)
        s = sqrt(T9)
        poly = -0.209351
        poly = 1.34596 + s * poly
        poly = -3.52204 + s * poly
        poly = 4.62949 + s * poly
        poly = -2.75037 + s * poly
        poly = 0.106597 + s * poly
        poly = 1.0 + s * poly
        return 44060.0 * poly * CM3_PER_MOL_S

    return sigma_v_pn


def _make_sigma_v_dp(*, exp, minimum):
    def sigma_v_dp(T):
        """<sigma v> for D + p -> 3He + gamma [MeV^-2]."""
        T9 = minimum(T / T9_MEV, T9_MAX_DP)
        c = T9 ** (1.0 / 3.0)
        poly = -77.3717
        poly = 100.688 + c * poly
        poly = 127.838 + c * poly
        poly = 173.239 + c * poly
        poly = -369.613 + c * poly
        poly = 2120.73 + c * poly
        poly = -721.914 + c * poly
        poly = -206.509 + c * poly
        poly = 126.821 + c * poly
        poly = -15.7097 + c * poly
        return exp(-1.29043 / c) * poly / (c * c) * CM3_PER_MOL_S

    return sigma_v_dp


def _make_sigma_v_dd(*, exp, minimum):
    def sigma_v_dd(T):
        """<sigma v> for D + D -> n + 3He plus D + D -> p + T [MeV^-2]."""
        T9 = minimum(T / T9_MEV, T9_MAX_DD)
        c = T9 ** (1.0 / 3.0)
        ddn = -252211.0
        ddn = 911550.0 + c * ddn
        ddn = -294342.0 + c * ddn
        ddn = -2.31204e6 + c * ddn
        ddn = 2.2519e6 + c * ddn
        ddn = 8.52947e6 + c * ddn
        ddn = -2.72404e7 + c * ddn
        ddn = -1.24333e7 + c * ddn
        ddn = 1.3329e8 + c * ddn
        ddn = -6.11628e7 + c * ddn
        ddn = -1.3761e7 + c * ddn
        ddn = 1.22986e7 + c * ddn
        ddn = -1.84664e6 + c * ddn
        ddp = -9.85542e6
        ddp = 5.28874e7 + c * ddp
        ddp = -1.18812e8 + c * ddp
        ddp = 2.32242e8 + c * ddp
        ddp = -1.70199e8 + c * ddp
        ddp = 5.23171e7 + c * ddp
        ddp = -5.85032e6 + c * ddp
        total = exp(-1.0 / c) * ddn + exp(-1.06765 / c) * ddp
        return total / (c * c) * CM3_PER_MOL_S

    return sigma_v_dd


def _make_deuterium_eq_inverse(*, exp):
    def deuterium_eq_inverse(T, n_b):
        """1 / K_D with K_D = Y_d^eq / (Y_n Y_p): Saha equation for D.

        Written as the inverse so it underflows to zero, rather than
        overflowing, once photodissociation has switched off.
        """
        return exp(-B_D / T) / (n_b * 0.75 * (4.0 * math.pi / (M_N * T)) ** 1.5)

    return deuterium_eq_inverse


HUBBLE = build_fn(_make_hubble)
N_BARYON = build_fn(_make_n_baryon)
WEAK_RATE_NP = build_fn(_make_weak_rate_np)
WEAK_RATE_PN = build_fn(_make_weak_rate_pn)
SIGMA_V_PN = build_fn(_make_sigma_v_pn)
SIGMA_V_DP = build_fn(_make_sigma_v_dp)
SIGMA_V_DD = build_fn(_make_sigma_v_dd)
DEUTERIUM_EQ_INVERSE = build_fn(_make_deuterium_eq_inverse)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

# Independent variable: x = Q/T (dimensionless, runs from ~0.13 to ~259).
# Entropy conservation gives dx/dt = H x / (1 + (1/3) d ln g_s / d ln T), so
# the equation of motion is dY/dx = rates * (1 + (1/3) d ln g_s / d ln T) / (H x).
# State: Y = [Y_n, Y_p, Y_d, Y_4He] (nucleon number fractions).
# Conservation: Y_n + Y_p + 2*Y_d + 4*Y_4He = 1.
# Parameters: [log10(eta_10), N_eff] where eta_10 = eta * 1e10.
#
# Reactions: the n <-> p weak processes, p(n,gamma)D with its reverse, and the
# three deuterium-burning reactions d(d,n)3He, d(d,p)T and d(p,gamma)3He.
# The T and 3He they make are burned on by t(d,n)4He and 3He(d,p)4He, which
# are far faster than the reactions that feed them, so the network treats
# them as instantaneous: each D + D reaction consumes a third deuteron and
# yields 4He + n + p, and each D + p capture consumes a second deuteron and
# yields 4He.  Only the four species are tracked.


def _make_bbn_ode(
    *,
    exp,
    sqrt,
    hubble,
    n_baryon,
    weak_rate_np,
    weak_rate_pn,
    sigma_v_pn,
    sigma_v_dp,
    sigma_v_dd,
    deuterium_eq_inverse,
):
    def bbn_ode(y, x, params):
        log_eta10, N_eff = params[0], params[1]
        eta = 10.0 ** (log_eta10 - 10.0)
        T = Q / x
        z = M_E / T

        # e+e- thermodynamics: rho/T^4, p/T^4 and d(rho/T^4)/dz of the pair
        # gas (g = 4: two spins each of e- and e+), by Gauss-Laguerre
        # quadrature over u = p/T.  ``node`` is one quadrature point; numba
        # inlines it.
        def node(u, w):
            E = sqrt(u * u + z * z)
            boltzmann = exp(E)
            occupation = 1.0 / (boltzmann + 1.0)
            return (
                w * u * u * E * occupation,
                w * u * u * u * u / E * occupation,
                w
                * u
                * u
                * (z / E)
                * (occupation - E * boltzmann * occupation * occupation),
            )

        rho_e, p_e, drho_e = node(0.08764941047892776, 0.2250363148642515)
        r, pr, dr = node(0.4626963289150804, 0.5258360527623329)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(1.141057774831227, 0.8319613916870876)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(2.129283645098381, 1.146099240963762)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(3.437086633893207, 1.471751316966821)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(5.078018614549768, 1.813134687381371)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(7.070338535048234, 2.175517519694626)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(9.438314336391938, 2.565762750165059)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(12.21422336886616, 2.993215086371418)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(15.44152736878162, 3.471234483102158)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(19.18015685675314, 4.020044086444779)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(23.51590569399191, 4.672516607732875)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(28.57872974288214, 5.487420657986193)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(34.58339870228662, 6.585361233289205)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(41.94045264768833, 8.276357984364385)
        rho_e += r
        p_e += pr
        drho_e += dr
        r, pr, dr = node(51.70116033954332, 11.82427755165846)
        rho_e += r
        p_e += pr
        drho_e += dr
        rho = RHO_PHOTON + 2.0 / math.pi**2 * rho_e
        pressure = P_PHOTON + 2.0 / math.pi**2 * p_e / 3.0
        drho_dz = 2.0 / math.pi**2 * drho_e
        g_rho_em = 30.0 / math.pi**2 * rho
        g_s_em = 45.0 / (2.0 * math.pi**2) * (rho + pressure)
        # T ds/dT = d rho/dT, so d ln g_s / d ln T = T (d rho/dT) / (rho + p) - 3
        # with rho = T^4 R(z): T d rho/dT = T^4 (4 R - z R'(z)).
        dlngs_dlnT = (4.0 * rho - z * drho_dz) / (rho + pressure) - 3.0

        H = hubble(T, N_eff, g_rho_em, g_s_em)
        n_b = n_baryon(T, eta, g_s_em)

        Gamma_np = weak_rate_np(T)
        Gamma_pn = weak_rate_pn(T)

        # n + p <-> D + gamma, net forward rate
        rate_np = (
            n_b * sigma_v_pn(T) * (y[0] * y[1] - y[2] * deuterium_eq_inverse(T, n_b))
        )
        # D + D -> (n 3He | p T), then (3He d | T d) -> 4He: 3 D -> 4He + n + p.
        # The 1/2 is the identical-particle factor.
        rate_dd = 0.5 * n_b * sigma_v_dd(T) * y[2] * y[2]
        # D + p -> 3He + gamma, then 3He + D -> 4He + p: 2 D -> 4He
        rate_dp = n_b * sigma_v_dp(T) * y[2] * y[1]

        dt_dx = (1.0 + dlngs_dlnT / 3.0) / (H * x)
        return (
            (Gamma_pn * y[1] - Gamma_np * y[0] - rate_np + rate_dd) * dt_dx,
            (Gamma_np * y[0] - Gamma_pn * y[1] - rate_np + rate_dd) * dt_dx,
            (rate_np - 3.0 * rate_dd - 2.0 * rate_dp) * dt_dx,
            (rate_dd + rate_dp) * dt_dx,
        )

    return bbn_ode


def make_bbn_ode():
    """Build the right-hand side in its device, traced and host forms."""
    return build_rhs(
        _make_bbn_ode,
        hubble=HUBBLE,
        n_baryon=N_BARYON,
        weak_rate_np=WEAK_RATE_NP,
        weak_rate_pn=WEAK_RATE_PN,
        sigma_v_pn=SIGMA_V_PN,
        sigma_v_dp=SIGMA_V_DP,
        sigma_v_dd=SIGMA_V_DD,
        deuterium_eq_inverse=DEUTERIUM_EQ_INVERSE,
    )


# dY/dx: ``.device`` is the 4-tuple for the modax kernel, ``.jax`` the array
# for the Diffrax and scipy backends.
BBN_ODE = make_bbn_ode()


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

SOLVER_RTOL = 1e-4
# Deuterium starts near 1e-11 and grows as exp(B_D/T) until the bottleneck
# opens, and a solver that lets it fall below the absolute tolerance during
# that growth loses control of when the bottleneck opens: with atol = 1e-9
# the SciPy stiff solvers return Y_P wrong by 0.2 on most of the parameter
# grid while reporting success. atol = 1e-13 keeps deuterium inside the error
# norm throughout, and all backends then agree to the rtol.
SOLVER_ATOL = 1e-13
SOLVER_FIRST_STEP = 0.1
SOLVER_MAX_STEPS = 1024

# The science uses the GPU-batched modax Rodas5P solver. For a like-for-like
# timing, ``--benchmark`` also runs the identical four-species stiff network
# on Diffrax Kvaerno5 (GPU, jax.vmap) and on serial scipy.solve_ivp Radau, the
# no-GPU baseline, each trajectory on its own CPU core. Radau rather than
# LSODA, the choice of codes such as the original ECHO21: at these tolerances
# LSODA's non-stiff start fails with repeated error-test failures on a
# fraction of the parameter grid, because deuterium equilibrates some 1e15
# times faster than the Universe expands from the first step, and BDF returns
# a wrong Y_P on one grid point in 2000 while reporting success. Radau is the
# one SciPy stiff solver that integrates the whole grid correctly at the
# tolerances the GPU solvers use; it runs with an automatic initial step, as
# serial codes do.
MODAX_KWARGS = dict(
    lu_precision="fp32",
    rtol=SOLVER_RTOL,
    atol=SOLVER_ATOL,
    first_step=SOLVER_FIRST_STEP,
    max_steps=SOLVER_MAX_STEPS,
)
BACKENDS = Backends(
    modax_solve=rodas5P_solve,
    modax_kwargs=MODAX_KWARGS,
    diffrax_method="kvaerno5",
    diffrax_kwargs=dict(
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        first_step=SOLVER_FIRST_STEP,
        max_steps=8192,
    ),
    scipy_kwargs=dict(
        method="Radau", rtol=SOLVER_RTOL, atol=SOLVER_ATOL, first_step=None
    ),
)


def initial_conditions(params):
    """Equilibrium state at T = 10 MeV (x = Q/10 ~ 0.13) for ``params``.

    Nucleons in weak equilibrium, deuterium at its Saha abundance (~1e-11,
    which depends on eta, so the state is per trajectory), no helium. The
    pairs are relativistic here, so n_b = (11/4) eta n_gamma to 0.02 per
    cent. Starting deuterium at zero instead leaves a 1e-15-of-the-window
    transient that trips the CPU solvers' initial step selection.
    """
    x0 = X_SPAN[0]
    T0 = Q / x0
    eta = 10.0 ** (params[..., 0] - 10.0)
    yn0 = 1.0 / (1.0 + jnp.exp(x0))
    yp0 = 1.0 - yn0
    n_b = eta * (2.0 * ZETA3 / math.pi**2) * T0**3 * (G_S_HIGH / 2.0)
    k_d = n_b * 0.75 * (4.0 * math.pi / (M_N * T0)) ** 1.5 * jnp.exp(B_D / T0)
    yd0 = k_d * yn0 * yp0
    return jnp.stack(jnp.broadcast_arrays(yn0, yp0, yd0, jnp.zeros_like(yd0)), axis=-1)


def predict_abundances(params):
    """Integrate BBN network and return [Y_P, D/H] for given params."""
    sol = rodas5P_solve(
        BBN_ODE.device, initial_conditions(params), X_SAVE, params, **MODAX_KWARGS
    )
    _, Yp, Yd, YHe = sol[0, -1]
    Y_P = 4.0 * YHe  # helium mass fraction
    D_H = Yd / Yp  # D/H number ratio
    return jnp.array([Y_P, D_H])


# ---------------------------------------------------------------------------
# Benchmark: the chi^2-grid use case, N independent (eta, N_eff) universes
# ---------------------------------------------------------------------------


def sample_grid_params(n):
    """A near-square (eta_10, N_eff) grid covering the prior box, flattened."""
    side = int(np.ceil(np.sqrt(n)))
    eta = np.linspace(LO[0], HI[0], side)
    neff = np.linspace(LO[1], HI[1], side)
    ee, nn = np.meshgrid(eta, neff)
    grid = np.column_stack([ee.ravel(), nn.ravel()])[:n]
    return jnp.asarray(grid, dtype=jnp.float64)


def benchmark(n, backends, repeats):
    params = sample_grid_params(n)
    run_benchmark(
        BACKENDS,
        BBN_ODE,
        initial_conditions(params),
        X_SAVE,
        params,
        backend_names=backends,
        repeats=repeats,
        title=f"BBN forward-solve benchmark: N = {n:,} stiff 4-species universes",
        column="Y_P, D/H (mid-grid)",
        # mid-grid sample for a sanity check on agreement across backends
        metric=lambda sol: (
            f"{4.0 * sol[n // 2, -1, 3]:.5f}, {sol[n // 2, -1, 2] / sol[n // 2, -1, 1]:.4e}"
        ),
    )


# ---------------------------------------------------------------------------
# Observational data and statistical model
# ---------------------------------------------------------------------------

# Synthetic observations: the network's own predictions at the Planck 2018
# parameters (eta_10 = 6.1, N_eff = 3.044), with the uncertainties of the
# real measurements -- Aver et al. (2015) for Y_P, Cooke et al. (2018) for
# D/H.  Fitting the network to its own output keeps the posterior a clean
# recovery test; its predictions sit within 0.1 per cent (Y_P) and 4 per
# cent (D/H) of the full-network codes, see README.md.
Y_P_OBS = 0.24700
SIGMA_YP = 0.0040

DH_OBS = 2.542e-5
SIGMA_DH = 0.030e-5

# Uniform box prior: log10(eta_10) in [0.5, 1.0], N_eff in [2.0, 4.0]
LO = jnp.array([0.5, 2.0])
HI = jnp.array([1.0, 4.0])


def log_prior(theta):
    return jnp.where(jnp.all((theta >= LO) & (theta <= HI)), 0.0, -jnp.inf)


def log_likelihood(theta):
    # Select rather than branch. The sampler always evaluates this under
    # jax.vmap, where lax.cond with a batched predicate is rewritten into a
    # select that runs both branches anyway -- so this costs nothing extra --
    # and the solver's custom_vmap rule cannot be traced inside a batched
    # lax.cond (JAX asserts that a custom_vmap's captured constants are
    # unbatched, which cond's batching rule violates).
    in_prior = jnp.all((theta >= LO) & (theta <= HI))
    preds = predict_abundances(theta)
    chi2 = ((preds[0] - Y_P_OBS) / SIGMA_YP) ** 2 + (
        (preds[1] - DH_OBS) / SIGMA_DH
    ) ** 2
    ll = -0.5 * chi2
    return jnp.where(in_prior & jnp.isfinite(ll), ll, -jnp.inf)


# ---------------------------------------------------------------------------
# Nested sampling
# ---------------------------------------------------------------------------

N_LIVE = 512
N_INNER = 3  # slice steps per dead-particle replacement
NUM_DEL = 256  # replacement chains run in parallel inside BlackJAX NSS (vmap batch)
TOL_LOGZ = 5.0  # convergence: stop when logZ - logZ_live > tol (remaining live contribution < exp(-tol) of accumulated)
NS_CHUNK_SIZE = 4
STATUS_EVERY_CHUNKS = 2
MAX_NS_STEPS = 10 * N_LIVE
SLICE_MAX_STEPS = 2
SLICE_MAX_SHRINKAGE = 5


def run_nested_sampling():
    import blackjax.ns.nss as nss
    from blackjax.ns import utils
    from blackjax.ns.base import NSInfo

    key = jax.random.key(42)
    key, init_key, pos_key = jax.random.split(key, 3)
    positions = jax.random.uniform(pos_key, (N_LIVE, 2)) * (HI - LO) + LO

    kernel = nss.as_top_level_api(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_inner_steps=N_INNER,
        num_delete=NUM_DEL,
        max_steps=SLICE_MAX_STEPS,
        max_shrinkage=SLICE_MAX_SHRINKAGE,
    )
    state = kernel.init(positions, rng_key=init_key)
    step = kernel.step

    def _strip_update_info(info):
        return NSInfo(info.particles, None)

    def _flatten_dead_info(info):
        particles = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
            info.particles,
        )
        return NSInfo(particles, None)

    def _chunk_body(carry, _):
        key, state = carry
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        return (key, state), _strip_update_info(info)

    @jax.jit
    def run_chunk(key, state):
        return jax.lax.scan(_chunk_body, (key, state), None, length=NS_CHUNK_SIZE)

    dead_info_chunks = []
    max_chunks = (MAX_NS_STEPS + NS_CHUNK_SIZE - 1) // NS_CHUNK_SIZE
    steps_done = 0
    for chunk in range(max_chunks):
        (key, state), chunk_info = run_chunk(key, state)
        dead_info_chunks.append(_flatten_dead_info(chunk_info))
        steps_done = min((chunk + 1) * NS_CHUNK_SIZE, MAX_NS_STEPS)
        logZ = state.integrator.logZ
        logZ_live = state.integrator.logZ_live
        if chunk % STATUS_EVERY_CHUNKS == 0:
            print(
                f"  step {steps_done:5d}  "
                f"logZ={float(logZ):.3f}  logZ_live={float(logZ_live):.3f}",
                flush=True,
            )
        if jnp.isfinite(logZ) and (float(logZ) - float(logZ_live)) > TOL_LOGZ:
            print(f"  Converged by step {steps_done}", flush=True)
            break

    # Combine dead particles + final live points into one NSInfo object
    final_info = utils.finalise(state, dead_info_chunks, update_info=False)
    dead_positions = final_info.particles.position  # shape (N_total, 2)

    # Compute posterior log-weights; shape=1 gives a single MC estimate
    key, w_key = jax.random.split(key)
    log_w = utils.log_weights(w_key, final_info, shape=1)  # (N_total, 1)
    log_w = log_w[:, 0]  # (N_total,)
    w = jnp.exp(log_w - jnp.max(log_w))
    w = w / w.sum()
    w_np = np.array(w)

    eta10_samples = 10.0 ** np.array(dead_positions[:, 0])
    neff_samples = np.array(dead_positions[:, 1])

    eta10_mean = float(np.average(eta10_samples, weights=w_np))
    eta10_std = float(
        np.sqrt(np.average((eta10_samples - eta10_mean) ** 2, weights=w_np))
    )
    neff_mean = float(np.average(neff_samples, weights=w_np))
    neff_std = float(np.sqrt(np.average((neff_samples - neff_mean) ** 2, weights=w_np)))

    print(f"\nlog Z = {float(state.integrator.logZ):.3f}")
    print(f"eta_10 = {eta10_mean:.3f} +/- {eta10_std:.3f}")
    print(f"N_eff  = {neff_mean:.3f} +/- {neff_std:.3f}")

    _plot_posterior(w_np, eta10_samples, neff_samples)


def _plot_posterior(w, eta10_samples, neff_samples):
    """Posterior scatter and marginals, in the house style of ``benchmarks/``."""
    configure_latex_plot_style(plt)
    print_plot_title("BBN posterior from nested slice sampling")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    # 2D scatter coloured by posterior weight
    sc = axes[0].scatter(
        eta10_samples,
        neff_samples,
        c=w,
        s=4,
        cmap="viridis",
        alpha=0.7,
        linewidths=0,
    )
    fig.colorbar(sc, ax=axes[0], label="Posterior weight")
    axes[0].set_xlabel(r"$\eta_{10}$")
    axes[0].set_ylabel(r"$N_\mathrm{eff}$")

    # marginals
    axes[1].hist(eta10_samples, weights=w, bins=40, color=MODAX_COLOR, density=True)
    axes[1].set_xlabel(r"$\eta_{10}$")
    axes[1].set_ylabel("Posterior density")
    axes[2].hist(neff_samples, weights=w, bins=40, color=MODAX_COLOR, density=True)
    axes[2].set_xlabel(r"$N_\mathrm{eff}$")
    axes[2].set_ylabel("Posterior density")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = Path(__file__).parent / "posterior.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args(__doc__, n_default=10_000, n_help="ensemble size")
    if args.benchmark:
        benchmark(args.n, args.backends, args.repeats)
        return

    print("Integrating BBN network (4 species, x = Q/T)", flush=True)
    print("Running nested sampling ...", flush=True)
    run_nested_sampling()


if __name__ == "__main__":
    main()
