"""Solve Mukhanov-Sasaki modes for a quadratic inflation example.

The Mukhanov-Sasaki equation describes the gauge-invariant scalar perturbation
that seeds the CMB temperature anisotropies and the large-scale distribution of
matter.  For a single inflaton field on an FLRW background, the canonical
perturbation variable v obeys

    v_k'' + (k^2 - z'' / z) v_k = 0,

where primes denote derivatives with respect to conformal time, k is the
comoving wavenumber, z = a dphi/dN in reduced Planck units, a is the scale
factor, phi is the homogeneous inflaton, and N = log(a) is e-fold time.  The
quantity z'' / z is a time-dependent effective mass that couples each quantum
fluctuation to the evolving background geometry.

The standard initial condition is the Bunch-Davies vacuum.  Deep inside the
horizon, k >> aH, curvature and expansion are negligible over the wavelength,
so each mode behaves like a flat-spacetime oscillator:

    v_k ~= exp(-i k tau) / sqrt(2 k).

The equation follows from expanding the Einstein-Hilbert plus scalar-field
action to second order around an FLRW background.  Metric and inflaton
perturbations are combined into one gauge-invariant variable, v, and applying
the Euler-Lagrange equations to the quadratic action gives the linear
Mukhanov-Sasaki equation.  Spatial homogeneity and isotropy make the background
depend only on time; after a Fourier transform, the linearized PDE splits into
independent ODEs labelled by |k|.  That independence is why this example maps
many k modes onto a batched ensemble solve.

The final frozen curvature perturbation is R_k = v_k / z.  Its dimensionless
power spectrum is P_R(k) = k^3 |R_k|^2 / (2 pi^2).  Einstein-Boltzmann solvers
commonly take this primordial spectrum through its amplitude A_s at a pivot
scale and its scalar spectral index n_s = d log(P_R) / d log(k).

In the code below the mode equation is solved in e-fold time N, not conformal
time.  With epsilon = -d log(H) / dN, q = (z'' / z) / (aH)^2, and
x = k / (aH), the complex mode equation becomes

    d^2 v_k/dN^2 + (1 - epsilon) dv_k/dN + (x^2 - q) v_k = 0.

The solver state stores real and imaginary parts separately:

    y = [Re(v_k), Im(v_k), Re(dv_k/dN), Im(dv_k/dN)].

Each trajectory is integrated over a normalized time s in [0, 1], where
N = N_start + s (N_stop - N_start), so the right-hand side returned to the
solver is multiplied by dN/ds = N_stop - N_start.

References:
    D. Baumann, "TASI Lectures on Inflation", arXiv:0907.5424.

Usage:
    uv run python examples/mukhanov_sasaki/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solvers.tsit5jax import solve as tsit5_solve

jax.config.update("jax_enable_x64", True)

N_MODES = 41  # Number of Fourier modes in the ensemble solve.
K_MIN_MPC = 1.0e-4  # Smallest physical comoving wavenumber, in Mpc^-1.
K_MAX_MPC = 1.0  # Largest physical comoving wavenumber, in Mpc^-1.
K_PIVOT_MPC = 0.05  # Pivot scale used for A_s and n_s, in Mpc^-1.

TARGET_A_S = 2.1e-9  # Observed scalar amplitude used to set the mass scale.
N_PIVOT_BEFORE_END = 55.0  # E-folds between pivot horizon exit and inflation end.

PHI_INITIAL = 17.5  # Initial homogeneous inflaton value, in reduced Planck units.
D_PHI_DN_INITIAL = -2.0 / PHI_INITIAL  # Slow-roll initial dphi/dN estimate.
N_BACKGROUND_MAX = 80.0  # Maximum e-fold time for the background integration.
N_BACKGROUND_SAMPLES = 2000  # Save points used to build interpolation tables.

X_INITIAL = 80.0  # Initial subhorizon ratio k/(aH) for each mode.
X_FINAL = 1.0e-3  # Final superhorizon ratio k/(aH) for each mode.

BACKGROUND_RTOL = 1.0e-9  # Relative tolerance for the background solve.
BACKGROUND_ATOL = 1.0e-11  # Absolute tolerance for the background solve.
MODE_RTOL = 1.0e-7  # Relative tolerance for the Mukhanov-Sasaki mode solve.
MODE_ATOL = 1.0e-9  # Absolute tolerance for the Mukhanov-Sasaki mode solve.
MODE_MAX_STEPS = 200000  # Step cap per mode trajectory.


def quadratic_mass_from_slow_roll(
    target_a_s: float = TARGET_A_S,
    n_star: float = N_PIVOT_BEFORE_END,
) -> float:
    """Return m for V = 0.5 m^2 phi^2 from the slow-roll amplitude."""
    phi_star_sq = 4.0 * n_star + 2.0
    epsilon_v = 2.0 / phi_star_sq
    return float(np.sqrt(target_a_s * 48.0 * np.pi**2 * epsilon_v / phi_star_sq))


MASS = quadratic_mass_from_slow_roll()


def potential(phi, mass):
    """Evaluate the quadratic inflaton potential in reduced Planck units."""
    return 0.5 * mass**2 * phi**2


def dpotential_dphi(phi, mass):
    """Evaluate dV/dphi for the quadratic inflaton potential."""
    return mass**2 * phi


def background_ode(y, n_efolds, params):
    """Return dphi/dN and d2phi/dN^2 for the homogeneous inflaton background."""
    del n_efolds
    mass = params[0]
    phi = y[0]
    dphi_dn = y[1]
    epsilon = 0.5 * dphi_dn**2
    h_sq = potential(phi, mass) / (3.0 - epsilon)
    d2phi_dn2 = -(3.0 - epsilon) * dphi_dn - dpotential_dphi(phi, mass) / h_sq
    return jnp.array([dphi_dn, d2phi_dn2])


def solve_background():
    """Integrate the homogeneous inflationary background over e-fold time."""
    times = jnp.linspace(0.0, N_BACKGROUND_MAX, N_BACKGROUND_SAMPLES)
    y0 = jnp.array([PHI_INITIAL, D_PHI_DN_INITIAL], dtype=jnp.float64)
    params = jnp.array([MASS], dtype=jnp.float64)
    solution = tsit5_solve(
        background_ode,
        y0,
        times,
        params,
        rtol=BACKGROUND_RTOL,
        atol=BACKGROUND_ATOL,
        first_step=1.0e-4,
    )[0]
    return np.asarray(times), np.asarray(solution)


def first_crossing(x, y, value):
    """Linearly interpolate the first x location where y crosses value upward."""
    below = np.nonzero(y >= value)[0]
    if below.size == 0 or below[0] == 0:
        raise RuntimeError(f"Could not find crossing for value {value}.")
    i = below[0]
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]
    return float(x0 + (value - y0) * (x1 - x0) / (y1 - y0))


def build_background_tables(n_grid, background):
    """Build inflationary interpolation tables used by the mode equations."""
    phi_full = background[:, 0]
    dphi_dn_full = background[:, 1]
    epsilon_full = 0.5 * dphi_dn_full**2
    n_end = first_crossing(n_grid, epsilon_full, 1.0)
    n_pivot = n_end - N_PIVOT_BEFORE_END
    if n_pivot <= n_grid[0]:
        raise RuntimeError("Background does not start early enough for the pivot.")

    inflationary = n_grid <= n_end
    n_grid = n_grid[inflationary]
    phi = phi_full[inflationary]
    dphi_dn = dphi_dn_full[inflationary]
    epsilon = 0.5 * dphi_dn**2

    h = np.sqrt(potential(phi, MASS) / (3.0 - epsilon))
    log_a_h = n_grid + np.log(h)
    z = np.exp(n_grid) * dphi_dn
    dz_dn = np.gradient(z, n_grid, edge_order=2)
    d2z_dn2 = np.gradient(dz_dn, n_grid, edge_order=2)
    q = d2z_dn2 / z + (1.0 - epsilon) * dz_dn / z

    return {
        "n": n_grid,
        "phi": phi,
        "dphi_dn": dphi_dn,
        "epsilon": epsilon,
        "h": h,
        "log_a_h": log_a_h,
        "z": z,
        "q": q,
        "n_end": n_end,
        "n_pivot": n_pivot,
    }


def interp_np(x, xp, fp):
    """Return one-dimensional NumPy linear interpolation as a Python float."""
    return float(np.interp(x, xp, fp))


def prepare_mode_problem(tables):
    """Create k values, per-mode integration windows, and Bunch-Davies y0."""
    physical_k = np.geomspace(K_MIN_MPC, K_MAX_MPC, N_MODES)
    log_a_h_pivot = interp_np(tables["n_pivot"], tables["n"], tables["log_a_h"])
    code_k_pivot = np.exp(log_a_h_pivot)
    code_k = code_k_pivot * physical_k / K_PIVOT_MPC
    log_code_k = np.log(code_k)

    n_start = np.interp(log_code_k - np.log(X_INITIAL), tables["log_a_h"], tables["n"])
    n_stop = np.interp(log_code_k - np.log(X_FINAL), tables["log_a_h"], tables["n"])

    phase = X_INITIAL
    norm = 1.0 / np.sqrt(2.0 * code_k)
    v_re = norm * np.cos(phase)
    v_im = norm * np.sin(phase)
    dv_dn_re = X_INITIAL * v_im
    dv_dn_im = -X_INITIAL * v_re
    y0 = np.stack([v_re, v_im, dv_dn_re, dv_dn_im], axis=1)
    params = np.stack([code_k, n_start, n_stop], axis=1)
    return physical_k, code_k, y0, params


def make_mode_ode(tables):
    """Create the Mukhanov-Sasaki RHS using background interpolation tables."""
    n_table = jnp.asarray(tables["n"], dtype=jnp.float64)
    epsilon_table = jnp.asarray(tables["epsilon"], dtype=jnp.float64)
    log_a_h_table = jnp.asarray(tables["log_a_h"], dtype=jnp.float64)
    q_table = jnp.asarray(tables["q"], dtype=jnp.float64)

    def mode_ode(y, s, params):
        code_k, n_start, n_stop = params
        delta_n = n_stop - n_start
        n_now = n_start + s * delta_n
        epsilon = jnp.interp(n_now, n_table, epsilon_table)
        log_a_h = jnp.interp(n_now, n_table, log_a_h_table)
        q = jnp.interp(n_now, n_table, q_table)
        k_over_a_h = code_k * jnp.exp(-log_a_h)

        v_re, v_im, dv_re, dv_im = y
        omega_sq = k_over_a_h**2 - q
        d2v_re = -(1.0 - epsilon) * dv_re - omega_sq * v_re
        d2v_im = -(1.0 - epsilon) * dv_im - omega_sq * v_im
        return delta_n * jnp.array([dv_re, dv_im, d2v_re, d2v_im])

    return mode_ode


def solve_modes(tables):
    """Solve all uncoupled Mukhanov-Sasaki Fourier modes as one ensemble."""
    physical_k, code_k, y0, params = prepare_mode_problem(tables)
    solution = tsit5_solve(
        make_mode_ode(tables),
        jnp.asarray(y0, dtype=jnp.float64),
        jnp.array([0.0, 1.0], dtype=jnp.float64),
        jnp.asarray(params, dtype=jnp.float64),
        rtol=MODE_RTOL,
        atol=MODE_ATOL,
        first_step=1.0e-5,
        max_steps=MODE_MAX_STEPS,
    )
    return physical_k, code_k, params[:, 2], np.asarray(solution[:, -1, :])


def compute_power_spectrum(tables, code_k, n_stop, final_state):
    """Convert final mode amplitudes into the dimensionless curvature spectrum."""
    z_stop = np.interp(n_stop, tables["n"], tables["z"])
    v_abs_sq = final_state[:, 0] ** 2 + final_state[:, 1] ** 2
    return code_k**3 * v_abs_sq / z_stop**2 / (2.0 * np.pi**2)


def local_spectral_index(k, power, pivot):
    """Estimate n_s from a local log-log slope of the numerical spectrum."""
    width = 7 if k.size >= 7 else k.size
    center = int(np.argmin(np.abs(np.log(k / pivot))))
    lo = max(0, center - width // 2)
    hi = min(k.size, lo + width)
    lo = max(0, hi - width)
    slope, _ = np.polyfit(np.log(k[lo:hi]), np.log(power[lo:hi]), deg=1)
    return float(1.0 + slope)


def log_interp(x, xp, fp):
    """Interpolate a positive quantity linearly in log-log space."""
    return float(np.exp(np.interp(np.log(x), np.log(xp), np.log(fp))))


def slow_roll_estimates(tables):
    """Compute approximate quadratic-inflation slow-roll values at the pivot."""
    n_star = tables["n_end"] - tables["n_pivot"]
    phi_pivot = interp_np(tables["n_pivot"], tables["n"], tables["phi"])
    epsilon_v = 1.0 / (2.0 * n_star + 1.0)
    eta_v = epsilon_v
    v_pivot = potential(phi_pivot, MASS)
    a_s = v_pivot / (24.0 * np.pi**2 * epsilon_v)
    n_s = 1.0 - 6.0 * epsilon_v + 2.0 * eta_v
    n_s_compact = 1.0 - 2.0 / n_star
    return {
        "n_star": n_star,
        "epsilon_v": epsilon_v,
        "eta_v": eta_v,
        "A_s": a_s,
        "n_s": n_s,
        "n_s_compact": n_s_compact,
    }


def print_results(results):
    """Print numerical and slow-roll primordial-spectrum outputs side by side."""
    a_s = results["A_s"]
    n_s = results["n_s"]
    a_s_sr = results["A_s_slow_roll"]
    n_s_sr = results["n_s_slow_roll"]
    print("Mukhanov-Sasaki scalar spectrum from quadratic inflation")
    print(f"m                 = {MASS:.6e} M_pl")
    print(f"N modes           = {N_MODES}")
    print(f"k pivot           = {K_PIVOT_MPC:.6g} Mpc^-1")
    print(f"N_end             = {results['background']['N_end']:.6f}")
    print(f"N_*               = {results['background']['N_star']:.6f}")
    print()
    print("quantity    numerical MS        slow-roll estimate    diff")
    print(
        f"A_s         {a_s: .8e}       {a_s_sr: .8e}       "
        f"{(a_s - a_s_sr) / a_s_sr: .3e} rel"
    )
    print(f"n_s         {n_s: .8f}       {n_s_sr: .8f}       {n_s - n_s_sr: .3e} abs")
    print(
        f"n_s compact slow-roll sanity estimate: {results['n_s_slow_roll_compact']:.8f}"
    )


def main():
    """Run the background solve, mode solve, spectrum extraction, and reporting."""
    n_grid, background = solve_background()
    tables = build_background_tables(n_grid, background)
    physical_k, code_k, n_stop, final_state = solve_modes(tables)
    power = compute_power_spectrum(tables, code_k, n_stop, final_state)
    a_s = log_interp(K_PIVOT_MPC, physical_k, power)
    n_s = local_spectral_index(physical_k, power, K_PIVOT_MPC)
    slow_roll = slow_roll_estimates(tables)

    results = {
        "A_s": a_s,
        "n_s": n_s,
        "A_s_slow_roll": slow_roll["A_s"],
        "n_s_slow_roll": slow_roll["n_s"],
        "n_s_slow_roll_compact": slow_roll["n_s_compact"],
        "k_pivot": K_PIVOT_MPC,
        "k": physical_k,
        "power_spectrum": power,
        "background": {
            "N_end": tables["n_end"],
            "N_pivot": tables["n_pivot"],
            "N_star": slow_roll["n_star"],
            "epsilon_V": slow_roll["epsilon_v"],
            "eta_V": slow_roll["eta_v"],
        },
    }
    print_results(results)
    return results


if __name__ == "__main__":
    main()
