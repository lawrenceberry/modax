# 21cm IGM Evolution Example

This example demonstrates Monte Carlo uncertainty quantification for the global
21cm signal using a massively batched, uncoupled ODE solve.  It draws uncertain
astrophysical parameters for the first galaxies, evolves many independent IGM
thermal histories, and plots the median plus 95% prior envelope of the
sky-averaged brightness temperature.

Run it with:

```bash
uv run python examples/21cm_igm_evolution/main.py
```

The default run uses 50,000 trajectories.  Tests use a much smaller batch.

## Solver-backend benchmark

The batched ensemble solve can be timed across solver backends on identical
right-hand-side code with the `--benchmark` flag: modax Rodas5P (GPU), Diffrax
Kvaerno5 (GPU), and serial `scipy.solve_ivp` LSODA (the no-GPU baseline used by
codes such as ECHO21).

```bash
# head-to-head at N=2000 (scipy is the slow one)
uv run python examples/21cm_igm_evolution/main.py --benchmark --n 2000
# GPU backends at the full default ensemble size
uv run python examples/21cm_igm_evolution/main.py --benchmark --backends modax diffrax --n 50000
```

## Model

The integration variable is

$$
u = \log\left(\frac{1 + z_\mathrm{initial}}{1 + z}\right),
$$

so redshift decreases monotonically from $z=100$ to $z=5$.  The solver state is

$$
y = [\log T_k,\ \mathrm{logit}(x_e),\ \mathrm{logit}(Q)],
$$

where $T_k$ is the gas kinetic temperature, $x_e$ is the residual electron
fraction in the bulk IGM, and $Q$ is a volume-filling ionized fraction.  The
log/logit state keeps the evolved quantities in physical ranges.

The thermal equation includes adiabatic cooling, Compton coupling to the CMB,
X-ray heating, and a smaller Ly-alpha heating term.  The ionization equations
include recombination, X-ray ionization, and an ionizing emissivity term that
tracks the star-formation source history.

This is a calibrated surrogate for a batching demonstration, not a precision
global-signal code.

## Uncertain Parameters

Each trajectory samples

| Parameter | Prior |
| --- | --- |
| $\log_{10} f_\star$ | Uniform $[-3, -0.5]$ |
| $\log_{10} f_X$ | Uniform $[-2, 2]$ |
| $\log_{10} T_\mathrm{vir}$ | Uniform $[3.5, 5.5]$ |

The analytic source history is a smooth stand-in for the star-formation rate
density.  Larger $f_\star$ increases the Ly-alpha, X-ray, and ionizing
emissivities.  Larger $T_\mathrm{vir}$ delays source turn-on because only rarer,
more massive haloes can form stars.

## Spin Temperature and Brightness

The spin temperature is derived algebraically using the Field equilibrium form,
with collisional coupling and Ly-alpha Wouthuysen-Field coupling:

$$
T_s^{-1}
=
\frac{T_\gamma^{-1} + (x_c + x_\alpha)T_k^{-1}}
{1 + x_c + x_\alpha}.
$$

The global brightness temperature follows the usual mean-density approximation:

$$
\delta T_b \approx
27\,\mathrm{mK}\,
x_\mathrm{HI}
\left(\frac{\Omega_b h^2}{0.023}\right)
\left(\frac{0.15}{\Omega_m h^2}\frac{1+z}{10}\right)^{1/2}
\left(1 - \frac{T_\gamma}{T_s}\right).
$$

Observed frequency is

$$
\nu = \frac{1420.40575177\,\mathrm{MHz}}{1 + z}.
$$

## References

1. S. Mittal, T. A. Voytek, G. Kulkarni, *ECHO21: a tool for modelling global 21-cm signal from dark ages to reionization*, arXiv:2503.11762; RASTI 5 (2026) rzag001. https://arxiv.org/abs/2503.11762
2. ECHO21 code: https://github.com/shikharmittal04/echo21
3. S. Furlanetto, S. P. Oh, F. Briggs, *Cosmology at low frequencies: The 21 cm transition and the high-redshift Universe*, Physics Reports 433, 181-301 (2006).
4. J. Pritchard, A. Loeb, *21 cm cosmology in the 21st century*, Reports on Progress in Physics 75, 086901 (2012).
5. S. Wouthuysen, *On the excitation mechanism of the 21-cm radio-frequency interstellar hydrogen emission line*, AJ 57, 31 (1952).
6. G. Field, *Excitation of the hydrogen 21-cm line*, Proceedings of the IRE 46, 240 (1958).
