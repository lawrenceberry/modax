# Parameter estimation: primordial abundances

Bayesian inference of cosmological parameters from primordial light-element
abundances. Every likelihood evaluation is a forward integration of a stiff
big bang nucleosynthesis (BBN) network, and the sampler evaluates its whole
population of proposals at once, so the inner loop of the inference is an
ensemble solve.

The forward model is a reduced network of four species -- neutrons, protons,
deuterium and helium-4 -- with the standard thermal history, and the
$n \leftrightarrow p$ weak rates and the four nuclear reactions that set the
helium and deuterium yields taken from the fits of Serpico et al. (2004) that
underlie the PArthENoPE code. It predicts $Y_P$ to within 0.1 per cent and
$D/H$ to within 4 per cent of LINX run with the twelve-reaction "key" network
and PRIMAT 2023 rates (the reduced network of PRIMAT, which PRyMordial shares),
which it reduces further by taking the tritium and helium-3 burning to be
instantaneous. What it keeps in full is the stiffness of BBN,
which comes from the wide separation between the fast weak-interaction rates
and the slower nuclear burning, and that makes it a test of the implicit
Rodas5P solver on an inference problem. Precision predictions still need a
full network and current rates; LINX in particular is written in JAX and
integrates its network with Diffrax, which makes it a natural candidate for
the batched solvers here.

The posterior is sampled with nested slice sampling (Yallup et al. 2026), a
vectorised form of nested sampling built for GPUs and shipped in BlackJAX. The
sampler evaluates its proposals under `jax.vmap`, which the modax solver's
batching rule turns into one ensemble launch.

Run it with:

```bash
uv run python examples/bbn_estimation/main.py
```

The modax solve runs as a Numba-CUDA kernel, so this needs a CUDA GPU. The
nested sampling needs BlackJAX 1.6, which needs Python 3.11 or newer.

## Result

With 512 live points the sampler recovers $\eta_{10} = 6.11 \pm 0.15$ and
$N_\mathrm{eff} = 3.05 \pm 0.29$, on the values the synthetic observations
were generated at. The script writes the posterior to `posterior.png`.

## Solver-backend benchmark

The batched forward solve -- the parameter grid, or the sampler population --
can be timed across solver backends on identical right-hand-side code with
the `--benchmark` flag: modax Rodas5P and Diffrax Kvaerno5 on the GPU (Diffrax
has no Rosenbrock-Wanner method), and `scipy.integrate.solve_ivp` Radau on the
CPU, each trajectory integrated independently and the ensemble spread across
every core, as established serial codes integrate.

```bash
# all three backends over a 2000-point (eta, N_eff) grid, as in the paper
uv run python examples/bbn_estimation/main.py --benchmark --n 2000
# GPU backends at a larger grid
uv run python examples/bbn_estimation/main.py --benchmark --backends modax diffrax --n 50000
```

All three run at `rtol = 1e-4`, `atol = 1e-13`, where they agree on $Y_P$ and
$D/H$ at $(\eta_{10}, N_\mathrm{eff}) = (4.11, 3.0)$ to $2 \times 10^{-5}$ in $Y_P$ (0.2424) and 0.005 per cent in $D/H$.
The paper measures the 2000-point grid on an NVIDIA GeForce RTX 4070 SUPER
against an AMD Ryzen 9 7950X across all 32 hardware threads -- hardware of
comparable price and power draw:

| Backend | Wall clock | Per solve |
|---|---|---|
| SciPy Radau (CPU, 32 threads) | 30.1 s | 15 000 µs |
| Diffrax Kvaerno5 (GPU) | 6.8 s | 3410 µs |
| **modax Rodas5P (GPU)** | **62 ms** | **31 µs** |

That is roughly 480× the parallel CPU baseline and 110× Diffrax.

The absolute tolerance is set by deuterium, which starts near $10^{-11}$ and
grows as $e^{B_D/T}$ until the bottleneck opens. A solver that lets it drop
out of the error norm during that growth loses control of when the bottleneck
opens: at `atol = 1e-9` SciPy's Radau and BDF return $Y_P$ wrong by $0.2$ on
most of the grid while reporting success, and against a tight-tolerance
reference modax Rodas5P and Diffrax Kvaerno5 stay within $2 \times 10^{-4}$
only because their controllers happen to step more conservatively there.
At `atol = 1e-13` every backend agrees with the reference to $10^{-4}$ in
$Y_P$ across the grid. Radau is the CPU baseline rather than LSODA, the usual
choice of serial codes, because LSODA's non-stiff start fails with repeated
error-test failures on a fraction of the grid at every tolerance tried --
deuterium equilibrates some $10^{15}$ times faster than the Universe expands
from the first step -- and BDF misses one grid point in 2000 at `rtol = 1e-4`.

## Thermal history

The integration runs from $T = 10\,\mathrm{MeV}$ to $5\,\mathrm{keV}$. The
independent variable is not cosmic time $t$ but $x(t) = Q/T(t)$, with $T(t)$
the photon temperature and $Q = 1.293\,\mathrm{MeV}$ the neutron--proton mass
difference; the rates are per unit time, so the equations carry the factor
$\mathrm{d}t/\mathrm{d}x$, which needs the history $T(t)$. That history comes
from the Friedmann equation for the expansion $\dot a/a = H(T)$ together with
the conservation of the plasma entropy, which ties $T$ to $a$ as derived
below. Across this range the $e^+e^-$ pairs annihilate, which changes the
expansion rate, heats the photons relative to the neutrinos and dilutes the
baryons relative to the photons; all three enter the yields at the per-cent
level and are kept.

The electromagnetic plasma (photons and pairs) has energy density
$\rho_\mathrm{em} = (\pi^2/30)\, g_\rho(T)\, T^4$ and entropy density
$s_\mathrm{em} = (2\pi^2/45)\, g_s(T)\, T^3$, with

$$g_\rho(T) = 2 + \frac{30}{\pi^2}\,\frac{\rho_{e}(T)}{T^4}, \qquad
  g_s(T) = 2 + \frac{45}{2\pi^2}\,\frac{\rho_{e}(T) + p_{e}(T)}{T^4},$$

where $\rho_e$ and $p_e$ are the Fermi--Dirac integrals of the pair gas
($g = 4$, $\mu = 0$). They have no closed form, polynomial fits in
$\ln(m_e/T)$ converge too slowly to be worth having, and a lookup table
cannot be used in the device callback because Enzyme cannot differentiate
through a constant-memory array, so the right-hand side evaluates them by
16-point Gauss--Laguerre quadrature over $u = p/T$, with the nodes and
weights written out as literals (a runtime index into a tuple of constants
emits a bounds check Enzyme will not differentiate either). Sixteen points
reproduce the integrals to $10^{-5}$ of their massless value, which moves
$Y_P$ by under $10^{-6}$. The same quadrature gives
$\partial\rho_e/\partial z$ at $z = m_e/T$, from which
$\mathrm{d}\ln g_s/\mathrm{d}\ln T = T\,(\mathrm{d}\rho/\mathrm{d}T)/(\rho + p) - 3$.
Both $g_\rho$ and $g_s$ run from 5.5 (photons and relativistic pairs) at high
temperature to the photon value 2 at low temperature. Each quadrature point
costs one square root and one exponential in double precision, which a
GeForce GPU executes at a 64th of its single-precision rate, and the
quadrature is about half the cost of the modax kernel.

The neutrinos decouple while the pairs are still relativistic and are not
heated by the annihilation, so with the plasma entropy conserved separately
$T_\nu/T = (g_s(T)/5.5)^{1/3}$, which tends to $(4/11)^{1/3}$. The Friedmann
equation in radiation domination is then

$$H(T) = \sqrt{\frac{4\pi^3}{45}\left[g_\rho(T) + \frac{7}{4} N_\mathrm{eff} \left(\frac{T_\nu}{T}\right)^4\right]}\;\frac{T^2}{M_\mathrm{Pl}},$$

with $M_\mathrm{Pl} = 1.22 \times 10^{22}\,\mathrm{MeV}$ the unreduced Planck
mass. After annihilation the bracket is $2 + 0.454\,N_\mathrm{eff}$, which is
3.38 for $N_\mathrm{eff} = 3.044$; $N_\mathrm{eff}$ enters the model only
here.

Conservation of the plasma entropy per comoving volume gives the clock,
$\mathrm{d}x/\mathrm{d}t = H x\,/\,(1 + \tfrac{1}{3}\,\mathrm{d}\ln g_s/\mathrm{d}\ln T)$,
and the baryon density. With $\eta$ the baryon-to-photon ratio today,

$$n_b(T) = \eta\, n_\gamma(T)\,\frac{g_s(T)}{2}, \qquad n_\gamma = \frac{2\zeta(3)}{\pi^2} T^3,$$

which is $\tfrac{11}{4}\eta n_\gamma$ before annihilation and $\eta n_\gamma$
after it.

## Weak interaction rates

The $n \leftrightarrow p$ rates are the fits of Serpico et al. (2004,
Appendix C) to the Born rates with radiative and finite-nucleon-mass
corrections, which the PArthENoPE code uses:

$$\Gamma_{n\to p} = \frac{1}{\tau_n}\, e^{-q_{np}/z} \sum_{l=0}^{13} a_l z^{-l}, \qquad
  \Gamma_{p\to n} = \frac{1}{\tau_n}\, e^{-q_{pn} z} \sum_{l=1}^{10} b_l z^{-l},$$

with $z = m_e/T$, accurate to 0.06 per cent over $0.01 \le T/\mathrm{MeV} \le 10$,
and normalised to the measured neutron lifetime $\tau_n = 879.4\,\mathrm{s}$.
$\Gamma_{n\to p}$ includes free neutron decay and tends to $1/\tau_n$ at low
temperature. The fits assume the standard $T_\nu(T)$ history, so
$N_\mathrm{eff}$ does not enter them.

## Nuclear reactions

Four reactions are followed, with rates from the same paper's fits to the
measured $S$-factors (Appendix E), in the form $N_A \langle\sigma v\rangle$
as a function of $T_9 = T / 10^9\,\mathrm{K}$:

| Reaction | Role | $N_A\langle\sigma v\rangle$ at $T_9 = 1$ |
|---|---|---|
| $p(n,\gamma)\mathrm{D}$ | deuterium formation; reverse rate by detailed balance | $2.6 \times 10^4$ |
| $\mathrm{D}(d,n){}^3\mathrm{He}$ | deuterium burning | $1.4 \times 10^7$ |
| $\mathrm{D}(d,p)\mathrm{T}$ | deuterium burning | $1.1 \times 10^7$ |
| $\mathrm{D}(p,\gamma){}^3\mathrm{He}$ | deuterium burning | $3.5 \times 10^2$ |

Each fit is used up to the $T_9$ its authors trust it to (1.5, 10, 10 and 4
respectively) and held constant above, where deuterium sits in Saha
equilibrium whatever the rate. Against the PRIMAT 2023 tables used by LINX
these 2004 fits are within 1 per cent for $p(n,\gamma)\mathrm{D}$ and 3 to 9
per cent low for the three burning reactions.

The deuterium-burning products are not tracked. Tritium and helium-3 are
destroyed by $\mathrm{T}(d,n){}^4\mathrm{He}$ and ${}^3\mathrm{He}(d,p){}^4\mathrm{He}$,
whose rates exceed those of the reactions that make them by orders of
magnitude, so the network takes that burning to be instantaneous: a
$\mathrm{D}+\mathrm{D}$ reaction consumes a third deuteron and yields
${}^4\mathrm{He} + n + p$, and a $\mathrm{D}+p$ capture consumes a second
deuteron and yields ${}^4\mathrm{He}$. The surviving ${}^3\mathrm{He}$, at
${}^3\mathrm{He}/\mathrm{H} \approx 10^{-5}$, is what this closure misses,
and it is the main reason the network's $D/H$ sits a few per cent above the
full codes'.

The deuterium photodissociation rate follows from the Saha equation,

$$\frac{Y_d^\mathrm{eq}}{Y_n Y_p} = K_D(T, \eta) = n_b \,\frac{3}{4} \left(\frac{4\pi}{M_N T}\right)^{3/2} e^{B_D/T},$$

with $B_D = 2.2246\,\mathrm{MeV}$ and $M_N = 938.3\,\mathrm{MeV}$.

## ODE system

With $Y = [Y_n, Y_p, Y_d, Y_{^4\mathrm{He}}]$ the nucleon number fractions and
$\mathrm{d}t/\mathrm{d}x = (1 + \tfrac{1}{3}\,\mathrm{d}\ln g_s/\mathrm{d}\ln T)/(Hx)$,

$$\frac{\mathrm{d}Y_n}{\mathrm{d}x} = \left(\Gamma_{pn} Y_p - \Gamma_{np} Y_n - R_{np} + R_{dd}\right)\frac{\mathrm{d}t}{\mathrm{d}x}$$

$$\frac{\mathrm{d}Y_p}{\mathrm{d}x} = \left(\Gamma_{np} Y_n - \Gamma_{pn} Y_p - R_{np} + R_{dd}\right)\frac{\mathrm{d}t}{\mathrm{d}x}$$

$$\frac{\mathrm{d}Y_d}{\mathrm{d}x} = \left(R_{np} - 3 R_{dd} - 2 R_{dp}\right)\frac{\mathrm{d}t}{\mathrm{d}x}$$

$$\frac{\mathrm{d}Y_{^4\mathrm{He}}}{\mathrm{d}x} = \left(R_{dd} + R_{dp}\right)\frac{\mathrm{d}t}{\mathrm{d}x}$$

where the net rates are

$$R_{np} = n_b \langle\sigma v\rangle_{np} \left(Y_n Y_p - \frac{Y_d}{K_D}\right), \qquad
  R_{dd} = \tfrac{1}{2}\, n_b \left(\langle\sigma v\rangle_{ddn} + \langle\sigma v\rangle_{ddp}\right) Y_d^2, \qquad
  R_{dp} = n_b \langle\sigma v\rangle_{dp\gamma} Y_d Y_p,$$

the $\tfrac{1}{2}$ in $R_{dd}$ being the identical-particle factor.

**Conservation law:** $Y_n + Y_p + 2 Y_d + 4 Y_{^4\mathrm{He}} = 1$ at all times.

The integration starts at $T = 10\,\mathrm{MeV}$ with the nucleons in weak
equilibrium, $Y_n/Y_p = e^{-Q/T}$, and no nuclei; deuterium reaches its Saha
abundance within the first step. It ends at $T = 5\,\mathrm{keV}$, by which
point $D/H$ has frozen out: continuing to $2\,\mathrm{keV}$ changes it by less
than 0.1 per cent.

## Parameters and observables

| Parameter | Symbol | Prior range |
|-----------|--------|-------------|
| Baryon-to-photon ratio (log) | $\log_{10}(\eta_{10})$ | $[0.5,\, 1.0]$ |
| Effective neutrino species | $N_\mathrm{eff}$ | $[2.0,\, 4.0]$ |

The observations are synthetic: the network's own predictions at the Planck
2018 parameters ($\eta_{10} = 6.1$, $N_\mathrm{eff} = 3.044$), with the
uncertainties of the real measurements, so that the posterior is a clean
recovery test.

| Observable | Synthetic value | 1-sigma | Measured |
| --- | --- | --- | --- |
| Helium mass fraction $Y_P$ | 0.2470 | 0.0040 | $0.2449 \pm 0.0040$ (Aver et al. 2015) |
| Deuterium ratio $D/H$ | $2.54 \times 10^{-5}$ | $0.03 \times 10^{-5}$ | $(2.527 \pm 0.030) \times 10^{-5}$ (Cooke et al. 2018) |

For comparison, LINX with its PRIMAT 2023 key network gives $Y_P = 0.2471$ and
$D/H = 2.43 \times 10^{-5}$ at $\Omega_b h^2 = 0.02242$ ($\eta_{10} \approx 6.14$),
where this network gives $0.2470$ and $2.52 \times 10^{-5}$.

## References

1. P. D. Serpico, S. Esposito, F. Iocco, G. Mangano, G. Miele, O. Pisanti (2004). *Nuclear reaction network for primordial nucleosynthesis: a detailed analysis of rates, uncertainties and light nuclei yields.* JCAP **12**, 010.
2. C. Pitrou, A. Coc, J.-P. Uzan, E. Vangioni (2018). *Precision big bang nucleosynthesis with improved Helium-4 predictions.* Phys. Rep. **754**, 1.
3. C. Giovanetti, M. Lisanti, H. Liu, S. Mishra-Sharma, J. T. Ruderman (2025). *LINX: A fast, differentiable, and extensible big bang nucleosynthesis package.* Phys. Rev. D.
4. R. J. Cooke, M. Pettini, C. C. Steidel (2018). *One Percent Determination of the Primordial Deuterium Abundance.* ApJ **855**, 102.
5. E. Aver, K. A. Olive, E. D. Skillman (2015). *The effects of He I λ10830 on helium abundance determinations.* JCAP **07**, 011.
6. J. Skilling (2006). *Nested sampling for general Bayesian computation.* Bayesian Analysis **1**, 833.
7. D. Yallup, N. Kroupa, W. Handley (2026). *Nested Slice Sampling: Vectorized Nested Sampling for GPU-Accelerated Inference.* arXiv:2601.23252.
