# Year-averaged spin coefficients $h_k(\theta_{\text{ecl}})$ under L2 scan

Closed-form derivation, validation against Falcons.jl, and literature search.
First written 2026-04-24 to back the novelty claim in
``augr.crosslinks``; see ``scripts/falcons_validation/README.md`` for
the reproduction recipe.

## Problem

For a CMB satellite in an L2-class orbit:

- Anti-sun direction rotates around the ecliptic at 1 rev/yr.
- Spin axis precesses around antisun at half-angle $\alpha_F$ (Falcons/Takase
  convention; LiteBIRD: $\alpha_F = 45^\circ$, Planck: $7.5^\circ$).
- Boresight scans around spin axis at half-angle $\beta_F$ (LiteBIRD:
  $\beta_F = 50^\circ$, Planck: $85^\circ$).
- Period hierarchy $T_\text{year} \gg T_\text{prec} \gg T_\text{spin}$.

The per-pixel "cross-link factor" / "spin coefficient" is
$h_k(\hat n) = \langle e^{-ik\psi} \rangle$, averaged over the time samples
landing in pixel $\hat n$, where $\psi$ is the scan-direction-vs-north angle
(Wallis et al. 2017 spin formalism). After year-averaging, the SO(2)
symmetry around the ecliptic pole reduces $h_k$ to a 1-D function
$h_k(\theta_\text{ecl})$.

The downstream use of $h_k$ is in differential-systematic propagation:
Wallis 2017 Eqs. 20-22 give the B-mode bias from differential gain /
pointing / ellipticity in terms of $h_1, h_2, h_4$.

## Closed form

$$
h_k(\theta_\text{ecl}) = (i)^k \cdot \frac{\int_{\theta_S^-}^{\theta_S^+}\!d\theta_S\;
w(\theta_S, \theta_\text{ecl})\,\cos(k A(\theta_S, \theta_\text{ecl}))}
{\int_{\theta_S^-}^{\theta_S^+}\!d\theta_S\;w(\theta_S, \theta_\text{ecl})}
$$

where $\theta_S$ is the **spin-axis ecliptic colatitude** (varies through
$[\pi/2 - \alpha_F, \pi/2 + \alpha_F]$ during precession), $A$ is the
**spherical-triangle vertex angle at the boresight** in the triangle
(ecliptic pole $P$, spin axis $S$, boresight $B$):

$$
\cos A = \frac{\cos\theta_S - \cos\theta_\text{ecl}\,\cos\beta_F}
              {\sin\theta_\text{ecl}\,\sin\beta_F}
$$

and the weight is the precession-Jacobian × spin-Jacobian product:

$$
w(\theta_S, \theta_\text{ecl}) = \frac{\sin\theta_S}{\sqrt{D_\alpha\, D_\beta}}
$$

with

$$
D_\alpha = \sin^2\!\alpha_F - \cos^2\!\theta_S
\quad\text{(precession turning-point Jacobian)}
$$

$$
D_\beta = (\sin\theta_S \sin\beta_F)^2 - (\cos\theta_\text{ecl} - \cos\theta_S\cos\beta_F)^2
\quad\text{(spin-tangent Jacobian)}
$$

Support: $D_\alpha > 0$ AND $D_\beta > 0$ AND spherical-triangle inequality
on $(\theta_S, \theta_\text{ecl}, \beta_F)$.

### Derivation outline

1. Parametrize phase space by $(\phi_y, \phi_p, \phi_s) \in T^3$ (year,
   precession, spin azimuths), uniform under year-long ergodic averaging.
2. SO(2) symmetry around ecliptic pole: $\phi_y$ marginalizes trivially;
   answer depends only on $(\theta_S, \xi)$ where $\xi$ is spin angle in
   the local frame at $S$.
3. Change of variables $(\phi_p, \phi_s) \to (\theta_S, \xi)$ has Jacobian
   $\sin\theta_S / \sqrt{D_\alpha}$ (precession turning-point density).
4. Constraint $\theta_b = \theta_\text{ecl}$ projects $\xi$ onto two
   crossings $\xi_\pm$ per spin, contributing
   $1/|\partial\theta_b/\partial\xi| = \sin\theta_b/\sqrt{D_\beta}$
   (spin-circle tangency density).
5. Combine: conditional density of $\theta_S$ given $\theta_b = \theta_\text{ecl}$
   is $\propto w(\theta_S, \theta_\text{ecl})$.
6. At each $\theta_S$, the two crossings have $\psi_+, \psi_- = \pi - \psi_+$
   (mirror symmetry across the meridian to ecliptic pole), so
   $e^{-ik\psi_+} + e^{-ik\psi_-} = 2 e^{-ik\pi/2} \cos(k\psi_+)$. With
   $\sin\psi_+ = \cos A$, $\cos\psi_+ = \sin\xi_+ \sin\theta_S/\sin\theta_b$,
   the per-spin contribution reduces to $2 e^{-ik\pi/2} \cos(kA)$.
   Average over $\theta_S$ with weight $w$ gives the closed form. The
   $(i)^k$ vs $(-i)^k$ phase choice depends on the $\psi$ sign convention;
   Falcons.jl matches $(i)^k$.

### Singularities

Both $D_\alpha = 0$ and $D_\beta = 0$ are integrable $1/\sqrt{\text{boundary
distance}}$ singularities corresponding to physical tangencies (precession
turning points; spin-circle tangent to the target parallel of constant
$\theta_\text{ecl}$). They contribute finite values to the integral but
**must be handled with adaptive quadrature** — `numpy.trapz` with 20k linearly
spaced points underestimates the boundary contributions by ~5%, propagating
into ~10-50% errors in $\langle\cos kA\rangle$. `scipy.integrate.quad` with
the singularities at the endpoints converges correctly; alternatively a
Chebyshev substitution
$\cos\theta_S = \tfrac{1}{2}(c_- + c_+) + \tfrac{1}{2}(c_+ - c_-)\cos s$
absorbs both singularities into $\sin s$ Jacobian factors and lets uniform
trapezoid in $s$ converge cleanly.

## Validation

### LiteBIRD-standard config

$\alpha_F = 45^\circ$, $\beta_F = 50^\circ$, $T_\text{prec} = 192.348$ min,
$T_\text{spin} = 20$ min, 1 yr at 1 Hz, $N_\text{side} = 128$.

Pre-frozen criterion: $|h_k^\text{Falcons} - h_k^\text{closed-form}| < 0.02$
across the observed sky for $k \in \{1, 2, 4\}$. After fixing the quadrature:

| $\theta_\text{ecl}$ | $k=1$ | $k=2$ | $k=4$ |
|:---:|:---:|:---:|:---:|
| 33° | 0.0010 | 0.0012 | 0.0021 |
| 60° | 0.0006 | 0.0013 | 0.0002 |
| 90° | 0.0023 | 0.0005 | 0.0014 |
| 135° | 0.0009 | 0.0010 | 0.0012 |

Max in bulk ($\theta \in [10^\circ, 170^\circ]$): **0.0078** (at
$\theta = 170^\circ$, $k=4$ — pole-adjacent finite-pixel-binning effect on
Falcons side, not a closed-form failure). **PASS.**

Cross-checked at the integrand level: $\langle\cos kA\rangle$ from the
closed form (scipy.quad) vs from a 30M-sample direct Monte Carlo of
$(\phi_p, \phi_s)$ on $T^2$ agrees to RMS $\le 0.003$.

### Planck-extreme config

$\alpha_F = 7.5^\circ$, $\beta_F = 85^\circ$ (very different geometry:
nearly-equatorial spin axis, near-orthogonal scan circle). Falcons preset
defaults: $T_\text{prec} = 6$ months, $T_\text{spin} = 1$ min.

**Bulk passes; poles fail; root cause is non-ergodicity of the Falcons
Planck preset.**

| $\theta_\text{ecl}$ | $k=1$ | $k=2$ | $k=4$ |
|:---:|:---:|:---:|:---:|
| 10° | 0.20 | 0.31 | 0.10 |
| 30° | 0.012 | 0.007 | 0.025 |
| 50° | 0.0003 | 0.0001 | 0.0002 |
| 70° | 0.006 | 0.001 | 0.003 |
| 90° | 0.004 | 0.001 | 0.002 |
| 130° | 0.004 | 0.001 | 0.002 |
| 150° | 0.013 | 0.015 | 0.048 |
| 170° | 0.13 | 0.17 | 0.26 |

Diagnosis:

1. **Bulk ($50^\circ \le \theta \le 130^\circ$) is excellent**: ≤ 0.005
   absolute, the closed form holds at extreme $\beta_F$ and $\alpha_F$.
2. **Poles fail badly**: up to 0.31 disagreement — and even sign flips for
   $k=2$ at $\theta=10^\circ$ (Falcons $-0.10$ vs closed form $+0.22$).
3. **Falcons output is not azimuthally symmetric**: at $\theta\in[60^\circ,
   80^\circ]$, Re($h_2$) varies from 0.947 to 0.993 in 10° $\phi$ bins
   with a 2-fold symmetry (peaks at $\phi=90^\circ, 270^\circ$). For a
   true year-averaged scan this should be uniform.
4. **Perturbing $T_\text{spin}$ to break commensurability with year and
   precession (set to 1.013 min instead of exact 1 min) does not fix
   pole disagreement.** Bulk gets slightly worse (because the
   integer-min original happened to land near the closed form by
   coincidence), pole errors stay at the same magnitude.

Root cause: Falcons' Planck preset has $T_\text{year}/T_\text{prec}
\approx 2.000$ within 0.07%. Together with $T_\text{spin}/T_\text{prec}
= 1/262{,}980$ (rational at integer minute resolution), the orbit on
$T^3 = (\phi_\text{year}, \phi_\text{prec}, \phi_\text{spin})$ is
quasi-periodic, not ergodic. After 1 year, the trajectory closes
back near its start. The conditional distribution of $\theta_S$ given
$\theta_b = \theta_\text{ecl}$ from Falcons is therefore sampled from
a non-uniform sub-region of the support, biasing $\langle\cos kA\rangle$.

**Pole regions are most sensitive** because the support of $\theta_S$ is
narrow there (e.g. $[82.5^\circ, 95^\circ]$ at $\theta_\text{ecl}=10^\circ$),
so non-uniform sampling within that range has a large effect on the
conditional moment. **Bulk regions hide the effect** because the support
is wide and the empirical sample of $\theta_S$ averages over many
geometric configurations even on a closed orbit.

**LiteBIRD passes by design**: $T_\text{prec} = 192.348$ min and
$T_\text{year} = 525{,}600$ min give $T_\text{year}/T_\text{prec} \approx
2733.04$ (irrational), making the orbit ergodic on $T^2$ ($\phi_\text{year},
\phi_\text{prec}$). The choice of $192.348$ min for the precession period
in the LiteBIRD scan strategy may have been deliberately incommensurate
with the year for exactly this reason — worth checking with Takase or the
LiteBIRD scan-strategy working group.

### Conclusion of validation

The closed form is the **year-averaged ergodic limit**. It applies exactly
when the orbital frequencies are mutually incommensurate (the generic case
for a real mission with detuned periods). Specific commensurate parameter
choices (Falcons' Planck preset is one) give quasi-periodic orbits that
empirically deviate from the ergodic limit by an amount that:

- Vanishes in the bulk where many pixels of similar geometry average
  together (≤ 0.005 in our tests).
- Can be 0.05-0.30 in narrow-support regions like near the ecliptic poles.

For augr design-study use this is the right behavior: the closed form
gives the "design-intent" $h_k$, free of artefacts from any specific
period-commensurability choice. A specific mission that wants to predict
$h_k$ for its as-built scan parameters can either (a) run a TOD
simulation, or (b) use the closed form with a small uncertainty band
attached to pole regions.

## Literature search

**Verdict: the year-averaged closed-form 1-D quadrature appears genuinely
novel as far as a thorough but non-exhaustive search has shown.**

Searched (by literature-researcher subagent + direct PDF read):

- **Wallis, Brown, Taylor 2017** (arXiv:1604.02290, MNRAS 466, 425) —
  the seminal paper introducing $h_k$ for CMB scan-strategy work.
  Eq. 16 defines $h_k$ as a discrete sum over pointing samples within a
  HEALPix pixel. Eqs. 20-22 propagate $h_k$ to B-mode bias. **No
  analytic year-averaged form.**
- **McCallum, Wallis et al. 2021** (arXiv:2008.00011, MNRAS 501, 802) —
  comprehensive spin-formalism follow-up. Eq. 4 defines
  $\tilde h_{k-k'}(\Omega) = (1/N_\text{hits})\sum_j e^{i(k-k')\psi_j}$,
  same discrete-sum construct as Wallis 2017. §4.4.1 runs an EPIC-like
  satellite simulation at exactly our LiteBIRD-like config (boresight
  $50^\circ$, precession $45^\circ$, spin 1 min, precession 3 hr) and
  feeds $h_k$ from TOD into power-spectrum-level analytic predictions
  (Eqs. 56-59, App. B4). **The analytic content lives at the
  power-spectrum level, not at $h_k$ itself.**
- **Takase et al. 2024** (arXiv:2408.03040, JCAP, LiteBIRD scan
  optimization) — extends $h_k$ to include HWP phase but as discrete sum
  (Eq. 3.2). Uses the public Julia simulator Falcons.jl numerically.
- **Falcons.jl** (github.com/yusuke-takase/Falcons.jl) — pure TOD-stream
  Monte Carlo. Spin coefficients $h_{n,m}$ accumulated in
  `function/scanfields.jl:get_scanfield()` over time samples; no analytic
  path.
- **Delabrouille et al. 1998** (astro-ph/9810478) — foundational Planck
  scan-strategy paper. Crossing-angle requirements treated qualitatively;
  no analytic forms.
- **Beam-asymmetry pseudo-$C_\ell$ literature** — Hanson et al. 2010
  (arXiv:1003.0198), FEBeCoP (arXiv:1005.1929), Hivon et al. 2017
  (QuickPol, arXiv:1608.08833), Mitra & Souradeep, Pant et al., etc.
  All treat $h_k$ as an **input summary statistic** computed from maps
  or pointing data, not derived from orbital geometry.
- **Leloup et al. 2024** (arXiv:2312.09001, LiteBIRD far-sidelobes),
  **Tomasi 2025**, **Bortolami 2025** (LBS framework) — fully numerical
  TOD/map pipelines.

**Closest prior work**: Wallis 2017 (Eq. 16) — discrete $h_k$ definition,
no continuous integral over orbital parameters. The split into a
precession-Jacobian factor ($\sin\theta_S/\sqrt{D_\alpha}$) and a
spin-Jacobian factor ($1/\sqrt{D_\beta}$), with the spherical-triangle
angle $A$ in the integrand, does not appear to have been written
explicitly.

**Caveats on novelty claim:**

- Search was non-exhaustive — a long tail of conference proceedings,
  theses, and internal collaboration notes was not covered.
- The mathematical setup is standard (spherical-triangle geometry +
  ergodic phase-space averaging); it's surprising no one has written
  this down, but I haven't ruled it out.
- Before any public claim of novelty: do another targeted search of
  PICO 2018 (Hanany et al. arXiv:1902.10541) collaboration documents,
  CORE concept papers, and the EPIC heritage from Bock et al.
  (referenced in McCallum 2021 §4.4.1). McCallum cites
  "Bock et al. 2009" — that's worth tracking down.
- Worth running by someone in the LiteBIRD scan-strategy working group
  (Takase, Galloni, Krachmalnicoff, Wallis) before submission.

## Files

- `validate_hk.jl` — Falcons driver, LiteBIRD config.
- `validate_hk_planck.jl` — Falcons driver, Planck config.
- `compare_yearavg.py` — closed-form-vs-Falcons comparator (LiteBIRD).
- `mc_truth.py` — direct $T^2$ Monte Carlo, used to ground-truth the
  conditional density and verify the analytic weight.
- `compare_hk_to_envelope.py` — original (failed) single-precession
  envelope comparator. Kept as a record of the false start.
- `data/falcons_validation/` — Falcons FITS outputs + comparison plots.
  Gitignored.

## Open questions / next steps

- **Planck validation** — done; clarified that closed form is ergodic
  limit. Validation passes in bulk, exposes Falcons-Planck preset
  non-ergodicity at poles. Open: confirm with Takase et al. that
  LiteBIRD's $T_\text{prec} = 192.348$ min was chosen for ergodicity.
- **Closed-form derivation in $u = \cos\theta_S$ variable** — the
  substitution moves singularities into elliptic-integral form
  ($\int du/\sqrt{(\text{quadratic in }u)}$); could give a clean
  expression in elliptic integrals of the first kind.
- **Generalization** — the same machinery should work for arbitrary
  HWP phase $\phi$ (extra factor $e^{-im\phi}$ in the per-spin sum,
  averages over the HWP phase if its frequency is incommensurate with
  $T_\text{spin}$). Not derived yet.
- **JAX-differentiable implementation** — `scipy.integrate.quad` is not
  JAX-compatible. Chebyshev substitution + uniform trapezoid in the
  substituted variable is differentiable and converges. Plan to use that
  in `augr/crosslinks.py`.
- **Bock et al. 2009 citation** — McCallum §2 cites Bock et al. 2009 for
  $\tilde h_1, \tilde h_2$ as scan-mitigation diagnostics. Worth tracking
  down to confirm it's not the prior derivation.
