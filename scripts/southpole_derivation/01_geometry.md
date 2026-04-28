# Stop 1 — South Pole geometry

Goal: derive `α(boresight, deck) = const + deck` at lat = −90°, with no
calculus.

## Setting

MAPO is at lat ≈ −89.99°. We treat it as exactly −90° to leading order;
the perturbative correction is bounded in Stop 5. The BICEP CMB field is
RA ∈ [−60°, +60°], Dec ∈ [−73°, −38°]. The instrument runs fixed-elevation
azimuth scans, with multiple deck angles between schedules.

## Two geometric facts

### Fact 1 — `alt = −δ`

From `sin(alt) = sin(φ) sin(δ) + cos(φ) cos(δ) cos(HA)` at φ = −90°, the
`cos(φ)` term dies and we get `sin(alt) = −sin(δ)`, i.e. `alt = −δ`. A
celestial object at fixed `(RA, Dec)` traces a horizontal diurnal circle
in the local sky at constant altitude. Fixed-elevation scans therefore
select fixed declination strips; HA only sets the azimuth at which the
boresight reaches a given pixel, not the elevation.

### Fact 2 — parallactic angle is frozen

Parallactic angle is the angle, at a sky pixel, between the great circle
from the pixel to the local zenith and the great circle from the pixel to
the celestial north pole.

At lat = −90° the local zenith *is* the south celestial pole. The great
circle from any pixel to the zenith is the pixel's own meridian. The
great circle from the pixel to the NCP is the same meridian, traversed in
the opposite direction. Parallactic angle = 180° at every pixel, every
time, independent of RA, HA, LST, azimuth, or season.

## Consequence

The only instrument-side rotation of the polarization-sensitive axis on
sky is the **telescope deck angle**. For the boresight (focal-plane radius
`r = 0`), the polarization angle on sky reduces to

    α(boresight, deck) = const + deck

with the constant absorbing the polarization fiducial (`χ`) and zero-point
conventions. The BK pipeline encodes the same fact in `chi2alpha.m:10`:

> "Polarization angle on sky for a given deck angle is a function of
> declination only, so only compute for unique declinations."

(Even simpler at the boresight: declination only enters through off-axis
focal-plane corrections, treated in Stop 2.)

## h_k at the boresight

The year-averaged crosslink coefficient `h_k = ⟨exp(−i k α)⟩_hits` becomes
a discrete weighted sum over the deck distribution `{w_d}`:

    h_k(boresight) = exp(−i k · const) · Σ_d w_d exp(−i k · d)

No orbital phase-space integral, no precession Jacobian, no spin-tangency
Jacobian, no Chebyshev quadrature. Just a finite weighted sum over the
deck angles in the schedule.

## Forward pointers

* **Stop 2** — read `chi2alpha.m` line by line; pin the constant from
  conventions; add the off-axis-detector correction; port to JAX.
* **Stop 3** — implement `h_k_southpole_boresight(deck_distribution, k)`;
  validate against a 5-line direct MC.
* **Stop 4** — lift to off-axis detectors and a 2-D BICEP-field map,
  with a `southpole_hit_map` for integration-time weighting.
* **Stop 5** — direct time-domain MC validation against a real BICEP
  schedule from `~/bicepkeck/gcp/config/sch/CMB/`; bound the lat = −89.99°
  perturbative correction.

## Diagram

`01_geometry_diagram.py` produces `01_pole_sky.png`: the celestial sphere
viewed from the pole's perspective (polar projection, `r = 90° + δ = alt`),
showing diurnal circles, the BICEP CMB field, and a representative
boresight pointing at fixed elevation with a deck rotation arc.
