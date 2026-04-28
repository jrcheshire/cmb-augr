# Stop 4 — Off-axis detectors and the 2-D field

## Lift from boresight to off-axis

For an off-axis detector at focal-plane offset `(r, theta_fp)`, the
polarization angle on sky is no longer `-90 + chi + deck` (the r=0
limit). Instead, ``chi2alpha`` reckons the detector's actual sky
position, takes the bearing from there back to the boresight, and adds
``chi`` plus a ``thetaref - theta`` fiducial correction. This bearing
depends on declination but *not* on RA (rotation-invariance, Stop 2),
so

    alpha(dec, r, theta_fp, chi, deck)
        = az(dec, r, theta_fp) - 180 + chi + thetaref - theta

with ``az`` the bearing back to the boresight.

## Surprise: the off-axis correction enters as a uniform phase

The deck angle enters ``alpha`` only through ``thetaref``; the ``az``
calculation does *not* depend on ``thetaref``. So at fixed
``(dec, r, theta_fp, chi)``, varying the deck shifts ``alpha`` by
``thetaref - thetaref_boresight`` — uniformly across all decks.
Therefore

    sum_d w_d exp(-i k alpha_d) at off-axis
        = exp(-i k * (off-axis shift)) * sum_d w_d exp(-i k alpha_d) at boresight,

i.e. the off-axis correction is a *global phase factor*. Consequently,
**for a single detector at lat = -90 deg:**

* ``|h_k|^2`` is set by the deck schedule alone — uniform across
  Dec, RA, and focal-plane position.
* ``arg(h_k)`` carries the focal-plane / Dec geometry; it varies
  smoothly with Dec at finite ``r``.

This is a *stronger* result than the naive Stop-1-prediction
"horizontal stripes": for any one detector the entire 2-D map is
amplitude-flat. The dec-variation lives entirely in the phase.

## Plot

`02_offaxis_map.py` produces `02_offaxis_map.png`: 1-D traces of
``|h_2|^2`` and ``arg(h_2)`` vs. Dec for ``r = 0, 1, 2, 4 deg`` at
``theta_fp = 45 deg``, ``chi = 11 deg``, BK 4-deck schedule. The
amplitude curves overlap at exactly 0.5; the phase fans out by ~5 deg
across the BICEP Dec range at ``r = 2 deg``.

(BK schedule note: with ``decks = {68, 113, 248, 293}``, ``|h_2|^2 =
0.5`` and ``|h_4|^2 = 0`` cleanly. The 180-deg outer pairs cancel odd
``k``; the 45-deg inner spacing puts the four ``4 alpha`` values into
two 180-deg pairs, killing ``h_4``. ``h_2`` falls in between and
survives at ``|h_2| = 1/sqrt(2)``. So differential gain
(``h_2``-coupled per Wallis 2017 Eq. 20) is *not* schedule-suppressed
for BK — it relies on deprojection (`dp1100`) for removal.)

## Design implication: amplitude diversity needs a detector ensemble

If one detector gives uniform ``|h_k|^2`` across the field, where does
amplitude diversity come from? Answer: when you sum h_k contributions
from a focal-plane *ensemble*, each detector at a different
``(r, theta_fp, chi)`` adds with a different phase pattern across Dec.
At each Dec, the ensemble |sum|^2 reflects how those phases interfere
— coherent buildup at some Decs, partial cancellation at others.

This is the right lever for thinking about ensemble-averaged
contamination, and the API supports it directly: vectorize
``h_k_offaxis`` over an array of detectors and sum.

## Public API (this stop)

* ``h_k_offaxis(dec_deg, deck_deg, weights, r_deg, theta_fp_deg, chi_deg, k)``
  — h_k at one or more declinations for a single detector.
* ``h_k_map_southpole(ra_grid_deg, dec_grid_deg, deck_deg, ..., k)``
  — flat-sky 2-D h_k map for a single detector.
* ``southpole_field_mask(ra_grid_deg, dec_grid_deg, ra_min, ra_max,
  dec_min, dec_max)`` — boolean field-bounds mask. Defaults to BICEP CMB
  field.

All JAX-differentiable in numeric arguments.

## What's still to add (Stop 5)

* Detector ensemble averaging — turn the single-detector phase
  pattern into a multi-detector ``<|h_k|^2>`` by summing.
* Per-pixel integration time and per-pixel deck weights from a real
  schedule (`~/bicepkeck/gcp/config/sch/CMB/`).
* Lat = -89.99 deg perturbative correction; check that
  ``|delta h_k| << 0.01``.
* Direct time-domain MC validation against the closed form using a
  parsed schedule.
