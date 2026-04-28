# Stop 5 — Validation, BA schedule, and the lat-offset bound

## Three things to close out

1. The closed form is consistent with a direct time-domain MC.
2. The actual BICEP Array deck schedule is special — it null-suppresses
   every spin moment in Wallis 2017's contamination list.
3. The lat = -89.99 deg perturbative correction is bounded at
   `O(eps^2) ~ 1e-8`, well below MC sampling noise.

## BICEP Array's 8-deck cycle is engineered to null h_k for k ≤ 7

The header of `~/bicepkeck/gcp/config/sch/CMB/9_baCMB_03_000.sch`
documents the schedule:

> BICEP Array observes at 8 dk angles, spaced in even 45 degree
> intervals. The dk angle used for observing rotates from schedule to
> schedule (every two days). The dk angle for each phase can be figured
> out by applying mod 8 to the schedule serial number ...

So one schedule = one deck angle, and over a 16-day cycle the deck
distribution is uniform 1/8 over {23, 68, 113, 158, 203, 248, 293, 338}
deg.

For 8 evenly-spaced decks, the discrete sum

    sum_{n=0}^{7} exp(-i k * (45 deg) * n)

is zero for any `k` not a multiple of 8 — a basic discrete-Fourier
identity. So at the boresight, the BA strategy gives `|h_k| = 0` for
`k ∈ {1, 2, 3, 4, 5, 6, 7}`. Wallis 2017 Eqs. 20-22 identify h_1
(differential pointing), h_2 (differential gain), and h_4 (differential
ellipticity) as the spin moments that contaminate B-modes from
asymmetric-beam systematics. **BA's deck cycle null-suppresses all
three at the boresight, by design.**

The constant `BA_DECK_ANGLES_8` exposes the 8-deck cycle; tests
`test_ba_8deck_preset_nulls_h_k[k]` for `k = 1..7` lock the result in.

## Comparison: deck-schedule design landscape

`03_validation.py` produces `03_validation.png` — bar chart of
`|h_k|^2` vs `k` for four schedules:

| schedule | nulls k = | survives k = |
|----------|-----------|---------------|
| 1 deck                              | (none)              | all (|h_k|=1) |
| 2 decks at 0, 180 deg               | odd                 | even          |
| BICEP 4-deck (68, 113, 248, 293)    | 1, 3, 4, 5, 7, 9    | 2 (0.5), 6 (0.5), 8 (1) |
| BA 8-deck (45 deg step + 23 offset) | 1..7                | 8 (1)         |

The BICEP 4-deck schedule has a 180-deg outer-pair structure that
nulls odd k, plus a 45-deg inner-pair structure that nulls k=4. h_2
falls between the symmetries and survives at half.

## Time-domain MC validation

`tests/test_crosslinks_southpole.py::test_mc_matches_closed_form_at_lat_minus_90`
samples 200k (HA, deck) hits at lat = -90 deg and confirms the
empirical mean of `exp(-i k alpha)` matches the closed form within MC
sampling noise (~5e-3 absolute at this n).

At lat = -90 deg, the MC reduces to per-pixel deck-distribution
sampling because `alpha` is time-independent given the deck (chi2alpha
is RA-invariant and HA-invariant). At lat != -90 deg the parallactic-
angle correction enters per sample, providing a true time-domain
probe.

## The lat = -89.99 deg perturbative bound

The exact parallactic angle is

    tan(eta) = sin(HA) cos(lat) / (sin(lat) cos(dec) - cos(lat) sin(dec) cos(HA)).

At lat = -90 deg exactly, `cos(lat) = 0`, `sin(lat) = -1`, so
`tan(eta) = 0 / (-cos(dec))` and `eta = 180 deg` (modulo conventions)
identically. At lat = -90 + eps, expanding to leading order:

    eta ≈ 180 deg + O(eps * sin(HA) / cos(dec)).

For `eps = 0.01 deg ~ 1.7e-4 rad` (MAPO offset from the geographic
South Pole), the leading term has amplitude `< 0.02 deg` over the BICEP
field. *Year-averaged*, this leading-order term integrates to zero over
any HA window symmetric around culmination (`<sin(HA)> = 0`). So the
first non-vanishing correction to `<exp(-i k alpha)>` is `O(eps^2)`:
expanding the Bessel-function average,

    <exp(i u sin(HA))>_HA ≈ J_0(u) ≈ 1 - u^2/4 + ...

with `u = k * eps / cos(dec) ~ 1e-3` for k=4, giving
`|delta h_k|/|h_k| ~ 3e-7`.

Locked in by tests `test_lat_offset_correction_is_negligible` (MC
diff between lat = -90 and lat = -89.99 < 1e-4 at 200k samples) and
`test_parallactic_deviation_*` (the deviation function itself).

**Conclusion**: the lat = -90 deg approximation in the closed form
is excellent for MAPO. No `lat_deg` knob added to the public API.

## What lives in this stop's tests

* `BA_DECK_ANGLES_8` constant + tests verifying the 8-deck nulling.
* `_hk_mc_time_domain` (test-private utility) + `_parallactic_deviation_deg`.
* `test_mc_matches_closed_form_at_lat_minus_90`,
  `test_lat_offset_correction_is_negligible`,
  `test_parallactic_deviation_zero_at_lat_minus_90`,
  `test_parallactic_deviation_scales_with_eps`.

## Future work (genuinely out of scope here)

* Real `.sch` parser to recover per-pixel integration time once a
  realistic hit map is needed (e.g., for sky-averaged
  `<|h_k|^2>` weighted by survey footprint).
* Detector ensemble: focal-plane summation `sum_d h_k(detector d)` to
  pick up the phase-interference effects at off-axis (Stop 4 setup
  already supports this; just plumbing).
* Tier 2.5 sidelobe x non-uniform-FG bias propagation -- separate
  effort, h_k factorization breaks for asymmetric sidelobes.
