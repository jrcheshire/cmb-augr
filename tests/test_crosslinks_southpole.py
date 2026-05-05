"""Tests for ``augr.crosslinks_southpole``."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr._chi2alpha import chi2alpha
from augr.crosslinks_southpole import (
    BA_DECK_ANGLES_8,
    h_k_boresight,
    h_k_map_southpole,
    h_k_offaxis,
    southpole_field_mask,
)


# -----------------------------------------------------------------------------
# Test helpers (Stop 5 time-domain MC).
# -----------------------------------------------------------------------------


def _parallactic_deviation_deg(ha_deg, dec_deg, lat_deg):
    """Deviation of the true parallactic angle from 180 deg at lat != -90 deg.

    Standard formula: tan(eta) = sin(HA) cos(lat) /
                                 (sin(lat) cos(dec) - cos(lat) sin(dec) cos(HA))
    At lat = -90 deg exactly, eta = 180 deg everywhere (the SCP-aligned-with-
    zenith fact from Stop 1). For lat = -90 + eps, eta wobbles around 180 deg
    by an amount that's leading-order eps * sin(HA) / cos(dec).
    """
    ha = np.deg2rad(ha_deg)
    dec = np.deg2rad(dec_deg)
    lat = np.deg2rad(lat_deg)
    num = np.sin(ha) * np.cos(lat)
    den = np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(ha)
    eta = np.rad2deg(np.arctan2(num, den))
    # eta is wrapped to (-180, 180]; deviation from 180 deg:
    return ((eta - 180.0 + 180.0) % 360.0) - 180.0


def _hk_mc_time_domain(
    dec_deg, deck_distribution, deck_weights, ha_window_deg,
    r_deg, theta_fp_deg, chi_deg, lat_deg, k, n_samples, rng,
):
    """Empirical h_k at one pixel via time-domain (HA, deck) sampling.

    At lat = -90 deg the parallactic correction is identically zero and this
    reduces to the per-pixel deck-distribution MC of Stop 3. At
    lat = -90 + eps the correction is included, so the closed-form-vs-MC
    comparison probes the lat-offset bound."""
    ha = rng.uniform(ha_window_deg[0], ha_window_deg[1], size=n_samples)
    deck_idx = rng.choice(len(deck_distribution), p=deck_weights, size=n_samples)
    decks = np.asarray(deck_distribution)[deck_idx]
    alphas = np.array([
        float(chi2alpha(0.0, dec_deg, r_deg, theta_fp_deg, chi_deg, float(d)))
        for d in decks
    ])
    if lat_deg != -90.0:
        alphas = alphas + _parallactic_deviation_deg(ha, dec_deg, lat_deg)
    return np.mean(np.exp(-1j * k * np.deg2rad(alphas)))


# =============================================================================
# Boresight tests (Stop 3)
# =============================================================================


# -- single deck: closed form match --------------------------------------------

@pytest.mark.parametrize("deck", [0.0, 45.0, 113.0, -68.0, 200.0])
@pytest.mark.parametrize("chi", [0.0, 11.0, -45.0])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_single_deck_closed_form(deck, chi, k):
    """Single-deck h_k = exp(-i k alpha)."""
    decks = jnp.array([deck])
    actual = complex(h_k_boresight(decks, chi_deg=chi, k=k))
    alpha_rad = np.deg2rad(-90.0 + chi + deck)
    expected = np.exp(-1j * k * alpha_rad)
    assert np.isclose(actual, expected, atol=1e-12)


# -- BK pipeline cross-check via chi2alpha -------------------------------------

def test_consistency_with_chi2alpha_r0():
    """h_k_boresight must use the same alpha as chi2alpha at r=0."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    chi = 17.0
    k = 2

    # chi2alpha at r=0 with thetaref = each deck angle.
    alpha_per_deck = jnp.array([
        float(chi2alpha(0.0, -55.0, 0.0, 0.0, chi, float(d))) for d in decks
    ])
    expected = complex(np.mean(np.exp(-1j * k * np.deg2rad(np.asarray(alpha_per_deck)))))

    actual = complex(h_k_boresight(decks, chi_deg=chi, k=k))
    assert np.isclose(actual, expected, atol=1e-12)


# -- symmetry: 4 evenly spaced decks kill k = 1, 2, 3 --------------------------

@pytest.mark.parametrize("k,expected_zero", [(1, True), (2, True), (3, True), (4, False)])
def test_uniform_4deck_symmetry(k, expected_zero):
    """4 equally-weighted decks at {0, 90, 180, 270} null h_1, h_2, h_3 but
    leave |h_4| = 1 (all four 4*alpha values land on the same angle mod 2 pi)."""
    decks = jnp.array([0.0, 90.0, 180.0, 270.0])
    h = complex(h_k_boresight(decks, chi_deg=0.0, k=k))
    if expected_zero:
        assert abs(h) < 1e-12, f"k={k}: expected ~0, got {h}"
    else:
        assert np.isclose(abs(h), 1.0, atol=1e-12), f"k={k}: expected |h|=1, got {abs(h)}"


# -- Schwarz inequality --------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_modulus_bounded_by_one(k):
    """|h_k| <= 1 for any deck distribution (Schwarz / Jensen)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 12))
        decks = jnp.asarray(rng.uniform(-180, 180, size=n))
        weights = jnp.asarray(rng.uniform(0.1, 5.0, size=n))
        h = complex(h_k_boresight(decks, weights=weights, k=k))
        assert abs(h) <= 1.0 + 1e-12, f"|h|={abs(h)} > 1"


# -- chi acts as a pure phase --------------------------------------------------

@pytest.mark.parametrize("chi", [0.0, 17.0, -45.0, 90.0])
def test_chi_is_pure_phase(chi):
    """|h_k| is invariant under chi; chi only rotates the complex phase."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    h0 = complex(h_k_boresight(decks, chi_deg=0.0, k=2))
    hc = complex(h_k_boresight(decks, chi_deg=chi, k=2))
    assert np.isclose(abs(h0), abs(hc), atol=1e-12)
    # Phase shift should be exactly -2 * chi (in degrees), modulo 360.
    if abs(h0) > 1e-12:
        dphi_deg = (np.angle(hc, deg=True) - np.angle(h0, deg=True)) % 360.0
        expected = (-2.0 * chi) % 360.0
        assert np.isclose(dphi_deg, expected, atol=1e-9)


# -- weights normalisation -----------------------------------------------------

def test_weights_normalisation():
    """Unnormalised weights must give the same result as normalised ones."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    raw_weights = jnp.array([2.0, 3.0, 5.0, 7.0])
    norm_weights = raw_weights / raw_weights.sum()
    h_raw = complex(h_k_boresight(decks, weights=raw_weights, k=2))
    h_norm = complex(h_k_boresight(decks, weights=norm_weights, k=2))
    assert np.isclose(h_raw, h_norm, atol=1e-12)


# -- 5-line direct MC validation ----------------------------------------------

def test_against_direct_mc():
    """Closed form vs. drawing samples from the deck distribution and
    averaging exp(-i k alpha) directly. The two should agree to MC
    sampling noise (~1/sqrt(N))."""
    decks = np.array([68.0, 113.0, 248.0, 293.0])
    weights_raw = np.array([2.0, 3.0, 5.0, 7.0])
    weights = weights_raw / weights_raw.sum()
    chi = 11.0
    k = 2

    # Direct MC: draw N samples from the discrete deck distribution, compute
    # alpha at each via the chi2alpha r=0 closed form, accumulate the
    # exp(-i k alpha) average.
    rng = np.random.default_rng(0)
    n = 200_000
    sampled_decks = rng.choice(decks, size=n, p=weights)
    alpha_rad = np.deg2rad(-90.0 + chi + sampled_decks)
    h_mc = np.mean(np.exp(-1j * k * alpha_rad))

    h_closed = complex(h_k_boresight(jnp.asarray(decks),
                                     weights=jnp.asarray(weights),
                                     chi_deg=chi, k=k))

    # Discrete-distribution MC has zero bias; the only error is from finite N.
    # For 200k samples and |h| <= 1, the per-sample variance is at most 1, so
    # rms error ~ 1/sqrt(N) ~ 2e-3.
    assert abs(h_mc - h_closed) < 5e-3, (
        f"closed form {h_closed} vs MC {h_mc}: |diff|={abs(h_mc - h_closed):.3e}"
    )


# -- JAX differentiability ----------------------------------------------------

def test_grad_through_weights():
    """|h_k|^2 should be jax.grad-able through weights."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])

    def loss(weights):
        h = h_k_boresight(decks, weights=weights, chi_deg=0.0, k=2)
        return jnp.real(h * jnp.conj(h))

    g = jax.grad(loss)(jnp.array([2.0, 3.0, 5.0, 7.0]))
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


def test_grad_through_decks():
    """|h_k|^2 differentiable in deck angles (use case: schedule design)."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])

    def loss(d):
        h = h_k_boresight(d, chi_deg=0.0, k=4)
        return jnp.real(h * jnp.conj(h))

    g = jax.grad(loss)(decks)
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


# -- input validation ----------------------------------------------------------

def test_k_must_be_positive_integer():
    decks = jnp.array([0.0, 90.0])
    with pytest.raises(ValueError):
        h_k_boresight(decks, k=0)
    with pytest.raises(ValueError):
        h_k_boresight(decks, k=-1)
    with pytest.raises(ValueError):
        h_k_boresight(decks, k=1.5)


# =============================================================================
# Off-axis and 2-D map tests (Stop 4)
# =============================================================================

# -- r=0 reduces to boresight -------------------------------------------------

@pytest.mark.parametrize("dec", [-90.0, -73.0, -55.0, -38.0])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_offaxis_r0_matches_boresight(dec, k):
    """h_k_offaxis(r=0) == h_k_boresight at any declination."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    chi = 11.0
    h_off = complex(h_k_offaxis(dec, decks, chi_deg=chi, r_deg=0.0, k=k))
    h_bs = complex(h_k_boresight(decks, chi_deg=chi, k=k))
    assert np.isclose(h_off, h_bs, atol=1e-12)


# -- consistency with chi2alpha applied per-deck -------------------------------

def test_offaxis_consistency_with_chi2alpha():
    """h_k_offaxis must match a manual chi2alpha-loop reconstruction."""
    dec = -55.0
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    chi = 17.0
    r = 2.0
    theta_fp = 45.0
    k = 2

    alphas = jnp.array([
        float(chi2alpha(0.0, dec, r, theta_fp, chi, float(d))) for d in decks
    ])
    expected = complex(np.mean(np.exp(-1j * k * np.deg2rad(np.asarray(alphas)))))

    actual = complex(h_k_offaxis(dec, decks, r_deg=r, theta_fp_deg=theta_fp,
                                 chi_deg=chi, k=k))
    assert np.isclose(actual, expected, atol=1e-12)


# -- vectorization over dec ---------------------------------------------------

def test_offaxis_vectorized_over_dec():
    decs = jnp.linspace(-73.0, -38.0, 8)
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    out = h_k_offaxis(decs, decks, r_deg=2.0, theta_fp_deg=45.0,
                      chi_deg=11.0, k=2)
    assert out.shape == (8,)
    # Each element should match a scalar call at that dec.
    for i, dec in enumerate(decs):
        expected = complex(h_k_offaxis(float(dec), decks, r_deg=2.0,
                                       theta_fp_deg=45.0, chi_deg=11.0, k=2))
        assert np.isclose(complex(out[i]), expected, atol=1e-12)


# -- continuity in r at r=0 ----------------------------------------------------

def test_offaxis_continuous_at_r0():
    """At small r, h_k_offaxis approaches the r=0 (boresight) value."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    h_0 = complex(h_k_offaxis(-55.0, decks, r_deg=0.0, theta_fp_deg=45.0))
    h_eps = complex(h_k_offaxis(-55.0, decks, r_deg=1e-5, theta_fp_deg=45.0))
    assert np.isclose(h_0, h_eps, atol=1e-6)


# -- 2-D map: RA-invariance ---------------------------------------------------

def test_map_ra_invariance():
    """Each row of the map (constant Dec) should be identical across RA."""
    ra_grid = jnp.linspace(-60.0, 60.0, 12)
    dec_grid = jnp.linspace(-73.0, -38.0, 8)
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    m = h_k_map_southpole(ra_grid, dec_grid, decks, r_deg=2.0,
                          theta_fp_deg=45.0, chi_deg=11.0, k=2)
    assert m.shape == (12, 8)
    # All rows equal the first row.
    for i in range(m.shape[0]):
        assert np.allclose(m[i, :], m[0, :], atol=1e-15)


def test_map_dec_dependence_at_boresight_is_flat():
    """At r=0 the map is constant in Dec too (boresight closed form is
    independent of Dec)."""
    ra_grid = jnp.linspace(-60.0, 60.0, 6)
    dec_grid = jnp.linspace(-73.0, -38.0, 6)
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    m = h_k_map_southpole(ra_grid, dec_grid, decks, r_deg=0.0)
    assert np.allclose(m, m[0, 0], atol=1e-15)


def test_map_phase_varies_with_dec_offaxis():
    """At finite r the complex h_k varies along Dec (the off-axis effect)
    -- in phase, not in amplitude (see test_offaxis_amplitude_invariance)."""
    ra_grid = jnp.linspace(-60.0, 60.0, 6)
    dec_grid = jnp.linspace(-73.0, -38.0, 6)
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    m = h_k_map_southpole(ra_grid, dec_grid, decks, r_deg=2.0,
                          theta_fp_deg=45.0, chi_deg=11.0, k=2)
    # First-vs-last Dec complex values should differ.
    assert not np.allclose(m[0, 0], m[0, -1], atol=1e-6)


# =============================================================================
# Stop 5 -- BA 8-deck preset and lat-offset validation.
# =============================================================================


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7])
def test_ba_8deck_preset_nulls_h_k(k):
    """BA's 8-deck schedule (45 deg evenly-spaced) nulls h_k for k = 1..7.

    Header of ``~/bicepkeck/gcp/config/sch/CMB/9_baCMB_03_000.sch`` documents
    the 8-deck cycle. The discrete sum ``sum_n exp(-i k * 45 deg * n)`` over
    n in {0..7} vanishes for any k not a multiple of 8 -- so the BA strategy
    null-suppresses every spin moment in Wallis 2017's contamination list
    (h_1, h_2, h_4) at the boresight, plus h_3, h_5, h_6, h_7 for free."""
    decks = jnp.asarray(BA_DECK_ANGLES_8)
    h = complex(h_k_boresight(decks, chi_deg=11.0, k=k))
    assert abs(h) < 1e-12, f"k={k}: |h|={abs(h):.3e}, expected ~0"


def test_ba_8deck_preset_nonzero_at_h8():
    """Sanity: at k=8, all 4*alpha values land on the same angle, h_8 = 1."""
    decks = jnp.asarray(BA_DECK_ANGLES_8)
    h = complex(h_k_boresight(decks, chi_deg=0.0, k=8))
    assert np.isclose(abs(h), 1.0, atol=1e-12)


# -- time-domain MC at lat = -90 deg matches closed form -----------------------

@pytest.mark.slow
def test_mc_matches_closed_form_at_lat_minus_90():
    """At lat = -90 deg exactly, the parallactic correction is zero and the
    time-domain MC reduces to the deck-distribution MC; should match the
    closed form within sampling noise."""
    rng = np.random.default_rng(0)
    decks = list(BA_DECK_ANGLES_8)
    weights = [1.0 / len(decks)] * len(decks)
    n = 200_000

    # Pick k=8 so the closed form is non-zero and the MC has a target to
    # compare against. (For k=1..7 the closed form is zero and the MC will
    # also be ~0; the test still passes, but is less informative.)
    h_mc = _hk_mc_time_domain(
        dec_deg=-55.0, deck_distribution=decks, deck_weights=weights,
        ha_window_deg=(-60.0, 60.0), r_deg=2.0, theta_fp_deg=45.0,
        chi_deg=11.0, lat_deg=-90.0, k=8, n_samples=n, rng=rng,
    )
    h_closed = complex(h_k_offaxis(
        -55.0, jnp.asarray(decks), r_deg=2.0, theta_fp_deg=45.0,
        chi_deg=11.0, k=8,
    ))
    # Discrete-distribution MC sampling noise: 1/sqrt(N).
    assert abs(h_mc - h_closed) < 5e-3, f"diff={abs(h_mc - h_closed):.3e}"


# -- lat = -89.99 deg deviation is bounded by O(eps^2) -------------------------

@pytest.mark.slow
def test_lat_offset_correction_is_negligible():
    """For lat = -89.99 deg (eps ~ 1.7e-4 rad), a symmetric HA scan window
    averages the leading-order eps*sin(HA) correction to zero, leaving an
    O(eps^2) residual -- well below MC sampling noise."""
    rng = np.random.default_rng(0)
    decks = list(BA_DECK_ANGLES_8)
    weights = [1.0 / len(decks)] * len(decks)
    n = 200_000

    h_polar = _hk_mc_time_domain(
        -55.0, decks, weights, (-60.0, 60.0),
        r_deg=2.0, theta_fp_deg=45.0, chi_deg=11.0,
        lat_deg=-90.0, k=8, n_samples=n, rng=rng,
    )
    rng2 = np.random.default_rng(0)  # match HA / deck samples for cleaner diff
    h_mapo = _hk_mc_time_domain(
        -55.0, decks, weights, (-60.0, 60.0),
        r_deg=2.0, theta_fp_deg=45.0, chi_deg=11.0,
        lat_deg=-89.99, k=8, n_samples=n, rng=rng2,
    )
    # Same HA samples, only the lat changes -> diff is purely the lat
    # correction. Bound: |diff| << MC noise of either estimate.
    diff = abs(h_polar - h_mapo)
    assert diff < 1e-4, (
        f"|h(-89.99) - h(-90)| = {diff:.3e}, expected << 1e-3 (MC noise)"
    )


# -- parallactic deviation: sanity at lat = -90 deg ----------------------------

def test_parallactic_deviation_zero_at_lat_minus_90():
    """At lat = -90 deg exactly, the deviation must be 0 at all HA, dec."""
    has = np.array([-60.0, -10.0, 0.0, 17.0, 60.0])
    decs = np.array([-73.0, -55.0, -38.0])
    for ha in has:
        for dec in decs:
            d = _parallactic_deviation_deg(ha, dec, -90.0)
            assert abs(d) < 1e-10, f"ha={ha} dec={dec}: deviation={d}"


def test_parallactic_deviation_scales_with_eps():
    """Magnitude of the deviation at small eps matches the leading-order
    bound |eps * sin(HA) / cos(dec)|. Sign depends on the convention for
    "polarization-angle increment direction" in chi2alpha and only flips
    the phase of h_k coherently across all pixels -- it does not affect
    |h_k| or the lat-offset bound."""
    ha = 30.0
    dec = -55.0
    eps_deg = 0.01  # lat = -89.99
    d = _parallactic_deviation_deg(ha, dec, -90.0 + eps_deg)
    expected_mag = abs(np.rad2deg(np.deg2rad(eps_deg) *
                                  np.sin(np.deg2rad(ha)) / np.cos(np.deg2rad(dec))))
    assert np.isclose(abs(d), expected_mag, rtol=1e-2), \
        f"got |d|={abs(d)}, expected ~{expected_mag}"


def test_offaxis_amplitude_invariance():
    """|h_k|^2 is invariant under (r, theta_fp, dec) for a single detector.

    The off-axis correction in chi2alpha is a deck-uniform shift in
    alpha (chi2alpha's az calculation does not depend on thetaref), so
    the off-axis h_k differs from the boresight h_k by a pure phase
    factor. Amplitude is identical at any focal-plane offset and any
    declination."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    chi = 11.0
    h_bs = complex(h_k_boresight(decks, chi_deg=chi, k=2))
    amp2_bs = abs(h_bs) ** 2

    # Sample a 2-D grid of (dec, r) and a couple theta_fp values.
    for dec in (-73.0, -55.0, -38.0):
        for r in (0.5, 2.0, 5.0):
            for theta_fp in (0.0, 45.0, 137.0):
                h = complex(h_k_offaxis(dec, decks, r_deg=r,
                                        theta_fp_deg=theta_fp, chi_deg=chi, k=2))
                assert np.isclose(abs(h) ** 2, amp2_bs, atol=1e-12), (
                    f"dec={dec} r={r} theta_fp={theta_fp}: "
                    f"|h|^2={abs(h)**2}, expected {amp2_bs}"
                )


# -- field mask shape and bounds ----------------------------------------------

def test_field_mask_shape_and_defaults():
    ra = jnp.linspace(-90.0, 90.0, 19)  # 10 deg spacing
    dec = jnp.linspace(-90.0, 0.0, 10)  # 10 deg spacing
    mask = southpole_field_mask(ra, dec)
    assert mask.shape == (19, 10)
    # Pixel at (ra=0, dec=-55): in default BK field.
    assert bool(mask[9, 4])  # ra_index 9 = 0 deg, dec_index 4 = -50 deg
    # Pixel at (ra=80, dec=-55): outside RA range.
    assert not bool(mask[17, 4])
    # Pixel at (ra=0, dec=-90): outside Dec range.
    assert not bool(mask[9, 0])


def test_field_mask_custom_bounds():
    ra = jnp.array([-30.0, 0.0, 30.0])
    dec = jnp.array([-60.0, -50.0, -40.0])
    mask = southpole_field_mask(ra, dec,
                                ra_min=-20.0, ra_max=20.0,
                                dec_min=-55.0, dec_max=-45.0)
    expected = np.array([
        [False, False, False],   # ra=-30: out of [-20, 20]
        [False, True,  False],   # ra=0: in. dec must be in [-55, -45]
        [False, False, False],   # ra=30: out
    ])
    assert np.array_equal(np.asarray(mask), expected)


# -- JAX differentiability through the off-axis path --------------------------

def test_grad_offaxis_through_r():
    """dh/dr should be finite and well-defined for r > 0."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])

    def loss(r):
        h = h_k_offaxis(-55.0, decks, r_deg=r, theta_fp_deg=45.0,
                        chi_deg=0.0, k=2)
        return jnp.real(h * jnp.conj(h))

    g = jax.grad(loss)(jnp.float64(2.0))
    assert jnp.isfinite(g)


def test_grad_map_through_chi():
    """Gradient through h_k_map_southpole w.r.t. chi_deg should run."""
    ra = jnp.linspace(-60.0, 60.0, 6)
    dec = jnp.linspace(-73.0, -38.0, 6)
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])

    def loss(chi):
        m = h_k_map_southpole(ra, dec, decks, r_deg=2.0, theta_fp_deg=45.0,
                              chi_deg=chi, k=2)
        return jnp.real(jnp.sum(m * jnp.conj(m)))

    g = jax.grad(loss)(jnp.float64(11.0))
    assert jnp.isfinite(g)


# -- sanity: |h_k| <= 1 in the map ---------------------------------------------

def test_map_modulus_bounded():
    ra = jnp.linspace(-60.0, 60.0, 6)
    dec = jnp.linspace(-73.0, -38.0, 6)
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    for k in (1, 2, 4):
        m = h_k_map_southpole(ra, dec, decks, r_deg=2.0, theta_fp_deg=45.0,
                              chi_deg=11.0, k=k)
        assert np.all(np.abs(m) <= 1.0 + 1e-12)
