"""Tests for instrument.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.bandpass import Bandpass
from augr.instrument import (
    ARCMIN_TO_RAD,
    L2_EFFICIENCY,
    SECONDS_PER_YEAR,
    Channel,
    Instrument,
    ScalarEfficiency,
    beam_bl,
    channel_bandpass,
    deconvolve_noise_bb,
    noise_nl,
    noise_nl_matrix,
    white_noise_power,
)


def _channel(**kw):
    base = dict(nu_ghz=150.0, n_detectors=100, net_per_detector=300.0,
                beam_fwhm_arcmin=20.0)
    base.update(kw)
    return Channel(**base)


def test_channel_defaults_monochromatic():
    """New fields default to the monochromatic (band-center) limit."""
    ch = _channel()
    assert ch.fractional_bandwidth == 0.0
    assert ch.bandpass is None
    assert channel_bandpass(ch) is None


def test_channel_bandpass_tophat_from_fractional_bandwidth():
    """A positive fractional bandwidth yields a top-hat centered on the band."""
    ch = _channel(fractional_bandwidth=0.25)
    bp = channel_bandpass(ch)
    assert bp is not None
    assert not bp.is_monochromatic
    assert abs(float(bp.nu_center_ghz) - 150.0) < 1e-9


def test_channel_bandpass_explicit_profile_wins():
    """An explicit Bandpass overrides the fractional-bandwidth top-hat."""
    prof = Bandpass.from_profile(
        jnp.array([140.0, 150.0, 160.0]), jnp.array([1.0, 2.0, 1.0])
    )
    ch = _channel(fractional_bandwidth=0.25, bandpass=prof)
    assert channel_bandpass(ch) is prof

# -----------------------------------------------------------------------
# ScalarEfficiency
# -----------------------------------------------------------------------

def test_efficiency_default_matches_l2():
    """Empty-arg ``ScalarEfficiency()`` must match the public ``L2_EFFICIENCY``
    preset, so a ``Channel`` constructed without explicit ``efficiency=``
    silently picks up the documented L2 baseline rather than a different,
    looser set of factors."""
    eff = ScalarEfficiency()
    expected = 0.85 * 0.85 * 0.90 * 0.97 * 0.95   # L2 preset, eta_total = 0.711
    assert abs(eff.total - expected) < 1e-6
    assert eff == L2_EFFICIENCY


def test_efficiency_all_ones():
    """All factors at 1.0 gives total = 1.0."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    assert eff.total == 1.0


def test_efficiency_frozen():
    """ScalarEfficiency is immutable."""
    eff = ScalarEfficiency()
    with pytest.raises(AttributeError):
        eff.detector_yield = 0.5


# -----------------------------------------------------------------------
# Channel and Instrument are frozen
# -----------------------------------------------------------------------

def test_channel_frozen():
    ch = Channel(nu_ghz=150.0, n_detectors=1000,
                 net_per_detector=300.0, beam_fwhm_arcmin=30.0)
    with pytest.raises(AttributeError):
        ch.nu_ghz = 200.0


def test_instrument_frozen():
    ch = Channel(nu_ghz=150.0, n_detectors=1000,
                 net_per_detector=300.0, beam_fwhm_arcmin=30.0)
    inst = Instrument(channels=(ch,))
    with pytest.raises(AttributeError):
        inst.f_sky = 0.5


# -----------------------------------------------------------------------
# White noise power — hand calculation
# -----------------------------------------------------------------------

def test_white_noise_hand_calc():
    """Verify w⁻¹ against an explicit hand calculation.

    Setup: NET_det = 300 μK√s, N_det = 1000, f_sky = 0.7,
           mission = 5 yr, η_total = 1.0 (perfect efficiency).

    w⁻¹ = (300 × √2)² × 4π × 0.7 / (1000 × 1.0 × 5 × 31557600)
         = 180000 × 8.7964 / 1.57788e11
         = 1.583e6 / 1.57788e11
         = 1.003e-5 μK² sr
    """
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    ch = Channel(nu_ghz=150.0, n_detectors=1000,
                 net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                 efficiency=eff)
    w_inv = float(white_noise_power(ch, mission_years=5.0, f_sky=0.7))

    # Hand calculation
    t_obs = 5.0 * SECONDS_PER_YEAR
    net_pol_sq = (300.0 * np.sqrt(2)) ** 2  # 180000
    omega = 4.0 * np.pi * 0.7
    expected = net_pol_sq * omega / (1000 * 1.0 * t_obs)

    assert abs(w_inv - expected) / expected < 1e-6


def test_white_noise_scales_with_ndet():
    """Doubling N_det halves the noise power."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    ch1 = Channel(nu_ghz=150.0, n_detectors=1000,
                  net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                  efficiency=eff)
    ch2 = Channel(nu_ghz=150.0, n_detectors=2000,
                  net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                  efficiency=eff)
    w1 = float(white_noise_power(ch1, 5.0, 0.7))
    w2 = float(white_noise_power(ch2, 5.0, 0.7))
    assert abs(w2 / w1 - 0.5) < 1e-6


def test_white_noise_scales_with_efficiency():
    """Halving one efficiency factor doubles noise power."""
    eff_full = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    eff_half = ScalarEfficiency(0.5, 1.0, 1.0, 1.0, 1.0)
    ch1 = Channel(nu_ghz=150.0, n_detectors=1000,
                  net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                  efficiency=eff_full)
    ch2 = Channel(nu_ghz=150.0, n_detectors=1000,
                  net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                  efficiency=eff_half)
    w1 = float(white_noise_power(ch1, 5.0, 0.7))
    w2 = float(white_noise_power(ch2, 5.0, 0.7))
    assert abs(w2 / w1 - 2.0) < 1e-6


# -----------------------------------------------------------------------
# Beam transfer function
# -----------------------------------------------------------------------

def test_beam_bl_at_ell_zero():
    """B_0 = exp(0) = 1 for any beam size."""
    bl = beam_bl(jnp.array([0.0]), fwhm_arcmin=30.0)
    assert abs(float(bl[0]) - 1.0) < 1e-6


def test_beam_bl_decreases():
    """Beam transfer function decreases with ℓ."""
    ells = jnp.arange(2, 500, dtype=float)
    bl = beam_bl(ells, fwhm_arcmin=30.0)
    assert jnp.all(bl[1:] <= bl[:-1])


def test_beam_bl_wider_beam_more_suppression():
    """Wider beam suppresses high ℓ more."""
    ells = jnp.array([200.0])
    bl_narrow = float(beam_bl(ells, fwhm_arcmin=5.0)[0])
    bl_wide = float(beam_bl(ells, fwhm_arcmin=30.0)[0])
    assert bl_narrow > bl_wide


def test_beam_sigma_value():
    """Check σ_beam for a 30 arcmin FWHM beam.
    σ = 30' × (π/10800) / √(8 ln 2) = 8.727e-3 / 2.3548 = 3.706e-3 rad."""
    fwhm_rad = 30.0 * float(ARCMIN_TO_RAD)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    assert abs(sigma - 3.706e-3) < 1e-5


def test_beam_bl_p1_byte_identical_to_gaussian():
    """p=1 reproduces the plain Gaussian beam exactly (the back-compat default)."""
    ells = jnp.arange(0, 600, dtype=float)
    np.testing.assert_array_equal(
        np.asarray(beam_bl(ells, fwhm_arcmin=20.0)),
        np.asarray(beam_bl(ells, fwhm_arcmin=20.0, p=1.0)),
    )


def test_beam_bl_p_below_one_fattens_high_ell_wings():
    """p<1 rolls off more slowly at high ℓ (x>1): less suppression there."""
    # x = ℓ(ℓ+1)σ²/2 crosses 1 near ℓ~380 for a 30' beam; ℓ=500 is in the wing.
    bl_gauss = float(beam_bl(jnp.array([500.0]), fwhm_arcmin=30.0, p=1.0)[0])
    bl_fat = float(beam_bl(jnp.array([500.0]), fwhm_arcmin=30.0, p=0.5)[0])
    assert bl_fat > bl_gauss


def test_beam_bl_grad_in_p_finite_at_low_ell():
    """∂B_ℓ/∂p must be finite at ℓ=0,1 (the x=0 NaN guard)."""
    ells = jnp.arange(0, 10, dtype=float)  # includes ℓ=0 where x=0

    def loss(p):
        return jnp.sum(beam_bl(ells, 30.0, p))

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


def test_beam_bl_grad_in_fwhm_finite():
    """∂B_ℓ/∂FWHM is finite across ℓ for the generalized form."""
    ells = jnp.arange(0, 200, dtype=float)

    def loss(fwhm):
        return jnp.sum(beam_bl(ells, fwhm, 0.8))

    g = jax.grad(loss)(20.0)
    assert jnp.isfinite(g)


def test_deconvolve_noise_bb_roundtrip():
    """Deconvolving beam-convolved noise should recover the input noise.

    The map-level auto-spectrum of a beam-smoothed noise realization is
    N_l^map = N_l^instr × B_l^2, so deconvolution divides by B_l^2.
    """
    ells = jnp.arange(2, 300, dtype=float)
    n_raw = jnp.full_like(ells, 1e-6)
    fwhm = 30.0
    n_conv = n_raw * beam_bl(ells, fwhm) ** 2      # anafast of beam-smoothed map
    n_back = deconvolve_noise_bb(n_conv, ells, fwhm)
    np.testing.assert_allclose(np.asarray(n_back), np.asarray(n_raw), rtol=1e-10)


def test_deconvolve_noise_bb_multichannel():
    """Works on (n_chan, n_ells) arrays as well as (n_ells,)."""
    ells = jnp.arange(2, 200, dtype=float)
    n_raw = jnp.stack([jnp.full_like(ells, 1e-6),
                       jnp.full_like(ells, 2e-6)])
    fwhm = 20.0
    n_conv = n_raw * beam_bl(ells, fwhm) ** 2
    n_back = deconvolve_noise_bb(n_conv, ells, fwhm)
    np.testing.assert_allclose(np.asarray(n_back), np.asarray(n_raw), rtol=1e-10)


# -----------------------------------------------------------------------
# Full noise N_ℓ
# -----------------------------------------------------------------------

def test_noise_nl_white_only():
    """With knee_ell=0, N_ℓ = w⁻¹ / B_ℓ² (pure white noise)."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    ch = Channel(nu_ghz=150.0, n_detectors=1000,
                 net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                 knee_ell=0.0, efficiency=eff)
    ells = jnp.arange(2, 300, dtype=float)
    nl = noise_nl(ch, ells, mission_years=5.0, f_sky=0.7)
    w_inv = white_noise_power(ch, 5.0, 0.7)
    bl = beam_bl(ells, 30.0)
    expected = w_inv / bl**2
    assert jnp.allclose(nl, expected, rtol=1e-5)


def test_noise_nl_1f_raises_low_ell():
    """1/f noise makes N_ℓ at low ℓ larger than at high ℓ (beyond beam)."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    ch = Channel(nu_ghz=150.0, n_detectors=1000,
                 net_per_detector=300.0, beam_fwhm_arcmin=1.0,  # tiny beam
                 knee_ell=50.0, alpha_knee=2.0, efficiency=eff)
    ells = jnp.array([5.0, 100.0])
    nl = noise_nl(ch, ells, mission_years=5.0, f_sky=0.7)
    # At ℓ=5 with knee=50, 1/f factor = 1 + (50/5)^2 = 101
    # At ℓ=100, 1/f factor = 1 + (50/100)^2 = 1.25
    # With 1' beam, beam suppression is negligible at these ℓ
    # So N_5 / N_100 ≈ 101 / 1.25 ≈ 80.8
    ratio = float(nl[0]) / float(nl[1])
    assert ratio > 50.0  # conservative check


def test_noise_nl_1f_factor_value():
    """Check exact 1/f factor at a specific ℓ."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    ch_white = Channel(nu_ghz=150.0, n_detectors=1000,
                       net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                       knee_ell=0.0, efficiency=eff)
    ch_1f = Channel(nu_ghz=150.0, n_detectors=1000,
                    net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                    knee_ell=30.0, alpha_knee=1.0, efficiency=eff)
    ells = jnp.array([10.0])
    nl_white = noise_nl(ch_white, ells, 5.0, 0.7)
    nl_1f = noise_nl(ch_1f, ells, 5.0, 0.7)
    # 1/f factor at ℓ=10 with knee=30, α=1: 1 + (30/10)^1 = 4.0
    ratio = float(nl_1f[0]) / float(nl_white[0])
    assert abs(ratio - 4.0) < 1e-4


# -----------------------------------------------------------------------
# Noise matrix
# -----------------------------------------------------------------------

def test_noise_matrix_diagonal():
    """Off-diagonal channel-channel entries are zero."""
    ch1 = Channel(nu_ghz=90.0, n_detectors=500,
                  net_per_detector=400.0, beam_fwhm_arcmin=30.0)
    ch2 = Channel(nu_ghz=150.0, n_detectors=1000,
                  net_per_detector=300.0, beam_fwhm_arcmin=20.0)
    inst = Instrument(channels=(ch1, ch2), mission_duration_years=5.0, f_sky=0.7)
    ells = jnp.arange(2, 100, dtype=float)
    nl_mat = noise_nl_matrix(inst, ells)

    assert nl_mat.shape == (2, 2, 98)
    assert jnp.all(nl_mat[0, 1, :] == 0.0)
    assert jnp.all(nl_mat[1, 0, :] == 0.0)


def test_noise_matrix_diagonal_matches_single():
    """Diagonal entries match noise_nl for each channel."""
    ch1 = Channel(nu_ghz=90.0, n_detectors=500,
                  net_per_detector=400.0, beam_fwhm_arcmin=30.0)
    ch2 = Channel(nu_ghz=150.0, n_detectors=1000,
                  net_per_detector=300.0, beam_fwhm_arcmin=20.0)
    inst = Instrument(channels=(ch1, ch2), mission_duration_years=5.0, f_sky=0.7)
    ells = jnp.arange(2, 100, dtype=float)
    nl_mat = noise_nl_matrix(inst, ells)

    nl1 = noise_nl(ch1, ells, 5.0, 0.7)
    nl2 = noise_nl(ch2, ells, 5.0, 0.7)
    assert jnp.allclose(nl_mat[0, 0, :], nl1, rtol=1e-5)
    assert jnp.allclose(nl_mat[1, 1, :], nl2, rtol=1e-5)


def test_noise_positive():
    """All noise values are positive."""
    ch = Channel(nu_ghz=150.0, n_detectors=1000,
                 net_per_detector=300.0, beam_fwhm_arcmin=30.0,
                 knee_ell=30.0)
    ells = jnp.arange(2, 300, dtype=float)
    nl = noise_nl(ch, ells, 5.0, 0.7)
    assert jnp.all(nl > 0)
