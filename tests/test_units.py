"""Tests for units.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.bandpass import Bandpass
from augr.units import (
    NU_DUST_REF_GHZ,
    NU_SYNC_REF_GHZ,
    cmb_to_rj,
    color_correct,
    dust_sed,
    dust_sed_deriv_beta,
    dust_sed_deriv_T,
    rj_to_cmb,
    sync_sed,
    sync_sed_curved,
    sync_sed_deriv_beta,
    sync_sed_deriv_c,
    x_factor,
)


def test_x_factor_150():
    """x(150 GHz) = hν/kT_CMB ≈ 2.641 with T_CMB=2.7255 K."""
    x = x_factor(150.0)
    assert abs(x - 2.641) < 0.001


def test_rj_to_cmb_reciprocal():
    """cmb_to_rj is the inverse of rj_to_cmb."""
    for nu in [30.0, 90.0, 150.0, 220.0, 353.0]:
        assert abs(rj_to_cmb(nu) * cmb_to_rj(nu) - 1.0) < 1e-6


def test_rj_to_cmb_known_value():
    """rj_to_cmb(150 GHz) ≈ 0.576 = ΔT_RJ/ΔT_CMB with x≈2.641."""
    val = rj_to_cmb(150.0)
    # x=2.641, eˣ=14.03, x²eˣ/(eˣ-1)² = 6.975*14.03/13.03² ≈ 0.576
    assert abs(val - 0.576) < 0.001


def test_cmb_to_rj_known_value():
    """cmb_to_rj(150 GHz) ≈ 1.735 — factor to multiply a RJ map to get CMB units."""
    val = cmb_to_rj(150.0)
    assert abs(val - 1.735) < 0.001


def test_rj_to_cmb_low_freq_approaches_one():
    """In the RJ limit (low ν, x→0), conversion factor → 1."""
    val = rj_to_cmb(1.0)  # 1 GHz, deep RJ regime
    assert abs(val - 1.0) < 0.01


def test_dust_sed_normalization():
    """dust_sed evaluated at reference frequency should equal 1."""
    val = dust_sed(NU_DUST_REF_GHZ, beta_d=1.6, T_d=19.6)
    assert abs(val - 1.0) < 1e-5


def test_dust_sed_normalization_nondefault_ref():
    """dust_sed at custom reference frequency should equal 1."""
    val = dust_sed(150.0, beta_d=1.6, T_d=19.6, nu_ref_ghz=150.0)
    assert abs(val - 1.0) < 1e-5


def test_sync_sed_normalization():
    """sync_sed evaluated at reference frequency should equal 1."""
    val = sync_sed(NU_SYNC_REF_GHZ, beta_s=-3.1)
    assert abs(val - 1.0) < 1e-5


def test_sync_sed_normalization_nondefault_ref():
    """sync_sed at custom reference frequency should equal 1."""
    val = sync_sed(150.0, beta_s=-3.1, nu_ref_ghz=150.0)
    assert abs(val - 1.0) < 1e-5


def test_dust_sed_positive():
    """Dust SED should be positive at all frequencies."""
    for nu in [20.0, 90.0, 150.0, 220.0, 353.0, 700.0]:
        assert dust_sed(nu, beta_d=1.6, T_d=19.6) > 0


def test_sync_sed_positive():
    """Synchrotron SED should be positive at all frequencies."""
    for nu in [20.0, 40.0, 90.0, 150.0, 220.0]:
        assert sync_sed(nu, beta_s=-3.1) > 0


def test_dust_sed_rising_with_frequency():
    """Dust SED rises steeply with frequency (dominant at high ν)."""
    f_low = dust_sed(100.0, beta_d=1.6, T_d=19.6)
    f_high = dust_sed(300.0, beta_d=1.6, T_d=19.6)
    assert f_high > f_low


def test_sync_sed_falling_with_frequency():
    """Synchrotron SED falls with frequency (dominant at low ν)."""
    f_low = sync_sed(30.0, beta_s=-3.1)
    f_high = sync_sed(90.0, beta_s=-3.1)
    assert f_low > f_high


# ---------------------------------------------------------------------------
# Curved synchrotron SED
# ---------------------------------------------------------------------------

def test_sync_sed_curved_normalization():
    """Curved SED = 1 at reference frequency for any curvature."""
    for c_s in [-0.1, 0.0, 0.1]:
        val = sync_sed_curved(NU_SYNC_REF_GHZ, beta_s=-3.1, c_s=c_s)
        assert abs(val - 1.0) < 1e-5


def test_sync_sed_curved_reduces_to_uncurved():
    """c_s = 0 gives identical result to sync_sed."""
    for nu in [30.0, 90.0, 150.0, 220.0]:
        curved = sync_sed_curved(nu, beta_s=-3.1, c_s=0.0)
        uncurved = sync_sed(nu, beta_s=-3.1)
        assert abs(curved - uncurved) < 1e-10


def test_sync_sed_curved_negative_c_steepens():
    """Negative curvature steepens the spectrum at high frequencies."""
    nu = 150.0  # well above nu_ref=23 GHz
    flat = sync_sed_curved(nu, beta_s=-3.1, c_s=0.0)
    steep = sync_sed_curved(nu, beta_s=-3.1, c_s=-0.05)
    assert steep < flat  # c_s < 0 makes high-freq dimmer


# ---------------------------------------------------------------------------
# SED log-derivatives
# ---------------------------------------------------------------------------

def test_dust_deriv_beta_vanishes_at_ref():
    """∂ ln f_d / ∂ β_d = ln(ν/ν_ref) = 0 at ν = ν_ref."""
    val = dust_sed_deriv_beta(NU_DUST_REF_GHZ)
    assert abs(val) < 1e-10


def test_dust_deriv_T_vanishes_at_ref():
    """∂ ln f_d / ∂ T_d = 0 at ν = ν_ref."""
    val = dust_sed_deriv_T(NU_DUST_REF_GHZ, T_d=19.6)
    assert abs(val) < 1e-10


def test_sync_deriv_beta_vanishes_at_ref():
    """∂ ln f_s / ∂ β_s = 0 at ν = ν_ref."""
    val = sync_sed_deriv_beta(NU_SYNC_REF_GHZ)
    assert abs(val) < 1e-10


def test_sync_deriv_c_vanishes_at_ref():
    """∂ ln f_s / ∂ c_s = 0 at ν = ν_ref."""
    val = sync_sed_deriv_c(NU_SYNC_REF_GHZ)
    assert abs(val) < 1e-10


def test_dust_deriv_beta_is_log_ratio():
    """∂ ln f_d / ∂ β_d = ln(ν/353) — exact analytic result."""
    for nu in [90.0, 150.0, 220.0]:
        val = dust_sed_deriv_beta(nu)
        expected = jnp.log(nu / NU_DUST_REF_GHZ)
        assert abs(val - expected) < 1e-10


def test_dust_deriv_T_numerical():
    """Compare analytic ∂ ln f_d/∂T_d against finite difference."""
    nu = 150.0
    T_d = 19.6
    dT = 0.001
    f_plus = dust_sed(nu, beta_d=1.6, T_d=T_d + dT)
    f_minus = dust_sed(nu, beta_d=1.6, T_d=T_d - dT)
    f_0 = dust_sed(nu, beta_d=1.6, T_d=T_d)
    numerical = (f_plus - f_minus) / (2.0 * dT * f_0)
    analytic = dust_sed_deriv_T(nu, T_d)
    assert abs(numerical - analytic) / abs(analytic) < 1e-4


def test_sync_deriv_c_is_log_ratio_squared():
    """∂ ln f_s / ∂ c_s = [ln(ν/23)]² — exact analytic result."""
    for nu in [30.0, 90.0, 150.0]:
        val = sync_sed_deriv_c(nu)
        expected = jnp.log(nu / NU_SYNC_REF_GHZ) ** 2
        assert abs(val - expected) < 1e-10


# ---------------------------------------------------------------------------
# Bandpass color correction
# ---------------------------------------------------------------------------

def test_color_correct_flat_sed_is_unity():
    """A flat SED (CMB column) band-averages to exactly 1 for any bandpass."""
    bp = Bandpass.tophat(150.0, 0.25)
    val = color_correct(lambda nu: jnp.ones_like(nu), bp)
    assert abs(float(val) - 1.0) < 1e-12


def test_color_correct_monochromatic_recovers_scalar():
    """A single-point bandpass reproduces the band-center evaluation exactly."""
    bp = Bandpass.monochromatic(150.0)
    cc = color_correct(dust_sed, bp, beta_d=1.6, T_d=19.6)
    mono = dust_sed(150.0, beta_d=1.6, T_d=19.6)
    assert abs(float(cc) - float(mono)) < 1e-12


def test_color_correct_within_band_extremes():
    """Band-averaged dust SED lies between its band-edge values (monotone rise)."""
    nu0, f = 150.0, 0.25
    bp = Bandpass.tophat(nu0, f)
    cc = color_correct(dust_sed, bp, beta_d=1.6, T_d=19.6)
    lo = dust_sed(nu0 * (1 - f / 2), beta_d=1.6, T_d=19.6)
    hi = dust_sed(nu0 * (1 + f / 2), beta_d=1.6, T_d=19.6)
    assert float(lo) < float(cc) < float(hi)


def test_color_correct_grad_wrt_nu_center():
    """d(band-averaged dust SED)/d(nu_center) matches central finite difference."""
    def cc(nu_c):
        return color_correct(
            lambda nu: dust_sed(nu, 1.6, 19.6), Bandpass.smooth_tophat(nu_c, 0.25)
        )

    g = jax.grad(cc)(150.0)
    h = 1e-2
    fd = (cc(150.0 + h) - cc(150.0 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert abs(float(g) - float(fd)) / abs(float(fd)) < 1e-4
    # Same under jit.
    g_jit = jax.jit(jax.grad(cc))(150.0)
    assert abs(float(g_jit) - float(g)) / abs(float(g)) < 1e-4


def test_color_correct_grad_wrt_frac_bw_finite():
    """Gradient flows w.r.t. fractional bandwidth via the smooth weights."""
    def cc(f):
        return color_correct(
            lambda nu: dust_sed(nu, 1.6, 19.6), Bandpass.smooth_tophat(150.0, f)
        )

    g = jax.grad(cc)(0.25)
    assert jnp.isfinite(g)


def test_color_correct_matches_pysm():
    """Kernel cross-check: color_correct reproduces PySM's bandpass integration.

    Build the μK_CMB band value the way PySM does — integrate the RJ emission
    (dust_sed · cmb_to_rj) with PySM-normalized weights, then convert to μK_CMB
    via PySM's bandpass_unit_conversion — and confirm color_correct matches it.
    Pure-numeric (no Sky build / network). Guards the g = w·ν²·cmb_to_rj kernel.
    """
    pytest.importorskip("pysm3")
    import pysm3.units as pu
    from pysm3.utils import bandpass_unit_conversion, normalize_weights

    # PySM 3.4.2's normalize_weights / bandpass_unit_conversion call np.trapz,
    # removed in numpy 2.0. Shim it locally so we can use them as ground truth.
    # (Production avoids these functions entirely — compsep_sims band-averages
    # monochromatic maps with this same kernel; see pysm_fg_iqu.)
    had_trapz = hasattr(np, "trapz")
    if not had_trapz:
        np.trapz = np.trapezoid  # noqa: NPY201  (shim PySM's removed call)
    try:
        for nu0, f in [(150.0, 0.25), (95.0, 0.30), (30.0, 0.20), (353.0, 0.25)]:
            bp = Bandpass.tophat(nu0, f, n_quad=64)
            nu = np.asarray(bp.nu_ghz)
            weights = np.asarray(bp.weights)

            # PySM ground truth: RJ emission integrated with PySM-normalized
            # weights, then unit-converted to μK_CMB as Sky.get_emission does.
            # A μK_CMB SED maps to μK_RJ emission via rj_to_cmb.
            m_rj = np.asarray(dust_sed(nu, 1.6, 19.6)) * np.asarray(rj_to_cmb(nu))
            w_n = normalize_weights(nu, weights)
            i_rj = np.trapezoid(m_rj * w_n, nu)
            factor = bandpass_unit_conversion(
                nu * pu.GHz, weights, pu.uK_CMB, pu.uK_RJ
            ).value
            pysm_band = i_rj * factor

            cc = float(color_correct(dust_sed, bp, beta_d=1.6, T_d=19.6))
            assert abs(cc - pysm_band) / abs(pysm_band) < 1e-3, (nu0, f, cc, pysm_band)
    finally:
        if not had_trapz:
            del np.trapz  # noqa: NPY201  (remove the shim)
