"""Tests for units.py."""

import jax.numpy as jnp
import pytest
from cmb_forecast.units import (
    x_factor, rj_to_cmb, cmb_to_rj, dust_sed, sync_sed,
    sync_sed_curved, dust_sed_deriv_beta, dust_sed_deriv_T,
    sync_sed_deriv_beta, sync_sed_deriv_c,
    NU_DUST_REF_GHZ, NU_SYNC_REF_GHZ,
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
