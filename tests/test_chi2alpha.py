"""Tests for ``augr._chi2alpha`` -- the JAX port of BK's ``chi2alpha.m``."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr._chi2alpha import chi2alpha


def _wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


# -- r = 0 boresight: closed-form match ----------------------------------------

@pytest.mark.parametrize("ra", [0.0, 30.0, -45.0, 200.0])
@pytest.mark.parametrize("dec", [-90.0, -55.0, -30.0, 0.0])
@pytest.mark.parametrize("chi", [0.0, 17.0, -45.0])
@pytest.mark.parametrize("thetaref", [0.0, 90.0, -120.0])
def test_boresight_closed_form(ra, dec, chi, thetaref):
    """At r=0, alpha = -90 + chi + thetaref, independent of (ra, dec, theta)."""
    for theta in (0.0, 75.0, -33.0):
        alpha = chi2alpha(ra, dec, 0.0, theta, chi, thetaref)
        expected = _wrap180(-90.0 + chi + thetaref)
        assert np.isclose(alpha, expected, atol=1e-10), (
            f"theta={theta} ra={ra} dec={dec} chi={chi} thetaref={thetaref}: "
            f"got {alpha}, expected {expected}"
        )


# -- RA invariance for off-axis detectors --------------------------------------

@pytest.mark.parametrize("delta_ra", [0.5, 12.0, 73.0, -150.0])
def test_offaxis_ra_invariance(delta_ra):
    """At fixed dec, alpha must be invariant under a common ra shift."""
    base = dict(dec=-55.0, r=2.5, theta=37.0, chi=11.0, thetaref=68.0)
    a0 = chi2alpha(ra=10.0, **base)
    a1 = chi2alpha(ra=10.0 + delta_ra, **base)
    assert np.isclose(a0, a1, atol=1e-10)


# -- deck equivariance ---------------------------------------------------------

@pytest.mark.parametrize("delta_d", [1.0, 25.0, 113.0, -180.0])
@pytest.mark.parametrize("r", [0.0, 0.5, 4.7])
def test_deck_equivariance(delta_d, r):
    """A deck rotation Delta increments alpha by Delta (mod 360)."""
    base = dict(ra=20.0, dec=-55.0, r=r, theta=37.0, chi=11.0)
    a0 = chi2alpha(thetaref=68.0, **base)
    a1 = chi2alpha(thetaref=68.0 + delta_d, **base)
    assert np.isclose(_wrap180(a1 - a0), _wrap180(delta_d), atol=1e-10)


# -- r -> 0 limit consistency --------------------------------------------------

def test_r_zero_limit_continuity():
    """Off-axis formula at small r should match the r=0 branch."""
    base = dict(ra=20.0, dec=-55.0, theta=37.0, chi=11.0, thetaref=68.0)
    a0 = chi2alpha(r=0.0, **base)
    a_eps = chi2alpha(r=1e-6, **base)
    # r=1e-6 deg means a ~1.7e-8 rad step; should match r=0 to ~1e-8 deg.
    assert np.isclose(a0, a_eps, atol=1e-5)


# -- spot check against an independent NumPy great-circle reference ------------

def _numpy_reference(ra, dec, r, theta, chi, thetaref):
    """Independent NumPy implementation of chi2alpha for cross-checking.

    Uses the same great-circle formulae as the JAX port, but re-derived
    from the textbook (Vincenty's reckon + initial bearing) so that the
    two paths share only the underlying spherical geometry, not code.
    """
    lat1 = np.deg2rad(dec)
    lon1 = np.deg2rad(ra)
    dist = np.deg2rad(r)
    bearing = np.deg2rad(theta - 90.0)

    if r == 0.0:
        alpha = -90.0 + theta + chi
    else:
        sin_l1, cos_l1 = np.sin(lat1), np.cos(lat1)
        sin_d, cos_d = np.sin(dist), np.cos(dist)
        sin_b, cos_b = np.sin(bearing), np.cos(bearing)
        sin_l2 = sin_l1 * cos_d + cos_l1 * sin_d * cos_b
        lat2 = np.arcsin(np.clip(sin_l2, -1.0, 1.0))
        lon2 = lon1 + np.arctan2(sin_b * sin_d * cos_l1,
                                 cos_d - sin_l1 * sin_l2)
        # Bearing from (lat2, lon2) back to (lat1, lon1).
        dlon = lon1 - lon2
        az = np.arctan2(
            np.sin(dlon) * np.cos(lat1),
            np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(dlon),
        )
        alpha = np.rad2deg(az) - 180.0 + chi

    alpha = alpha + thetaref - theta
    return (alpha + 180.0) % 360.0 - 180.0


# Spot-check geometries shared by the NumPy and MATLAB reference tests.
_SPOT_CASES = [
    dict(ra=0.0, dec=-30.0, r=10.0, theta=0.0, chi=0.0, thetaref=0.0),
    dict(ra=20.0, dec=-55.0, r=2.5, theta=45.0, chi=11.0, thetaref=68.0),
    dict(ra=-30.0, dec=-73.0, r=8.0, theta=180.0, chi=-13.5, thetaref=113.0),
    dict(ra=10.0, dec=-38.0, r=5.5, theta=270.0, chi=22.0, thetaref=-45.0),
]


@pytest.mark.parametrize("case", _SPOT_CASES)
def test_against_numpy_reference(case):
    expected = _numpy_reference(**case)
    actual = float(chi2alpha(**case))
    assert np.isclose(actual, expected, atol=1e-10), f"case={case}"


# Reference values produced by running BK's chi2alpha.m on _SPOT_CASES (raw
# MATLAB output is in [0, 360); these have been wrapped to [-180, 180) to
# match the JAX port's convention). Generated 2026-04-27 via:
#
#   matlab -batch "addpath(genpath('/Users/jamie/bicepkeck/bk_analysis/pipeline/util'));
#                  for case = _SPOT_CASES: chi2alpha(...)"
_MATLAB_ALPHA = [
    -84.274894826624,   # 275.725105173376 wrapped
    -8.609064823796,    # 351.390935176204 wrapped
    -14.975725873472,   # already in [-180, 180)
    -113.000000000000,  # already in [-180, 180)
]


@pytest.mark.parametrize("case,expected",
                         list(zip(_SPOT_CASES, _MATLAB_ALPHA, strict=False)))
def test_against_matlab_chi2alpha(case, expected):
    """Bit-exact match against BK's production chi2alpha.m at spot points."""
    actual = float(chi2alpha(**case))
    assert np.isclose(actual, expected, atol=1e-10), f"case={case}"


# -- JAX differentiability -----------------------------------------------------

def test_grad_thetaref():
    """d alpha / d thetaref should be 1 (modulo wrap-discontinuity points)."""
    def f(thetaref):
        return chi2alpha(20.0, -55.0, 2.5, 37.0, 11.0, thetaref)
    g = jax.grad(f)
    assert np.isclose(float(g(50.0)), 1.0, atol=1e-10)


def test_grad_chi():
    def f(chi):
        return chi2alpha(20.0, -55.0, 2.5, 37.0, chi, 68.0)
    g = jax.grad(f)
    assert np.isclose(float(g(11.0)), 1.0, atol=1e-10)


def test_grad_dec_offaxis():
    """d alpha / d dec is non-trivial off-axis but should be finite."""
    def f(dec):
        return chi2alpha(20.0, dec, 2.5, 37.0, 11.0, 68.0)
    g = jax.grad(f)
    val = float(g(-55.0))
    assert np.isfinite(val), f"got {val}"


# -- vectorization -------------------------------------------------------------

def test_vectorized():
    """Inputs of compatible shape broadcast as expected."""
    n = 5
    ra = jnp.linspace(0.0, 60.0, n)
    dec = jnp.full((n,), -55.0)
    out = chi2alpha(ra, dec, 0.0, 37.0, 11.0, 68.0)
    assert out.shape == (n,)
    # All r=0 outputs should be equal regardless of ra.
    assert np.allclose(out, out[0], atol=1e-10)
