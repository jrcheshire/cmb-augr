"""Tests for foregrounds.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from cmb_forecast.foregrounds import (
    GaussianForegroundModel,
    MomentExpansionModel,
    ForegroundModel,
    ELL_REF,
    _dust_moment_factor,
    _sync_moment_factor,
)
from cmb_forecast.units import NU_DUST_REF_GHZ, NU_SYNC_REF_GHZ


@pytest.fixture
def model():
    return GaussianForegroundModel()


# Fiducial parameters matching BK15 ML (Buza thesis Sec. 6.3.2)
FIDUCIAL = jnp.array([
    4.7,     # A_dust  [μK² at 353 GHz, ℓ=80]
    1.6,     # beta_dust
    -0.58,   # alpha_dust
    19.6,    # T_dust [K]
    1.5,     # A_sync [μK² at 23 GHz, ℓ=80]
    -3.1,    # beta_sync
    -0.6,    # alpha_sync
    0.0,     # epsilon (no cross-correlation)
    0.0,     # Delta_dust (no decorrelation)
])


# -----------------------------------------------------------------------
# Protocol conformance
# -----------------------------------------------------------------------

def test_satisfies_protocol(model):
    """GaussianForegroundModel satisfies ForegroundModel Protocol."""
    assert isinstance(model, ForegroundModel)


def test_parameter_names(model):
    """Has 9 named parameters in expected order."""
    names = model.parameter_names
    assert len(names) == 9
    assert names[0] == "A_dust"
    assert names[-1] == "Delta_dust"


# -----------------------------------------------------------------------
# Dust spectrum
# -----------------------------------------------------------------------

def test_dust_amplitude_at_reference():
    """At ν=353 GHz and ℓ=80, dust D_ℓ = A_dust (with ε=0, Δ=0)."""
    model = GaussianForegroundModel()
    ells = jnp.array([ELL_REF])
    cl = model.cl_bb(NU_DUST_REF_GHZ, NU_DUST_REF_GHZ, ells, FIDUCIAL)
    # Model returns C_ℓ; convert to D_ℓ = ℓ(ℓ+1)C_ℓ/(2π)
    dl = float(cl[0]) * ELL_REF * (ELL_REF + 1) / (2 * np.pi)
    A_d = float(FIDUCIAL[0])
    # At ref freq and ref ℓ: D_ℓ ≈ A_d (sync negligible at 353 GHz)
    assert dl > A_d * 0.99


def test_dust_ell_scaling(model):
    """Dust D_ℓ follows (ℓ/80)^alpha_dust power law."""
    ells = jnp.array([40.0, 80.0, 160.0])
    # Use dust-only: set A_sync=0, epsilon=0
    params = FIDUCIAL.at[4].set(0.0)  # A_sync = 0
    cl = model.cl_bb(NU_DUST_REF_GHZ, NU_DUST_REF_GHZ, ells, params)
    # Convert to D_ℓ to check power-law scaling
    dl = cl * ells * (ells + 1) / (2 * np.pi)
    ratio = float(dl[0]) / float(dl[1])  # D_ℓ(40) / D_ℓ(80)
    expected = (40.0 / 80.0) ** float(FIDUCIAL[2])
    assert abs(ratio - expected) / expected < 1e-4


# -----------------------------------------------------------------------
# Synchrotron spectrum
# -----------------------------------------------------------------------

def test_sync_amplitude_at_reference():
    """At ν=23 GHz and ℓ=80, with no dust, D_ℓ ≈ A_sync."""
    model = GaussianForegroundModel()
    ells = jnp.array([ELL_REF])
    # Zero dust, zero cross
    params = FIDUCIAL.at[0].set(0.0)  # A_dust = 0
    cl = model.cl_bb(NU_SYNC_REF_GHZ, NU_SYNC_REF_GHZ, ells, params)
    dl = float(cl[0]) * ELL_REF * (ELL_REF + 1) / (2 * np.pi)
    A_s = float(FIDUCIAL[4])
    assert abs(dl - A_s) < 1e-4


def test_sync_ell_scaling(model):
    """Synchrotron D_ℓ follows (ℓ/80)^alpha_sync power law."""
    ells = jnp.array([40.0, 80.0])
    # Sync-only at ref freq
    params = FIDUCIAL.at[0].set(0.0)  # A_dust = 0
    cl = model.cl_bb(NU_SYNC_REF_GHZ, NU_SYNC_REF_GHZ, ells, params)
    # Convert to D_ℓ to check power-law scaling
    dl = cl * ells * (ells + 1) / (2 * np.pi)
    ratio = float(dl[0]) / float(dl[1])
    expected = (40.0 / 80.0) ** float(FIDUCIAL[6])  # alpha_sync
    assert abs(ratio - expected) / expected < 1e-4


# -----------------------------------------------------------------------
# Cross-correlation
# -----------------------------------------------------------------------

def test_cross_zero_when_epsilon_zero(model):
    """With ε=0, dust+sync cross term vanishes."""
    ells = jnp.array([80.0])
    params_e0 = FIDUCIAL.at[7].set(0.0)  # epsilon = 0
    # Total = dust + sync (no cross)
    cl_e0 = model.cl_bb(150.0, 150.0, ells, params_e0)
    # Now manually compute dust + sync
    params_no_sync = params_e0.at[4].set(0.0)
    cl_dust = model.cl_bb(150.0, 150.0, ells, params_no_sync)
    params_no_dust = params_e0.at[0].set(0.0)
    cl_sync = model.cl_bb(150.0, 150.0, ells, params_no_dust)
    assert abs(float(cl_e0[0]) - float(cl_dust[0]) - float(cl_sync[0])) < 1e-8


def test_cross_symmetric(model):
    """Cross term is symmetric: C(ν_i, ν_j) = C(ν_j, ν_i)."""
    ells = jnp.arange(20, 200, dtype=float)
    params = FIDUCIAL.at[7].set(0.5)  # nonzero epsilon
    cl_ij = model.cl_bb(90.0, 220.0, ells, params)
    cl_ji = model.cl_bb(220.0, 90.0, ells, params)
    assert jnp.allclose(cl_ij, cl_ji, rtol=1e-5)


def test_cross_single_freq_reduces(model):
    """At a single frequency, cross = ε × √(dust × sync) (no factor of 2 issue)."""
    ells = jnp.array([80.0])
    nu = 150.0
    eps = 0.3
    params = FIDUCIAL.at[7].set(eps)

    # Total with cross
    cl_total = model.cl_bb(nu, nu, ells, params)

    # Dust-only and sync-only at this freq
    params_dust_only = params.at[4].set(0.0).at[7].set(0.0)  # A_sync=0, eps=0
    cl_d = model.cl_bb(nu, nu, ells, params_dust_only)
    params_sync_only = params.at[0].set(0.0).at[7].set(0.0)  # A_dust=0, eps=0
    cl_s = model.cl_bb(nu, nu, ells, params_sync_only)

    # Expected: dust + sync + ε × √(dust × sync)
    expected = float(cl_d[0]) + float(cl_s[0]) + eps * np.sqrt(float(cl_d[0]) * float(cl_s[0]))
    assert abs(float(cl_total[0]) - expected) / expected < 1e-4


# -----------------------------------------------------------------------
# Frequency decorrelation
# -----------------------------------------------------------------------

def test_decorrelation_no_effect_auto(model):
    """Decorrelation has no effect on auto-spectra (same freq)."""
    ells = jnp.arange(20, 200, dtype=float)
    params_d0 = FIDUCIAL.at[8].set(0.0)
    params_d1 = FIDUCIAL.at[8].set(0.5)
    cl_0 = model.cl_bb(150.0, 150.0, ells, params_d0)
    cl_1 = model.cl_bb(150.0, 150.0, ells, params_d1)
    assert jnp.allclose(cl_0, cl_1, rtol=1e-5)


def test_decorrelation_reduces_cross(model):
    """Decorrelation reduces cross-frequency dust power."""
    ells = jnp.array([80.0])
    # Dust-only (no sync, no cross)
    params_base = FIDUCIAL.at[4].set(0.0).at[7].set(0.0)
    params_d0 = params_base.at[8].set(0.0)
    params_d1 = params_base.at[8].set(0.5)
    cl_0 = model.cl_bb(150.0, 220.0, ells, params_d0)
    cl_1 = model.cl_bb(150.0, 220.0, ells, params_d1)
    assert float(cl_1[0]) < float(cl_0[0])


def test_decorrelation_value(model):
    """Check exact decorrelation factor at specific frequencies."""
    ells = jnp.array([80.0])
    Delta = 0.3
    # Dust-only
    params_base = FIDUCIAL.at[4].set(0.0).at[7].set(0.0)
    params_d0 = params_base.at[8].set(0.0)
    params_d1 = params_base.at[8].set(Delta)
    cl_0 = float(model.cl_bb(150.0, 220.0, ells, params_d0)[0])
    cl_1 = float(model.cl_bb(150.0, 220.0, ells, params_d1)[0])
    expected_factor = np.exp(-Delta * abs(np.log(150.0 / 220.0)))
    assert abs(cl_1 / cl_0 - expected_factor) < 1e-4


# -----------------------------------------------------------------------
# Positivity and general sanity
# -----------------------------------------------------------------------

def test_all_positive_same_freq(model):
    """Total foreground power is positive at all ℓ for same-frequency spectra."""
    ells = jnp.arange(2, 300, dtype=float)
    for nu in [30.0, 90.0, 150.0, 220.0, 353.0]:
        cl = model.cl_bb(nu, nu, ells, FIDUCIAL)
        assert jnp.all(cl > 0), f"Negative power at {nu} GHz"


def test_dust_dominates_high_freq(model):
    """At 353 GHz, dust dominates over synchrotron."""
    ells = jnp.array([80.0])
    # Dust-only
    params_dust = FIDUCIAL.at[4].set(0.0).at[7].set(0.0)
    cl_d = float(model.cl_bb(353.0, 353.0, ells, params_dust)[0])
    # Sync-only
    params_sync = FIDUCIAL.at[0].set(0.0).at[7].set(0.0)
    cl_s = float(model.cl_bb(353.0, 353.0, ells, params_sync)[0])
    assert cl_d > 1000 * cl_s, "Dust should strongly dominate at 353 GHz"


def test_sync_dominates_low_freq(model):
    """At 23 GHz, synchrotron dominates over dust."""
    ells = jnp.array([80.0])
    params_dust = FIDUCIAL.at[4].set(0.0).at[7].set(0.0)
    cl_d = float(model.cl_bb(23.0, 23.0, ells, params_dust)[0])
    params_sync = FIDUCIAL.at[0].set(0.0).at[7].set(0.0)
    cl_s = float(model.cl_bb(23.0, 23.0, ells, params_sync)[0])
    assert cl_s > 100 * cl_d, "Sync should strongly dominate at 23 GHz"


# =======================================================================
# MomentExpansionModel tests
# =======================================================================

@pytest.fixture
def moment_model():
    return MomentExpansionModel()


# Fiducial for moment model: first 9 params same as Gaussian, rest zero
FIDUCIAL_MOMENT = jnp.array([
    4.7,     # A_dust
    1.6,     # beta_dust
    -0.58,   # alpha_dust
    19.6,    # T_dust
    1.5,     # A_sync
    -3.1,    # beta_sync
    -0.6,    # alpha_sync
    0.0,     # epsilon
    0.0,     # Delta_dust
    # --- new params, all zero ---
    0.0,     # c_sync
    0.0,     # Delta_sync
    0.0,     # omega_d_beta
    0.0,     # omega_d_T
    0.0,     # omega_d_betaT
    0.0,     # omega_s_beta
    0.0,     # omega_s_c
    0.0,     # omega_s_betac
])


# -----------------------------------------------------------------------
# Protocol and parameter conformance
# -----------------------------------------------------------------------

def test_moment_satisfies_protocol(moment_model):
    """MomentExpansionModel satisfies ForegroundModel Protocol."""
    assert isinstance(moment_model, ForegroundModel)


def test_moment_parameter_count(moment_model):
    """Has 17 named parameters."""
    names = moment_model.parameter_names
    assert len(names) == 17
    assert names[0] == "A_dust"
    assert names[9] == "c_sync"
    assert names[-1] == "omega_s_betac"


# -----------------------------------------------------------------------
# Critical: reduction to GaussianForegroundModel
# -----------------------------------------------------------------------

def test_moment_reduces_to_gaussian():
    """With new params = 0, MomentExpansionModel == GaussianForegroundModel."""
    gauss = GaussianForegroundModel()
    moment = MomentExpansionModel()
    ells = jnp.arange(20, 300, dtype=float)

    for nu_i, nu_j in [(150.0, 150.0), (90.0, 220.0), (30.0, 353.0)]:
        cl_g = gauss.cl_bb(nu_i, nu_j, ells, FIDUCIAL)
        cl_m = moment.cl_bb(nu_i, nu_j, ells, FIDUCIAL_MOMENT)
        assert jnp.allclose(cl_g, cl_m, rtol=1e-8), \
            f"Mismatch at ({nu_i}, {nu_j}) GHz"


def test_moment_reduces_with_epsilon():
    """Reduction also holds with nonzero epsilon and Delta_dust."""
    gauss = GaussianForegroundModel()
    moment = MomentExpansionModel()
    ells = jnp.arange(20, 200, dtype=float)

    params_g = FIDUCIAL.at[7].set(0.5).at[8].set(0.3)
    params_m = FIDUCIAL_MOMENT.at[7].set(0.5).at[8].set(0.3)

    cl_g = gauss.cl_bb(90.0, 220.0, ells, params_g)
    cl_m = moment.cl_bb(90.0, 220.0, ells, params_m)
    assert jnp.allclose(cl_g, cl_m, rtol=1e-8)


# -----------------------------------------------------------------------
# Moment corrections
# -----------------------------------------------------------------------

def test_moment_dust_beta_increases_auto():
    """Positive omega_d_beta increases auto-spectrum power."""
    moment = MomentExpansionModel()
    ells = jnp.array([80.0])
    nu = 150.0  # not at reference freq (where derivative = 0)

    params_0 = FIDUCIAL_MOMENT
    params_w = FIDUCIAL_MOMENT.at[11].set(0.1)  # omega_d_beta = 0.1

    cl_0 = float(moment.cl_bb(nu, nu, ells, params_0)[0])
    cl_w = float(moment.cl_bb(nu, nu, ells, params_w)[0])
    assert cl_w > cl_0


def test_moment_produces_decorrelation():
    """Moment expansion reduces cross-freq correlation coefficient."""
    moment = MomentExpansionModel()
    ells = jnp.array([80.0])
    # Dust-only
    params = FIDUCIAL_MOMENT.at[4].set(0.0).at[7].set(0.0)  # A_sync=0, eps=0
    params_w = params.at[11].set(0.1)  # omega_d_beta = 0.1

    cl_auto_i = float(moment.cl_bb(150.0, 150.0, ells, params_w)[0])
    cl_auto_j = float(moment.cl_bb(220.0, 220.0, ells, params_w)[0])
    cl_cross = float(moment.cl_bb(150.0, 220.0, ells, params_w)[0])
    rho = cl_cross / np.sqrt(cl_auto_i * cl_auto_j)
    assert rho < 1.0, "Moment expansion should produce decorrelation"

    # Without moments, should be perfectly correlated
    cl_auto_i_0 = float(moment.cl_bb(150.0, 150.0, ells, params)[0])
    cl_auto_j_0 = float(moment.cl_bb(220.0, 220.0, ells, params)[0])
    cl_cross_0 = float(moment.cl_bb(150.0, 220.0, ells, params)[0])
    rho_0 = cl_cross_0 / np.sqrt(cl_auto_i_0 * cl_auto_j_0)
    assert abs(rho_0 - 1.0) < 1e-6


def test_moment_no_effect_at_ref_freq():
    """Dust moments have no effect at dust reference frequency (derivatives vanish)."""
    moment = MomentExpansionModel()
    ells = jnp.arange(20, 200, dtype=float)
    # Dust-only at 353 GHz
    params_base = FIDUCIAL_MOMENT.at[4].set(0.0).at[7].set(0.0)
    params_w = params_base.at[11].set(0.5).at[12].set(0.5)

    cl_0 = moment.cl_bb(NU_DUST_REF_GHZ, NU_DUST_REF_GHZ, ells, params_base)
    cl_w = moment.cl_bb(NU_DUST_REF_GHZ, NU_DUST_REF_GHZ, ells, params_w)
    assert jnp.allclose(cl_0, cl_w, rtol=1e-8)


# -----------------------------------------------------------------------
# Synchrotron curvature
# -----------------------------------------------------------------------

def test_sync_curvature_steepens():
    """Negative c_s reduces synchrotron power at high freq (away from ref)."""
    moment = MomentExpansionModel()
    ells = jnp.array([80.0])
    # Sync-only at 150 GHz
    params = FIDUCIAL_MOMENT.at[0].set(0.0).at[7].set(0.0)  # A_dust=0
    params_curved = params.at[9].set(-0.05)  # c_sync = -0.05

    cl_flat = float(moment.cl_bb(150.0, 150.0, ells, params)[0])
    cl_curved = float(moment.cl_bb(150.0, 150.0, ells, params_curved)[0])
    assert cl_curved < cl_flat


# -----------------------------------------------------------------------
# Synchrotron decorrelation
# -----------------------------------------------------------------------

def test_delta_sync_no_effect_auto(moment_model):
    """Delta_sync has no effect on auto-spectra."""
    ells = jnp.arange(20, 200, dtype=float)
    params_0 = FIDUCIAL_MOMENT.at[10].set(0.0)
    params_1 = FIDUCIAL_MOMENT.at[10].set(0.5)
    cl_0 = moment_model.cl_bb(30.0, 30.0, ells, params_0)
    cl_1 = moment_model.cl_bb(30.0, 30.0, ells, params_1)
    assert jnp.allclose(cl_0, cl_1, rtol=1e-5)


def test_delta_sync_reduces_cross(moment_model):
    """Delta_sync reduces cross-frequency synchrotron power."""
    ells = jnp.array([80.0])
    # Sync-only
    params = FIDUCIAL_MOMENT.at[0].set(0.0).at[7].set(0.0)
    params_d0 = params.at[10].set(0.0)
    params_d1 = params.at[10].set(0.5)
    cl_0 = moment_model.cl_bb(30.0, 90.0, ells, params_d0)
    cl_1 = moment_model.cl_bb(30.0, 90.0, ells, params_d1)
    assert float(cl_1[0]) < float(cl_0[0])


# -----------------------------------------------------------------------
# Symmetry
# -----------------------------------------------------------------------

def test_moment_symmetric(moment_model):
    """C(ν_i, ν_j) = C(ν_j, ν_i) with all features active."""
    ells = jnp.arange(20, 200, dtype=float)
    params = FIDUCIAL_MOMENT.at[7].set(0.3).at[9].set(-0.02)
    params = params.at[11].set(0.05).at[14].set(0.03)
    cl_ij = moment_model.cl_bb(90.0, 220.0, ells, params)
    cl_ji = moment_model.cl_bb(220.0, 90.0, ells, params)
    assert jnp.allclose(cl_ij, cl_ji, rtol=1e-5)


# -----------------------------------------------------------------------
# Moment helper functions
# -----------------------------------------------------------------------

def test_dust_moment_factor_unity_at_zero():
    """Moment factor = 1 when all omega = 0."""
    factor = _dust_moment_factor(150.0, 220.0, 19.6, 0.0, 0.0, 0.0)
    assert abs(factor - 1.0) < 1e-10


def test_sync_moment_factor_unity_at_zero():
    """Moment factor = 1 when all omega = 0."""
    factor = _sync_moment_factor(30.0, 90.0, 0.0, 0.0, 0.0)
    assert abs(factor - 1.0) < 1e-10


def test_dust_moment_factor_at_ref_freq():
    """Dust moment factor = 1 at reference frequency (derivatives vanish)."""
    factor = _dust_moment_factor(NU_DUST_REF_GHZ, NU_DUST_REF_GHZ,
                                 19.6, 1.0, 1.0, 1.0)
    assert abs(factor - 1.0) < 1e-10
