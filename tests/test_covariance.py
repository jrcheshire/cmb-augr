"""Tests for covariance.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from cmb_forecast.covariance import bandpower_covariance, _nu_b, _build_M
from cmb_forecast.signal import SignalModel, flatten_params
from cmb_forecast.instrument import Channel, Instrument, ScalarEfficiency
from cmb_forecast.foregrounds import GaussianForegroundModel
from cmb_forecast.spectra import CMBSpectra


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_chan_instrument():
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    return Instrument(channels=(
        Channel(90.0,  500, 400.0, 30.0, efficiency=eff),
        Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)


@pytest.fixture(scope="module")
def signal_model(two_chan_instrument):
    return SignalModel(
        two_chan_instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
    )


FIDUCIAL = {
    "r": 0.0, "A_lens": 1.0,
    "A_dust": 4.7, "beta_dust": 1.6, "alpha_dust": -0.58, "T_dust": 19.6,
    "A_sync": 1.5, "beta_sync": -3.1, "alpha_sync": -0.6,
    "epsilon": 0.0, "Delta_dust": 0.0,
}


# -----------------------------------------------------------------------
# nu_b (effective modes)
# -----------------------------------------------------------------------

def test_nu_b_per_ell():
    """Per-ℓ bin at ℓ: ν_b = f_sky × (2ℓ+1)."""
    edges = [(10, 10), (11, 11), (50, 50)]
    nu = _nu_b(edges, f_sky=1.0)
    assert abs(float(nu[0]) - 21.0) < 1e-10
    assert abs(float(nu[1]) - 23.0) < 1e-10
    assert abs(float(nu[2]) - 101.0) < 1e-10


def test_nu_b_wide_bin():
    """Wide bin [30, 64]: ν_b = f_sky × (64-30+1) × (30+64+1) = 35 × 95 = 3325."""
    edges = [(30, 64)]
    nu = _nu_b(edges, f_sky=1.0)
    assert abs(float(nu[0]) - 35 * 95) < 1e-10


def test_nu_b_fsky_scaling():
    """ν_b scales linearly with f_sky."""
    edges = [(50, 84)]
    nu1 = float(_nu_b(edges, f_sky=0.5)[0])
    nu2 = float(_nu_b(edges, f_sky=1.0)[0])
    assert abs(nu2 / nu1 - 2.0) < 1e-10


# -----------------------------------------------------------------------
# Covariance matrix structure
# -----------------------------------------------------------------------

def test_covariance_shape(signal_model, two_chan_instrument):
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    n = signal_model.n_data
    assert cov.shape == (n, n)


def test_covariance_symmetric(signal_model, two_chan_instrument):
    """Covariance matrix is symmetric."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    assert jnp.allclose(cov, cov.T, rtol=1e-6)


def test_covariance_positive_diagonal(signal_model, two_chan_instrument):
    """Diagonal entries (variances) are positive."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    assert jnp.all(jnp.diag(cov) > 0)


def test_covariance_block_diagonal(signal_model, two_chan_instrument):
    """Covariance is block-diagonal: different bins are uncorrelated."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    n_spec = signal_model.n_spectra
    n_bins = signal_model.n_bins

    # Check a few off-diagonal bin pairs are zero
    for s1 in range(n_spec):
        for s2 in range(n_spec):
            for b1 in range(min(n_bins, 3)):
                for b2 in range(min(n_bins, 3)):
                    if b1 != b2:
                        val = float(cov[s1 * n_bins + b1, s2 * n_bins + b2])
                        assert val == 0.0, f"Non-zero off-bin covariance at s1={s1},b1={b1},s2={s2},b2={b2}"


def test_covariance_positive_semidefinite(signal_model, two_chan_instrument):
    """Covariance matrix has non-negative eigenvalues."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    eigvals = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigvals >= -1e-10 * jnp.max(eigvals))


# -----------------------------------------------------------------------
# Knox formula values
# -----------------------------------------------------------------------

def test_auto_spectrum_variance(signal_model, two_chan_instrument):
    """Variance of auto-spectrum (i,i) at bin b equals 2*M[i,i,b]^2 / nu_b.

    For an auto-spectrum (i,j) = (i,i), the Knox formula gives:
        Cov = (M_ii * M_ii + M_ii * M_ii) / nu_b = 2 * M_ii^2 / nu_b
    """
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    M = _build_M(signal_model, two_chan_instrument, params)

    from cmb_forecast.covariance import _nu_b
    nu = _nu_b(signal_model._bin_edges, two_chan_instrument.f_sky)

    n_bins = signal_model.n_bins
    # Spectrum (0,0) is the first spectrum (index 0)
    # Its variance at bin b is at index [0 * n_bins + b, 0 * n_bins + b]
    for b in range(min(5, n_bins)):
        expected = 2.0 * float(M[0, 0, b]) ** 2 / float(nu[b])
        actual = float(cov[b, b])
        assert abs(actual - expected) / expected < 1e-6, \
            f"Auto-spectrum variance wrong at bin {b}"


def test_noise_increases_variance(signal_model, two_chan_instrument):
    """Higher noise → larger diagonal covariance entries."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov_normal = bandpower_covariance(signal_model, two_chan_instrument, params)

    # Create instrument with 10× worse NET
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    noisy_inst = Instrument(channels=(
        Channel(90.0,  500, 4000.0, 30.0, efficiency=eff),
        Channel(150.0, 1000, 3000.0, 20.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)
    cov_noisy = bandpower_covariance(signal_model, noisy_inst, params)

    # Auto-spectrum variance should be larger
    assert float(jnp.diag(cov_noisy).mean()) > float(jnp.diag(cov_normal).mean())
