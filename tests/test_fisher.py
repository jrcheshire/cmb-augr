"""Tests for fisher.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from cmb_forecast.fisher import FisherForecast
from cmb_forecast.signal import SignalModel, flatten_params
from cmb_forecast.instrument import Channel, Instrument, ScalarEfficiency
from cmb_forecast.foregrounds import GaussianForegroundModel
from cmb_forecast.spectra import CMBSpectra


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

FIDUCIAL = {
    "r": 0.0, "A_lens": 1.0,
    "A_dust": 4.7, "beta_dust": 1.6, "alpha_dust": -0.58, "T_dust": 19.6,
    "A_sync": 1.5, "beta_sync": -3.1, "alpha_sync": -0.6,
    "epsilon": 0.0, "Delta_dust": 0.0,
}


@pytest.fixture(scope="module")
def instrument():
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    return Instrument(channels=(
        Channel(90.0,  500, 400.0, 30.0, efficiency=eff),
        Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
        Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)


@pytest.fixture(scope="module")
def signal_model(instrument):
    return SignalModel(
        instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
    )


@pytest.fixture(scope="module")
def fisher(signal_model, instrument):
    return FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )


# -----------------------------------------------------------------------
# Fisher matrix structure
# -----------------------------------------------------------------------

def test_fisher_matrix_shape(fisher):
    """Fisher matrix shape = (n_free, n_free). T_dust is fixed → 10 free params."""
    F = fisher.fisher_matrix
    n = fisher.n_free
    assert F.shape == (n, n)
    assert n == 10  # 11 total - 1 fixed


def test_fisher_matrix_symmetric(fisher):
    """Fisher matrix is symmetric."""
    F = fisher.fisher_matrix
    assert jnp.allclose(F, F.T, rtol=1e-5)


def test_fisher_matrix_positive_diagonal(fisher):
    """Diagonal entries of Fisher matrix are positive."""
    F = fisher.fisher_matrix
    assert jnp.all(jnp.diag(F) > 0)


def test_fisher_matrix_positive_definite(fisher):
    """Fisher matrix is positive definite (all eigenvalues > 0)."""
    F = fisher.fisher_matrix
    eigvals = jnp.linalg.eigvalsh(F)
    assert jnp.all(eigvals > 0), f"Non-positive eigenvalue: {float(eigvals.min())}"


def test_free_params_exclude_fixed(fisher):
    """T_dust is not in the free parameter list."""
    assert "T_dust" not in fisher.free_parameter_names
    assert "r" in fisher.free_parameter_names
    assert "A_dust" in fisher.free_parameter_names


# -----------------------------------------------------------------------
# Constraints
# -----------------------------------------------------------------------

def test_sigma_r_positive(fisher):
    """σ(r) is positive and finite."""
    sr = fisher.sigma("r")
    assert sr > 0
    assert np.isfinite(sr)


def test_sigma_conditional_leq_marginalised(fisher):
    """Conditional σ ≤ marginalised σ (marginalization adds uncertainty)."""
    for param in fisher.free_parameter_names:
        sc = fisher.sigma_conditional(param)
        sm = fisher.sigma(param)
        assert sc <= sm * (1 + 1e-6), \
            f"Conditional σ({param})={sc:.4e} > marginalised {sm:.4e}"


def test_sigma_r_improves_with_more_detectors(signal_model, instrument):
    """Doubling detector count should improve (lower) σ(r)."""
    fisher_base = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    sr_base = fisher_base.sigma("r")

    # More detectors
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    inst2 = Instrument(channels=(
        Channel(90.0,  1000, 400.0, 30.0, efficiency=eff),
        Channel(150.0, 2000, 300.0, 20.0, efficiency=eff),
        Channel(220.0, 1000, 500.0, 15.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)
    model2 = SignalModel(inst2, GaussianForegroundModel(), CMBSpectra(),
                         ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30)
    fisher2 = FisherForecast(
        model2, inst2, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    sr_more = fisher2.sigma("r")
    assert sr_more < sr_base, f"More detectors didn't help: {sr_more:.4e} >= {sr_base:.4e}"


def test_fixing_params_improves_sigma_r(signal_model, instrument):
    """Fixing more foreground params → fewer free params → lower σ(r)."""
    fisher_many_free = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    fisher_more_fixed = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust", "alpha_dust", "alpha_sync", "epsilon", "Delta_dust"],
    )
    sr_many = fisher_many_free.sigma("r")
    sr_fewer = fisher_more_fixed.sigma("r")
    assert sr_fewer < sr_many, \
        f"Fixing params didn't help: {sr_fewer:.4e} >= {sr_many:.4e}"


# -----------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------

def test_prior_tightens_constraint(signal_model, instrument):
    """Adding a tight prior on a degenerate parameter improves σ(r)."""
    fisher_no_prior = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={},
        fixed_params=["T_dust"],
    )
    fisher_with_prior = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.01},  # very tight prior
        fixed_params=["T_dust"],
    )
    sr_no = fisher_no_prior.sigma("r")
    sr_with = fisher_with_prior.sigma("r")
    assert sr_with <= sr_no * (1 + 1e-6)


# -----------------------------------------------------------------------
# 2D marginalized
# -----------------------------------------------------------------------

def test_marginalized_2d_structure(fisher):
    """marginalized_2d returns correct keys and consistent values."""
    result = fisher.marginalized_2d("r", "A_dust")
    assert "cov_2d" in result
    assert result["cov_2d"].shape == (2, 2)
    assert result["sigma_i"] == pytest.approx(fisher.sigma("r"), rel=1e-5)
    assert result["sigma_j"] == pytest.approx(fisher.sigma("A_dust"), rel=1e-5)
    assert -1.0 <= result["rho"] <= 1.0


# -----------------------------------------------------------------------
# Inverse consistency
# -----------------------------------------------------------------------

def test_fisher_times_inverse_is_identity(fisher):
    """F × F⁻¹ ≈ I."""
    F = fisher.fisher_matrix
    F_inv = fisher.inverse
    product = F @ F_inv
    assert jnp.allclose(product, jnp.eye(fisher.n_free), atol=1e-6)
