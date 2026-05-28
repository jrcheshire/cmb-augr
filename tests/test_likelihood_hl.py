"""Tests for the augr HL + Gaussian bandpower likelihoods (Phase A)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.likelihood import GaussianLikelihood, HLLikelihood, SignalSpectrumModel
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

FIDUCIAL = {
    "r": 0.01,
    "A_lens": 1.0,
    "A_dust": 4.7,
    "beta_dust": 1.6,
    "alpha_dust": -0.58,
    "T_dust": 19.6,
    "A_sync": 1.5,
    "beta_sync": -3.1,
    "alpha_sync": -0.6,
    "epsilon": 0.0,
    "Delta_dust": 0.0,
}


@pytest.fixture(scope="module")
def instrument():
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    return Instrument(
        channels=(
            Channel(90.0, 500, 400.0, 30.0, efficiency=eff),
            Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
            Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
        ),
        mission_duration_years=5.0,
        f_sky=0.7,
    )


@pytest.fixture(scope="module")
def signal_model(instrument):
    # Per-ℓ bins: bandpower_covariance_full (HL's dense M_f) reduces exactly to
    # FisherForecast's per-bin block path only for single-ℓ bins, so per-ℓ
    # binning lets the likelihood Fishers be cross-checked against FisherForecast.
    return SignalModel(
        instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=30,
        ell_max=130,
        delta_ell=1,
        ell_per_bin_below=1,
    )


@pytest.fixture(scope="module")
def fid_vec(signal_model):
    return flatten_params(FIDUCIAL, signal_model.parameter_names)


@pytest.fixture(scope="module")
def model(signal_model):
    return SignalSpectrumModel(signal_model)


@pytest.fixture(scope="module")
def hl(signal_model, instrument, fid_vec):
    return HLLikelihood.from_forecast(signal_model, instrument, fid_vec)


@pytest.fixture(scope="module")
def gauss(signal_model, instrument, fid_vec):
    return GaussianLikelihood.from_forecast(signal_model, instrument, fid_vec)


def _sigma_r_from_fisher(fisher: jax.Array, r_idx: int) -> float:
    cov = np.linalg.inv(np.asarray(fisher))
    return float(np.sqrt(cov[r_idx, r_idx]))


class TestAsimovPeak:
    def test_log_prob_zero_at_fiducial(self, hl, gauss, model, fid_vec):
        pred = model.predict(fid_vec)
        # Asimov: data == model at fiducial → residual 0 → log L = 0.
        assert abs(float(hl.log_prob(pred))) < 1e-8
        assert abs(float(gauss.log_prob(pred))) < 1e-8

    def test_log_prob_negative_off_fiducial(self, hl, gauss, model, fid_vec):
        off = fid_vec.at[0].add(0.05)  # bump r
        pred = model.predict(off)
        assert float(hl.log_prob(pred)) < -1e-6
        assert float(gauss.log_prob(pred)) < -1e-6


class TestOrderingConsistency:
    def test_bin_matrices_match_data_vector(self, signal_model, model, fid_vec):
        # The per-bin-matrix view must reproduce SignalModel.data_vector, pair by
        # pair, validating the spec-slowest / bin-fastest layout against augr.
        pred = model.predict(fid_vec)
        mats = pred.as_bin_matrices()  # (M, M, n_bins)
        dv = np.asarray(signal_model.data_vector(fid_vec))
        for i, j in signal_model.freq_pairs:
            sl = signal_model.spectrum_slice(i, j)
            np.testing.assert_allclose(np.asarray(mats[i, j]), dv[sl], rtol=1e-12)


class TestFisherEqualsKnox:
    """HL must not move the *Fisher*: HL curvature == Gaussian == augr Knox.

    The HL transform g(λ) is autodiff-safe but has a kink at λ=1 (the exact
    fiducial, where all per-bin eigenvalues are 1): the differentiate-through-
    ``where`` returns a finite but *zero* gradient there. So HL's curvature is
    validated by a finite-difference second derivative slightly off the peak
    (λ≠1, where g'=1 holds); the Gaussian (kink-free) gets the full autodiff
    marginalized check.
    """

    def test_gaussian_marginalized_sigma_r_equals_knox(
        self, signal_model, instrument, gauss, model, fid_vec
    ):
        # Gaussian residual is the raw data-vector difference -> autodiff is clean
        # and J^T C^-1 J is exactly the Knox Fisher; marginalized sigma(r) matches.
        r_idx = signal_model.parameter_names.index("r")
        ff = FisherForecast(signal_model, instrument, FIDUCIAL, priors={}, fixed_params=[])
        jac = jax.jacobian(lambda p: gauss.residual_vector(model.predict(p)))(fid_vec)
        fisher_g = jac.T @ gauss.cov_inv @ jac
        np.testing.assert_allclose(_sigma_r_from_fisher(fisher_g, r_idx), ff.sigma("r"), rtol=1e-3)

    def test_hl_conditional_curvature_matches_knox(
        self, signal_model, instrument, hl, gauss, model, fid_vec
    ):
        # Conditional curvature along r: -2 logL ≈ F[r,r] (Δr)^2 about the peak.
        # F[r,r] = 1 / sigma_conditional(r)^2. Central 2nd difference off the peak
        # sidesteps the g(1) kink.
        ff = FisherForecast(signal_model, instrument, FIDUCIAL, priors={}, fixed_params=[])
        r_idx = signal_model.parameter_names.index("r")
        sigma_cond = ff.sigma_conditional("r")
        knox_curv = 1.0 / sigma_cond**2
        h = 0.2 * sigma_cond

        def fd_curvature(like):
            lp = [
                float(like.log_prob(model.predict(fid_vec.at[r_idx].add(s * h))))
                for s in (+1.0, -1.0)
            ]
            return -(lp[0] + lp[1]) / h**2  # logL(fid)=0 → -(lp+ + lp-)/h^2

        # Gaussian recovers Knox to FD precision; HL shares the curvature near peak.
        np.testing.assert_allclose(fd_curvature(gauss), knox_curv, rtol=1e-2)
        np.testing.assert_allclose(fd_curvature(hl), knox_curv, rtol=2e-2)


class TestAutodiffSafety:
    def test_grad_finite_and_matches_fd_off_fiducial(
        self, signal_model, instrument, hl, model, fid_vec
    ):
        # _safe_g keeps the gradient finite once M is well off identity. A
        # generic offset of ~1σ_marg in every free parameter is comfortably away
        # from the Asimov degeneracy; the autodiff gradient is finite there and
        # matches a finite-difference reference along r.
        ff = FisherForecast(signal_model, instrument, FIDUCIAL, priors={}, fixed_params=[])
        sigmas = jnp.array([ff.sigma(n) for n in signal_model.parameter_names])
        off = fid_vec + sigmas

        def lp(p):
            return hl.log_prob(model.predict(p))

        g = jax.grad(lp)(off)
        assert np.all(np.isfinite(np.asarray(g)))
        r_idx = 0
        step = 1e-4
        fd = (float(lp(off.at[r_idx].add(step))) - float(lp(off.at[r_idx].add(-step)))) / (2 * step)
        np.testing.assert_allclose(float(g[r_idx]), fd, rtol=1e-3)

    def test_grad_is_nan_exactly_at_asimov(self, hl, model, fid_vec):
        # KNOWN EDGE (documents, not endorses): at the exact Asimov fiducial every
        # per-bin M = R·Ĉ·R = I has fully-degenerate eigenvalues, so eigh's
        # eigenvector gradient is NaN and 0·NaN poisons grad. The likelihood
        # *value* is still correct (peak = 0, see TestAsimovPeak). Phase B must
        # init the sampler off-fiducial / jitter the Asimov data. This test pins
        # the edge so a future change (e.g. a degeneracy-safe eigh) is noticed.
        g = jax.grad(lambda p: hl.log_prob(model.predict(p)))(fid_vec)
        assert np.all(np.isnan(np.asarray(g)))


class TestNonGaussianity:
    def test_hl_matches_gaussian_near_peak_then_diverges(
        self, signal_model, instrument, hl, gauss, model, fid_vec
    ):
        # Displace along r by k * conditional-sigma(r); near the peak HL and
        # Gaussian agree (shared curvature), and the gap grows with displacement
        # (the non-Gaussian skew that sampling will pick up in Phase B).
        ff = FisherForecast(signal_model, instrument, FIDUCIAL, priors={}, fixed_params=[])
        r_idx = signal_model.parameter_names.index("r")
        sigma_r_cond = ff.sigma_conditional("r")

        def gap(k):
            p = fid_vec.at[r_idx].add(k * sigma_r_cond)
            pred = model.predict(p)
            lp_hl = float(hl.log_prob(pred))
            lp_g = float(gauss.log_prob(pred))
            return lp_hl, lp_g, abs(lp_hl - lp_g)

        lp_hl_small, lp_g_small, gap_small = gap(0.5)
        _, _, gap_large = gap(3.0)

        # Near the peak the two likelihoods nearly coincide.
        np.testing.assert_allclose(lp_hl_small, lp_g_small, rtol=5e-2)
        # ... and the HL-vs-Gaussian discrepancy grows as we move away.
        assert gap_large > gap_small
