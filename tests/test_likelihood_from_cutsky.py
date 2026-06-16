"""Tests for the cut-sky-MC -> inference-layer bridge (Direction A).

The fast tests pin the *constructors* without any sampling or component
separation: a synthetic ``CutskyMC`` is built from the analytic Knox covariance,
so ``from_external`` fed that covariance must reproduce ``from_forecast``
bit-for-bit, every likelihood must peak at the fiducial, and the KS decider must
identify a chi^2 vs Gaussian ensemble. The slow test runs the full
``hl_forecast_from_cutsky_mc`` orchestrator through a short NUTS to pin the four
sigma(r) gates (Gaussian-NUTS == Gaussian-Fisher parity, HL widening, HL-NUTS ==
HL-profile, convergence) — marked ``slow`` and behind the ``[sampling]`` extra.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from augr.covariance import _build_M, bandpower_covariance_full
from augr.likelihood import (
    GaussianLikelihood,
    HLLikelihood,
    MCCalibratedLikelihood,
    bandpower_ks,
    build_cutsky_signal_model,
    build_likelihood,
    posterior_from_cutsky_mc,
)
from augr.likelihood.ordering import SpectrumLayout, matrices_to_spectra
from augr.likelihood.protocols import SignalSpectrumModel
from augr.signal import flatten_params

R_FID = 0.01


class _SyntheticMC:
    """Duck-typed ``CutskyMC``: analytic Knox covariance + a chi^2 ensemble.

    Lets the bridge be tested without the ``[compsep]`` extra or a real masked-Wiener
    Monte-Carlo ensemble. ``mean_bandpower`` is the analytic total S+N (= the binned
    ``_build_M`` diagonal), so ``from_external`` fed ``covariance`` reproduces the
    analytic-Knox ``from_forecast`` exactly.
    """

    def __init__(self, covariance, mean_bandpower, debiased_bandpowers):
        self.covariance = covariance
        self.mean_bandpower = mean_bandpower
        self.debiased_bandpowers = debiased_bandpowers


def _make_case(*, ell_max=120, delta_ell=10, n_sims=200, seed=0):
    """Build a cleaned-map SignalModel + a synthetic MC (analytic Knox cov + chi^2 ensemble)."""
    ells = jnp.arange(2, 260, dtype=float)
    template_cl = 5e-5 * (ells / 80.0) ** (-0.4)
    sm, inst = build_cutsky_signal_model(
        ells, template_cl, f_sky=0.6, ell_min=2, ell_max=ell_max, delta_ell=delta_ell
    )
    fid = {"r": R_FID, "A_lens": 1.0, "A_res": 1.0}
    fid_vec = flatten_params(fid, sm.parameter_names)
    layout = SpectrumLayout.from_freq_pairs(sm.freq_pairs, sm.n_bins)
    total = np.asarray(matrices_to_spectra(_build_M(sm, inst, fid_vec), layout))
    cov = np.asarray(bandpower_covariance_full(sm, inst, fid_vec))
    var = np.diag(cov)
    nu = 2.0 * total**2 / var
    rng = np.random.default_rng(seed)
    ens = np.stack(
        [(total[b] / nu[b]) * rng.chisquare(nu[b], size=n_sims) for b in range(sm.n_bins)],
        axis=1,
    )
    mc = _SyntheticMC(cov, total, ens)
    return sm, inst, fid, fid_vec, mc, ells, template_cl


# --- Fast: constructors -----------------------------------------------------


def test_gaussian_from_external_matches_from_forecast():
    sm, inst, _fid, fid_vec, mc, *_ = _make_case()
    g_ext = GaussianLikelihood.from_external(sm, fid_vec, mc.covariance)
    g_for = GaussianLikelihood.from_forecast(sm, inst, fid_vec)
    np.testing.assert_allclose(np.asarray(g_ext.cov_inv), np.asarray(g_for.cov_inv), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(g_ext.data), np.asarray(g_for.data), rtol=0, atol=0)


def test_hl_from_external_matches_from_forecast():
    # Fed total = binned _build_M (analytic S+N) and the analytic Knox cov, from_external
    # must reproduce from_forecast: same log_prob at several off-fiducial points.
    sm, inst, _fid, fid_vec, mc, *_ = _make_case()
    hl_ext = HLLikelihood.from_external(sm, fid_vec, mc.mean_bandpower, mc.covariance)
    hl_for = HLLikelihood.from_forecast(sm, inst, fid_vec)
    # Constructor fields agree to fp64 (noise reconstruction total - data_vector).
    for nm in ("m_f_inv", "c_fl_12", "data_matrices"):
        np.testing.assert_allclose(
            np.asarray(getattr(hl_ext, nm)), np.asarray(getattr(hl_for, nm)), rtol=0, atol=0
        )
    np.testing.assert_allclose(
        np.asarray(hl_ext.noise_matrices), np.asarray(hl_for.noise_matrices), rtol=1e-10, atol=1e-18
    )
    model = SignalSpectrumModel(sm)
    for r in (0.0, 0.03, 0.08):
        pred = model.predict(
            flatten_params({"r": r, "A_lens": 1.0, "A_res": 1.2}, sm.parameter_names)
        )
        np.testing.assert_allclose(
            float(hl_ext.log_prob(pred)), float(hl_for.log_prob(pred)), rtol=1e-9, atol=1e-12
        )


def test_all_likelihoods_peak_at_fiducial():
    sm, _inst, _fid, fid_vec, mc, *_ = _make_case()
    model = SignalSpectrumModel(sm)
    pred_fid = model.predict(fid_vec)
    pred_off = model.predict(
        flatten_params({"r": 0.06, "A_lens": 1.0, "A_res": 1.0}, sm.parameter_names)
    )
    for kind in ("gaussian", "hl", "mc_calibrated"):
        lik = build_likelihood(kind, sm, fid_vec, mc)
        lp_fid = float(lik.log_prob(pred_fid))
        lp_off = float(lik.log_prob(pred_off))
        assert abs(lp_fid) < 1e-9, f"{kind} does not peak at fiducial: {lp_fid}"
        assert lp_off < lp_fid, f"{kind} not lower off-fiducial"


def test_non_gaussian_likelihoods_widen():
    # At a fixed off-fiducial r the HL / MC-calibrated forms penalize *less* than the
    # Gaussian (the asymmetric low-mode skew) -> a wider sigma(r). Directional gate.
    sm, _inst, _fid, fid_vec, mc, *_ = _make_case(ell_max=50, delta_ell=8)
    model = SignalSpectrumModel(sm)
    pred = model.predict(
        flatten_params({"r": 0.05, "A_lens": 1.0, "A_res": 1.0}, sm.parameter_names)
    )
    lp = {
        k: float(build_likelihood(k, sm, fid_vec, mc).log_prob(pred))
        for k in ("gaussian", "hl", "mc_calibrated")
    }
    assert lp["hl"] > lp["gaussian"]
    assert lp["mc_calibrated"] > lp["gaussian"]


def test_mc_calibrated_rejects_multifield():
    sm, _inst, _fid, fid_vec, _mc, *_ = _make_case()

    class _MultiField:
        # Duck-typed signal model with n_field = 2 (freq_pairs span two channels).
        freq_pairs = ((0, 0), (0, 1), (1, 1))
        n_bins = sm.n_bins

        def data_vector(self, params):
            return jnp.zeros(3 * self.n_bins)

    with pytest.raises(ValueError, match="single-field"):
        MCCalibratedLikelihood.from_external(
            _MultiField(), fid_vec, jnp.ones(3 * sm.n_bins), jnp.eye(3 * sm.n_bins)
        )


def test_posterior_builds_and_finite_off_fiducial():
    sm, _inst, fid, _fid_vec, mc, *_ = _make_case()
    for kind in ("gaussian", "hl", "mc_calibrated"):
        post, transform, free_names, _lik = posterior_from_cutsky_mc(
            mc, sm, fid, likelihood_kind=kind
        )
        assert list(free_names) == ["r", "A_lens", "A_res"]
        u = post.fiducial_unconstrained(transform) + 0.2 * jnp.ones(len(free_names))
        assert np.isfinite(float(post.log_prob(u)))


# --- Fast: KS decider -------------------------------------------------------


def test_bandpower_ks_recommends_hl_for_chi2_ensemble():
    nb = 8
    total = np.linspace(5e-3, 1e-1, nb)
    nu = np.linspace(3, 60, nb)
    var = 2.0 * total**2 / nu
    cov = np.diag(var)
    rng = np.random.default_rng(0)
    ens = np.stack([(total[b] / nu[b]) * rng.chisquare(nu[b], size=400) for b in range(nb)], axis=1)
    ks = bandpower_ks(ens, total, cov)
    assert ks["recommend"] == "hl"
    assert ks["gauss_rejected_bump"] is True
    assert ks["chi2_rejected_bump"] is False
    np.testing.assert_allclose(ks["nu_eff"], nu, rtol=0.2)  # recovered effective modes


def test_bandpower_ks_recommends_gaussian_for_normal_ensemble():
    nb = 8
    total = np.linspace(5e-3, 1e-1, nb)
    var = (0.1 * total) ** 2
    cov = np.diag(var)
    rng = np.random.default_rng(1)
    ens = np.stack(
        [total[b] + np.sqrt(var[b]) * rng.standard_normal(400) for b in range(nb)], axis=1
    )
    ks = bandpower_ks(ens, total, cov)
    # At high effective mode count the chi^2 and Gaussian shapes coincide, so the
    # chi^2 form is never *materially* better -> the verdict stays "gaussian" even if
    # an unlucky bump bin trips the (Bonferroni-corrected) Gaussian rejection.
    assert ks["recommend"] == "gaussian"
    assert ks["hl_better_bump"] is False


# --- Slow: the full orchestrator through NUTS -------------------------------


@pytest.mark.slow
def test_hl_forecast_from_cutsky_mc_gates():
    pytest.importorskip("blackjax")
    pytest.importorskip("optax")
    from augr.likelihood import hl_forecast_from_cutsky_mc

    # Low-mode bump config (ell 2-50): the regime where HL is non-Gaussian.
    _sm, _inst, _fid, _fid_vec, mc, ells, template_cl = _make_case(ell_max=50, delta_ell=8)
    res = hl_forecast_from_cutsky_mc(
        mc,
        template_ells=ells,
        template_cl=template_cl,
        f_sky=0.6,
        r_fid=R_FID,
        ell_min=2,
        ell_max=50,
        delta_ell=8,
        n_chains=2,
        num_warmup=300,
        num_samples=500,
        n_starts_mle=4,
        profile=True,
        profile_n_grid=11,
        seed=0,
    )
    # Parity: sampling the Gaussian likelihood reproduces the Gaussian-Fisher sigma(r).
    np.testing.assert_allclose(res.sigma_r_gauss_nuts, res.sigma_r_gauss_fisher, rtol=0.12)
    # Headline: HL is wider than both Gaussian-NUTS and Gaussian-Fisher.
    assert res.sigma_r_hl_nuts > res.sigma_r_gauss_nuts
    assert res.sigma_r_hl_nuts > res.sigma_r_gauss_fisher
    # The two HL estimators (sampling vs profile) agree.
    np.testing.assert_allclose(res.sigma_r_hl_nuts, res.sigma_r_hl_profile, rtol=0.2)
    # Convergence + KS verdict (the synthetic ensemble is chi^2 -> HL).
    assert res.converged_gauss
    assert res.converged_hl
    assert res.ks["recommend"] == "hl"
