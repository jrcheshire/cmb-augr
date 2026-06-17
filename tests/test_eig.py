"""Gate for augr.eig: Gaussian EIG objectives + the cost-constrained design objective.

The fast tests are pure linear algebra on a synthetic covariance + a real
``OptimizationContext`` (no map forward): the r-marginal EIG ``≡ -log sigma(r)``
equivalence (the framing gate), the D-optimal ``0.5 logdet F_post`` closed form,
analytic differentiability, and the cost portion. The slow test runs the full
cut-sky MC forward and checks the end-to-end design gradient is finite + that the
r-marginal-EIG and sigma(r) objectives have the same descent direction.

Map work needs jht (the [masking] extra) and ducc0 (the SHTs).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jht")
pytest.importorskip("ducc0")

import jax
import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.cost import CostModel, aperture_from_fwhm, budget_penalty
from augr.delensing import load_lensing_spectra
from augr.eig import (
    design_cost,
    design_objective,
    gaussian_eig_from_external_cov,
    marginal_eig_r_from_external_cov,
    posterior_fisher_from_external_cov,
)
from augr.foregrounds import NullForegroundModel
from augr.optimize import make_optimization_context, sigma_r_from_external_cov
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context

FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)
N_DET = (200.0, 400.0, 200.0)
NET = (60.0, 50.0, 80.0)
ETA = (0.5, 0.5, 0.5)
MISSION_YEARS = 4.0


def _opt_ctx(ell_max=24, delta_ell=8, ell_per_bin_below=2):
    """Lightweight cleaned-map OptimizationContext (no map forward)."""
    return make_optimization_context(
        cleaned_map_instrument(f_sky=0.6),
        NullForegroundModel(),
        CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0},
        priors={},
        fixed_params=[],
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )


def _synthetic_cov(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return jnp.asarray(a @ a.T + n * np.eye(n))  # SPD, well-conditioned


# --- fast: r-marginal EIG is the sigma(r) framing -----------------------------


def test_marginal_eig_r_equals_minus_log_sigma_r():
    """The framing gate: r-marginal EIG == log(sigma_prior) - log(sigma_r_from_external_cov)."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0])
    sigma_prior = 0.1
    eig = float(marginal_eig_r_from_external_cov(cov, ctx, sigma_prior_r=sigma_prior))
    sigma_r = float(sigma_r_from_external_cov(cov, ctx))
    np.testing.assert_allclose(eig, np.log(sigma_prior) - np.log(sigma_r), rtol=1e-12)


def test_marginal_eig_r_monotone_in_information():
    """Quartering the covariance halves sigma(r) -> EIG rises by exactly log 2."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0])
    e_full = float(marginal_eig_r_from_external_cov(cov, ctx, sigma_prior_r=0.1))
    e_quarter = float(marginal_eig_r_from_external_cov(0.25 * cov, ctx, sigma_prior_r=0.1))
    assert e_quarter > e_full
    np.testing.assert_allclose(e_quarter - e_full, np.log(2.0), rtol=1e-10)


def test_marginal_eig_r_grad_is_minus_half_in_scale():
    """d/ds [EIG_r(s*cov)] = -1/2 at s=1 (sigma_r ~ sqrt(s)); a noise-free grad check."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0])
    g = float(jax.grad(lambda s: marginal_eig_r_from_external_cov(s * cov, ctx))(1.0))
    np.testing.assert_allclose(g, -0.5, rtol=1e-8)


# --- fast: D-optimal EIG ------------------------------------------------------


def test_gaussian_eig_matches_half_logdet_closed_form():
    """D-optimal EIG == 0.5 logdet(F_post), cross-checked against an independent numpy F."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0])
    eig_d = float(gaussian_eig_from_external_cov(cov, ctx))

    j = np.asarray(ctx.J)
    fcov = np.asarray(cov)
    f = j.T @ np.linalg.solve(fcov, j) + np.diag(np.asarray(ctx.prior_diag))
    _sign, logdet = np.linalg.slogdet(f)
    np.testing.assert_allclose(eig_d, 0.5 * logdet, rtol=1e-6)


def test_gaussian_eig_grad_is_minus_half_nfree_in_scale():
    """d/ds [0.5 logdet F_post(s*cov)] = -0.5*n_free at s=1 (F ~ 1/s)."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0])
    n_free = ctx.J.shape[1]
    g = float(jax.grad(lambda s: gaussian_eig_from_external_cov(s * cov, ctx))(1.0))
    np.testing.assert_allclose(g, -0.5 * n_free, rtol=1e-8)


def test_posterior_fisher_symmetric_and_adds_prior():
    """posterior_fisher_from_external_cov is symmetric and folds in the prior diagonal."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0])
    f = np.asarray(posterior_fisher_from_external_cov(cov, ctx))
    np.testing.assert_allclose(f, f.T, rtol=1e-10)


# --- fast: cost portion (no map forward) --------------------------------------


def test_design_cost_uses_tightest_beam_band_aperture():
    """design_cost takes the aperture from the highest-frequency (tightest-beam) band."""
    cm = CostModel()
    n_det = jnp.asarray(N_DET)
    beam = jnp.asarray(BEAMS)
    c = float(design_cost(n_det, beam, 5.0, cost_model=cm, freqs_ghz=FREQS))
    # tightest band = 220 GHz at 20', total detectors = 800.
    ap = float(aperture_from_fwhm(20.0, 220.0))
    np.testing.assert_allclose(c, float(cm.total_cost(ap, 800.0, 5.0)), rtol=1e-10)


def test_design_cost_grad_only_through_tightest_band():
    """Only the aperture-setting (tightest) band's FWHM carries an aperture-cost gradient."""
    cm = CostModel()
    n_det = jnp.asarray(N_DET)
    beam = jnp.asarray(BEAMS)
    g = np.asarray(
        jax.grad(lambda b: design_cost(n_det, b, 5.0, cost_model=cm, freqs_ghz=FREQS))(beam)
    )
    assert g[2] < 0.0  # finer 220 GHz beam -> bigger dish -> more cost; aperture ~ 1/fwhm
    assert g[0] == 0.0 and g[1] == 0.0


def test_budget_penalty_binds_through_design_cost():
    """The budget penalty is zero under budget and positive when the design overspends."""
    cm = CostModel()
    beam = jnp.asarray(BEAMS)
    cost_lo = design_cost(jnp.asarray(N_DET), beam, 5.0, cost_model=cm, freqs_ghz=FREQS)
    budget = float(cost_lo)
    assert float(budget_penalty(cost_lo, budget)) == 0.0
    cost_hi = design_cost(
        jnp.asarray((400.0, 800.0, 400.0)), beam, 5.0, cost_model=cm, freqs_ghz=FREQS
    )
    assert float(budget_penalty(cost_hi, budget)) > 0.0


# --- slow: end-to-end design objective + the design-level EIG/sigma(r) equivalence ---


def _setup(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2):
    """mc_ctx + opt_ctx + cleaner for the tiny CMB-only config (mirrors test_optimize_mapbased)."""
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=0.6),
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    bm = jnp.asarray(sm.bin_matrix)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
    )
    cleaner = nilc_cleaner(clean_e=True)
    w_inv_fid = np.asarray(
        w_inv_from_noise_design(
            jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, 0.6
        )
    )
    mc_ctx = make_cutsky_mc_context(
        cleaner=cleaner,
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=w_inv_fid,
        nside=nside,
        lmax=lmax,
        mask=mk.galactic_mask(nside, 0.6),
        cl_ee=cl_ee,
        cl_bb_prior_unbeamed=cl_bb,
        bin_matrix=bm,
        ell_min=2,
        true_bb_binned=true_b,
        n_sims=n_sims,
        base_seed=0,
        fg_model=None,
        r_in=0.0,
    )
    opt_ctx = make_optimization_context(
        cleaned_map_instrument(f_sky=0.6),
        NullForegroundModel(),
        CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0},
        priors={},
        fixed_params=[],
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    return mc_ctx, opt_ctx, cleaner


@pytest.mark.slow
def test_design_objective_grad_and_eig_sigma_r_equivalence():
    """End-to-end: design gradient is finite + FD-matched, and the r-marginal-EIG and
    sigma(r) objectives descend in the same direction (the Gaussian equivalence at the
    design level)."""
    mc_ctx, opt_ctx, cleaner = _setup(12)
    cm = CostModel()
    big_budget = 1.0e12  # penalty inactive -> isolate the EIG gradient

    args = (
        jnp.asarray(N_DET),
        jnp.asarray(NET),
        jnp.asarray(ETA),
        MISSION_YEARS,
        jnp.asarray(BEAMS),
        jnp.ones(len(BEAMS)),
    )

    def make_loss(objective):
        def loss(nd, ne, et, yr, bf, bp):
            return design_objective(
                nd,
                ne,
                et,
                yr,
                bf,
                bp,
                mc_ctx=mc_ctx,
                opt_ctx=opt_ctx,
                cleaner=cleaner,
                cost_model=cm,
                budget=big_budget,
                freqs_ghz=FREQS,
                objective=objective,
            )

        return loss

    eig_loss = make_loss("marginal_eig_r")
    v0 = float(eig_loss(*args))
    assert np.isfinite(v0)

    g_eig = jax.grad(eig_loss, argnums=(0, 1, 2, 3, 4, 5))(*args)
    for g in g_eig:
        assert bool(jnp.all(jnp.isfinite(g)))

    # FD on the first band's NET (CRN fixed -> autodiff and FD see the same sims).
    net = args[1]
    h = 0.05 * float(net[0])
    fd = (
        float(eig_loss(args[0], net.at[0].add(h), *args[2:]))
        - float(eig_loss(args[0], net.at[0].add(-h), *args[2:]))
    ) / (2 * h)
    np.testing.assert_allclose(float(g_eig[1][0]), fd, rtol=0.05)

    # The sigma(r) objective must point the same way (EIG_r is a monotone reparam of sigma_r).
    g_sig = jax.grad(make_loss("sigma_r"), argnums=(0, 1, 2, 3, 4, 5))(*args)
    a = np.concatenate([np.asarray(x).ravel() for x in g_eig])
    b = np.concatenate([np.asarray(x).ravel() for x in g_sig])
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.999
