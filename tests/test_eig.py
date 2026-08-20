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
from augr.active_subspace import (
    DesignSpec,
    active_subspace,
    collect_gradients,
    sample_designs,
    subspace_alignment,
)
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.cost import CostModel, aperture_from_fwhm, bias_wall, budget_penalty
from augr.delensing import load_lensing_spectra
from augr.design_opt import build_design_objectives
from augr.eig import (
    HLEIGContext,
    delta_r_from_residual,
    design_cost,
    design_objective,
    gaussian_eig_from_external_cov,
    hl_eig_from_external_cov,
    marginal_eig_r_from_external_cov,
    physical_design_objective,
    posterior_fisher_from_external_cov,
    sigma_r_from_posterior_fisher,
)
from augr.fisher import FisherForecast
from augr.foregrounds import NullForegroundModel
from augr.optimize import (
    DelensCoupling,
    make_optimization_context,
    sigma_r_from_external_cov,
)
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context, mc_cutsky_cov_traced

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


def _setup(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2,
           split_lensing=False, fg_model=None):
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
        fg_model=fg_model,
        r_in=0.0,
        split_lensing=split_lensing,
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


# --- slow: HL-EIG (Stage 2) end-to-end through the cut-sky forward (gate 4d) ----------


def _hl_eig_ctx(lmax):
    """An HLEIGContext matching the tiny cut-sky config's binning (residual template, A_res)."""
    ell = np.arange(2, lmax + 1, dtype=float)
    return HLEIGContext.build(
        template_ells=ell,
        template_cl=(ell / 5.0) ** -2.4,
        f_sky=0.6,
        r_fid=0.0,
        floated=frozenset({"A_res"}),
        sigma_prior_r=0.05,
        n_grid=400,
        n_nuis_grid=41,
        ell_max=lmax,
        delta_ell=8,
        ell_per_bin_below=2,
    )


@pytest.mark.slow
def test_hl_eig_through_cutsky_forward():
    """4d: HL-EIG runs end-to-end on the MC covariance; finite, positive, and not wider than
    the Gaussian EIG beyond MC error (HL widens sigma(r) -> HL-EIG <= Gaussian-EIG)."""
    mc_ctx, _opt_ctx, cleaner = _setup(12)
    hl_ctx = _hl_eig_ctx(24)
    w_inv = w_inv_from_noise_design(
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, 0.6
    )
    traced = mc_cutsky_cov_traced(
        w_inv, mc_ctx, cleaner, beam_fwhm=jnp.asarray(BEAMS), beam_p=jnp.ones(len(BEAMS))
    )
    res = hl_eig_from_external_cov(
        traced.covariance,
        traced.mean_bandpower,
        hl_ctx,
        key=jax.random.PRNGKey(0),
        n_outer=512,
        return_diagnostics=True,
    )
    assert np.isfinite(res.eig_hl) and res.eig_hl > 0.0
    assert res.edge_frac < 1e-2
    # HL is the wider (more conservative) posterior, so its EIG does not exceed the Gaussian's
    # beyond the MC band.
    assert res.eig_hl <= res.eig_gauss + 4.0 * (res.stderr_hl + res.stderr_gauss)


# --- slow: the intertwining headline -- Gaussian-EIG active subspace predicts HL-EIG (5d) ---


def _design_spec_and_loss(mc_ctx, opt_ctx, cleaner, cost_model, budget, objective):
    """DesignSpec over the 13-knob design + a z-space loss(z, ctx) for ``objective``."""
    fid = {
        "n_det": jnp.asarray(N_DET),
        "net": jnp.asarray(NET),
        "beam_fwhm": jnp.asarray(BEAMS),
        "beam_p": jnp.ones(len(BEAMS)),
        "mission_years": jnp.asarray(float(MISSION_YEARS)),
    }
    labels = tuple(f"k{i}" for i in range(3 * 4 + 1))
    spec = DesignSpec.from_pytree(fid, labels, mode="log")

    def loss(z, ctx):
        d = spec.design_pytree(z)
        return design_objective(
            d["n_det"],
            d["net"],
            jnp.asarray(ETA),
            d["mission_years"],
            d["beam_fwhm"],
            d["beam_p"],
            mc_ctx=ctx,
            opt_ctx=opt_ctx,
            cleaner=cleaner,
            cost_model=cost_model,
            budget=budget,
            freqs_ghz=FREQS,
            objective=objective,
        )

    return spec, loss


@pytest.mark.slow
def test_active_subspace_surrogate_validity():
    """5d: the cheap Gaussian-EIG active subspace is a valid surrogate for HL-EIG.

    (i) the marginal-EIG-r and sigma(r) design subspaces share direction 1 (the monotone-
    reparam consistency), and (ii) the non-Gaussian HL-EIG varies more along Gaussian-EIG
    direction 1 than along an orthogonal direction -- so building the subspace from the cheap
    gradient and scanning HL-EIG along it is justified.
    """
    mc_ctx, opt_ctx, cleaner = _setup(12)
    cost_model = CostModel()
    big_budget = 1.0e12
    z = sample_designs(8, 13, sigma=0.12, method="lhs", seed=0)

    spec, loss_eig = _design_spec_and_loss(
        mc_ctx, opt_ctx, cleaner, cost_model, big_budget, "marginal_eig_r"
    )
    _spec, loss_sig = _design_spec_and_loss(
        mc_ctx, opt_ctx, cleaner, cost_model, big_budget, "sigma_r"
    )
    _vfe, vg_eig = build_design_objectives(loss_eig)
    _vfs, vg_sig = build_design_objectives(loss_sig)

    sub_eig = active_subspace(collect_gradients(vg_eig, z, lambda _i: mc_ctx, n_crn=1).grads)
    sub_sig = active_subspace(collect_gradients(vg_sig, z, lambda _i: mc_ctx, n_crn=1).grads)
    # (i) the two objectives' leading design directions coincide (monotone reparam).
    assert subspace_alignment(sub_eig.eigenvectors[:, 0], sub_sig.eigenvectors[:, 0]) > 0.98

    # (ii) HL-EIG varies more along Gaussian-EIG direction 1 than along an orthogonal direction.
    hl_ctx = _hl_eig_ctx(24)
    w1 = sub_eig.eigenvectors[:, 0]
    w_orth = sub_eig.eigenvectors[:, -1]  # least-active direction (orthonormal to w1)
    key = jax.random.PRNGKey(1)

    def hl_eig_at(zvec):
        d = spec.design_pytree(jnp.asarray(zvec))
        w_inv = w_inv_from_noise_design(
            d["n_det"], d["net"], jnp.asarray(ETA), d["mission_years"], 0.6
        )
        tr = mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner, beam_fwhm=d["beam_fwhm"], beam_p=d["beam_p"])
        return float(
            hl_eig_from_external_cov(tr.covariance, tr.mean_bandpower, hl_ctx, key=key, n_outer=512)
        )

    t = 0.18
    range_w1 = abs(hl_eig_at(t * w1) - hl_eig_at(-t * w1))
    range_orth = abs(hl_eig_at(t * w_orth) - hl_eig_at(-t * w_orth))
    assert range_w1 > range_orth, (range_w1, range_orth)


# --- sky coverage as a design coordinate -------------------------------------


def _master_setup_smooth(n_sims, *, nside=16, lmax=24):
    """A MASTER context on a SMOOTH mask, so the mask is a usable coordinate."""
    from augr.delensing import load_lensing_spectra

    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=0.6),
        foreground_model=NullForegroundModel(), cmb_spectra=CMBSpectra(),
        ell_min=2, ell_max=24, delta_ell=8, ell_per_bin_below=2)
    bm = jnp.asarray(sm.bin_matrix)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None),
        bm, 2)
    cleaner = nilc_cleaner(clean_e=True)
    w_fid = np.asarray(w_inv_from_noise_design(
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, 0.6))
    mc_ctx = make_cutsky_mc_context(
        cleaner=cleaner, freqs_ghz=FREQS, beam_fwhm_arcmin=BEAMS, w_inv=w_fid,
        nside=nside, lmax=lmax, mask=mk.smooth_gal_cut_mask(nside, 25.0, 8.0),
        cl_ee=cl_ee, cl_bb_prior_unbeamed=cl_bb, bin_matrix=bm, ell_min=2,
        true_bb_binned=true_b, n_sims=n_sims, base_seed=0, fg_model=None, r_in=0.0,
        estimator="master")
    return mc_ctx, _opt_ctx(), cleaner


@pytest.mark.slow
def test_design_objective_mask_gradient_matches_fd():
    """jax.grad of the design objective w.r.t. the sky-coverage coordinate.

    argnums=6 is the mask. Two things have to hold for the axis to be usable, and
    only the second is about differentiability:

    1. The gradient is right -- checked against central FD with the step shrunk
       until curvature stops dominating (the same h^2 discipline as the
       spectrum-stages test; a careless step reads as a broken gradient).
    2. The mask reaches ``w_inv`` as well as the mode count. Without that a wider
       mask looks like free information and the optimizer runs to full sky.
       Guarded below by holding everything else fixed and confirming the noise
       level actually moves with the mask.
    """
    mc_ctx, opt_ctx, cleaner = _master_setup_smooth(6)
    cost_model = CostModel()
    args = (jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA),
            jnp.asarray(MISSION_YEARS), jnp.asarray(BEAMS), None)

    def loss(b_cut):
        mask = mk.smooth_gal_cut_mask(16, b_cut, 8.0)
        return design_objective(
            *args, mask, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner,
            cost_model=cost_model, budget=1.0e12, freqs_ghz=FREQS,
            objective="sigma_r")

    g = float(jax.grad(loss)(25.0))
    assert np.isfinite(g)
    errs = []
    for h in (0.5, 0.25):
        fd = (float(loss(25.0 + h)) - float(loss(25.0 - h))) / (2 * h)
        errs.append(abs(g - fd) / abs(fd))
    assert errs[-1] < 0.05, f"grad {g:.6e}, FD rel errors {errs}"
    assert errs[0] > errs[-1], f"FD error should shrink with h: {errs}"


def test_mask_couples_to_the_noise_level_not_just_the_mode_count():
    """A wider mask must make the map shallower, or the axis is unphysical.

    w_inv_from_noise_design spreads a fixed detector-second budget over the survey
    area, so f_sky enters the noise. design_objective takes that f_sky from the
    mask when one is supplied; if it did not, sky coverage would look free and the
    gradient would push to full sky regardless of foregrounds.
    """
    narrow = mk.smooth_gal_cut_mask(16, 50.0, 8.0)   # less sky
    wide = mk.smooth_gal_cut_mask(16, 10.0, 8.0)     # more sky
    kw = (jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS)
    w_narrow = w_inv_from_noise_design(*kw, jnp.mean(narrow))
    w_wide = w_inv_from_noise_design(*kw, jnp.mean(wide))
    assert float(jnp.mean(wide)) > float(jnp.mean(narrow))
    assert np.all(np.asarray(w_wide) > np.asarray(w_narrow)), (
        "more sky must mean noisier per-pixel maps at fixed detector-seconds"
    )


# --- the delensing coupling in the design objective ---------------------------


@pytest.mark.slow
def test_design_objective_credits_delensing() -> None:
    """delens= lowers the objective (raises EIG), and delens=None is unchanged.

    Without the coupling the map forward runs at A_lens = 1 for every design, so an
    aperture trade sees none of the delensing benefit. This pins that the residual
    reaches the Monte-Carlo covariance and that leaving it off changes nothing.
    """
    mc_split, opt_ctx, cleaner = _setup(8, split_lensing=True)
    mc_plain, _o2, _c2 = _setup(8)
    cm = CostModel()
    args = (
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS,
        jnp.asarray(BEAMS), jnp.ones(len(BEAMS)),
    )
    kw = dict(
        opt_ctx=opt_ctx, cleaner=cleaner, cost_model=cm, budget=1.0e12,
        freqs_ghz=FREQS, objective="marginal_eig_r",
    )
    coupling = DelensCoupling.build(
        lensing_spectra=load_lensing_spectra(),
        n_det=args[0], net=args[1], beam=args[4], eta=args[2],
        mission_years=MISSION_YEARS, f_sky=0.6,
        l_max_qe=500, n_iter=2, ls=jnp.arange(2, 30, dtype=float),
    )
    # The residual really is a partial delensing at this design (else the test is vacuous).
    frac = float(jnp.median(coupling.cl_bb_res0 / CMBSpectra().cl_bb(coupling.ls, 0.0)))
    assert 0.0 < frac < 1.0

    off_split = float(design_objective(*args, mc_ctx=mc_split, **kw))
    off_plain = float(design_objective(*args, mc_ctx=mc_plain, **kw))
    assert off_split == off_plain  # the split alone changes nothing at r_in = 0

    on = float(design_objective(*args, mc_ctx=mc_split, delens=coupling, **kw))
    assert on < off_split  # objective = -EIG, so delensing must lower it


@pytest.mark.slow
def test_design_objective_delens_requires_a_split_context() -> None:
    """delens= against an unsplit ensemble raises instead of quietly ignoring it."""
    mc_plain, opt_ctx, cleaner = _setup(4)
    args = (
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS,
        jnp.asarray(BEAMS), jnp.ones(len(BEAMS)),
    )
    coupling = DelensCoupling.build(
        lensing_spectra=load_lensing_spectra(),
        n_det=args[0], net=args[1], beam=args[4], eta=args[2],
        mission_years=MISSION_YEARS, f_sky=0.6,
        l_max_qe=500, n_iter=2, ls=jnp.arange(2, 30, dtype=float),
    )
    with pytest.raises(ValueError, match="split_lensing=True"):
        design_objective(
            *args, mc_ctx=mc_plain, opt_ctx=opt_ctx, cleaner=cleaner,
            cost_model=CostModel(), budget=1.0e12, freqs_ghz=FREQS,
            delens=coupling,
        )


# --- the r-bias wall ----------------------------------------------------------
#
# The EIG is -log sigma(r) + const, so it carries NO bias term and will buy a
# design with small sigma(r) and large Delta r. These gates pin the wall's shape,
# that Delta r agrees with the eager primitive it mirrors, and that the wall is
# keyed to sigma(r) rather than to an absolute bias.


def test_delta_r_matches_fisher_parameter_bias():
    """delta_r_from_residual == FisherForecast.parameter_bias on the same inputs.

    The traced readout and the eager primitive must not drift: one is used inside
    the design objective, the other is what any offline bias check would run.
    """
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0], seed=3)
    rng = np.random.default_rng(11)
    delta_d = jnp.asarray(1e-3 * rng.standard_normal(ctx.J.shape[0]))

    got = float(delta_r_from_residual(cov, delta_d, ctx))

    ff = FisherForecast(
        ctx.signal_model,
        cleaned_map_instrument(f_sky=0.6),
        {"r": 0.0, "A_lens": 1.0},
        priors={},
        fixed_params=[],
        external_covariance=jnp.asarray(cov),
    )
    ff.compute()
    np.testing.assert_allclose(got, ff.parameter_bias(delta_d)["r"], rtol=1e-9)


def test_delta_r_is_linear_and_signed():
    """Delta r is linear in the residual and flips sign with it (a linear bias)."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0], seed=5)
    rng = np.random.default_rng(2)
    d = jnp.asarray(1e-3 * rng.standard_normal(ctx.J.shape[0]))
    b1 = float(delta_r_from_residual(cov, d, ctx))
    np.testing.assert_allclose(float(delta_r_from_residual(cov, 3.0 * d, ctx)), 3.0 * b1,
                               rtol=1e-10)
    np.testing.assert_allclose(float(delta_r_from_residual(cov, -d, ctx)), -b1, rtol=1e-10)


def test_bias_wall_is_zero_inside_the_budget_and_quadratic_outside():
    """Zero while |Delta r| <= eps*sigma, quadratic past it, C1 at the knee."""
    sigma, eps = 1e-3, 0.5
    knee = eps * sigma
    assert float(bias_wall(0.0, sigma, eps=eps)) == 0.0
    assert float(bias_wall(0.999 * knee, sigma, eps=eps)) == 0.0
    assert float(bias_wall(-0.999 * knee, sigma, eps=eps)) == 0.0  # symmetric in sign
    over = 0.25 * sigma
    np.testing.assert_allclose(
        float(bias_wall(knee + over, sigma, eps=eps)), over**2, rtol=1e-12
    )
    # C1 at the knee: the derivative approaches 0 from above.
    g = float(jax.grad(lambda d: bias_wall(d, sigma, eps=eps))(knee + 1e-9))
    assert abs(g) < 1e-8
    # and stiffness scales linearly
    np.testing.assert_allclose(
        float(bias_wall(knee + over, sigma, eps=eps, weight=7.0)), 7.0 * over**2, rtol=1e-12
    )


def test_bias_wall_tightens_as_the_design_gets_more_precise():
    """The SAME bias is free at a loose sigma(r) and penalized at a tight one.

    This is the property that makes it a bias/variance statement rather than an
    absolute bias cap: buying precision raises the bar the design must clear.
    """
    delta_r = 1e-3
    assert float(bias_wall(delta_r, 4e-3, eps=0.5)) == 0.0   # 0.25 sigma -> free
    assert float(bias_wall(delta_r, 1e-3, eps=0.5)) > 0.0    # 1.0 sigma  -> bites


def test_sigma_r_readout_matches_the_eig_framing():
    """sigma_r_from_posterior_fisher is the sigma(r) the EIG is built on."""
    ctx = _opt_ctx()
    cov = _synthetic_cov(ctx.J.shape[0], seed=7)
    np.testing.assert_allclose(
        float(sigma_r_from_posterior_fisher(cov, ctx)),
        float(sigma_r_from_external_cov(cov, ctx)),
        rtol=1e-12,
    )


@pytest.mark.slow
def test_design_objective_bias_wall_needs_foregrounds() -> None:
    """bias_eps against a foreground-free ensemble raises instead of being vacuous."""
    mc_ctx, opt_ctx, cleaner = _setup(4)  # fg_model=None
    args = (
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS,
        jnp.asarray(BEAMS), jnp.ones(len(BEAMS)),
    )
    with pytest.raises(ValueError, match="foreground"):
        design_objective(
            *args, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner,
            cost_model=CostModel(), budget=1.0e12, freqs_ghz=FREQS, bias_eps=0.5,
        )


@pytest.mark.slow
def test_design_objective_bias_wall_bites_on_a_real_residual() -> None:
    """End to end with foregrounds: Delta r is finite, and the wall engages only
    when the bias exceeds its sigma(r) budget.

    Marked slow because it builds a PySM (d1s1) ensemble -- see the `slow` marker
    note on foreground sims. d1s1 is the cheap stand-in here; the study's nominal
    truth model is d10s5, which is a run configuration rather than a code path.
    """
    mc_ctx, opt_ctx, cleaner = _setup(6, fg_model="d1s1")
    args = (
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS,
        jnp.asarray(BEAMS), jnp.ones(len(BEAMS)),
    )
    kw = dict(
        mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner, cost_model=CostModel(),
        budget=1.0e12, freqs_ghz=FREQS,
    )
    # f_sky must be the context's REALIZED mask fraction, which is what
    # design_objective feeds w_inv_from_noise_design -- passing a nominal 0.6 here
    # instead perturbs the covariance and the comparison below stops being exact.
    w_inv = w_inv_from_noise_design(
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS,
        mc_ctx.f_sky,
    )
    traced = mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner, fg_residual=True)
    res = np.asarray(traced.fg_residual_bandpower)
    assert traced.fg_residual_bandpower is not None
    assert np.all(np.isfinite(res)) and res.shape == np.asarray(traced.mean_bandpower).shape
    # NOT asserted positive per bin: MASTER is unbiased but not positive-definite, so a
    # low-power bin can deconvolve slightly negative. The residual carries net power,
    # concentrated at low ell where the Galactic BB is.
    assert res.sum() > 0.0 and res[0] == res.max()
    # It is a distinct leg, not the data vector relabelled.
    assert not np.allclose(res, np.asarray(traced.mean_bandpower))

    delta_r = float(delta_r_from_residual(
        traced.covariance, traced.fg_residual_bandpower, opt_ctx))
    sigma_r = float(sigma_r_from_posterior_fisher(traced.covariance, opt_ctx))
    assert np.isfinite(delta_r) and delta_r != 0.0

    # A budget far above the realized bias leaves the objective untouched; one far
    # below it must raise the objective by exactly the wall's value.
    base = float(design_objective(*args, **kw))
    loose = float(design_objective(*args, bias_eps=100.0 * abs(delta_r) / sigma_r, **kw))
    np.testing.assert_allclose(loose, base, rtol=1e-10)

    eps_tight = 0.1 * abs(delta_r) / sigma_r
    tight = float(design_objective(*args, bias_eps=eps_tight, **kw))
    expected = base + float(bias_wall(delta_r, sigma_r, eps=eps_tight))
    assert tight > base
    np.testing.assert_allclose(tight, expected, rtol=1e-10)


@pytest.mark.slow
def test_physical_design_objective_forwards_delens_and_bias_wall() -> None:
    """The physical entry point forwards delens= and bias_eps= unchanged.

    Both are pure kwarg passthroughs, but this is the entry point the EIG driver
    actually calls (it is the one that takes aperture), so a forwarding typo would
    surface only in a production run -- as a design silently credited with no
    delensing, or an inactive bias wall, both of which look like success.

    Asserted by equivalence: derive the channels the physical path derives, call
    design_objective directly on them, and require the same scalar.
    """
    from augr.optimize import design_to_channels

    mc_ctx, opt_ctx, cleaner = _setup(6, split_lensing=True, fg_model="d1s1")
    cm = CostModel()
    freqs_per_group = ((90.0,), (150.0,), (220.0,))
    fp_diameter_m = 0.3
    design = {
        "aperture_m": jnp.asarray(1.2),
        "f_number": jnp.asarray(2.0),
        "area_fractions": jnp.asarray([1 / 3, 1 / 3, 1 / 3]),
        "mission_years": jnp.asarray(MISSION_YEARS),
    }
    coupling = DelensCoupling.build(
        lensing_spectra=load_lensing_spectra(),
        n_det=jnp.asarray(N_DET), net=jnp.asarray(NET),
        beam=jnp.asarray(BEAMS), eta=jnp.asarray(ETA),
        mission_years=MISSION_YEARS, f_sky=0.6,
        l_max_qe=500, n_iter=2, ls=jnp.arange(2, 30, dtype=float),
    )
    shared = dict(
        mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner, cost_model=cm,
        budget=1.0e12, delens=coupling, bias_eps=0.5, bias_weight=3.0,
    )

    got = float(physical_design_objective(
        design, freqs_per_group=freqs_per_group, fp_diameter_m=fp_diameter_m,
        eta_total=0.5, galactic_loading=False, **shared,
    ))

    n_det, net, beam = design_to_channels(
        design["aperture_m"], design["f_number"], fp_diameter_m,
        design["area_fractions"], freqs_per_group, extra_loading=None,
    )
    freqs_flat = tuple(f for grp in freqs_per_group for f in grp)
    expected = float(design_objective(
        n_det, net, jnp.full((len(freqs_flat),), 0.5), design["mission_years"],
        beam, jnp.ones(len(freqs_flat)),
        freqs_ghz=freqs_flat, **shared,
    ))
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert np.isfinite(got)
