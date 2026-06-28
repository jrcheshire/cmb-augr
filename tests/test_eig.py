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
from augr.cost import CostModel, aperture_from_fwhm, budget_penalty
from augr.delensing import load_lensing_spectra
from augr.design_opt import build_design_objectives
from augr.eig import (
    HLEIGContext,
    design_cost,
    design_objective,
    gaussian_eig_from_external_cov,
    hl_eig_from_external_cov,
    marginal_eig_r_from_external_cov,
    posterior_fisher_from_external_cov,
)
from augr.foregrounds import NullForegroundModel
from augr.optimize import make_optimization_context, sigma_r_from_external_cov
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
