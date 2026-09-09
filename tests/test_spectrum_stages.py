"""Gate for augr.spectrum_stages: the cut-sky masked-Wiener Monte-Carlo spectrum stage.

Fast tests cover the prior beaming, the single-map estimator wrapper shape, and the
MC driver's output shapes / covariance positivity / Hartlap guard at a tiny nside with
``fg_model=None`` (no PySM). The slow test is the science checkpoint: the E→B leakage
template through the *full spin-2 cleaner* must sit well below the lensing-BB floor (the
purity null), so the leaked-E cosmic variance does not dominate σ(r).

Map work needs jht (the [masking] extra) and ducc0 (the SHTs).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jht")
pytest.importorskip("ducc0")

import jax
import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.delensing import load_lensing_spectra
from augr.foregrounds import NullForegroundModel
from augr.instrument import beam_bl
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import (
    _edges_from_bin_matrix,
    beamed_prior,
    cutsky_bb_bandpower,
    make_cutsky_mc_context,
    mc_cutsky_bandpowers,
    mc_cutsky_cov_traced,
)

FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)
W_INV = (1e-4, 8e-5, 1.2e-4)


def _bin_matrix(ell_min, ell_max, delta_ell, ell_per_bin_below, f_sky=0.6):
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=f_sky),
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=ell_min,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    return jnp.asarray(sm.bin_matrix)


def _priors(lmax):
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    return cl_ee, cl_bb


# --- fast --------------------------------------------------------------------


def test_beamed_prior_scales_by_bc_squared() -> None:
    lmax = 24
    cl = jnp.ones(lmax + 1)
    bc = 30.0
    out = beamed_prior(cl, bc, lmax)
    expected = beam_bl(jnp.arange(lmax + 1, dtype=float), bc) ** 2
    assert out.shape == (lmax + 1,)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-12)


def test_cutsky_bb_bandpower_shape_finite() -> None:
    import jax

    nside, lmax = 16, 24
    cl_ee, cl_bb = _priors(lmax)
    bc = float(min(BEAMS))
    qu = jax.random.normal(jax.random.PRNGKey(0), (2, 12 * nside * nside)) * 1e-2
    mask = mk.galactic_mask(nside, 0.6)
    invn = mk.inv_noise_map(jnp.ones(12 * nside * nside), 1e-4, mask=mask)
    bm = _bin_matrix(2, 24, 8, 2)
    out = cutsky_bb_bandpower(
        qu,
        invn,
        beamed_prior(cl_ee, bc, lmax),
        beamed_prior(cl_bb, bc, lmax),
        bin_matrix=bm,
        ell_min=2,
        nside=nside,
        lmax=lmax,
    )
    assert out.shape == (bm.shape[0],)
    assert bool(jnp.all(jnp.isfinite(out)))


def _run_mc(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2, workers=1):
    cl_ee, cl_bb = _priors(lmax)
    bm = _bin_matrix(2, ell_max, delta_ell, ell_per_bin_below)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
    )
    return mc_cutsky_bandpowers(
        cleaner=nilc_cleaner(clean_e=True),
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=W_INV,
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
        workers=workers,
    )


@pytest.mark.slow
def test_mc_workers_parity() -> None:
    """workers>1 (spawn pool, picklable nilc_cleaner) is bit-identical to serial (per-sim CRN)."""
    serial = _run_mc(8, workers=1)
    parallel = _run_mc(8, workers=2)
    assert np.array_equal(serial.debiased_bandpowers, parallel.debiased_bandpowers)
    assert np.array_equal(serial.rec_full, parallel.rec_full)


@pytest.mark.slow
def test_mc_cutsky_bandpowers_shapes_and_covariance() -> None:
    res = _run_mc(12)
    n_bins = res.transfer.shape[0]
    assert res.debiased_bandpowers.shape == (12, n_bins)
    assert res.covariance.shape == (n_bins, n_bins)
    assert np.allclose(res.covariance, res.covariance.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(res.covariance) > 0)  # positive-definite
    assert np.all(np.isfinite(res.transfer)) and np.all(res.transfer > 0)
    assert res.f_sky == pytest.approx(0.6, abs=0.02)
    assert res.var_pix_ref > 0


@pytest.mark.slow
def test_mc_hartlap_guard_raises_for_too_few_sims() -> None:
    # n_bins = 3 here, so n_sims = 4 <= n_bins + 2 trips the Hartlap guard.
    with pytest.raises(ValueError, match="Hartlap"):
        _run_mc(4)


# --- slow: purity null through the full cleaner ------------------------------


@pytest.mark.slow
def test_purity_null_through_cleaner() -> None:
    """E→B leakage (cleaned CMB-E through the masked-Wiener filter) ≪ lensing-BB floor.

    The leakage template ``res.leakage`` is the E-only leg's mean: the cleaned CMB-E
    projected through each sim's weights, then masked-Wiener filtered and debiased onto
    the true scale. Compared to the binned lensing-BB floor it must be sub-dominant so
    the leaked-E cosmic variance does not dominate σ(r).
    """
    # nside / lmax stay within jht's validated band-limit ceiling (1.5*nside); over it
    # the Wiener accuracy degrades and spuriously inflates the leakage.
    nside, lmax = 64, 96
    _cl_ee, cl_bb = _priors(lmax)
    res = _run_mc(16, nside=nside, lmax=lmax, ell_max=80, delta_ell=20, ell_per_bin_below=2)
    bm = _bin_matrix(2, 80, 20, 2)
    floor = np.asarray(mk.bin_spectrum(cl_bb, bm, 2))
    # Put the leakage on the same (true, beam-free) scale as the floor by dividing the
    # raw E-only mean by the transfer (which absorbs the filter suppression + B_c^2).
    leak_true = np.asarray(res.leakage) / np.asarray(res.transfer)
    ratio = np.abs(leak_true) / floor
    # Sub-dominant to the lensing-BB floor everywhere; the lowest bin (ℓ≲21) carries the
    # most E→B mask ambiguity (~0.08 of the floor at nside=64), tailing to <1% above the
    # bump. nside≥128 science runs do markedly better.
    assert np.max(ratio) < 0.1, (
        f"E->B leakage not sub-dominant: max leak/floor = {np.max(ratio):.3e}"
    )


# --- differentiable (jax.grad-in-w_inv) cut-sky MC ---------------------------


def _traced_setup(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2):
    """Build a CutskyMCContext + cleaner mirroring _run_mc's tiny CMB-only config."""
    cl_ee, cl_bb = _priors(lmax)
    bm = _bin_matrix(2, ell_max, delta_ell, ell_per_bin_below)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
    )
    cleaner = nilc_cleaner(clean_e=True)
    ctx = make_cutsky_mc_context(
        cleaner=cleaner,
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=W_INV,
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
        estimator="wiener",
    )
    return ctx, cleaner


@pytest.mark.slow
def test_traced_matches_forward_covariance() -> None:
    """mc_cutsky_cov_traced reproduces mc_cutsky_bandpowers at the fiducial w_inv.

    Same seeds, same math; the traced path differs only in execution structure
    (precomputed batched sky + ``lax.map`` scan vs process-pool + per-sim build).
    The two therefore agree only to the wiener CG tolerance (default 1e-8), not to
    fp64: ``scan`` and the python loop fuse differently in XLA, and that ~fp-level
    difference is amplified through the CG solve. A tol sweep confirms the agreement
    tracks the CG tol -- max rel diff 1.0e-6 / 9.6e-9 / 3.3e-11 at wiener tol
    1e-8 / 1e-10 / 1e-12 -- so rtol=1e-5 (~10x margin over the observed 1.0e-6) is
    the right gate; a real bug would be %-level. (The internal-consistency check
    ``test_driver_matches_raw_w_inv_path`` still holds traced-vs-traced to 1e-12.)"""
    ctx, cleaner = _traced_setup(12)
    traced = mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner)
    fwd = _run_mc(12)
    np.testing.assert_allclose(
        np.asarray(traced.covariance), np.asarray(fwd.covariance), rtol=1e-5, atol=1e-16
    )
    np.testing.assert_allclose(
        np.asarray(traced.debiased_bandpowers),
        np.asarray(fwd.debiased_bandpowers),
        rtol=1e-5,
        atol=1e-16,
    )
    assert traced.f_sky == pytest.approx(fwd.f_sky)


@pytest.mark.slow
def test_traced_sigma_r_grad_in_w_inv() -> None:
    """End-to-end jax.grad of the map-based sigma(r) w.r.t. w_inv: finite, FD-matched.

    The crown-jewel path: w_inv -> cut-sky MC compsep -> sample covariance ->
    sigma_r_from_external_cov. CRN is fixed (ctx.noise_keys), so autodiff and the
    central finite difference see the same sims and must agree."""
    import jax

    from augr.optimize import make_optimization_context, sigma_r_from_external_cov

    ctx, cleaner = _traced_setup(12)
    opt_ctx = make_optimization_context(
        cleaned_map_instrument(f_sky=0.6),
        NullForegroundModel(),
        CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0},
        priors={},
        fixed_params=[],
        ell_min=2,
        ell_max=24,
        delta_ell=8,
        ell_per_bin_below=2,
    )

    def loss(w):
        cov = mc_cutsky_cov_traced(w, ctx, cleaner).covariance
        return sigma_r_from_external_cov(cov, opt_ctx)

    w0 = jnp.asarray(W_INV)
    s0 = float(loss(w0))
    assert np.isfinite(s0) and s0 > 0

    g = jax.grad(loss)(w0)
    assert bool(jnp.all(jnp.isfinite(g)))

    # Central FD on the first band (CRN-fixed => autodiff and FD use identical sims).
    h = 0.05 * float(w0[0])
    g_fd0 = (float(loss(w0.at[0].add(h))) - float(loss(w0.at[0].add(-h)))) / (2 * h)
    np.testing.assert_allclose(float(g[0]), g_fd0, rtol=0.05)


@pytest.mark.slow
def test_traced_cov_jit_matches_eager() -> None:
    """mc_cutsky_cov_traced is jax.jit-able, including with *traced* beams.

    The jnp + stop_gradient needlet-channel mask removes the last ``np.asarray``/
    ``float()`` boundary in the cleaner body, so the whole map path (sky beaming ->
    clean -> masked-Wiener -> sample covariance) compiles. jit vs eager agree to the
    XLA-fusion / wiener-CG floor (the GPU enabler)."""
    import jax

    ctx, cleaner = _traced_setup(8)
    w0 = jnp.asarray(W_INV)
    bf = jnp.asarray(BEAMS)
    bp = jnp.ones(len(BEAMS))

    def cov_of(w, a, b):
        return mc_cutsky_cov_traced(w, ctx, cleaner, beam_fwhm=a, beam_p=b).covariance

    cov_eager = cov_of(w0, bf, bp)
    cov_jit = jax.jit(cov_of)(w0, bf, bp)
    np.testing.assert_allclose(np.asarray(cov_jit), np.asarray(cov_eager), rtol=1e-6, atol=1e-30)


# --- MASTER estimator path ---------------------------------------------------


def _master_setup(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8):
    """As _traced_setup, but estimator='master' and a SMOOTH mask.

    MASTER needs a tapered mask -- a hard step puts ~1.6e-3 of its power above
    the band limit versus 1.7e-8 for a resolved taper -- so the sigmoid family is
    the right companion here, not galactic_mask.
    """
    cl_ee, cl_bb = _priors(lmax)
    bm = _bin_matrix(2, ell_max, delta_ell, 2)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
    )
    cleaner = nilc_cleaner(clean_e=True)
    ctx = make_cutsky_mc_context(
        cleaner=cleaner, freqs_ghz=FREQS, beam_fwhm_arcmin=BEAMS, w_inv=W_INV,
        nside=nside, lmax=lmax, mask=mk.smooth_gal_cut_mask(nside, 25.0, 8.0),
        cl_ee=cl_ee, cl_bb_prior_unbeamed=cl_bb, bin_matrix=bm, ell_min=2,
        true_bb_binned=true_b, n_sims=n_sims, base_seed=0, fg_model=None, r_in=0.0,
        estimator="master",
    )
    return ctx, cleaner


def _ctx_kwargs(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8):
    """The _master_setup kwargs, minus cleaner/estimator, so callers can vary them."""
    cl_ee, cl_bb = _priors(lmax)
    bm = _bin_matrix(2, ell_max, delta_ell, 2)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None),
        bm, 2,
    )
    return dict(
        freqs_ghz=FREQS, beam_fwhm_arcmin=BEAMS, w_inv=W_INV, nside=nside, lmax=lmax,
        mask=mk.smooth_gal_cut_mask(nside, 25.0, 8.0), cl_ee=cl_ee,
        cl_bb_prior_unbeamed=cl_bb, bin_matrix=bm, ell_min=2, true_bb_binned=true_b,
        n_sims=n_sims, base_seed=0, fg_model=None, r_in=0.0,
    )


def test_master_context_skips_the_var_pix_ref_setup_clean():
    """MASTER builds no inverse-noise filter, so its setup clean must not run.

    Three things are asserted together because they are one change: the clean is
    *actually* skipped (a call count -- the covariance alone cannot tell a live
    skip from a dead one), the fields it fed are None, and the Wiener path still
    runs it. The last case is the anti-vacuity half: a guard keyed on the wrong
    thing would skip everywhere and still pass the first two.
    """
    kw = _ctx_kwargs(6)
    calls = {"n": 0}
    base = nilc_cleaner(clean_e=True)

    def counting(*a, **k):
        calls["n"] += 1
        return base(*a, **k)

    for estimator, expected in (("master", 0), ("wiener", 1)):
        calls["n"] = 0
        ctx = make_cutsky_mc_context(cleaner=counting, estimator=estimator, **kw)
        assert calls["n"] == expected, f"{estimator}: cleaner ran {calls['n']} times"
        if estimator == "master":
            assert ctx.var_pix_ref is None
            assert ctx.inv_noise is None
        else:
            assert ctx.var_pix_ref is not None
            assert ctx.inv_noise is not None


def test_master_bandpowers_are_unchanged_by_skipping_the_setup_clean():
    """Skipping it changes nothing numerically: MASTER never reads what it produced.

    Byte-identical, not ``allclose`` -- the skipped quantity is genuinely unread on
    this path, so any difference at all would mean it was not.
    """
    kw = _ctx_kwargs(6)
    cleaner = nilc_cleaner(clean_e=True)
    wiener = make_cutsky_mc_context(cleaner=cleaner, estimator="wiener", **kw)

    skipped = make_cutsky_mc_context(cleaner=cleaner, estimator="master", **kw)
    supplied = make_cutsky_mc_context(
        cleaner=cleaner, estimator="master",
        var_pix_ref=float(wiener.var_pix_ref), **kw,
    )
    a = mc_cutsky_cov_traced(jnp.asarray(W_INV), skipped, cleaner)
    b = mc_cutsky_cov_traced(jnp.asarray(W_INV), supplied, cleaner)
    assert np.array_equal(np.asarray(a.covariance), np.asarray(b.covariance))


def test_master_accepts_a_clean_e_false_cleaner():
    """The saving the skip unlocks: no setup clean means no project_e() at build.

    Before this, a ``clean_e=False`` cleaner raised at *context build* on every
    estimator, because the setup clean's noise leg projected E. MASTER reads only
    the B solution, which ``tests/test_nilc.py`` pins as byte-identical either way,
    so the covariance must match exactly.
    """
    kw = _ctx_kwargs(6)
    out = {}
    for flag in (True, False):
        cleaner = nilc_cleaner(clean_e=flag)
        ctx = make_cutsky_mc_context(cleaner=cleaner, estimator="master", **kw)
        out[flag] = np.asarray(
            mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner).covariance
        )
    assert np.array_equal(out[True], out[False])


def test_wiener_rejects_a_context_built_without_inv_noise():
    """A MASTER-built context fed to the Wiener branch fails loudly, not deep inside."""
    kw = _ctx_kwargs(6)
    cleaner = nilc_cleaner(clean_e=True)
    ctx = make_cutsky_mc_context(cleaner=cleaner, estimator="master", **kw)
    # estimator is a static field (in the treedef, not a leaf), so tree_at cannot
    # touch it; eqx.Module is a frozen dataclass, so dataclasses.replace can.
    ctx_w = dataclasses.replace(ctx, estimator="wiener")
    with pytest.raises(ValueError, match=r"needs ctx\.inv_noise"):
        mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx_w, cleaner)


def test_edges_from_bin_matrix_matches_the_signal_model_binning():
    """Edges are read off bin_matrix, so MASTER bins cannot drift from Fisher bins."""
    bm = _bin_matrix(2, 24, 8, 2)
    assert _edges_from_bin_matrix(bm, 2) == ((2, 9), (10, 17), (18, 24))


def test_edges_from_bin_matrix_rejects_a_gappy_bin():
    """A non-contiguous bin would mean different things to the two estimators."""
    bm = np.zeros((2, 23))
    bm[0, [0, 1, 5]] = 1.0   # gap
    bm[1, 6:] = 1.0
    with pytest.raises(ValueError, match="non-contiguous"):
        _edges_from_bin_matrix(jnp.asarray(bm), 2)


def test_master_context_carries_the_mask_as_a_traced_leaf():
    """f_sky is static; the mask is a leaf. That asymmetry is the design axis.

    The Wiener context folds its mask into inv_noise at build time and never
    stores it, which is exactly why sky coverage could not previously be
    differentiated.
    """
    import jax

    ctx, _ = _master_setup(2)
    leaves = jax.tree_util.tree_leaves(ctx)
    assert any(leaf.shape == ctx.mask.shape and jnp.allclose(leaf, ctx.mask)
               for leaf in leaves if hasattr(leaf, "shape"))
    assert isinstance(ctx.f_sky, float)


@pytest.mark.slow
def test_master_path_has_unit_transfer_and_zero_leakage():
    """MASTER is unbiased by construction, so there is nothing to calibrate.

    The Wiener path measures a multiplicative transfer and an additive E->B
    leakage template from two extra per-sim projections. MASTER needs neither, so
    it runs one leg per sim instead of three -- and the returned transfer/leakage
    are exactly 1 and 0 rather than approximately so.
    """
    ctx, cleaner = _master_setup(6)
    out = mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner)
    np.testing.assert_array_equal(np.asarray(out.transfer), 1.0)
    np.testing.assert_array_equal(np.asarray(out.leakage), 0.0)
    np.testing.assert_allclose(np.asarray(out.debiased_bandpowers).mean(axis=0),
                               np.asarray(out.mean_bandpower), rtol=1e-12)
    assert np.all(np.isfinite(np.asarray(out.covariance)))


def test_wiener_path_rejects_a_mask_override():
    """Silently ignoring it would be worse than refusing it.

    The Wiener mask is baked into inv_noise at context-build time, so a mask=
    passed to the traced forward could not take effect -- and a caller sweeping
    the mask would get a flat, entirely wrong answer with no error.
    """
    ctx, cleaner = _traced_setup(2)
    with pytest.raises(ValueError, match="only meaningful for estimator"):
        mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner,
                             mask=mk.smooth_gal_cut_mask(16, 25.0, 8.0))


@pytest.mark.slow
def test_master_mask_override_is_differentiable():
    """d sigma(r) / d(mask parameter) through the full MC forward.

    Step size matters here and a careless choice reads as a broken gradient. The
    FD error is dominated by curvature, not Monte-Carlo noise: measured at
    nside=16/lmax=24 with 6 sims, |grad - FD|/|FD| runs 0.44 / 0.087 / 0.021 /
    0.0051 / 0.0008 at h = 2 / 1 / 0.5 / 0.25 / 0.1 degrees -- clean h^2
    convergence. A one-degree step alone would fail a 5% gate while the gradient
    is perfectly correct.

    The test asserts convergence as well as agreement, since the h^2 trend is the
    part that distinguishes a right answer from a lucky one.
    """
    import jax

    from augr.config import cleaned_map_instrument as _inst
    from augr.optimize import make_optimization_context, sigma_r_from_external_cov

    ctx, cleaner = _master_setup(6)
    opt_ctx = make_optimization_context(
        _inst(f_sky=0.6), NullForegroundModel(), CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0}, priors={}, fixed_params=[],
        ell_min=2, ell_max=24, delta_ell=8, ell_per_bin_below=2)

    def sigma_of_cut(b_cut):
        cov = mc_cutsky_cov_traced(
            jnp.asarray(W_INV), ctx, cleaner,
            mask=mk.smooth_gal_cut_mask(16, b_cut, 8.0)).covariance
        return sigma_r_from_external_cov(cov, opt_ctx)

    g = float(jax.grad(sigma_of_cut)(25.0))
    assert np.isfinite(g)
    errs = []
    for h in (0.5, 0.25):
        fd = (float(sigma_of_cut(25.0 + h)) - float(sigma_of_cut(25.0 - h))) / (2 * h)
        errs.append(abs(g - fd) / abs(fd))
    assert errs[-1] < 0.05, f"grad {g:.6e}, FD rel errors {errs}"
    assert errs[0] > errs[-1], f"FD error should shrink with h: {errs}"


@pytest.mark.slow
def test_master_and_wiener_agree_on_sigma_r():
    """The two estimators must land close, and this is what guards the E-leak trap.

    They are different estimators with different failure modes -- MASTER is
    prior-free and unbiased by construction; the masked-Wiener filter carries a
    signal prior frozen at r=0 and is documented as biased-by-construction, a
    bias-for-variance trade -- so they will not agree exactly. Measured at
    nside=16/lmax=24 with 24 sims on shared CRN: 1.95e-3 vs 1.65e-3, a ratio of
    1.18, with MASTER the looser of the two. That is the expected direction: the
    prior buys variance by suppressing ambiguous low-l modes.

    A factor-2 gate is deliberately loose on that comparison and deliberately
    tight on the failure it exists to catch. Feeding MASTER the full cleaned Q/U
    instead of the cleaned B alm leaks real lensing E (~100x the B power) into
    pseudo-BB; the 2x2 decoupling removes it in the mean but not in variance, and
    sigma(r) jumps to 8.7e-2 -- a ratio of ~13, which this test fails loudly.
    """
    ctx_m, cleaner = _master_setup(24)
    ctx_w, _ = _traced_setup(24)
    from augr.config import cleaned_map_instrument as _inst
    from augr.optimize import make_optimization_context, sigma_r_from_external_cov

    opt_ctx = make_optimization_context(
        _inst(f_sky=0.6), NullForegroundModel(), CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0}, priors={}, fixed_params=[],
        ell_min=2, ell_max=24, delta_ell=8, ell_per_bin_below=2)

    def sigma(ctx):
        out = mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner)
        return float(sigma_r_from_external_cov(out.covariance, opt_ctx))

    s_master, s_wiener = sigma(ctx_m), sigma(ctx_w)
    ratio = s_master / s_wiener
    assert 0.5 < ratio < 2.0, (
        f"MASTER {s_master:.3e} vs Wiener {s_wiener:.3e} (ratio {ratio:.2f}). "
        "A ratio near 13 means MASTER is being fed the E+B cleaned map instead "
        "of the cleaned B alm."
    )


# ---------------------------------------------------------------------------
# Gradient checkpointing and batching on the per-sim scan (_sim_map)
# ---------------------------------------------------------------------------
#
# Fast tier holds the trace-level checks -- a dead knob and a correct answer look
# identical, and tracing is cheap. The executing value / gradient / memory gates
# are slow, matching every other test that runs this scan.

_SETUPS = {"master": _master_setup, "wiener": _traced_setup}


def _top_level_scan_lengths(jaxpr) -> list[int]:
    """Lengths of the scans at the top of the trace.

    The per-sim map is one of these; the other is the MASTER coupling build over
    ell. Nested scans (the vmapped body's own) are deliberately not collected --
    they are what ``sim_batch`` puts *inside* a step, not the step count.
    """
    return [e.params["length"] for e in jaxpr.jaxpr.eqns if e.primitive.name == "scan"]


def _sq_cov(w, ctx, cleaner, **kw):
    """Scalar off the covariance, so there is something to differentiate."""
    return jnp.sum(mc_cutsky_cov_traced(w, ctx, cleaner, **kw).covariance ** 2)


def _grad_temp_bytes(ctx, cleaner, **kw) -> int:
    """Compiled reverse-mode scratch. Deterministic, unlike sampled RSS.

    ``temp_size_in_bytes``, not ``peak_memory_in_bytes`` -- the latter reads ~0 on
    a tape of hundreds of MB (see ``TestRematMemory`` in ``test_delensing.py``).
    """
    g = jax.jit(jax.grad(lambda w: _sq_cov(w, ctx, cleaner, **kw)))
    return g.lower(W_INV).compile().memory_analysis().temp_size_in_bytes


def test_remat_and_sim_batch_are_live_in_the_trace() -> None:
    """Both knobs checked on the trace, because values cannot distinguish them.

    Values are bit-identical across ``remat`` and agree to 3e-14 across
    ``sim_batch`` (both gated below), so agreement demonstrates nothing about
    whether anything was actually checkpointed or batched. ``sim_batch`` gives
    ``ceil(n_sims / sim_batch)`` sequential steps; 3 does not divide 6, so the
    loop also covers the padding path.
    """
    ctx, cleaner = _master_setup(6)
    on = str(jax.make_jaxpr(lambda w: _sq_cov(w, ctx, cleaner, remat=True))(W_INV))
    off = str(jax.make_jaxpr(lambda w: _sq_cov(w, ctx, cleaner, remat=False))(W_INV))

    # The MASTER coupling build checkpoints its own per-l2 map unconditionally
    # (pseudo_cl_jax.coupling_matrices, remat=True by default), so "no checkpoint
    # anywhere" is no longer a proxy for "the sim scan is not checkpointed" --
    # that assertion would be testing two loops at once. Count instead: turning
    # this knob off must strictly reduce the checkpoints, and must not remove
    # the coupling's.
    n_on, n_off = on.count("remat2["), off.count("remat2[")
    assert n_off >= 1, "the coupling build's own remat vanished from the trace"
    assert n_on > n_off, (
        f"remat=True added no checkpoint to the sim scan ({n_on} vs {n_off})"
    )

    for sim_batch, expected in ((1, 6), (2, 3), (3, 2), (4, 2)):
        lengths = _top_level_scan_lengths(
            jax.make_jaxpr(
                lambda w, b=sim_batch: _sq_cov(w, ctx, cleaner, remat=True, sim_batch=b)
            )(W_INV)
        )
        assert expected in lengths, (
            f"sim_batch={sim_batch}: no top-level scan of length {expected} "
            f"in {lengths}"
        )


def test_rejects_bad_knob_values() -> None:
    """``sim_batch`` is a step count; ``remat`` is a trace-time decision.

    A traced ``remat`` cannot express a choice made while tracing, so it is
    refused rather than silently coerced to True.
    """
    ctx, cleaner = _master_setup(6)
    for bad in (0, -1, 1.0, True, "4", None):
        with pytest.raises(ValueError, match="sim_batch"):
            mc_cutsky_cov_traced(W_INV, ctx, cleaner, sim_batch=bad)
    with pytest.raises(ValueError, match="remat"):
        mc_cutsky_cov_traced(W_INV, ctx, cleaner, remat=jnp.array(True))


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["master", "wiener"])
def test_remat_leaves_values_bit_identical(estimator) -> None:
    """``jax.checkpoint`` rewrites the reverse-mode tape, not the forward jaxpr."""
    ctx, cleaner = _SETUPS[estimator](6)
    on = mc_cutsky_cov_traced(W_INV, ctx, cleaner, remat=True)
    off = mc_cutsky_cov_traced(W_INV, ctx, cleaner, remat=False)
    np.testing.assert_array_equal(np.asarray(on.covariance), np.asarray(off.covariance))
    np.testing.assert_array_equal(
        np.asarray(on.debiased_bandpowers), np.asarray(off.debiased_bandpowers)
    )


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["master", "wiener"])
def test_remat_gradient_agrees_to_a_few_ulp(estimator) -> None:
    """Measured max relative difference 1.9e-16 (master) / 1.2e-15 (wiener).

    Gated at 1e-13, i.e. ~100-500x margin. The residual is XLA re-fusing the
    recomputed forward -- the same 1-2 ulp drift the delensing scans show.
    """
    ctx, cleaner = _SETUPS[estimator](6)
    g_on = jax.grad(lambda w: _sq_cov(w, ctx, cleaner, remat=True))(W_INV)
    g_off = jax.grad(lambda w: _sq_cov(w, ctx, cleaner, remat=False))(W_INV)
    assert np.all(np.isfinite(np.asarray(g_on)))
    np.testing.assert_allclose(np.asarray(g_on), np.asarray(g_off), rtol=1e-13, atol=0.0)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["master", "wiener"])
def test_remat_makes_the_tape_flat_in_n_sims(estimator) -> None:
    """The point of the change: the tape stops carrying a factor of ``n_sims``.

    Gating flatness rather than a ratio, because the ratio is just whatever
    ``n_sims`` happens to be -- at this fixture it is only 2.3-4.9x, while the
    same structural change is worth orders of magnitude at the nside=1024
    production shape. Flatness is the invariant; the ratio is a property of the
    fixture.

    Measured: remat 6.11 MB at both 6 and 12 sims (master) and 11.41 MB (wiener),
    against a plain arm growing 14.31 -> 22.94 MB and 33.73 -> 55.34 MB.
    """
    setup = _SETUPS[estimator]
    lo_ctx, lo_cleaner = setup(6)
    hi_ctx, hi_cleaner = setup(12)

    on_lo = _grad_temp_bytes(lo_ctx, lo_cleaner, remat=True)
    on_hi = _grad_temp_bytes(hi_ctx, hi_cleaner, remat=True)
    off_lo = _grad_temp_bytes(lo_ctx, lo_cleaner, remat=False)
    off_hi = _grad_temp_bytes(hi_ctx, hi_cleaner, remat=False)

    assert on_hi == pytest.approx(on_lo, rel=0.05), (
        f"remat'd tape grew with n_sims: {on_lo} -> {on_hi} bytes. The per-sim "
        "residuals are being retained after all."
    )
    # Anti-vacuity: if the control arm ever stops growing for an unrelated reason,
    # the flatness assertion above becomes trivially true and must fail loudly.
    assert off_hi > 1.3 * off_lo, (
        f"control arm did not grow with n_sims ({off_lo} -> {off_hi} bytes), so "
        "the flatness check above proves nothing."
    )
    assert on_hi < off_hi


@pytest.mark.slow
def test_sim_batch_agrees_with_the_unbatched_scan() -> None:
    """Measured, not exact: vmapping reassociates the trailing reductions.

    Max relative difference 3.3e-14 on values and 2.4e-15 on the gradient across
    sim_batch 2/3/4. Gated at 1e-11, ~300x margin.
    """
    ctx, cleaner = _master_setup(6)
    ref = mc_cutsky_cov_traced(W_INV, ctx, cleaner, remat=True).covariance
    g_ref = jax.grad(lambda w: _sq_cov(w, ctx, cleaner, remat=True))(W_INV)
    for b in (2, 3, 4):
        got = mc_cutsky_cov_traced(
            W_INV, ctx, cleaner, remat=True, sim_batch=b
        ).covariance
        g = jax.grad(
            lambda w, bb=b: _sq_cov(w, ctx, cleaner, remat=True, sim_batch=bb)
        )(W_INV)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-11)
        np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-11)
