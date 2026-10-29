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
import functools
import os

import numpy as np
import pytest

pytest.importorskip("jht")
pytest.importorskip("ducc0")

import jax
import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import cmilc_cleaner, nilc_cleaner
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
    share_constant_fg,
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


def _run_mc(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2, workers=1,
            **mc_kw):
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
        **mc_kw,
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
    assert res.rec_transfer.shape == (12, n_bins)


@pytest.mark.slow
def test_analytic_transfer_matches_the_mc_transfer() -> None:
    """``masking.analytic_transfer`` from the exact ``K`` == the NILC MC transfer ``F_b``.

    ``K`` is built for the run's own filter (``var_pix_ref``, B_c-beamed priors, mask,
    uniform hits), so this checks that the cleaned CMB-B reaches the filter at unit gain
    and that ``K``, the beam and the truth follow the MC's conventions. Per-bin z from
    the per-sim ``rec_transfer`` spread. Measured at 16 sims: |z| <= 1.32 with ``K``;
    5.8-8.6 with ``K``'s diagonal (the mutation). Gates |z| < 3.5 and max |z| > 5.
    """
    lmax, nside, n_sims = 24, 16, 16
    res = _run_mc(n_sims)
    cl_ee, cl_bb = _priors(lmax)
    bc = float(min(BEAMS))
    invn = mk.inv_noise_map(jnp.ones(12 * nside**2), res.var_pix_ref,
                            mask=mk.galactic_mask(nside, 0.6))
    K = mk.wiener_bb_response(
        mk.wiener_normal_matrix(invn, nside=nside, lmax=lmax),
        beamed_prior(cl_ee, bc, lmax), beamed_prior(cl_bb, bc, lmax), lmax=lmax,
    )
    bm = _bin_matrix(2, 24, 8, 2)
    c_true = np.clip(np.asarray(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0)), 0.0, None)
    per_sim = res.rec_transfer / np.asarray(mk.bin_spectrum(c_true, bm, 2))[None, :]
    np.testing.assert_allclose(per_sim.mean(0), res.transfer, rtol=1e-12)
    err = per_sim.std(0, ddof=1) / np.sqrt(n_sims)

    def z(kernel):
        f = mk.analytic_transfer(kernel, bm, 2, common_fwhm_arcmin=bc, true_cl_bb=c_true)
        return (f - res.transfer) / err

    assert np.max(np.abs(z(K))) < 3.5
    assert np.max(np.abs(z(np.diag(np.diag(K))))) > 5.0


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


def _traced_setup(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2,
                  **ctx_kw):
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
        **ctx_kw,
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


# --- masked-Wiener preconditioner + convergence check ------------------------


def test_wiener_context_builds_the_preconditioner() -> None:
    """Wiener contexts carry a preconditioner by default; MASTER and opt-out carry none.

    Spin-2 DOFs with ``l <= ell_split`` number ``2((ell_split+1)^2 - 4)``; the default
    32 exceeds lmax=24 here, so the block is the whole operator.
    """
    ctx, cleaner = _traced_setup(6)
    assert ctx.wiener_precond.low_idx.shape == (2 * (25**2 - 4),)
    vpr = ctx.var_pix_ref  # skip the setup clean on the rebuilds
    ctx8, _ = _traced_setup(6, wiener_ell_split=8, var_pix_ref=vpr)
    assert ctx8.wiener_precond.low_idx.shape == (2 * (9**2 - 4),)
    off, _ = _traced_setup(6, wiener_precond=False, var_pix_ref=vpr)
    assert off.wiener_precond is None
    master = make_cutsky_mc_context(cleaner=cleaner, estimator="master", **_ctx_kwargs(6))
    assert master.wiener_precond is None


def test_traced_wiener_records_convergence_and_raises_when_truncated() -> None:
    """A truncated solve raises in the traced forward; ``wiener_check=False`` only records.

    ``ell_split=8`` keeps the preconditioned CG genuinely iterative (the default block
    would be exact at lmax=24). The record-only arm is the anti-vacuity half: it shows
    the same context really does truncate, so the raise comes from the guard.
    """
    ctx, cleaner = _traced_setup(6, wiener_ell_split=8)
    w = jnp.asarray(W_INV)
    ok = mc_cutsky_cov_traced(w, ctx, cleaner)
    assert 0 < float(ok.wiener_max_rel_residual) <= ctx.tol
    assert 0 < int(ok.wiener_max_n_iter) < ctx.max_iter

    trunc, _ = _traced_setup(6, wiener_ell_split=8, max_iter=3, var_pix_ref=ctx.var_pix_ref)
    with pytest.raises(RuntimeError, match="did not converge"):
        np.asarray(mc_cutsky_cov_traced(w, trunc, cleaner).covariance)
    rec = mc_cutsky_cov_traced(w, trunc, cleaner, wiener_check=False)
    assert float(rec.wiener_max_rel_residual) > trunc.tol
    assert int(rec.wiener_max_n_iter) == 3


@pytest.mark.slow
def test_traced_wiener_truncation_raises_under_jit_and_grad() -> None:
    """The ``eqx.error_if`` guard survives ``jax.jit`` and ``jax.grad``."""
    import jax

    trunc, cleaner = _traced_setup(6, wiener_ell_split=8, max_iter=3)
    w = jnp.asarray(W_INV)
    with pytest.raises(RuntimeError, match="did not converge"):
        np.asarray(jax.jit(lambda v: mc_cutsky_cov_traced(v, trunc, cleaner).covariance)(w))
    with pytest.raises(RuntimeError, match="did not converge"):
        np.asarray(jax.grad(
            lambda v: jnp.sum(mc_cutsky_cov_traced(v, trunc, cleaner).covariance))(w))


@pytest.mark.slow
def test_eager_mc_records_convergence_and_raises_when_truncated() -> None:
    """mc_cutsky_bandpowers reports its worst solve, and raises on a truncated one."""
    res = _run_mc(6, wiener_ell_split=8)
    assert 0 < res.wiener_max_rel_residual <= 1e-8
    assert 0 < res.wiener_max_n_iter < 1000
    with pytest.raises(RuntimeError, match="did not converge"):
        _run_mc(6, wiener_ell_split=8, max_iter=3, var_pix_ref=res.var_pix_ref)


# Two CONVERGED Wiener paths agree only to the CG-tolerance floor (tol 1e-8).
# Measured in these tests' config as max|a-b| / max|b|: debiased 2.9e-7,
# covariance 5.1e-7, transfer 3.0e-9, leakage 3.5e-6, sigma(r) gradient 3.6e-7.
_PRECOND_FLOOR = 1e-5


def _max_rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.slow
def test_preconditioned_forward_matches_the_unpreconditioned_solve() -> None:
    """The preconditioner changes iteration counts, not the solution.

    Both arms converge (the plain one at ``max_iter=5000``) on the same sims and
    filter, so they differ only at the CG-tolerance floor.
    """
    pc, cleaner = _traced_setup(8, wiener_ell_split=8)
    plain, _ = _traced_setup(8, wiener_precond=False, max_iter=5000, var_pix_ref=pc.var_pix_ref)
    w = jnp.asarray(W_INV)
    a = mc_cutsky_cov_traced(w, pc, cleaner)
    b = mc_cutsky_cov_traced(w, plain, cleaner)
    for name in ("debiased_bandpowers", "covariance", "transfer", "leakage"):
        assert _max_rel(getattr(a, name), getattr(b, name)) < _PRECOND_FLOOR, name
    # 44 vs 201 iterations measured: the preconditioner must actually be doing work.
    assert int(a.wiener_max_n_iter) < int(b.wiener_max_n_iter) // 2


@pytest.mark.slow
def test_preconditioned_gradient_matches_and_backward_solves_converge() -> None:
    """jax.grad of sigma(r) agrees with and without the preconditioner, and at 5x budget.

    jht reports nothing about the backward pass's transpose solves, and they need
    more iterations than the forward ones (capping at the forward need, 44, moves the
    gradient by 1.2e-9). If they needed more than ``max_iter`` the gradient would move
    when the budget grows; bitwise equality at 1000 vs 5000 shows they converged
    inside the default.
    """
    import jax

    from augr.optimize import make_optimization_context, sigma_r_from_external_cov

    pc, cleaner = _traced_setup(8, wiener_ell_split=8)
    vpr = pc.var_pix_ref
    plain, _ = _traced_setup(8, wiener_precond=False, max_iter=5000, var_pix_ref=vpr)
    big, _ = _traced_setup(8, wiener_ell_split=8, max_iter=5000, var_pix_ref=vpr)
    opt_ctx = make_optimization_context(
        cleaned_map_instrument(f_sky=0.6), NullForegroundModel(), CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0}, priors={}, fixed_params=[], ell_min=2, ell_max=24,
        delta_ell=8, ell_per_bin_below=2,
    )

    def grad_for(ctx):
        return np.asarray(jax.grad(lambda w: sigma_r_from_external_cov(
            mc_cutsky_cov_traced(w, ctx, cleaner).covariance, opt_ctx))(jnp.asarray(W_INV)))

    g = grad_for(pc)
    assert np.all(np.isfinite(g))
    assert _max_rel(g, grad_for(plain)) < _PRECOND_FLOOR
    assert np.array_equal(g, grad_for(big))


# --- masked-Wiener dense direct solve ----------------------------------------
#
# _traced_setup's binary GAL60 mask with non-uniform hits and a two-component
# isotropic foreground, so every leg (data, CMB-B, CMB-E, FG total, per component)
# goes through the solve. The CG reference iterates for real (ell_split=8) to tol 1e-12.
# Margins measured 2026-10-25.

_DENSE_NS, _DENSE_LMAX = 16, 24


@functools.lru_cache(maxsize=1)
def _dense_inputs(n_sims):
    hits = jnp.asarray(np.linspace(1.0, 3.0, 12 * _DENSE_NS**2))
    comps = jnp.stack([_isotropic_fg(n_sims + 1, _DENSE_LMAX, slope=1.5),
                       _isotropic_fg(n_sims + 1, _DENSE_LMAX, slope=-3.0, seed0=77_000)], axis=1)
    unit = mk.inv_noise_map(hits, 1.0, mask=mk.galactic_mask(_DENSE_NS, 0.6))
    normal = mk.wiener_normal_matrix(unit, nside=_DENSE_NS, lmax=_DENSE_LMAX)
    return hits, comps, normal


def _dense_setup(n_sims, **ctx_kw):
    """(dense ctx, cleaner, kwargs that rebuild the same sims and filter for a CG twin)."""
    hits, comps, normal = _dense_inputs(n_sims)
    kw = dict(hit_map=hits, fg_comp_eb_alm=comps)
    ctx, cleaner = _traced_setup(n_sims, wiener_solver="dense", wiener_normal=normal, **kw,
                                 **ctx_kw)
    return ctx, cleaner, dict(kw, var_pix_ref=ctx.var_pix_ref)


def test_dense_context_carries_a_factor_and_validates_its_inputs(tmp_path) -> None:
    """A dense context holds the factor (no preconditioner) for its own ``var_pix_ref``; a
    ``.npy`` path gives the same factor bit for bit; bad configurations are refused."""
    ctx, cleaner, kw = _dense_setup(6)
    assert ctx.wiener_solver == "dense" and ctx.wiener_precond is None
    assert ctx.wiener_factor.var_pix_ref == ctx.var_pix_ref
    _hits, _comps, normal = _dense_inputs(6)
    path = tmp_path / "normal.npy"
    np.save(path, normal)
    from_path, _ = _traced_setup(6, wiener_solver="dense", wiener_normal=str(path), **kw)
    assert np.array_equal(np.asarray(from_path.wiener_factor.chol_t),
                          np.asarray(ctx.wiener_factor.chol_t))
    assert np.array_equal(np.load(path), normal)  # the caller's array was not touched

    with pytest.raises(ValueError, match="'cg' or 'dense'"):
        _traced_setup(6, wiener_solver="direct", **kw)
    with pytest.raises(ValueError, match="needs wiener_normal"):
        _traced_setup(6, wiener_solver="dense", **kw)
    with pytest.raises(ValueError, match="does not describe"):
        _traced_setup(6, wiener_solver="dense", wiener_normal=2.0 * normal, **kw)
    with pytest.raises(ValueError, match="estimator='wiener' only"):
        make_cutsky_mc_context(cleaner=cleaner, estimator="master", wiener_solver="dense",
                               wiener_normal=normal, **_ctx_kwargs(6))


def test_dense_context_factors_on_the_host_on_a_gpu_backend(monkeypatch) -> None:
    """A dense context factors on the host even when the default backend is a GPU, whose
    Cholesky is wrong at lmax 192 (Vista job 1105246); the device route stays opt-in."""
    calls = []
    real = mk.wiener_factor

    def spy(*args, **kwargs):
        calls.append(kwargs["on_device"])
        return real(*args, **kwargs)

    monkeypatch.setattr(mk, "wiener_factor", spy)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    hits, comps, normal = _dense_inputs(6)
    kw = dict(hit_map=hits, fg_comp_eb_alm=comps, wiener_solver="dense", wiener_normal=normal)
    host, _ = _traced_setup(6, **kw)
    _traced_setup(6, wiener_factor_on_device=True, var_pix_ref=host.var_pix_ref, **kw)
    assert calls == [False, True]


_DENSE_OUTPUTS = ("debiased_bandpowers", "covariance", "transfer", "leakage",
                  "fg_residual_bandpower", "fg_residual_components", "rec_transfer")


@pytest.mark.slow
def test_dense_forward_matches_the_converged_cg() -> None:
    """Every output of the dense forward == the CG forward at tol 1e-14, same sims and filter.

    The gap is the CG's own error: it falls ~100x per 100x in tol (leakage 6.8e-7, 1.2e-8,
    7.2e-11, 3.9e-13 at tol 1e-8..1e-14). At tol 1e-14 measured max|Δ| / max|ref| <= 3.9e-13
    (leakage; covariance 2.6e-13, debiased 5.0e-14); gate 1e-10. The tol-1e-10 arm (debiased
    5.0e-10) shows the comparison resolves CG error.
    """
    ctx, cleaner, kw = _dense_setup(6)
    w = jnp.asarray(W_INV)
    a = mc_cutsky_cov_traced(w, ctx, cleaner, fg_residual=True)
    assert int(a.wiener_max_n_iter) == 0 and float(a.wiener_max_rel_residual) < 1e-12
    cg, _ = _traced_setup(6, wiener_ell_split=8, tol=1e-14, max_iter=50000, **kw)
    b = mc_cutsky_cov_traced(w, cg, cleaner, fg_residual=True)
    assert int(b.wiener_max_n_iter) > 5
    for name in _DENSE_OUTPUTS:
        assert _max_rel(getattr(a, name), getattr(b, name)) < 1e-10, name
    loose, _ = _traced_setup(6, wiener_ell_split=8, tol=1e-10, max_iter=50000, **kw)
    c = mc_cutsky_cov_traced(w, loose, cleaner, fg_residual=True)
    tight_gap = _max_rel(a.debiased_bandpowers, b.debiased_bandpowers)
    assert _max_rel(a.debiased_bandpowers, c.debiased_bandpowers) > 10 * tight_gap


def test_dense_forward_raises_on_a_filter_it_was_not_built_for() -> None:
    """The residual guard fires when ``inv_noise`` and the factor disagree, naming the
    residual's decade; with ``wiener_check=False`` nothing is checked and the residual
    reads NaN. (NaN firing is covered at function level in ``test_dense_residual_decades``.)"""
    ctx, cleaner, _kw = _dense_setup(6)
    bad = dataclasses.replace(ctx, inv_noise=2.0 * ctx.inv_noise)
    w = jnp.asarray(W_INV)
    with pytest.raises(RuntimeError, match=r"residual in \[1e-1, 1e0\) \(tol 1e-08\)"):
        np.asarray(mc_cutsky_cov_traced(w, bad, cleaner).covariance)
    rec = mc_cutsky_cov_traced(w, bad, cleaner, wiener_check=False)
    assert np.isnan(float(rec.wiener_max_rel_residual))


def test_dense_residual_decades() -> None:
    """Each residual maps to the message naming its decade, non-finite to the NaN one; the
    check fires above ``tol`` and on NaN."""
    from augr.spectrum_stages import _dense_residual_bad, _dense_residual_msgs, _residual_decade

    bad = [bool(_dense_residual_bad(jnp.asarray(r), 1e-8)) for r in (1e-9, 1e-8, 2e-8, np.nan)]
    assert bad == [False, False, True, True]
    msgs = _dense_residual_msgs(1e-8)
    for rel, span in [(3e-16, "[1e-16, 1e-15)"), (2e-7, "[1e-7, 1e-6)"), (0.5, "[1e-1, 1e0)"),
                      (1.0, "[1e0, 1e1)"), (1e5, ">= 1e2"), (1e-20, "[1e-16, 1e-15)"),
                      (np.nan, "NaN or inf"), (np.inf, "NaN or inf")]:
        assert span in msgs[int(_residual_decade(jnp.asarray(rel)))], (rel, span)


@pytest.mark.slow
def test_dense_gradient_matches_the_converged_cg() -> None:
    """``jax.grad`` of sigma(r) in ``w_inv``: dense vs CG at tol 1e-14. Measured 2.5e-14
    (1.7e-11 at tol 1e-12, 1.2e-9 at 1e-10: the CG's error); gate 1e-11."""
    from augr.optimize import make_optimization_context, sigma_r_from_external_cov

    ctx, cleaner, kw = _dense_setup(8)
    cg, _ = _traced_setup(8, wiener_ell_split=8, tol=1e-14, max_iter=50000, **kw)
    opt_ctx = make_optimization_context(
        cleaned_map_instrument(f_sky=0.6), NullForegroundModel(), CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0}, priors={}, fixed_params=[], ell_min=2, ell_max=24,
        delta_ell=8, ell_per_bin_below=2,
    )

    def grad_for(c):
        return np.asarray(jax.grad(lambda w: sigma_r_from_external_cov(
            mc_cutsky_cov_traced(w, c, cleaner).covariance, opt_ctx))(jnp.asarray(W_INV)))

    g = grad_for(ctx)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)
    assert _max_rel(g, grad_for(cg)) < 1e-11


@pytest.mark.slow
def test_dense_sim_batch_agrees_with_the_unbatched_map() -> None:
    """``sim_batch`` 2 / 3 on the dense branch, values and gradient. Measured <= 5.0e-15
    (leakage) and 2.1e-15 (gradient); gate 1e-12."""
    ctx, cleaner, _kw = _dense_setup(6)
    w = jnp.asarray(W_INV)
    ref = mc_cutsky_cov_traced(w, ctx, cleaner, fg_residual=True)
    g_ref = jax.grad(lambda v: _sq_cov(v, ctx, cleaner))(w)
    for b in (2, 3):
        got = mc_cutsky_cov_traced(w, ctx, cleaner, fg_residual=True, sim_batch=b)
        for name in ("debiased_bandpowers", "covariance", "leakage", "fg_residual_components"):
            assert _max_rel(getattr(got, name), getattr(ref, name)) < 1e-12, (b, name)
        g = jax.grad(lambda v, bb=b: _sq_cov(v, ctx, cleaner, sim_batch=bb))(w)
        assert _max_rel(g, g_ref) < 1e-12, b


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


# --- foreground-residual leg on the masked-Wiener path ------------------------


def _with_fg(ctx, fg_eb_alm):
    """``ctx`` with a per-sim ``(n_sims, n_band, 2, n_alm)`` foreground attached."""
    hs = dataclasses.replace(ctx.harmonic_skies, fg_eb_alm=fg_eb_alm)
    return dataclasses.replace(ctx, harmonic_skies=hs)


def _twin_fg(e_alm, b_alm):
    """A flat-SED (CMB-unit) foreground: the same ``(n_sims, n_alm)`` E/B in every band.

    The cleaner's weights sum to one on a flat SED, so this foreground comes out of
    the clean exactly as a CMB component would -- which is what makes its residual
    predictable.
    """
    fg = jnp.stack([e_alm, b_alm], axis=1)[:, None]
    return jnp.broadcast_to(fg, (fg.shape[0], len(FREQS), *fg.shape[2:]))


def _isotropic_fg(n_sims, lmax, *, amp=10.0, slope=-3.0, seed0=10_000):
    """Independent isotropic B-only foreground per sim: lensing-BB shape, power-law SED.

    Lensing-shaped on purpose: the Wiener transfer is calibrated on that shape, and
    a sharp-mask Wiener filter couples neighbouring l, so a differently shaped
    spectrum would carry a genuine (and data-vector-consistent) calibration offset
    that is not what the cross-estimator check is after. B-only because MASTER reads
    only the cleaned B, so foreground E->B leakage is a real estimator difference.
    """
    import healpy as hp

    _, cl_bb = _priors(lmax)
    sed = jnp.asarray([(f / 150.0) ** slope for f in FREQS])
    g = []
    for s in range(n_sims):
        np.random.seed(seed0 + s)  # noqa: NPY002 - healpy.synalm uses the global RNG
        g.append(hp.synalm(np.asarray(cl_bb) * amp, lmax=lmax, new=True))
    g = jnp.asarray(np.stack(g))
    fg = jnp.zeros((n_sims, len(FREQS), 2, g.shape[1]), dtype=g.dtype)
    return fg.at[:, :, 1].set(sed[None, :, None] * g[:, None, :])


def test_cmilc_runs_in_the_traced_forward():
    """cMILC cleans inside the per-sim map and passes the CMB with unit response.

    On a shared Wiener filter (one context, so one inv_noise) any unit-CMB-response
    cleaner leaves the same cleaned CMB B, so cMILC's transfer must equal NILC's.
    Measured 3e-10. (Each cleaner's *own* context differs by ~4x: the context's
    var_pix_ref comes from a setup clean, and cMILC's extra constraint leaves more
    noise, so its filter suppresses more -- a filter setting, not a transfer error.)
    """
    ctx = make_cutsky_mc_context(
        cleaner=nilc_cleaner(clean_e=True), estimator="wiener", **_ctx_kwargs(6))
    w = jnp.asarray(W_INV)
    nilc = mc_cutsky_cov_traced(w, ctx, nilc_cleaner(clean_e=True))
    cmilc = mc_cutsky_cov_traced(
        w, ctx, cmilc_cleaner(FREQS, moments=("f_dust",), clean_e=True), fg_residual=True)
    assert np.all(np.isfinite(np.asarray(cmilc.covariance)))
    np.testing.assert_allclose(
        np.asarray(cmilc.transfer), np.asarray(nilc.transfer), rtol=3e-9)


def test_wiener_transfer_follows_a_delensing_residual():
    """A uniform A_L rescale of the sims' lensing B must leave the Wiener transfer alone.

    At r_in=0 a uniform residual ``A_L * C_lens`` scales every sim's CMB B by
    ``sqrt(A_L)``; the cleaned CMB B scales with it (the ILC has unit CMB response)
    and the Wiener filter is linear, so ``rec_b`` scales by ``A_L`` exactly and the
    transfer must not move. Measured 5.3e-6 at A_L=0.1 (pixel-quadrature E/B mixing:
    the CMB E is not rescaled). Calibrating against the unrescaled truth gives
    ``A_L * F_b`` -- off by 90% here, and every debiased bandpower 1/A_L too large.
    At A_L=1 the knob must reproduce the no-delensing forward exactly.
    """
    cleaner = nilc_cleaner(clean_e=True)
    kw = _ctx_kwargs(6)
    ctx = make_cutsky_mc_context(
        cleaner=cleaner, estimator="wiener", split_lensing=True, **kw)
    ells = jnp.arange(kw["lmax"] + 1, dtype=float)
    lens = jnp.asarray(CMBSpectra().cl_lensing(ells))
    w = jnp.asarray(W_INV)
    ref = mc_cutsky_cov_traced(w, ctx, cleaner)

    full = mc_cutsky_cov_traced(w, ctx, cleaner, cl_bb_res=lens, cl_bb_res_ells=ells)
    assert np.array_equal(np.asarray(full.covariance), np.asarray(ref.covariance))

    delensed = mc_cutsky_cov_traced(
        w, ctx, cleaner, cl_bb_res=0.1 * lens, cl_bb_res_ells=ells)
    np.testing.assert_allclose(
        np.asarray(delensed.transfer), np.asarray(ref.transfer), rtol=5e-5)


def test_wiener_fg_residual_leaves_the_data_untouched():
    """Asking for the leg changes no existing output, and with no foreground it is 0."""
    cleaner = nilc_cleaner(clean_e=True)
    ctx = make_cutsky_mc_context(cleaner=cleaner, estimator="wiener", **_ctx_kwargs(6))
    off = mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner)
    on = mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner, fg_residual=True)
    assert off.fg_residual_bandpower is None
    for name in ("covariance", "debiased_bandpowers", "transfer", "leakage"):
        assert np.array_equal(np.asarray(getattr(off, name)), np.asarray(getattr(on, name)))
    np.testing.assert_array_equal(np.asarray(on.fg_residual_bandpower), 0.0)


def test_wiener_fg_residual_twins_land_on_the_debias_pieces():
    """A foreground that IS the sim's own CMB must reproduce the debias legs.

    * **B twin** (foreground = CMB B, E = 0): its cleaned map is the cleaned CMB B,
      i.e. the transfer leg, so the residual is ``mean(rec_b) / F_b = true_bb_binned``.
      Measured 2.4e-6 relative (pixel-quadrature E/B mixing, since the CMB maps the
      transfer leg projects also carry E). Dropping ``/ F_b`` is off by 96%.
    * **E twin** (foreground = CMB E, B = 0): the residual is the leakage leg on the
      debiased scale, ``leakage / F_b``. This pins that the E projection is included
      and that the leakage template is NOT subtracted (doing so is off by 99.6%;
      dropping the E projection leaves only quadrature, off by ~100%). Measured
      3.9e-3 relative: the cleaner's B projection of an E-only map carries a
      quadrature E->B leak (4.6e-5 of the power alone) that adds coherently with the
      leaked E through the cross term; the E projection alone matches to 2e-7.
    """
    cleaner = nilc_cleaner(clean_e=True)
    ctx = make_cutsky_mc_context(cleaner=cleaner, estimator="wiener", **_ctx_kwargs(6))
    hs = ctx.harmonic_skies
    zero = jnp.zeros_like(hs.cmb_b_alm)
    w = jnp.asarray(W_INV)

    b = mc_cutsky_cov_traced(
        w, _with_fg(ctx, _twin_fg(zero, hs.cmb_b_alm)), cleaner, fg_residual=True)
    np.testing.assert_allclose(
        np.asarray(b.fg_residual_bandpower), np.asarray(ctx.true_bb_binned), rtol=2e-5)

    e = mc_cutsky_cov_traced(
        w, _with_fg(ctx, _twin_fg(hs.cmb_e_alm, zero)), cleaner, fg_residual=True)
    np.testing.assert_allclose(
        np.asarray(e.fg_residual_bandpower),
        np.asarray(e.leakage / e.transfer),
        rtol=1e-2,
    )


def _component_run(comps, n_sims, *, var_pix_ref=None):
    """Wiener forward with ``fg_residual=True`` on a per-sim component stack."""
    cleaner = nilc_cleaner(clean_e=True)
    ctx = make_cutsky_mc_context(
        cleaner=cleaner, estimator="wiener", var_pix_ref=var_pix_ref,
        fg_comp_eb_alm=comps, **_ctx_kwargs(n_sims))
    return ctx, mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner, fg_residual=True)


def test_fg_components_leave_every_existing_output_unchanged():
    """A component stack is the summed foreground plus extra legs, nothing else.

    Against the same sims carrying the component sum as a plain foreground (and the
    same var_pix_ref), every pre-existing output is bit-identical -- measured exact.
    The component means match the per-sim legs' mean to 2e-16 (summation order).
    """
    n = 6
    lmax = _ctx_kwargs(n)["lmax"]
    dust = _isotropic_fg(n + 1, lmax, slope=1.5)
    sync = _isotropic_fg(n + 1, lmax, slope=-3.0, seed0=77_000)
    comps = jnp.stack([dust, sync], axis=1)
    ctx, comp = _component_run(comps, n)

    cleaner = nilc_cleaner(clean_e=True)
    plain = make_cutsky_mc_context(
        cleaner=cleaner, estimator="wiener", var_pix_ref=ctx.var_pix_ref, **_ctx_kwargs(n))
    ref = mc_cutsky_cov_traced(
        jnp.asarray(W_INV), _with_fg(plain, comps[:n].sum(axis=1)), cleaner, fg_residual=True)
    for name in ("covariance", "debiased_bandpowers", "transfer", "leakage",
                 "fg_residual_bandpower", "fg_residual_per_sim"):
        assert np.array_equal(np.asarray(getattr(comp, name)), np.asarray(getattr(ref, name))), name
    assert ref.fg_residual_components is None
    assert comp.fg_residual_components.shape == (2, comp.transfer.shape[0])
    np.testing.assert_allclose(
        np.asarray(comp.fg_residual_components),
        np.asarray(comp.fg_residual_components_per_sim).mean(axis=0), rtol=1e-13)


def test_fg_component_legs_are_linear_in_the_foreground():
    """Exact limits of the split, which a mis-wired leg cannot satisfy.

    * One component zero: the other's leg IS the total leg and the zero leg is 0
      (bitwise -- identical input maps, identical solves).
    * Two identical components g: the total is the leg of 2g, which a converged
      linear filter returns as exactly 4x the leg of g, so the cross
      (total - a - b)/2 equals the auto. Measured exact / 2e-16. Wiring a component
      leg to the total (or to the other component) breaks the first; dropping the
      E projection from one side breaks the second.
    """
    n = 6
    lmax = _ctx_kwargs(n)["lmax"]
    g = _isotropic_fg(n + 1, lmax, slope=1.5)

    _, z = _component_run(jnp.stack([g, jnp.zeros_like(g)], axis=1), n)
    tot = np.asarray(z.fg_residual_per_sim)
    legs = np.asarray(z.fg_residual_components_per_sim)
    assert np.array_equal(tot, legs[:, 0])
    np.testing.assert_array_equal(legs[:, 1], 0.0)

    _, same = _component_run(jnp.stack([g, g], axis=1), n)
    tot = np.asarray(same.fg_residual_per_sim)
    legs = np.asarray(same.fg_residual_components_per_sim)
    np.testing.assert_allclose(tot, 4.0 * legs[:, 0], rtol=1e-12)
    np.testing.assert_allclose(0.5 * (tot - legs[:, 0] - legs[:, 1]), legs[:, 0], rtol=1e-12)


@pytest.mark.slow
def test_fg_component_cross_of_independent_components_is_consistent_with_zero():
    """Independent isotropic components have zero expected cross residual.

    Per-bin t-statistic of the per-sim cross (total - a - b)/2 at 12 sims: measured
    -1.8 / -0.7 / -0.2; gated at |t| < 4. Identical components (the mutation) give
    cross == auto exactly, i.e. an unbounded t.
    """
    n = 12
    lmax = _ctx_kwargs(n)["lmax"]
    dust = _isotropic_fg(n + 1, lmax, slope=1.5)
    sync = _isotropic_fg(n + 1, lmax, slope=-3.0, seed0=77_000)
    _, r = _component_run(jnp.stack([dust, sync], axis=1), n)
    legs = np.asarray(r.fg_residual_components_per_sim)
    cross = 0.5 * (np.asarray(r.fg_residual_per_sim) - legs[:, 0] - legs[:, 1])
    t = cross.mean(axis=0) / (cross.std(axis=0, ddof=1) / np.sqrt(n))
    assert np.all(np.abs(t) < 4.0), t


def test_fg_seed_offset_reaches_the_foreground_draw(monkeypatch):
    """Generated foregrounds are drawn at seed + offset, the setup sim included.

    The CMB reseeds numpy with seed, seed+1, seed+2 -- the first stochastic
    component's seeds at offset 0 -- so the offset is what decorrelates a d11
    realization from its sim's CMB. Recorded at the call, without PySM.
    """
    import augr.spectrum_stages as ss

    seen = []
    real = ss.harmonic_sky

    def spy(*args, **kw):
        seen.append((kw["cmb_seed"], kw["fg_seed"]))
        return real(*args, **{**kw, "fg_model": None})

    monkeypatch.setattr(ss, "harmonic_sky", spy)
    monkeypatch.setattr(ss, "fg_model_is_static", lambda m: False)
    kw = {**_ctx_kwargs(4), "base_seed": 100, "fg_model": "d11s6"}
    ss.make_cutsky_mc_context(
        cleaner=nilc_cleaner(clean_e=True), estimator="wiener", fg_seed_offset=10_000_000, **kw)
    assert seen == [(s, s + 10_000_000) for s in range(100, 105)]

    seen.clear()
    ss.make_cutsky_mc_context(cleaner=nilc_cleaner(clean_e=True), estimator="wiener", **kw)
    assert seen == [(s, None) for s in range(100, 105)]


def test_fg_component_context_shapes_and_guards():
    """Shared and per-sim stacks land on the right ranks; misuse raises."""
    n = 6
    kw = _ctx_kwargs(n)
    lmax = kw["lmax"]
    cleaner = nilc_cleaner(clean_e=True)
    g = _isotropic_fg(n + 1, lmax)

    shared = jnp.stack([g[0], 2.0 * g[0]], axis=0)  # (n_comp, n_band, 2, n_alm)
    ctx = make_cutsky_mc_context(
        cleaner=cleaner, estimator="wiener", fg_comp_eb_alm=shared, **kw)
    assert ctx.harmonic_skies.fg_comp_eb_alm.ndim == 4
    assert ctx.harmonic_skies.fg_eb_alm.ndim == 3
    np.testing.assert_array_equal(
        np.asarray(ctx.harmonic_skies.fg_eb_alm), np.asarray(shared.sum(axis=0)))

    # A per-sim stack that does not vary collapses to the shared form.
    const = jnp.broadcast_to(shared, (n + 1, *shared.shape))
    ctx_c = make_cutsky_mc_context(
        cleaner=cleaner, estimator="wiener", var_pix_ref=ctx.var_pix_ref,
        fg_comp_eb_alm=const, **kw)
    assert ctx_c.harmonic_skies.fg_comp_eb_alm.ndim == 4

    with pytest.raises(ValueError, match="fg_comp_eb_alm has shape"):
        make_cutsky_mc_context(
            cleaner=cleaner, estimator="wiener", fg_comp_eb_alm=const[:n], **kw)
    with pytest.raises(ValueError, match="exclusive"):
        make_cutsky_mc_context(
            cleaner=cleaner, estimator="wiener", fg_comp_eb_alm=shared,
            fg_eb_alm=shared[0], **kw)

    master = make_cutsky_mc_context(
        cleaner=cleaner, estimator="master", fg_comp_eb_alm=shared, **kw)
    with pytest.raises(NotImplementedError, match="masked-Wiener"):
        mc_cutsky_cov_traced(jnp.asarray(W_INV), master, cleaner, fg_residual=True)
    out = mc_cutsky_cov_traced(jnp.asarray(W_INV), master, cleaner)
    assert np.all(np.isfinite(np.asarray(out.covariance)))


@pytest.mark.slow
def test_wiener_and_master_fg_residual_agree_on_an_isotropic_foreground():
    """On an isotropic foreground both legs measure the full-sky residual power.

    The reference is estimator-free: per sim, the cleaner's own B projection of the
    foreground maps on the full sky (``alm2cl / B_c^2``, binned). Measured at 30
    sims (nside=16, 3 bins): Wiener within 1.2% of it, MASTER within 3.8%. The
    shared realizations make the three strongly correlated, so these offsets are
    the estimators' extra cut-sky variance, not the ~5% cosmic variance of the
    reference mean.
    """
    from augr.compsep_sims import assemble_band_maps, beam_harmonic_sky
    from augr.sht import alm2cl

    n_sims = 30
    kw = _ctx_kwargs(n_sims)
    lmax = kw["lmax"]
    fg = _isotropic_fg(n_sims, lmax)
    cleaner = nilc_cleaner(clean_e=True)
    w = jnp.asarray(W_INV)
    res = {}
    for estimator in ("wiener", "master"):
        ctx = _with_fg(
            make_cutsky_mc_context(cleaner=cleaner, estimator=estimator, **kw), fg)
        res[estimator] = np.asarray(
            mc_cutsky_cov_traced(w, ctx, cleaner, fg_residual=True).fg_residual_bandpower
        )

    bf = ctx.beam_fwhm_arcmin
    bl2 = beam_bl(jnp.arange(lmax + 1, dtype=float), min(bf)) ** 2
    ref = []
    for s in range(n_sims):
        sky = beam_harmonic_sky(
            jax.tree.map(lambda a, s=s: a[s], ctx.harmonic_skies), bf, None,
            beam_fwhm_ref=bf)
        total = assemble_band_maps(
            sky, w, ctx.hit_map, noise_key=ctx.noise_keys[s],
            knee_ell=ctx.knee_ell, alpha_knee=ctx.alpha_knee)
        result = cleaner(total, bf, None, lmax=lmax, nside=ctx.nside)
        cl = alm2cl(result.project(sky.fg_qu), lmax) / bl2
        ref.append(np.asarray(mk.bin_spectrum(cl, ctx.bin_matrix, ctx.ell_min)))
    ref = np.mean(ref, axis=0)

    np.testing.assert_allclose(res["wiener"], ref, rtol=0.04)
    np.testing.assert_allclose(res["master"], ref, rtol=0.10)


@pytest.mark.slow
def test_wiener_fg_residual_gradient_matches_finite_differences():
    """d(sum fg_residual)/d(log w_inv) through the Wiener CG matches central FD.

    The residual depends on the noise through the cleaner's weights. Measured at
    h=1e-3: agreement to 1.1e-7, shrinking as h^2 from 1.1e-5 at h=1e-2.
    """
    n_sims = 6
    kw = _ctx_kwargs(n_sims)
    cleaner = nilc_cleaner(clean_e=True)
    ctx = _with_fg(
        make_cutsky_mc_context(cleaner=cleaner, estimator="wiener", **kw),
        _isotropic_fg(n_sims, kw["lmax"]),
    )

    def f(log_w):
        out = mc_cutsky_cov_traced(jnp.exp(log_w), ctx, cleaner, fg_residual=True)
        return jnp.sum(out.fg_residual_bandpower) * 1e6

    x0 = jnp.log(jnp.asarray(W_INV))
    grad = np.asarray(jax.grad(f)(x0))
    h = 1e-3
    fd = np.asarray([
        float((f(x0 + h * jnp.eye(len(W_INV))[i]) - f(x0 - h * jnp.eye(len(W_INV))[i]))
              / (2 * h))
        for i in range(len(W_INV))
    ])
    assert np.all(fd != 0.0)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


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


# --- sim-axis device sharding ------------------------------------------------
#
# JAX fixes the device count at import, so every multi-device arm runs in a child
# process with JAX_NUM_CPU_DEVICES set. n_sims=6 over 4 devices pads to 8
# (sim_batch 1) and 12 (sim_batch 3), so both arms cover the padding path.

_SHARD_CHILD = r'''
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[2])
import test_spectrum_stages as T
from augr import sht
from augr.spectrum_stages import _sim_shard_devices, mc_cutsky_cov_traced

n_dev = int(os.environ["JAX_NUM_CPU_DEVICES"])
assert jax.device_count() == n_dev, (jax.device_count(), n_dev)
expect = 1 if os.environ.get("AUGR_SIM_NO_SHARD") else n_dev
assert _sim_shard_devices() == expect, (_sim_shard_devices(), expect)
sht.set_sht_backend(os.environ["AUGR_TEST_BACKEND"])
mode = os.environ["AUGR_TEST_MODE"]

w = jnp.asarray(T.W_INV)
out = {}
if os.environ.get("AUGR_TEST_DENSE_SHARD"):
    import augr.spectrum_stages as ss
    ss._DENSE_SHARD_CPU = True
if mode == "dense":
    dctx, dcleaner, _kw = T._dense_setup(6)
    for b in (1, 3):
        def f(ww, bb=b):
            return T._sq_cov(ww, dctx, dcleaner, remat=True, sim_batch=bb)
        out[f"sharded{b}"] = np.array(int("shard_map" in str(jax.make_jaxpr(f)(w))))
        res = mc_cutsky_cov_traced(w, dctx, dcleaner, remat=True, sim_batch=b, fg_residual=True)
        out[f"cov{b}"] = np.asarray(res.covariance)
        out[f"fgc{b}"] = np.asarray(res.fg_residual_components)
        out[f"grad{b}"] = np.asarray(jax.grad(f)(w))
    np.savez(sys.argv[1], **out)
    sys.exit(0)
ctx, cleaner = T._master_setup(6)
for b in (1, 3):
    def f(ww, bb=b):
        return T._sq_cov(ww, ctx, cleaner, remat=True, sim_batch=bb)
    out[f"sharded{b}"] = np.array(int("shard_map" in str(jax.make_jaxpr(f)(w))))
    if mode == "run":
        out[f"cov{b}"] = np.asarray(
            mc_cutsky_cov_traced(w, ctx, cleaner, remat=True, sim_batch=b).covariance)
        out[f"grad{b}"] = np.asarray(jax.grad(f)(w))
if mode == "trace":
    wctx, wcleaner = T._traced_setup(6)
    txt = str(jax.make_jaxpr(lambda ww: T._sq_cov(ww, wctx, wcleaner))(w))
    out["wiener_sharded"] = np.array(int("shard_map" in txt))
    dctx, dcleaner, _kw = T._dense_setup(6)
    txt = str(jax.make_jaxpr(lambda ww: T._sq_cov(ww, dctx, dcleaner))(w))
    out["dense_sharded"] = np.array(int("shard_map" in txt))
np.savez(sys.argv[1], **out)
'''


def _shard_child(tmp_path, tag, *, devices, backend="ducc", mode="run", opt_out=False,
                 dense_shard=False):
    """Run ``_SHARD_CHILD`` under ``devices`` CPU devices; return its npz.

    ``dense_shard`` forces the dense Wiener branch to shard on CPU, under
    ``OMP_NUM_THREADS=1`` (see ``spectrum_stages._DENSE_SHARD_CPU``).
    """
    import subprocess
    import sys

    script = tmp_path / "shard_child.py"
    script.write_text(_SHARD_CHILD)
    npz = tmp_path / f"{tag}.npz"
    env = {k: v for k, v in os.environ.items()
           if k not in ("AUGR_SIM_NO_SHARD", "AUGR_TEST_DENSE_SHARD")}
    env.update(JAX_NUM_CPU_DEVICES=str(devices), AUGR_TEST_BACKEND=backend,
               AUGR_TEST_MODE=mode)
    if opt_out:
        env["AUGR_SIM_NO_SHARD"] = "1"
    if dense_shard:
        env.update(AUGR_TEST_DENSE_SHARD="1", OMP_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, str(script), str(npz), os.path.dirname(os.path.abspath(__file__))],
        capture_output=True, text=True, timeout=900, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return np.load(npz)


def test_master_is_sharded_and_wiener_is_not_on_cpu(tmp_path) -> None:
    """Trace-only, 4 CPU devices: a live ``shard_map`` on MASTER; none on the CG Wiener
    branch (its ``while_loop`` makes a batch cost its slowest member) nor, on CPU, the
    dense one (concurrent LAPACK under OpenMP OpenBLAS, ``_DENSE_SHARD_CPU``) unless forced.

    Values cannot tell a sharded map from an unsharded one (they agree to ~1e-15),
    so liveness is checked on the trace.
    """
    out = _shard_child(tmp_path, "trace", devices=4, mode="trace")
    assert out["sharded1"] == 1 and out["sharded3"] == 1, "MASTER sim map not sharded"
    assert out["wiener_sharded"] == 0, "CG masked-Wiener sim map picked up a shard_map"
    assert out["dense_sharded"] == 0, "dense masked-Wiener sharded over CPU devices"
    forced = _shard_child(tmp_path, "trace_forced", devices=4, mode="trace", dense_shard=True)
    assert forced["dense_sharded"] == 1, "forced dense masked-Wiener sim map not sharded"


@pytest.mark.slow
def test_dense_wiener_sharded_matches_one_device(tmp_path) -> None:
    """Dense branch forced onto 4 CPU devices (``OMP_NUM_THREADS=1``), values (incl. the FG
    components) and gradient, at ``sim_batch`` 1 and 3.

    Against the one-device path in the same child env (``AUGR_SIM_NO_SHARD=1``): exact on
    values, 2.3e-16 on one gradient. Against this process (default OpenBLAS threading,
    which reorders the solve's sums): <= 8.4e-15. Both gated at 1e-12 as for MASTER.
    """
    ctx, cleaner, _kw = _dense_setup(6)
    w = jnp.asarray(W_INV)
    sharded = _shard_child(tmp_path, "dense", devices=4, mode="dense", dense_shard=True)
    optout = _shard_child(tmp_path, "dense_optout", devices=4, mode="dense", opt_out=True,
                          dense_shard=True)
    for b in (1, 3):
        res = mc_cutsky_cov_traced(w, ctx, cleaner, remat=True, sim_batch=b, fg_residual=True)
        grad = np.asarray(jax.grad(
            lambda ww, bb=b: _sq_cov(ww, ctx, cleaner, remat=True, sim_batch=bb))(w))
        assert sharded[f"sharded{b}"] == 1 and optout[f"sharded{b}"] == 0, b
        for key, ref in ((f"cov{b}", res.covariance), (f"fgc{b}", res.fg_residual_components),
                         (f"grad{b}", grad)):
            np.testing.assert_allclose(sharded[key], optout[key], rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(optout[key], np.asarray(ref), rtol=1e-12, atol=0.0)


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["ducc", "jht"])
def test_sharded_sim_axis_matches_one_device(tmp_path, backend) -> None:
    """Sharded value and gradient against the same ``sim_batch`` on one device.

    Measured on 2 and 4 CPU devices, both backends, sim_batch 1/2/3: max relative
    difference 6.5e-16 on the covariance and 2.6e-15 on the gradient. Gated at
    1e-12, ~400x margin. ``AUGR_SIM_NO_SHARD=1`` is the one-device path: exact.
    """
    from augr import sht

    w = jnp.asarray(W_INV)
    ref = {}
    with sht.sht_backend(backend):
        ctx, cleaner = _master_setup(6)
        for b in (1, 3):
            ref[f"cov{b}"] = np.asarray(
                mc_cutsky_cov_traced(w, ctx, cleaner, remat=True, sim_batch=b).covariance
            )
            ref[f"grad{b}"] = np.asarray(
                jax.grad(lambda ww, bb=b: _sq_cov(ww, ctx, cleaner, remat=True, sim_batch=bb))(w)
            )
    assert np.all(np.isfinite(ref["grad1"])) and np.any(ref["grad1"] != 0.0)

    sharded = _shard_child(tmp_path, "sharded", devices=4, backend=backend)
    optout = _shard_child(tmp_path, "optout", devices=4, backend=backend, opt_out=True)
    for b in (1, 3):
        assert sharded[f"sharded{b}"] == 1, f"sim_batch={b}: no shard_map in the trace"
        assert optout[f"sharded{b}"] == 0, f"sim_batch={b}: opt-out still sharded"
        np.testing.assert_array_equal(optout[f"cov{b}"], ref[f"cov{b}"])
        np.testing.assert_array_equal(optout[f"grad{b}"], ref[f"grad{b}"])
        np.testing.assert_allclose(sharded[f"cov{b}"], ref[f"cov{b}"], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(sharded[f"grad{b}"], ref[f"grad{b}"], rtol=1e-12, atol=0.0)


def _skies_with_constant_fg(n_sims, n_band, lmax, *, vary=False):
    """A batched CMB-only sky given a synthetic per-band foreground.

    Built by hand rather than through PySM so these stay in the fast tier: the
    mechanism under test is the sharing, not the foreground model. ``vary=True``
    gives an ensemble that genuinely differs per sim, standing in for a
    stochastic preset (d10s6 / s6).
    """
    from augr.compsep_sims import HarmonicSky
    from augr.sht import alm_size

    n_alm = alm_size(lmax)
    rng = np.random.default_rng(0)
    one = (rng.normal(size=(n_band, 2, n_alm))
           + 1j * rng.normal(size=(n_band, 2, n_alm)))
    fg = np.broadcast_to(one, (n_sims, n_band, 2, n_alm)).copy()
    if vary:
        fg[1] += 1.0
    cmb = (rng.normal(size=(n_sims, n_alm)) + 1j * rng.normal(size=(n_sims, n_alm)))
    return HarmonicSky(
        freqs_ghz=FREQS[:n_band], nside=16, lmax=lmax, r_in=0.0,
        cmb_b_alm=jnp.asarray(cmb), fg_eb_alm=jnp.asarray(fg),
        cmb_e_alm=jnp.asarray(cmb),
    )


def test_share_constant_fg_collapses_only_a_constant_ensemble():
    """Rank marks the shared case; a varying ensemble is refused, not averaged.

    The refusal is the anti-vacuity half: a check that accepted anything would
    silently replace a stochastic foreground ensemble with its first realization.
    """
    keep = _skies_with_constant_fg(4, 3, 24)
    shared = share_constant_fg(keep)
    assert shared.fg_eb_alm.ndim == 3
    assert np.array_equal(np.asarray(shared.fg_eb_alm),
                          np.asarray(keep.fg_eb_alm)[0])
    assert shared.fg_eb_alm.nbytes * 4 == keep.fg_eb_alm.nbytes

    # Idempotent, and a no-op on a CMB-only sky.
    assert share_constant_fg(shared) is shared
    assert share_constant_fg(dataclasses.replace(keep, fg_eb_alm=None)).fg_eb_alm is None

    with pytest.raises(ValueError, match="varies across sims"):
        share_constant_fg(_skies_with_constant_fg(4, 3, 24, vary=True))


def test_shared_fg_is_bit_identical_and_survives_the_scan_knobs():
    """Sharing changes storage, not arithmetic -- so nothing may move at all.

    Byte-identical rather than ``allclose``: the shared foreground is re-attached
    inside the body, so every sim sees exactly the array it saw before. The
    ``sim_batch`` leg is the real trap -- ``jax.vmap`` maps over every positional
    argument by default, so a scan constant passed through the vmap rather than
    bound before it would be mapped along the sim axis and silently mismatch.
    """
    n_sims, lmax = 6, 24  # > n_bins + 2, or the Hartlap guard fires first
    per_sim = _skies_with_constant_fg(n_sims, 3, lmax)
    shared = share_constant_fg(per_sim)
    kw = _ctx_kwargs(n_sims, lmax=lmax, ell_max=lmax)
    cleaner = nilc_cleaner(clean_e=True)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in range(n_sims)], axis=0)

    def cov(skies, **extra):
        ctx = make_cutsky_mc_context(
            cleaner=cleaner, estimator="master", harmonic_skies=skies,
            noise_keys=keys, **kw,
        )
        return np.asarray(
            mc_cutsky_cov_traced(jnp.asarray(W_INV), ctx, cleaner, **extra).covariance
        )

    # Vary ONE thing: sharing, at each fixed setting of the other knobs. Comparing
    # a shared sim_batch=2 run against an unshared sim_batch=1 run would fold in
    # that knob's own documented reassociation (measured 2.7e-15 here, and
    # identical with and without sharing -- so it is not ours).
    for sim_batch in (1, 2, 3):
        assert np.array_equal(
            cov(per_sim, sim_batch=sim_batch), cov(shared, sim_batch=sim_batch)
        ), f"sharing moved the covariance at sim_batch={sim_batch}"
    for remat in (True, False):
        assert np.array_equal(cov(per_sim, remat=remat), cov(shared, remat=remat)), (
            f"sharing moved the covariance at remat={remat}"
        )


def test_share_fg_defaults_on_and_keeps_a_varying_ensemble():
    """The default shares a constant ensemble and leaves a varying one alone.

    Defaulting to the strict form would break the stochastic presets (d10s6 / s6)
    at context build, where a per-sim ensemble is the *correct* representation
    rather than an error -- so the builder uses the non-raising form. The varying
    leg is the anti-vacuity half: a builder that shared unconditionally would
    replace the ensemble with its first realization and still pass the first.
    """
    n_sims, lmax = 6, 24
    kw = _ctx_kwargs(n_sims, lmax=lmax, ell_max=lmax)
    cleaner = nilc_cleaner(clean_e=True)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in range(n_sims)], axis=0)

    def built(skies, **extra):
        return make_cutsky_mc_context(
            cleaner=cleaner, estimator="master", harmonic_skies=skies,
            noise_keys=keys, **kw, **extra,
        ).harmonic_skies.fg_eb_alm

    constant = _skies_with_constant_fg(n_sims, 3, lmax)
    varying = _skies_with_constant_fg(n_sims, 3, lmax, vary=True)

    assert built(constant).ndim == 3, "the default did not share a constant ensemble"
    assert built(varying).ndim == 4, "a varying ensemble was collapsed"
    assert np.array_equal(
        np.asarray(built(varying)), np.asarray(varying.fg_eb_alm)
    ), "a varying ensemble was modified"
    assert built(constant, share_fg=False).ndim == 4, "share_fg=False was ignored"
