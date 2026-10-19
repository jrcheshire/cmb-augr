"""Gate for augr.masking: cut-sky B-mode estimation via jht's masked Wiener filter.

Fast tests cover the mask / inverse-noise / binning / debias algebra and an
estimator shape-smoke. Slow tests are the science checkpoints (require the
heavier nside): the **purity null** (E-only input → B leakage ≪ the lensing-BB
floor, on ≥2 masks, per-sim), the **fidelity** companion (B-only input recovered
post-debias), and a **NaMaster** cross-check of the mean bandpower normalization.

The acceptance bar (per the design): the Wiener filter must control E→B leakage
to well below the lensing-BB level (so the leaked-E cosmic variance does not
dominate σ(r)), and the multiplicative transfer + additive leakage-template
debias must recover an unbiased B bandpower. Map work needs jht (the [masking]
extra); the NaMaster cross-check additionally needs pymaster.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("jht")

import healpy as hp
import jax.numpy as jnp

from augr import masking as mk
from augr.compsep_sims import cmb_b_alm, cmb_e_alm, cmb_eb_qu
from augr.delensing import load_lensing_spectra
from augr.instrument import beam_bl
from augr.signal import _build_bin_matrix, _make_bin_edges
from augr.spectra import CMBSpectra

# ---------------------------------------------------------------------------
# fast: masks
# ---------------------------------------------------------------------------


def test_gal_cut_mask_full_sky():
    """b_cut = 0 leaves the whole sky (all ones)."""
    m = mk.gal_cut_mask(16, 0.0)
    assert m.shape == (hp.nside2npix(16),)
    assert float(jnp.min(m)) == 1.0


def test_galactic_mask_hits_target_fsky():
    """galactic_mask(nside, f_sky) realizes ~f_sky up to pixelization."""
    for target in (0.4, 0.6, 0.8):
        m = mk.galactic_mask(128, target)
        assert set(np.unique(np.asarray(m))) <= {0.0, 1.0}  # binary
        assert mk.f_sky_of(m) == pytest.approx(target, abs=0.02)


def test_load_mask_roundtrip(tmp_path):
    """load_mask reads a FITS mask and ud_grades to the requested nside."""
    m = np.asarray(mk.galactic_mask(64, 0.7))
    path = str(tmp_path / "mask.fits")
    hp.write_map(path, m, overwrite=True)
    loaded = mk.load_mask(path)
    np.testing.assert_allclose(np.asarray(loaded), m)
    up = mk.load_mask(path, nside=128)
    assert up.shape == (hp.nside2npix(128),)


# ---------------------------------------------------------------------------
# fast: inverse-noise
# ---------------------------------------------------------------------------


def test_inv_noise_map_mask_and_hits():
    """inv_noise is 0 off-mask / at zero hits, and scales linearly with hits."""
    nside = 16
    npix = hp.nside2npix(nside)
    hits = jnp.asarray(np.linspace(1.0, 5.0, npix))
    mask = mk.gal_cut_mask(nside, 30.0)
    var_pix = 2.0
    inv = mk.inv_noise_map(hits, var_pix, mask=mask)
    # off-mask pixels are exactly zero
    assert float(jnp.max(jnp.where(jnp.asarray(mask) > 0, 0.0, inv))) == 0.0
    # on-mask: inv = hits / (var_pix * max_hits)
    expect = np.asarray(hits) / (var_pix * float(jnp.max(hits)))
    on = np.asarray(mask) > 0
    np.testing.assert_allclose(np.asarray(inv)[on], expect[on], rtol=1e-12)

    # zero-hit pixels → zero inv_noise (no mask)
    h2 = jnp.asarray(np.where(np.arange(npix) % 2 == 0, 0.0, 3.0))
    inv2 = mk.inv_noise_map(h2, 1.0)
    assert float(jnp.max(inv2[::2])) == 0.0


def test_var_pix_ref_uniform_hits_is_the_plain_mean():
    """Uniform hits: bitwise the observed-pixel mean of (Q² + U²)/2 (pre-hits behaviour)."""
    nside = 16
    npix = hp.nside2npix(nside)
    mask = mk.gal_cut_mask(nside, 30.0)
    qu = jnp.asarray(np.random.default_rng(3).standard_normal((2, npix)) * 1.7)
    obs = jnp.asarray(mask) > 0
    plain = float(jnp.mean((qu[0] ** 2 + qu[1] ** 2)[obs]) / 2.0)
    assert mk.var_pix_ref_from_noise(qu, jnp.ones(npix), mask) == plain


def test_var_pix_ref_recovers_the_max_exposure_level():
    """Noise drawn at ``σ²(p) = v · max(H)/H(p)`` on the L2 envelope returns ``v``.

    That is :func:`inv_noise_map`'s convention, so the filter's ``N⁻¹`` matches the noise.
    Measured at nside 32 / 30° cut (6272 pixels) over 200 seeds: estimate/v mean 0.9997,
    std 0.013, max |dev| 0.036. The observed-pixel mean (the pre-fix estimate) gives
    4.6-4.8x. Gate |estimate/v - 1| < 0.07 (~5 sigma).
    """
    from augr.hit_maps import l2_hit_map

    nside = 32
    hits = np.asarray(l2_hit_map(nside), dtype=float)
    mask = mk.gal_cut_mask(nside, 30.0)
    v = 2.0
    sig2 = v * hits.max() / hits
    qu = jnp.asarray(np.random.default_rng(0).standard_normal((2, hits.size)) * np.sqrt(sig2))
    est = mk.var_pix_ref_from_noise(qu, jnp.asarray(hits), mask)
    assert abs(est / v - 1.0) < 0.07


# ---------------------------------------------------------------------------
# fast: binning + debias algebra
# ---------------------------------------------------------------------------


def test_bin_spectrum_matches_manual():
    """bin_spectrum slices from ell_min and applies the bin matrix."""
    ell_min, ell_max = 2, 50
    edges = _make_bin_edges(ell_min, ell_max, 2, 10)
    bm, _ = _build_bin_matrix(np.arange(ell_min, ell_max + 1), edges, "tophat")
    cl = jnp.asarray(np.arange(0, ell_max + 5, dtype=float) ** 2)  # full ℓ=0.. grid
    binned = mk.bin_spectrum(cl, bm, ell_min)
    manual = np.asarray(bm) @ np.asarray(cl)[ell_min : ell_min + bm.shape[1]]
    assert binned.shape == (bm.shape[0],)
    np.testing.assert_allclose(np.asarray(binned), manual, rtol=1e-12)


def test_debias_inverts_model_nondegenerate():
    """debias recovers C_true from C_rec = F·C_true + leak with distinct per-bin F, leak.

    Non-degenerate inputs (distinct F_b, leak_b, C_true_b per bin) so the test is
    not measure-zero — cf. feedback_controlled_tests_measure_zero.
    """
    F = jnp.asarray([0.6, 0.7, 0.8, 0.9, 0.95])
    leak = jnp.asarray([3.0, 2.0, 1.0, 0.5, 0.2])
    c_true = jnp.asarray([10.0, 8.0, 6.0, 4.0, 2.0])

    # B-only sims (E=0, so leak=0): rec = F·c_true, two sims with identical CRN mean
    rec_b_only = jnp.stack([F * c_true, F * c_true])
    F_rec = mk.transfer_function(rec_b_only, c_true)
    np.testing.assert_allclose(np.asarray(F_rec), np.asarray(F), rtol=1e-12)

    # E-only sims (B=0): rec = leak
    rec_e_only = jnp.stack([leak, leak])
    leak_rec = mk.leakage_template(rec_e_only)
    np.testing.assert_allclose(np.asarray(leak_rec), np.asarray(leak), rtol=1e-12)

    # full model and inversion
    c_rec = F * c_true + leak
    recovered = mk.debias_bandpower(c_rec, F_rec, leak_rec)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(c_true), rtol=1e-12)


# ---------------------------------------------------------------------------
# fast: estimator shape-smoke
# ---------------------------------------------------------------------------


def test_masked_wiener_bb_shape_finite():
    """Estimator returns a finite per-ℓ C_BB of length lmax+1."""
    nside, lmax = 32, 48
    npix = hp.nside2npix(nside)
    cl_ee = jnp.ones(lmax + 1).at[:2].set(0.0) * 1e-2
    cl_bb = jnp.ones(lmax + 1).at[:2].set(0.0) * 1e-3
    rng = np.random.default_rng(0)
    qu = jnp.asarray(rng.standard_normal((2, npix)))
    mask = mk.galactic_mask(nside, 0.7)
    invn = mk.inv_noise_map(jnp.ones(npix), 1.0, mask=mask)
    cl = mk.masked_wiener_bb(qu, invn, cl_ee, cl_bb, nside=nside, lmax=lmax)
    assert cl.shape == (lmax + 1,)
    assert bool(jnp.all(jnp.isfinite(cl)))


def _small_wiener_problem(nside=16, lmax=24):
    npix = hp.nside2npix(nside)
    cl_ee = jnp.ones(lmax + 1).at[:2].set(0.0) * 1e-2
    cl_bb = jnp.ones(lmax + 1).at[:2].set(0.0) * 1e-3
    qu = jnp.asarray(np.random.default_rng(1).standard_normal((2, npix)))
    invn = mk.inv_noise_map(jnp.ones(npix), 1.0, mask=mk.galactic_mask(nside, 0.6))
    return qu, invn, cl_ee, cl_bb, dict(nside=nside, lmax=lmax)


def test_masked_wiener_default_call_is_the_stock_jht_solve():
    """With no precond / return_info the call is jht's stock-``cg`` path, bitwise."""
    import jht

    qu, invn, cl_ee, cl_bb, kw = _small_wiener_problem()
    ours = mk.masked_wiener_eb_alm(qu, invn, cl_ee, cl_bb, **kw)
    stock = jht.wiener(qu, (cl_ee, cl_bb), kw["nside"], kw["lmax"], spin=2,
                       inv_noise=invn, max_iter=1000, tol=1e-8)
    assert bool(jnp.array_equal(ours, stock))


def test_masked_wiener_return_info_reports_a_truncated_solve():
    """``max_iter=3`` stops unconverged and says so; the jht stopping rule is ``rel <= tol``."""
    qu, invn, cl_ee, cl_bb, kw = _small_wiener_problem()
    _alm, info = mk.masked_wiener_eb_alm(qu, invn, cl_ee, cl_bb, max_iter=3,
                                         return_info=True, **kw)
    assert int(info.n_iter) == 3
    assert float(info.rel_residual) > 1e-8
    cl, info = mk.masked_wiener_bb(qu, invn, cl_ee, cl_bb, return_info=True, **kw)
    assert cl.shape == (kw["lmax"] + 1,)
    assert float(info.rel_residual) <= 1e-8 and int(info.n_iter) < 1000


def test_wiener_preconditioner_low_block_size():
    """The exact block covers the spin-2 DOFs with ``l <= ell_split``: ``2((l_s+1)^2 - 4)``."""
    qu, invn, cl_ee, cl_bb, kw = _small_wiener_problem()
    p8 = mk.wiener_preconditioner(invn, cl_ee, cl_bb, ell_split=8, **kw)
    assert p8.low_idx.shape == (2 * (9**2 - 4),)
    assert p8.low_inv.shape == (154, 154)
    diag = mk.wiener_preconditioner(invn, cl_ee, cl_bb, ell_split=None, **kw)
    assert diag.low_idx.shape == (0,)
    # ell_split >= lmax: the block is the whole operator, so the PCG finishes at once.
    full = mk.wiener_preconditioner(invn, cl_ee, cl_bb, ell_split=kw["lmax"], **kw)
    _alm, info = mk.masked_wiener_eb_alm(qu, invn, cl_ee, cl_bb, precond=full,
                                         return_info=True, **kw)
    assert int(info.n_iter) <= 2


# ---------------------------------------------------------------------------
# exact bandpower window (dense B→B response)
# ---------------------------------------------------------------------------
# nside 16 / lmax 24 (n_dof 1242), lensing priors, non-uniform hits, binary f_sky 0.6
# mask. var_pix 1e-5 puts the filter mid-way between noise- and signal-dominated
# (diag K 0.03-0.29 at l = 2..7, off-diagonal mass ~0.4 of the diagonal), so the
# l-coupling the window exists for is large. Margins measured 2026-10-13.

_WNS, _WLMAX, _WVAR = 16, 24, 1e-5


@pytest.fixture(scope="module")
def dense_window():
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: _WLMAX + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: _WLMAX + 1], 0.0, None)
    hits = jnp.asarray(np.linspace(1.0, 3.0, hp.nside2npix(_WNS)))
    invn = mk.inv_noise_map(hits, _WVAR, mask=mk.galactic_mask(_WNS, 0.6))
    normal = mk.wiener_normal_matrix(invn, nside=_WNS, lmax=_WLMAX)
    K = mk.wiener_bb_response(normal, cl_ee, cl_bb, lmax=_WLMAX)
    return invn, cl_ee, cl_bb, normal, K


def test_wiener_normal_matrix_reproduces_the_jht_wiener_mean(dense_window):
    """``P (P A P + I)⁻¹ P b`` from the dense ``A`` == ``jht.wiener`` on a random map.

    Checks the operator, the DOF layout and the per-DOF ℓ index at once. Measured
    4.4e-12 (max|Δx| / max|x|) against a tol-1e-13 CG solve; gate 1e-10.
    """
    import jht

    invn, cl_ee, cl_bb, normal, _K = dense_window
    npix = hp.nside2npix(_WNS)
    qu = jnp.asarray(np.random.default_rng(3).standard_normal((2, npix)))
    ell = mk._dof_ell(_WLMAX)
    n_e = ell.size // 2
    pw = np.sqrt(np.concatenate([np.asarray(cl_ee)[ell[:n_e]], np.asarray(cl_bb)[ell[n_e:]]]))
    b = np.asarray(jht.alm_to_real(
        jht.adjoint_synthesis(invn[None, :] * qu, _WNS, _WLMAX, spin=2), _WLMAX, spin=2))
    x_dense = pw * np.linalg.solve(pw[:, None] * normal * pw[None, :] + np.eye(ell.size), pw * b)
    alm = jht.wiener(qu, (cl_ee, cl_bb), _WNS, _WLMAX, spin=2, inv_noise=invn, tol=1e-13,
                     max_iter=5000)
    x_cg = np.asarray(jht.alm_to_real(alm, _WLMAX, spin=2))
    assert np.max(np.abs(x_dense - x_cg)) / np.max(np.abs(x_cg)) < 1e-10


@pytest.mark.parametrize("l0", [2, 9])
def test_wiener_bb_response_matches_direct_cg_solves(dense_window, l0):
    """``K[:, l0]`` == output B power summed over CG solves of every unit B DOF at ``l0``.

    An independent path: no dense algebra, no prior-ratio formula. Measured 2.0e-11
    (l0=2) and 3.5e-13 (l0=9), max|ΔK| / max|K[:, l0]|; gate 1e-9.
    """
    import jht

    invn, cl_ee, cl_bb, _normal, K = dense_window
    ell = mk._dof_ell(_WLMAX)
    n_e = ell.size // 2
    col = np.zeros(_WLMAX + 1)
    for d in np.flatnonzero(ell[n_e:] == l0):
        unit = np.zeros(ell.size)
        unit[n_e + d] = 1.0
        qu = jht.synthesis(jht.real_to_alm(jnp.asarray(unit), _WLMAX, spin=2), _WNS, _WLMAX, spin=2)
        out, info = jht.wiener(qu, (cl_ee, cl_bb), _WNS, _WLMAX, spin=2, inv_noise=invn,
                               tol=1e-13, max_iter=20000, return_info=True)
        assert float(info.rel_residual) <= 1e-13
        col += np.asarray(jht.bandpower(out, _WLMAX, spin=2)[1])
    assert np.max(np.abs(K[:, l0] - col)) / np.max(np.abs(col)) < 1e-9


def test_wiener_bb_response_scale_matches_scaled_noise(dense_window):
    """``K(A(N⁻¹), scale=s)`` == ``K(A(s·N⁻¹))``: one normal matrix serves every ``var_pix_ref``.

    Measured 4.1e-15; gate 1e-12.
    """
    invn, cl_ee, cl_bb, normal, _K = dense_window
    s = 3.7
    k_scaled = mk.wiener_bb_response(normal, cl_ee, cl_bb, lmax=_WLMAX, scale=s)
    normal_s = mk.wiener_normal_matrix(invn * s, nside=_WNS, lmax=_WLMAX)
    k_direct = mk.wiener_bb_response(normal_s, cl_ee, cl_bb, lmax=_WLMAX)
    assert np.max(np.abs(k_scaled - k_direct)) / np.max(np.abs(k_direct)) < 1e-12


def test_debiased_window_reproduces_its_calibration_spectrum(dense_window):
    """With ``transfer`` = the analytic ``F_b`` at ``C``, ``W @ C`` == ``bin(C)``.

    Algebra, but it pins the ℓ offsets between ``bin_matrix`` (from ``ell_min``), ``K``
    (from 0) and ``W`` (from ``ell_min`` to lmax). Measured 2e-16; gate 1e-12.
    """
    _invn, _ee, cl_bb, _normal, K = dense_window
    edges = _make_bin_edges(2, 20, 2, 5)
    bm, _ = _build_bin_matrix(np.arange(2, 21), edges, "tophat")
    c = np.asarray(cl_bb) * 1.3
    F = mk.analytic_transfer(K, bm, 2, common_fwhm_arcmin=60.0, true_cl_bb=c)
    W, ells = mk.debiased_window(K, bm, 2, common_fwhm_arcmin=60.0, transfer=F)
    assert ells[0] == 2 and ells[-1] == _WLMAX and W.shape == (bm.shape[0], _WLMAX - 1)
    ratio = (W @ c[ells]) / np.asarray(mk.bin_spectrum(c, bm, 2))
    assert np.max(np.abs(ratio - 1.0)) < 1e-12


def test_wiener_bb_response_rejects_a_zero_bb_prior(dense_window):
    """The ``P⁻¹`` form needs ``C_BB > 0`` on ℓ = 2..lmax."""
    _invn, cl_ee, cl_bb, normal, _K = dense_window
    with pytest.raises(ValueError, match="positive"):
        mk.wiener_bb_response(normal, cl_ee, cl_bb.at[7].set(0.0), lmax=_WLMAX)


@pytest.mark.slow
def test_wiener_bb_response_predicts_the_mc_mean(dense_window):
    """Mean raw Wiener BB of noise-free B-only sims == ``K @ C`` for a non-lensing ``C``.

    The truth is a power law (not the lensing shape the prior carries), drawn with
    ``healpy.synalm``, so neither the DOF-variance convention nor the spectrum comes
    from ``K``'s own construction. Every solve is checked converged: at higher S/N
    (var 1e-7, cond(M+I) 4.6e7) plain CG needs ~15,500 iterations and a silently
    truncated solve fails this test. Measured over ℓ = 2..24 with 48 sims: χ²/dof 1.07
    with ``K``, 36.5 with ``K``'s diagonal (the mutation); gates < 2 and > 10.
    """
    import jax
    import jht

    invn, cl_ee, cl_bb, _normal, K = dense_window
    ell = np.arange(_WLMAX + 1)
    c = np.where(ell >= 2, 1e-6 * (np.maximum(ell, 2) / 10.0) ** -1.0, 0.0)
    n_sims = 48
    alms = []
    for i in range(n_sims):
        np.random.seed(1000 + i)  # noqa: NPY002 - healpy.synalm uses the global RNG
        alms.append(hp.synalm(c, lmax=_WLMAX, new=True))
    b = jnp.asarray(np.array(alms))
    qu = jax.vmap(lambda a: jht.synthesis(jnp.stack([jnp.zeros_like(a), a]), _WNS, _WLMAX, spin=2))(b)
    solve = jax.jit(jax.vmap(lambda q: mk.masked_wiener_bb(
        q, invn, cl_ee, cl_bb, nside=_WNS, lmax=_WLMAX, tol=1e-10, max_iter=20000,
        return_info=True)))
    recs, info = solve(qu)
    assert float(jnp.max(info.rel_residual)) <= 1e-10
    recs = np.asarray(recs)
    mean, err = recs.mean(0), recs.std(0, ddof=1) / np.sqrt(n_sims)

    def chi2_dof(pred):
        z = (mean - pred)[2:] / err[2:]
        return float(np.mean(z**2))

    assert chi2_dof(K @ c) < 2.0
    assert chi2_dof(np.diag(K) * c) > 10.0


# ---------------------------------------------------------------------------
# slow: science checkpoints  (tolerances set from the nside=256 validation run)
# ---------------------------------------------------------------------------

# Validation config. nside=128/lmax=192 keeps the reion (ℓ≲10) and recombination
# (ℓ~80) bumps where the r-information lives while keeping the heavy spin-2 Wiener
# solves affordable for a `slow` test. The science was cross-checked at
# nside=256/lmax=300 during development (same conclusions, ~4× slower).
_NS, _LMAX, _FWHM = 128, 192, 30.0
_ELL_MIN, _ELL_MAX = 2, 150


def _validation_setup():
    ls = load_lensing_spectra()
    ell = np.arange(_LMAX + 1)
    bl = np.asarray(beam_bl(jnp.asarray(ell, dtype=float), _FWHM))
    cl_ee = jnp.clip(ls.cl_ee_len[: _LMAX + 1], 0.0, None)
    cl_bb_len = jnp.clip(ls.cl_bb_len[: _LMAX + 1], 0.0, None)
    # beamed priors describe the (beamed) data covariance; the transfer function
    # absorbs the beam multiplicatively so debias recovers the unbeamed C_ℓ.
    cl_ee_p = cl_ee * bl**2
    cl_bb_p = cl_bb_len * bl**2
    spec = CMBSpectra()
    edges = _make_bin_edges(_ELL_MIN, _ELL_MAX, 2, 20)
    bm, centers = _build_bin_matrix(np.arange(_ELL_MIN, _ELL_MAX + 1), edges, "tophat")
    return ls, ell, bl, cl_ee, cl_bb_len, cl_ee_p, cl_bb_p, spec, bm, centers


def _estimate(qu, invn, cl_ee_p, cl_bb_p, bm, *, npix, var_pix, seed):
    noise = jnp.asarray(np.random.default_rng(seed).standard_normal((2, npix)) * np.sqrt(var_pix))
    cl = mk.masked_wiener_bb(qu + noise, invn, cl_ee_p, cl_bb_p, nside=_NS, lmax=_LMAX)
    return mk.bin_spectrum(cl, bm, _ELL_MIN)


def _b_only_qu(spec, r, seed):
    b = cmb_b_alm(spec, r, _LMAX, seed=seed)
    return cmb_eb_qu(jnp.zeros_like(b), b, _FWHM, _LMAX, _NS)


def _e_only_qu(cl_ee, seed):
    e = cmb_e_alm(cl_ee, _LMAX, seed=seed)
    return cmb_eb_qu(e, jnp.zeros_like(e), _FWHM, _LMAX, _NS)


@pytest.mark.slow
def test_purity_null_multimask():
    """E-only (B=0) input → recovered B ≪ the lensing-BB floor, per-sim, ≥2 masks.

    The signal-dominated limit isolates the estimator's E→B leakage. Asserts the
    per-sim leakage/lensing-BB ratio (not just the mean) stays small on every
    mask — cf. feedback_distribution_comparison_rigor. Validation runs show
    ~2–7e-4; the 0.05 bound is a loose robustness gate, not a tuned tolerance.
    """
    _ls, _ell, _bl, cl_ee, cl_bb_len, cl_ee_p, cl_bb_p, _spec, bm, _c = _validation_setup()
    npix = hp.nside2npix(_NS)
    var_pix = 1e-5
    true_bb_floor = mk.bin_spectrum(cl_bb_len, bm, _ELL_MIN)

    for f_sky in (0.6, 0.4):
        mask = mk.galactic_mask(_NS, f_sky)
        invn = mk.inv_noise_map(jnp.ones(npix), var_pix, mask=mask)
        for s in range(2):
            rec = _estimate(
                _e_only_qu(cl_ee, 200 + s),
                invn,
                cl_ee_p,
                cl_bb_p,
                bm,
                npix=npix,
                var_pix=var_pix,
                seed=s + 50,
            )
            ratio = np.asarray(rec) / np.asarray(true_bb_floor)
            # leakage is everywhere well below 5% of the lensing-BB floor
            assert np.max(np.abs(ratio)) < 0.05, (f_sky, s, np.max(np.abs(ratio)))


@pytest.mark.slow
def test_fidelity_recovers_input_bb():
    """B-only input → transfer+leakage debias recovers the input BB (unbiased in the mean).

    Smooth-non-constant companion to the purity null. The multiplicative transfer
    F_b must be sane (0 < F_b ≲ 1; it absorbs the beam B_ℓ² and the Wiener gain),
    and the debias must be unbiased. Recovery is asserted on the **mean over
    held-out sims** rather than a single realization: at the reion band (lowest
    bin, ~few modes) a single realization scatters ±40% by cosmic variance, so a
    single-sim assertion would test cosmic variance, not the debias. Averaging
    isolates the (multiplicative + additive) bias the debias is meant to remove.
    """
    _ls, ell, _bl, cl_ee, _cl_bb_len, cl_ee_p, cl_bb_p, spec, bm, centers = _validation_setup()
    npix = hp.nside2npix(_NS)
    var_pix = 1e-5
    mask = mk.galactic_mask(_NS, 0.6)
    invn = mk.inv_noise_map(jnp.ones(npix), var_pix, mask=mask)
    cl_bb_true = jnp.clip(spec.cl_bb(jnp.asarray(ell, dtype=float), 0.01), 0.0, None)
    true_b = mk.bin_spectrum(cl_bb_true, bm, _ELL_MIN)

    # transfer (B-only) and leakage (E-only) templates from independent sim sets
    rec_b = jnp.stack(
        [
            _estimate(
                _b_only_qu(spec, 0.01, 100 + s),
                invn,
                cl_ee_p,
                cl_bb_p,
                bm,
                npix=npix,
                var_pix=var_pix,
                seed=s,
            )
            for s in range(3)
        ]
    )
    F = mk.transfer_function(rec_b, true_b)
    assert np.all(np.asarray(F) > 0) and np.all(np.asarray(F) < 1.2)

    leak = mk.leakage_template(
        jnp.stack(
            [
                _estimate(
                    _e_only_qu(cl_ee, 300 + s),
                    invn,
                    cl_ee_p,
                    cl_bb_p,
                    bm,
                    npix=npix,
                    var_pix=var_pix,
                    seed=s + 50,
                )
                for s in range(3)
            ]
        )
    )

    # mean debiased recovery over held-out B-only sims (cosmic variance averaged down)
    deb = jnp.stack(
        [
            mk.debias_bandpower(
                _estimate(
                    _b_only_qu(spec, 0.01, 500 + s),
                    invn,
                    cl_ee_p,
                    cl_bb_p,
                    bm,
                    npix=npix,
                    var_pix=var_pix,
                    seed=700 + s,
                ),
                F,
                leak,
            )
            for s in range(3)
        ]
    )
    ratio = np.asarray(jnp.mean(deb, axis=0)) / np.asarray(true_b)
    # mean recovery is unbiased to ~15% across the band; the lowest bin keeps
    # the most residual cosmic variance even after averaging 4 sims → 25% there.
    lowest = np.asarray(centers) <= 21
    assert np.all(np.abs(ratio[~lowest] - 1.0) < 0.15), ratio
    assert np.all(np.abs(ratio[lowest] - 1.0) < 0.25), ratio


@pytest.mark.slow
def test_namaster_mean_bandpower_crosscheck():
    """Cross-check the masked-Wiener BB normalization against NaMaster (MASTER).

    NaMaster's mode-coupling-corrected pseudo-Cℓ is a non-differentiable,
    independent estimator; this validates the *mean bandpower normalization* (a
    different axis from purity). A **B-only** input sim is used, so there is no
    E→B mixing and ``purify_b=False`` (plain MASTER) is the right, robust choice
    — purification on a sharp binary mask is both unnecessary here and
    numerically fragile. Skips if pymaster is not installed.

    .. warning::
       **CIRCULAR AS WRITTEN — do not trust this as a masked-Wiener gate.**
       ``transfer_function(rec[None, :], true_b)`` returns ``rec / true_b``, so
       ``debiased`` below equals ``true_b`` *identically* for a single sim. What is
       actually asserted is "NaMaster's decoupled BB matches the analytic input BB
       to 30% in the median above ℓ=40" — a NaMaster-vs-truth check with the
       masked-Wiener estimator algebraically cancelled out of it. A real gate needs
       a transfer frozen on an *independent* B-only ensemble, as
       :func:`test_fidelity_recovers_input_bb` does. Left in place (it is a valid,
       if weak, NaMaster sanity check) rather than silently "fixed"; the
       non-circular pattern lives in ``tests/test_pseudo_cl.py``.
    """
    pymaster = pytest.importorskip("pymaster")
    _ls, ell, _bl, _cl_ee, _cl_bb_len, cl_ee_p, cl_bb_p, spec, bm, centers = _validation_setup()
    npix = hp.nside2npix(_NS)
    var_pix = 1e-5
    mask = mk.galactic_mask(_NS, 0.6)
    invn = mk.inv_noise_map(jnp.ones(npix), var_pix, mask=mask)

    # one B-only sim (no E → no leakage; purification unnecessary)
    b = cmb_b_alm(spec, 0.05, _LMAX, seed=7)
    qu = np.asarray(cmb_eb_qu(jnp.zeros_like(b), b, _FWHM, _LMAX, _NS))
    qu = qu + np.random.default_rng(7).standard_normal((2, npix)) * np.sqrt(var_pix)

    # NaMaster MASTER BB on the same mask + binning
    # field lmax must match the bin lmax (NmtField defaults to 3·nside−1)
    fld = pymaster.NmtField(np.asarray(mask), [qu[0], qu[1]], spin=2, purify_b=False, lmax=_LMAX)
    edges = _make_bin_edges(_ELL_MIN, _ELL_MAX, 2, 20)
    bpws = np.full(_LMAX + 1, -1, dtype=int)
    for ib, (lo, hi) in enumerate(edges):
        bpws[lo : hi + 1] = ib
    bins = pymaster.NmtBin(
        bpws=bpws, ells=np.arange(_LMAX + 1), weights=np.ones(_LMAX + 1), lmax=_LMAX
    )
    wsp = pymaster.NmtWorkspace.from_fields(fld, fld, bins)
    cl_nmt = wsp.decouple_cell(pymaster.compute_coupled_cell(fld, fld))[3]  # BB
    nmt_b = cl_nmt[: len(centers)]

    # masked-Wiener debiased (transfer-only; B-only so no E leakage to subtract)
    cl_bb_true = jnp.clip(spec.cl_bb(jnp.asarray(ell, dtype=float), 0.05), 0.0, None)
    true_b = mk.bin_spectrum(cl_bb_true, bm, _ELL_MIN)
    rec = mk.bin_spectrum(
        mk.masked_wiener_bb(jnp.asarray(qu), invn, cl_ee_p, cl_bb_p, nside=_NS, lmax=_LMAX),
        bm,
        _ELL_MIN,
    )
    debiased = np.asarray(rec / mk.transfer_function(rec[None, :], true_b))

    # both estimate the same sky BB → agree in the mean over the well-measured
    # band (skip the lowest, few-mode bins where single-sim scatter is large)
    sel = np.asarray(centers) > 40
    ratio = debiased[sel] / nmt_b[sel]
    assert np.all(np.isfinite(ratio))
    assert np.abs(np.median(ratio) - 1.0) < 0.3, np.median(ratio)


# ---------------------------------------------------------------------------
# Differentiable mask families (the sky-coverage design axis)
# ---------------------------------------------------------------------------


def test_smooth_gal_cut_mask_is_differentiable_in_both_parameters():
    """The whole point: gal_cut_mask/galactic_mask carry no gradient at all.

    Both threshold with a numpy comparison, so ``d f_sky / d b_cut`` does not
    exist for them. The sigmoid family is smooth in the cut latitude *and* in
    the width, which is what lets sky coverage be a design coordinate.
    """
    import jax

    nside = 32
    d_bcut = float(jax.grad(
        lambda b: jnp.mean(mk.smooth_gal_cut_mask(nside, b, 5.0)))(25.0))
    d_width = float(jax.grad(
        lambda w: jnp.mean(mk.smooth_gal_cut_mask(nside, 25.0, w)))(5.0))
    assert np.isfinite(d_bcut) and np.isfinite(d_width)
    # cutting to higher |b| can only remove sky
    assert d_bcut < 0.0


def test_smooth_gal_cut_mask_f_sky_is_monotone_and_bounded():
    nside = 32
    f = [float(jnp.mean(mk.smooth_gal_cut_mask(nside, b, 5.0)))
         for b in (10.0, 20.0, 30.0, 40.0, 50.0)]
    assert all(a > b for a, b in itertools.pairwise(f)), f
    assert min(f) > 0.0 and max(f) < 1.0


def test_smooth_mask_approaches_the_sharp_cut_as_width_shrinks():
    """The family really is a smoothing of gal_cut_mask -- but pixelization floors it.

    Measured |f_sky - sharp| at nside=32, cut 25 deg: 9.1e-3 / 4.7e-3 / 4.5e-3 /
    1.1e-5 at width = 5 / 1 / 0.5 / 0.05 deg. Note the **plateau** near 4.5e-3
    between widths 1 and 0.5: once the taper is narrower than the HEALPix ring
    spacing it stops being resolved, and only below that does it collapse onto
    the sharp cut. That floor is pixelization, not an implementation defect, and
    it is the other half of why width_deg should stay at or above a pixel scale.
    """
    nside = 32
    sharp = np.asarray(mk.gal_cut_mask(nside, 25.0)).mean()
    wide = np.abs(np.asarray(mk.smooth_gal_cut_mask(nside, 25.0, 5.0)).mean() - sharp)
    narrow = np.abs(np.asarray(mk.smooth_gal_cut_mask(nside, 25.0, 0.05)).mean() - sharp)
    assert wide < 2e-2, wide
    assert narrow < 1e-4, narrow
    assert narrow < wide


def test_admission_rank_and_level_sets_reproduce_a_nested_family():
    """At integer t with a narrow width, the level set IS the family member."""
    nside = 32
    cuts = [10.0, 20.0, 30.0, 40.0]
    stack = jnp.stack([mk.gal_cut_mask(nside, c) for c in cuts])
    rank = mk.admission_rank(stack)
    assert set(np.asarray(rank).astype(int).tolist()) == {0, 1, 2, 3, 4}

    # rank >= k <=> admitted by the k most permissive cuts
    for k, cut in enumerate(sorted(cuts), start=1):
        recovered = np.asarray(mk.level_set_mask(rank, k - 0.5, width=1e-3))
        np.testing.assert_allclose(recovered, np.asarray(mk.gal_cut_mask(nside, cut)),
                                   atol=1e-6)


def test_level_set_mask_f_sky_is_monotone_in_t():
    """No folds along the design coordinate for an optimizer to catch on."""
    nside = 32
    rank = mk.admission_rank(
        jnp.stack([mk.gal_cut_mask(nside, c) for c in (10.0, 20.0, 30.0, 40.0)]))
    f = [float(jnp.mean(mk.level_set_mask(rank, t))) for t in np.arange(-0.5, 4.0, 0.5)]
    assert all(a >= b for a, b in itertools.pairwise(f)), f


def test_admission_rank_rejects_a_non_nested_family():
    """A non-nested stack still sums to a plausible-looking field -- so it raises.

    This is the silent-wrong-answer case: without the check the rank field is a
    well-formed array and level_set_mask returns a well-formed mask, but the
    level sets are no longer the input masks and interpolating between them
    means nothing.
    """
    nside = 32
    overlapping = jnp.stack([mk.gal_cut_mask(nside, 20.0),
                             1.0 - mk.gal_cut_mask(nside, 40.0)])
    with pytest.raises(ValueError, match="not nested"):
        mk.admission_rank(overlapping)
    # ...and is computable when the caller takes responsibility
    assert mk.admission_rank(overlapping, check=False).shape == (hp.nside2npix(nside),)
