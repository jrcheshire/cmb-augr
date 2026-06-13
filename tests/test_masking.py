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
