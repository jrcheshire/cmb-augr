"""Gate for augr.spectrum_stages: the cut-sky masked-Wiener Monte-Carlo spectrum stage.

Fast tests cover the prior beaming, the single-map estimator wrapper shape, and the
MC driver's output shapes / covariance positivity / Hartlap guard at a tiny nside with
``fg_model=None`` (no PySM). The slow test is the science checkpoint: the E→B leakage
template through the *full spin-2 cleaner* must sit well below the lensing-BB floor (the
purity null), so the leaked-E cosmic variance does not dominate σ(r).

Map work needs jht (the [masking] extra) and ducc0 (the SHTs).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jht")
pytest.importorskip("ducc0")

import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.delensing import load_lensing_spectra
from augr.foregrounds import NullForegroundModel
from augr.instrument import beam_bl
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import beamed_prior, cutsky_bb_bandpower, mc_cutsky_bandpowers

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


def _run_mc(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2):
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
        workers=1,
    )


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
