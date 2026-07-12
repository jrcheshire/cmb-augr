"""Tests for the foreground-sky cache (save_sky_cache / load_sky_cache + bypass).

The cache lets a pysm3-less env (the aarch64 GPU) rebuild a cut-sky MC context from a
precomputed sky ensemble. These tests use a CMB-only ctx (fg_model=None) so they need no
pysm3 and run no forward (var_pix_ref supplied -> the setup clean is skipped): they cover
the serialization roundtrip, the make_cutsky_mc_context bypass, and its validation.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.delensing import load_lensing_spectra
from augr.foregrounds import NullForegroundModel
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import load_sky_cache, make_cutsky_mc_context, save_sky_cache

NSIDE, LMAX, F_SKY = 16, 24, 0.6
FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)


def _build_ctx(
    *, harmonic_skies=None, noise_keys=None, var_pix_ref=1.0, n_sims=3, base_seed=0
):
    """A tiny CMB-only cut-sky MC context (no pysm3; var_pix_ref supplied -> no setup clean)."""
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: LMAX + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: LMAX + 1], 0.0, None)
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=F_SKY),
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=2,
        ell_max=LMAX,
        delta_ell=8,
        ell_per_bin_below=2,
    )
    bm = jnp.asarray(sm.bin_matrix)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(LMAX + 1, dtype=float), 0.0), 0.0, None),
        bm,
        2,
    )
    return make_cutsky_mc_context(
        cleaner=nilc_cleaner(clean_e=True),
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=np.full(3, 1e-4),
        nside=NSIDE,
        lmax=LMAX,
        mask=mk.galactic_mask(NSIDE, F_SKY),
        cl_ee=cl_ee,
        cl_bb_prior_unbeamed=cl_bb,
        bin_matrix=bm,
        ell_min=2,
        true_bb_binned=true_b,
        n_sims=n_sims,
        base_seed=base_seed,
        fg_model=None,
        r_in=0.0,
        var_pix_ref=var_pix_ref,
        harmonic_skies=harmonic_skies,
        noise_keys=noise_keys,
    )


def test_sky_cache_roundtrip(tmp_path):
    ctx = _build_ctx(n_sims=3, base_seed=7)
    p = str(tmp_path / "cache.npz")
    save_sky_cache(p, ctx, fg_model="none", base_seed=7)
    cache = load_sky_cache(p)

    assert cache.n_sims == ctx.n_sims == 3
    assert cache.base_seed == 7
    assert cache.fg_model == "none"
    assert cache.freqs_ghz == ctx.harmonic_skies.freqs_ghz
    assert cache.beam_fwhm_arcmin == tuple(float(b) for b in ctx.beam_fwhm_arcmin)
    assert cache.var_pix_ref == pytest.approx(float(ctx.var_pix_ref))
    assert np.allclose(
        np.asarray(cache.harmonic_skies.cmb_b_alm),
        np.asarray(ctx.harmonic_skies.cmb_b_alm),
    )
    assert np.allclose(
        np.asarray(cache.harmonic_skies.cmb_e_alm),
        np.asarray(ctx.harmonic_skies.cmb_e_alm),
    )
    assert np.array_equal(np.asarray(cache.noise_keys), np.asarray(ctx.noise_keys))
    assert cache.harmonic_skies.fg_eb_alm is None  # CMB-only ctx


def test_make_ctx_from_cache_bypasses_generation(tmp_path):
    """A ctx rebuilt from the cache uses the cached arrays (n_sims from the cache, not the arg)."""
    ctx_a = _build_ctx(n_sims=3, base_seed=100)
    p = str(tmp_path / "c.npz")
    save_sky_cache(p, ctx_a, fg_model="none")
    cache = load_sky_cache(p)

    # n_sims=999 / base_seed=0 here are deliberately wrong -- they must be ignored in favor
    # of the cached ensemble (proving generation is bypassed).
    ctx_b = _build_ctx(
        harmonic_skies=cache.harmonic_skies,
        noise_keys=cache.noise_keys,
        var_pix_ref=cache.var_pix_ref,
        n_sims=999,
        base_seed=0,
    )
    assert ctx_b.n_sims == ctx_a.n_sims == 3
    assert np.allclose(
        np.asarray(ctx_b.harmonic_skies.cmb_b_alm),
        np.asarray(ctx_a.harmonic_skies.cmb_b_alm),
    )
    assert np.array_equal(np.asarray(ctx_b.noise_keys), np.asarray(ctx_a.noise_keys))
    assert float(ctx_b.var_pix_ref) == pytest.approx(float(ctx_a.var_pix_ref))


def test_fg_alm_roundtrip(tmp_path):
    """The foreground alm (has_fg=True, the headline path) round-trip -- no pysm3 needed."""
    ctx = _build_ctx(
        n_sims=2
    )  # CMB-only; inject a synthetic FG ensemble to exercise has_fg
    hs = ctx.harmonic_skies
    n_sims, n_alm = hs.cmb_b_alm.shape
    fake_fg = (1.0 + 2.0j) * jnp.ones((n_sims, len(FREQS), 2, n_alm))
    hs_fg = eqx.tree_at(lambda h: h.fg_eb_alm, hs, fake_fg, is_leaf=lambda x: x is None)
    ctx_fg = eqx.tree_at(lambda c: c.harmonic_skies, ctx, hs_fg)

    p = str(tmp_path / "fg.npz")
    save_sky_cache(p, ctx_fg, fg_model="d1s1")
    cache = load_sky_cache(p)
    assert cache.fg_model == "d1s1"
    assert cache.harmonic_skies.fg_eb_alm is not None
    assert np.allclose(np.asarray(cache.harmonic_skies.fg_eb_alm), np.asarray(fake_fg))


def test_cache_bypass_validation():
    ctx = _build_ctx(n_sims=3)
    hs = ctx.harmonic_skies
    with pytest.raises(ValueError, match="noise_keys must be supplied"):
        _build_ctx(harmonic_skies=hs, noise_keys=None, var_pix_ref=1.0)
    with pytest.raises(ValueError, match="var_pix_ref must be supplied"):
        _build_ctx(harmonic_skies=hs, noise_keys=ctx.noise_keys, var_pix_ref=None)
    with pytest.raises(ValueError, match="noise_keys has"):
        _build_ctx(harmonic_skies=hs, noise_keys=ctx.noise_keys[:2], var_pix_ref=1.0)
