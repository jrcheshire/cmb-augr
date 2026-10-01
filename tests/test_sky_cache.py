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
from augr.sht import alm_size
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import (
    load_fg_cache,
    load_sky_cache,
    make_cutsky_mc_context,
    save_fg_cache,
    save_sky_cache,
)

NSIDE, LMAX, F_SKY = 16, 24, 0.6
FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)


def _build_ctx(
    *, harmonic_skies=None, noise_keys=None, var_pix_ref=1.0, n_sims=3, base_seed=0,
    split_lensing=False, estimator="master", fg_model=None, fg_eb_alm=None,
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
        fg_model=fg_model,
        r_in=0.0,
        var_pix_ref=var_pix_ref,
        harmonic_skies=harmonic_skies,
        noise_keys=noise_keys,
        fg_eb_alm=fg_eb_alm,
        split_lensing=split_lensing,
        estimator=estimator,
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

    # A MASTER ctx carries no var_pix_ref at all; the absence must survive the
    # round trip as None rather than resurfacing as 0.0 or NaN.
    ctx_none = _build_ctx(n_sims=3, base_seed=7, var_pix_ref=None, estimator="master")
    assert ctx_none.var_pix_ref is None
    p_none = str(tmp_path / "cache_no_vpr.npz")
    save_sky_cache(p_none, ctx_none, fg_model="none", base_seed=7)
    assert load_sky_cache(p_none).var_pix_ref is None


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
    with pytest.raises(ValueError, match="noise_keys has"):
        _build_ctx(harmonic_skies=hs, noise_keys=ctx.noise_keys[:2], var_pix_ref=1.0)
    # var_pix_ref is required only where inv_noise is: the masked-Wiener path.
    with pytest.raises(ValueError, match="var_pix_ref must be supplied"):
        _build_ctx(
            harmonic_skies=hs, noise_keys=ctx.noise_keys, var_pix_ref=None,
            estimator="wiener",
        )
    # ...and MASTER builds happily without one, since it never reads inv_noise.
    master = _build_ctx(
        harmonic_skies=hs, noise_keys=ctx.noise_keys, var_pix_ref=None,
        estimator="master",
    )
    assert master.var_pix_ref is None
    assert master.inv_noise is None


def test_sky_cache_carries_the_lensing_split(tmp_path):
    """A split ensemble survives the round trip, and rebuilds into a delens-capable ctx.

    The production EIG runs load their skies from this cache on pysm3-less nodes, so
    a cache that dropped ``cmb_b_lens_alm`` would strand them at A_lens = 1 -- the
    exact bias the delensing seam exists to remove.
    """
    ctx = _build_ctx(n_sims=3, base_seed=11, split_lensing=True)
    p = str(tmp_path / "split.npz")
    save_sky_cache(p, ctx, fg_model="none", base_seed=11)
    cache = load_sky_cache(p)

    assert cache.harmonic_skies.cmb_b_lens_alm is not None
    np.testing.assert_array_equal(
        np.asarray(cache.harmonic_skies.cmb_b_lens_alm),
        np.asarray(ctx.harmonic_skies.cmb_b_lens_alm),
    )
    rebuilt = _build_ctx(
        harmonic_skies=cache.harmonic_skies,
        noise_keys=cache.noise_keys,
        var_pix_ref=cache.var_pix_ref,
        split_lensing=True,
    )
    assert rebuilt.cl_bb_lens_ref is not None


def test_unsplit_cache_refuses_a_delensing_context(tmp_path):
    """Asking for split_lensing against a cache without it raises, rather than
    silently rebuilding an ensemble that cannot delens."""
    ctx = _build_ctx(n_sims=3, base_seed=12)
    p = str(tmp_path / "plain.npz")
    save_sky_cache(p, ctx, fg_model="none", base_seed=12)
    cache = load_sky_cache(p)
    assert cache.harmonic_skies.cmb_b_lens_alm is None
    with pytest.raises(ValueError, match="split_lensing=True"):
        _build_ctx(
            harmonic_skies=cache.harmonic_skies,
            noise_keys=cache.noise_keys,
            var_pix_ref=cache.var_pix_ref,
            split_lensing=True,
        )


# --- static foreground generated / cached once ------------------------------------------


def _fake_fg():
    """A deterministic stand-in for one static PySM sky, ``(n_band, 2, n_alm)``."""
    rng = np.random.default_rng(3)
    shape = (len(FREQS), 2, alm_size(LMAX))
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


@pytest.fixture
def fake_pysm(monkeypatch):
    """Replace PySM generation with a call-counting fake; ``static`` sets the preset type.

    Patches the per-band generator in ``compsep_sims`` (reached by both the per-sim
    ``harmonic_sky`` path and the real ``static_fg_eb_alm``), so the count is the
    number of PySM runs whichever path builds the ensemble.
    """
    import augr.compsep_sims as cs
    import augr.spectrum_stages as ss

    state = {"calls": 0, "static": True}
    fg = _fake_fg()

    def gen(freqs_ghz, fg_model, lmax, nside, *, fg_seed=0, bandpasses=None):
        state["calls"] += 1
        return fg

    monkeypatch.setattr(cs, "_fg_eb_alm", gen)
    monkeypatch.setattr(cs, "fg_model_is_static", lambda m: state["static"])
    monkeypatch.setattr(ss, "fg_model_is_static", lambda m: state["static"])
    state["fg"] = fg
    return state


def _assert_same_ensemble(a, b):
    ha, hb = a.harmonic_skies, b.harmonic_skies
    for name in ("cmb_b_alm", "cmb_e_alm", "fg_eb_alm"):
        np.testing.assert_array_equal(np.asarray(getattr(ha, name)), np.asarray(getattr(hb, name)))
    np.testing.assert_array_equal(np.asarray(a.noise_keys), np.asarray(b.noise_keys))


def test_static_fg_is_generated_once_and_matches_per_sim_generation(fake_pysm):
    """A static preset runs PySM once, not n_sims times, and builds the same ensemble.

    The stochastic leg is the anti-vacuity half: it must still generate per sim, or a
    builder that always shared would pass the first leg while replacing a genuinely
    varying foreground with one realization.
    """
    n_sims = 4
    fake_pysm["static"] = False
    per_sim = _build_ctx(n_sims=n_sims, base_seed=5, fg_model="d1s1")
    assert fake_pysm["calls"] == n_sims

    fake_pysm["calls"], fake_pysm["static"] = 0, True
    once = _build_ctx(n_sims=n_sims, base_seed=5, fg_model="d1s1")
    assert fake_pysm["calls"] == 1
    assert once.harmonic_skies.fg_eb_alm.ndim == 3

    fake_pysm["calls"] = 0
    supplied = _build_ctx(n_sims=n_sims, base_seed=5, fg_eb_alm=fake_pysm["fg"])
    assert fake_pysm["calls"] == 0

    # per_sim was collapsed by share_fg, so all three carry the same rank-3 foreground.
    _assert_same_ensemble(per_sim, once)
    _assert_same_ensemble(per_sim, supplied)


def test_fg_cache_roundtrip_rebuilds_the_generated_ensemble(tmp_path, fake_pysm):
    """Generate -> save_fg_cache -> load -> fg_eb_alm= reproduces the ensemble bitwise."""
    generated = _build_ctx(n_sims=3, base_seed=9, fg_model="d10s5")
    p = str(tmp_path / "fg_once.npz")
    save_fg_cache(
        p, generated.harmonic_skies.fg_eb_alm, fg_model="d10s5",
        freqs_ghz=FREQS, nside=NSIDE, lmax=LMAX,
    )
    cache = load_fg_cache(p)
    assert (cache.fg_model, cache.freqs_ghz, cache.nside, cache.lmax) == ("d10s5", FREQS, NSIDE, LMAX)

    fake_pysm["calls"] = 0
    rebuilt = _build_ctx(n_sims=3, base_seed=9, fg_eb_alm=cache.fg_eb_alm)
    assert fake_pysm["calls"] == 0
    _assert_same_ensemble(generated, rebuilt)


def test_fg_eb_alm_validation(tmp_path):
    fg = _fake_fg()
    ctx = _build_ctx(n_sims=3)
    with pytest.raises(ValueError, match="not both"):
        _build_ctx(harmonic_skies=ctx.harmonic_skies, noise_keys=ctx.noise_keys, fg_eb_alm=fg)
    with pytest.raises(ValueError, match="fg_eb_alm has shape"):
        _build_ctx(n_sims=3, fg_eb_alm=fg[:, :, :-1])
    with pytest.raises(ValueError, match="fg_eb_alm has shape"):
        save_fg_cache(
            str(tmp_path / "bad.npz"), fg[:2], fg_model="d1s1",
            freqs_ghz=FREQS, nside=NSIDE, lmax=LMAX,
        )
