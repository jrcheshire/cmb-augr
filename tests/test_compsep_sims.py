"""Stage-2 gate for augr.compsep_sims: per-band map calibration + differentiability.

The calibration gates (noise level, CMB+beam amplitude, additivity,
differentiability in w_inv) need only healpy + ducc0 and stay in the fast suite.
The PySM foreground path is gated behind importorskip + the slow marker.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

import healpy as hp  # noqa: E402

from augr.compsep_sims import (  # noqa: E402
    BandSky,
    assemble_band_maps,
    beam_harmonic_sky,
    generate_band_sky,
    harmonic_sky,
)
from augr.instrument import beam_bl  # noqa: E402
from augr.spectra import CMBSpectra  # noqa: E402

NSIDE = 64
LMAX = 128
NPIX = 12 * NSIDE * NSIDE


def _bb(qu: np.ndarray, lmax: int) -> np.ndarray:
    """anafast BB of a Q/U map (I set to zero), full-sky."""
    iqu = np.zeros((3, qu.shape[-1]))
    iqu[1:] = qu
    cls = hp.anafast(iqu, lmax=lmax, pol=True)
    return cls[2]  # [TT, EE, BB, TE, EB, TB] -> BB


def _zero_sky(freqs, beams) -> BandSky:
    n = len(freqs)
    z = jnp.zeros((n, 2, NPIX))
    return BandSky(
        freqs_ghz=tuple(freqs),
        beam_fwhm_arcmin=tuple(beams),
        nside=NSIDE,
        lmax=LMAX,
        r_in=0.0,
        cmb_qu=z,
        fg_qu=z,
    )


# --- noise level matches the polarization white-noise spec -----------------


def test_noise_only_bb_matches_w_inv() -> None:
    """Q/U white noise at power w_inv gives a flat N_ℓ^{BB} ≈ w_inv."""
    w_inv = jnp.array([1.0e-4, 4.0e-4])
    sky = _zero_sky([30.0, 100.0], [30.0, 18.0])
    hit = jnp.ones(NPIX)
    maps = np.asarray(assemble_band_maps(sky, w_inv, hit, noise_key=jax.random.PRNGKey(0)))
    ells = np.arange(LMAX + 1)
    band = (ells >= 10) & (ells <= 120)
    for i, wv in enumerate([1.0e-4, 4.0e-4]):
        bb = _bb(maps[i], LMAX)
        np.testing.assert_allclose(np.mean(bb[band]), wv, rtol=0.05)


# --- CMB amplitude + beam ---------------------------------------------------


def test_cmb_band_bb_matches_theory() -> None:
    """Beam-deconvolved CMB-only BB matches C_ℓ^{BB}(r) within cosmic variance."""
    spectra = CMBSpectra()
    fwhm = 30.0
    r_in = 0.05
    sky = generate_band_sky(
        (100.0,),
        (fwhm,),
        spectra=spectra,
        r_in=r_in,
        nside=NSIDE,
        lmax=LMAX,
        fg_model=None,
        cmb_seed=0,
    )
    bb_meas = _bb(np.asarray(sky.cmb_qu[0]), LMAX)
    ells = np.arange(LMAX + 1)
    bl2 = np.asarray(beam_bl(jnp.asarray(ells, dtype=float), fwhm)) ** 2
    theory = np.asarray(spectra.cl_bb(jnp.asarray(ells, dtype=float), r_in))

    band = (ells >= 40) & (ells <= 110)
    deconv = (bb_meas / bl2)[band]
    # Summed bandpower integrates over ~1e4 modes -> few-% cosmic-variance scatter.
    np.testing.assert_allclose(deconv.sum(), theory[band].sum(), rtol=0.15)


# --- additivity of the component maps --------------------------------------


def test_total_bb_is_sum_of_components() -> None:
    """anafast(total) ≈ anafast(CMB) + anafast(noise) (cross terms subdominant)."""
    spectra = CMBSpectra()
    sky = generate_band_sky(
        (100.0,),
        (30.0,),
        spectra=spectra,
        r_in=0.05,
        nside=NSIDE,
        lmax=LMAX,
        fg_model=None,
        cmb_seed=1,
    )
    w_inv = jnp.array([1.0e-3])
    hit = jnp.ones(NPIX)
    total = np.asarray(assemble_band_maps(sky, w_inv, hit, noise_key=jax.random.PRNGKey(2)))[0]
    cmb_only = np.asarray(sky.cmb_qu[0])
    noise_only = total - cmb_only

    ells = np.arange(LMAX + 1)
    band = (ells >= 20) & (ells <= 120)
    bb_total = _bb(total, LMAX)[band].sum()
    bb_parts = (_bb(cmb_only, LMAX) + _bb(noise_only, LMAX))[band].sum()
    np.testing.assert_allclose(bb_total, bb_parts, rtol=0.10)


# --- differentiability in w_inv (the allocation lever) ---------------------


def test_assemble_grad_wrt_w_inv_matches_fd() -> None:
    sky = _zero_sky([30.0, 100.0], [30.0, 18.0])
    hit = jnp.asarray(1.0 + 0.5 * np.random.default_rng(0).random(NPIX))
    key = jax.random.PRNGKey(5)
    w0 = jnp.array([2.0e-4, 5.0e-4])

    def power(scale):
        maps = assemble_band_maps(sky, scale * w0, hit, noise_key=key)
        return jnp.mean(maps**2)

    g = float(jax.grad(power)(1.0))
    assert np.isfinite(g) and g > 0
    eps = 1e-4
    fd = float((power(1.0 + eps) - power(1.0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-5)


# --- PySM foreground path (heavy; gated) -----------------------------------


@pytest.mark.slow
def test_generate_band_sky_with_pysm_foregrounds() -> None:
    pytest.importorskip("pysm3")
    spectra = CMBSpectra()
    nside, lmax = 32, 64
    freqs = (30.0, 100.0)
    beams = (40.0, 12.0)
    sky = generate_band_sky(
        freqs,
        beams,
        spectra=spectra,
        r_in=0.0,
        nside=nside,
        lmax=lmax,
        fg_model="d1s1",
        cmb_seed=0,
    )
    assert sky.fg_qu.shape == (2, 2, 12 * nside * nside)
    fg = np.asarray(sky.fg_qu)
    assert np.all(np.isfinite(fg))
    assert not np.allclose(fg, 0.0)
    # Synchrotron makes the 30 GHz foreground far brighter than the 100 GHz band.
    bb30 = _bb(fg[0], lmax)
    bb100 = _bb(fg[1], lmax)
    ells = np.arange(lmax + 1)
    band = (ells >= 10) & (ells <= 50)
    assert bb30[band].sum() > bb100[band].sum()


# --- foreground seed reproducibility (CRN for the aperture sweep) -----------
#
# The load-bearing logic (which PySM component gets an explicit `seeds`, and that
# the seed tracks `fg_seed`) is tested network-free against the preset dicts. The
# actual map-level reproducibility is a slow test against an s6-only Sky -- only
# the small synch alm/cl realization templates download, not the heavy GNILC dust
# maps the full d10s6 path would pull.


def test_seeded_component_config_seeds_only_realizations() -> None:
    """Only stochastic *Realization components get a `seeds` field injected."""
    pytest.importorskip("pysm3")
    from augr.compsep_sims import _seeded_component_config

    cfg = _seeded_component_config("d10s6", fg_seed=0)
    assert "seeds" in cfg["s6"]  # PowerLawRealization -> seeded for reproducibility
    assert "seeds" not in cfg["d10"]  # ModifiedBlackBody is a fixed template

    # Fixed-template models have no stochastic component -> nothing is seeded.
    for model in ("d1s1", "d10s5"):
        cfg_fixed = _seeded_component_config(model, fg_seed=0)
        assert all("seeds" not in c for c in cfg_fixed.values())


def test_seeded_component_config_seed_varies_with_fg_seed() -> None:
    pytest.importorskip("pysm3")
    from augr.compsep_sims import _seeded_component_config

    s0 = _seeded_component_config("d10s6", fg_seed=0)["s6"]["seeds"]
    s1 = _seeded_component_config("d10s6", fg_seed=1)["s6"]["seeds"]
    assert s0 != s1


def test_seeded_component_config_does_not_mutate_presets() -> None:
    """Deep-copy guard: the global PRESET_MODELS dict is never mutated."""
    pytest.importorskip("pysm3")
    from pysm3.sky import PRESET_MODELS

    from augr.compsep_sims import _seeded_component_config

    assert "seeds" not in PRESET_MODELS["s6"]
    _seeded_component_config("d10s6", fg_seed=0)
    assert "seeds" not in PRESET_MODELS["s6"]


@pytest.mark.slow
def test_s6_realization_reproducible_and_seed_varying() -> None:
    """s6 small scales are bit-identical at a fixed fg_seed and differ across seeds.

    This is the core of the CRN fix: with the preset default (seeds=None) s6
    reseeds numpy from entropy at construction, so the realization is neither
    reproducible nor held fixed across apertures. The injected seeds make it both.
    """
    pysm3 = pytest.importorskip("pysm3")
    import pysm3.units as u

    from augr.compsep_sims import _seeded_component_config

    nside = 16

    def synch_emission(seed: int) -> np.ndarray:
        # s6-only Sky -> downloads just the synch realization templates, no dust.
        cfg = {"s6": _seeded_component_config("d10s6", seed)["s6"]}
        sky = pysm3.Sky(nside=nside, component_config=cfg)
        return np.asarray(sky.get_emission(23.0 * u.GHz).value)

    a0 = synch_emission(0)
    a0_again = synch_emission(0)
    a1 = synch_emission(1)

    assert np.all(np.isfinite(a0))
    np.testing.assert_array_equal(a0, a0_again)  # same seed -> bit-identical
    assert not np.allclose(a0, a1)  # different seed -> different realization


# --- harmonic-sky reuse across apertures (the aperture-sweep cost fix) ------
#
# harmonic_sky() builds the aperture-independent CMB + FG harmonic sky once;
# beam_harmonic_sky() applies the per-aperture beam. The aperture sweep relies on
# beaming one shared harmonic sky giving bit-identical results to regenerating the
# sky from scratch at each aperture.


def test_beam_harmonic_sky_matches_generate_band_sky_cmb_only() -> None:
    """Beaming a shared harmonic sky == fresh generate_band_sky, at two apertures."""
    spectra = CMBSpectra()
    freqs = (30.0, 100.0)
    hsky = harmonic_sky(
        freqs, spectra=spectra, r_in=0.01, nside=NSIDE, lmax=LMAX, fg_model=None, cmb_seed=7
    )
    for beams in [(40.0, 12.0), (20.0, 8.0)]:
        reused = beam_harmonic_sky(hsky, beams)
        fresh = generate_band_sky(
            freqs,
            beams,
            spectra=spectra,
            r_in=0.01,
            nside=NSIDE,
            lmax=LMAX,
            fg_model=None,
            cmb_seed=7,
        )
        np.testing.assert_array_equal(np.asarray(reused.cmb_qu), np.asarray(fresh.cmb_qu))
        np.testing.assert_array_equal(np.asarray(reused.fg_qu), np.asarray(fresh.fg_qu))
        assert np.allclose(np.asarray(reused.fg_qu), 0.0)  # CMB-only -> no foreground


def test_beam_harmonic_sky_rejects_wrong_band_count() -> None:
    spectra = CMBSpectra()
    hsky = harmonic_sky(
        (30.0, 100.0), spectra=spectra, r_in=0.0, nside=NSIDE, lmax=LMAX, fg_model=None
    )
    with pytest.raises(ValueError, match="bands"):
        beam_harmonic_sky(hsky, (40.0, 12.0, 5.0))


@pytest.mark.slow
def test_beam_harmonic_sky_matches_generate_band_sky_with_fg() -> None:
    """Foreground path too: shared harmonic sky beamed == fresh, at two apertures."""
    pytest.importorskip("pysm3")
    spectra = CMBSpectra()
    nside, lmax = 32, 64
    freqs = (30.0, 100.0)
    hsky = harmonic_sky(
        freqs, spectra=spectra, r_in=0.0, nside=nside, lmax=lmax, fg_model="d1s1", cmb_seed=3
    )
    for beams in [(40.0, 12.0), (25.0, 9.0)]:
        reused = beam_harmonic_sky(hsky, beams)
        fresh = generate_band_sky(
            freqs,
            beams,
            spectra=spectra,
            r_in=0.0,
            nside=nside,
            lmax=lmax,
            fg_model="d1s1",
            cmb_seed=3,
        )
        np.testing.assert_array_equal(np.asarray(reused.fg_qu), np.asarray(fresh.fg_qu))
        np.testing.assert_array_equal(np.asarray(reused.cmb_qu), np.asarray(fresh.cmb_qu))
