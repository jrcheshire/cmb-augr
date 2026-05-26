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
    generate_band_sky,
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
