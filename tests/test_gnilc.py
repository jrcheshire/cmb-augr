"""Gate for augr.gnilc: GNILC FG-estimator algebra + a reasonable residual template.

The acceptance bar (per the design discussion) is that the GNILC residual template is
**reasonable on its own** — positive, foreground-shaped, CMB-suppressed, correctly
noise-debiased, differentiable — and tracks the augr oracle (true FG through the NILC
weights) to O(1) across ℓ at the Carones m_bias=1. A tight tolerance / BROOM head-to-head
is deliberately not gated. Map-based tests need ducc0 (the [compsep] extra).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

import healpy as hp

from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.gnilc import _gnilc_fg_estimator, alm2cl, build_gnilc, gnilc_residual_template
from augr.spectra import CMBSpectra

FREQS = (30.0, 95.0, 150.0, 353.0)
BEAMS = (60.0, 40.0, 30.0, 25.0)
W_INV = jnp.array([5.0e-4, 8.0e-5, 8.0e-5, 3.0e-4])


def _sim(nside: int, lmax: int, *, seed: int = 1, r_in: float = 0.0, fg_model: str = "d1s1"):
    """Beamed sky + total/noise component map sets for a small multi-band sim."""
    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=r_in,
        nside=nside,
        lmax=lmax,
        fg_model=fg_model,
        cmb_seed=seed,
    )
    hit = jnp.ones(12 * nside * nside)
    total = assemble_band_maps(sky, W_INV, hit, noise_key=jax.random.PRNGKey(seed))
    noise = total - sky.cmb_qu - sky.fg_qu
    return sky, total, noise


def _band_means(cl: np.ndarray, edges) -> list[float]:
    ell = np.arange(cl.shape[-1])
    return [float(cl[(ell >= lo) & (ell < hi)].mean()) for lo, hi in edges]


# --- alm2cl normalization (no SHT, but ducc-gated with the file) ------------


def test_alm2cl_matches_healpy() -> None:
    lmax = 40
    rng = np.random.default_rng(1)
    cl_in = np.abs(rng.normal(size=lmax + 1)) + 0.1
    alm = hp.synalm(cl_in, lmax=lmax, new=True)
    mine = np.asarray(alm2cl(jnp.asarray(alm), lmax))
    ref = hp.alm2cl(alm, lmax=lmax)
    np.testing.assert_allclose(mine, ref, rtol=1e-10, atol=1e-12)


# --- GNILC FG-estimator algebra (pure linear algebra, no maps) -------------


def test_gnilc_estimator_recovers_fg_dim_and_projects() -> None:
    """AIC recovers the injected FG dimension; W is the C_n^{1/2}-projector onto it."""
    rng = np.random.default_rng(0)
    n, k = 5, 2
    a = rng.normal(size=(n, n))
    cov_n = a @ a.T + n * np.eye(n)  # random PD nuisance (CMB+noise)
    fdirs = rng.normal(size=(n, k))
    cov_t = cov_n + fdirs @ np.diag([200.0, 80.0]) @ fdirs.T  # + k strong FG modes

    w, m = _gnilc_fg_estimator(jnp.asarray(cov_t), jnp.asarray(cov_n), m_bias=0)
    w = np.asarray(w)
    assert int(m) == k  # AIC selects exactly the injected FG dimension
    np.testing.assert_allclose(w @ w, w, atol=1e-8)  # idempotent (oblique projector)
    # range(W) = span(fdirs): W passes any FG-subspace vector unchanged
    f = fdirs[:, 0]
    np.testing.assert_allclose(w @ f, f, rtol=1e-6, atol=1e-8)


def test_gnilc_estimator_m_bias_shifts_dimension() -> None:
    rng = np.random.default_rng(3)
    n = 5
    a = rng.normal(size=(n, n))
    cov_n = a @ a.T + n * np.eye(n)
    fdirs = rng.normal(size=(n, 2))
    cov_t = cov_n + fdirs @ np.diag([200.0, 80.0]) @ fdirs.T
    _, m0 = _gnilc_fg_estimator(jnp.asarray(cov_t), jnp.asarray(cov_n), m_bias=0)
    _, m1 = _gnilc_fg_estimator(jnp.asarray(cov_t), jnp.asarray(cov_n), m_bias=1)
    assert int(m1) == int(m0) + 1


# --- residual template: reasonable on its own ------------------------------


def test_gnilc_template_positive_and_tracks_oracle() -> None:
    """At m_bias=1 the template is positive and tracks the oracle to O(1) across ℓ."""
    nside, lmax = 64, 128
    sky, total, noise = _sim(nside, lmax)
    ells, cl, res = gnilc_residual_template(
        total, sky.cmb_qu, noise, BEAMS, lmax=lmax, nside=nside, m_bias=1, return_result=True
    )
    cl = np.asarray(cl)
    ell = np.asarray(ells)

    # oracle: true FG through the NILC weights, beam-deconvolved like the template
    from augr.instrument import beam_bl

    bl2 = np.maximum(np.asarray(beam_bl(ell.astype(float), res.common_fwhm_arcmin)) ** 2, 1e-8)
    oracle = np.asarray(alm2cl(res.nilc_clean_alm(sky.fg_qu), lmax)) / bl2

    edges = [(4, 10), (10, 20), (20, 40), (40, 70), (70, 110)]
    cl_bands = np.array(_band_means(cl, edges))
    oracle_bands = np.array(_band_means(oracle, edges))
    # Band-averaged positive (the Eq. 3.7-debiased single-realization C_ℓ can dip
    # negative at isolated ℓ from MC noise in the subtraction).
    assert np.all(cl_bands > 0), cl_bands
    ratios = cl_bands / oracle_bands
    # Loose, shape-level agreement (amplitude is absorbed by A_res); the m_bias=0
    # template fails this (ratios swing ~0.02-0.65) — see the module docstring.
    assert np.all((ratios > 0.4) & (ratios < 2.5)), ratios


def test_gnilc_template_cmb_suppressed() -> None:
    """The template carries little CMB: the CMB leaking through the composed weights is a
    small fraction of the foreground template. (m_bias=1 leaks more CMB than pure AIC —
    the leakage Carones' paired depro_cmb=0 removes; deferred to v2, so this checks it is
    small relative to the FG residual rather than zero.)"""
    nside, lmax = 64, 128
    sky, total, noise = _sim(nside, lmax)
    res = build_gnilc(total, sky.cmb_qu + noise, BEAMS, lmax=lmax, nside=nside, m_bias=1)
    ell = np.arange(lmax + 1)
    band = (ell >= 20) & (ell <= 80)
    cmb_v = np.asarray(alm2cl(res.fg_residual_alm(sky.cmb_qu), lmax))[band].mean()
    fg_v = np.asarray(alm2cl(res.fg_residual_alm(sky.fg_qu), lmax))[band].mean()
    assert cmb_v / fg_v < 0.1, cmb_v / fg_v


def test_gnilc_noise_term_vanishes_for_zero_noise() -> None:
    """The Eq. 3.7 noise-debias term is exactly zero when the noise input is zero."""
    nside, lmax = 32, 64
    sky, total, noise = _sim(nside, lmax)
    res = build_gnilc(total, sky.cmb_qu + noise, BEAMS, lmax=lmax, nside=nside, m_bias=1)
    zero = jnp.zeros_like(total)
    nres = np.asarray(alm2cl(res.fg_residual_alm(zero), lmax))
    np.testing.assert_allclose(nres, 0.0, atol=1e-30)


# --- differentiability (gates the eigh-degeneracy risk) --------------------


def test_gnilc_template_differentiable_in_noise() -> None:
    nside, lmax = 32, 64
    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=0.0,
        nside=nside,
        lmax=lmax,
        fg_model="d1s1",
        cmb_seed=2,
    )
    hit = jnp.ones(12 * nside * nside)
    key = jax.random.PRNGKey(2)
    ell = jnp.arange(lmax + 1)
    band = (ell >= 4) & (ell <= 50)

    def power(scale):
        total = assemble_band_maps(sky, scale * W_INV, hit, noise_key=key)
        noise = total - sky.cmb_qu - sky.fg_qu
        _, cl = gnilc_residual_template(
            total, sky.cmb_qu, noise, BEAMS, lmax=lmax, nside=nside, m_bias=1
        )
        return jnp.sum(jnp.where(band, cl, 0.0))

    g = float(jax.grad(power)(1.0))
    assert np.isfinite(g)
    eps = 1e-3
    fd = float((power(1.0 + eps) - power(1.0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=2e-2, atol=1e-12)
