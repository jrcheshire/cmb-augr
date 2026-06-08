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


def _sim(nside: int, lmax: int, *, seed: int = 1, r_in: float = 0.0, fg_model: str | None = "d1s1"):
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
#
# These build a realistic PySM (d1s1) sky, which downloads sky templates over the
# network on a cold cache -- so they carry `slow` (excluded from the parallel per-PR
# gate; run serially in the weekly/local suite). The network-free GNILC end-to-end
# smoke kept in the gate is test_gnilc_noise_term_vanishes_for_zero_noise below.


@pytest.mark.slow
def test_gnilc_template_positive_and_tracks_oracle() -> None:
    """At m_bias=1 the template is positive and tracks the oracle to O(1) across ℓ."""
    pytest.importorskip("pysm3")
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


@pytest.mark.slow
def test_gnilc_template_cmb_suppressed() -> None:
    """The template carries little CMB: the CMB leaking through the composed weights is a
    small fraction of the foreground template. (m_bias=1 leaks more CMB than pure AIC —
    the leakage Carones' paired depro_cmb=0 removes; deferred to v2, so this checks it is
    small relative to the FG residual rather than zero.)"""
    pytest.importorskip("pysm3")
    nside, lmax = 64, 128
    sky, total, noise = _sim(nside, lmax)
    res = build_gnilc(total, sky.cmb_qu + noise, BEAMS, lmax=lmax, nside=nside, m_bias=1)
    ell = np.arange(lmax + 1)
    band = (ell >= 20) & (ell <= 80)
    cmb_v = np.asarray(alm2cl(res.fg_residual_alm(sky.cmb_qu), lmax))[band].mean()
    fg_v = np.asarray(alm2cl(res.fg_residual_alm(sky.fg_qu), lmax))[band].mean()
    assert cmb_v / fg_v < 0.1, cmb_v / fg_v


def test_gnilc_noise_term_vanishes_for_zero_noise() -> None:
    """The Eq. 3.7 noise-debias term is exactly zero when the noise input is zero.

    Network-free GNILC end-to-end smoke kept in the per-PR gate: `fg_model=None` so no
    PySM templates are fetched, but build_gnilc still exercises the full needlet /
    common-resolution / weight / recompose wiring.
    """
    nside, lmax = 32, 64
    sky, total, noise = _sim(nside, lmax, fg_model=None)
    res = build_gnilc(total, sky.cmb_qu + noise, BEAMS, lmax=lmax, nside=nside, m_bias=1)
    zero = jnp.zeros_like(total)
    nres = np.asarray(alm2cl(res.fg_residual_alm(zero), lmax))
    np.testing.assert_allclose(nres, 0.0, atol=1e-30)


# --- differentiability (gates the eigh-degeneracy risk) --------------------


@pytest.mark.slow
def test_gnilc_template_differentiable_in_noise() -> None:
    pytest.importorskip("pysm3")
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


# --- localized (per-pixel) GNILC -------------------------------------------


def test_gnilc_estimator_batches_over_pixels() -> None:
    """The per-pixel ``_gnilc_fg_estimator`` equals the global one pixel-by-pixel — the
    batch-safety of the eigh descending-reorder and the matrix-sqrt column scaling. Also
    the regression that those fixes are identity for the 2-D global path."""
    rng = np.random.default_rng(7)
    n, k = 6, 2
    a = rng.normal(size=(n, n))
    cov_n = jnp.asarray(a @ a.T + n * np.eye(n))
    fd = rng.normal(size=(n, k))
    cov_t = cov_n + jnp.asarray(fd @ np.diag([200.0, 80.0]) @ fd.T)
    w0, m0 = _gnilc_fg_estimator(cov_t, cov_n, m_bias=1)
    npix = 5
    wb, mb = _gnilc_fg_estimator(
        jnp.broadcast_to(cov_t, (npix, n, n)), jnp.broadcast_to(cov_n, (npix, n, n)), m_bias=1
    )
    np.testing.assert_allclose(
        np.asarray(wb), np.broadcast_to(np.asarray(w0), (npix, n, n)), atol=1e-12
    )
    assert all(int(x) == int(m0) for x in mb)


def test_gnilc_per_pixel_aic_recovers_varying_dim() -> None:
    """Per-pixel AIC selects a different FG subspace dimension per pixel — the mechanism
    behind localized GNILC adapting where the foregrounds are more complex."""
    rng = np.random.default_rng(8)
    n = 6
    a = rng.normal(size=(n, n))
    cov_n = a @ a.T + n * np.eye(n)
    fd = rng.normal(size=(n, 3))
    cov_t0 = cov_n + fd[:, :1] @ np.diag([200.0]) @ fd[:, :1].T  # 1 FG mode
    cov_t1 = cov_n + fd @ np.diag([300.0, 150.0, 90.0]) @ fd.T  # 3 FG modes
    cov_t = jnp.stack([jnp.asarray(cov_t0), jnp.asarray(cov_t1)])
    cov_n_b = jnp.broadcast_to(jnp.asarray(cov_n), (2, n, n))
    _, m = _gnilc_fg_estimator(cov_t, cov_n_b, m_bias=0)
    np.testing.assert_array_equal(np.asarray(m), [1, 3])


@pytest.mark.slow
def test_gnilc_localized_smoke() -> None:
    """Localized build_gnilc returns per-pixel weights / m-maps and a finite residual.
    Network-free (fg_model=None) end-to-end wiring check. ``slow`` not for the compute
    (nside is small) but because the localized path traces ~3·J·n_act² SHT ops, so its
    first-call compilation is ~10s regardless of nside — too heavy for the parallel gate.
    The fast estimator-batching / per-pixel-AIC tests above cover the core in-gate."""
    nside, lmax = 16, 32
    sky, total, noise = _sim(nside, lmax, fg_model=None)
    res = build_gnilc(
        total,
        sky.cmb_qu + noise,
        BEAMS,
        lmax=lmax,
        nside=nside,
        localization_fwhm_arcmin=600.0,
        m_bias=1,
    )
    npix = 12 * nside * nside
    assert res.residual_weights.shape == (res.needlet_bands.shape[0], len(BEAMS), npix)
    assert np.asarray(res.m_per_band[0]).shape == (npix,)
    assert np.all(np.isfinite(np.asarray(res.fg_residual_alm(sky.fg_qu))))


@pytest.mark.slow
def test_gnilc_localized_reduces_near_plane_residual() -> None:
    """Region-split by galactic |b|: localized GNILC's per-pixel NILC weights leave less FG
    residual in the high-β-gradient near-plane region than global weights, and the
    relative improvement is larger near the plane than at high latitude (where β varies
    little). The whole point of localization."""
    pytest.importorskip("pysm3")
    from augr.sht import synthesis

    nside, lmax = 64, 128
    sky, total, noise = _sim(nside, lmax, fg_model="d10s5", seed=1)
    res_glob = build_gnilc(total, sky.cmb_qu + noise, BEAMS, lmax=lmax, nside=nside, m_bias=1)
    res_loc = build_gnilc(
        total,
        sky.cmb_qu + noise,
        BEAMS,
        lmax=lmax,
        nside=nside,
        localization_fwhm_arcmin=600.0,
        m_bias=1,
    )

    def resid_map(res):
        # oracle FG residual through the (localized) NILC CMB weights, as a real B-field map
        return np.asarray(synthesis(res.nilc_clean_alm(sky.fg_qu)[None, :], 0, lmax, nside)[0])

    _, lat = hp.pix2ang(nside, np.arange(12 * nside * nside), lonlat=True)
    near, far = np.abs(lat) < 20.0, np.abs(lat) > 40.0
    g, lo = resid_map(res_glob), resid_map(res_loc)
    near_glob, near_loc = g[near].var(), lo[near].var()
    far_glob, far_loc = g[far].var(), lo[far].var()
    assert near_loc < near_glob, (near_loc, near_glob)  # localized cleans the plane better
    assert (near_loc / near_glob) < (far_loc / far_glob)  # ... more so than at high |b|


# No localized-GNILC grad gate, by design: the per-pixel top-m eigenvector projector is
# non-smooth at eigenvalue crossings (faint-FG pixels sit near them), so the localized
# gradient is unreliable — NaN with no FG, tens-of-% off FD with FG. The localized path is
# forward-only (a diagnostic / realism cross-check); the *global* GNILC
# (test_gnilc_template_differentiable_in_noise) is the differentiable path for optimization.
# cMILC's localized path is solve-based (no eigenvector projector) and is unaffected.
