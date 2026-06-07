"""Gate for augr.noise_sims.correlated_noise_maps: 1/f spectral recovery, CRN, AD.

The correlated draw adds a 1/f tilt N_ell = w_inv * (1 + (ell_knee/ell)^alpha) to the
anisotropic noise, drawn in harmonic space. These tests pin: (a) the angular power
spectrum is recovered (white limit and 1/f shape) in the uniform-hits case where the
construction is exact; (b) the sqrt(w_inv) common-random-number scaling; (c)
differentiability in w_inv and in ell_knee; (d) per-band independence. The SHT needs
ducc0 (the [compsep] extra), so the whole file skips without it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

import healpy as hp

from augr.noise_sims import correlated_noise_maps, one_over_f_factor

NSIDE = 64
LMAX = 2 * NSIDE  # conservative band limit (augr.sht.band_limit default)
NPIX = 12 * NSIDE * NSIDE
W_INV = 4.0e-4  # μK²·sr


def _uniform_hits() -> jnp.ndarray:
    return jnp.ones(NPIX)


def _anisotropic_hits(seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(1.0 + 0.8 * rng.random(NPIX))


def _target_nl(w_inv: float, knee: float, alpha: float) -> np.ndarray:
    ells = np.arange(LMAX + 1)
    factor = np.where((knee > 0) & (ells > 0), (knee / np.maximum(ells, 1.0)) ** alpha, 0.0)
    return w_inv * (1.0 + factor)


def _mc_cl(hit, w_inv, knee, alpha, *, n_real: int, seed0: int = 0) -> np.ndarray:
    """MC-averaged angular power spectrum of the (single-band) correlated noise map."""
    cls = []
    for r in range(n_real):
        m = np.asarray(
            correlated_noise_maps(
                hit,
                w_inv,
                knee,
                alpha,
                jax.random.PRNGKey(seed0 + r),
                lmax=LMAX,
                nside=NSIDE,
            )
        )[0]
        alm = hp.map2alm(m, lmax=LMAX, iter=3)
        cls.append(hp.alm2cl(alm, lmax=LMAX))
    return np.mean(cls, axis=0)


def _band_ratio(cl_mc: np.ndarray, target: np.ndarray, lo: int, hi: int) -> float:
    """Mode-weighted ratio of measured to target C_ell over [lo, hi]."""
    ells = np.arange(LMAX + 1)
    band = (ells >= lo) & (ells <= hi)
    modes = 2 * ells + 1
    return float(np.sum(modes[band] * cl_mc[band]) / np.sum(modes[band] * target[band]))


# --- one_over_f_factor convention (matches instrument.noise_nl) -------------


def test_one_over_f_factor_white_and_guards() -> None:
    ells = jnp.arange(LMAX + 1, dtype=float)
    # knee=0 -> pure white (factor 0 everywhere)
    f0 = np.asarray(one_over_f_factor(ells, jnp.array(0.0), jnp.array(1.0)))
    np.testing.assert_array_equal(f0, 0.0)
    # knee>0: (knee/ell)^alpha, with ell=0 guarded to 0
    f = np.asarray(one_over_f_factor(ells, jnp.array(80.0), jnp.array(2.0)))
    assert f[0] == 0.0  # ell=0 guard
    np.testing.assert_allclose(f[10], (80.0 / 10.0) ** 2, rtol=1e-12)
    np.testing.assert_allclose(f[80], 1.0, rtol=1e-12)


# --- spectral recovery (uniform hits = exact construction) ------------------


def test_correlated_recovers_white_spectrum() -> None:
    """knee=0 reproduces a flat N_ell = w_inv angular spectrum (the white limit)."""
    cl = _mc_cl(_uniform_hits(), W_INV, 0.0, 1.0, n_real=40)
    target = _target_nl(W_INV, 0.0, 1.0)
    np.testing.assert_allclose(_band_ratio(cl, target, 10, 110), 1.0, rtol=0.05)


def test_correlated_recovers_1f_shape() -> None:
    """knee>0 recovers the 1/f shape at low ell AND the white floor at high ell."""
    knee, alpha = 80.0, 2.0
    cl = _mc_cl(_uniform_hits(), W_INV, knee, alpha, n_real=40)
    target = _target_nl(W_INV, knee, alpha)
    # low-ell band is strongly 1/f-tilted (N_ell ~ 65 w_inv at ell=10); high-ell ~ white
    np.testing.assert_allclose(_band_ratio(cl, target, 5, 20), 1.0, rtol=0.08)
    np.testing.assert_allclose(_band_ratio(cl, target, 80, 120), 1.0, rtol=0.06)
    # sanity: the low-ell power really is well above the white floor
    assert cl[8] > 10.0 * W_INV


# --- common random numbers: sqrt(w_inv) amplitude scaling ------------------


def test_correlated_crn_sqrt_scaling() -> None:
    hit = _anisotropic_hits()
    key = jax.random.PRNGKey(7)
    a = np.asarray(correlated_noise_maps(hit, 1.0e-4, 80.0, 2.0, key, lmax=LMAX, nside=NSIDE))
    b = np.asarray(correlated_noise_maps(hit, 4.0e-4, 80.0, 2.0, key, lmax=LMAX, nside=NSIDE))
    # Same key + same knee => same unit field => map scales by sqrt(w ratio) = 2.
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-10, atol=1e-14)


# --- anisotropy: low-hit regions are noisier --------------------------------


def test_correlated_envelope_modulates_variance() -> None:
    hit = np.asarray(_anisotropic_hits(seed=1))
    m = np.asarray(
        correlated_noise_maps(
            jnp.asarray(hit), W_INV, 80.0, 2.0, jax.random.PRNGKey(2), lmax=LMAX, nside=NSIDE
        )
    )[0]
    lo = hit < np.quantile(hit, 0.25)  # fewest hits -> noisiest
    hi = hit > np.quantile(hit, 0.75)
    assert np.var(m[lo]) > np.var(m[hi])


# --- differentiability ------------------------------------------------------


def test_correlated_grad_w_inv_matches_fd() -> None:
    hit = _anisotropic_hits()
    key = jax.random.PRNGKey(3)

    def power(w_inv):
        m = correlated_noise_maps(hit, w_inv, 80.0, 2.0, key, lmax=LMAX, nside=NSIDE)
        return jnp.mean(m**2)

    w0 = 2.5e-4
    g = float(jax.grad(power)(w0))
    assert np.isfinite(g) and g > 0
    eps = 1e-6
    fd = float((power(w0 + eps) - power(w0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def test_correlated_grad_knee_matches_fd() -> None:
    hit = _anisotropic_hits()
    key = jax.random.PRNGKey(5)

    def power(knee_ell):
        m = correlated_noise_maps(hit, W_INV, knee_ell, 2.0, key, lmax=LMAX, nside=NSIDE)
        return jnp.mean(m**2)

    k0 = 80.0
    g = float(jax.grad(power)(k0))
    assert np.isfinite(g) and g > 0  # more 1/f power -> more variance
    eps = 1e-2
    fd = float((power(k0 + eps) - power(k0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=2e-3)


# --- per-band knees ---------------------------------------------------------


def test_correlated_per_band_knees() -> None:
    """Per-band knee array: band 0 white, band 1 strongly 1/f; bands independent."""
    hit = _uniform_hits()
    w_inv = jnp.array([W_INV, W_INV])
    knee = jnp.array([0.0, 120.0])
    alpha = jnp.array([1.0, 2.0])
    m = np.asarray(
        correlated_noise_maps(
            hit, w_inv, knee, alpha, jax.random.PRNGKey(9), lmax=LMAX, nside=NSIDE
        )
    )
    assert m.shape == (2, NPIX)
    cl0 = hp.alm2cl(hp.map2alm(m[0], lmax=LMAX, iter=3), lmax=LMAX)
    cl1 = hp.alm2cl(hp.map2alm(m[1], lmax=LMAX, iter=3), lmax=LMAX)
    # Band 1 has far more low-ell power than band 0 (white); high-ell comparable.
    assert cl1[6] > 20.0 * cl0[6]
    # Independent realizations (split keys), not rescaled copies.
    assert abs(np.corrcoef(m[0], m[1])[0, 1]) < 0.1
