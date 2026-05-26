"""Stage-1 gate for augr.noise_sims: normalization, CRN scaling, differentiability."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.noise_sims import hit_weight, noise_map, noise_maps

NSIDE = 64
NPIX = 12 * NSIDE * NSIDE
OMEGA_PIX = 4.0 * np.pi / NPIX


def _anisotropic_hits(seed: int = 0) -> np.ndarray:
    """Smooth-ish positive hit map with real spatial variation."""
    rng = np.random.default_rng(seed)
    return (1.0 + 0.8 * rng.random(NPIX)).astype(np.float64)


# --- hit_weight normalization (deterministic) ------------------------------


def test_hit_weight_mean_square_is_one() -> None:
    H = _anisotropic_hits()
    w = np.asarray(hit_weight(jnp.asarray(H)))
    surveyed = H > 0
    np.testing.assert_allclose(np.mean(w[surveyed] ** 2), 1.0, rtol=1e-12)


def test_hit_weight_uniform_is_unity() -> None:
    H = np.full(NPIX, 3.7)
    w = np.asarray(hit_weight(jnp.asarray(H)))
    np.testing.assert_allclose(w, 1.0, rtol=1e-12)


def test_hit_weight_zeros_are_masked() -> None:
    H = _anisotropic_hits()
    H[:100] = 0.0
    w = np.asarray(hit_weight(jnp.asarray(H)))
    assert np.all(w[:100] == 0.0)
    np.testing.assert_allclose(np.mean(w[100:] ** 2), 1.0, rtol=1e-12)


# --- noise level matches spec (MC) -----------------------------------------


def test_noise_map_sky_avg_variance_matches_spec() -> None:
    H = _anisotropic_hits()
    w_inv = 4.0e-4  # μK²·sr
    nm = np.asarray(noise_map(jnp.asarray(H), w_inv, jax.random.PRNGKey(0)))
    # surveyed-sky-averaged per-pixel variance ≈ w_inv / Ω_pix (MC, ~sqrt(2/Npix))
    np.testing.assert_allclose(np.mean(nm**2), w_inv / OMEGA_PIX, rtol=0.03)


# --- common random numbers: sqrt(w_inv) amplitude scaling ------------------


def test_noise_map_crn_sqrt_scaling() -> None:
    H = _anisotropic_hits()
    key = jax.random.PRNGKey(7)
    a = np.asarray(noise_map(jnp.asarray(H), 1.0e-4, key))
    b = np.asarray(noise_map(jnp.asarray(H), 4.0e-4, key))
    # Same key ⇒ same z ⇒ map scales by sqrt(w_inv ratio) = 2 pixelwise.
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-12, atol=1e-15)


# --- differentiability in w_inv --------------------------------------------


def test_noise_map_grad_matches_fd() -> None:
    H = jnp.asarray(_anisotropic_hits())
    key = jax.random.PRNGKey(3)

    def power(w_inv):
        nm = noise_map(H, w_inv, key)
        return jnp.mean(nm**2)

    w0 = 2.5e-4
    g = float(jax.grad(power)(w0))
    assert np.isfinite(g) and g > 0
    eps = 1e-6
    fd = float((power(w0 + eps) - power(w0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-5)


# --- multi-band ------------------------------------------------------------


def test_noise_maps_per_band_independent_and_scaled() -> None:
    H = jnp.asarray(_anisotropic_hits())
    w_inv = jnp.array([1.0e-4, 4.0e-4, 9.0e-4])
    nm = np.asarray(noise_maps(H, w_inv, jax.random.PRNGKey(11)))
    assert nm.shape == (3, NPIX)
    # Each band's sky-averaged variance matches its own spec.
    for i, wv in enumerate([1.0e-4, 4.0e-4, 9.0e-4]):
        np.testing.assert_allclose(np.mean(nm[i] ** 2), wv / OMEGA_PIX, rtol=0.03)
    # Bands are independent realizations (not just rescaled copies).
    assert abs(np.corrcoef(nm[0], nm[1])[0, 1]) < 0.1
