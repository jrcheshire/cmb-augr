"""noise_sims.py — anisotropic white-noise map synthesis for map-based comp-sep.

Draws per-pixel white noise realizations whose spatial pattern follows the inverse
hit count (more hits → lower variance) and whose surveyed-sky-averaged power matches
the white-noise spec ``w_inv`` [μK²·sr] from :func:`augr.instrument.white_noise_power`.

Differentiable in ``w_inv`` (hence ``n_det``, hence focal-plane allocation) under
**common random numbers**: the design enters only through the ``sqrt(w_inv)``
amplitude, so a fixed unit realization ``z`` (fixed ``key``) and a fixed hit map can
be reused across allocations to give a low-variance pathwise gradient
``d(noise)/d(w_inv) = noise / (2 w_inv)``.

The per-band **beam** dependence of the noise is *not* applied here — it enters at the
NILC common-resolution step (:mod:`augr.nilc`), where deconvolving each band's beam to
a common resolution inflates that band's noise by ``(B_c/B_ν)²`` at fine scales. Stage 1
produces the raw white, anisotropic map-domain noise; Stage 3 makes it beam-aware.

Normalization convention
-------------------------
``hit_weight`` is normalized so the mean of its square over *surveyed* pixels (hits > 0)
is exactly 1. Then the per-pixel variance ``v(p) = (w_inv / Ω_pix) · hit_weight(p)²``
has surveyed-sky-averaged value ``w_inv / Ω_pix`` (with ``Ω_pix = 4π / N_pix``), so the
mask-averaged white noise power spectrum is ``w_inv`` — the spec, sky-averaged rather
than best-pixel. This is the "sky-averaged variance matches spec" convention; cf.
:func:`augr.hit_maps.mean_pixel_rescale_factor`, which instead pins the *best* pixel to
spec (BROOM's internal max-normalized-hits convention). Unsurveyed pixels (hits == 0)
get zero noise; the caller is responsible for masking them out of any spectrum estimate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def hit_weight(hit_map: jax.Array) -> jax.Array:
    """Per-pixel noise-amplitude weight from a hit map, ``mean_surveyed(w²) = 1``.

    ``w(p) = sqrt[ (1/H(p)) / mean_surveyed(1/H) ]`` on surveyed pixels (H > 0), else 0.
    Dimensionless and invariant to the overall normalization of ``H`` (hit maps carry
    arbitrary units). Larger where there are fewer hits.

    Parameters
    ----------
    hit_map
        Relative exposure / hit count per pixel, shape ``(npix,)``. Non-negative.

    Returns
    -------
    Weight array, shape ``(npix,)``; 0 on unsurveyed pixels.
    """
    H = jnp.asarray(hit_map)
    surveyed = H > 0
    inv = jnp.where(surveyed, 1.0 / jnp.where(surveyed, H, 1.0), 0.0)
    n_surveyed = jnp.sum(surveyed)
    mean_inv = jnp.sum(inv) / n_surveyed
    return jnp.where(surveyed, jnp.sqrt(inv / mean_inv), 0.0)


def noise_map(hit_map: jax.Array, w_inv: jax.Array, key: jax.Array) -> jax.Array:
    """Single-channel anisotropic white-noise map [μK], differentiable in ``w_inv``.

    Parameters
    ----------
    hit_map
        Relative exposure per pixel, shape ``(npix,)``.
    w_inv
        White-noise power [μK²·sr] (scalar), e.g. from
        :func:`augr.instrument.white_noise_power`.
    key
        JAX PRNG key. Fix it across allocations for common-random-number gradients.

    Returns
    -------
    Noise map [μK], shape ``(npix,)``; 0 on unsurveyed pixels.
    """
    H = jnp.asarray(hit_map)
    npix = H.shape[-1]
    omega_pix = 4.0 * jnp.pi / npix
    z = jax.random.normal(key, (npix,), dtype=H.dtype)
    return jnp.sqrt(w_inv / omega_pix) * hit_weight(H) * z


def noise_maps(hit_map: jax.Array, w_inv: jax.Array, key: jax.Array) -> jax.Array:
    """Per-band anisotropic white-noise maps [μK], differentiable in ``w_inv``.

    The scan (hit map) is shared across bands; each band gets an independent noise
    realization (split keys) and its own ``w_inv``.

    Parameters
    ----------
    hit_map
        Shared relative exposure per pixel, shape ``(npix,)``.
    w_inv
        Per-band white-noise power [μK²·sr], shape ``(n_band,)``.
    key
        JAX PRNG key; split into one sub-key per band.

    Returns
    -------
    Noise maps [μK], shape ``(n_band, npix)``.
    """
    H = jnp.asarray(hit_map)
    w_inv = jnp.atleast_1d(jnp.asarray(w_inv))
    n_band = w_inv.shape[0]
    npix = H.shape[-1]
    omega_pix = 4.0 * jnp.pi / npix
    keys = jax.random.split(key, n_band)
    z = jax.vmap(lambda k: jax.random.normal(k, (npix,), dtype=H.dtype))(keys)
    return jnp.sqrt(w_inv[:, None] / omega_pix) * hit_weight(H)[None, :] * z
