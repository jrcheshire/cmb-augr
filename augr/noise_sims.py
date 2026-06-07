"""noise_sims.py — anisotropic noise map synthesis for map-based comp-sep.

Draws per-band noise realizations whose spatial pattern follows the inverse hit count
(more hits → lower variance) and whose surveyed-sky-averaged power matches the
white-noise spec ``w_inv`` [μK²·sr] from :func:`augr.instrument.white_noise_power`,
optionally with a 1/f spectral tilt ``N_ℓ = w_inv · (1 + (ℓ_knee/ℓ)^α)`` matching the
:func:`augr.instrument.noise_nl` convention.

Two draws are provided:

* :func:`noise_maps` — **white**, per-pixel iid. Pixel-domain only, ducc-free.
* :func:`correlated_noise_maps` — **white + 1/f**, drawn in harmonic space (a unit-power
  Gaussian alm field filtered by ``sqrt(N_ℓ)``, synthesized, then multiplied by the
  ``hit_weight`` envelope). Requires the ``[compsep]`` extra (ducc0) for the SHT.

Both are differentiable in ``w_inv`` (hence ``n_det``, hence focal-plane allocation)
under **common random numbers**: the design enters only through the ``sqrt(w_inv)``
amplitude, so a fixed unit realization (fixed ``key``) and a fixed hit map can be reused
across allocations to give a low-variance pathwise gradient. ``correlated_noise_maps`` is
*additionally* differentiable in ``knee_ell`` / ``alpha_knee`` (they enter only through
``sqrt(N_ℓ)``, with the unit field held fixed). At ``knee_ell = 0`` it reduces to the
white case (``N_ℓ = w_inv`` flat), reproducing :func:`noise_maps`' *angular* power
spectrum over ``[0, lmax]`` (the harmonic draw is band-limited to ``lmax`` where the
pixel-domain white draw is not, so the pixel realizations differ but the measured
``C_ℓ`` agree — and ``C_ℓ`` is what the downstream forecast consumes).

The per-band **beam** dependence of the noise is *not* applied here — it enters at the
NILC common-resolution step (:mod:`augr.nilc`), where deconvolving each band's beam to
a common resolution inflates that band's noise by ``(B_c/B_ν)²`` at fine scales. Stage 1
produces the raw anisotropic map-domain noise; Stage 3 makes it beam-aware.

Anisotropy × 1/f is approximate by construction
-----------------------------------------------
Anisotropic 1/f noise is not separable: multiplying a correlated field by the spatial
``hit_weight`` envelope *convolves* its power spectrum, so the realized sky-averaged
``N_ℓ`` only approximately equals ``w_inv · (1 + (ℓ_knee/ℓ)^α)`` once the hit map is
non-uniform. The recovery is **exact in the uniform-hits limit** (where ``hit_weight ≡
1``) and mild for the smooth L2 coverage envelope at the bump scales that drive σ(r);
the exact anisotropic-1/f covariance is a per-pixel correlated object and is out of
scope (it needs the time-ordered-data path, deferred to ``bk-jax``). The 1/f is also
drawn independently in Q and U (isotropic, no scan-direction structure) — appropriate
for a space mission, but it does not capture the correlated common mode that scan
cross-linking suppresses; that, too, is a TOD-level effect.

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

from .sht import _m_zero_mask, alm_size, almxfl, synthesis


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


# ---------------------------------------------------------------------------
# correlated (white + 1/f) anisotropic noise, drawn in harmonic space
# ---------------------------------------------------------------------------


def one_over_f_factor(ells: jax.Array, knee_ell: jax.Array, alpha_knee: jax.Array) -> jax.Array:
    """1/f spectral factor ``(ℓ_knee/ℓ)^α`` with the :func:`augr.instrument.noise_nl` guards.

    Returns ``(knee_ell / max(ℓ, 1))^alpha_knee`` where ``knee_ell > 0`` and ``ℓ > 0``,
    else 0 — so ``N_ℓ = w_inv · (1 + factor)`` is pure white (``= w_inv``) when
    ``knee_ell = 0`` and never divides by ``ℓ = 0``. Same convention as the analytic
    spectrum-domain noise, so the two paths agree at the spectrum level.
    """
    return jnp.where(
        (knee_ell > 0) & (ells > 0),
        (knee_ell / jnp.maximum(ells, 1.0)) ** alpha_knee,
        0.0,
    )


def _unit_white_alm(key: jax.Array, lmax: int) -> jax.Array:
    """Draw a unit-power Gaussian alm field (``⟨C_ℓ⟩ = 1``) in healpy triangular packing.

    Matches the ``healpy.synalm`` normalization so that ``synthesis(almxfl(ξ, sqrt(C)))``
    has angular power ``C_ℓ``: ``a_{ℓ,0}`` is real ``N(0, 1)``; ``a_{ℓ,m>0} = (x + i y) /
    √2`` with ``x, y ~ N(0, 1)`` (so ``⟨|a_{ℓm}|²⟩ = 1``). Differentiability-free (the
    draw carries no design parameters); fixing ``key`` makes it a common-random-number
    seed reused across allocations and apertures.
    """
    nlm = alm_size(int(lmax))
    is_m0 = jnp.asarray(_m_zero_mask(int(lmax)))
    key_re, key_im = jax.random.split(key)
    re = jax.random.normal(key_re, (nlm,))
    im = jax.random.normal(key_im, (nlm,))
    return jnp.where(is_m0, re.astype(jnp.complex128), (re + 1j * im) / jnp.sqrt(2.0))


def correlated_noise_maps(
    hit_map: jax.Array,
    w_inv: jax.Array,
    knee_ell: jax.Array,
    alpha_knee: jax.Array,
    key: jax.Array,
    *,
    lmax: int,
    nside: int,
) -> jax.Array:
    """Per-band anisotropic noise maps with a 1/f tilt [μK], differentiable.

    Each band's noise has target angular power ``N_ℓ = w_inv · (1 + (ℓ_knee/ℓ)^α)``
    (uniform-hits limit), drawn as ``hit_weight(H) · synthesis(sqrt(N_ℓ) · ξ)`` with
    ``ξ`` a fixed unit-power Gaussian alm field (:func:`_unit_white_alm`). The
    ``hit_weight`` envelope imposes the 1/hits anisotropy; see the module docstring for
    why that makes the realized ``N_ℓ`` approximate away from uniform coverage.

    Differentiable in ``w_inv`` and in ``knee_ell`` / ``alpha_knee`` (all enter only
    through ``sqrt(N_ℓ)``, with ``ξ`` and the hit map held fixed for common random
    numbers). At ``knee_ell = 0`` this reduces to the white :func:`noise_maps` angular
    spectrum. Requires the ``[compsep]`` extra (ducc0) for the SHT.

    Parameters
    ----------
    hit_map
        Shared relative exposure per pixel, shape ``(npix,)`` with ``npix = 12·nside²``.
    w_inv
        Per-band polarization white-noise power [μK²·sr], shape ``(n_band,)``.
    knee_ell, alpha_knee
        Per-band 1/f knee multipole and slope. Scalars broadcast to all bands;
        ``knee_ell = 0`` disables the 1/f term for that band.
    key
        JAX PRNG key; split into one sub-key per band. Fix across allocations for CRN.
    lmax
        Band limit of the harmonic draw (use :func:`augr.sht.band_limit`).
    nside
        HEALPix resolution; must satisfy ``12·nside² == hit_map.shape[-1]``.

    Returns
    -------
    Noise maps [μK], shape ``(n_band, npix)``; the ``hit_weight`` envelope zeroes
    unsurveyed pixels.
    """
    H = jnp.asarray(hit_map)
    w_inv = jnp.atleast_1d(jnp.asarray(w_inv))
    n_band = w_inv.shape[0]
    knee_ell = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(knee_ell, dtype=float)), (n_band,))
    alpha_knee = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(alpha_knee, dtype=float)), (n_band,))

    ells = jnp.arange(int(lmax) + 1, dtype=float)
    hw = hit_weight(H)
    keys = jax.random.split(key, n_band)

    out = []
    for b in range(n_band):
        nl = w_inv[b] * (1.0 + one_over_f_factor(ells, knee_ell[b], alpha_knee[b]))
        xi = _unit_white_alm(keys[b], int(lmax))
        alm = almxfl(xi, jnp.sqrt(nl), int(lmax))
        n_corr = synthesis(alm[None, :], 0, int(lmax), int(nside))[0]
        out.append(hw * n_corr)
    return jnp.stack(out, axis=0)
