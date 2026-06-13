"""masking.py — cut-sky B-mode bandpowers via jht's masked Wiener filter.

The in-house compsep sim path (:mod:`augr.compsep_sims`) is full-sky and forms
spectra with ``alm2cl`` and a scalar ``1/f_sky`` correction, so it does not
model realistic sky coverage: the spin-2 E/B split is ambiguous on a cut sky,
and since ``C_ℓ^{EE} ≫ C_ℓ^{BB}`` the leaked E *cosmic variance* inflates the B
error bars by a mask-dependent factor — contaminating instrument *design
comparisons*, not just absolute σ(r). This module replaces that path with a
masked, noise-aware **Wiener-filter** B-mode estimator built on the JAX-native
SHT package `jht` (PyPI ``jaxht``).

Estimator chain (all differentiable; ``jht`` runs its CG in fp64 because augr
enables JAX x64 at import):

    Q/U map  --jht.wiener(spin=2, prior=(C_EE,C_BB)@r=0, N^-1)-->  E/B alm
             --jht.bandpower(spin=2)[1]-->  raw C_ℓ^{BB}
             --bin_spectrum-->  binned bandpowers

The Wiener mean is a *suppressed* estimate of the sky (the prior bounds the
near-null E/B-ambiguous modes — a bias-for-variance trade), so the raw
bandpower is **biased** and must be debiased:

* a **multiplicative transfer** ``F_b`` from B-only sims (corrects the filter's
  suppression of true B), and
* an **always-subtracted additive** E→B leakage template ``⟨C_b^{BB,leak}⟩``
  from E-only (B=0) sims (E→B leakage is additive; ``F_b`` does not remove it).

``C_b^{true} = (C_b^{rec} − leak_b) / F_b`` (:func:`debias_bandpower`). The two
sim sets separate the two corrections cleanly: B-only inputs carry no E, so
``F_b`` is leakage-free; E-only inputs carry no B, so the recovered power is
pure leakage.

Conventions
-----------
* **Signal prior fixed at r=0** (LCDM): ``C_EE`` = lensed EE, ``C_BB`` = lensing
  BB. The filter is an estimator choice, not the signal model — fixing it at
  r=0 keeps ``∂(filter)/∂r`` out of the σ(r) derivative (a production pipeline
  fixes the filter before knowing r). Use :func:`augr.delensing.load_lensing_spectra`
  (``cl_ee_len`` / ``cl_bb_len``) for the priors.
* **Maps are in the galactic frame** (PySM native; matches
  ``hit_maps.l2_hit_map(coord="G")``), so a ``|b|``-cut mask thresholds pixel
  latitude directly with no rotation. RING ordering, HEALPix-internal Q/U.
* **Sharp masks, no apodization** (the design choice; the Wiener prior, not an
  apodization taper, controls the ambiguous modes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _require_jht():
    """Import jht or raise a helpful error pointing at the [masking] extra."""
    try:
        import jht

        return jht
    except ImportError as exc:  # pragma: no cover - exercised only without jht
        raise ImportError(
            "augr.masking requires 'jht' (PyPI distribution 'jaxht'), which "
            "ships with the masking extra. Install it with:\n"
            "    pip install 'cmb-augr[masking]'\n"
            "or, in the development env:\n"
            "    pixi add --pypi 'jaxht>=0.1.2'"
        ) from exc


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def gal_cut_mask(nside: int, b_cut_deg: float) -> jax.Array:
    """Sharp galactic ``|b| > b_cut`` binary mask (1 = observed, 0 = cut), RING.

    Thresholds pixel latitude directly; assumes galactic-frame maps (see module
    docstring). ``b_cut_deg = 0`` is full sky.
    """
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    theta = hp.pix2ang(int(nside), np.arange(npix))[0]  # colatitude [rad]
    lat_deg = 90.0 - np.degrees(theta)
    mask = (np.abs(lat_deg) >= float(b_cut_deg)).astype(np.float64)
    return jnp.asarray(mask)


def galactic_mask(nside: int, f_sky: float) -> jax.Array:
    """``|b|``-cut binary mask hitting a target ``f_sky``.

    For a symmetric ``|b| > b_cut`` cut over both hemispheres, the observed sky
    fraction is ``f_sky = 1 − sin(b_cut)``, so ``b_cut = arcsin(1 − f_sky)``.
    The realized ``f_sky`` matches the target up to HEALPix pixelization.
    """
    b_cut_deg = float(np.degrees(np.arcsin(np.clip(1.0 - float(f_sky), -1.0, 1.0))))
    return gal_cut_mask(nside, b_cut_deg)


def load_mask(path: str, *, nside: int | None = None, field: int = 0) -> jax.Array:
    """Load a HEALPix mask FITS (RING). Optionally ``ud_grade`` to ``nside``."""
    import healpy as hp

    m = np.asarray(hp.read_map(path, field=field), dtype=np.float64)
    if nside is not None and hp.get_nside(m) != int(nside):
        m = hp.ud_grade(m, int(nside))
    return jnp.asarray(m)


def f_sky_of(mask: jax.Array) -> float:
    """Effective sky fraction ``⟨mask⟩`` (mean of the per-pixel weight)."""
    return float(jnp.mean(jnp.asarray(mask)))


# ---------------------------------------------------------------------------
# per-pixel inverse-noise
# ---------------------------------------------------------------------------


def inv_noise_map(
    hit_map: jax.Array, var_pix_ref: float, *, mask: jax.Array | None = None
) -> jax.Array:
    """Per-pixel inverse-noise ``N^-1`` [μK⁻²] for :func:`masked_wiener_bb`.

    The polarization pixel noise variance scales inversely with exposure:
    ``σ²(p) = var_pix_ref · max(H) / H(p)``, so ``N^-1(p) = H(p) / (var_pix_ref ·
    max(H))``. Zero-hit pixels and (if ``mask`` is given) cut pixels are set to
    0 — the mask must be folded into ``inv_noise`` for jht's Wiener solve (it
    does not apply a separate mask when ``inv_noise`` is supplied).

    Parameters
    ----------
    hit_map
        Relative exposure per pixel ``(npix,)`` (any positive scaling; only the
        shape matters, e.g. ``hit_maps.l2_hit_map``).
    var_pix_ref
        Per-pixel polarization noise variance [μK²] at the maximum-exposure
        pixel (the cleaned-map white level there).
    mask
        Optional binary/weight mask ``(npix,)``; pixels with ``mask <= 0`` are
        zeroed.
    """
    h = jnp.asarray(hit_map)
    hmax = jnp.max(h)
    inv = jnp.where(h > 0, h / (float(var_pix_ref) * hmax), 0.0)
    if mask is not None:
        inv = inv * (jnp.asarray(mask) > 0)
    return inv


# ---------------------------------------------------------------------------
# masked-Wiener B-mode estimator
# ---------------------------------------------------------------------------


def masked_wiener_eb_alm(
    qu_map: jax.Array,
    inv_noise: jax.Array,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    nside: int,
    lmax: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> jax.Array:
    """Wiener-filtered E/B alm from a masked Q/U map → ``(2, K)`` ``[E, B]``.

    Solves ``(SᵀN⁻¹S + C⁻¹) a = SᵀN⁻¹ m`` (jht's prior-whitened CG). The signal
    prior is fixed at r=0: ``cl_ee_prior`` = lensed EE, ``cl_bb_prior`` = lensing
    BB (each ``(lmax+1,)``). ``inv_noise`` ``(npix,)`` must have the mask folded
    in (see :func:`inv_noise_map`).
    """
    jht = _require_jht()
    cl_ee = jnp.asarray(cl_ee_prior)
    cl_bb = jnp.asarray(cl_bb_prior)
    return jht.wiener(
        jnp.asarray(qu_map),
        (cl_ee, cl_bb),
        int(nside),
        int(lmax),
        spin=2,
        inv_noise=jnp.asarray(inv_noise),
        max_iter=max_iter,
        tol=tol,
    )


def masked_wiener_bb(
    qu_map: jax.Array,
    inv_noise: jax.Array,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    nside: int,
    lmax: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> jax.Array:
    """Raw per-ℓ ``C_ℓ^{BB}`` (length ``lmax+1``) of the Wiener-filtered map.

    This is the **filter-suppressed, leakage-contaminated** B power — apply
    :func:`transfer_function` and subtract :func:`leakage_template` (via
    :func:`debias_bandpower`) for an unbiased estimate. Differentiable in the
    map and the noise level.
    """
    jht = _require_jht()
    eb_alm = masked_wiener_eb_alm(
        qu_map,
        inv_noise,
        cl_ee_prior,
        cl_bb_prior,
        nside=nside,
        lmax=lmax,
        max_iter=max_iter,
        tol=tol,
    )
    return jht.bandpower(eb_alm, int(lmax), spin=2)[1]  # C_BB channel


def bin_spectrum(cl: jax.Array, bin_matrix: jax.Array, ell_min: int) -> jax.Array:
    """Bin a per-ℓ spectrum (indexed from ℓ=0) onto SignalModel bins.

    ``bin_matrix`` is ``(n_bins, n_ells)`` over ``[ell_min, ell_max]`` (e.g.
    ``SignalModel.bin_matrix``); the spectrum is sliced to that range, so
    ``ell_min`` is the SignalModel's ``ell_min``. Returns ``(n_bins,)``.
    """
    bm = jnp.asarray(bin_matrix)
    n_ells = bm.shape[1]
    seg = jnp.asarray(cl)[int(ell_min) : int(ell_min) + n_ells]
    return bm @ seg


# ---------------------------------------------------------------------------
# debias: multiplicative transfer (B-only) + additive E→B leakage (E-only)
# ---------------------------------------------------------------------------


def transfer_function(rec_binned_b_only: jax.Array, true_binned_bb: jax.Array) -> jax.Array:
    """Multiplicative transfer ``F_b = ⟨C_b^{rec}⟩ / C_b^{true}`` from B-only sims.

    Parameters
    ----------
    rec_binned_b_only
        Recovered binned BB bandpowers from B-only (E=0) inputs, ``(n_sims,
        n_bins)`` (averaged over sims) or already-averaged ``(n_bins,)``.
    true_binned_bb
        The binned input BB truth ``(n_bins,)`` (same binning).
    """
    rec = jnp.asarray(rec_binned_b_only)
    mean_rec = jnp.mean(rec, axis=0) if rec.ndim == 2 else rec
    return mean_rec / jnp.asarray(true_binned_bb)


def leakage_template(rec_binned_e_only: jax.Array) -> jax.Array:
    """Additive E→B leakage template ``⟨C_b^{BB,leak}⟩`` from E-only (B=0) sims.

    ``rec_binned_e_only`` is ``(n_sims, n_bins)`` (averaged) or ``(n_bins,)``.
    """
    rec = jnp.asarray(rec_binned_e_only)
    return jnp.mean(rec, axis=0) if rec.ndim == 2 else rec


def debias_bandpower(rec_binned: jax.Array, transfer: jax.Array, leakage: jax.Array) -> jax.Array:
    """Unbiased binned BB: ``(C_b^{rec} − leakage_b) / F_b``.

    Inverts the bandpower model ``C^{rec} = F · C^{true,B} + leakage`` (the
    multiplicative filter suppression plus the additive E→B leakage).
    """
    return (jnp.asarray(rec_binned) - jnp.asarray(leakage)) / jnp.asarray(transfer)
