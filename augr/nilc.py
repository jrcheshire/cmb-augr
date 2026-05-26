"""nilc.py — differentiable empirical needlet ILC for B-mode component separation.

A blind, empirically-weighted needlet internal-linear-combination (NILC) cleaner
built entirely on the differentiable ``augr.sht`` primitives, so the cleaned-map
power spectra — and hence σ(r) and the FG-leakage bias Δr — are differentiable in
the per-band noise (and therefore the focal-plane allocation). "Empirical" means
the ILC weights come from the *data* covariance, not a foreground model: that is
required here, because the science question is exactly how well a blind cleaner
removes foregrounds the analytic model does not know about.

Pipeline (B-only, full-sky v1)
------------------------------
1. **Common resolution.** Per band, extract the B-mode alm (``map2alm`` spin-2,
   keep B) and bring every band to a common beam ``B_c`` (default the finest band's
   beam) by ``almxfl(B_alm, B_c/B_ν)`` in harmonic space. **This is the load-bearing
   step**: a coarse low-frequency beam (small aperture) is deconvolved up to the
   fine common resolution, inflating that band's *map-domain* white noise by
   ``(B_c/B_ν)²`` at small scales. The band then carries huge variance in the fine
   needlet bands and the ILC down-weights it to ~0, so it cannot clean small-scale
   synchrotron there — the mechanism that sets the minimum aperture ``D_min``.
2. **Needlet decomposition.** Cosine-needlet bandpass windows ``h_j(ℓ)`` (partition
   of unity, ``Σ_j h_j² = 1``) split each band's common-resolution B alm into
   needlet coefficient maps ``β_{j,b}`` (spin-0 synthesis of the windowed alm).
3. **Empirical covariance + ILC weights.** Per needlet band ``j`` build the
   cross-band covariance ``C_j`` (global full-sky average by default, or a
   Gaussian-localized per-pixel covariance if ``localization_fwhm`` is set) and the
   weights ``w_j = C_j⁻¹ a / (aᵀ C_j⁻¹ a)`` with ``a = 1`` (the CMB response is
   identical across bands in thermodynamic units once at common resolution). The
   constraint ``aᵀ w_j = 1`` is exact by construction, so the CMB is preserved.
4. **Recompose.** Clean each needlet band, ``s_j = Σ_b w_{j,b} β_{j,b}``, and sum
   back, ``cleaned_B_alm = Σ_j h_j · map2alm(s_j)`` — the cleaned B at resolution
   ``B_c``.

The same weights are applied to passive map sets (FG-only / noise-only / CMB-only)
via :meth:`NILCResult.project`, giving the residual-FG, post-NILC-noise, and
signal-transfer decompositions the forecast (Stage 4) consumes.

Conventions: B alm in healpy triangular packing; maps ``(n_band, 2, npix)`` Q/U
(HEALPix-internal); beams Gaussian via :func:`augr.instrument.beam_bl`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .instrument import beam_bl
from .sht import almxfl, check_band_limit, map2alm, synthesis

# ---------------------------------------------------------------------------
# cosine needlet bands
# ---------------------------------------------------------------------------


def cosine_needlet_bands(lmax: int, peaks) -> jax.Array:
    """Cosine-needlet bandpass windows ``h_j(ℓ)``, shape ``(J, lmax+1)``.

    ``peaks`` is an increasing sequence of ``J`` peak multipoles. Band 0 is flat
    (=1) below ``peaks[0]`` then cosine-tapers to ``peaks[1]``; the last band rises
    from ``peaks[-2]`` and is flat above ``peaks[-1]``; interior bands rise on
    ``[peaks[j-1], peaks[j]]`` and fall on ``[peaks[j], peaks[j+1]]``. Adjacent
    windows satisfy ``cos² + sin² = 1`` on their overlap, so ``Σ_j h_j(ℓ)² = 1``
    on ``[0, lmax]`` — a partition of unity in power, making the decompose/recompose
    round-trip exact.
    """
    peaks = sorted(int(p) for p in peaks)
    ell = np.arange(lmax + 1)
    n_bands = len(peaks)
    h = np.zeros((n_bands, lmax + 1))
    for j in range(n_bands):
        pj = peaks[j]
        left = peaks[j - 1] if j > 0 else None
        right = peaks[j + 1] if j < n_bands - 1 else None
        if left is None:  # flat below the first peak
            h[j, ell <= pj] = 1.0
        else:  # cosine rising edge [left, pj]
            mask = (ell >= left) & (ell <= pj)
            h[j, mask] = np.cos(np.pi / 2 * (pj - ell[mask]) / (pj - left))
        if right is None:  # flat above the last peak
            h[j, ell >= pj] = 1.0
        else:  # cosine falling edge [pj, right]
            mask = (ell >= pj) & (ell <= right)
            h[j, mask] = np.cos(np.pi / 2 * (ell[mask] - pj) / (right - pj))
    return jnp.asarray(h)


def default_needlet_peaks(lmax: int, n_bands: int = 6) -> list[int]:
    """Geometrically-spaced needlet peaks from ``ℓ≈8`` to ``lmax`` (``n_bands`` peaks)."""
    peaks = np.unique(np.round(np.geomspace(8, lmax, n_bands)).astype(int))
    peaks[-1] = lmax
    return peaks.tolist()


# ---------------------------------------------------------------------------
# common-resolution B-mode extraction
# ---------------------------------------------------------------------------


def common_resolution_b_alm(
    band_qu: jax.Array,
    beam_fwhm_arcmin,
    *,
    lmax: int,
    nside: int,
    n_iter: int = 3,
    common_fwhm_arcmin: float | None = None,
) -> tuple[jax.Array, float]:
    """Per-band Q/U → common-resolution B-mode alm, shape ``(n_band, Nlm)``.

    Each band's B alm is deconvolved from its own beam and reconvolved to
    ``common_fwhm_arcmin`` (default: the finest band's beam). Returns the stacked
    B alms and the common FWHM used. The ``B_c/B_ν`` ratio is the noise-inflation
    factor; choose ``lmax`` so the coarsest beam's ``B_ν(lmax)`` stays representable
    (a tiny floor guards against division by an underflowed beam).
    """
    beams = [float(b) for b in beam_fwhm_arcmin]
    if common_fwhm_arcmin is None:
        common_fwhm_arcmin = min(beams)
    ells = jnp.arange(lmax + 1, dtype=float)
    bl_common = beam_bl(ells, common_fwhm_arcmin)
    out = []
    for qu_b, fwhm_b in zip(band_qu, beams, strict=True):
        eb = map2alm(qu_b, 2, lmax, nside, n_iter)  # (2, Nlm) = (E, B)
        bl_band = beam_bl(ells, fwhm_b)
        ratio = bl_common / jnp.maximum(bl_band, 1e-30)
        out.append(almxfl(eb[1], ratio, lmax))
    return jnp.stack(out, axis=0), float(common_fwhm_arcmin)


# ---------------------------------------------------------------------------
# needlet decomposition / recomposition
# ---------------------------------------------------------------------------


def needlet_beta(b_alm: jax.Array, needlet_bands: jax.Array, *, lmax: int, nside: int) -> jax.Array:
    """Common-resolution B alms → needlet coefficient maps, shape ``(J, n_band, npix)``."""
    beta = []
    for hj in needlet_bands:
        per_band = [
            synthesis(almxfl(alm_b, hj, lmax)[None, :], 0, lmax, nside)[0] for alm_b in b_alm
        ]
        beta.append(jnp.stack(per_band, axis=0))
    return jnp.stack(beta, axis=0)


def combine_needlets(
    weights: jax.Array,
    beta: jax.Array,
    needlet_bands: jax.Array,
    *,
    lmax: int,
    nside: int,
    n_iter: int = 3,
) -> jax.Array:
    """Weight + recompose needlet maps into a single cleaned B alm, shape ``(Nlm,)``.

    ``s_j = Σ_b w_{j,b} β_{j,b}``; ``cleaned = Σ_j h_j · map2alm(s_j)``.
    """
    s = jnp.sum(weights * beta, axis=1)  # (J, npix)
    acc = [
        almxfl(map2alm(s[j][None, :], 0, lmax, nside, n_iter)[0], hj, lmax)
        for j, hj in enumerate(needlet_bands)
    ]
    return jnp.sum(jnp.stack(acc, axis=0), axis=0)


# ---------------------------------------------------------------------------
# empirical ILC weights
# ---------------------------------------------------------------------------


def _ilc_weights_from_cov(cov: jax.Array) -> jax.Array:
    """ILC weights ``w = C⁻¹ a / (aᵀ C⁻¹ a)`` with ``a = 1``.

    ``cov`` is ``(..., n_band, n_band)``; returns ``(..., n_band)``. The
    constraint ``Σ_b w_b = 1`` holds exactly.
    """
    n_band = cov.shape[-1]
    a = jnp.ones((*cov.shape[:-2], n_band))
    # Explicit rhs column so the batched (npix, n_band, n_band) solve is unambiguous.
    cinv_a = jnp.linalg.solve(cov, a[..., None])[..., 0]
    return cinv_a / jnp.sum(a * cinv_a, axis=-1, keepdims=True)


def _ridge(cov: jax.Array, ridge: float) -> jax.Array:
    """Add a relative diagonal ridge ``ridge · tr(C)/n · I`` for invertibility."""
    n_band = cov.shape[-1]
    scale = jnp.trace(cov, axis1=-2, axis2=-1) / n_band
    eye = jnp.eye(n_band)
    eye = jnp.broadcast_to(eye, cov.shape)
    return cov + ridge * scale[..., None, None] * eye


def _needlet_channel_mask(
    needlet_bands: jax.Array,
    beam_fwhm_arcmin,
    common_fwhm_arcmin: float,
    lmax: int,
    threshold: float,
) -> np.ndarray:
    """Which channels participate in each needlet band, shape ``(J, n_band)`` bool.

    A channel joins needlet band ``j`` only where its beam retains enough support
    relative to the common-resolution beam over the band — specifically
    ``B_ν(ℓ_hi) / B_c(ℓ_hi) >= threshold`` at the band's upper support edge
    ``ℓ_hi``. This caps the common-resolution deconvolution amplification
    ``B_c/B_ν`` at ``1/threshold`` for any included channel, so a coarse low-ν
    beam is *excluded* from fine needlet bands instead of being deconvolved to
    astronomical noise there (the small-aperture / high-ℓ blow-up). The finest
    channel is the common beam, so its ratio is 1 and it is active in every band;
    every band therefore retains at least one channel.

    The mask depends only on the (static) beams, not the maps, so it does not
    affect differentiability with respect to the maps / allocation.
    """
    nb = np.asarray(needlet_bands)
    ells = jnp.arange(lmax + 1, dtype=float)
    bl_common = np.asarray(beam_bl(ells, common_fwhm_arcmin))
    beams = [float(b) for b in beam_fwhm_arcmin]
    n_j = nb.shape[0]
    mask = np.zeros((n_j, len(beams)), dtype=bool)
    for j in range(n_j):
        support = np.nonzero(nb[j] > 1e-3)[0]
        ell_hi = int(support[-1]) if support.size else 0
        bc = max(float(bl_common[ell_hi]), 1e-30)
        for b, fwhm in enumerate(beams):
            bl_band = float(np.asarray(beam_bl(jnp.asarray([float(ell_hi)]), fwhm))[0])
            mask[j, b] = (bl_band / bc) >= threshold
    return mask


def _global_weights(beta: jax.Array, ridge: float, active: np.ndarray) -> jax.Array:
    """Spatially-constant weights per needlet band, shape ``(J, n_band, npix)``.

    The empirical covariance and ILC solve run over the *active* channels of each
    needlet band (``active[j]``); excluded channels get weight exactly 0, so their
    (possibly deconvolution-inflated) maps never enter the cleaned product.
    """
    n_j, n_band, npix = beta.shape
    ws = []
    for j in range(n_j):
        idx = np.nonzero(active[j])[0]
        bj = beta[j][idx]  # (n_active, npix)
        cov = (bj @ bj.T) / npix  # (n_active, n_active)
        w_active = _ilc_weights_from_cov(_ridge(cov, ridge))  # (n_active,)
        w = jnp.zeros(n_band).at[jnp.asarray(idx)].set(w_active)
        ws.append(jnp.broadcast_to(w[:, None], (n_band, npix)))
    return jnp.stack(ws, axis=0)


def _localized_weights(
    beta: jax.Array,
    localization_fwhm_arcmin: float,
    *,
    lmax: int,
    nside: int,
    n_iter: int,
    ridge: float,
    active: np.ndarray,
) -> jax.Array:
    """Per-pixel weights from a Gaussian-localized empirical covariance.

    ``C_j(p)_{bb'} = smooth(β_{j,b}·β_{j,b'})`` with a Gaussian of FWHM
    ``localization_fwhm_arcmin``. The localization scale sets the modes-per-domain
    and hence the ILC bias; choose it so domains hold ≫ n_band modes. As in
    :func:`_global_weights`, the per-pixel solve runs over the active channels of
    each needlet band; excluded channels get weight 0.
    """
    n_j, n_band, _npix = beta.shape
    ells = jnp.arange(lmax + 1, dtype=float)
    gauss = beam_bl(ells, localization_fwhm_arcmin)

    def smooth(x):  # spatial Gaussian smoothing of a real map (npix,)
        alm = map2alm(x[None, :], 0, lmax, nside, n_iter)
        return synthesis(almxfl(alm, gauss, lmax), 0, lmax, nside)[0]

    ws = []
    for j in range(n_j):
        idx = np.nonzero(active[j])[0]
        bj = beta[j][idx]  # (n_active, npix)
        n_act = len(idx)
        rows = [
            jnp.stack([smooth(bj[i] * bj[k]) for k in range(n_act)], axis=0) for i in range(n_act)
        ]
        cov = jnp.stack(rows, axis=0)  # (n_active, n_active, npix)
        cov = jnp.moveaxis(cov, 2, 0)  # (npix, n_active, n_active)
        w_act = _ilc_weights_from_cov(_ridge(cov, ridge))  # (npix, n_active)
        w = jnp.zeros((n_band, _npix)).at[jnp.asarray(idx)].set(w_act.T)
        ws.append(w)  # (n_band, npix)
    return jnp.stack(ws, axis=0)


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NILCResult:
    """Output of :func:`nilc_clean`, with everything needed to project passives.

    Attributes
    ----------
    cleaned_b_alm
        Cleaned B-mode alm at the common resolution, shape ``(Nlm,)``.
    weights
        ILC weights, shape ``(J, n_band, npix)`` (constant over pixels in the
        global case).
    needlet_bands
        Cosine-needlet windows used, shape ``(J, lmax+1)``.
    beam_fwhm_arcmin, common_fwhm_arcmin
        Per-band input beams and the common resolution they were brought to.
    lmax, nside, n_iter
        Transform configuration (so passives are projected identically).
    """

    cleaned_b_alm: jax.Array
    weights: jax.Array
    needlet_bands: jax.Array
    beam_fwhm_arcmin: tuple[float, ...]
    common_fwhm_arcmin: float
    lmax: int
    nside: int
    n_iter: int

    def project(self, passive_band_qu: jax.Array) -> jax.Array:
        """Apply the stored weights to another map set → its cleaned B alm.

        Use for FG-only / noise-only / CMB-only maps to get the residual-FG,
        post-NILC-noise, and signal-transfer pieces (same instrument beams).
        """
        b_alm, _ = common_resolution_b_alm(
            passive_band_qu,
            self.beam_fwhm_arcmin,
            lmax=self.lmax,
            nside=self.nside,
            n_iter=self.n_iter,
            common_fwhm_arcmin=self.common_fwhm_arcmin,
        )
        beta = needlet_beta(b_alm, self.needlet_bands, lmax=self.lmax, nside=self.nside)
        return combine_needlets(
            self.weights,
            beta,
            self.needlet_bands,
            lmax=self.lmax,
            nside=self.nside,
            n_iter=self.n_iter,
        )


def nilc_clean(
    band_qu: jax.Array,
    beam_fwhm_arcmin,
    *,
    lmax: int,
    nside: int,
    needlet_peaks=None,
    localization_fwhm_arcmin: float | None = None,
    common_fwhm_arcmin: float | None = None,
    n_iter: int = 3,
    ridge: float = 1e-10,
    beam_band_limit: float = 0.1,
) -> NILCResult:
    """Run the differentiable empirical needlet ILC on per-band Q/U maps.

    Parameters
    ----------
    band_qu
        Per-band Q/U maps, shape ``(n_band, 2, npix)`` [μK_CMB].
    beam_fwhm_arcmin
        Per-band beam FWHM [arcmin], length ``n_band``.
    lmax, nside
        Transform band limit and HEALPix resolution.
    needlet_peaks
        Peak multipoles for the cosine needlets (default
        :func:`default_needlet_peaks`).
    localization_fwhm_arcmin
        ``None`` → spatially-constant (global) weights per needlet band (default);
        a finite FWHM → spatially-varying weights from a Gaussian-localized
        empirical covariance.
    common_fwhm_arcmin
        Common resolution to bring all bands to (default: finest band's beam).
    n_iter
        ``map2alm`` Jacobi iterations.
    ridge
        Relative diagonal regularization on the covariance before inversion.
    beam_band_limit
        A channel joins a needlet band only where ``B_ν/B_c >= beam_band_limit``
        at the band's upper edge, capping the deconvolution amplification at
        ``1/beam_band_limit`` (default 0.1 → ≤10×). Excludes coarse low-ν channels
        from fine needlet bands instead of deconvolving them to astronomical noise
        — the fix for the small-aperture / high-ℓ covariance blow-up. At low lmax
        every channel resolves every band, so the mask is all-True and this is a
        no-op.

    Returns
    -------
    :class:`NILCResult`.
    """
    check_band_limit(lmax, nside)
    beams = tuple(float(b) for b in beam_fwhm_arcmin)
    if needlet_peaks is None:
        needlet_peaks = default_needlet_peaks(lmax)
    needlet_bands = cosine_needlet_bands(lmax, needlet_peaks)

    b_alm, common_fwhm = common_resolution_b_alm(
        band_qu,
        beams,
        lmax=lmax,
        nside=nside,
        n_iter=n_iter,
        common_fwhm_arcmin=common_fwhm_arcmin,
    )
    beta = needlet_beta(b_alm, needlet_bands, lmax=lmax, nside=nside)

    active = _needlet_channel_mask(needlet_bands, beams, common_fwhm, lmax, beam_band_limit)
    if localization_fwhm_arcmin is None:
        weights = _global_weights(beta, ridge, active)
    else:
        weights = _localized_weights(
            beta,
            localization_fwhm_arcmin,
            lmax=lmax,
            nside=nside,
            n_iter=n_iter,
            ridge=ridge,
            active=active,
        )

    cleaned = combine_needlets(weights, beta, needlet_bands, lmax=lmax, nside=nside, n_iter=n_iter)
    return NILCResult(
        cleaned_b_alm=cleaned,
        weights=weights,
        needlet_bands=needlet_bands,
        beam_fwhm_arcmin=beams,
        common_fwhm_arcmin=common_fwhm,
        lmax=int(lmax),
        nside=int(nside),
        n_iter=int(n_iter),
    )
