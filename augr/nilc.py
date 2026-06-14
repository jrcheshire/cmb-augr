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

from .instrument import ARCMIN_TO_RAD, beam_bl
from .sht import almxfl, check_band_limit, map2alm, synthesis, synthesis_pol

# ---------------------------------------------------------------------------
# cosine needlet bands
# ---------------------------------------------------------------------------


def cosine_needlet_bands(lmax: int, peaks) -> np.ndarray:
    """Cosine-needlet bandpass windows ``h_j(ℓ)``, shape ``(J, lmax+1)`` (numpy).

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
    # NumPy (not jnp): the windows are static (lmax/peaks only, never the maps,
    # never differentiated). Keeping them concrete lets the cleaner body run under
    # jax.jit / lax.map -- `_needlet_channel_mask` does `np.asarray(needlet_bands)`,
    # which fails on a tracer. almxfl/synthesis promote the numpy constant to the
    # device, so the differentiable map path is unchanged. (lax.map + jit gate.)
    return np.asarray(h)


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


def common_resolution_eb(
    band_qu: jax.Array,
    beam_fwhm_arcmin,
    *,
    lmax: int,
    nside: int,
    n_iter: int = 3,
    common_fwhm_arcmin: float | None = None,
) -> tuple[jax.Array, jax.Array, float]:
    """Per-band Q/U → common-resolution E *and* B alm, each shape ``(n_band, Nlm)``.

    The spin-2 companion to :func:`common_resolution_b_alm`: one ``map2alm(spin=2)``
    per band yields both E and B, each deconvolved from its band beam and reconvolved
    to ``common_fwhm_arcmin`` (default: the finest band's beam). Returns ``(e_alm,
    b_alm, common_fwhm)``. Used by the spin-2 Q/U cleaner (``clean_e=True``) so the
    cut-sky masked-Wiener estimator has a cleaned Q/U map to act on; the B-only path
    keeps using :func:`common_resolution_b_alm` unchanged (byte-identical, and without
    paying for the unused E leg).
    """
    beams = [float(b) for b in beam_fwhm_arcmin]
    if common_fwhm_arcmin is None:
        common_fwhm_arcmin = min(beams)
    ells = jnp.arange(lmax + 1, dtype=float)
    bl_common = beam_bl(ells, common_fwhm_arcmin)
    out_e = []
    out_b = []
    for qu_b, fwhm_b in zip(band_qu, beams, strict=True):
        eb = map2alm(qu_b, 2, lmax, nside, n_iter)  # (2, Nlm) = (E, B)
        bl_band = beam_bl(ells, fwhm_b)
        ratio = bl_common / jnp.maximum(bl_band, 1e-30)
        out_e.append(almxfl(eb[0], ratio, lmax))
        out_b.append(almxfl(eb[1], ratio, lmax))
    return jnp.stack(out_e, axis=0), jnp.stack(out_b, axis=0), float(common_fwhm_arcmin)


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

    ``s_j = Σ_b w_{j,b} β_{j,b}``; ``cleaned = Σ_j h_j · map2alm(s_j)``. ``weights``
    is ``(J, n_band)`` (global, pixel-constant) or ``(J, n_band, npix)``
    (localized); the ``einsum`` fuses the multiply+reduce so the
    ``(J, n_band, npix)`` product is never materialised — the memory saving for
    the global default at high nside.
    """
    if weights.ndim == 2:
        s = jnp.einsum("jb,jbp->jp", weights, beta)  # global: pixel-constant weights
    else:
        s = jnp.einsum("jbp,jbp->jp", weights, beta)  # localized: per-pixel weights
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


def _gaussian_smooth_map(
    x: jax.Array, fwhm_arcmin: float, *, lmax: int, nside: int, n_iter: int
) -> jax.Array:
    """Spatial Gaussian smoothing of a real map ``x (npix,)`` to FWHM ``fwhm_arcmin``.

    ``map2alm`` (spin-0) → multiply by the Gaussian ``b_ℓ`` → ``synthesis``. Shared by
    the localized-covariance paths of :func:`_localized_weights` (NILC) and the
    constrained-moment NILC (:mod:`augr.cmilc`) / localized GNILC, so there is one
    validated implementation of the per-pixel covariance localization.
    """
    ells = jnp.arange(lmax + 1, dtype=float)
    gauss = beam_bl(ells, fwhm_arcmin)
    alm = map2alm(x[None, :], 0, lmax, nside, n_iter)
    return synthesis(almxfl(alm, gauss, lmax), 0, lmax, nside)[0]


def _beam_bl_np(ells: np.ndarray, fwhm_arcmin: float) -> np.ndarray:
    """NumPy Gaussian beam ``B_ℓ`` -- static-config twin of :func:`instrument.beam_bl`.

    Used only inside :func:`_needlet_channel_mask`, which must stay pure-numpy so
    the cleaner body runs under ``jax.jit`` / ``lax.map`` (the jnp-based
    ``beam_bl`` returns a tracer there). Same formula / ``ARCMIN_TO_RAD`` constant,
    so the values match ``beam_bl`` to fp64.
    """
    sigma = fwhm_arcmin * ARCMIN_TO_RAD / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-ells * (ells + 1.0) * sigma**2 / 2.0)


def _needlet_channel_mask(
    needlet_bands: np.ndarray,
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
    ells = np.arange(lmax + 1, dtype=float)
    bl_common = _beam_bl_np(ells, common_fwhm_arcmin)
    beams = [float(b) for b in beam_fwhm_arcmin]
    n_j = nb.shape[0]
    mask = np.zeros((n_j, len(beams)), dtype=bool)
    for j in range(n_j):
        support = np.nonzero(nb[j] > 1e-3)[0]
        ell_hi = int(support[-1]) if support.size else 0
        bc = max(float(bl_common[ell_hi]), 1e-30)
        for b, fwhm in enumerate(beams):
            bl_band = float(_beam_bl_np(np.array([float(ell_hi)]), fwhm)[0])
            mask[j, b] = (bl_band / bc) >= threshold
    return mask


def _global_weights(beta: jax.Array, ridge: float, active: np.ndarray) -> jax.Array:
    """Spatially-constant weights per needlet band, shape ``(J, n_band)``.

    The empirical covariance and ILC solve run over the *active* channels of each
    needlet band (``active[j]``); excluded channels get weight exactly 0, so their
    (possibly deconvolution-inflated) maps never enter the cleaned product. The
    weights are pixel-independent, so they are returned as ``(J, n_band)`` and
    broadcast at apply time (``combine_needlets``) — *not* materialised to
    ``(J, n_band, npix)``, which would be a redundant ~13 GB at nside=1024.
    """
    n_j, n_band, npix = beta.shape
    ws = []
    for j in range(n_j):
        idx = np.nonzero(active[j])[0]
        bj = beta[j][idx]  # (n_active, npix)
        cov = (bj @ bj.T) / npix  # (n_active, n_active)
        w_active = _ilc_weights_from_cov(_ridge(cov, ridge))  # (n_active,)
        ws.append(jnp.zeros(n_band).at[jnp.asarray(idx)].set(w_active))  # (n_band,)
    return jnp.stack(ws, axis=0)  # (J, n_band)


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

    def smooth(x):  # spatial Gaussian smoothing of a real map (npix,)
        return _gaussian_smooth_map(x, localization_fwhm_arcmin, lmax=lmax, nside=nside, n_iter=n_iter)

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
        ILC weights: ``(J, n_band)`` for the global (pixel-constant) case, or
        ``(J, n_band, npix)`` for localized weights. ``combine_needlets`` /
        :meth:`project` broadcast the global form at apply time.
    needlet_bands
        Cosine-needlet windows used, shape ``(J, lmax+1)``.
    beam_fwhm_arcmin, common_fwhm_arcmin
        Per-band input beams and the common resolution they were brought to.
    lmax, nside, n_iter
        Transform configuration (so passives are projected identically).
    cleaned_e_alm, weights_e
        The spin-2 (Q/U cleaner) extension, populated only when the cleaner is run
        with ``clean_e=True`` (both ``None`` otherwise — the default B-only result is
        unchanged). ``cleaned_e_alm`` is the cleaned E-mode alm at the common
        resolution from an independent E-mode ILC solve (its own empirical
        covariance / constrained system), and ``weights_e`` are the E-mode needlet
        weights (so passive E maps project identically via :meth:`project_e`). With
        both legs present, :meth:`cleaned_qu` builds the cut-sky cleaned Q/U map.
    """

    cleaned_b_alm: jax.Array
    weights: jax.Array
    needlet_bands: jax.Array
    beam_fwhm_arcmin: tuple[float, ...]
    common_fwhm_arcmin: float
    lmax: int
    nside: int
    n_iter: int
    cleaned_e_alm: jax.Array | None = None
    weights_e: jax.Array | None = None

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

    def project_e(self, passive_band_qu: jax.Array) -> jax.Array:
        """Apply the stored E-mode weights to another map set → its cleaned E alm.

        The E-mode companion to :meth:`project`; requires the cleaner to have been run
        with ``clean_e=True`` (``weights_e`` is otherwise ``None``). Used to project
        passive (noise-only / FG-only / CMB-only) E content through the same E ILC.
        """
        if self.weights_e is None:
            raise ValueError(
                "this NILCResult has no E-mode weights (cleaner run without "
                "clean_e=True); cannot project E."
            )
        e_alm, _b_alm, _fwhm = common_resolution_eb(
            passive_band_qu,
            self.beam_fwhm_arcmin,
            lmax=self.lmax,
            nside=self.nside,
            n_iter=self.n_iter,
            common_fwhm_arcmin=self.common_fwhm_arcmin,
        )
        beta = needlet_beta(e_alm, self.needlet_bands, lmax=self.lmax, nside=self.nside)
        return combine_needlets(
            self.weights_e,
            beta,
            self.needlet_bands,
            lmax=self.lmax,
            nside=self.nside,
            n_iter=self.n_iter,
        )

    def cleaned_qu(self) -> jax.Array:
        """Cleaned Q/U map ``(2, npix)`` at the common resolution from cleaned E and B.

        Requires ``clean_e=True`` at clean time. This is the spin-2 cleaned map the
        cut-sky masked-Wiener estimator (:mod:`augr.masking`) consumes. Both E and B
        sit at the common resolution ``B_c``, so the estimator's signal priors must be
        beamed by ``B_c`` (the transfer then absorbs ``B_c²`` on debias).
        """
        if self.cleaned_e_alm is None:
            raise ValueError(
                "this NILCResult has no cleaned E alm (cleaner run without "
                "clean_e=True); cannot build a Q/U map."
            )
        _t, q, u = synthesis_pol(
            jnp.zeros_like(self.cleaned_e_alm),
            self.cleaned_e_alm,
            self.cleaned_b_alm,
            lmax=self.lmax,
            nside=self.nside,
        )
        return jnp.stack([q, u], axis=0)


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
    clean_e: bool = False,
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
    clean_e
        ``False`` (default) → the B-only cleaner, byte-identical to before
        (``cleaned_e_alm`` / ``weights_e`` left ``None``). ``True`` → additionally run
        an *independent* E-mode ILC (its own empirical covariance, same needlet bands
        and active-channel mask) so :meth:`NILCResult.cleaned_qu` can build the cut-sky
        cleaned Q/U map for the masked-Wiener spectrum stage. Roughly doubles the clean
        cost (a second needlet decomposition + weight solve + recompose).

    Returns
    -------
    :class:`NILCResult`.
    """
    check_band_limit(lmax, nside)
    beams = tuple(float(b) for b in beam_fwhm_arcmin)
    if needlet_peaks is None:
        needlet_peaks = default_needlet_peaks(lmax)
    needlet_bands = cosine_needlet_bands(lmax, needlet_peaks)

    if clean_e:
        e_alm, b_alm, common_fwhm = common_resolution_eb(
            band_qu,
            beams,
            lmax=lmax,
            nside=nside,
            n_iter=n_iter,
            common_fwhm_arcmin=common_fwhm_arcmin,
        )
    else:
        e_alm = None
        b_alm, common_fwhm = common_resolution_b_alm(
            band_qu,
            beams,
            lmax=lmax,
            nside=nside,
            n_iter=n_iter,
            common_fwhm_arcmin=common_fwhm_arcmin,
        )

    active = _needlet_channel_mask(needlet_bands, beams, common_fwhm, lmax, beam_band_limit)

    def _weights(beta_field):
        if localization_fwhm_arcmin is None:
            return _global_weights(beta_field, ridge, active)
        return _localized_weights(
            beta_field,
            localization_fwhm_arcmin,
            lmax=lmax,
            nside=nside,
            n_iter=n_iter,
            ridge=ridge,
            active=active,
        )

    beta = needlet_beta(b_alm, needlet_bands, lmax=lmax, nside=nside)
    weights = _weights(beta)
    cleaned = combine_needlets(weights, beta, needlet_bands, lmax=lmax, nside=nside, n_iter=n_iter)

    cleaned_e = None
    weights_e = None
    if clean_e:
        beta_e = needlet_beta(e_alm, needlet_bands, lmax=lmax, nside=nside)
        weights_e = _weights(beta_e)
        cleaned_e = combine_needlets(
            weights_e, beta_e, needlet_bands, lmax=lmax, nside=nside, n_iter=n_iter
        )

    return NILCResult(
        cleaned_b_alm=cleaned,
        weights=weights,
        needlet_bands=needlet_bands,
        beam_fwhm_arcmin=beams,
        common_fwhm_arcmin=common_fwhm,
        lmax=int(lmax),
        nside=int(nside),
        n_iter=int(n_iter),
        cleaned_e_alm=cleaned_e,
        weights_e=weights_e,
    )
