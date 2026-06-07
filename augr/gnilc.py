"""gnilc.py — differentiable generalized needlet ILC (GNILC) foreground template.

A data-driven foreground-residual template for the post-component-separation Fisher
forecast, replacing the *oracle* template :mod:`augr.nilc_forecast` builds by projecting
the true ``fg_qu`` through the NILC weights. GNILC (Remazeilles+ 2011; Carones 2025,
arXiv:2510.20785) estimates the foregrounds *from the data* via a nuisance-whitened
subspace projection, then propagates that estimate through the NILC CMB weights and
noise-debiases (Carones Eq. 3.7).

Algorithm (per needlet band ``j``, over the active channels; matches BROOM ``gilcs.py``)
---------------------------------------------------------------------------------------
1. Total covariance ``R = (β_tot @ β_totᵀ)/npix`` and nuisance covariance
   ``R_n = (β_nuis @ β_nuisᵀ)/npix``, where ``β_nuis`` are the needlet coefficients of
   the **CMB+noise** maps. In a forecast we have those component maps exactly, so ``R_n``
   is formed directly — no separate nuisance simulations.
2. Whiten and eigendecompose ``M = R_n^{-1/2} R R_n^{-1/2}`` → eigenvalues ``λ`` (the
   nuisance/CMB modes sit at ``λ ≈ 1``, the foreground modes at ``λ > 1``).
3. Choose the foreground subspace dimension ``m`` by the eigenvalue AIC
   ``A(m) = 2m + Σ_{k≥m}(λ_k − ln λ_k − 1)`` (Carones Eq. 3.5), in pure JAX (no host
   callback). The discrete ``m`` is frozen for the backward pass via ``stop_gradient`` on
   the selector mask — the data-dependent analog of the static beam-band-limit mask in
   :mod:`augr.nilc`.
4. The GNILC foreground-estimator matrix (BROOM ``W = F(FᵀR⁻¹F)⁻¹FᵀR⁻¹`` with
   ``F = R_n^{1/2}U_s``) collapses for the no-CMB-deprojection case to the fixed-shape
   spectral form ``W = R_n^{1/2} P_s R_n^{-1/2}``, where ``P_s = Σ_{k<m} u_k u_kᵀ`` is the
   orthogonal projector onto the top-``m`` whitened eigenvectors. This is differentiable
   in ``R``/``R_n``/``U`` with no masked-Gram inverse and no dynamic shapes.

Residual template
-----------------
``W`` is composed with the NILC CMB weights ``w`` (from :func:`augr.nilc._global_weights`
on the total). The composition collapses to a per-band vector ``v_j = w_jᵀ W_j``, so the
foreground residual in the cleaned map is ``combine_needlets(v, β)`` — reusing the NILC
recomposition. The template is

    T_res(ℓ) = [ Cℓ(v·β_tot) − Cℓ(v·β_noise) ] / f_sky / B_c²        (Carones Eq. 3.7)

(total minus the noise-only path, beam-deconvolved to the common resolution). Because
``W`` suppresses the nuisance subspace (CMB lives there), ``v·CMB ≈ 0`` and the template
carries foregrounds, like the oracle.

``m_bias`` (verified finding)
-----------------------------
Pure AIC (``m_bias=0``) **under-selects** the foreground dimension for the residual-
template use case: the spatial-SED-variation leak modes — the part of the foreground that
actually survives NILC — sit just above the nuisance eigenvalue floor, so AIC's ``2m``
penalty drops them and the resulting template decouples in shape from the true residual
(verified at nside=64/d1s1: the GNILC/oracle ratio swings ~0.02–0.65 across ℓ). Including
**one extra mode** (``m_bias=1``) recovers them, and the global template then tracks the
oracle (true FG through the NILC weights) to O(1) across ℓ. This matches Carones 2025's
documented ``m_bias=[0,1,1,1]`` (+1 mode at the FG-relevant needlet scales), so the
user-facing defaults here are **``m_bias=1``**.

Scope (v1): **global** per-needlet-band weights (with ``m_bias=1`` this is adequate — the
template is faithful without per-pixel domains); ``depro_cmb`` (explicit CMB deprojection)
deferred — the foreground-subspace projector already suppresses CMB to the ~few-percent
level. B-only, full-sky, same conventions as :mod:`augr.nilc`. Requires the ``[compsep]``
extra (ducc0) for the SHTs.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .instrument import beam_bl
from .nilc import (
    _global_weights,
    _needlet_channel_mask,
    _ridge,
    combine_needlets,
    common_resolution_b_alm,
    cosine_needlet_bands,
    default_needlet_peaks,
    needlet_beta,
)
from .sht import _ell_of_alm, _m_zero_mask, check_band_limit

# ---------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------


def alm2cl(alm: jax.Array, lmax: int) -> jax.Array:
    """Angular power spectrum ``C_ℓ`` of a B-only alm, JAX-differentiable.

    ``C_ℓ = [|a_{ℓ0}|² + 2 Σ_{m>0} |a_{ℓm}|²] / (2ℓ+1)`` in healpy triangular packing —
    the differentiable analog of ``healpy.alm2cl`` for the GNILC residual spectra.
    """
    ell = jnp.asarray(_ell_of_alm(int(lmax)))
    is_m0 = jnp.asarray(_m_zero_mask(int(lmax)))
    power = jnp.where(is_m0, jnp.abs(alm) ** 2, 2.0 * jnp.abs(alm) ** 2)
    cl = jnp.zeros(int(lmax) + 1, dtype=power.dtype).at[ell].add(power)
    return cl / (2.0 * jnp.arange(int(lmax) + 1) + 1.0)


def _matsqrt_pair(cov: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Symmetric ``(cov^{1/2}, cov^{-1/2})`` of a symmetric PD matrix via ``eigh``."""
    s, V = jnp.linalg.eigh(0.5 * (cov + cov.swapaxes(-1, -2)))
    s = jnp.maximum(s, jnp.finfo(cov.dtype).tiny)
    sqrt_s = jnp.sqrt(s)
    half = (V * sqrt_s) @ V.swapaxes(-1, -2)
    ihalf = (V / sqrt_s) @ V.swapaxes(-1, -2)
    return half, ihalf


def _aic_m(lam_desc: jax.Array) -> jax.Array:
    """Eigenvalue-AIC foreground-subspace dimension (Carones Eq. 3.5 / BROOM ``_get_gilc_m``).

    ``A(m) = 2m + Σ_{k≥m}(λ_k − ln λ_k − 1)``, ``m = argmin_m A`` over ``m ∈ [0, n]``.
    Pure JAX (suffix cumulative-sum + ``argmin``), so it traces / jits / vmaps with **no
    host callback** — ``jax.pure_callback`` under ``jax.grad`` deadlocks on some platforms,
    and the callback is unnecessary here: the discrete ``m`` is non-differentiable and is
    frozen downstream by ``stop_gradient`` on the selector mask anyway. ``lam_desc`` is the
    descending eigenvalues, shape ``(..., n)``; returns the integer ``m``, shape ``(...,)``
    (a leading batch is supported for per-pixel localization).
    """
    lam = jnp.maximum(lam_desc, 1e-12)
    n = lam.shape[-1]
    g = lam - jnp.log(lam) - 1.0  # (..., n)
    suffix = jnp.flip(jnp.cumsum(jnp.flip(g, axis=-1), axis=-1), axis=-1)  # Σ_{k≥m}, m=0..n-1
    a_head = 2.0 * jnp.arange(n, dtype=lam.dtype) + suffix  # A(m), m=0..n-1
    a_last = 2.0 * n * jnp.ones((*lam.shape[:-1], 1), dtype=lam.dtype)  # A(n) = 2n
    a_m = jnp.concatenate([a_head, a_last], axis=-1)  # (..., n+1)
    return jnp.argmin(a_m, axis=-1)


def _gnilc_fg_estimator(
    cov_total: jax.Array, cov_nuis: jax.Array, *, m_bias: int = 0, ridge: float = 1e-10
) -> tuple[jax.Array, jax.Array]:
    """GNILC foreground-estimator matrix ``W`` (n×n) and AIC ``m`` for one needlet band.

    ``W = R_n^{1/2} P_s R_n^{-1/2}`` with ``P_s`` the projector onto the top-``m``
    whitened eigenvectors (``m`` from the AIC, frozen for the backward pass). Applied to a
    channel-stacked map it returns the per-channel foreground estimate. Differentiable in
    ``cov_total`` / ``cov_nuis``.
    """
    r = _ridge(cov_total, ridge)
    r_n = _ridge(cov_nuis, ridge)
    cn_half, cn_ihalf = _matsqrt_pair(r_n)

    m_mat = cn_ihalf @ r @ cn_ihalf
    m_mat = 0.5 * (m_mat + m_mat.swapaxes(-1, -2))
    lam, u = jnp.linalg.eigh(m_mat)  # ascending
    lam_desc = lam[::-1]
    u_desc = u[:, ::-1]  # eigenvectors ordered by descending eigenvalue

    n = m_mat.shape[-1]
    m = _aic_m(lam_desc) + int(m_bias)
    # 0/1 selector for the top-m foreground eigenvectors; frozen (discrete selection).
    # m[..., None] keeps this correct under a leading per-pixel batch.
    p = jax.lax.stop_gradient((jnp.arange(n) < m[..., None]).astype(m_mat.dtype))
    proj = (u_desc * p) @ u_desc.swapaxes(-1, -2)  # Σ_{k<m} u_k u_kᵀ
    w = cn_half @ proj @ cn_ihalf
    return w, m


# ---------------------------------------------------------------------------
# GNILC result + builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GNILCResult:
    """GNILC weights for foreground-residual estimation, with the NILC CMB weights.

    Attributes
    ----------
    nilc_weights
        NILC CMB ILC weights ``w`` from the total data, shape ``(J, n_band)``.
    residual_weights
        Composed foreground-residual weights ``v_j = w_jᵀ W_j``, shape ``(J, n_band)``.
        ``combine_needlets(residual_weights, β)`` is the foreground residual in the
        cleaned map.
    needlet_bands
        Cosine-needlet windows, shape ``(J, lmax+1)``.
    beam_fwhm_arcmin, common_fwhm_arcmin
        Per-band input beams and the common resolution.
    lmax, nside, n_iter
        Transform configuration (so passives project identically).
    m_per_band
        AIC foreground-subspace dimension per needlet band (diagnostic), as JAX int
        scalars (concrete in eager use; cast with ``int(...)`` for display).
    """

    nilc_weights: jax.Array
    residual_weights: jax.Array
    needlet_bands: jax.Array
    beam_fwhm_arcmin: tuple[float, ...]
    common_fwhm_arcmin: float
    lmax: int
    nside: int
    n_iter: int
    m_per_band: tuple[jax.Array, ...]

    def _project(self, weights: jax.Array, band_qu: jax.Array) -> jax.Array:
        b_alm, _ = common_resolution_b_alm(
            band_qu,
            self.beam_fwhm_arcmin,
            lmax=self.lmax,
            nside=self.nside,
            n_iter=self.n_iter,
            common_fwhm_arcmin=self.common_fwhm_arcmin,
        )
        beta = needlet_beta(b_alm, self.needlet_bands, lmax=self.lmax, nside=self.nside)
        return combine_needlets(
            weights, beta, self.needlet_bands, lmax=self.lmax, nside=self.nside, n_iter=self.n_iter
        )

    def fg_residual_alm(self, band_qu: jax.Array) -> jax.Array:
        """Foreground-residual B alm: the composed GNILC×NILC weights applied to ``band_qu``."""
        return self._project(self.residual_weights, band_qu)

    def nilc_clean_alm(self, band_qu: jax.Array) -> jax.Array:
        """NILC-cleaned B alm: the CMB weights applied to ``band_qu``.

        Applied to the true ``fg_qu`` this is the *oracle* foreground residual the GNILC
        template approximates; applied to the total it is the NILC CMB map.
        """
        return self._project(self.nilc_weights, band_qu)


def build_gnilc(
    total_qu: jax.Array,
    nuisance_qu: jax.Array,
    beam_fwhm_arcmin,
    *,
    lmax: int,
    nside: int,
    needlet_peaks=None,
    m_bias: int = 1,
    ridge: float = 1e-10,
    n_iter: int = 3,
    beam_band_limit: float = 0.1,
    common_fwhm_arcmin: float | None = None,
) -> GNILCResult:
    """Build GNILC foreground-residual weights from total + nuisance (CMB+noise) maps.

    Parameters mirror :func:`augr.nilc.nilc_clean`; ``nuisance_qu`` is the CMB+noise
    map set whose covariance whitens the GNILC eigenproblem. ``m_bias`` defaults to 1
    (one mode beyond the AIC), which is needed for a faithful residual template and
    matches Carones 2025 — see the module docstring. Pass ``m_bias=0`` for pure AIC.
    """
    check_band_limit(lmax, nside)
    beams = tuple(float(b) for b in beam_fwhm_arcmin)
    if needlet_peaks is None:
        needlet_peaks = default_needlet_peaks(lmax)
    bands = cosine_needlet_bands(lmax, needlet_peaks)

    b_tot, common = common_resolution_b_alm(
        total_qu,
        beams,
        lmax=lmax,
        nside=nside,
        n_iter=n_iter,
        common_fwhm_arcmin=common_fwhm_arcmin,
    )
    b_nuis, _ = common_resolution_b_alm(
        nuisance_qu, beams, lmax=lmax, nside=nside, n_iter=n_iter, common_fwhm_arcmin=common
    )
    beta_tot = needlet_beta(b_tot, bands, lmax=lmax, nside=nside)
    beta_nuis = needlet_beta(b_nuis, bands, lmax=lmax, nside=nside)

    active = _needlet_channel_mask(bands, beams, common, lmax, beam_band_limit)
    w_nilc = _global_weights(beta_tot, ridge, active)  # (J, n_band) NILC CMB weights

    n_j, n_band, npix = beta_tot.shape
    v_rows = []
    m_per_band = []
    for j in range(n_j):
        idx = np.nonzero(active[j])[0]
        idx_j = jnp.asarray(idx)
        bt = beta_tot[j][idx]  # (n_active, npix)
        bn = beta_nuis[j][idx]
        cov_t = (bt @ bt.T) / npix
        cov_n = (bn @ bn.T) / npix
        w_mat, m = _gnilc_fg_estimator(cov_t, cov_n, m_bias=m_bias, ridge=ridge)
        # v_active = w_NILC (active) propagated through the GNILC FG estimator W.
        v_active = w_nilc[j][idx_j] @ w_mat  # (n_active,)
        v_rows.append(jnp.zeros(n_band).at[idx_j].set(v_active))
        # Keep the AIC m as a JAX scalar (no int() — build_gnilc is traced under
        # jax.grad of the residual template); concrete in eager use, cast by callers.
        m_per_band.append(m)
    v = jnp.stack(v_rows, axis=0)  # (J, n_band)

    return GNILCResult(
        nilc_weights=w_nilc,
        residual_weights=v,
        needlet_bands=bands,
        beam_fwhm_arcmin=beams,
        common_fwhm_arcmin=common,
        lmax=int(lmax),
        nside=int(nside),
        n_iter=int(n_iter),
        m_per_band=tuple(m_per_band),
    )


def gnilc_residual_template(
    total_qu: jax.Array,
    cmb_qu: jax.Array,
    noise_qu: jax.Array,
    beam_fwhm_arcmin,
    *,
    lmax: int,
    nside: int,
    needlet_peaks=None,
    m_bias: int = 1,
    ridge: float = 1e-10,
    n_iter: int = 3,
    f_sky: float = 1.0,
    beam_band_limit: float = 0.1,
    common_fwhm_arcmin: float | None = None,
    return_result: bool = False,
):
    """GNILC foreground-residual BB template ``T_res(ℓ)`` for the ``A_res`` Fisher path.

    Builds the GNILC weights with the **CMB+noise** nuisance covariance, propagates the
    foreground estimate through the NILC CMB weights, and noise-debiases (Carones Eq. 3.7):

        T_res(ℓ) = [ Cℓ(v·β_tot) − Cℓ(v·β_noise) ] / f_sky / B_c²

    beam-deconvolved to the common resolution (so it is the beam-free form
    :class:`augr.signal.SignalModel` expects as ``residual_template_cl``).

    Parameters
    ----------
    total_qu, cmb_qu, noise_qu
        Total / CMB-only / noise-only per-band Q/U map sets, shape ``(n_band, 2, npix)``.
        The nuisance covariance uses ``cmb_qu + noise_qu``.
    beam_fwhm_arcmin, lmax, nside, needlet_peaks, m_bias, ridge, n_iter, beam_band_limit,
    common_fwhm_arcmin
        As in :func:`build_gnilc`.
    f_sky
        Sky fraction (1/f_sky correction on the spectra).
    return_result
        If True, also return the :class:`GNILCResult` (for diagnostics / the oracle
        comparison ``result.nilc_clean_alm(fg_qu)``).

    Returns
    -------
    ``(ells, cl_res)`` (and the :class:`GNILCResult` if ``return_result``).
    """
    result = build_gnilc(
        total_qu,
        cmb_qu + noise_qu,
        beam_fwhm_arcmin,
        lmax=lmax,
        nside=nside,
        needlet_peaks=needlet_peaks,
        m_bias=m_bias,
        ridge=ridge,
        n_iter=n_iter,
        beam_band_limit=beam_band_limit,
        common_fwhm_arcmin=common_fwhm_arcmin,
    )

    res_alm = result.fg_residual_alm(total_qu)
    nres_alm = result.fg_residual_alm(noise_qu)
    cl_res = (alm2cl(res_alm, lmax) - alm2cl(nres_alm, lmax)) / f_sky

    ells = jnp.arange(int(lmax) + 1, dtype=float)
    bl2 = jnp.maximum(beam_bl(ells, result.common_fwhm_arcmin) ** 2, 1e-8)
    cl_res = cl_res / bl2

    if return_result:
        return ells, cl_res, result
    return ells, cl_res
