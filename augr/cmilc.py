"""cmilc.py — constrained-moment needlet ILC (cMILC) for B-mode component separation.

cMILC (Remazeilles, Rotti & Chluba 2021, arXiv:2006.08628) generalizes the blind NILC
of :mod:`augr.nilc` by *deprojecting* foreground moment SEDs. Where NILC enforces the
single CMB constraint ``aᵀw = 1`` and minimizes the empirical variance, cMILC enforces a
multi-constraint system

    minimize  wᵀ C w     subject to   Aᵀ w = e ,
    A = [a_cmb | f_dust | f_sync | ∂_β f_dust | ∂_β f_sync | ∂_T f_dust | ...] ,
    e = [1, 0, 0, ...]ᵀ ,

whose closed form is

    w = C⁻¹ A (Aᵀ C⁻¹ A)⁻¹ e .

By construction ``Aᵀw = e``: the CMB is preserved (transfer 1) and every listed
foreground moment SED is nulled. NILC is the special case ``A = a_cmb`` (a column of
ones — the CMB SED is flat in CMB thermodynamic units), recovering
:func:`augr.nilc._ilc_weights_from_cov`.

Why moments. A spatially-varying dust spectral index β_dust(n̂) is, via the Chluba+ 2017
moment expansion, algebraically an extra component whose SED is the derivative
``∂I_ν/∂β`` of the baseline modified blackbody. That moment is exactly the residual a
blind NILC leaves on a real (varying-β) sky — the dust-dominated, aperture-dependent
r-bias seen in the diff-NILC sweep. Deprojecting ``∂_β f_dust`` removes it explicitly,
and *which* moment must be nulled is interpretable (the cross-check on the data-driven
GNILC of :mod:`augr.gnilc`).

Cost. Each deprojected column is one fewer degree of freedom for variance minimization (a
noise penalty), so a band needs more active channels than constraints to retain any
freedom. The default :data:`CMILC08_MOMENTS` deprojects the zeroth-order dust + sync and
the three leading first-order moments {∂_β dust, ∂_β sync, ∂_T dust} — the PICO-optimal
trade-off in Remazeilles+ 2021 Table 1 for wide-frequency-coverage missions. The
assembler is configurable, so cMILC06 (:data:`CMILC06_MOMENTS`) or higher-order sets are
a ``moments=`` change.

Implementation. cMILC is a drop-in for NILC: the same common-resolution → needlet → weight
→ recompose pipeline, returning a :class:`augr.nilc.NILCResult`, so the forecast consumers
(:func:`augr.nilc_forecast.nilc_spectra` / ``nilc_forecast`` / ``NILCResult.project``)
work unchanged. The only new inputs vs ``nilc_clean`` are the band-center ``freqs`` (cMILC
is *not* blind — it needs the SEDs) and the fiducial pivots. Moment SED columns reuse the
validated :mod:`augr.units` primitives. B-only, full-sky, ducc0 ([compsep] extra) for the
SHTs.

Active-channel handling. The beam-band-limit mask (shared with NILC) can leave a fine
needlet band with fewer active channels ``n_act`` than constraints ``k`` → ``AᵀC⁻¹A``
singular. That band then uses the **leading ``min(k, n_act)`` columns** of ``A`` (the CMB
constraint is column 0 and is always kept; ``moments`` is ordered low→high Taylor order,
so the highest-order moment drops first — the simplest form of ocMILC's locally-varying
moment count). The retained-column count per band is available via
``cmilc_clean(..., return_diagnostics=True)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import units
from .bandpass import Bandpass
from .config import FIDUCIAL_BK15
from .nilc import (
    NILCResult,
    _gaussian_smooth_map,
    _needlet_channel_mask,
    _ridge,
    combine_needlets,
    common_resolution_b_alm,
    common_resolution_eb,
    cosine_needlet_bands,
    default_needlet_peaks,
    needlet_beta,
)
from .sht import check_band_limit

# Default cMILC08 deprojection set (PICO-optimal; Remazeilles+ 2021 Table 1): zeroth-order
# dust + sync, plus the three leading first-order moments.
CMILC08_MOMENTS: tuple[str, ...] = ("f_dust", "f_sync", "dbeta_dust", "dbeta_sync", "dT_dust")

# Minimal cMILC06: zeroth-order dust + sync + the dust β moment — the cheapest set that
# still nulls the headline spatial-β_dust moment.
CMILC06_MOMENTS: tuple[str, ...] = ("f_dust", "f_sync", "dbeta_dust")


# ---------------------------------------------------------------------------
# moment SED mixing matrix
# ---------------------------------------------------------------------------


def _moment_columns(
    nu: jax.Array,
    fiducial: Mapping[str, float],
    bandpasses: Sequence[Bandpass | None] | None = None,
) -> dict[str, jax.Array]:
    """Moment SED columns at frequencies ``nu`` (GHz), in CMB thermodynamic units.

    Each column is the baseline SED times the corresponding :mod:`augr.units`
    log-derivative (Chluba+ 2017 / Remazeilles+ 2021 moment spectral functions). The
    per-column normalization is immaterial to the deprojection (which fixes only each
    column's *span*), so the convenient ``units`` normalization (SEDs = 1 at ν_ref) is used
    directly — and keeps a single in-repo source of truth for the SED forms.

    If ``bandpasses`` is given (one ``Bandpass`` per band, ``None`` entries = the
    monochromatic band-center limit), each *whole* moment SED is band-averaged over its
    bandpass via :func:`augr.units.color_correct`, so cMILC deprojects the effective
    (color-corrected) SED that the bandpass-integrated sky actually presents. The whole
    moment SED ``f_X·∂ln f_X`` is averaged, not the factors separately. ``bandpasses=None``
    (default) keeps the original vectorized monochromatic columns byte-for-byte.
    """
    bd = float(fiducial["beta_dust"])
    td = float(fiducial["T_dust"])
    bs = float(fiducial["beta_sync"])
    if bandpasses is None:
        f_dust = units.dust_sed(nu, bd, td)
        f_sync = units.sync_sed(nu, bs)
        return {
            "f_dust": f_dust,  # zeroth-order dust (modified blackbody)
            "f_sync": f_sync,  # zeroth-order sync (power law)
            "dbeta_dust": f_dust * units.dust_sed_deriv_beta(nu),  # ∂_β f_dust = f_dust·ln(ν/ν_d)
            "dbeta_sync": f_sync * units.sync_sed_deriv_beta(nu),  # ∂_β f_sync = f_sync·ln(ν/ν_s)
            "dT_dust": f_dust * units.dust_sed_deriv_T(nu, td),  # ∂_T f_dust
        }

    # Bandpass-integrated: band-average each whole moment SED over its bandpass.
    sed_fns = {
        "f_dust": lambda v: units.dust_sed(v, bd, td),
        "f_sync": lambda v: units.sync_sed(v, bs),
        "dbeta_dust": lambda v: units.dust_sed(v, bd, td) * units.dust_sed_deriv_beta(v),
        "dbeta_sync": lambda v: units.sync_sed(v, bs) * units.sync_sed_deriv_beta(v),
        "dT_dust": lambda v: units.dust_sed(v, bd, td) * units.dust_sed_deriv_T(v, td),
    }
    band_bps = [
        bp if bp is not None else Bandpass.monochromatic(nu[i])
        for i, bp in enumerate(bandpasses)
    ]
    return {
        key: jnp.stack([units.color_correct(fn, bp) for bp in band_bps])
        for key, fn in sed_fns.items()
    }


def moment_sed_vectors(
    freqs: Sequence[float],
    *,
    fiducial: Mapping[str, float] = FIDUCIAL_BK15,
    moments: Sequence[str] = CMILC08_MOMENTS,
    bandpasses: Sequence[Bandpass | None] | None = None,
) -> jax.Array:
    """Constrained-ILC mixing matrix ``A``, shape ``(n_band, 1 + len(moments))``.

    Column 0 is the CMB constraint (ones — the CMB SED is flat in CMB thermodynamic units,
    so cMILC preserves the CMB exactly, like NILC's ``a = 1``). The remaining columns are
    the requested foreground-moment SEDs (see :func:`_moment_columns`) at each band-center
    frequency, ordered as ``moments``. Pivots default to ``FIDUCIAL_BK15`` (β̄_d=1.6,
    T̄_d=19.6, β̄_s=−3.1); reference frequencies from ``units.NU_{DUST,SYNC}_REF_GHZ``.

    If ``bandpasses`` (one ``Bandpass`` per band, ``None`` = monochromatic) is supplied,
    the moment SED columns are band-averaged (color-corrected) over each bandpass so they
    match the bandpass-integrated sky; the CMB column stays flat (a flat SED band-averages
    to 1). ``bandpasses=None`` is the original monochromatic behavior, byte-for-byte.

    The returned ``A`` is constant in the data, so the cMILC weights are differentiable in
    the data covariance (the maps); ``A`` is itself differentiable in ``freqs`` / pivots /
    bandpass band centers + widths if those are passed as traced arrays (use
    ``Bandpass.smooth_tophat`` for clean band-center/bandwidth gradients).
    """
    nu = jnp.asarray(freqs, dtype=float)
    if bandpasses is not None and len(bandpasses) != nu.shape[0]:
        raise ValueError(
            f"bandpasses (len {len(bandpasses)}) must match freqs (len {nu.shape[0]})."
        )
    cols = _moment_columns(nu, fiducial, bandpasses)
    unknown = [m for m in moments if m not in cols]
    if unknown:
        raise ValueError(f"unknown moment key(s) {unknown}; available: {sorted(cols)}")
    columns = [jnp.ones_like(nu)] + [cols[m] for m in moments]
    return jnp.stack(columns, axis=1)


# ---------------------------------------------------------------------------
# constrained-ILC weights
# ---------------------------------------------------------------------------


def _cilc_weights_from_cov(cov: jax.Array, A: jax.Array, e: jax.Array) -> jax.Array:
    """Constrained-ILC weights ``w = C⁻¹ A (Aᵀ C⁻¹ A)⁻¹ e``.

    ``cov`` is ``(..., n, n)``, ``A`` is ``(n, k)``, ``e`` is ``(k,)``; returns
    ``(..., n)``. ``Aᵀ w = e`` holds exactly. Reduces to
    :func:`augr.nilc._ilc_weights_from_cov` when ``A = ones((n, 1))`` and ``e = [1]``.
    Batches over a leading pixel dimension (the localized path) like the ``a = 1`` form.
    """
    n, k = A.shape
    batch = cov.shape[:-2]
    a_b = jnp.broadcast_to(A, (*batch, n, k))
    cia = jnp.linalg.solve(cov, a_b)  # (..., n, k) = C⁻¹ A
    atcia = jnp.einsum("...nk,...nl->...kl", a_b, cia)  # (..., k, k) = Aᵀ C⁻¹ A
    e_b = jnp.broadcast_to(e, (*batch, k))
    x = jnp.linalg.solve(atcia, e_b[..., None])[..., 0]  # (..., k) = (Aᵀ C⁻¹ A)⁻¹ e
    return jnp.einsum("...nk,...k->...n", cia, x)  # (..., n)


def _retained_k(k_full: int, n_act: int) -> int:
    """Constraints to keep for a band with ``n_act`` active channels: the CMB column plus
    as many moments as fit, highest-order dropped first. Always ≥ 1 (CMB)."""
    return max(1, min(k_full, n_act))


def _global_cilc_weights(beta, A, e, ridge, active):
    """Spatially-constant cMILC weights per needlet band, shape ``(J, n_band)``.

    Mirrors :func:`augr.nilc._global_weights` but solves the constrained system over the
    active channels of each band, sub-selecting both the covariance and the SED rows
    (``A[idx]``). Returns ``(weights, retained_columns_per_band)``.
    """
    n_j, n_band, npix = beta.shape
    k_full = A.shape[1]
    ws = []
    cols = []
    for j in range(n_j):
        idx = np.nonzero(active[j])[0]
        k = _retained_k(k_full, len(idx))
        a_j = A[jnp.asarray(idx)][:, :k]
        e_j = e[:k]
        bj = beta[j][idx]  # (n_act, npix)
        cov = (bj @ bj.T) / npix
        w_act = _cilc_weights_from_cov(_ridge(cov, ridge), a_j, e_j)  # (n_act,)
        ws.append(jnp.zeros(n_band).at[jnp.asarray(idx)].set(w_act))
        cols.append(k)
    return jnp.stack(ws, axis=0), tuple(cols)


def _localized_cilc_weights(
    beta, A, e, localization_fwhm_arcmin, *, lmax, nside, n_iter, ridge, active
):
    """Per-pixel cMILC weights from a Gaussian-localized covariance, ``(J, n_band, npix)``.

    Mirrors :func:`augr.nilc._localized_weights` (shared
    :func:`augr.nilc._gaussian_smooth_map`) with the constrained solve. Returns
    ``(weights, retained_columns_per_band)``.
    """
    n_j, n_band, _npix = beta.shape
    k_full = A.shape[1]
    ws = []
    cols = []
    for j in range(n_j):
        idx = np.nonzero(active[j])[0]
        n_act = len(idx)
        k = _retained_k(k_full, n_act)
        a_j = A[jnp.asarray(idx)][:, :k]
        e_j = e[:k]
        bj = beta[j][idx]  # (n_act, npix)
        rows = [
            jnp.stack(
                [
                    _gaussian_smooth_map(
                        bj[i] * bj[m],
                        localization_fwhm_arcmin,
                        lmax=lmax,
                        nside=nside,
                        n_iter=n_iter,
                    )
                    for m in range(n_act)
                ],
                axis=0,
            )
            for i in range(n_act)
        ]
        cov = jnp.moveaxis(jnp.stack(rows, axis=0), 2, 0)  # (npix, n_act, n_act)
        w_act = _cilc_weights_from_cov(_ridge(cov, ridge), a_j, e_j)  # (npix, n_act)
        w = jnp.zeros((n_band, _npix)).at[jnp.asarray(idx)].set(w_act.T)
        ws.append(w)
        cols.append(k)
    return jnp.stack(ws, axis=0), tuple(cols)


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------


def cmilc_clean(
    band_qu,
    beam_fwhm_arcmin,
    freqs,
    *,
    lmax,
    nside,
    moments=CMILC08_MOMENTS,
    fiducial=FIDUCIAL_BK15,
    bandpasses=None,
    needlet_peaks=None,
    localization_fwhm_arcmin=None,
    common_fwhm_arcmin=None,
    n_iter=3,
    ridge=1e-10,
    beam_band_limit=0.1,
    clean_e=False,
    return_diagnostics=False,
):
    """Constrained-moment needlet ILC on per-band Q/U maps → :class:`augr.nilc.NILCResult`.

    Same pipeline and return type as :func:`augr.nilc.nilc_clean` (so the forecast
    consumers work unchanged) but with the cMILC weight solve. cMILC is *not* blind, so it
    needs the band-center ``freqs`` (GHz) and the fiducial spectral pivots in addition to
    the NILC arguments.

    Parameters
    ----------
    band_qu, beam_fwhm_arcmin, lmax, nside, needlet_peaks, localization_fwhm_arcmin,
    common_fwhm_arcmin, n_iter, ridge, beam_band_limit
        As in :func:`augr.nilc.nilc_clean`.
    freqs
        Band-center frequencies [GHz], length ``n_band`` — used to build the moment SEDs.
    moments
        Ordered foreground moment-SED keys to deproject (default :data:`CMILC08_MOMENTS`).
        See :func:`_moment_columns` for the available keys.
    fiducial
        Fiducial spectral parameters (pivots) for the moment SEDs (default
        ``FIDUCIAL_BK15``).
    bandpasses
        Optional per-band ``Bandpass`` (one per ``freqs`` entry, ``None`` =
        monochromatic). When given, the moment SED columns are band-averaged
        (color-corrected) to match a bandpass-integrated sky. ``None`` (default)
        is the monochromatic band-center behavior, byte-identical to before.
    clean_e
        ``False`` (default) → B-only, byte-identical to before. ``True`` → additionally
        run an independent constrained-ILC solve on the E modes (same mixing matrix
        ``A`` — moment SEDs are frequency scalings, identical for E and B — and the same
        active-channel mask) so :meth:`augr.nilc.NILCResult.cleaned_qu` can build the
        cut-sky cleaned Q/U map for the masked-Wiener spectrum stage.
    return_diagnostics
        If True, return ``(result, info)`` where ``info`` carries the mixing matrix, the
        moment list, the constraint count, and the per-band retained-column count (the
        active-channel degradation diagnostic).

    Returns
    -------
    :class:`augr.nilc.NILCResult`, or ``(NILCResult, dict)`` if ``return_diagnostics``.
    """
    check_band_limit(lmax, nside)
    beams = tuple(float(b) for b in beam_fwhm_arcmin)
    freqs = tuple(float(f) for f in freqs)
    if len(freqs) != len(beams):
        raise ValueError(
            f"freqs (len {len(freqs)}) and beam_fwhm_arcmin (len {len(beams)}) must both be n_band."
        )
    if bandpasses is not None and len(bandpasses) != len(freqs):
        raise ValueError(
            f"bandpasses (len {len(bandpasses)}) must match freqs (len {len(freqs)})."
        )
    if needlet_peaks is None:
        needlet_peaks = default_needlet_peaks(lmax)
    needlet_bands = cosine_needlet_bands(lmax, needlet_peaks)

    if clean_e:
        e_alm, b_alm, common_fwhm = common_resolution_eb(
            band_qu, beams, lmax=lmax, nside=nside, n_iter=n_iter, common_fwhm_arcmin=common_fwhm_arcmin
        )
    else:
        e_alm = None
        b_alm, common_fwhm = common_resolution_b_alm(
            band_qu, beams, lmax=lmax, nside=nside, n_iter=n_iter, common_fwhm_arcmin=common_fwhm_arcmin
        )
    active = _needlet_channel_mask(needlet_bands, beams, common_fwhm, lmax, beam_band_limit)

    A = moment_sed_vectors(
        freqs, fiducial=fiducial, moments=moments, bandpasses=bandpasses
    )  # (n_band, 1 + n_moments)
    e = jnp.zeros(A.shape[1]).at[0].set(1.0)

    def _cilc_weights(beta_field):
        if localization_fwhm_arcmin is None:
            return _global_cilc_weights(beta_field, A, e, ridge, active)
        return _localized_cilc_weights(
            beta_field,
            A,
            e,
            localization_fwhm_arcmin,
            lmax=lmax,
            nside=nside,
            n_iter=n_iter,
            ridge=ridge,
            active=active,
        )

    beta = needlet_beta(b_alm, needlet_bands, lmax=lmax, nside=nside)
    weights, cols = _cilc_weights(beta)
    cleaned = combine_needlets(weights, beta, needlet_bands, lmax=lmax, nside=nside, n_iter=n_iter)

    cleaned_e = None
    weights_e = None
    if clean_e:
        beta_e = needlet_beta(e_alm, needlet_bands, lmax=lmax, nside=nside)
        weights_e, _cols_e = _cilc_weights(beta_e)
        cleaned_e = combine_needlets(
            weights_e, beta_e, needlet_bands, lmax=lmax, nside=nside, n_iter=n_iter
        )

    result = NILCResult(
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
    if return_diagnostics:
        info = {
            "mixing_matrix": A,
            "moments": tuple(moments),
            "n_constraints": int(A.shape[1]),
            "retained_columns_per_band": cols,
        }
        return result, info
    return result
