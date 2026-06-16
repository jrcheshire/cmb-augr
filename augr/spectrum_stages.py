"""spectrum_stages.py — cut-sky masked-Wiener Monte-Carlo spectrum stage.

The full-sky in-house path (:func:`augr.nilc_forecast.nilc_spectra`) forms cleaned
B-mode spectra with ``alm2cl`` and a scalar ``1/f_sky`` correction — it does not model
realistic sky coverage, where the spin-2 E/B split is ambiguous and the leaked-E
*cosmic variance* inflates the B error bars by a mask-dependent factor. This module is
the cut-sky replacement: a Monte-Carlo ensemble that, per sim,

1. builds a per-band sky (CMB E+B + optional PySM FG) + noise,
2. runs a **spin-2 Q/U cleaner** (``clean_e=True``) → cleaned Q/U map at the common
   resolution ``B_c``,
3. estimates the cut-sky ``C_ℓ^{BB}`` with the masked-Wiener filter
   (:func:`augr.masking.masked_wiener_bb`) and bins it,
4. debiases (multiplicative transfer ``F_b`` + additive E→B leakage),

and returns the per-sim debiased bandpowers plus their Monte-Carlo covariance
(:func:`augr.covariance.mc_bandpower_covariance`) for
``FisherForecast(external_covariance=...)``.

Transfer / leakage via projection (not re-cleaning)
---------------------------------------------------
The multiplicative transfer ``F_b`` (filter suppression of true B) and the additive
E→B leakage template come from **projecting the CMB component of each sim through that
sim's own cleaner weights**, not from separately re-cleaning B-only / E-only sims. For
a *blind* ILC the weights are realization-dependent (derived from the full data), so
re-cleaning a foreground-free B-only sim would solve a different (noise-only)
covariance and measure the transfer of a differently-weighted map. Projecting the CMB
component through the matched full-sim weights — exactly the decomposition
:func:`augr.nilc_forecast.nilc_spectra` already uses for ``nl_post`` / ``cl_residual_fg``
— is self-consistent and keeps ``F_b`` / leakage free of noise and FG-residual
contamination. On the full sky a pure-E map carries no B and vice versa, so
``result.project`` (B) and ``result.project_e`` (E) cleanly separate the cleaned CMB-B
(transfer) from the cleaned CMB-E (leakage) out of the single ``cmb_qu`` band set.

Because the leakage enters the debias as a per-bin *additive constant* and the transfer
as a per-bin *multiplicative rescale*, neither shifts the **covariance** beyond the
deterministic ``Cov_debiased = diag(1/F) · Cov(C_rec) · diag(1/F)`` — they set the data
vector, while the σ(r) width is carried by ``Cov(C_rec)`` rescaled onto the true,
beam-free bandpower scale (matching the Fisher Jacobian).

Conventions follow :mod:`augr.masking`: galactic-frame maps, sharp ``|b|``-cut mask
folded into ``inv_noise``, signal priors fixed at r=0 (lensed EE / lensing BB) and
beamed by ``B_c`` so the transfer absorbs ``B_c²`` on debias.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import masking as mk
from .cleaning import Cleaner
from .compsep_sims import BandSky, assemble_band_maps, beam_harmonic_sky, harmonic_sky
from .covariance import mc_bandpower_covariance
from .instrument import beam_bl
from .parallel import parallel_map
from .sht import synthesis_pol
from .spectra import CMBSpectra


@dataclass(frozen=True)
class CutskyMC:
    """Output of :func:`mc_cutsky_bandpowers`: debiased bandpowers + MC covariance.

    Attributes
    ----------
    debiased_bandpowers
        Per-sim debiased binned ``C_b^{BB}`` (beam-free, true scale), ``(n_sims,
        n_bins)`` — the ensemble whose sample covariance is :attr:`covariance` and
        whose mean is :attr:`mean_bandpower`.
    covariance
        Hartlap-corrected MC covariance ``(n_bins, n_bins)`` for
        ``FisherForecast(external_covariance=...)``.
    transfer, leakage
        The debias pieces: multiplicative transfer ``F_b`` (from the cleaned CMB-B
        through the masked-Wiener filter) and additive E→B leakage template ``leak_b``
        (from the cleaned CMB-E), each ``(n_bins,)``.
    mean_bandpower
        Ensemble-mean debiased bandpower ``(n_bins,)`` (the data vector; not used by
        the σ(r) width, which depends only on the Jacobian and covariance).
    f_sky
        Realized sky fraction ``⟨mask⟩``.
    n_sims, var_pix_ref
        Ensemble size and the per-pixel noise variance used in the Wiener filter's
        ``inv_noise`` (a filter knob; a mismatch only de-tunes the filter, absorbed by
        ``F_b``).
    """

    debiased_bandpowers: np.ndarray
    covariance: np.ndarray
    transfer: np.ndarray
    leakage: np.ndarray
    mean_bandpower: np.ndarray
    f_sky: float
    n_sims: int
    var_pix_ref: float


def beamed_prior(cl: jax.Array, common_fwhm_arcmin: float, lmax: int) -> jax.Array:
    """Beam a beam-free signal prior by the common-resolution beam ``B_c²``.

    The masked-Wiener priors describe the *data* (beamed) covariance, so the lensed-EE
    / lensing-BB priors are multiplied by ``beam_bl(ℓ, B_c)²``; the transfer ``F_b``
    then absorbs the ``B_c²`` and debias recovers the beam-free ``C_ℓ``.
    """
    bl = beam_bl(jnp.arange(int(lmax) + 1, dtype=float), float(common_fwhm_arcmin))
    return jnp.clip(jnp.asarray(cl)[: int(lmax) + 1], 0.0, None) * bl**2


def cutsky_bb_bandpower(
    cleaned_qu: jax.Array,
    inv_noise: jax.Array,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    bin_matrix: jax.Array,
    ell_min: int,
    nside: int,
    lmax: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> jax.Array:
    """Raw binned ``C_b^{BB}`` of one cleaned Q/U map via the masked-Wiener estimator.

    Thin compose of :func:`augr.masking.masked_wiener_bb` (filter-suppressed,
    leakage-contaminated per-ℓ B power) and :func:`augr.masking.bin_spectrum`. The
    debias (:func:`augr.masking.debias_bandpower`) is applied at the ensemble level by
    :func:`mc_cutsky_bandpowers`.
    """
    cl = mk.masked_wiener_bb(
        cleaned_qu,
        inv_noise,
        cl_ee_prior,
        cl_bb_prior,
        nside=int(nside),
        lmax=int(lmax),
        max_iter=max_iter,
        tol=tol,
    )
    return mk.bin_spectrum(cl, bin_matrix, ell_min)


def _cleaned_b_qu(result, band_qu: jax.Array) -> jax.Array:
    """Cleaned **B-only** Q/U from projecting ``band_qu`` through the B weights."""
    b_alm = result.project(band_qu)
    _t, q, u = synthesis_pol(
        jnp.zeros_like(b_alm),
        jnp.zeros_like(b_alm),
        b_alm,
        lmax=result.lmax,
        nside=result.nside,
    )
    return jnp.stack([q, u], axis=0)


def _cleaned_e_qu(result, band_qu: jax.Array) -> jax.Array:
    """Cleaned **E-only** Q/U from projecting ``band_qu`` through the E weights."""
    e_alm = result.project_e(band_qu)
    _t, q, u = synthesis_pol(
        jnp.zeros_like(e_alm),
        e_alm,
        jnp.zeros_like(e_alm),
        lmax=result.lmax,
        nside=result.nside,
    )
    return jnp.stack([q, u], axis=0)


def _build_sim(
    seed: int,
    *,
    freqs_ghz,
    beam_fwhm_arcmin,
    w_inv,
    nside,
    lmax,
    fg_model,
    r_in,
    cl_ee,
    spectra,
    hit_map,
    knee_ell,
    alpha_knee,
    bandpasses=None,
):
    """One sim's beamed sky + total band maps. Returns ``(sky, total)``."""
    hsky = harmonic_sky(
        tuple(freqs_ghz),
        spectra=spectra,
        r_in=float(r_in),
        nside=int(nside),
        lmax=int(lmax),
        fg_model=fg_model,
        cmb_seed=int(seed),
        cl_ee=cl_ee,
        bandpasses=bandpasses,
    )
    sky = beam_harmonic_sky(hsky, tuple(beam_fwhm_arcmin))
    total = assemble_band_maps(
        sky,
        jnp.asarray(w_inv),
        hit_map,
        noise_key=jax.random.PRNGKey(int(seed)),
        knee_ell=knee_ell,
        alpha_knee=alpha_knee,
    )
    return sky, total


def mc_cutsky_bandpowers(
    *,
    cleaner: Cleaner,
    freqs_ghz,
    beam_fwhm_arcmin,
    w_inv,
    nside: int,
    lmax: int,
    mask: jax.Array,
    cl_ee: jax.Array,
    cl_bb_prior_unbeamed: jax.Array,
    bin_matrix: jax.Array,
    ell_min: int,
    true_bb_binned: jax.Array,
    n_sims: int,
    base_seed: int = 0,
    fg_model: str | None = "d1s1",
    r_in: float = 0.0,
    hit_map: jax.Array | None = None,
    var_pix_ref: float | None = None,
    knee_ell: jax.Array | None = None,
    alpha_knee: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
    workers: int = 1,
    spectra: CMBSpectra | None = None,
    bandpasses=None,
) -> CutskyMC:
    """Run the cut-sky masked-Wiener MC ensemble → debiased bandpowers + covariance.

    The ``cleaner`` MUST be a spin-2 Q/U cleaner (built with ``clean_e=True``, e.g.
    ``nilc_cleaner(clean_e=True)``); each sim is cleaned once and the data / transfer /
    leakage legs are projected from that single result (see module docstring).

    Parameters
    ----------
    cleaner
        A spin-2 :class:`augr.cleaning.Cleaner` (``clean_e=True``).
    freqs_ghz, beam_fwhm_arcmin, w_inv
        Per-band centers [GHz], beams [arcmin], white-noise power [μK²·sr].
    nside, lmax
        HEALPix resolution and band limit.
    mask
        Binary/weight mask ``(npix,)`` (folded into ``inv_noise``).
    cl_ee
        Lensed EE on ℓ=0..lmax — the source of the CMB E realization AND (beamed by
        ``B_c``) the masked-Wiener EE prior. Use
        ``delensing.load_lensing_spectra().cl_ee_len``.
    cl_bb_prior_unbeamed
        Lensing BB on ℓ=0..lmax — the masked-Wiener BB prior at r=0 (beamed by ``B_c``).
    bin_matrix, ell_min
        Binning from the forecast ``SignalModel`` (so the MC bins match the Fisher bins
        exactly). ``bin_matrix`` is ``(n_bins, n_ells)`` over ``[ell_min, ell_max]``.
    true_bb_binned
        Binned input ``C_b^{BB}`` (beam-free) at ``r_in`` — the transfer denominator.
    n_sims, base_seed
        Ensemble size and the first CRN seed (sims use ``base_seed .. base_seed +
        n_sims - 1``; the ``var_pix_ref`` setup clean uses ``base_seed + n_sims``).
    fg_model, r_in
        PySM preset (``None`` = CMB-only) and the input tensor ratio of the sims.
    hit_map
        Relative exposure ``(npix,)``; ``None`` → uniform.
    var_pix_ref
        Wiener-filter per-pixel noise variance [μK²]. ``None`` → estimated from a
        representative cleaned-noise map (recommended). A filter knob only: a mismatch
        de-tunes the filter (absorbed by ``F_b``), it does not bias the result.
    knee_ell, alpha_knee
        Optional 1/f noise (``None`` → white).
    workers
        Process-pool workers for the ensemble (``1`` → serial; >1 requires a picklable
        cleaner — local closures from the factories are not picklable, so use serial or
        a module-level cleaner).
    spectra
        ``CMBSpectra`` provider (default: a fresh ``CMBSpectra()``).
    bandpasses
        Optional per-band ``Bandpass`` to bandpass-integrate the foreground sims
        (match the cleaner's color-corrected SEDs); ``None`` → monochromatic.
    """
    spectra = CMBSpectra() if spectra is None else spectra
    freqs_ghz = tuple(float(f) for f in freqs_ghz)
    beam_fwhm_arcmin = tuple(float(b) for b in beam_fwhm_arcmin)
    npix = 12 * int(nside) ** 2
    hit_map = jnp.ones(npix) if hit_map is None else jnp.asarray(hit_map)
    common_fwhm = float(min(beam_fwhm_arcmin))
    cl_ee_prior = beamed_prior(cl_ee, common_fwhm, lmax)
    cl_bb_prior = beamed_prior(cl_bb_prior_unbeamed, common_fwhm, lmax)

    sim_kw = dict(
        freqs_ghz=freqs_ghz,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
        w_inv=w_inv,
        nside=nside,
        lmax=lmax,
        fg_model=fg_model,
        r_in=r_in,
        cl_ee=cl_ee,
        spectra=spectra,
        hit_map=hit_map,
        knee_ell=knee_ell,
        alpha_knee=alpha_knee,
        bandpasses=bandpasses,
    )

    # var_pix_ref: per-pixel noise variance of a representative cleaned map (filter
    # knob). Derived from one setup clean's cleaned-noise map over observed pixels.
    if var_pix_ref is None:
        sky0, total0 = _build_sim(base_seed + n_sims, **sim_kw)
        res0 = cleaner(total0, beam_fwhm_arcmin, lmax=lmax, nside=nside)
        noise0 = total0 - sky0.cmb_qu - sky0.fg_qu
        cn = _cleaned_b_qu(res0, noise0) + _cleaned_e_qu(res0, noise0)
        obs = jnp.asarray(mask) > 0
        var_pix_ref = float(jnp.mean((cn[0] ** 2 + cn[1] ** 2)[obs]) / 2.0)

    inv_noise = mk.inv_noise_map(hit_map, var_pix_ref, mask=mask)

    bp_kw = dict(
        bin_matrix=jnp.asarray(bin_matrix),
        ell_min=int(ell_min),
        nside=int(nside),
        lmax=int(lmax),
        max_iter=max_iter,
        tol=tol,
    )

    def _one(seed: int):
        sky, total = _build_sim(seed, **sim_kw)
        result = cleaner(total, beam_fwhm_arcmin, lmax=lmax, nside=nside)
        rec_full = cutsky_bb_bandpower(
            result.cleaned_qu(), inv_noise, cl_ee_prior, cl_bb_prior, **bp_kw
        )
        rec_b = cutsky_bb_bandpower(
            _cleaned_b_qu(result, sky.cmb_qu), inv_noise, cl_ee_prior, cl_bb_prior, **bp_kw
        )
        rec_e = cutsky_bb_bandpower(
            _cleaned_e_qu(result, sky.cmb_qu), inv_noise, cl_ee_prior, cl_bb_prior, **bp_kw
        )
        return np.asarray(rec_full), np.asarray(rec_b), np.asarray(rec_e)

    seeds = list(range(int(base_seed), int(base_seed) + int(n_sims)))
    out = parallel_map(_one, seeds, workers=workers)
    rec_full = jnp.asarray(np.stack([o[0] for o in out], axis=0))
    rec_b = jnp.asarray(np.stack([o[1] for o in out], axis=0))
    rec_e = jnp.asarray(np.stack([o[2] for o in out], axis=0))

    transfer = mk.transfer_function(rec_b, jnp.asarray(true_bb_binned))
    leakage = mk.leakage_template(rec_e)
    debiased = mk.debias_bandpower(rec_full, transfer, leakage)
    cov = mc_bandpower_covariance(debiased, hartlap=True)

    return CutskyMC(
        debiased_bandpowers=np.asarray(debiased),
        covariance=np.asarray(cov),
        transfer=np.asarray(transfer),
        leakage=np.asarray(leakage),
        mean_bandpower=np.asarray(jnp.mean(debiased, axis=0)),
        f_sky=mk.f_sky_of(mask),
        n_sims=int(n_sims),
        var_pix_ref=float(var_pix_ref),
    )


# ---------------------------------------------------------------------------
# differentiable (jax.grad-in-w_inv) cut-sky MC
# ---------------------------------------------------------------------------
#
# :func:`mc_cutsky_bandpowers` above is the forward path: a process-pool ensemble
# returning numpy arrays, right for large-nsims forecasts but NOT differentiable
# (parallel_map + np.asarray break the JAX trace, and each sim rebuilds the PySM /
# synalm sky inside the loop, which is not traceable either).
#
# The pair below is the differentiable sibling for end-to-end instrument
# optimization. It splits the work at the traceability boundary:
#
#   * :func:`make_cutsky_mc_context` (EAGER) does the non-traceable, design-
#     independent work ONCE: the per-sim HarmonicSky -> beamed BandSky ensemble
#     (PySM emission + synalm draws + per-band beaming) and the frozen Wiener-filter
#     ``inv_noise`` (a fiducial ``var_pix_ref``). Beams are concrete here.
#   * :func:`mc_cutsky_cov_traced` (TRACED) runs noise -> clean -> masked-Wiener ->
#     bandpowers -> debias -> sample covariance in-trace, all ``jnp``, as a function
#     of the per-band noise amplitude ``w_inv``. Compose it with
#     ``optimize.sigma_r_from_external_cov`` for a ``jax.grad``-able map-based σ(r).
#
# The differentiable lever here is ``w_inv`` (set by NET / n_det / efficiency /
# mission_years / f_sky). **Beams are held concrete** (see the plan's beam-path
# constraint: the cleaner's needlet-channel mask is discrete in beams, and
# ``common_resolution_*`` ``float()`` the beams) — so aperture / band-center
# optimization stays on the analytic path for now. ``var_pix_ref`` is frozen at the
# fiducial design (a filter knob; a mismatch only de-tunes the Wiener filter and is
# absorbed by the transfer ``F_b``, not a bias), the standard CRN treatment.


class CutskyMCContext(eqx.Module):
    """Eager, design-independent precompute for :func:`mc_cutsky_cov_traced`.

    Holds the non-traceable, ``w_inv``-independent pieces of the cut-sky MC so the
    traced forward only carries the noise -> clean -> spectrum path. Built by
    :func:`make_cutsky_mc_context`; passed (un-differentiated) to
    :func:`mc_cutsky_cov_traced`.

    An ``eqx.Module`` (pytree): the array fields + the batched ``band_skies`` /
    ``noise_keys`` are traced leaves, the binning / resolution / loop config are
    ``static`` (in the treedef). So the whole context can be passed as a traced
    argument to a jitted ``mc_cutsky_cov_traced`` -- different CRN ensembles share
    one compiled executable -- and the ``band_skies`` leaves carry the leading
    sim axis that :func:`mc_cutsky_cov_traced`'s ``lax.map`` iterates.
    """

    band_skies: BandSky  # batched BandSky (array leaves carry a leading sim axis)
    noise_keys: jax.Array  # (n_sims, 2) stacked PRNGKeys (CRN; fixed across the gradient)
    beam_fwhm_arcmin: tuple = eqx.field(static=True)  # concrete per-band beams
    inv_noise: jax.Array
    cl_ee_prior: jax.Array
    cl_bb_prior: jax.Array
    bin_matrix: jax.Array
    ell_min: int = eqx.field(static=True)
    true_bb_binned: jax.Array
    hit_map: jax.Array
    knee_ell: jax.Array | None
    alpha_knee: float = eqx.field(static=True)
    nside: int = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    max_iter: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    n_sims: int = eqx.field(static=True)
    var_pix_ref: float = eqx.field(static=True)
    f_sky: float = eqx.field(static=True)


class CutskyMCTraced(eqx.Module):
    """Output of :func:`mc_cutsky_cov_traced` — the differentiable analogue of
    :class:`CutskyMC`, with all fields as traced ``jnp`` arrays (no ``np.asarray``).

    An ``eqx.Module`` (pytree) so ``mc_cutsky_cov_traced`` can be ``jax.jit``-ed
    (its output must be a pytree). Feed :attr:`covariance` to
    ``optimize.sigma_r_from_external_cov`` (or ``FisherForecast(external_covariance=...)``
    for the forward value).
    """

    covariance: jax.Array
    debiased_bandpowers: jax.Array
    transfer: jax.Array
    leakage: jax.Array
    mean_bandpower: jax.Array
    f_sky: float = eqx.field(static=True)
    n_sims: int = eqx.field(static=True)


def make_cutsky_mc_context(
    *,
    cleaner: Cleaner,
    freqs_ghz,
    beam_fwhm_arcmin,
    w_inv,
    nside: int,
    lmax: int,
    mask: jax.Array,
    cl_ee: jax.Array,
    cl_bb_prior_unbeamed: jax.Array,
    bin_matrix: jax.Array,
    ell_min: int,
    true_bb_binned: jax.Array,
    n_sims: int,
    base_seed: int = 0,
    fg_model: str | None = "d1s1",
    r_in: float = 0.0,
    hit_map: jax.Array | None = None,
    var_pix_ref: float | None = None,
    knee_ell: jax.Array | None = None,
    alpha_knee: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
    spectra: CMBSpectra | None = None,
    bandpasses=None,
) -> CutskyMCContext:
    """Eager precompute for the differentiable cut-sky MC (beams held concrete).

    Builds, once, the per-sim ``HarmonicSky`` -> beamed ``BandSky`` ensemble (the
    non-traceable PySM emission + ``synalm`` draws + per-band beaming) plus the
    Wiener-filter ``inv_noise`` at a frozen ``var_pix_ref``, and packs the binning /
    prior statics. The result feeds :func:`mc_cutsky_cov_traced`, which is
    ``jax.grad``-able in ``w_inv``.

    Parameters mirror :func:`mc_cutsky_bandpowers`; ``w_inv`` here is the *fiducial*
    per-band noise power used to set ``var_pix_ref`` (the traced forward varies
    ``w_inv`` around it). ``var_pix_ref`` may be supplied to skip the setup clean.
    """
    spectra = CMBSpectra() if spectra is None else spectra
    freqs_ghz = tuple(float(f) for f in freqs_ghz)
    beam_fwhm_arcmin = tuple(float(b) for b in beam_fwhm_arcmin)
    npix = 12 * int(nside) ** 2
    hit_map = jnp.ones(npix) if hit_map is None else jnp.asarray(hit_map)
    common_fwhm = float(min(beam_fwhm_arcmin))
    cl_ee_prior = beamed_prior(cl_ee, common_fwhm, lmax)
    cl_bb_prior = beamed_prior(cl_bb_prior_unbeamed, common_fwhm, lmax)

    def _beamed_sky(seed: int):
        hsky = harmonic_sky(
            freqs_ghz,
            spectra=spectra,
            r_in=float(r_in),
            nside=int(nside),
            lmax=int(lmax),
            fg_model=fg_model,
            cmb_seed=int(seed),
            cl_ee=cl_ee,
            bandpasses=bandpasses,
        )
        return beam_harmonic_sky(hsky, beam_fwhm_arcmin)

    seeds = list(range(int(base_seed), int(base_seed) + int(n_sims)))
    # Stack the per-sim ensemble into ONE batched BandSky (the cmb_qu/fg_qu leaves
    # get a leading sim axis; the static fields are shared) + stacked keys, so the
    # traced forward can lax.map over the sim axis instead of Python-unrolling the
    # cleaner n_sims times (O(1) compile, scan-accumulated memory -> higher n_sims).
    _band_sky_tuple = tuple(_beamed_sky(s) for s in seeds)
    band_skies = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *_band_sky_tuple)
    noise_keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds], axis=0)

    # var_pix_ref: frozen filter knob from one setup clean at the fiducial w_inv
    # (matches mc_cutsky_bandpowers; uses the base_seed + n_sims setup sim).
    if var_pix_ref is None:
        sky0 = _beamed_sky(base_seed + n_sims)
        total0 = assemble_band_maps(
            sky0,
            jnp.asarray(w_inv),
            hit_map,
            noise_key=jax.random.PRNGKey(int(base_seed + n_sims)),
            knee_ell=knee_ell,
            alpha_knee=alpha_knee,
        )
        res0 = cleaner(total0, beam_fwhm_arcmin, lmax=lmax, nside=nside)
        noise0 = total0 - sky0.cmb_qu - sky0.fg_qu
        cn = _cleaned_b_qu(res0, noise0) + _cleaned_e_qu(res0, noise0)
        obs = jnp.asarray(mask) > 0
        var_pix_ref = float(jnp.mean((cn[0] ** 2 + cn[1] ** 2)[obs]) / 2.0)

    inv_noise = mk.inv_noise_map(hit_map, var_pix_ref, mask=mask)

    return CutskyMCContext(
        band_skies=band_skies,
        noise_keys=noise_keys,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
        inv_noise=inv_noise,
        cl_ee_prior=cl_ee_prior,
        cl_bb_prior=cl_bb_prior,
        bin_matrix=jnp.asarray(bin_matrix),
        ell_min=int(ell_min),
        true_bb_binned=jnp.asarray(true_bb_binned),
        hit_map=hit_map,
        knee_ell=knee_ell,
        alpha_knee=alpha_knee,
        nside=int(nside),
        lmax=int(lmax),
        max_iter=int(max_iter),
        tol=float(tol),
        n_sims=int(n_sims),
        var_pix_ref=float(var_pix_ref),
        f_sky=mk.f_sky_of(mask),
    )


def mc_cutsky_cov_traced(
    w_inv: jax.Array,
    ctx: CutskyMCContext,
    cleaner: Cleaner,
) -> CutskyMCTraced:
    """Differentiable cut-sky MC bandpower covariance as a function of ``w_inv``.

    The ``jax.grad``-able sibling of :func:`mc_cutsky_bandpowers`: with the per-sim
    beamed sky ensemble + ``inv_noise`` precomputed in ``ctx`` (beams concrete,
    ``var_pix_ref`` frozen), this runs noise -> clean -> masked-Wiener ->
    bandpowers -> debias -> sample covariance entirely in ``jnp``, differentiable in
    the per-band noise amplitude ``w_inv`` under common random numbers (the
    ``ctx.noise_keys`` are fixed). Compose with
    ``optimize.sigma_r_from_external_cov(result.covariance, opt_ctx)`` for a
    ``jax.grad``-able map-based σ(r). Select the on-device jht SHT backend
    (``with sht.sht_backend("jht"): ...``) to run the whole thing on a GPU.

    The per-sim loop is a ``jax.lax.map`` (sequential scan, no ``batch_size``) over
    the batched ``ctx.band_skies`` -- the cleaner body is traced once, so compile is
    O(1) in ``n_sims`` and the scan accumulates outputs instead of holding an
    ``n_sims``-deep unroll live. Scan (not ``vmap``) so the cleaner's inner
    ``while_loop`` map2alm iteration runs per sim. The straight-through gradient
    flows through a *sample* covariance, so it carries Monte-Carlo noise --
    characterise grad std vs ``n_sims`` before trusting a descent step (see the
    plan's Phase 2 verification).
    """
    w_inv = jnp.asarray(w_inv)
    bp_kw = dict(
        bin_matrix=ctx.bin_matrix,
        ell_min=ctx.ell_min,
        nside=ctx.nside,
        lmax=ctx.lmax,
        max_iter=ctx.max_iter,
        tol=ctx.tol,
    )

    def _one(band_sky, key):
        total = assemble_band_maps(
            band_sky,
            w_inv,
            ctx.hit_map,
            noise_key=key,
            knee_ell=ctx.knee_ell,
            alpha_knee=ctx.alpha_knee,
        )
        result = cleaner(total, ctx.beam_fwhm_arcmin, lmax=ctx.lmax, nside=ctx.nside)
        rec_full = cutsky_bb_bandpower(
            result.cleaned_qu(), ctx.inv_noise, ctx.cl_ee_prior, ctx.cl_bb_prior, **bp_kw
        )
        rec_b = cutsky_bb_bandpower(
            _cleaned_b_qu(result, band_sky.cmb_qu),
            ctx.inv_noise,
            ctx.cl_ee_prior,
            ctx.cl_bb_prior,
            **bp_kw,
        )
        rec_e = cutsky_bb_bandpower(
            _cleaned_e_qu(result, band_sky.cmb_qu),
            ctx.inv_noise,
            ctx.cl_ee_prior,
            ctx.cl_bb_prior,
            **bp_kw,
        )
        return rec_full, rec_b, rec_e

    # Sequential scan over the sim axis (lax.map, no batch_size): the cleaner body
    # is traced ONCE and reused per sim -- O(1) compile in n_sims, scan-accumulated
    # outputs (no live n_sims-deep unroll). Scan, not vmap, so the cleaner's inner
    # map2alm while_loop runs per sim. ctx.band_skies is a batched BandSky (leading
    # sim axis on its leaves); lax.map slices it back to a per-sim BandSky.
    rec_full, rec_b, rec_e = jax.lax.map(
        lambda bk: _one(bk[0], bk[1]), (ctx.band_skies, ctx.noise_keys)
    )

    transfer = mk.transfer_function(rec_b, ctx.true_bb_binned)
    leakage = mk.leakage_template(rec_e)
    debiased = mk.debias_bandpower(rec_full, transfer, leakage)
    cov = mc_bandpower_covariance(debiased, hartlap=True)

    return CutskyMCTraced(
        covariance=cov,
        debiased_bandpowers=debiased,
        transfer=transfer,
        leakage=leakage,
        mean_bandpower=jnp.mean(debiased, axis=0),
        f_sky=ctx.f_sky,
        n_sims=ctx.n_sims,
    )
