"""nilc_forecast.py — turn NILC cleaned maps into σ(r) and the FG-leakage bias Δr.

Bridges :mod:`augr.nilc` to the Fisher forecast. Two steps:

1. :func:`nilc_spectra` projects the FG-only / noise-only / CMB-only map sets
   through the *same* ILC weights (:meth:`augr.nilc.NILCResult.project`) and forms
   their B-mode auto/cross spectra:

   * ``nl_post``        — post-NILC noise ``N_ℓ^{BB}`` (noise-only projection),
   * ``cl_residual_fg`` — residual foreground ``ΔC_ℓ`` (FG-only projection),
   * ``cl_cleaned``     — the cleaned total BB (diagnostic),
   * ``transfer``       — cleaned-CMB / common-resolution-CMB (signal-loss check).

   The cleaned map lives at the common resolution ``B_c``, so the spectra are
   deconvolved by ``B_c²`` to the beam-free form that
   :class:`augr.fisher.FisherForecast` (and ``SignalModel``) assume.

2. :func:`nilc_forecast` mirrors ``scripts/validate_carones.run_fisher_variants``:
   ``cleaned_map_instrument`` + ``NullForegroundModel`` + ``SignalModel`` with the
   residual template, feeding ``nl_post`` as ``external_noise_bb``. It returns σ(r)
   under (baseline / flat-A_res / Gaussian-A_res), and the FG-leakage bias **Δr**
   from :meth:`FisherForecast.bias_from_truth_model` — the debias-OFF primary bias
   (the residual is left unmodelled in the fit), with the A_res runs as the
   debias-ON companion.

f_sky correction: the B-mode spectra are divided by ``f_sky`` (= 1 full-sky in v1).
The signal-loss transfer is validated ≈ 1 by Stage 3; it is reported here and only
applied (dividing through ``nl_post`` / ``cl_residual_fg``) if ``apply_transfer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .config import DEFAULT_PRIORS_POST_COMPSEP, cleaned_map_instrument
from .fisher import FisherForecast
from .foregrounds import NullForegroundModel
from .instrument import beam_bl
from .nilc import NILCResult, common_resolution_b_alm
from .signal import SignalModel
from .spectra import CMBSpectra


def cl_bb(b_alm, lmax: int, b_alm2=None, f_sky: float = 1.0) -> np.ndarray:
    """B-mode auto (or cross) ``C_ℓ`` from B-only alm, with a 1/f_sky correction."""
    import healpy as hp

    a1 = np.asarray(b_alm)
    a2 = None if b_alm2 is None else np.asarray(b_alm2)
    return np.asarray(hp.alm2cl(a1, a2, lmax=int(lmax))) / f_sky


@dataclass(frozen=True)
class NILCSpectra:
    """B-mode spectra extracted from a NILC run, beam-deconvolved to ``B_c``-free.

    Attributes
    ----------
    ells
        Integer multipoles ``0..lmax``.
    cl_cleaned
        Cleaned total BB (diagnostic).
    nl_post
        Post-NILC noise ``N_ℓ^{BB}`` → ``external_noise_bb``.
    cl_residual_fg
        Residual foreground ``ΔC_ℓ`` → A_res template and Δr source.
    transfer
        Cleaned-CMB / common-resolution-CMB transfer (≈ 1 if no signal loss).
    common_fwhm_arcmin
        Common resolution the maps were brought to (the deconvolved beam).
    f_sky
        Sky fraction used in the 1/f_sky correction.
    """

    ells: np.ndarray
    cl_cleaned: np.ndarray
    nl_post: np.ndarray
    cl_residual_fg: np.ndarray
    transfer: np.ndarray
    common_fwhm_arcmin: float
    f_sky: float


def nilc_spectra(
    result: NILCResult,
    *,
    total_qu,
    noise_qu,
    fg_qu,
    cmb_qu,
    f_sky: float = 1.0,
    deconvolve_common_beam: bool = True,
) -> NILCSpectra:
    """Project FG/noise/CMB-only maps through the NILC weights → beam-free BB spectra.

    Parameters
    ----------
    result
        A completed :class:`augr.nilc.NILCResult` (weights derived from the total).
    total_qu, noise_qu, fg_qu, cmb_qu
        The matched ``(n_band, 2, npix)`` map sets: full data, noise-only, FG-only,
        CMB-only. ``total_qu`` is only used for the diagnostic ``cl_cleaned``.
    f_sky
        Observed sky fraction (1/f_sky correction on every spectrum).
    deconvolve_common_beam
        Divide ``nl_post`` / ``cl_residual_fg`` / ``cl_cleaned`` by ``B_c²`` so they
        are beam-free (required by ``FisherForecast``). Leave on unless the caller
        wants the at-resolution spectra.
    """
    lmax = result.lmax
    ells = np.arange(lmax + 1)

    cleaned_total = result.cleaned_b_alm
    cleaned_noise = result.project(noise_qu)
    cleaned_fg = result.project(fg_qu)
    cleaned_cmb = result.project(cmb_qu)

    # Reference: the CMB brought to common resolution but NOT ILC-combined.
    cmb_common, _ = common_resolution_b_alm(
        cmb_qu,
        result.beam_fwhm_arcmin,
        lmax=lmax,
        nside=result.nside,
        n_iter=result.n_iter,
    )
    cmb_ref = cmb_common[0]  # all bands identical at common resolution

    cl_total = cl_bb(cleaned_total, lmax, f_sky=f_sky)
    nl_post = cl_bb(cleaned_noise, lmax, f_sky=f_sky)
    cl_fg = cl_bb(cleaned_fg, lmax, f_sky=f_sky)

    cross = cl_bb(cleaned_cmb, lmax, b_alm2=cmb_ref, f_sky=f_sky)
    cl_ref = cl_bb(cmb_ref, lmax, f_sky=f_sky)
    transfer = np.divide(cross, cl_ref, out=np.ones_like(cross), where=cl_ref > 0)

    if deconvolve_common_beam:
        bl = np.asarray(beam_bl(jnp.asarray(ells, dtype=float), result.common_fwhm_arcmin))
        bl2 = np.maximum(bl**2, 1e-8)  # floor guards the un-used high-ℓ tail
        nl_post = nl_post / bl2
        cl_fg = cl_fg / bl2
        cl_total = cl_total / bl2

    return NILCSpectra(
        ells=ells,
        cl_cleaned=cl_total,
        nl_post=nl_post,
        cl_residual_fg=cl_fg,
        transfer=transfer,
        common_fwhm_arcmin=result.common_fwhm_arcmin,
        f_sky=f_sky,
    )


def nilc_leakage_correlation(result: NILCResult, fg_qu, *, f_sky: float = 1.0):
    """Cross-correlation ρ_ℓ between the cleaned FG residual and the input FG.

    Projects the FG-only maps through the ILC weights, then correlates the cleaned
    residual against the common-resolution input foreground at the lowest-frequency
    (most FG-dominated) band. Characterizes the *morphology* of the residual:

    * ``ρ → 1`` would indicate a single-component leakage that is essentially a
      rescaling of one band (e.g. resolution-limited under-cleaning of one FG).
    * ``ρ → 0`` is the usual multi-band-ILC case: with enough channels the ILC nulls
      the dominant FG components, so the surviving residual is the *spatial-SED-
      variation* leakage, which is decorrelated from any single input band (it is
      not noise — it is FG, just not coherent with one band's morphology).

    Diagnostic only (the bias-relevant quantity is the residual auto-spectrum that
    feeds Δr). Returns ``(ells, rho)`` with ``ells = 0..lmax``.
    """
    lmax = result.lmax
    cleaned_fg = result.project(fg_qu)
    fg_common, _ = common_resolution_b_alm(
        fg_qu, result.beam_fwhm_arcmin, lmax=lmax, nside=result.nside, n_iter=result.n_iter
    )
    fg_ref = fg_common[0]  # lowest-frequency band: most foreground

    cross = cl_bb(cleaned_fg, lmax, b_alm2=fg_ref, f_sky=f_sky)
    auto_res = cl_bb(cleaned_fg, lmax, f_sky=f_sky)
    auto_ref = cl_bb(fg_ref, lmax, f_sky=f_sky)
    denom = np.sqrt(auto_res * auto_ref)
    rho = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)
    return np.arange(lmax + 1), rho


def analytic_mv_noise_floor(w_inv, beam_fwhm_arcmin, lmax: int) -> np.ndarray:
    """Beam-deconvolved inverse-variance ILC noise floor ``1/Σ_b B_b²/w_inv_b``.

    The best a (per-ℓ, a=1) ILC can do against independent per-band white noise
    ``w_inv_b`` with Gaussian beams. Post-NILC noise approaches this in the
    noise-dominated regime; needlet banding makes the match piecewise.
    """
    ells = jnp.arange(lmax + 1, dtype=float)
    inv = jnp.zeros_like(ells)
    for w, fwhm in zip(w_inv, beam_fwhm_arcmin, strict=True):
        inv = inv + beam_bl(ells, fwhm) ** 2 / w
    return np.asarray(1.0 / inv)


def nilc_forecast(
    spectra: NILCSpectra,
    *,
    f_sky: float,
    r_fid: float = 0.0,
    ell_min: int = 2,
    ell_max: int = 180,
    delta_ell: int = 5,
    ell_per_bin_below: int = 30,
    a_res_prior: float | None = None,
    apply_transfer: bool = False,
    delensed_bb=None,
    delensed_bb_ells=None,
) -> dict:
    """σ(r) variants + the FG-leakage bias Δr from a :class:`NILCSpectra`.

    Mirrors ``validate_carones.run_fisher_variants``: ``cleaned_map_instrument`` +
    ``NullForegroundModel`` + a residual-template ``SignalModel``, with ``nl_post``
    as beam-free ``external_noise_bb``.

    ``delta_r`` is the linear bias on the *baseline* fit's r induced by the
    unmodelled residual (debias-OFF, primary). The flat / Gaussian-A_res σ(r) are
    the debias-ON companion (residual fit with a nuisance amplitude).

    ``delensed_bb`` / ``delensed_bb_ells`` (optional): a self-consistent residual
    lensing ``C_ℓ^{BB}`` (e.g. ``iterate_delensing(...).cl_bb_res`` on its ``ls``
    grid) to use in place of the ``A_lens`` lensing multiplier — the same hook as
    ``validate_carones.run_fisher_variants(delensed_bb=...)``. ``delensed_bb_ells``
    must span ``[ell_min, ell_max]``. Without it the forecast sits on the full
    lensing-B floor (fine for the lensing-independent Δr, but it inflates σ(r)).

    Returns a dict of printable scalars (σ(r) baseline/flat/gauss, σ(A_res),
    delta_r, transfer_mean, the (r, A_res) condition number, and the inputs used).
    """
    if a_res_prior is None:
        a_res_prior = DEFAULT_PRIORS_POST_COMPSEP["A_res"]
    if (delensed_bb is None) != (delensed_bb_ells is None):
        raise ValueError("delensed_bb and delensed_bb_ells must be supplied together.")

    nl_post = np.asarray(spectra.nl_post)
    cl_fg = np.asarray(spectra.cl_residual_fg)
    band = (spectra.ells >= ell_min) & (spectra.ells <= ell_max)
    transfer_mean = float(np.mean(spectra.transfer[band]))
    if apply_transfer and transfer_mean > 0:
        nl_post = nl_post / transfer_mean
        cl_fg = cl_fg / transfer_mean

    inst = cleaned_map_instrument(f_sky=f_sky)
    cmb = CMBSpectra()

    def _signal(with_template):
        kw = dict(
            instrument=inst,
            foreground_model=NullForegroundModel(),
            cmb_spectra=cmb,
            ell_min=ell_min,
            ell_max=ell_max,
            delta_ell=delta_ell,
            ell_per_bin_below=ell_per_bin_below,
        )
        if with_template:
            kw["residual_template_cl"] = jnp.asarray(cl_fg)
            kw["residual_template_ells"] = jnp.asarray(spectra.ells, dtype=float)
        if delensed_bb is not None:
            kw["delensed_bb"] = jnp.asarray(delensed_bb)
            kw["delensed_bb_ells"] = jnp.asarray(delensed_bb_ells, dtype=float)
        return SignalModel(**kw)

    baseline = _signal(with_template=False)
    nl_interp = jnp.interp(
        baseline.ells, jnp.asarray(spectra.ells, dtype=float), jnp.asarray(nl_post)
    )
    external_noise_bb = nl_interp[None, :]

    fid_base = {"r": r_fid, "A_lens": 1.0}
    fisher_baseline = FisherForecast(
        baseline, inst, fid_base, priors={}, fixed_params=[], external_noise_bb=external_noise_bb
    )
    fisher_baseline.compute()

    signal = _signal(with_template=True)
    fid = {**fid_base, "A_res": 1.0}
    fisher_flat = FisherForecast(
        signal, inst, fid, priors={}, fixed_params=[], external_noise_bb=external_noise_bb
    )
    fisher_flat.compute()
    fisher_gauss = FisherForecast(
        signal,
        inst,
        fid,
        priors={"A_res": a_res_prior},
        fixed_params=[],
        external_noise_bb=external_noise_bb,
    )
    fisher_gauss.compute()

    # Δr (debias-OFF): bias on the baseline fit from the unmodelled residual.
    # truth = baseline + residual template at A_res=1; ΔD = the binned residual.
    delta = fisher_baseline.bias_from_truth_model(signal, fid)

    # (r, A_res) marginalized condition number, as in validate_carones. A zero
    # residual template (no FG) leaves A_res unconstrained -> F singular -> inf.
    F = np.asarray(fisher_flat.fisher_matrix)
    names = fisher_flat.free_parameter_names
    ix = np.array([names.index("r"), names.index("A_res")])
    try:
        cond = float(np.linalg.cond(np.linalg.inv(F)[np.ix_(ix, ix)]))
    except np.linalg.LinAlgError:
        cond = float("inf")

    return {
        "sigma_r_baseline": fisher_baseline.sigma("r"),
        "sigma_r_flat": fisher_flat.sigma("r"),
        "sigma_r_gauss": fisher_gauss.sigma("r"),
        "sigma_A_res_flat": fisher_flat.sigma("A_res"),
        "sigma_A_res_gauss": fisher_gauss.sigma("A_res"),
        "delta_r": delta["r"],
        "transfer_mean": transfer_mean,
        "cond_r_Ares": cond,
        "a_res_prior": a_res_prior,
        "r_fid": r_fid,
    }
