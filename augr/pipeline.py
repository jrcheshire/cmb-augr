"""pipeline.py — one config-driven driver for the in-house compsep → σ(r) spine.

The diagnostic/study scripts (``nilc_diagnostics`` / ``cmilc_diagnostics`` /
``gnilc_diagnostics`` / the JPL aperture sweep) all re-implement the same spine:
**sim sky → component-separation cleaner → spectra → Fisher σ(r)/Δr**. This
collapses that spine into a single :func:`run_forecast` taking a frozen
:class:`ForecastConfig`. The cleaner is a pluggable :class:`augr.cleaning.Cleaner`
(build one with :func:`augr.cleaning.nilc_cleaner` / ``cmilc_cleaner``); the
residual-foreground template — the one genuine branch in the spine — is selected
by :class:`ResidualTemplateSource` (the oracle from the true ``fg_qu`` through the
cleaner weights, or the data-driven GNILC template).

The sim path imports :mod:`augr.compsep_sims`, so this driver needs the
``[compsep]`` extra (ducc0; pysm3 once ``fg_model`` is not ``None``).
:func:`augr.forecast.forecast_from_spectra` itself stays on the light Fisher path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import jax
import jax.numpy as jnp

from .cleaning import Cleaner, CleanerResult
from .compsep_sims import assemble_band_maps, generate_band_sky
from .forecast import ForecastResult, forecast_from_spectra
from .gnilc import gnilc_residual_template
from .nilc_forecast import NILCSpectra, nilc_spectra
from .spectra import CMBSpectra


class SpectrumSource(Enum):
    """How the post-separation B-mode bandpower covariance is formed.

    ``FULLSKY_SCALAR`` (default) is the in-house full-sky path: cleaned-map spectra via
    ``alm2cl`` and an analytic Knox covariance with a scalar ``1/f_sky`` correction
    (:func:`augr.nilc_forecast.nilc_spectra` → ``external_noise_bb``). ``CUTSKY_MC``
    runs a spin-2 Q/U cleaner over a Monte-Carlo ensemble and estimates the cut-sky
    ``C_ℓ^{BB}`` with the masked-Wiener filter, so the covariance — including the
    mask-dependent E→B leakage variance — comes from the sim ensemble
    (:func:`augr.spectrum_stages.mc_cutsky_bandpowers` → ``external_covariance``).
    Both share the same residual-foreground template and Fisher Jacobian, so the σ(r)
    difference isolates what the scalar-``1/f_sky`` approximation costs.
    """

    FULLSKY_SCALAR = "fullsky_scalar"
    CUTSKY_MC = "cutsky_mc"


class ResidualTemplateSource(Enum):
    """Where the residual-foreground template ``ΔC_ℓ`` (the ``A_res`` template) comes from.

    ``ORACLE`` projects the *true* ``fg_qu`` through the cleaner weights
    (:attr:`augr.nilc_forecast.NILCSpectra.cl_residual_fg`) — exact, but needs the
    ground-truth foreground map, so it is a forecasting diagnostic, not a data
    estimator. ``GNILC`` builds the data-driven
    :func:`augr.gnilc.gnilc_residual_template` from the total + (CMB+noise)
    nuisance maps — what a real pipeline can compute. ``nl_post`` (the noise) always
    comes from the configured cleaner regardless of the template source.
    """

    ORACLE = "oracle"
    GNILC = "gnilc"


@dataclass(frozen=True)
class ForecastConfig:
    """Inputs for one :func:`run_forecast` run (sky → cleaner → spectra → forecast).

    ``cleaner`` is any :class:`augr.cleaning.Cleaner` — typically
    ``nilc_cleaner(...)`` or ``cmilc_cleaner(freqs, ...)``. ``w_inv`` is the per-band
    polarization white-noise power [μK²·sr]. ``hit_map`` ``None`` → uniform exposure.
    The forecast knobs mirror :func:`augr.forecast.forecast_from_spectra`.
    """

    # sky / instrument
    freqs_ghz: tuple[float, ...]
    beam_fwhm_arcmin: tuple[float, ...]
    w_inv: tuple[float, ...]
    cleaner: Cleaner
    nside: int
    lmax: int
    # sky realization
    fg_model: str | None = "d1s1"
    r_in: float = 0.0
    seed: int = 0
    hit_map: jax.Array | None = None
    # residual-template source
    residual_source: ResidualTemplateSource = ResidualTemplateSource.ORACLE
    gnilc_m_bias: int = 1
    # forecast
    f_sky: float = 1.0
    r_fid: float = 0.0
    ell_min: int = 2
    ell_max: int = 180
    delta_ell: int = 5
    ell_per_bin_below: int = 30
    a_res_prior: float | None = None
    apply_transfer: bool = False
    delensed_bb: jax.Array | None = None
    delensed_bb_ells: jax.Array | None = None
    # spectrum stage (cut-sky masked-Wiener MC; CUTSKY_MC fields below are ignored for
    # the default FULLSKY_SCALAR path). CUTSKY_MC requires a spin-2 cleaner
    # (``nilc_cleaner(clean_e=True)`` / ``cmilc_cleaner(..., clean_e=True)``) and a mask;
    # ``f_sky`` is overridden by the realized ``⟨mask⟩``.
    spectrum_source: SpectrumSource = SpectrumSource.FULLSKY_SCALAR
    mask: jax.Array | None = None
    n_sims_mc: int = 50
    base_seed_mc: int = 0
    var_pix_ref: float | None = None
    cl_ee_prior: jax.Array | None = None
    cl_bb_prior: jax.Array | None = None
    mc_workers: int = 1


@dataclass(frozen=True)
class CleanedSky:
    """Shared intermediates of one sim + clean: cleaner result, spectra, component maps.

    The prefix of the :func:`run_forecast` spine (sky → cleaner → spectra) exposed so
    diagnostic callers can read the intermediates they plot — post-separation noise,
    residual FG, transfer, leakage — from a single sim+clean. ``cleaner_result.project``
    and the component maps (``fg_qu`` etc.) drive those diagnostics; a second cleaner
    (cMILC vs NILC) or a second template (GNILC vs oracle) reuses the same maps.
    """

    cleaner_result: CleanerResult
    spectra: NILCSpectra
    total_qu: jax.Array
    noise_qu: jax.Array
    fg_qu: jax.Array
    cmb_qu: jax.Array


def clean_sky(config: ForecastConfig) -> CleanedSky:
    """Build the sim, clean it, and project the post-separation spectra.

    The sky → cleaner → ``nilc_spectra`` prefix of :func:`run_forecast`, exposed so
    diagnostic scripts share one sim+clean and read the intermediates: 1.
    ``generate_band_sky`` → beamed CMB + FG; 2. ``assemble_band_maps`` adds CRN
    noise; 3. ``config.cleaner`` produces the cleaned-map result; 4. ``nilc_spectra``
    projects noise/FG/CMB through the weights.
    """
    sky = generate_band_sky(
        config.freqs_ghz,
        config.beam_fwhm_arcmin,
        spectra=CMBSpectra(),
        r_in=config.r_in,
        nside=config.nside,
        lmax=config.lmax,
        fg_model=config.fg_model,
        cmb_seed=config.seed,
    )
    hit_map = jnp.ones(sky.npix) if config.hit_map is None else config.hit_map
    total = assemble_band_maps(
        sky,
        jnp.asarray(config.w_inv),
        hit_map,
        noise_key=jax.random.PRNGKey(config.seed),
    )
    noise = total - sky.cmb_qu - sky.fg_qu

    result = config.cleaner(total, config.beam_fwhm_arcmin, lmax=config.lmax, nside=config.nside)
    spectra = nilc_spectra(
        result,
        total_qu=total,
        noise_qu=noise,
        fg_qu=sky.fg_qu,
        cmb_qu=sky.cmb_qu,
        f_sky=config.f_sky,
    )
    return CleanedSky(
        cleaner_result=result,
        spectra=spectra,
        total_qu=total,
        noise_qu=noise,
        fg_qu=sky.fg_qu,
        cmb_qu=sky.cmb_qu,
    )


def _residual_template(cleaned: CleanedSky, config: ForecastConfig):
    """Select the residual-foreground template ``(ells, ΔC_ℓ)`` per ``residual_source``.

    The oracle ``cl_residual_fg`` (true ``fg_qu`` through the cleaner weights) or the
    data-driven GNILC template. Shared by the full-sky and cut-sky forecast tails so
    both consume an identical ``A_res`` template (only the covariance differs).
    """
    spectra = cleaned.spectra
    cl_residual_fg = spectra.cl_residual_fg
    if config.residual_source is ResidualTemplateSource.GNILC:
        _ells, cl_residual_fg = gnilc_residual_template(
            cleaned.total_qu,
            cleaned.cmb_qu,
            cleaned.noise_qu,
            config.beam_fwhm_arcmin,
            lmax=config.lmax,
            nside=config.nside,
            m_bias=config.gnilc_m_bias,
            f_sky=config.f_sky,
        )
    return spectra.ells, cl_residual_fg


def forecast_cleaned(cleaned: CleanedSky, config: ForecastConfig) -> ForecastResult:
    """Forecast from an already-:func:`clean_sky` result (the spine's tail).

    The full-sky-scalar tail: selects the residual-foreground template per
    ``config.residual_source`` and feeds the cleaned-map spectra (with the analytic
    Knox / scalar-``1/f_sky`` covariance) to :func:`augr.forecast.forecast_from_spectra`.
    """
    spectra = cleaned.spectra
    template_ells, cl_residual_fg = _residual_template(cleaned, config)
    return forecast_from_spectra(
        nl_ells=spectra.ells,
        nl_post=spectra.nl_post,
        template_ells=template_ells,
        template_cl=cl_residual_fg,
        f_sky=config.f_sky,
        transfer=spectra.transfer,
        r_fid=config.r_fid,
        ell_min=config.ell_min,
        ell_max=config.ell_max,
        delta_ell=config.delta_ell,
        ell_per_bin_below=config.ell_per_bin_below,
        a_res_prior=config.a_res_prior,
        apply_transfer=config.apply_transfer,
        delensed_bb=config.delensed_bb,
        delensed_bb_ells=config.delensed_bb_ells,
    )


def forecast_cleaned_cutsky_mc(cleaned: CleanedSky, config: ForecastConfig) -> ForecastResult:
    """Cut-sky forecast tail: masked-Wiener MC covariance + the shared residual template.

    Reuses ``cleaned`` only for the residual-foreground template (so the σ(r) here is
    apples-to-apples with :func:`forecast_cleaned`: identical ``A_res`` template and
    Fisher Jacobian). The covariance comes from a fresh spin-2-cleaner Monte-Carlo
    ensemble through the masked-Wiener estimator
    (:func:`augr.spectrum_stages.mc_cutsky_bandpowers`). ``config.f_sky`` is the realized
    ``⟨mask⟩`` (set by :func:`run_forecast`).
    """
    from . import masking as mk
    from .config import cleaned_map_instrument
    from .delensing import load_lensing_spectra
    from .foregrounds import NullForegroundModel
    from .signal import SignalModel
    from .spectrum_stages import mc_cutsky_bandpowers

    if config.mask is None:
        raise ValueError("spectrum_source=CUTSKY_MC requires config.mask.")

    template_ells, cl_residual_fg = _residual_template(cleaned, config)

    # Binning identical to the Fisher bins: take the bin matrix from a throwaway
    # SignalModel built with the same ell knobs forecast_from_spectra uses.
    inst = cleaned_map_instrument(f_sky=config.f_sky)
    sm = SignalModel(
        instrument=inst,
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=config.ell_min,
        ell_max=config.ell_max,
        delta_ell=config.delta_ell,
        ell_per_bin_below=config.ell_per_bin_below,
    )
    bin_matrix = sm.bin_matrix

    ls = load_lensing_spectra()
    cl_ee = (
        config.cl_ee_prior
        if config.cl_ee_prior is not None
        else jnp.clip(ls.cl_ee_len[: config.lmax + 1], 0.0, None)
    )
    cl_bb = (
        config.cl_bb_prior
        if config.cl_bb_prior is not None
        else jnp.clip(ls.cl_bb_len[: config.lmax + 1], 0.0, None)
    )
    ells = jnp.arange(config.lmax + 1, dtype=float)
    true_bb = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(ells, config.r_in), 0.0, None), bin_matrix, config.ell_min
    )

    mc = mc_cutsky_bandpowers(
        cleaner=config.cleaner,
        freqs_ghz=config.freqs_ghz,
        beam_fwhm_arcmin=config.beam_fwhm_arcmin,
        w_inv=config.w_inv,
        nside=config.nside,
        lmax=config.lmax,
        mask=config.mask,
        cl_ee=cl_ee,
        cl_bb_prior_unbeamed=cl_bb,
        bin_matrix=bin_matrix,
        ell_min=config.ell_min,
        true_bb_binned=true_bb,
        n_sims=config.n_sims_mc,
        base_seed=config.base_seed_mc,
        fg_model=config.fg_model,
        r_in=config.r_in,
        hit_map=config.hit_map,
        var_pix_ref=config.var_pix_ref,
        workers=config.mc_workers,
    )

    return forecast_from_spectra(
        template_ells=template_ells,
        template_cl=cl_residual_fg,
        f_sky=config.f_sky,
        external_covariance=mc.covariance,
        r_fid=config.r_fid,
        ell_min=config.ell_min,
        ell_max=config.ell_max,
        delta_ell=config.delta_ell,
        ell_per_bin_below=config.ell_per_bin_below,
        a_res_prior=config.a_res_prior,
        delensed_bb=config.delensed_bb,
        delensed_bb_ells=config.delensed_bb_ells,
    )


def run_forecast(config: ForecastConfig) -> ForecastResult:
    """Run the full spine for ``config`` and return the σ(r) variants + Δr.

    ``FULLSKY_SCALAR`` (default) is ``forecast_cleaned(clean_sky(config), config)``.
    ``CUTSKY_MC`` reuses one representative clean for the residual template and swaps in
    the masked-Wiener Monte-Carlo covariance; ``config.f_sky`` is forced to the realized
    ``⟨mask⟩`` so the representative clean and the forecast share one sky fraction.
    """
    if config.spectrum_source is SpectrumSource.CUTSKY_MC:
        from . import masking as mk

        if config.mask is None:
            raise ValueError("spectrum_source=CUTSKY_MC requires config.mask.")
        config = replace(config, f_sky=mk.f_sky_of(config.mask))
        return forecast_cleaned_cutsky_mc(clean_sky(config), config)
    return forecast_cleaned(clean_sky(config), config)
