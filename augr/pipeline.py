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

from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp

from .cleaning import Cleaner, CleanerResult
from .compsep_sims import assemble_band_maps, generate_band_sky
from .forecast import ForecastResult, forecast_from_spectra
from .gnilc import gnilc_residual_template
from .nilc_forecast import NILCSpectra, nilc_spectra
from .spectra import CMBSpectra


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


def forecast_cleaned(cleaned: CleanedSky, config: ForecastConfig) -> ForecastResult:
    """Forecast from an already-:func:`clean_sky` result (the spine's tail).

    Selects the residual-foreground template per ``config.residual_source`` (the
    oracle ``cl_residual_fg`` or the data-driven GNILC template) and feeds the
    spectra to :func:`augr.forecast.forecast_from_spectra`.
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

    return forecast_from_spectra(
        nl_ells=spectra.ells,
        nl_post=spectra.nl_post,
        template_ells=spectra.ells,
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


def run_forecast(config: ForecastConfig) -> ForecastResult:
    """Run the full spine for ``config`` and return the σ(r) variants + Δr.

    Equivalent to ``forecast_cleaned(clean_sky(config), config)``.
    """
    return forecast_cleaned(clean_sky(config), config)
