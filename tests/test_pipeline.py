"""Gates for augr.pipeline.run_forecast: the consolidated sky→cleaner→forecast spine.

``run_forecast`` with a NILC config must reproduce the manual nilc_diagnostics
spine bit-for-bit; the cMILC cleaner and the GNILC residual-source must run and
give a finite σ(r). All need ducc0 (real SHTs); ``fg_model=None`` keeps the fast
gates pysm-free, and the meaningful GNILC-with-foreground gate is marked slow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

from augr.cleaning import cmilc_cleaner, nilc_cleaner
from augr.cmilc import CMILC06_MOMENTS
from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.forecast import ForecastResult
from augr.nilc import nilc_clean
from augr.nilc_forecast import nilc_forecast, nilc_spectra
from augr.pipeline import ForecastConfig, ResidualTemplateSource, run_forecast
from augr.spectra import CMBSpectra

NSIDE, LMAX = 32, 48
FREQS = (30.0, 90.0, 150.0)
BEAMS = (40.0, 20.0, 10.0)
PEAKS = [8, 24, LMAX]
W_INV = (5e-4, 5e-4, 5e-4)

FC_KW = dict(f_sky=1.0, r_fid=0.01, ell_min=2, ell_max=40, delta_ell=5, ell_per_bin_below=20)


def _nilc_config(**over) -> ForecastConfig:
    base = dict(
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=W_INV,
        cleaner=nilc_cleaner(needlet_peaks=PEAKS),
        nside=NSIDE,
        lmax=LMAX,
        fg_model=None,
        r_in=0.01,
        seed=0,
        **FC_KW,
    )
    base.update(over)
    return ForecastConfig(**base)


def test_run_forecast_returns_forecast_result() -> None:
    out = run_forecast(_nilc_config())
    assert isinstance(out, ForecastResult)
    assert np.isfinite(out.sigma_r_baseline) and out.sigma_r_baseline > 0
    assert abs(out.delta_r) < 1e-12  # no foreground -> no leakage bias


def test_run_forecast_matches_manual_nilc_spine() -> None:
    """run_forecast == the inlined generate_band_sky → nilc_clean → nilc_forecast spine."""
    cfg = _nilc_config()
    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=cfg.r_in,
        nside=NSIDE,
        lmax=LMAX,
        fg_model=None,
        cmb_seed=cfg.seed,
    )
    total = assemble_band_maps(
        sky, jnp.asarray(W_INV), jnp.ones(sky.npix), noise_key=jax.random.PRNGKey(cfg.seed)
    )
    noise = total - sky.cmb_qu - sky.fg_qu
    res = nilc_clean(total, BEAMS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    spec = nilc_spectra(
        res, total_qu=total, noise_qu=noise, fg_qu=sky.fg_qu, cmb_qu=sky.cmb_qu, f_sky=1.0
    )
    manual = nilc_forecast(spec, **FC_KW)

    out = run_forecast(cfg).as_dict()
    assert set(out) == set(manual)
    for k in manual:
        np.testing.assert_allclose(out[k], manual[k], rtol=1e-9, atol=1e-12)


def test_run_forecast_cmilc_cleaner_runs() -> None:
    cfg = _nilc_config(cleaner=cmilc_cleaner(FREQS, moments=CMILC06_MOMENTS, needlet_peaks=PEAKS))
    out = run_forecast(cfg)
    assert np.isfinite(out.sigma_r_baseline) and out.sigma_r_baseline > 0


@pytest.mark.slow
def test_run_forecast_gnilc_source_with_foreground() -> None:
    pytest.importorskip("pysm3")
    cfg = _nilc_config(
        fg_model="d1s1",
        r_in=0.0,
        w_inv=(1e-4, 1e-4, 1e-4),
        r_fid=0.0,
        residual_source=ResidualTemplateSource.GNILC,
    )
    out = run_forecast(cfg)
    assert np.isfinite(out.sigma_r_baseline) and out.sigma_r_baseline > 0
    assert np.isfinite(out.delta_r)
