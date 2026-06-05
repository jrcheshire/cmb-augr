"""Stage-4 gates for augr.nilc_forecast: spectra extraction + Fisher wiring.

The fast gate (CMB + noise, no FG) checks the transfer ≈ 1, positive post-NILC
noise, zero residual -> zero Δr, and finite σ(r). The PySM gate (foregrounds
actually leak) is slow and confirms a positive residual and a finite Δr.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.nilc import common_resolution_b_alm, nilc_clean
from augr.nilc_forecast import (
    cl_bb,
    nilc_forecast,
    nilc_leakage_correlation,
    nilc_spectra,
)
from augr.noise_sims import noise_maps
from augr.spectra import CMBSpectra

NSIDE, LMAX = 32, 48
FREQS = (30.0, 90.0, 150.0)
BEAMS = [40.0, 20.0, 10.0]
PEAKS = [8, 24, LMAX]


def _decompose(fg_model, *, r_in, w_inv, seed=0):
    """Build total / noise / fg / cmb map sets sharing one realization."""
    sky = generate_band_sky(
        FREQS,
        tuple(BEAMS),
        spectra=CMBSpectra(),
        r_in=r_in,
        nside=NSIDE,
        lmax=LMAX,
        fg_model=fg_model,
        cmb_seed=seed,
    )
    total = assemble_band_maps(
        sky, jnp.asarray(w_inv), jnp.ones(sky.npix), noise_key=jax.random.PRNGKey(seed)
    )
    noise = total - sky.cmb_qu - sky.fg_qu
    return total, noise, sky.fg_qu, sky.cmb_qu


def test_nilc_forecast_cmb_plus_noise() -> None:
    total, noise, fg, cmb = _decompose(None, r_in=0.01, w_inv=[5e-4, 5e-4, 5e-4])
    res = nilc_clean(total, BEAMS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    spec = nilc_spectra(res, total_qu=total, noise_qu=noise, fg_qu=fg, cmb_qu=cmb, f_sky=1.0)

    band = (spec.ells >= 10) & (spec.ells <= 40)
    # Signal is preserved (transfer ~ 1) and the noise spectrum is positive.
    assert abs(np.mean(spec.transfer[band]) - 1.0) < 0.05
    assert np.all(spec.nl_post[band] > 0)
    # No foreground -> exactly zero residual.
    assert np.max(np.abs(spec.cl_residual_fg)) == 0.0

    out = nilc_forecast(
        spec, f_sky=1.0, r_fid=0.01, ell_min=2, ell_max=40, delta_ell=5, ell_per_bin_below=20
    )
    assert np.isfinite(out["sigma_r_baseline"]) and out["sigma_r_baseline"] > 0
    assert abs(out["delta_r"]) < 1e-12  # no FG -> no leakage bias


def test_nilc_forecast_delensing_tightens_sigma_r() -> None:
    """A residual lensing C_ℓ^BB below the full floor must tighten σ(r)."""
    total, noise, fg, cmb = _decompose(None, r_in=0.01, w_inv=[5e-4, 5e-4, 5e-4])
    res = nilc_clean(total, BEAMS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    spec = nilc_spectra(res, total_qu=total, noise_qu=noise, fg_qu=fg, cmb_qu=cmb, f_sky=1.0)

    kw = dict(f_sky=1.0, r_fid=0.01, ell_min=2, ell_max=40, delta_ell=5, ell_per_bin_below=20)
    base = nilc_forecast(spec, **kw)

    ells = spec.ells.astype(float)
    lens_bb = np.asarray(CMBSpectra().cl_bb(jnp.asarray(ells), 0.0))  # full lensing-B floor
    delensed = nilc_forecast(spec, **kw, delensed_bb=0.1 * lens_bb, delensed_bb_ells=ells)
    assert delensed["sigma_r_baseline"] < base["sigma_r_baseline"]

    with pytest.raises(ValueError, match="supplied together"):
        nilc_forecast(spec, **kw, delensed_bb=0.1 * lens_bb)


@pytest.mark.slow
def test_nilc_forecast_with_pysm_foregrounds() -> None:
    pytest.importorskip("pysm3")
    total, noise, fg, cmb = _decompose("d1s1", r_in=0.0, w_inv=[1e-4, 1e-4, 1e-4])
    res = nilc_clean(total, BEAMS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    spec = nilc_spectra(res, total_qu=total, noise_qu=noise, fg_qu=fg, cmb_qu=cmb, f_sky=1.0)

    band = (spec.ells >= 2) & (spec.ells <= 40)
    assert np.sum(spec.cl_residual_fg[band]) > 0.0  # foreground leaks through

    out = nilc_forecast(
        spec, f_sky=1.0, r_fid=0.0, ell_min=2, ell_max=40, delta_ell=5, ell_per_bin_below=20
    )
    assert np.isfinite(out["delta_r"])
    assert out["sigma_r_baseline"] > 0


def test_post_nilc_noise_matches_mv_floor() -> None:
    """Noise-only, equal beams: ILC = inverse-variance combine → flat w/n_band."""
    nside, lmax = 64, 96
    n_band, w = 3, 1.0e-3
    beams = [20.0] * n_band  # equal beams: no inflation, white noise stays flat
    hit = jnp.ones(12 * nside * nside)
    kq, ku = jax.random.split(jax.random.PRNGKey(0))
    w_inv = jnp.array([w] * n_band)
    noise_qu = jnp.stack([noise_maps(hit, w_inv, kq), noise_maps(hit, w_inv, ku)], axis=1)

    res = nilc_clean(noise_qu, beams, lmax=lmax, nside=nside, needlet_peaks=[8, 32, lmax])
    cl = cl_bb(res.cleaned_b_alm, lmax)
    ells = np.arange(lmax + 1)
    band = (ells >= 10) & (ells <= lmax - 10)
    # Independent equal-variance bands -> w/n_band, flat across ℓ.
    np.testing.assert_allclose(np.mean(cl[band]), w / n_band, rtol=0.12)


@pytest.mark.slow
def test_ilc_suppresses_foreground_but_leaves_residual() -> None:
    """3-band ILC removes realistic FG by a large factor, leaving a small residual."""
    pytest.importorskip("pysm3")
    total, _noise, fg, _cmb = _decompose("d1s1", r_in=0.0, w_inv=[1e-4, 1e-4, 1e-4])
    res = nilc_clean(total, BEAMS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    cleaned_fg = res.project(fg)
    fg_common, _ = common_resolution_b_alm(fg, BEAMS, lmax=LMAX, nside=NSIDE, n_iter=3)

    ells = np.arange(LMAX + 1)
    band = (ells >= 2) & (ells <= 30)
    cl_resid = cl_bb(cleaned_fg, LMAX)
    cl_input = cl_bb(fg_common[0], LMAX)  # lowest-freq (most FG) band, common res
    suppression = np.mean(cl_input[band]) / np.mean(cl_resid[band])
    assert suppression > 100.0  # ILC removes most FG (observed ~5000x at this config)
    assert np.mean(cl_resid[band]) > 0.0  # but a residual leaks through

    # The leakage correlation is a valid coefficient (typically small here: the
    # residual is decorrelated spatial-SED-variation leakage, not a rescaled band).
    _ells, rho = nilc_leakage_correlation(res, fg)
    assert np.all(np.isfinite(rho[band])) and np.all(np.abs(rho[band]) <= 1.0 + 1e-9)
