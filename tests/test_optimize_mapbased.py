"""Gate for augr.optimize_mapbased: differentiable map-based sigma(r) from the
noise design vector.

The driver composes ``white_noise_power_continuous`` -> ``mc_cutsky_cov_traced``
-> ``sigma_r_from_external_cov``. The fast test checks the ``w_inv`` helper
against the Channel-based ``white_noise_power`` (so the design-vector -> w_inv
mapping stays consistent with the analytic path) plus a broadcast check. The slow
tests check that the driver reproduces the raw-``w_inv`` path bit-for-bit and that
``jax.grad`` through the noise vector is finite and finite-difference-matched
under common random numbers.

Map work needs jht (the [masking] extra) and ducc0 (the SHTs).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jht")
pytest.importorskip("ducc0")

import jax
import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument, simple_probe
from augr.delensing import load_lensing_spectra
from augr.foregrounds import NullForegroundModel
from augr.instrument import white_noise_power
from augr.optimize import make_optimization_context, sigma_r_from_external_cov
from augr.optimize_mapbased import (
    sigma_r_from_beam_design,
    sigma_r_from_noise_design,
    w_inv_from_noise_design,
)
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context, mc_cutsky_cov_traced

FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)
# A fiducial noise design for the three FREQS bands (per-band detector counts,
# NETs [uK.sqrt(s)], efficiencies). Chosen to land w_inv near the W_INV used in
# test_spectrum_stages so the cleaner/filter are in a sensible regime.
N_DET = (200.0, 400.0, 200.0)
NET = (60.0, 50.0, 80.0)
ETA = (0.5, 0.5, 0.5)
MISSION_YEARS = 4.0


# --- fast --------------------------------------------------------------------


def test_w_inv_matches_channel_white_noise_power() -> None:
    """w_inv_from_noise_design over a preset's channels == per-channel white_noise_power.

    Validates that the design-vector -> w_inv mapping uses the same fields and
    formula as the Channel-based path (n_detectors, net_per_detector,
    efficiency.total)."""
    inst = simple_probe()
    years, f_sky = 5.0, 0.7
    n_det = jnp.array([float(ch.n_detectors) for ch in inst.channels])
    net = jnp.array([ch.net_per_detector for ch in inst.channels])
    eta = jnp.array([ch.efficiency.total for ch in inst.channels])

    got = w_inv_from_noise_design(n_det, net, eta, years, f_sky)
    expected = jnp.array([white_noise_power(ch, years, f_sky) for ch in inst.channels])

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12)


def test_w_inv_broadcasts_scalars() -> None:
    """Scalar efficiency broadcasts against per-band n_det / net."""
    n_det = jnp.array([100.0, 200.0, 300.0])
    net = jnp.array([50.0, 60.0, 70.0])
    out_scalar_eta = w_inv_from_noise_design(n_det, net, 0.5, 4.0, 0.6)
    out_array_eta = w_inv_from_noise_design(n_det, net, jnp.full(3, 0.5), 4.0, 0.6)
    assert out_scalar_eta.shape == (3,)
    np.testing.assert_allclose(np.asarray(out_scalar_eta), np.asarray(out_array_eta))


# --- shared slow setup -------------------------------------------------------


def _priors(lmax):
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    return cl_ee, cl_bb


def _bin_matrix(ell_min, ell_max, delta_ell, ell_per_bin_below, f_sky=0.6):
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=f_sky),
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=ell_min,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    return jnp.asarray(sm.bin_matrix)


def _setup(n_sims, *, nside=16, lmax=24, ell_max=24, delta_ell=8, ell_per_bin_below=2):
    """Build mc_ctx + opt_ctx + cleaner for the tiny CMB-only config (no PySM)."""
    cl_ee, cl_bb = _priors(lmax)
    bm = _bin_matrix(2, ell_max, delta_ell, ell_per_bin_below)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
    )
    cleaner = nilc_cleaner(clean_e=True)
    # Fiducial w_inv for the context's var_pix_ref setup clean; the driver
    # recomputes w_inv from the design and varies it around this.
    w_inv_fid = np.asarray(
        w_inv_from_noise_design(
            jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, 0.6
        )
    )
    mc_ctx = make_cutsky_mc_context(
        cleaner=cleaner,
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=w_inv_fid,
        nside=nside,
        lmax=lmax,
        mask=mk.galactic_mask(nside, 0.6),
        cl_ee=cl_ee,
        cl_bb_prior_unbeamed=cl_bb,
        bin_matrix=bm,
        ell_min=2,
        true_bb_binned=true_b,
        n_sims=n_sims,
        base_seed=0,
        fg_model=None,
        r_in=0.0,
    )
    opt_ctx = make_optimization_context(
        cleaned_map_instrument(f_sky=0.6),
        NullForegroundModel(),
        CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0},
        priors={},
        fixed_params=[],
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    return mc_ctx, opt_ctx, cleaner


# --- slow: driver consistency + end-to-end gradient --------------------------


@pytest.mark.slow
def test_driver_matches_raw_w_inv_path() -> None:
    """sigma_r_from_noise_design == the explicit w_inv -> cov -> sigma_r composition.

    The driver is a thin wrapper; at the fiducial design it must reproduce the
    raw path (same w_inv, same context, same seeds) to fp64."""
    mc_ctx, opt_ctx, cleaner = _setup(12)
    n_det = jnp.asarray(N_DET)
    net = jnp.asarray(NET)
    eta = jnp.asarray(ETA)

    s_driver = float(
        sigma_r_from_noise_design(
            n_det,
            net,
            eta,
            MISSION_YEARS,
            mc_ctx=mc_ctx,
            opt_ctx=opt_ctx,
            cleaner=cleaner,
        )
    )

    # Explicit composition with f_sky tied to the mask (= mc_ctx.f_sky default).
    w_inv = w_inv_from_noise_design(n_det, net, eta, MISSION_YEARS, mc_ctx.f_sky)
    cov = mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner).covariance
    s_raw = float(sigma_r_from_external_cov(cov, opt_ctx))

    assert np.isfinite(s_driver) and s_driver > 0
    np.testing.assert_allclose(s_driver, s_raw, rtol=1e-12)


@pytest.mark.slow
def test_noise_design_grad_finite_and_fd_matched() -> None:
    """End-to-end jax.grad of map-based sigma(r) w.r.t. the noise vector: FD-matched.

    Differentiates through n_det / net / eta_total / mission_years. CRN is fixed
    (mc_ctx.noise_keys), so autodiff and the central finite difference see the
    same sims and must agree. The chain rule adds white_noise_power_continuous on
    top of the already-FD-validated w_inv -> sigma(r) gradient."""
    mc_ctx, opt_ctx, cleaner = _setup(12)
    n_det = jnp.asarray(N_DET)
    net = jnp.asarray(NET)
    eta = jnp.asarray(ETA)
    years = MISSION_YEARS

    def loss(nd, ne, et, yr):
        return sigma_r_from_noise_design(
            nd, ne, et, yr, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner
        )

    s0 = float(loss(n_det, net, eta, years))
    assert np.isfinite(s0) and s0 > 0

    g_ndet, g_net, g_eta, g_years = jax.grad(loss, argnums=(0, 1, 2, 3))(n_det, net, eta, years)
    for g in (g_ndet, g_net, g_eta):
        assert bool(jnp.all(jnp.isfinite(g)))
    assert np.isfinite(float(g_years))

    # Central FD on the first band's NET and on mission_years.
    h_net = 0.05 * float(net[0])
    g_fd_net0 = (
        float(loss(n_det, net.at[0].add(h_net), eta, years))
        - float(loss(n_det, net.at[0].add(-h_net), eta, years))
    ) / (2 * h_net)
    np.testing.assert_allclose(float(g_net[0]), g_fd_net0, rtol=0.05)

    h_yr = 0.05 * years
    g_fd_years = (
        float(loss(n_det, net, eta, years + h_yr)) - float(loss(n_det, net, eta, years - h_yr))
    ) / (2 * h_yr)
    np.testing.assert_allclose(float(g_years), g_fd_years, rtol=0.05)


@pytest.mark.slow
def test_beam_design_grad_finite_and_fd_matched() -> None:
    """End-to-end jax.grad of map-based sigma(r) w.r.t. the per-band beams: FD-matched.

    Differentiates through the per-band FWHM and the shape exponent ``p`` (beamed
    in-trace). CRN is fixed (mc_ctx.noise_keys), so autodiff and the central finite
    difference see the same sims and must agree. Also checks that at the reference beams
    (FWHM = BEAMS, p = 1) the beam-design path reproduces the frozen noise-only path —
    i.e. moving the beaming into the trace did not change the value."""
    mc_ctx, opt_ctx, cleaner = _setup(12)
    w_inv = w_inv_from_noise_design(
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, mc_ctx.f_sky
    )
    fwhm = jnp.asarray(BEAMS)
    p = jnp.ones(len(BEAMS))

    def loss(bf, bp):
        return sigma_r_from_beam_design(
            bf, bp, w_inv=w_inv, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner
        )

    s0 = float(loss(fwhm, p))
    assert np.isfinite(s0) and s0 > 0

    # Reference beams reproduce the noise-only (frozen-beam) path to fp64.
    cov_ref = mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner).covariance
    s_ref = float(sigma_r_from_external_cov(cov_ref, opt_ctx))
    np.testing.assert_allclose(s0, s_ref, rtol=1e-9)

    g_fwhm, g_p = jax.grad(loss, argnums=(0, 1))(fwhm, p)
    assert bool(jnp.all(jnp.isfinite(g_fwhm)))
    assert bool(jnp.all(jnp.isfinite(g_p)))

    # Central FD on band-0 FWHM and band-0 shape exponent (CRN-fixed => identical sims).
    h = 0.05 * float(fwhm[0])
    g_fd_fwhm0 = (float(loss(fwhm.at[0].add(h), p)) - float(loss(fwhm.at[0].add(-h), p))) / (2 * h)
    np.testing.assert_allclose(float(g_fwhm[0]), g_fd_fwhm0, rtol=0.05)

    hp = 0.05
    g_fd_p0 = (float(loss(fwhm, p.at[0].add(hp))) - float(loss(fwhm, p.at[0].add(-hp)))) / (2 * hp)
    np.testing.assert_allclose(float(g_p[0]), g_fd_p0, rtol=0.05)
