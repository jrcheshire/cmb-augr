"""Tests for optimize.py — differentiable Fisher forecast for instrument optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.config import FIDUCIAL_BK15, simple_probe
from augr.covariance import (
    bandpower_covariance_blocks,
    bandpower_covariance_blocks_from_noise,
)
from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import (
    noise_nl,
    noise_nl_continuous,
    white_noise_power,
    white_noise_power_continuous,
)
from augr.optimize import (
    make_optimization_context,
    sigma_r_from_channels,
    sigma_r_from_design,
)
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra
from augr.telescope import count_pixels, count_pixels_continuous

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def instrument():
    return simple_probe()


@pytest.fixture(scope="module")
def ctx(instrument):
    return make_optimization_context(
        instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        dict(FIDUCIAL_BK15),
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust", "Delta_dust"],
        ell_min=2, ell_max=300, delta_ell=30, ell_per_bin_below=10,
    )


# -----------------------------------------------------------------------
# Step 1: Functional noise consistency
# -----------------------------------------------------------------------

class TestNoiseConsistency:
    def test_white_noise_power_matches(self, instrument):
        """white_noise_power_continuous matches white_noise_power for each channel."""
        for ch in instrument.channels:
            expected = white_noise_power(
                ch, instrument.mission_duration_years, instrument.f_sky)
            got = white_noise_power_continuous(
                ch.net_per_detector, float(ch.n_detectors),
                ch.efficiency.total,
                instrument.mission_duration_years, instrument.f_sky)
            np.testing.assert_allclose(float(got), float(expected), rtol=1e-12)

    def test_noise_nl_matches(self, instrument):
        """noise_nl_continuous matches noise_nl for each channel."""
        ells = jnp.arange(2, 301, dtype=float)
        for ch in instrument.channels:
            expected = noise_nl(
                ch, ells, instrument.mission_duration_years, instrument.f_sky)
            got = noise_nl_continuous(
                ch.net_per_detector, float(ch.n_detectors),
                ch.beam_fwhm_arcmin, ch.efficiency.total,
                ells, instrument.mission_duration_years, instrument.f_sky,
                ch.knee_ell, ch.alpha_knee)
            np.testing.assert_allclose(np.array(got), np.array(expected),
                                       rtol=1e-12)


# -----------------------------------------------------------------------
# Step 2: Covariance consistency
# -----------------------------------------------------------------------

class TestCovarianceConsistency:
    def test_blocks_from_noise_matches(self, instrument):
        """bandpower_covariance_blocks_from_noise matches the Instrument version."""
        fg = GaussianForegroundModel()
        cmb = CMBSpectra()
        sig = SignalModel(instrument, fg, cmb,
                          ell_min=2, ell_max=300, delta_ell=30,
                          ell_per_bin_below=10)
        params = flatten_params(dict(FIDUCIAL_BK15), sig.parameter_names)

        # Reference: from Instrument
        cov_ref = bandpower_covariance_blocks(sig, instrument, params)

        # New: from pre-computed noise arrays
        ells = sig.ells
        noise_nls = jnp.stack([
            noise_nl(ch, ells, instrument.mission_duration_years,
                     instrument.f_sky)
            for ch in instrument.channels
        ])
        cov_new = bandpower_covariance_blocks_from_noise(
            sig, noise_nls, instrument.f_sky, params)

        np.testing.assert_allclose(np.array(cov_new), np.array(cov_ref),
                                   rtol=1e-10)


# -----------------------------------------------------------------------
# Step 3: count_pixels_continuous
# -----------------------------------------------------------------------

class TestCountPixelsContinuous:
    def test_matches_at_integers(self):
        """Continuous version matches discrete when result is an integer."""
        # 80 cells * packing 1.0 = exactly 80 pixels
        discrete = count_pixels(80e-4, 1e-4, 1.0)
        continuous = count_pixels_continuous(80e-4, 1e-4, 1.0)
        assert discrete == 80
        np.testing.assert_allclose(float(continuous), 80.0, rtol=1e-10)

    def test_continuous_is_smooth(self):
        """Continuous version returns non-integer values."""
        val = count_pixels_continuous(85e-4, 1e-4, 1.0)
        assert float(val) == pytest.approx(85.0, abs=0.01)

    def test_non_negative(self):
        """Returns 0 for negative area ratios."""
        val = count_pixels_continuous(-1.0, 1e-4, 1.0)
        assert float(val) == 0.0


# -----------------------------------------------------------------------
# Step 4: End-to-end sigma(r) consistency
# -----------------------------------------------------------------------

class TestSigmaRConsistency:
    def test_matches_fisher_forecast(self, instrument, ctx):
        """sigma_r_from_channels approximately matches FisherForecast.sigma('r').

        The optimize path uses jnp.linalg.solve for gradient stability,
        while FisherForecast uses eigendecomposition with eigenvalue zeroing.
        For instruments with high condition-number covariance blocks (~10^17),
        these differ by a few percent due to near-degenerate directions.
        """
        ff = FisherForecast(
            ctx.signal_model, instrument, dict(FIDUCIAL_BK15),
            priors={"beta_dust": 0.11, "beta_sync": 0.3},
            fixed_params=["T_dust", "Delta_dust"],
        )
        sigma_ref = ff.sigma("r")

        sigma_opt = sigma_r_from_channels(
            ctx.n_det, ctx.net, ctx.beam, ctx.eta,
            ctx,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
        )

        np.testing.assert_allclose(float(sigma_opt), sigma_ref, rtol=0.05)


# -----------------------------------------------------------------------
# Step 5: Gradient tests
# -----------------------------------------------------------------------

class TestGradients:
    def test_gradient_exists(self, ctx, instrument):
        """jax.grad(sigma_r) w.r.t. n_det is finite and nonzero."""
        grad_fn = jax.grad(sigma_r_from_channels, argnums=0)
        grads = grad_fn(
            ctx.n_det, ctx.net, ctx.beam, ctx.eta,
            ctx,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
        )
        assert jnp.all(jnp.isfinite(grads)), f"Non-finite gradients: {grads}"
        assert jnp.any(grads != 0.0), "All gradients are zero"

    def test_gradient_sign_n_det(self, ctx, instrument):
        """More detectors should decrease sigma(r): d(sigma_r)/d(n_det) < 0."""
        grad_fn = jax.grad(sigma_r_from_channels, argnums=0)
        grads = grad_fn(
            ctx.n_det, ctx.net, ctx.beam, ctx.eta,
            ctx,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
        )
        # All gradients should be negative (more detectors = lower sigma(r))
        assert jnp.all(grads < 0), (
            f"Expected all negative, got {np.array(grads)}")

    def test_gradient_vs_finite_differences(self, ctx, instrument):
        """Analytical gradient matches central finite differences.

        Uses a multiplicative step (10% of n_det) to get clean FD estimates.
        The simple_probe has O(10-100) detectors per channel and gradients
        ~1e-7, so small absolute steps produce FD noise from matrix
        inversion precision.
        """
        def loss(n_det):
            return sigma_r_from_channels(
                n_det, ctx.net, ctx.beam, ctx.eta,
                ctx,
                mission_years=instrument.mission_duration_years,
                f_sky=instrument.f_sky,
            )

        grad_analytical = jax.grad(loss)(ctx.n_det)

        # Central finite differences with 10% relative step
        grad_fd = jnp.zeros_like(ctx.n_det)
        for i in range(len(ctx.n_det)):
            h = 0.1 * float(ctx.n_det[i])
            n_det_plus = ctx.n_det.at[i].add(h)
            n_det_minus = ctx.n_det.at[i].add(-h)
            grad_fd = grad_fd.at[i].set(
                (float(loss(n_det_plus)) - float(loss(n_det_minus))) / (2 * h))

        np.testing.assert_allclose(
            np.array(grad_analytical), np.array(grad_fd), rtol=0.02,
            err_msg="Analytical gradient disagrees with finite differences")

    def test_gradient_net(self, ctx, instrument):
        """Gradient w.r.t. NET exists and is positive (higher NET = worse)."""
        grad_fn = jax.grad(sigma_r_from_channels, argnums=1)
        grads = grad_fn(
            ctx.n_det, ctx.net, ctx.beam, ctx.eta,
            ctx,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
        )
        assert jnp.all(jnp.isfinite(grads))
        assert jnp.all(grads > 0), (
            f"Expected all positive (higher NET = worse), got {np.array(grads)}")

    def test_gradient_beam(self, ctx, instrument):
        """Gradient w.r.t. beam FWHM exists and is finite."""
        grad_fn = jax.grad(sigma_r_from_channels, argnums=2)
        grads = grad_fn(
            ctx.n_det, ctx.net, ctx.beam, ctx.eta,
            ctx,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
        )
        assert jnp.all(jnp.isfinite(grads))


# -----------------------------------------------------------------------
# Step 6: JIT compilation
# -----------------------------------------------------------------------

class TestJIT:
    def test_jit_compiles(self, ctx, instrument):
        """sigma_r_from_channels works under jax.jit."""
        from functools import partial

        jitted = jax.jit(
            partial(sigma_r_from_channels,
                    ctx=ctx,
                    mission_years=instrument.mission_duration_years,
                    f_sky=instrument.f_sky),
        )
        result = jitted(ctx.n_det, ctx.net, ctx.beam, ctx.eta)
        assert jnp.isfinite(result)
        assert float(result) > 0


# -----------------------------------------------------------------------
# Step 7: Tier 2 — design-level optimization
# -----------------------------------------------------------------------

class TestDesignLevel:
    def test_sigma_r_from_design_runs(self, ctx, instrument):
        """sigma_r_from_design produces a finite result."""
        # Probe design: 3 dichroic groups
        freqs_per_group = ((30., 40.), (85., 150.), (220., 340.))
        area_fracs = jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        result = sigma_r_from_design(
            aperture_m=1.5,
            f_number=2.0,
            fp_diameter_m=0.4,
            area_fractions=area_fracs,
            ctx=ctx,
            freqs_per_group=freqs_per_group,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
            net_override=ctx.net,  # use pre-computed NETs
            eta_total=ctx.eta,
        )
        assert jnp.isfinite(result)
        assert float(result) > 0

    def test_area_fraction_gradient(self, ctx, instrument):
        """Gradient w.r.t. area_fractions exists and is finite."""
        freqs_per_group = ((30., 40.), (85., 150.), (220., 340.))
        area_fracs = jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        def loss(fracs):
            return sigma_r_from_design(
                aperture_m=1.5,
                f_number=2.0,
                fp_diameter_m=0.4,
                area_fractions=fracs,
                ctx=ctx,
                freqs_per_group=freqs_per_group,
                mission_years=instrument.mission_duration_years,
                f_sky=instrument.f_sky,
                net_override=ctx.net,
                eta_total=ctx.eta,
            )

        grads = jax.grad(loss)(area_fracs)
        assert jnp.all(jnp.isfinite(grads)), f"Non-finite gradients: {grads}"
        assert jnp.any(grads != 0.0), "All gradients are zero"
