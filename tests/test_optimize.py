"""Tests for optimize.py — differentiable Fisher forecast for instrument optimization."""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.config import FIDUCIAL_BK15, cleaned_map_instrument, simple_probe
from augr.covariance import (
    bandpower_covariance_blocks,
    bandpower_covariance_blocks_from_noise,
)
from augr.delensing import load_lensing_spectra
from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel, NullForegroundModel
from augr.instrument import (
    noise_nl,
    noise_nl_continuous,
    white_noise_power,
    white_noise_power_continuous,
)
from augr.optimize import (
    DelensCoupling,
    _combined_white_nl_bb,
    _delens_from_combined_bb,
    make_optimization_context,
    sigma_r_from_channels,
    sigma_r_from_design,
    sigma_r_from_external_cov,
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


# -----------------------------------------------------------------------
# Step 8: External-covariance sigma(r) (cut-sky masked-Wiener MC consumer)
# -----------------------------------------------------------------------

class TestExternalCovSigmaR:
    """sigma_r_from_external_cov: the jnp-returning sigma(r) for the dense
    external-covariance path (cut-sky masked-Wiener Monte-Carlo covariance),
    the differentiable consumer end of the end-to-end map-based optimization."""

    @pytest.fixture(scope="class")
    def cleaned_ctx(self):
        inst = cleaned_map_instrument(f_sky=0.6)
        ctx = make_optimization_context(
            inst,
            NullForegroundModel(),
            CMBSpectra(),
            {"r": 0.0, "A_lens": 1.0},
            priors={},
            fixed_params=[],
            ell_min=2, ell_max=30, delta_ell=5, ell_per_bin_below=2,
        )
        return ctx, inst

    @staticmethod
    def _random_cov(n_data, seed=0):
        """A well-conditioned symmetric positive-definite (n_data, n_data) cov."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n_data, n_data))
        return jnp.asarray(A @ A.T + n_data * np.eye(n_data))

    def test_matches_fisher_forecast(self, cleaned_ctx):
        """fp64 parity with FisherForecast(external_covariance=...).sigma('r').

        Both route through the same prewhitened dense solve
        (fisher._fisher_from_full); the optimize path just keeps the JAX array
        instead of casting to float in sigma()."""
        ctx, inst = cleaned_ctx
        n_data = len(ctx.signal_model.freq_pairs) * ctx.signal_model.n_bins
        cov = self._random_cov(n_data)

        ff = FisherForecast(
            ctx.signal_model, inst, {"r": 0.0, "A_lens": 1.0},
            priors={}, fixed_params=[], external_covariance=cov,
        )
        ff.compute()
        sigma_ref = ff.sigma("r")
        sigma_opt = float(sigma_r_from_external_cov(cov, ctx))
        np.testing.assert_allclose(sigma_opt, sigma_ref, rtol=1e-10)

    def test_gradient_finite_and_matches_fd(self, cleaned_ctx):
        """jax.grad w.r.t. a covariance scaling is finite, matches central FD,
        and hits the closed form: with priors off, scaling cov by s scales F by
        1/s, so sigma(r)(s) = sigma0 * sqrt(s) and d(sigma_r)/ds|_{s=1} = sigma0/2."""
        ctx, _ = cleaned_ctx
        n_data = len(ctx.signal_model.freq_pairs) * ctx.signal_model.n_bins
        cov0 = self._random_cov(n_data)

        def loss(s):
            return sigma_r_from_external_cov(s * cov0, ctx)

        sigma0 = float(loss(1.0))
        g = float(jax.grad(loss)(1.0))
        assert np.isfinite(g)

        h = 1e-4
        g_fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2 * h)
        np.testing.assert_allclose(g, g_fd, rtol=1e-4)
        # Closed form (priors off): grad = sigma0 / 2.
        np.testing.assert_allclose(g, 0.5 * sigma0, rtol=1e-6)

    def test_jit(self, cleaned_ctx):
        """Runs under jax.jit — the traceability prerequisite shared with grad."""
        from functools import partial

        ctx, _ = cleaned_ctx
        n_data = len(ctx.signal_model.freq_pairs) * ctx.signal_model.n_bins
        cov = self._random_cov(n_data)
        jitted = jax.jit(partial(sigma_r_from_external_cov, ctx=ctx))
        out = jitted(cov)
        assert jnp.isfinite(out)
        assert float(out) > 0


# -----------------------------------------------------------------------
# Self-consistent delensing in the design forward (issue #45 Stage 2)
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def lensing():
    return load_lensing_spectra()


# Small forecast + delensing config (fast gate).
_S2_SIG_KW = dict(ell_min=2, ell_max=300, delta_ell=30, ell_per_bin_below=10)
_S2_DELENS_KW = dict(delens_l_max_qe=300, delens_n_iter=2)


class TestDelensedDesignForward:
    """Design-dependent delensing wired into the analytic forward (#45 Stage 2)."""

    def _common(self, instrument):
        return dict(
            instrument=instrument,
            foreground_model=GaussianForegroundModel(),
            cmb_spectra=CMBSpectra(),
            fiducial_params=dict(FIDUCIAL_BK15),
            fixed_params=["T_dust", "Delta_dust"],
        )

    def _eval(self, ctx, instrument, n_det=None):
        return sigma_r_from_channels(
            ctx.n_det if n_det is None else n_det,
            ctx.net, ctx.beam, ctx.eta, ctx,
            mission_years=instrument.mission_duration_years,
            f_sky=instrument.f_sky,
        )

    def test_off_is_default_and_fields_none(self, instrument):
        """delens=None (default) leaves the delensing context empty."""
        ctx = make_optimization_context(**self._common(instrument), **_S2_SIG_KW)
        assert ctx.delens_mode is None
        assert ctx.delens_cl_bb_res0 is None
        assert ctx.delens_jac is None

    def test_bad_mode_raises(self, instrument, lensing):
        with pytest.raises(ValueError, match="delens must be"):
            make_optimization_context(
                **self._common(instrument), delens="nope",
                lensing_spectra=lensing, **_S2_SIG_KW)

    def test_recompute_reproduces_frozen_and_tightens(self, instrument, lensing):
        """recompute at the reference design == frozen-delensed forecast, and
        delensing tightens sigma(r) vs the A_lens=1 forecast."""
        ctx_rec = make_optimization_context(
            **self._common(instrument), delens="recompute",
            lensing_spectra=lensing, **_S2_DELENS_KW, **_S2_SIG_KW)
        # A_lens drops from the parameter vector in delensed mode.
        assert "A_lens" not in ctx_rec.signal_model.parameter_names

        ctx_frozen = make_optimization_context(
            **self._common(instrument),
            delensed_bb=ctx_rec.delens_cl_bb_res0,
            delensed_bb_ells=ctx_rec.delens_ls, **_S2_SIG_KW)
        s_rec = float(self._eval(ctx_rec, instrument))
        s_frozen = float(self._eval(ctx_frozen, instrument))
        # Exact: the reference recompute reproduces the frozen residual.
        np.testing.assert_allclose(s_rec, s_frozen, rtol=1e-9)

        ctx_alens = make_optimization_context(
            **self._common(instrument), **_S2_SIG_KW)
        assert s_rec < float(self._eval(ctx_alens, instrument))

    def test_recompute_grad_finite_and_sign(self, instrument, lensing):
        """grad wrt n_det is finite and negative (more detectors -> lower sigma)."""
        ctx = make_optimization_context(
            **self._common(instrument), delens="recompute",
            lensing_spectra=lensing, **_S2_DELENS_KW, **_S2_SIG_KW)
        g = jax.grad(lambda nd: self._eval(ctx, instrument, n_det=nd))(ctx.n_det)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(g < 0)

    def test_delens_chain_grad_matches_fd(self, instrument, lensing):
        """Rigorous gradient check on the design -> delensing chain, isolated
        from the ill-conditioned Fisher inversion.

        The scalar is sum(cl_bb_res) as a function of n_det, routed through the
        exact same combined-white-noise -> QE-residual chain the forward uses.
        Central FD along the design direction (multiplicative n_det step)
        converges to autodiff as h shrinks (the sum(cl_bb_res) map is smooth;
        only the downstream sigma(r) inv() carries the known FD-noise floor).
        """
        ctx = make_optimization_context(
            **self._common(instrument), delens="recompute",
            lensing_spectra=lensing, **_S2_DELENS_KW, **_S2_SIG_KW)
        my = instrument.mission_duration_years
        fsky = instrument.f_sky

        def f(n_det):
            nl_bb = _combined_white_nl_bb(
                n_det, ctx.net, ctx.beam, ctx.eta, ctx.delens_ells, my, fsky)
            return jnp.sum(_delens_from_combined_bb(
                lensing, nl_bb, ctx.delens_ls,
                ctx.delens_l_max_qe, ctx.delens_n_iter))

        g = jax.grad(f)(ctx.n_det)
        assert jnp.all(jnp.isfinite(g))
        ad = float(jnp.dot(g, ctx.n_det))         # directional along n_det
        # sum(cl_bb_res) at l_max_qe=300 is strongly curved and carries a
        # ~1e-7 fp64 floor from the lax.scan reductions, so central FD is only
        # reliable at a small step (h ~ 1e-5); larger h is curvature/noise
        # dominated. At h = 1e-5 FD reproduces autodiff to < 1e-3.
        h = 1e-5
        fd = float((f(ctx.n_det * (1 + h)) - f(ctx.n_det * (1 - h))) / (2 * h))
        np.testing.assert_allclose(fd, ad, rtol=1e-3)

    @pytest.mark.slow
    def test_linearized_matches_recompute_at_reference(self, instrument, lensing):
        """linearized == recompute at the reference (nl_bb - nl_bb0 = 0), and
        its gradient is finite. Uses a small ell range so the jacrev precompute
        (O(n_ls) delensing solves) stays cheap."""
        sig_kw = dict(ell_min=2, ell_max=60, delta_ell=10, ell_per_bin_below=10)
        dkw = dict(delens_l_max_qe=200, delens_n_iter=2,
                   delens_ls=jnp.arange(2, 61, dtype=float))
        ctx_rec = make_optimization_context(
            **self._common(instrument), delens="recompute",
            lensing_spectra=lensing, **dkw, **sig_kw)
        ctx_lin = make_optimization_context(
            **self._common(instrument), delens="linearized",
            lensing_spectra=lensing, **dkw, **sig_kw)
        s_rec = float(self._eval(ctx_rec, instrument))
        s_lin = float(self._eval(ctx_lin, instrument))
        np.testing.assert_allclose(s_lin, s_rec, rtol=1e-9)
        g = jax.grad(
            lambda nd: self._eval(ctx_lin, instrument, n_det=nd))(ctx_lin.n_det)
        assert jnp.all(jnp.isfinite(g))


# --- DelensCoupling: the design-keyed residual for the map-based forward ------


@pytest.fixture(scope="module")
def _coupling_design():
    """A tiny 3-band reference design + a light QE config (keeps the solve fast)."""
    return dict(
        n_det=jnp.asarray((200.0, 400.0, 200.0)),
        net=jnp.asarray((60.0, 50.0, 80.0)),
        beam=jnp.asarray((40.0, 30.0, 20.0)),
        eta=jnp.asarray((0.5, 0.5, 0.5)),
        mission_years=4.0,
        f_sky=0.6,
    )


@pytest.mark.slow
def test_delens_coupling_reference_is_exact(_coupling_design):
    """residual() at the reference design reproduces cl_bb_res0 to the last bit."""
    d = _coupling_design
    c = DelensCoupling.build(
        lensing_spectra=load_lensing_spectra(), l_max_qe=500, n_iter=2,
        ls=jnp.arange(2, 30, dtype=float), **d,
    )
    got = c.residual(d["n_det"], d["net"], d["beam"], d["eta"],
                     d["mission_years"], d["f_sky"])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(c.cl_bb_res0))


@pytest.mark.slow
def test_delens_coupling_deeper_design_delenses_better(_coupling_design):
    """More detectors -> lower noise -> a smaller residual, at every multipole.

    The monotonicity is the physics the design forward is meant to see; a coupling
    that ignored its arguments would return the reference residual and pass a
    "finite and positive" check.
    """
    d = _coupling_design
    c = DelensCoupling.build(
        lensing_spectra=load_lensing_spectra(), l_max_qe=500, n_iter=2,
        ls=jnp.arange(2, 30, dtype=float), **d,
    )
    deeper = c.residual(4.0 * d["n_det"], d["net"], d["beam"], d["eta"],
                        d["mission_years"], d["f_sky"])
    ref = np.asarray(c.cl_bb_res0)
    assert np.all(np.asarray(deeper) < ref)
    # and it is a real delensing, not a rounding-level move
    assert np.median(np.asarray(deeper) / ref) < 0.97


@pytest.mark.slow
def test_delens_coupling_residual_is_below_full_lensing(_coupling_design):
    """The residual is a *fraction* of the lensing BB it replaces -- 0 < res < C_lens."""
    d = _coupling_design
    spec = load_lensing_spectra()
    ls = jnp.arange(2, 30, dtype=float)
    c = DelensCoupling.build(lensing_spectra=spec, l_max_qe=500, n_iter=2, ls=ls, **d)
    res = np.asarray(c.cl_bb_res0)
    full = np.asarray(spec.cl_bb_len[2:30])
    assert np.all(res > 0.0)
    assert np.all(res < full)


@pytest.mark.slow
def test_delens_coupling_accepts_scalar_eta(_coupling_design):
    """A SCALAR eta must work: it is what the physical entry point defaults to.

    ``physical_design_objective``'s ``eta_total`` defaults to the scalar 0.5 and is
    forwarded verbatim into ``DelensCoupling.residual``, but every other test here
    passes a per-channel array. ``_combined_white_nl_bb`` indexes ``eta[i]``, so the
    scalar raised ``IndexError: array is 0-dimensional`` and ``delens=`` was
    unusable from the physical entry point -- the exact combination a design run
    uses. Broadcasting a scalar must agree with passing it per channel.
    """
    d = dict(_coupling_design)
    eta_scalar = 0.5
    assert np.allclose(np.asarray(d["eta"]), eta_scalar), "fixture eta must be uniform"
    kw = dict(lensing_spectra=load_lensing_spectra(), l_max_qe=500, n_iter=2,
              ls=jnp.arange(2, 30, dtype=float))

    c_arr = DelensCoupling.build(**kw, **d)
    c_sca = DelensCoupling.build(**kw, **{**d, "eta": eta_scalar})
    np.testing.assert_allclose(np.asarray(c_sca.cl_bb_res0),
                               np.asarray(c_arr.cl_bb_res0), rtol=0, atol=0)

    # and through residual(), which is the per-design eval path
    got = c_sca.residual(d["n_det"], d["net"], d["beam"], eta_scalar,
                         d["mission_years"], d["f_sky"])
    np.testing.assert_allclose(np.asarray(got), np.asarray(c_arr.cl_bb_res0),
                               rtol=0, atol=0)


@pytest.mark.slow
def test_delens_coupling_remat_flag_is_transparent(_coupling_design):
    """``remat`` must reach BOTH build() and residual(), and change nothing.

    The failure mode this guards is asymmetry, not arithmetic: remat is
    forward-transparent, so wiring it into ``build`` but not ``residual`` (or
    vice versa) leaves one path on the O(l_max_qe**2) reverse-mode tape while
    every value assertion in this file still passes. Compare both halves.
    """
    d = _coupling_design
    kw = dict(lensing_spectra=load_lensing_spectra(), l_max_qe=300, n_iter=2,
              ls=jnp.arange(2, 30, dtype=float))
    c_on = DelensCoupling.build(**kw, **d, remat=True)
    c_off = DelensCoupling.build(**kw, **d, remat=False)

    assert c_on.remat is True and c_off.remat is False
    np.testing.assert_array_equal(np.asarray(c_on.cl_bb_res0),
                                  np.asarray(c_off.cl_bb_res0))

    args = (d["n_det"], d["net"], d["beam"], d["eta"],
            d["mission_years"], d["f_sky"])
    np.testing.assert_array_equal(np.asarray(c_on.residual(*args)),
                                  np.asarray(c_off.residual(*args)))


class TestCombinedWhiteNoiseOverflow:
    """``_combined_white_nl_bb`` accumulates ``b_l**2 / w_inv``, not ``1 / nl_i``.

    Regression for the NaN that killed a 96-design Vista EIG production run
    (job 945271): every gradient came back non-finite while every *value* was
    exact, and the only symptom was ``np.linalg.eigh`` raising "Eigenvalues did
    not converge" seven hours in.
    """

    ELLS = jnp.arange(0.0, 5001.0)
    #: 20 GHz on a ~1.3 m aperture. The lowest band of a probe-class design has
    #: the largest beam, so it is the one whose b_l**2 underflows first.
    BIG_BEAM = 47.83
    ARGS: ClassVar[dict] = dict(mission_years=4.0, f_sky=0.6)

    @staticmethod
    def _naive(n_det, net, beam, eta, ells, mission_years, f_sky):
        """The pre-fix accumulation, kept as the negative control."""
        inv = jnp.zeros_like(ells)
        for i in range(n_det.shape[0]):
            inv = inv + 1.0 / noise_nl_continuous(
                net[i], n_det[i], beam[i], eta[i], ells, mission_years, f_sky, 0.0, 1.0)
        return 1.0 / inv

    def _design(self, big_beam):
        return (jnp.array([22.4, 2.1e4]),          # n_det
                jnp.array([49.8, 61.2]),           # net
                jnp.array([big_beam, 1.56]),       # beam: one huge, one small
                jnp.array([0.5, 0.5]))             # eta

    def test_large_beam_gradient_is_finite(self):
        """The whole point: a large-beam channel must not poison the gradient."""
        n_det, net, beam, eta = self._design(self.BIG_BEAM)

        def f(b):
            return jnp.sum(_combined_white_nl_bb(
                n_det, net, b, eta, self.ELLS, **self.ARGS))

        val = f(beam)
        grad = jax.grad(f)(beam)
        assert jnp.isfinite(val), "forward was never the broken half"
        assert jnp.all(jnp.isfinite(grad)), f"non-finite beam gradient: {grad}"

    def test_naive_accumulation_is_the_negative_control(self):
        """Pin the mechanism: forming ``nl_i = w_inv / b_l**2`` overflows, and the
        backward pass then multiplies ``-1/nl**2 -> -0`` by ``dnl/dbeam -> inf``.
        Value finite, gradient NaN -- exactly the production signature.
        """
        n_det, net, beam, eta = self._design(self.BIG_BEAM)

        def f_naive(b):
            return jnp.sum(self._naive(n_det, net, b, eta, self.ELLS, **self.ARGS))

        assert jnp.isfinite(f_naive(beam)), "the naive form's *value* is fine"
        assert not jnp.all(jnp.isfinite(jax.grad(f_naive)(beam))), (
            "the naive accumulation no longer overflows -- if fp64 or the beam "
            "convention changed, re-derive BIG_BEAM/ELLS rather than deleting this"
        )

    def test_matches_naive_form_where_nothing_overflows(self):
        """Away from the overflow the two forms are the same number: the fix is a
        reassociation of a division, not a change of physics.
        """
        n_det, net, beam, eta = self._design(big_beam=8.0)
        ells = jnp.arange(0.0, 1001.0)
        got = _combined_white_nl_bb(n_det, net, beam, eta, ells, **self.ARGS)
        want = self._naive(n_det, net, beam, eta, ells, **self.ARGS)
        assert jnp.all(jnp.isfinite(want)), "control regime must not overflow"
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-14)
