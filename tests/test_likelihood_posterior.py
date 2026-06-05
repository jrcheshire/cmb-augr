"""Fast tests for the Phase B prior + posterior assembly (no sampling)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.likelihood import (
    GaussianLikelihood,
    GaussianPrior,
    HLLikelihood,
    PositivityTransform,
    SignalSpectrumModel,
    make_log_posterior,
)
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

FIDUCIAL = {
    "r": 0.01,
    "A_lens": 1.0,
    "A_dust": 4.7,
    "beta_dust": 1.6,
    "alpha_dust": -0.58,
    "T_dust": 19.6,
    "A_sync": 1.5,
    "beta_sync": -3.1,
    "alpha_sync": -0.6,
    "epsilon": 0.0,
    "Delta_dust": 0.0,
}
FIXED = ["T_dust"]


@pytest.fixture(scope="module")
def instrument():
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    return Instrument(
        channels=(
            Channel(90.0, 500, 400.0, 30.0, efficiency=eff),
            Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
            Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
        ),
        mission_duration_years=5.0,
        f_sky=0.7,
    )


@pytest.fixture(scope="module")
def signal_model(instrument):
    return SignalModel(
        instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=30,
        ell_max=130,
        delta_ell=1,
        ell_per_bin_below=1,
    )


@pytest.fixture(scope="module")
def free_names(signal_model):
    return [n for n in signal_model.parameter_names if n not in FIXED]


@pytest.fixture(scope="module")
def model(signal_model):
    return SignalSpectrumModel(signal_model)


@pytest.fixture(scope="module")
def fid_vec(signal_model):
    return flatten_params(FIDUCIAL, signal_model.parameter_names)


class TestPositivityTransform:
    def test_roundtrip(self, free_names):
        t = PositivityTransform.from_names(free_names)  # default positive set
        x = jnp.array([abs(v) + 0.5 for v in range(len(free_names))])  # all > 0
        np.testing.assert_allclose(np.asarray(t.forward(t.inverse(x))), np.asarray(x), rtol=1e-6)

    def test_identity_slots_untouched(self, free_names):
        t = PositivityTransform.from_names(free_names, positive_params=frozenset())
        u = jnp.linspace(-2.0, 2.0, len(free_names))
        np.testing.assert_allclose(np.asarray(t.forward(u)), np.asarray(u))
        assert float(t.forward_log_det_jacobian(u)) == 0.0

    def test_log_det_jacobian_matches_autodiff(self, free_names):
        t = PositivityTransform.from_names(free_names)
        u = jnp.linspace(-1.5, 1.5, len(free_names))
        jac = jax.jacobian(t.forward)(u)  # diagonal (per-param bijector)
        ref = float(jnp.sum(jnp.log(jnp.abs(jnp.diag(jac)))))
        np.testing.assert_allclose(float(t.forward_log_det_jacobian(u)), ref, rtol=1e-6)


class TestGaussianPrior:
    def test_zero_at_means(self, free_names):
        prior = GaussianPrior.from_priors(
            free_names, FIDUCIAL, {"beta_dust": 0.11, "beta_sync": 0.3}
        )
        means = jnp.array([FIDUCIAL[n] for n in free_names])
        assert abs(float(prior.log_prob(means))) < 1e-12

    def test_value_matches_gaussian(self, free_names):
        priors = {"beta_dust": 0.11, "beta_sync": 0.3}
        prior = GaussianPrior.from_priors(free_names, FIDUCIAL, priors)
        x = jnp.array([FIDUCIAL[n] for n in free_names])
        i = free_names.index("beta_dust")
        x = x.at[i].add(0.05)
        expected = -0.5 * (0.05 / 0.11) ** 2
        np.testing.assert_allclose(float(prior.log_prob(x)), expected, rtol=1e-10)


class TestPosterior:
    def test_free_names_exclude_fixed(self, model):
        free = [n for n in model.parameter_names if n not in FIXED]
        prior = GaussianPrior.from_priors(free, FIDUCIAL, {})
        t = PositivityTransform.from_names(free, positive_params=frozenset())
        post = make_log_posterior(
            model, _ZeroLikelihood(), prior, t, fiducial=FIDUCIAL, fixed=FIXED
        )
        assert "T_dust" not in post.free_names
        assert set(post.free_names) == {n for n in model.parameter_names if n != "T_dust"}

    def test_gaussian_grad_finite_everywhere(
        self, signal_model, instrument, model, fid_vec, free_names
    ):
        gauss = GaussianLikelihood.from_forecast(signal_model, instrument, fid_vec)
        prior = GaussianPrior.from_priors(free_names, FIDUCIAL, {})
        t = PositivityTransform.from_names(free_names, positive_params=frozenset())
        post = make_log_posterior(model, gauss, prior, t, fiducial=FIDUCIAL, fixed=FIXED)
        u_fid = post.fiducial_unconstrained(t)
        # Gaussian has no eigh degeneracy: grad finite at the fiducial and off it.
        assert np.all(np.isfinite(np.asarray(jax.grad(post.log_prob)(u_fid))))
        assert np.all(np.isfinite(np.asarray(jax.grad(post.log_prob)(u_fid + 0.3))))

    def test_hl_grad_nan_at_fiducial_finite_off(
        self, signal_model, instrument, model, fid_vec, free_names
    ):
        # The pinned Phase B property: the HL gradient is NaN exactly at the Asimov
        # fiducial (degenerate per-bin eigh) but finite once the chain is off it,
        # which is why the NUTS sampler must init off-fiducial.
        hl = HLLikelihood.from_forecast(signal_model, instrument, fid_vec)
        prior = GaussianPrior.from_priors(free_names, FIDUCIAL, {})
        t = PositivityTransform.from_names(free_names, positive_params=frozenset())
        post = make_log_posterior(model, hl, prior, t, fiducial=FIDUCIAL, fixed=FIXED)
        u_fid = post.fiducial_unconstrained(t)
        assert np.all(np.isnan(np.asarray(jax.grad(post.log_prob)(u_fid))))
        g_off = jax.grad(post.log_prob)(u_fid + 0.3)
        assert np.all(np.isfinite(np.asarray(g_off)))
        # log_prob value is finite at the fiducial (only the gradient degenerates).
        assert np.isfinite(float(post.log_prob(u_fid)))


class _ZeroLikelihood:
    """A trivial Likelihood (log_prob ≡ 0) for structural posterior tests."""

    def log_prob(self, prediction):
        return jnp.asarray(0.0)
