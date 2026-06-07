"""Tests for augr.likelihood.profile — Hessian-at-MLE + profile-likelihood σ.

Two tiers:

* **Fast** synthetic tests against analytic ground truth (need only ``optax``):
  the Hessian-at-MLE recovers a known precision matrix, and the profile σ of a
  correlated multivariate Gaussian equals its *marginal* σ (profiling a Gaussian
  gives the marginal — the off-diagonal correlation makes this distinct from the
  conditional, so the test isn't measure-zero).
* **Slow** posterior tests: Gaussian-likelihood profile σ(r) and Hessian-at-MLE
  σ(r) agree with the Knox ``FisherForecast``; the Hamimeche-Lewis profile σ(r) is
  *wider* than Fisher at the low-mode bump — the same non-Gaussian widening as
  HL-NUTS, with no sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax")

from augr.config import DEFAULT_PRIORS, FIDUCIAL_BK15
from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.likelihood import (
    FisherAtMLE,
    GaussianLikelihood,
    GaussianPrior,
    HLLikelihood,
    PositivityTransform,
    SignalSpectrumModel,
    compute_fisher_at_mle,
    compute_profile_sigma,
    draw_fisher_inits,
    make_log_posterior,
    run_mle_search,
)
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

# ---------------------------------------------------------------------------
# Fast synthetic tests (analytic ground truth)
# ---------------------------------------------------------------------------


def test_fisher_at_mle_recovers_precision():
    # -log_prob = 0.5 dᵀ P d  ⇒  Hessian = P, cov = inv(P), gradient = 0 at the mode.
    precision = jnp.array([[4.0, 1.0, 0.5], [1.0, 2.0, -0.3], [0.5, -0.3, 1.0]])
    mu = jnp.array([0.5, -1.0, 2.0])

    def log_prob(x):
        d = x - mu
        return -0.5 * d @ precision @ d

    res = compute_fisher_at_mle(log_prob, mu, ["a", "b", "c"])
    assert isinstance(res, FisherAtMLE)
    cov = np.asarray(jnp.linalg.inv(precision))
    np.testing.assert_allclose(np.asarray(res.hessian), np.asarray(precision), atol=1e-6)
    np.testing.assert_allclose(np.asarray(res.cov), cov, atol=1e-6)
    np.testing.assert_allclose(np.asarray(res.sigmas), np.sqrt(np.diag(cov)), rtol=1e-6)
    assert np.all(np.abs(np.asarray(res.gradient)) < 1e-6)


def test_profile_sigma_equals_marginal_on_correlated_gaussian():
    cov = jnp.array([[1.0, 0.6, 0.3], [0.6, 2.0, 0.5], [0.3, 0.5, 0.5]])  # PD
    precision = jnp.linalg.inv(cov)
    mu = jnp.array([0.1, -0.2, 0.3])

    def log_prob(x):
        d = x - mu
        return -0.5 * d @ precision @ d

    names = ["a", "b", "c"]
    transform = PositivityTransform.from_names(names, positive_params=frozenset())  # identity
    fisher_sigma = jnp.sqrt(jnp.diag(cov))
    sig = compute_profile_sigma(
        log_prob, names, "a", mu, fisher_sigma, transform, key=jax.random.PRNGKey(0)
    )
    cov_np = np.asarray(cov)
    marginal = float(np.sqrt(cov_np[0, 0]))  # profiling a Gaussian → the marginal
    conditional = float(1.0 / np.sqrt(np.asarray(precision)[0, 0]))
    np.testing.assert_allclose(sig, marginal, rtol=3e-2)
    # Non-trivial: the profile is the marginal, materially distinct from the
    # conditional (so a diagonal-cov version wouldn't have caught a profile/condition mixup).
    assert abs(marginal - conditional) / conditional > 0.1


def test_profile_sigma_return_curve_and_guard():
    mu = jnp.zeros(2)

    def log_prob(x):
        return -0.5 * (x @ x)

    names = ["a", "b"]
    transform = PositivityTransform.from_names(names, positive_params=frozenset())
    sig, curve = compute_profile_sigma(
        log_prob,
        names,
        "a",
        mu,
        jnp.ones(2),
        transform,
        n_grid=11,
        key=jax.random.PRNGKey(0),
        return_curve=True,
    )
    np.testing.assert_allclose(sig, 1.0, rtol=3e-2)
    assert set(curve) == {"grid", "logp_profile", "fit_window_mask", "x0", "half_width"}
    assert curve["grid"].shape == (11,)
    with pytest.raises(ValueError):
        compute_profile_sigma(
            log_prob, names, "z", mu, jnp.ones(2), transform, key=jax.random.PRNGKey(0)
        )


# ---------------------------------------------------------------------------
# Slow posterior tests
# ---------------------------------------------------------------------------

FIXED = ["T_dust"]
R_FID = 0.01


def _setup(likelihood_cls, ell_max, delta_ell):
    """Build (post, transform, fisher_cov, fisher_sigma, x_mle, free_names, fisher_sigma_r)."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    inst = Instrument(
        channels=(
            Channel(90.0, 500, 400.0, 30.0, efficiency=eff),
            Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
            Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
        ),
        mission_duration_years=5.0,
        f_sky=0.6,
    )
    fid = {**FIDUCIAL_BK15, "r": R_FID}
    sm = SignalModel(
        inst,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
    )
    model = SignalSpectrumModel(sm)
    fid_vec = flatten_params(fid, sm.parameter_names)
    free_names = [n for n in sm.parameter_names if n not in FIXED]

    likelihood = likelihood_cls.from_forecast(sm, inst, fid_vec)
    prior = GaussianPrior.from_priors(free_names, fid, DEFAULT_PRIORS)
    transform = PositivityTransform.from_names(free_names, positive_params=frozenset())
    post = make_log_posterior(model, likelihood, prior, transform, fiducial=fid, fixed=FIXED)

    ff = FisherForecast(sm, inst, fid, priors=DEFAULT_PRIORS, fixed_params=FIXED)
    fisher_cov = jnp.asarray(np.linalg.inv(np.asarray(ff.compute())))
    fisher_sigma = jnp.sqrt(jnp.diag(fisher_cov))

    x_fid = post.fiducial_full[post.free_idx]
    inits = draw_fisher_inits(x_fid, fisher_cov, transform, jax.random.PRNGKey(0), 4, scale=0.3)
    mle = run_mle_search(post.log_prob, inits)
    return {
        "post": post,
        "transform": transform,
        "fisher_sigma": fisher_sigma,
        "x_mle": mle.best.x,
        "free_names": post.free_names,
        "sigma_fisher_r": float(ff.sigma("r")),
    }


@pytest.mark.slow
def test_gaussian_profile_and_hessian_match_fisher():
    s = _setup(GaussianLikelihood, ell_max=200, delta_ell=10)
    prof_r = compute_profile_sigma(
        s["post"].log_prob,
        s["free_names"],
        "r",
        s["x_mle"],
        s["fisher_sigma"],
        s["transform"],
        key=jax.random.PRNGKey(1),
    )
    fam = compute_fisher_at_mle(s["post"].log_prob, s["x_mle"], s["free_names"])
    r_idx = list(s["free_names"]).index("r")
    hess_r = float(fam.sigmas[r_idx])
    # Gaussian likelihood → quadratic posterior: profile σ(r) == Hessian σ(r)
    # (both the full-covariance Gaussian curvature).
    np.testing.assert_allclose(prof_r, hess_r, rtol=0.05)
    # Both agree with the Knox FisherForecast; the residual gap is the per-bin
    # block (FisherForecast) vs dense (likelihood) Knox covariance at wide bins.
    np.testing.assert_allclose(prof_r, s["sigma_fisher_r"], rtol=0.12)
    np.testing.assert_allclose(hess_r, s["sigma_fisher_r"], rtol=0.12)


@pytest.mark.slow
def test_hl_profile_wider_than_fisher_at_bump():
    # Low-mode bump config: the HL likelihood is non-Gaussian → profile σ(r) wider
    # than the Knox curvature (the sampling-free analogue of the HL-NUTS widening).
    s = _setup(HLLikelihood, ell_max=50, delta_ell=8)
    prof_r = compute_profile_sigma(
        s["post"].log_prob,
        s["free_names"],
        "r",
        s["x_mle"],
        s["fisher_sigma"],
        s["transform"],
        key=jax.random.PRNGKey(1),
    )
    assert np.isfinite(prof_r)
    assert prof_r > s["sigma_fisher_r"]
    # Sanity bound: the widening is O(10%), not a blow-up.
    assert prof_r < 1.5 * s["sigma_fisher_r"]
