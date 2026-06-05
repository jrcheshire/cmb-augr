"""Slow NUTS tests over the HL / Gaussian bandpower posteriors (Phase B).

These run the BlackJAX sampler, so they are marked ``slow`` (excluded from the
default fast suite; run via ``pytest -m slow`` or the full-suite CI step) and
require the ``[sampling]`` extra. They pin the two Phase B facts:

* sampling the Gaussian/Knox likelihood recovers the Fisher σ(r) (parity check —
  validates the whole posterior + sampler chain);
* sampling the Hamimeche-Lewis likelihood gives a *wider* σ(r) at low mode count
  (the non-Gaussian reionization-bump widening that the Knox Fisher misses).

Reference numbers at this config (ell 2-50, f_sky 0.6, r_fid 0.01, unbounded,
target_accept 0.99): Gaussian-NUTS within a few % of Fisher; HL-NUTS ~10-15%
wider; divergences < 1% (the residual divergences come from the weakly-
constrained foreground-index banana geometry, not the likelihood).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("blackjax")

from augr.config import DEFAULT_PRIORS, FIDUCIAL_BK15
from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.likelihood import (
    GaussianLikelihood,
    GaussianPrior,
    HLLikelihood,
    PositivityTransform,
    SignalSpectrumModel,
    constrain,
    draw_fisher_init,
    make_log_posterior,
    marginal_sigma,
    run_nuts,
)
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

FIXED = ["T_dust"]
R_FID = 0.01
TARGET_ACCEPT = 0.99
NUM_WARMUP = 800
NUM_SAMPLES = 1500


@pytest.fixture(scope="module")
def forecast():
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
    # Low-mode (reionization-bump-weighted) config: the regime where the HL
    # likelihood is non-Gaussian enough to widen σ(r) measurably.
    sm = SignalModel(
        inst, GaussianForegroundModel(), CMBSpectra(), ell_min=2, ell_max=50, delta_ell=8
    )
    return inst, sm, fid


def _sigma_r_nuts(likelihood, model, sm, inst, fid, key):
    """Unbounded NUTS over the likelihood's Asimov posterior → (σ(r), n_div, n)."""
    free_names = [n for n in sm.parameter_names if n not in FIXED]
    prior = GaussianPrior.from_priors(free_names, fid, DEFAULT_PRIORS)
    transform = PositivityTransform.from_names(free_names, positive_params=frozenset())
    post = make_log_posterior(model, likelihood, prior, transform, fiducial=fid, fixed=FIXED)

    ff = FisherForecast(sm, inst, fid, priors=DEFAULT_PRIORS, fixed_params=FIXED)
    fisher_cov = jnp.asarray(np.linalg.inv(np.asarray(ff.compute())))
    x_fid = post.fiducial_full[post.free_idx]

    k_init, k_run = jax.random.split(key)
    u0 = draw_fisher_init(x_fid, fisher_cov, transform, k_init, scale=1.0)
    positions, info = run_nuts(
        post.log_prob,
        u0,
        k_run,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        target_acceptance_rate=TARGET_ACCEPT,
    )
    constrained = constrain(positions, transform)
    return (
        marginal_sigma(constrained, post.free_names, "r"),
        int(jnp.sum(info.is_divergent)),
        positions.shape[0],
    )


@pytest.fixture(scope="module")
def sampled(forecast):
    inst, sm, fid = forecast
    model = SignalSpectrumModel(sm)
    fid_vec = flatten_params(fid, sm.parameter_names)

    ff = FisherForecast(sm, inst, fid, priors=DEFAULT_PRIORS, fixed_params=FIXED)
    sigma_fisher = float(ff.sigma("r"))

    gauss = GaussianLikelihood.from_forecast(sm, inst, fid_vec)
    hl = HLLikelihood.from_forecast(sm, inst, fid_vec)
    key_g, key_h = jax.random.split(jax.random.PRNGKey(0))
    sigma_g, div_g, n = _sigma_r_nuts(gauss, model, sm, inst, fid, key_g)
    sigma_h, div_h, _ = _sigma_r_nuts(hl, model, sm, inst, fid, key_h)
    return {
        "fisher": sigma_fisher,
        "gauss": sigma_g,
        "hl": sigma_h,
        "div_g": div_g,
        "div_h": div_h,
        "n": n,
    }


@pytest.mark.slow
def test_gaussian_nuts_recovers_fisher(sampled):
    # Sampling the Gaussian/Knox likelihood must return the Fisher σ(r).
    np.testing.assert_allclose(sampled["gauss"], sampled["fisher"], rtol=0.12)


@pytest.mark.slow
def test_hl_nuts_wider_than_gaussian_and_fisher(sampled):
    # The headline: HL is non-Gaussian at low mode count → wider σ(r). Measured
    # ~14% wider than Gaussian-NUTS / ~10% wider than Fisher at this config; the
    # gap is well above Monte-Carlo noise, so a 5% margin is a safe pin.
    assert sampled["hl"] > sampled["gauss"] * 1.05
    assert sampled["hl"] > sampled["fisher"]


@pytest.mark.slow
def test_sampler_divergences_low(sampled):
    # Sampler health: divergences stay a small fraction (FG-index geometry, not
    # the likelihood). A regression that breaks the geometry would blow this up.
    assert sampled["div_g"] < 0.05 * sampled["n"]
    assert sampled["div_h"] < 0.05 * sampled["n"]
