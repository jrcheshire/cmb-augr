"""Tests for augr.likelihood.mle — L-BFGS MLE + dithered multistart.

Two tiers:

* **Fast** synthetic-loss unit tests of the optimizer and dither generator
  (need only the ``[sampling]`` extra's ``optax``; no forecast machinery). These
  mirror bk-jax's ``test_likelihood_mle.py`` but are phrased for augr's
  *maximize*-``log_prob`` convention (``run_mle`` maximizes, so the targets are
  *negated* losses with their peak at the optimum).
* **Slow** posterior-level tests: a Gaussian-likelihood Asimov forecast whose MLE
  must recover the fiducial, and the MLE→NUTS init bridge (dither the chains
  around the located mode, sample, gate on convergence).
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
    GaussianLikelihood,
    GaussianPrior,
    MLEResult,
    MLESearchResult,
    PositivityTransform,
    SignalSpectrumModel,
    constrain,
    converged,
    diagnostics_summary,
    draw_fisher_inits,
    make_dithered_starts,
    make_log_posterior,
    marginal_sigma,
    run_mle,
    run_mle_search,
    run_nuts_chains,
)
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

# ---------------------------------------------------------------------------
# Fast synthetic-loss unit tests
# ---------------------------------------------------------------------------


def _neg_quadratic(center):
    """log_prob = -0.5 ||x - center||^2 (peak at center, max value 0)."""
    c = jnp.asarray(center, dtype=jnp.float64)

    def log_prob(x):
        return -0.5 * jnp.sum((x - c) ** 2)

    return log_prob


def _neg_rosenbrock(x):
    """log_prob = -Rosenbrock(x); peak at all-ones, max value 0."""
    return -jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def test_run_mle_recovers_quadratic_peak():
    res = run_mle(_neg_quadratic(jnp.zeros(3)), jnp.array([1.5, -2.0, 0.7]))
    assert isinstance(res, MLEResult)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.x), 0.0, atol=1e-6)
    assert float(res.grad_norm) <= 1e-6
    np.testing.assert_allclose(float(res.log_prob), 0.0, atol=1e-10)


def test_run_mle_recovers_offset_anisotropic_quadratic():
    center = jnp.array([2.0, -1.0, 5.0, 0.0])
    scales = jnp.array([1.0, 10.0, 0.1, 3.0])  # anisotropic curvature

    def log_prob(x):
        return -0.5 * jnp.sum(scales * (x - center) ** 2)

    res = run_mle(log_prob, jnp.zeros(4))
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.x), np.asarray(center), atol=1e-5)


def test_run_mle_rosenbrock_4d():
    res = run_mle(_neg_rosenbrock, jnp.array([-1.0, 1.0, -1.0, 1.0]), max_iter=500)
    np.testing.assert_allclose(np.asarray(res.x), 1.0, atol=1e-3)


def test_run_mle_jit_traceable():
    log_prob = _neg_quadratic(jnp.zeros(3))
    res = jax.jit(lambda x0: run_mle(log_prob, x0))(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(res.x), 0.0, atol=1e-6)


def test_run_mle_already_at_optimum_zero_iters():
    res = run_mle(_neg_quadratic(jnp.zeros(2)), jnp.zeros(2))
    assert int(res.n_iter) == 0
    assert bool(res.converged)


def test_run_mle_max_iter_cap():
    # Rosenbrock from a hard start needs many L-BFGS steps; cap at 2 -> not done.
    res = run_mle(_neg_rosenbrock, jnp.array([-1.2, 1.0]), max_iter=2)
    assert int(res.n_iter) <= 2
    assert not bool(res.converged)


def test_make_dithered_starts_shape_and_stats():
    center = jnp.array([1.0, -2.0, 3.0])
    dither = jnp.array([0.1, 0.5, 0.2])
    starts = make_dithered_starts(center, dither, 4000, jax.random.PRNGKey(0))
    assert starts.shape == (4000, 3)
    np.testing.assert_allclose(np.asarray(starts.mean(0)), np.asarray(center), atol=0.05)
    np.testing.assert_allclose(np.asarray(starts.std(0)), np.asarray(dither), rtol=0.1)


def test_make_dithered_starts_reproducible():
    key = jax.random.PRNGKey(7)
    a = make_dithered_starts(jnp.zeros(4), jnp.ones(4), 16, key)
    b = make_dithered_starts(jnp.zeros(4), jnp.ones(4), 16, key)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_make_dithered_starts_shape_guards():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        make_dithered_starts(jnp.zeros(3), jnp.ones(2), 5, key)
    with pytest.raises(ValueError):
        make_dithered_starts(jnp.zeros((2, 2)), jnp.zeros((2, 2)), 5, key)


def test_run_mle_search_best_on_convex():
    log_prob = _neg_quadratic(jnp.array([1.0, -1.0]))
    inits = make_dithered_starts(jnp.zeros(2), jnp.ones(2), 5, jax.random.PRNGKey(1))
    res = run_mle_search(log_prob, inits)
    assert isinstance(res, MLESearchResult)
    assert res.all_x.shape == (5, 2)
    assert res.init_positions.shape == (5, 2)
    np.testing.assert_allclose(np.asarray(res.best.x), np.array([1.0, -1.0]), atol=1e-5)
    # best is exactly the argmax over the per-start log_probs.
    assert float(res.best.log_prob) == float(jnp.max(res.all_log_prob))


def test_run_mle_search_finds_global_on_bimodal():
    # Two peaks: a taller one near x=-3, a shorter near x=+3. Wide dither explores
    # both basins; the best-of-n must land in the taller (global) basin.
    def log_prob(x):
        xv = x[0]
        peak_neg = -0.5 * (xv + 3.0) ** 2 + 1.0  # taller (offset +1)
        peak_pos = -0.5 * (xv - 3.0) ** 2  # shorter
        return jnp.logaddexp(peak_neg, peak_pos)

    inits = make_dithered_starts(jnp.zeros(1), jnp.array([3.0]), 8, jax.random.PRNGKey(3))
    res = run_mle_search(log_prob, inits)
    assert abs(float(res.best.x[0]) + 3.0) < 0.2


def test_run_mle_search_init_positions_guard():
    with pytest.raises(ValueError):
        run_mle_search(_neg_quadratic(jnp.zeros(2)), jnp.zeros(2))  # 1-D not allowed


# ---------------------------------------------------------------------------
# Slow posterior-level tests
# ---------------------------------------------------------------------------

FIXED = ["T_dust"]
R_FID = 0.01


@pytest.fixture(scope="module")
def gaussian_posterior():
    """A 3-channel Gaussian-likelihood Asimov posterior (peaks at the fiducial).

    Uses a well-conditioned ℓ range (2–200): the deliberately ill-conditioned
    low-mode bump config of ``test_likelihood_sampler.py`` is the right regime for
    the *sampler* (where the HL likelihood widens σ(r)) but the wrong one for the
    *optimizer* — its weakly-constrained foreground directions stall L-BFGS. More
    modes tighten the posterior so the MLE cleanly recovers the fiducial.
    """
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
        inst, GaussianForegroundModel(), CMBSpectra(), ell_min=2, ell_max=200, delta_ell=10
    )
    model = SignalSpectrumModel(sm)
    fid_vec = flatten_params(fid, sm.parameter_names)
    free_names = [n for n in sm.parameter_names if n not in FIXED]

    gauss = GaussianLikelihood.from_forecast(sm, inst, fid_vec)
    prior = GaussianPrior.from_priors(free_names, fid, DEFAULT_PRIORS)
    transform = PositivityTransform.from_names(free_names, positive_params=frozenset())
    post = make_log_posterior(model, gauss, prior, transform, fiducial=fid, fixed=FIXED)

    ff = FisherForecast(sm, inst, fid, priors=DEFAULT_PRIORS, fixed_params=FIXED)
    fisher_cov = jnp.asarray(np.linalg.inv(np.asarray(ff.compute())))
    return {
        "post": post,
        "transform": transform,
        "fisher_cov": fisher_cov,
        "free_names": post.free_names,
        "x_fid": post.fiducial_full[post.free_idx],
        "sigma_fisher": float(ff.sigma("r")),
    }


def _r_of(x_unconstrained, gp):
    r_idx = list(gp["free_names"]).index("r")
    return float(constrain(x_unconstrained[None, :], gp["transform"])[0, r_idx])


@pytest.mark.slow
def test_mle_recovers_fiducial_gaussian(gaussian_posterior):
    # Recovery, not the convergence flag: the g_tol=1e-6 gate is unreachable on
    # ill-conditioned foreground posteriors (a near-flat FG direction keeps the
    # gradient ~0.05 even at the mode), so we check that the located point IS the
    # Asimov optimum — r at the fiducial, log_prob at the peak (=0).
    gp = gaussian_posterior
    u0 = draw_fisher_inits(
        gp["x_fid"], gp["fisher_cov"], gp["transform"], jax.random.PRNGKey(0), 1, scale=0.3
    )[0]
    res = run_mle(gp["post"].log_prob, u0)
    assert np.isfinite(float(res.log_prob))
    assert float(res.log_prob) > -1e-2  # at the peak (max log_prob is 0 for Asimov)
    # A single modest start stops at the FG-direction gradient floor ~1-2% of σ
    # short of the exact mode; the multistart best-of-n below tightens this.
    assert abs(_r_of(res.x, gp) - R_FID) < 5e-2 * gp["sigma_fisher"]


@pytest.mark.slow
def test_mle_search_best_recovers_fiducial_gaussian(gaussian_posterior):
    gp = gaussian_posterior
    inits = draw_fisher_inits(
        gp["x_fid"], gp["fisher_cov"], gp["transform"], jax.random.PRNGKey(2), 4, scale=0.3
    )
    res = run_mle_search(gp["post"].log_prob, inits)
    assert res.all_x.shape[0] == 4
    # best is the (nan-safe) argmax over the per-start log_probs.
    assert float(res.best.log_prob) == float(jnp.nanmax(res.all_log_prob))
    assert abs(_r_of(res.best.x, gp) - R_FID) < 1e-2 * gp["sigma_fisher"]


@pytest.mark.slow
def test_mle_initialized_nuts_converges(gaussian_posterior):
    """The MLE→NUTS bridge: dither chains around the mode, sample, gate on convergence."""
    pytest.importorskip("blackjax")
    gp = gaussian_posterior
    k_search, k_dither, k_run = jax.random.split(jax.random.PRNGKey(5), 3)
    inits = draw_fisher_inits(
        gp["x_fid"], gp["fisher_cov"], gp["transform"], k_search, 4, scale=0.3
    )
    mle = run_mle_search(gp["post"].log_prob, inits)

    dither = 0.5 * jnp.sqrt(jnp.diag(gp["fisher_cov"]))
    u0 = make_dithered_starts(mle.best.x, dither, 4, k_dither)
    positions, info = run_nuts_chains(
        gp["post"].log_prob,
        u0,
        k_run,
        num_warmup=300,
        num_samples=500,
        target_acceptance_rate=0.99,
    )
    constrained = constrain(positions, gp["transform"])
    sigma_r = marginal_sigma(constrained, gp["free_names"], "r")
    diag = diagnostics_summary(positions, info, gp["free_names"])

    assert np.isfinite(sigma_r)
    assert converged(diag)
    # Gaussian-likelihood parity: sampled sigma(r) recovers the Fisher sigma(r).
    np.testing.assert_allclose(sigma_r, gp["sigma_fisher"], rtol=0.15)
