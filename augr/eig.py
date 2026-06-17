"""
eig.py -- Expected Information Gain (EIG) objectives for Bayesian-optimal design.

The capstone of the differentiable map-based forward: choose the instrument design xi
to maximize the information a forecast carries about r, under a cost budget. This
module supplies the EIG utilities and the cost-constrained design objective; the
non-Gaussian Hamimeche-Lewis EIG (the science novelty) is designed here but deferred
to Stage 2 (see :func:`design_objective` ``objective="hl_eig"`` and the recipe below).

**Gaussian EIG, two flavors, off one posterior Fisher.** Every utility reads a scalar
off ``F_post = J^T Sigma_hat(xi)^-1 J + diag(prior)`` -- the same posterior Fisher
that :func:`augr.optimize.sigma_r_from_external_cov` builds (via
``fisher._fisher_from_full``), differentiable in the covariance:

* **r-marginal EIG** (:func:`marginal_eig_r_from_external_cov`) --
  ``log(sigma_prior_r / sigma_post(r))``. Since ``sigma_post(r) = sqrt((F_post^-1)_rr)``
  is exactly what ``sigma_r_from_external_cov`` returns, the r-marginal Gaussian EIG is
  ``-log sigma(r) + const`` -- maximizing it is *identical* to minimizing sigma(r). The
  ``sigma_prior_r`` only sets the additive offset, **not** the optimum (its design
  gradient is zero), so the EIG-optimal design equals the sigma(r)-optimal design. This
  is the primary objective: the science QoI is r.
* **D-optimal EIG** (:func:`gaussian_eig_from_external_cov`) --
  ``0.5 * logdet(F_post)``, the joint information on the full parameter vector
  (r + nuisances). Exposed as a secondary/general BOED objective; it rewards nuisance
  information (e.g. pinning ``A_res``), so it is *not* focused on r.

**The cost-constrained loop is the new capability.** The prior sigma(r) optimization had
no budget and ran a free FWHM/NET to the boundary; :func:`design_objective` adds the
convex cost (:mod:`augr.cost`) as a soft budget wall so the optimum is interior.

**Structural gift (makes Stage-2 HL-EIG tractable).** ``Sigma_hat(xi)`` is
design-dependent but cosmology-parameter-independent, so one map forward per design
yields ``(Sigma_hat, N_b)`` and the cheap analytic ``data_vector(theta)`` carries the
parameter dependence -- there is **no map forward inside any EIG/sampling inner loop**.

Stage-2 HL-EIG recipe (deferred -- ``objective="hl_eig"`` raises):
  Per design xi, one map forward -> ``(Sigma_hat, N_b)``. Estimate the r-marginal EIG by
  **nested Monte Carlo over the cheap analytic HL likelihood**: sample
  ``theta=(r, eta) ~ prior``, simulate ``d_hat ~ HL(. | theta, Sigma_hat)``, and average
  ``log p(d_hat | theta) - log p_marg(d_hat)`` (nuisances ``eta`` marginalized). Reuse
  :func:`augr.likelihood.from_cutsky.posterior_from_cutsky_mc` /
  ``build_likelihood("hl")`` for ``log p(d_hat | theta)``. The design gradient flows
  through ``Sigma_hat(xi)``; the ``d_hat`` draw depends on ``Sigma_hat(xi)`` so
  reparametrize ``d_hat = mean(theta) + chol(Sigma_hat(xi)) @ z``. Gradient variance is
  the hard part (compounded MC) -- the n_sims/GPU characterization sets that budget.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from augr.cleaning import Cleaner
from augr.cost import CostModel, aperture_from_fwhm, budget_penalty
from augr.fisher import _fisher_from_full
from augr.optimize import OptimizationContext
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.spectrum_stages import CutskyMCContext, mc_cutsky_cov_traced

_OBJECTIVES = ("marginal_eig_r", "d_optimal", "sigma_r")


def posterior_fisher_from_external_cov(cov, ctx: OptimizationContext):
    """Posterior Fisher ``F = J^T cov^-1 J + diag(prior)`` (jnp, differentiable in ``cov``).

    The same prewhitened dense solve as
    :func:`augr.optimize.sigma_r_from_external_cov` /
    ``FisherForecast(external_covariance=...).compute()``; factored out so the EIG
    utilities share it. ``ctx`` supplies the structural Jacobian ``J`` and the prior
    diagonal ``1/sigma_prior^2``.
    """
    F = _fisher_from_full(ctx.J, jnp.asarray(cov))
    return F + jnp.diag(ctx.prior_diag)


def marginal_eig_r_from_external_cov(cov, ctx: OptimizationContext, *, sigma_prior_r: float = 1.0):
    """r-marginal Gaussian EIG ``log(sigma_prior_r / sigma_post(r))`` -- the primary objective.

    Differentiable in ``cov``; consistent by construction with
    :func:`augr.optimize.sigma_r_from_external_cov` (same ``F_post``, same
    ``(F^-1)_rr``). ``sigma_prior_r`` sets only the additive offset -- its design
    gradient is zero, so the EIG-optimal design equals the sigma(r)-optimal design.
    """
    F = posterior_fisher_from_external_cov(cov, ctx)
    sigma_r = jnp.sqrt(jnp.linalg.inv(F)[ctx.r_idx, ctx.r_idx])
    return jnp.log(sigma_prior_r) - jnp.log(sigma_r)


def gaussian_eig_from_external_cov(cov, ctx: OptimizationContext, *, prior_fisher_logdet=None):
    """D-optimal Gaussian EIG ``0.5 * logdet(F_post)`` -- the secondary/general objective.

    Joint information on the full parameter vector. With ``prior_fisher_logdet`` supplied
    (proper priors), returns the calibrated ``0.5 * (logdet F_post - logdet F_prior)``;
    with improper priors (some ``prior_diag`` are zero) ``logdet F_prior`` is undefined,
    but the design-dependent part ``0.5 * logdet(F_post)`` -- and hence the optimization
    gradient -- is well-defined, so the absolute EIG is offset-only.
    """
    F = posterior_fisher_from_external_cov(cov, ctx)
    _sign, logdet = jnp.linalg.slogdet(F)
    eig = 0.5 * logdet
    if prior_fisher_logdet is not None:
        eig = eig - 0.5 * prior_fisher_logdet
    return eig


def _utility(cov, ctx, objective, sigma_prior_r):
    """EIG utility (to be MAXIMIZED) for the named ``objective``."""
    if objective == "marginal_eig_r":
        return marginal_eig_r_from_external_cov(cov, ctx, sigma_prior_r=sigma_prior_r)
    if objective == "d_optimal":
        return gaussian_eig_from_external_cov(cov, ctx)
    if objective == "sigma_r":
        # Utility = -sigma(r), so maximizing utility == minimizing sigma(r). Lets the
        # design driver reuse one code path to reproduce the sigma(r) optimum.
        F = posterior_fisher_from_external_cov(cov, ctx)
        return -jnp.sqrt(jnp.linalg.inv(F)[ctx.r_idx, ctx.r_idx])
    if objective == "hl_eig":
        raise NotImplementedError(
            "HL-EIG is the deferred Stage-2 novelty: nested-MC over the analytic HL "
            "likelihood with a reparametrized d_hat draw. See the augr.eig module docstring."
        )
    raise ValueError(f"unknown objective {objective!r}; expected one of {_OBJECTIVES} or 'hl_eig'.")


def design_cost(n_det, beam_fwhm, mission_years, *, cost_model: CostModel, freqs_ghz):
    """Mission cost [$M] from the design knobs (no map forward) -- differentiable.

    The aperture is taken from the **tightest-beam (highest-frequency) band**, the one
    that sets the diffraction limit of the single dish (``aperture_from_fwhm``); the
    detector cost from ``sum(n_det)``; the ops cost from ``mission_years``. Reconciling
    per-band free FWHMs with a single physical aperture is a modeling choice deferred to
    a real optics model (see :func:`augr.cost.aperture_from_fwhm`).
    """
    freqs = tuple(float(f) for f in freqs_ghz)
    i_tight = int(np.argmax(freqs))  # static: freqs are concrete
    aperture = aperture_from_fwhm(beam_fwhm[i_tight], freqs[i_tight])
    n_det_total = jnp.sum(jnp.asarray(n_det))
    return cost_model.total_cost(aperture, n_det_total, mission_years)


def design_objective(
    n_det,
    net,
    eta_total,
    mission_years,
    beam_fwhm,
    beam_p,
    *,
    mc_ctx: CutskyMCContext,
    opt_ctx: OptimizationContext,
    cleaner: Cleaner,
    cost_model: CostModel,
    budget: float,
    freqs_ghz,
    penalty_weight: float = 1.0,
    objective: str = "marginal_eig_r",
    sigma_prior_r: float = 1.0,
):
    """Cost-constrained Bayesian-design objective to MINIMIZE -- ``jax.grad``-able.

    Composes the noise + beam design into ``w_inv`` and per-band beams, runs the cut-sky
    masked-Wiener Monte-Carlo compsep forward
    (:func:`augr.spectrum_stages.mc_cutsky_cov_traced`) to ``Sigma_hat``, evaluates the
    EIG ``objective`` (default r-marginal), and adds the convex budget penalty
    (:func:`augr.cost.budget_penalty`). Differentiable via
    ``jax.grad(design_objective, argnums=(0, 1, 2, 3, 4, 5))`` -> the joint noise+beam
    design gradient; ``mc_ctx`` / ``opt_ctx`` / ``cleaner`` / ``cost_model`` are static.

    Returns ``-utility + budget_penalty`` (a minimization scalar): descent maximizes EIG
    while staying on the affordable side of the budget surface.
    """
    w_inv = w_inv_from_noise_design(n_det, net, eta_total, mission_years, mc_ctx.f_sky)
    cov = mc_cutsky_cov_traced(
        w_inv, mc_ctx, cleaner, beam_fwhm=beam_fwhm, beam_p=beam_p
    ).covariance
    util = _utility(cov, opt_ctx, objective, sigma_prior_r)
    cost = design_cost(n_det, beam_fwhm, mission_years, cost_model=cost_model, freqs_ghz=freqs_ghz)
    return -util + budget_penalty(cost, budget, penalty_weight)
