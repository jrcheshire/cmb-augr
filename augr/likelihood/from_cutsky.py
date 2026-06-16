"""from_cutsky.py — bridge the cut-sky masked-Wiener MC forecast into the inference layer.

The cut-sky map-based forecast (:func:`augr.pipeline.run_forecast` ``CUTSKY_MC`` ->
:func:`augr.spectrum_stages.mc_cutsky_bandpowers`) reports sigma(r) via a
Gaussian-Fisher on the Monte-Carlo bandpower covariance — carrying the documented
Knox/Gaussian few-to-ten-percent optimism at the reionization bump
(``reference_knox_gaussian_likelihood_bias``). This module feeds the *same* MC
outputs into augr's Bayesian inference layer (:mod:`augr.likelihood`) to get the
**honest non-Gaussian Hamimeche-Lewis posterior sigma(r)** at a fixed design, plus
the sampling-free profile-likelihood sigma(r) as the cheap cross-check.

It is built as a **reusable primitive**: :func:`posterior_from_cutsky_mc` is the
per-design log-posterior builder, and the profile-sigma path is the per-design
objective, that an experimental-design (EIG) loop reuses without a map forward in
its inner loop — the covariance ``Sigma_hat(xi)`` is design-dependent but
cosmology-parameter-independent, so one map forward per design produces
``(Sigma_hat, N_b)`` and the cheap analytic ``data_vector(theta)`` carries the
parameter dependence.

The bridge needs no change to the forward: ``CutskyMC.mean_bandpower`` is the
**total** debiased bandpower ``S + N + residual`` (the masked-Wiener debias does
not subtract a noise bias — ``augr.masking.debias_bandpower``), so the noise floor
is ``N_b = mean_bandpower - data_vector(fid)`` and the HL likelihood peaks at the
fiducial by construction (see :meth:`augr.likelihood.hl.HLLikelihood.from_external`).

The NUTS / MLE paths import blackjax / optax lazily (the ``[sampling]`` extra), so
importing this module does not require the extra; the constructors + KS calibration
do not need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from augr.config import cleaned_map_instrument
from augr.fisher import FisherForecast
from augr.foregrounds import NullForegroundModel
from augr.likelihood.gaussian import GaussianLikelihood
from augr.likelihood.hl import HLLikelihood
from augr.likelihood.mc_calibrated import MCCalibratedLikelihood, bandpower_ks
from augr.likelihood.nuts import (
    constrain,
    converged,
    diagnostics_summary,
    marginal_sigma,
    run_nuts_chains,
)
from augr.likelihood.posterior import Posterior, make_log_posterior
from augr.likelihood.prior import GaussianPrior, PositivityTransform
from augr.likelihood.protocols import SignalSpectrumModel
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

# Single cleaned BB map: r + A_lens + A_res, no foreground-index banana, so the
# amplitudes can be sampled unbounded (the model is linear in r / A_lens / A_res
# with no sqrt — no NaN region) for a clean Gaussian-NUTS == Gaussian-Fisher parity.
_DEFAULT_POSITIVE: frozenset[str] = frozenset()


def build_cutsky_signal_model(
    template_ells,
    template_cl,
    f_sky: float,
    *,
    ell_min: int = 2,
    ell_max: int = 180,
    delta_ell: int = 5,
    ell_per_bin_below: int = 30,
    delensed_bb=None,
    delensed_bb_ells=None,
):
    """Build the cleaned-map residual-template ``SignalModel`` + instrument.

    Mirrors :func:`augr.forecast.forecast_from_spectra`'s ``with_template=True``
    signal so the Bayesian sigma(r) here is apples-to-apples with the Gaussian-Fisher
    number: identical ``A_res`` template, binning, and Jacobian. Returns
    ``(signal_model, instrument)``.
    """
    inst = cleaned_map_instrument(f_sky=f_sky)
    kw = dict(
        instrument=inst,
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=ell_min,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
        residual_template_cl=jnp.asarray(template_cl),
        residual_template_ells=jnp.asarray(template_ells, dtype=float),
    )
    if delensed_bb is not None:
        kw["delensed_bb"] = jnp.asarray(delensed_bb)
        kw["delensed_bb_ells"] = jnp.asarray(delensed_bb_ells, dtype=float)
    return SignalModel(**kw), inst


def build_likelihood(kind: str, signal_model, fiducial_vec, mc):
    """Build a likelihood of ``kind`` in ``{"gaussian", "hl", "mc_calibrated"}`` from MC outputs.

    ``mc`` is a :class:`augr.spectrum_stages.CutskyMC` (or any object exposing
    ``covariance`` and ``mean_bandpower``). All three peak at the fiducial.
    """
    if kind == "gaussian":
        return GaussianLikelihood.from_external(signal_model, fiducial_vec, mc.covariance)
    if kind == "hl":
        return HLLikelihood.from_external(
            signal_model, fiducial_vec, mc.mean_bandpower, mc.covariance
        )
    if kind == "mc_calibrated":
        return MCCalibratedLikelihood.from_external(
            signal_model, fiducial_vec, mc.mean_bandpower, mc.covariance
        )
    raise ValueError(f"unknown likelihood kind {kind!r}; expected gaussian / hl / mc_calibrated.")


def posterior_from_cutsky_mc(
    mc,
    signal_model,
    fiducial: dict[str, float],
    *,
    likelihood_kind: str = "hl",
    priors: dict[str, float] | None = None,
    fixed: tuple[str, ...] | list[str] = (),
    positive_params: frozenset[str] = _DEFAULT_POSITIVE,
) -> tuple[Posterior, PositivityTransform, list[str], object]:
    """Assemble the cut-sky-MC log-posterior over the free parameters.

    The per-design log-posterior builder (the reusable primitive for an EIG/design
    loop): chooses the likelihood from the MC outputs (:func:`build_likelihood`),
    wraps the ``signal_model`` as the forward model, and attaches the Gaussian prior
    + positivity bijector. Returns ``(posterior, transform, free_names, likelihood)``.
    """
    fid_vec = flatten_params(fiducial, signal_model.parameter_names)
    likelihood = build_likelihood(likelihood_kind, signal_model, fid_vec, mc)
    model = SignalSpectrumModel(signal_model)
    free_names = [n for n in signal_model.parameter_names if n not in set(fixed)]
    prior = GaussianPrior.from_priors(free_names, fiducial, priors or {})
    transform = PositivityTransform.from_names(free_names, positive_params=positive_params)
    post = make_log_posterior(
        model, likelihood, prior, transform, fiducial=fiducial, fixed=tuple(fixed)
    )
    return post, transform, free_names, likelihood


def _fisher_cov(signal_model, instrument, fiducial, mc, priors, fixed):
    """Free-parameter Fisher covariance on the MC covariance — for sampler inits + profile grid.

    Uses a floored eigendecomposition inverse (mirroring ``FisherForecast.sigma``'s
    eigh-clipping) so a near-singular Fisher does not crash: this covariance only
    seeds the sampler starts and sets the profile-grid widths, not the reported
    sigma(r). A zero residual template (e.g. ``fg_model=None``) leaves ``A_res``
    unconstrained -> singular Fisher -> the floored direction just gets a wide
    (finite) init spread.
    """
    ff = FisherForecast(
        signal_model,
        instrument,
        fiducial,
        priors=priors or {},
        fixed_params=list(fixed),
        external_covariance=jnp.asarray(mc.covariance),
    )
    fisher = np.asarray(ff.compute())
    w, vecs = np.linalg.eigh(0.5 * (fisher + fisher.T))
    w_floored = np.maximum(w, 1e-12 * np.max(w))
    cov = (vecs / w_floored) @ vecs.T
    return jnp.asarray(0.5 * (cov + cov.T)), ff


@dataclass(frozen=True)
class _Sampled:
    sigma_r: float
    diagnostics: dict
    converged: bool
    mle_x: jax.Array


def _sample_posterior(
    post,
    transform,
    fisher_cov,
    *,
    key,
    n_chains: int,
    num_warmup: int,
    num_samples: int,
    target_acceptance_rate: float,
    n_starts_mle: int,
    mle_scale: float,
) -> _Sampled:
    """MLE multistart -> dithered multi-chain NUTS -> sigma(r) + convergence diagnostics."""
    from augr.likelihood.mle import make_dithered_starts, run_mle_search
    from augr.likelihood.nuts import draw_fisher_inits

    x_fid = post.fiducial_full[post.free_idx]
    k_mle, k_init, k_run = jax.random.split(key, 3)

    # Locate the mode off the Asimov fiducial (HL gradient is NaN exactly at it).
    inits_mle = draw_fisher_inits(
        x_fid, fisher_cov, transform, k_mle, n_starts_mle, scale=mle_scale
    )
    mle = run_mle_search(post.log_prob, inits_mle)

    # Over-dispersed NUTS starts dithered around the located mode.
    dither = jnp.sqrt(jnp.diag(fisher_cov))
    inits = make_dithered_starts(mle.best.x, dither, n_chains, k_init)
    positions, info = run_nuts_chains(
        post.log_prob,
        inits,
        k_run,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_acceptance_rate=target_acceptance_rate,
    )
    constrained = constrain(positions, transform)
    diag = diagnostics_summary(positions, info, post.free_names)
    return _Sampled(
        sigma_r=marginal_sigma(constrained, post.free_names, "r"),
        diagnostics=diag,
        converged=converged(diag),
        mle_x=mle.best.x,
    )


@dataclass(frozen=True)
class CutskyHLForecast:
    """sigma(r) four ways from a cut-sky MC ensemble + the KS calibration verdict.

    ``sigma_r_gauss_fisher`` is the existing Knox/Gaussian-Fisher baseline (what
    ``forecast_from_spectra(external_covariance=...)`` reports). ``sigma_r_gauss_nuts``
    samples that same Gaussian likelihood (the parity check). ``sigma_r_hl_nuts`` is
    the headline honest non-Gaussian width; ``sigma_r_hl_profile`` is its
    sampling-free profile-likelihood cross-check. ``hl_widening`` is
    ``sigma_r_hl_nuts / sigma_r_gauss_fisher``. ``ks`` is the
    :func:`augr.likelihood.mc_calibrated.bandpower_ks` verdict;
    ``sigma_r_mc_calibrated_nuts`` is populated only when the MC-calibrated
    cross-check is run.
    """

    sigma_r_gauss_fisher: float
    sigma_r_gauss_nuts: float
    sigma_r_hl_nuts: float
    sigma_r_hl_profile: float
    hl_widening: float
    converged_gauss: bool
    converged_hl: bool
    diagnostics_gauss: dict
    diagnostics_hl: dict
    ks: dict
    r_fid: float
    free_names: tuple[str, ...]
    sigma_r_mc_calibrated_nuts: float | None = None


def hl_forecast_from_cutsky_mc(
    mc,
    *,
    template_ells,
    template_cl,
    f_sky: float,
    r_fid: float = 0.0,
    ell_min: int = 2,
    ell_max: int = 180,
    delta_ell: int = 5,
    ell_per_bin_below: int = 30,
    priors: dict[str, float] | None = None,
    fixed: tuple[str, ...] | list[str] = (),
    positive_params: frozenset[str] = _DEFAULT_POSITIVE,
    delensed_bb=None,
    delensed_bb_ells=None,
    n_chains: int = 4,
    num_warmup: int = 800,
    num_samples: int = 1500,
    target_acceptance_rate: float = 0.99,
    n_starts_mle: int = 8,
    mle_scale: float = 0.5,
    profile: bool = True,
    profile_n_grid: int = 15,
    run_mc_calibrated: bool = False,
    seed: int = 0,
) -> CutskyHLForecast:
    """The honest non-Gaussian sigma(r) for a cut-sky MC forecast: HL-NUTS + profile + parity.

    Builds the cleaned-map residual-template ``SignalModel`` (apples-to-apples with
    :func:`augr.forecast.forecast_from_spectra`), then reports sigma(r) as:

    * **Gaussian-Fisher** — the existing Knox baseline (``forecast_from_spectra``).
    * **Gaussian-NUTS** — sampling that Gaussian likelihood (parity check: must
      reproduce Gaussian-Fisher).
    * **HL-NUTS** — the headline non-Gaussian width.
    * **HL-profile** — the sampling-free profile-likelihood cross-check.

    plus the :func:`augr.likelihood.mc_calibrated.bandpower_ks` calibration verdict
    on ``mc.debiased_bandpowers``. With ``run_mc_calibrated=True`` it also samples the
    ensemble-calibrated offset-lognormal cross-check.

    ``priors`` / ``fixed`` mirror :class:`augr.fisher.FisherForecast`; the fiducial is
    ``{"r": r_fid, "A_lens": 1.0, "A_res": 1.0}``. Needs the ``[sampling]`` extra
    (blackjax + optax) for the NUTS / MLE / profile paths.
    """
    signal_model, instrument = build_cutsky_signal_model(
        template_ells,
        template_cl,
        f_sky,
        ell_min=ell_min,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
        delensed_bb=delensed_bb,
        delensed_bb_ells=delensed_bb_ells,
    )
    fiducial = {"r": r_fid, "A_lens": 1.0, "A_res": 1.0}

    fisher_cov, ff_gauss = _fisher_cov(signal_model, instrument, fiducial, mc, priors, fixed)
    sigma_r_gauss_fisher = float(ff_gauss.sigma("r"))

    key = jax.random.PRNGKey(seed)
    k_gauss, k_hl, k_mcc = jax.random.split(key, 3)

    def _run(kind, k):
        post, transform, free_names, _lik = posterior_from_cutsky_mc(
            mc,
            signal_model,
            fiducial,
            likelihood_kind=kind,
            priors=priors,
            fixed=fixed,
            positive_params=positive_params,
        )
        sampled = _sample_posterior(
            post,
            transform,
            fisher_cov,
            key=k,
            n_chains=n_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_acceptance_rate=target_acceptance_rate,
            n_starts_mle=n_starts_mle,
            mle_scale=mle_scale,
        )
        return post, transform, free_names, sampled

    _post_g, _t_g, free_names, sampled_g = _run("gaussian", k_gauss)
    post_h, transform_h, _free_h, sampled_h = _run("hl", k_hl)

    sigma_r_hl_profile = float("nan")
    if profile:
        from augr.likelihood.profile import compute_profile_sigma

        fisher_sigma = jnp.sqrt(jnp.diag(fisher_cov))
        sigma_r_hl_profile = float(
            compute_profile_sigma(
                post_h.log_prob,
                free_names,
                "r",
                sampled_h.mle_x,
                fisher_sigma,
                transform_h,
                n_grid=profile_n_grid,
                key=jax.random.PRNGKey(seed + 1),
            )
        )

    sigma_r_mcc = None
    if run_mc_calibrated:
        _post_m, _t_m, _f_m, sampled_m = _run("mc_calibrated", k_mcc)
        sigma_r_mcc = float(sampled_m.sigma_r)

    ks = bandpower_ks(mc.debiased_bandpowers, mc.mean_bandpower, mc.covariance)

    return CutskyHLForecast(
        sigma_r_gauss_fisher=sigma_r_gauss_fisher,
        sigma_r_gauss_nuts=float(sampled_g.sigma_r),
        sigma_r_hl_nuts=float(sampled_h.sigma_r),
        sigma_r_hl_profile=sigma_r_hl_profile,
        hl_widening=float(sampled_h.sigma_r) / sigma_r_gauss_fisher,
        converged_gauss=bool(sampled_g.converged),
        converged_hl=bool(sampled_h.converged),
        diagnostics_gauss=sampled_g.diagnostics,
        diagnostics_hl=sampled_h.diagnostics,
        ks=ks,
        r_fid=r_fid,
        free_names=tuple(free_names),
        sigma_r_mc_calibrated_nuts=sigma_r_mcc,
    )
