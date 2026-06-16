"""
optimize_mapbased.py -- Differentiable map-based sigma(r) for instrument design.

The map-based counterpart of :func:`augr.optimize.sigma_r_from_design`: sigma(r)
as a differentiable function of the *noise design vector*, threaded through the
cut-sky masked-Wiener Monte-Carlo component-separation forward
(:func:`augr.spectrum_stages.mc_cutsky_cov_traced`) instead of the analytic Knox
covariance. This is the gradient entry point for design optimization *through*
map-based compsep.

Two differentiable levers:

* The **noise** vector (:func:`sigma_r_from_noise_design`): per-band detector count
  ``n_det``, NET ``net``, total efficiency ``eta_total``, and the scalar
  ``mission_years`` -- everything that enters the white-noise power ``w_inv`` via
  :func:`augr.instrument.white_noise_power_continuous`.
* The **beam** design (:func:`sigma_r_from_beam_design`): per-band FWHM and the
  generalized-Gaussian shape exponent ``p``, beamed onto the sky *in-trace* by
  :func:`augr.spectrum_stages.mc_cutsky_cov_traced`. The cleaner's discrete
  needlet-channel mask is handled by a jnp + ``stop_gradient`` mask (exact forward,
  frozen jump), so the beams flow the gradient -- it is no longer on the analytic path.

The noise chain is:

    n_det, net, eta_total, mission_years
      -> w_inv  (white_noise_power_continuous, per band)          [this module]
      -> mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner).covariance  [spectrum_stages]
      -> sigma_r_from_external_cov(cov, opt_ctx)                  [optimize]

and the beam chain feeds ``beam_fwhm`` / ``beam_p`` into the same traced covariance.

``f_sky`` is deliberately *not* a gradient knob here. The analysis sky fraction
is set by the (fixed, non-differentiable) mask baked into ``mc_ctx``, so the
``f_sky`` used to spread the integration time in ``w_inv`` is tied to
``mc_ctx.f_sky`` for self-consistency and held static. Exposing it as a
noise-only knob with the mode count fixed would be a footgun -- the gradient
would always favor ``f_sky -> 0`` (deeper noise, no penalty for fewer modes). A
genuine f_sky optimization needs a differentiable mask, deferred with the beam
path.

Usage:
    import jax
    from augr.config import cleaned_map_instrument
    from augr.foregrounds import NullForegroundModel
    from augr.optimize import make_optimization_context
    from augr.optimize_mapbased import sigma_r_from_noise_design
    from augr.spectrum_stages import make_cutsky_mc_context

    mc_ctx = make_cutsky_mc_context(...)        # eager: sky ensemble + frozen filter
    opt_ctx = make_optimization_context(        # the cleaned-map SignalModel
        cleaned_map_instrument(f_sky), NullForegroundModel(), cmb, fiducial, ...)
    grad_fn = jax.grad(sigma_r_from_noise_design, argnums=(0, 1, 2, 3))
    g = grad_fn(n_det, net, eta_total, mission_years,
                mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner)

    # Beam sensitivity: per-band d sigma(r)/d(FWHM, p) at fixed noise.
    beam_grad = jax.grad(sigma_r_from_beam_design, argnums=(0, 1))
    g_fwhm, g_p = beam_grad(beam_fwhm, beam_p,
                            w_inv=w_inv, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner)
"""

from __future__ import annotations

import jax.numpy as jnp

from augr.cleaning import Cleaner
from augr.instrument import white_noise_power_continuous
from augr.optimize import OptimizationContext, sigma_r_from_external_cov
from augr.spectrum_stages import CutskyMCContext, mc_cutsky_cov_traced


def w_inv_from_noise_design(
    n_det: jnp.ndarray,
    net: jnp.ndarray,
    eta_total: jnp.ndarray,
    mission_years: jnp.ndarray | float,
    f_sky: jnp.ndarray | float,
) -> jnp.ndarray:
    """Per-band white-noise power ``w_inv`` [μK²·sr] from the noise design vector.

    Vectorized form of :func:`augr.instrument.white_noise_power_continuous`:
    ``n_det``, ``net``, ``eta_total`` are per-band (scalars broadcast against
    each other); ``mission_years`` and ``f_sky`` are shared scalars. Fully
    differentiable in every argument.

    Args:
        n_det:         Effective detector counts per band (float, continuous).
        net:           NET per detector [μK√s] per band.
        eta_total:     Total efficiency per band.
        mission_years: Mission duration [years].
        f_sky:         Sky fraction over which the integration time is spread.

    Returns:
        ``w_inv`` array broadcast to the common shape of the per-band inputs.
    """
    n_det, net, eta_total = jnp.broadcast_arrays(
        jnp.asarray(n_det), jnp.asarray(net), jnp.asarray(eta_total)
    )
    return white_noise_power_continuous(net, n_det, eta_total, mission_years, f_sky)


def sigma_r_from_noise_design(
    n_det: jnp.ndarray,
    net: jnp.ndarray,
    eta_total: jnp.ndarray,
    mission_years: jnp.ndarray | float,
    *,
    mc_ctx: CutskyMCContext,
    opt_ctx: OptimizationContext,
    cleaner: Cleaner,
    f_sky: float | None = None,
) -> jnp.ndarray:
    """Differentiable map-based sigma(r) from the noise design vector.

    Composes the noise vector into the per-band ``w_inv``, runs the differentiable
    cut-sky masked-Wiener Monte-Carlo compsep forward
    (:func:`augr.spectrum_stages.mc_cutsky_cov_traced`, ``jax.grad``-able in
    ``w_inv`` under the fixed common random numbers in ``mc_ctx``), and feeds the
    resulting sample covariance to
    :func:`augr.optimize.sigma_r_from_external_cov`. The four positional
    arguments are differentiable via ``jax.grad(..., argnums=(0, 1, 2, 3))``;
    ``mc_ctx`` / ``opt_ctx`` / ``cleaner`` are static.

    Because the gradient flows through a *sample* covariance, it carries
    Monte-Carlo noise -- characterise grad std vs ``n_sims`` before trusting a
    descent step (the plan's Phase 2 verification). Select the on-device jht SHT
    backend (``with augr.sht.sht_backend("jht"): ...``) to run on a GPU.

    Args:
        n_det:         Detector counts per band, shape ``(n_bands,)`` (or scalar).
        net:           NET per detector [μK√s] per band, shape ``(n_bands,)``.
        eta_total:     Total efficiency per band, shape ``(n_bands,)``.
        mission_years: Mission duration [years] (scalar).
        mc_ctx:        Eager cut-sky MC context (sky ensemble, frozen filter,
                       mask, binning) from :func:`make_cutsky_mc_context`.
        opt_ctx:       OptimizationContext built on the cleaned-map SignalModel
                       (NullForegroundModel + residual template); supplies the
                       structural Jacobian, priors, and ``r_idx``.
        cleaner:       Differentiable Cleaner (NILC / cMILC, global).
        f_sky:         Sky fraction used in ``w_inv``. Defaults to
                       ``mc_ctx.f_sky`` (the analysis mask fraction) and is held
                       static -- see the module docstring on why f_sky is not a
                       gradient knob in the map path.

    Returns:
        Scalar sigma(r).
    """
    n_bands = len(mc_ctx.beam_fwhm_arcmin)
    shape = (n_bands,)
    try:
        n_det_b = jnp.broadcast_to(jnp.asarray(n_det), shape)
        net_b = jnp.broadcast_to(jnp.asarray(net), shape)
        eta_b = jnp.broadcast_to(jnp.asarray(eta_total), shape)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"noise design vector must broadcast to {n_bands} bands "
            f"(mc_ctx has {n_bands} beams); got shapes "
            f"n_det={jnp.shape(n_det)}, net={jnp.shape(net)}, "
            f"eta_total={jnp.shape(eta_total)}."
        ) from exc

    fsky_noise = mc_ctx.f_sky if f_sky is None else f_sky
    w_inv = w_inv_from_noise_design(n_det_b, net_b, eta_b, mission_years, fsky_noise)
    cov = mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner).covariance
    return sigma_r_from_external_cov(cov, opt_ctx)


def sigma_r_from_beam_design(
    beam_fwhm: jnp.ndarray,
    beam_p: jnp.ndarray,
    *,
    w_inv: jnp.ndarray,
    mc_ctx: CutskyMCContext,
    opt_ctx: OptimizationContext,
    cleaner: Cleaner,
) -> jnp.ndarray:
    """Differentiable map-based sigma(r) from the per-band beam design (FWHM + shape ``p``).

    The beam counterpart of :func:`sigma_r_from_noise_design`: holds the noise ``w_inv``
    fixed and varies the per-band beams through the cut-sky masked-Wiener Monte-Carlo
    compsep forward (:func:`augr.spectrum_stages.mc_cutsky_cov_traced` beams the frozen
    harmonic sky in-trace), then feeds the sample covariance to
    :func:`augr.optimize.sigma_r_from_external_cov`. Differentiable via
    ``jax.grad(sigma_r_from_beam_design, argnums=(0, 1))`` → ``(∂σ(r)/∂fwhm,
    ∂σ(r)/∂p)`` per band.

    The beam is the generalized Gaussian ``b_ℓ = exp(-(ℓ(ℓ+1)σ²/2)^p)`` (``p=1`` →
    ordinary Gaussian); both knobs are optics-agnostic stand-ins for the (TBD) true beam
    profile. The cleaner's discrete needlet-channel mask is jnp + ``stop_gradient``, so the
    gradient is exact within a mask configuration and frozen across the (measure-zero)
    channel enter/leave-a-band jumps. As with the noise gradient, the loss flows through a
    *sample* covariance and carries Monte-Carlo noise — characterise grad std vs ``n_sims``
    before trusting a descent step, and note a free FWHM has no cost penalty (this is a
    *sensitivity* tool, not an unconstrained optimizer).

    Args:
        beam_fwhm:  Per-band beam FWHM [arcmin], shape ``(n_bands,)`` (or scalar).
        beam_p:     Per-band beam shape exponent, shape ``(n_bands,)`` (or scalar).
        w_inv:      Fixed per-band white-noise power [μK²·sr], shape ``(n_bands,)``.
        mc_ctx:     Eager cut-sky MC context (unbeamed sky ensemble, frozen filter,
                    mask, binning) from :func:`make_cutsky_mc_context`.
        opt_ctx:    OptimizationContext on the cleaned-map SignalModel (beam-independent:
                    the transfer-debiased bandpowers carry the beam, the Jacobian does not).
        cleaner:    Differentiable Cleaner (NILC, global).

    Returns:
        Scalar sigma(r).
    """
    n_bands = len(mc_ctx.beam_fwhm_arcmin)
    shape = (n_bands,)
    try:
        beam_fwhm_b = jnp.broadcast_to(jnp.asarray(beam_fwhm), shape)
        beam_p_b = jnp.broadcast_to(jnp.asarray(beam_p), shape)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"beam design vectors must broadcast to {n_bands} bands; got shapes "
            f"beam_fwhm={jnp.shape(beam_fwhm)}, beam_p={jnp.shape(beam_p)}."
        ) from exc

    cov = mc_cutsky_cov_traced(
        jnp.asarray(w_inv), mc_ctx, cleaner, beam_fwhm=beam_fwhm_b, beam_p=beam_p_b
    ).covariance
    return sigma_r_from_external_cov(cov, opt_ctx)
