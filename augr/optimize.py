"""
optimize.py — Differentiable Fisher forecast for gradient-based instrument optimization.

Provides a functional interface where sigma(r) is a differentiable function
of continuous instrument parameters (detector counts, NETs, beam sizes,
telescope geometry). This enables gradient-based optimization of telescope
designs via jax.grad.

The key insight: instrument parameters enter the Fisher calculation only
through the noise covariance. The data vector and Jacobian depend on
foreground/cosmological parameters and channel frequencies (structural),
not on detector counts, NETs, or beams. So the Jacobian can be pre-computed
once, and only the noise → covariance → Fisher path needs to be traced.

Two tiers:
  - **Tier 1** (sigma_r_from_channels): Optimize channel-level parameters
    directly — n_det (float), NET, beam FWHM. Fastest for "given these
    frequencies, how should I allocate detectors?"
  - **Tier 2** (sigma_r_from_design): Optimize telescope design parameters
    — aperture, f_number, focal plane diameter, area fractions — that
    derive channel parameters via the physics.

Usage:
    from augr.optimize import make_optimization_context, sigma_r_from_channels

    ctx = make_optimization_context(instrument, fg_model, cmb, fiducial,
                                    priors, fixed_params)
    # Gradient of sigma(r) w.r.t. detector counts:
    grad_fn = jax.grad(sigma_r_from_channels, argnums=0)
    d_sigma_d_ndet = grad_fn(ctx.n_det, ctx.net, ctx.beam, ctx.eta,
                             ctx, mission_years=5.0, f_sky=0.7)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from augr.covariance import bandpower_covariance_blocks_from_noise
from augr.fisher import _fisher_from_blocks, _fisher_from_full
from augr.instrument import Instrument, noise_nl_continuous
from augr.signal import SignalModel, flatten_params
from augr.telescope import (
    beam_fwhm_arcmin,
    count_pixels_continuous,
    horn_diameter,
    photon_noise_net_jax,
)

# Backward-compat alias: the optimize and FisherForecast paths share the
# same primitive (jnp.linalg.solve per bin). Kept as a name in case any
# downstream code imports it; new code should use _fisher_from_blocks
# directly.
_fisher_from_blocks_solve = _fisher_from_blocks


@dataclass(frozen=True)
class OptimizationContext:
    """Pre-computed quantities that are static during instrument optimization.

    Built once by make_optimization_context(). Holds the signal model,
    pre-computed Jacobian blocks, prior structure, and initial channel
    parameters extracted from the reference instrument.

    Attributes:
        signal_model: Pre-built SignalModel (defines frequencies, binning).
        J_blocks:     Jacobian reshaped to (n_bins, n_spec, n_free).
        J:            Full Jacobian (n_data, n_free), n_data = n_spec * n_bins
                      (the un-blocked form for the dense external-covariance solve
                      in sigma_r_from_external_cov).
        fiducial_params: Flat parameter array for signal evaluation.
        prior_diag:   1/sigma^2 for each free parameter (0 = no prior).
        r_idx:        Index of 'r' in the free parameter list.
        ells:         Multipole grid from the signal model.
        n_det:        Initial detector counts, shape (n_chan,).
        net:          Initial NETs per detector, shape (n_chan,).
        beam:         Initial beam FWHM [arcmin], shape (n_chan,).
        eta:          Initial total efficiency per channel, shape (n_chan,).
        freqs:        Channel frequencies [GHz], tuple of floats.
    """

    signal_model: SignalModel
    J_blocks: jnp.ndarray
    J: jnp.ndarray
    fiducial_params: jnp.ndarray
    prior_diag: jnp.ndarray
    r_idx: int
    ells: jnp.ndarray
    n_det: jnp.ndarray
    net: jnp.ndarray
    beam: jnp.ndarray
    eta: jnp.ndarray
    freqs: tuple[float, ...]


def make_optimization_context(
    instrument: Instrument,
    foreground_model,
    cmb_spectra,
    fiducial_params: dict[str, float],
    priors: dict[str, float] | None = None,
    fixed_params: list[str] | None = None,
    **signal_kwargs,
) -> OptimizationContext:
    """One-time setup for differentiable sigma(r) optimization.

    Builds the SignalModel, pre-computes the Jacobian, assembles the prior
    structure, and extracts channel parameters as JAX arrays. The returned
    context is passed to sigma_r_from_channels or sigma_r_from_design.

    Args:
        instrument:      Reference instrument (defines frequencies, structure).
        foreground_model: ForegroundModel (Gaussian or Moment).
        cmb_spectra:     CMBSpectra instance.
        fiducial_params: Dict of fiducial parameter values.
        priors:          Dict mapping param name -> prior sigma.
        fixed_params:    List of params to hold fixed.
        **signal_kwargs: Passed to SignalModel (ell_min, ell_max, delta_ell,
                         ell_per_bin_below, delensed_bb, etc.)

    Returns:
        OptimizationContext for use with sigma_r_from_channels.
    """
    priors = priors or {}
    fixed_params = fixed_params or []

    # Build signal model (defines data vector structure, Jacobian)
    sig = SignalModel(instrument, foreground_model, cmb_spectra, **signal_kwargs)

    # Parameter bookkeeping
    all_names = sig.parameter_names
    free_names = [n for n in all_names if n not in set(fixed_params)]
    free_idx = jnp.array([all_names.index(n) for n in free_names])

    # Flatten fiducial params
    params = flatten_params(fiducial_params, all_names)

    # Pre-compute Jacobian (depends on foreground params + frequencies, not
    # on instrument noise/beam/n_det)
    J_full = sig.jacobian(params)  # (n_data, n_all_params)
    J = J_full[:, free_idx]  # (n_data, n_free)

    n_spec = len(sig.freq_pairs)
    n_bins = sig.n_bins
    J_blocks = J.reshape(n_spec, n_bins, -1).transpose(1, 0, 2)

    # Prior diagonal: 1/sigma^2 for each free param, 0 if no prior
    prior_diag = jnp.zeros(len(free_names))
    for name, sigma_prior in priors.items():
        if name in free_names:
            idx = free_names.index(name)
            prior_diag = prior_diag.at[idx].set(1.0 / sigma_prior**2)

    # Index of r in free params
    r_idx = free_names.index("r")

    # Extract channel parameters as JAX arrays
    channels = instrument.channels
    n_det = jnp.array([float(ch.n_detectors) for ch in channels])
    net = jnp.array([ch.net_per_detector for ch in channels])
    beam = jnp.array([ch.beam_fwhm_arcmin for ch in channels])
    eta = jnp.array([ch.efficiency.total for ch in channels])
    freqs = tuple(ch.nu_ghz for ch in channels)

    return OptimizationContext(
        signal_model=sig,
        J_blocks=J_blocks,
        J=J,
        fiducial_params=params,
        prior_diag=prior_diag,
        r_idx=r_idx,
        ells=sig.ells,
        n_det=n_det,
        net=net,
        beam=beam,
        eta=eta,
        freqs=freqs,
    )


def sigma_r_from_channels(
    n_det: jnp.ndarray,
    net: jnp.ndarray,
    beam_fwhm: jnp.ndarray,
    eta_total: jnp.ndarray,
    ctx: OptimizationContext,
    mission_years: float = 5.0,
    f_sky: float = 0.7,
    knee_ell: jnp.ndarray | float = 0.0,
    alpha_knee: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Differentiable sigma(r) as a function of channel-level instrument params.

    Tier 1 optimization: directly optimize detector counts (as floats),
    NETs, and beam sizes. Channel frequencies are fixed (structural).

    All four positional arrays can be differentiated via jax.grad.

    Args:
        n_det:       Effective detector counts per channel, shape (n_chan,).
                     Float (continuous relaxation of integer counts).
        net:         NET per detector [μK√s], shape (n_chan,).
        beam_fwhm:   Beam FWHM [arcmin], shape (n_chan,).
        eta_total:   Total efficiency per channel, shape (n_chan,).
        ctx:         Pre-computed OptimizationContext.
        mission_years: Mission duration [years].
        f_sky:       Sky fraction.
        knee_ell:    1/f knee multipole (scalar or per-channel).
        alpha_knee:  1/f spectral index (scalar or per-channel).

    Returns:
        Scalar sigma(r) -- marginalized Fisher constraint on r.

    Note:
        Uses ``fisher._fisher_from_blocks`` (jnp.linalg.solve per bin),
        the same primitive as ``FisherForecast.sigma``. The two paths
        agree to fp64 precision at any allocation. Validated as
        essentially exact (rel error < 1e-7 outside bin 0, < 1% at bin 0)
        against mpmath @ 30 dps on the PICO 21-channel moment-FG fixture.
    """
    ells = ctx.ells
    n_chan = n_det.shape[0]

    # Broadcast scalar knee_ell / alpha_knee to per-channel arrays
    knee_arr = jnp.broadcast_to(jnp.asarray(knee_ell), (n_chan,))
    alpha_arr = jnp.broadcast_to(jnp.asarray(alpha_knee), (n_chan,))

    # Compute noise N_ell per channel: (n_chan, n_ells)
    noise_nls = jnp.stack(
        [
            noise_nl_continuous(
                net[i],
                n_det[i],
                beam_fwhm[i],
                eta_total[i],
                ells,
                mission_years,
                f_sky,
                knee_arr[i],
                alpha_arr[i],
            )
            for i in range(n_chan)
        ]
    )

    # Covariance blocks: (n_bins, n_spec, n_spec)
    cov_blocks = bandpower_covariance_blocks_from_noise(
        ctx.signal_model, noise_nls, f_sky, ctx.fiducial_params
    )

    # Fisher matrix: J^T Sigma^{-1} J via the unified primitive.
    F = _fisher_from_blocks(ctx.J_blocks, cov_blocks)

    # Add priors
    F = F + jnp.diag(ctx.prior_diag)

    # Invert and extract sigma(r)
    F_inv = jnp.linalg.inv(F)
    return jnp.sqrt(F_inv[ctx.r_idx, ctx.r_idx])


def sigma_r_from_external_cov(
    external_covariance: jnp.ndarray,
    ctx: OptimizationContext,
) -> jnp.ndarray:
    """Differentiable sigma(r) from a full bandpower covariance (cut-sky / MC path).

    The jnp-returning analogue of
    ``FisherForecast(external_covariance=...).sigma("r")``: builds
    ``F = Jᵀ C⁻¹ J`` via the same prewhitened dense solve
    (``fisher._fisher_from_full``), adds Gaussian priors on the diagonal,
    inverts, and returns ``sqrt((F⁻¹)_{rr})`` as a JAX scalar — no ``float()``
    boundary, so it is differentiable in ``external_covariance``.

    This is the *consumer* end of the end-to-end map-based optimization: the
    covariance is the output of the cut-sky masked-Wiener Monte-Carlo stage
    (:func:`augr.spectrum_stages.mc_cutsky_bandpowers`), itself a function of
    the instrument design. Composing this with that traced stage gives a
    ``jax.grad``-able σ(r) through component separation. The analytic
    block-diagonal counterpart is :func:`sigma_r_from_channels`.

    The Jacobian ``ctx.J`` is structural (it depends on the cleaned-map
    ``SignalModel`` — frequencies, binning, residual template — not on the
    covariance), so it is pre-computed once and held fixed; only the noise →
    covariance path carries the design dependence, exactly as in
    :func:`sigma_r_from_channels`.

    Args:
        external_covariance: full ``(n_data, n_data)`` bandpower covariance,
            ``n_data = n_spec × n_bins`` (just ``n_bins`` for a single cleaned
            map), on the same binning as ``ctx.signal_model``. E.g.
            ``mc_cutsky_bandpowers(...).covariance``.
        ctx: ``OptimizationContext`` built on the cleaned-map ``SignalModel``
            (the residual-template / ``NullForegroundModel`` forecast); supplies
            ``J``, ``prior_diag``, and ``r_idx``.

    Returns:
        Scalar sigma(r).

    Note:
        Routes through ``fisher._fisher_from_full`` — the same prewhitened dense
        solve as ``FisherForecast(external_covariance=...).compute()`` — so the
        two paths agree to fp64 precision (the optimize path simply keeps the
        result as a JAX array instead of casting to ``float`` in ``sigma()``).
    """
    F = _fisher_from_full(ctx.J, jnp.asarray(external_covariance))
    F = F + jnp.diag(ctx.prior_diag)
    F_inv = jnp.linalg.inv(F)
    return jnp.sqrt(F_inv[ctx.r_idx, ctx.r_idx])


def design_to_channels(
    aperture_m,
    f_number,
    fp_diameter_m,
    area_fractions,
    freqs_per_group: tuple[tuple[float, ...], ...],
    *,
    net_override=None,
    illumination_factor: float = 1.22,
    packing_efficiency: float = 0.80,
):
    """Differentiable telescope design -> per-channel ``(n_det, net, beam)``.

    The focal-plane packing physics shared by the analytic (:func:`sigma_r_from_design`)
    and map-based (:func:`augr.eig.physical_design_objective`) forecasts:

    - horn diameter set by the lowest band in each pixel group (Griffin 2002,
      ``d = 2 F lambda``);
    - hex cell area + continuous pixel count over the group's allocated focal-plane area
      ``area_fractions[g] * pi (fp_diameter / 2)^2``;
    - dichroic groups share a horn, so ``n_det = 2 * n_pixels`` (dual-pol) at *each* band
      in the group;
    - NET per channel from photon noise (:func:`augr.telescope.photon_noise_net_jax`)
      unless ``net_override`` is supplied;
    - beam per channel from the single physical aperture (:func:`beam_fwhm_arcmin`).

    Args:
        aperture_m:      Primary mirror diameter [m].
        f_number:        Focal ratio f/D.
        fp_diameter_m:   Usable focal plane diameter [m] (sets the fixed total area).
        area_fractions:  Focal-plane area allocation per group, shape ``(n_groups,)``.
        freqs_per_group: Per-group frequency tuples (1 or 2 bands each), e.g.
                         ``((20.,), (35.,), (80., 115.), ...)``.
        net_override:    If given, per-channel NETs to use instead of photon noise,
                         shape ``(n_chan,)`` in flattened-group order.
        illumination_factor: FWHM = factor x lambda/D (1.22 for Airy).
        packing_efficiency:  Fraction of ideal hex packing achieved.

    Returns:
        ``(n_det, net, beam)``, each ``(n_chan,)`` in flattened ``freqs_per_group`` order.
        Differentiable in ``aperture_m``, ``f_number``, ``area_fractions`` (and
        ``net_override`` if given).
    """
    a_fp = jnp.pi * (fp_diameter_m / 2.0) ** 2
    n_det_list, beam_list, net_list = [], [], []
    chan_idx = 0
    for g, freqs in enumerate(freqs_per_group):
        nu_low = min(freqs)  # static (Python float)
        d_horn = horn_diameter(nu_low, f_number)
        a_cell = (jnp.sqrt(3.0) / 2.0) * d_horn**2  # hex_cell_area, JAX
        a_alloc = area_fractions[g] * a_fp
        n_pixels = count_pixels_continuous(a_alloc, a_cell, packing_efficiency)
        n_det_group = 2.0 * n_pixels  # dual-pol; dichroic bands share the horn
        for nu in freqs:
            n_det_list.append(n_det_group)
            beam_list.append(beam_fwhm_arcmin(nu, aperture_m, illumination_factor))
            if net_override is not None:
                net_list.append(net_override[chan_idx])
            else:
                net_list.append(photon_noise_net_jax(nu))
            chan_idx += 1
    return jnp.stack(n_det_list), jnp.stack(net_list), jnp.stack(beam_list)


def sigma_r_from_design(
    aperture_m: jnp.ndarray,
    f_number: jnp.ndarray,
    fp_diameter_m: jnp.ndarray,
    area_fractions: jnp.ndarray,
    ctx: OptimizationContext,
    freqs_per_group: tuple[tuple[float, ...], ...],
    mission_years: float = 5.0,
    f_sky: float = 0.7,
    net_override: jnp.ndarray | None = None,
    illumination_factor: float = 1.22,
    packing_efficiency: float = 0.80,
    eta_total: jnp.ndarray | float = 0.50,
    knee_ell: jnp.ndarray | float = 0.0,
    alpha_knee: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Differentiable sigma(r) from telescope design parameters.

    Tier 2 optimization: optimize aperture, f_number, focal plane diameter,
    and area fractions. Detector counts, beam sizes, and (optionally) NETs
    are derived from the physics.

    Args:
        aperture_m:      Primary mirror diameter [m].
        f_number:        Focal ratio f/D.
        fp_diameter_m:   Usable focal plane diameter [m].
        area_fractions:  Focal plane area allocation per group, shape (n_groups,).
        ctx:             Pre-computed OptimizationContext.
        freqs_per_group: Channel frequencies per pixel group,
                         e.g. ((30., 40.), (85., 150.), (220., 340.)).
        mission_years:   Mission duration [years].
        f_sky:           Sky fraction.
        net_override:    If provided, use these NETs instead of computing
                         from photon noise. Shape (n_chan,).
        illumination_factor: FWHM = factor × λ/D (1.22 for Airy).
        packing_efficiency:  Fraction of ideal hex packing achieved.
        eta_total:       Total efficiency (scalar or per-channel).
        knee_ell:        1/f knee multipole.
        alpha_knee:      1/f spectral index.

    Returns:
        Scalar sigma(r).

    Note:
        Delegates to sigma_r_from_channels, which routes through
        ``fisher._fisher_from_blocks`` -- the same primitive as
        ``FisherForecast.sigma``. The two paths agree to fp64 precision.
    """
    n_det_arr, net_arr, beam_arr = design_to_channels(
        aperture_m,
        f_number,
        fp_diameter_m,
        area_fractions,
        freqs_per_group,
        net_override=net_override,
        illumination_factor=illumination_factor,
        packing_efficiency=packing_efficiency,
    )

    eta_arr = (
        jnp.full(n_det_arr.shape, eta_total) if jnp.ndim(eta_total) == 0 else eta_total
    )

    return sigma_r_from_channels(
        n_det_arr,
        net_arr,
        beam_arr,
        eta_arr,
        ctx,
        mission_years,
        f_sky,
        knee_ell,
        alpha_knee,
    )
