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

import jax
import jax.numpy as jnp

from augr.covariance import bandpower_covariance_blocks_from_noise
from augr.instrument import Instrument, noise_nl_continuous
from augr.signal import SignalModel, flatten_params
from augr.telescope import (
    beam_fwhm_arcmin,
    count_pixels_continuous,
    horn_diameter,
)


@jax.jit
def _fisher_from_blocks_solve(J_blocks: jnp.ndarray,
                               cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Fisher matrix via Cholesky solve (gradient-friendly).

    Same result as fisher._fisher_from_blocks for well-conditioned blocks,
    but uses jnp.linalg.solve instead of eigendecomposition. This gives
    stable gradients even when covariance blocks have condition numbers
    ~10^17 (common for deep multifrequency instruments), where the eigh
    gradient amplifies numerical noise through near-zero eigenvalues.

    For the forward computation the eigh approach is slightly more robust
    (projects out degenerate directions). For gradient-based optimization
    this solve approach is preferred.
    """
    def _one_bin(carry, inputs):
        J_b, cov_b = inputs
        SinvJ = jnp.linalg.solve(cov_b, J_b)
        return carry + J_b.T @ SinvJ, None

    n_free = J_blocks.shape[2]
    F, _ = jax.lax.scan(_one_bin, jnp.zeros((n_free, n_free)),
                         (J_blocks, cov_blocks))
    return F


@dataclass(frozen=True)
class OptimizationContext:
    """Pre-computed quantities that are static during instrument optimization.

    Built once by make_optimization_context(). Holds the signal model,
    pre-computed Jacobian blocks, prior structure, and initial channel
    parameters extracted from the reference instrument.

    Attributes:
        signal_model: Pre-built SignalModel (defines frequencies, binning).
        J_blocks:     Jacobian reshaped to (n_bins, n_spec, n_free).
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
    sig = SignalModel(instrument, foreground_model, cmb_spectra,
                      **signal_kwargs)

    # Parameter bookkeeping
    all_names = sig.parameter_names
    free_names = [n for n in all_names if n not in set(fixed_params)]
    free_idx = jnp.array([all_names.index(n) for n in free_names])

    # Flatten fiducial params
    params = flatten_params(fiducial_params, all_names)

    # Pre-compute Jacobian (depends on foreground params + frequencies, not
    # on instrument noise/beam/n_det)
    J_full = sig.jacobian(params)  # (n_data, n_all_params)
    J = J_full[:, free_idx]        # (n_data, n_free)

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
        Scalar sigma(r) — marginalized Fisher constraint on r.

    Note:
        This function uses a ``jnp.linalg.solve``-based Fisher block
        inversion (_fisher_from_blocks_solve) rather than the
        eigendecomposition path used by ``FisherForecast.sigma``.  The
        solve path is smoother for autodiff (no conditional zero-clipping
        of eigenvalues) but the two can disagree by up to a few percent
        in degenerate-cov regimes where ``FisherForecast`` would clip
        near-zero eigenvalues.  For optimization work the solve path is
        the one that is actually being minimized; for reporting absolute
        sigma(r) numbers prefer ``FisherForecast.sigma``.
    """
    ells = ctx.ells
    n_chan = n_det.shape[0]

    # Broadcast scalar knee_ell / alpha_knee to per-channel arrays
    knee_arr = jnp.broadcast_to(jnp.asarray(knee_ell), (n_chan,))
    alpha_arr = jnp.broadcast_to(jnp.asarray(alpha_knee), (n_chan,))

    # Compute noise N_ell per channel: (n_chan, n_ells)
    noise_nls = jnp.stack([
        noise_nl_continuous(
            net[i], n_det[i], beam_fwhm[i], eta_total[i],
            ells, mission_years, f_sky,
            knee_arr[i], alpha_arr[i],
        )
        for i in range(n_chan)
    ])

    # Covariance blocks: (n_bins, n_spec, n_spec)
    cov_blocks = bandpower_covariance_blocks_from_noise(
        ctx.signal_model, noise_nls, f_sky, ctx.fiducial_params)

    # Fisher matrix: J^T Sigma^{-1} J (solve-based for stable gradients)
    F = _fisher_from_blocks_solve(ctx.J_blocks, cov_blocks)

    # Add priors
    F = F + jnp.diag(ctx.prior_diag)

    # Invert and extract sigma(r)
    F_inv = jnp.linalg.inv(F)
    return jnp.sqrt(F_inv[ctx.r_idx, ctx.r_idx])


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
        Delegates to sigma_r_from_channels, which uses a solve-based
        Fisher inversion that can disagree with FisherForecast.sigma by
        a few percent in degenerate-cov regimes.  See the note on
        sigma_r_from_channels for details.
    """
    a_fp = jnp.pi * (fp_diameter_m / 2.0) ** 2

    # Derive per-channel params from design
    n_det_list = []
    beam_list = []
    net_list = []

    chan_idx = 0
    for g, freqs in enumerate(freqs_per_group):
        nu_low = min(freqs)  # static (Python float)
        d_horn = horn_diameter(nu_low, f_number)
        a_cell = (jnp.sqrt(3.0) / 2.0) * d_horn ** 2  # hex_cell_area, JAX
        a_alloc = area_fractions[g] * a_fp
        n_pixels = count_pixels_continuous(a_alloc, a_cell, packing_efficiency)
        n_det_group = 2.0 * n_pixels

        for nu in freqs:
            beam = beam_fwhm_arcmin(nu, aperture_m, illumination_factor)
            n_det_list.append(n_det_group)
            beam_list.append(beam)
            if net_override is not None:
                net_list.append(net_override[chan_idx])
            else:
                from augr.telescope import photon_noise_net_jax
                net_list.append(photon_noise_net_jax(nu))
            chan_idx += 1

    n_det_arr = jnp.stack(n_det_list)
    beam_arr = jnp.stack(beam_list)
    net_arr = jnp.stack(net_list)

    eta_arr = jnp.full(n_det_arr.shape, eta_total) if jnp.ndim(eta_total) == 0 else eta_total

    return sigma_r_from_channels(
        n_det_arr, net_arr, beam_arr, eta_arr,
        ctx, mission_years, f_sky, knee_ell, alpha_knee)
