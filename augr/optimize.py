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
from augr.delensing import AUTO_N_L_SAMPLE, LensingSpectra, delens_residual_bb
from augr.fisher import _fisher_from_blocks, _fisher_from_full
from augr.instrument import (
    Instrument,
    beam_bl,
    noise_nl_continuous,
    white_noise_power_continuous,
)
from augr.signal import SignalModel, flatten_params
from augr.telescope import (
    beam_fwhm_arcmin,
    count_pixels_continuous,
    horn_diameter,
    photon_noise_net_jax,
)

# Backward-compat alias; new code should use _fisher_from_blocks directly.
_fisher_from_blocks_solve = _fisher_from_blocks


def _combined_white_nl_bb(n_det: jnp.ndarray,
                          net: jnp.ndarray,
                          beam: jnp.ndarray,
                          eta: jnp.ndarray,
                          ells: jnp.ndarray,
                          mission_years: float,
                          f_sky: float) -> jnp.ndarray:
    """Inverse-variance-combined *white* polarization noise N_l^BB.

    White (no 1/f) by design: the QE reconstruction is dominated by the E/B lensing
    peak (l ~ 500-2000) where 1/f is negligible, and it keeps the context-build
    reference solve consistent with the per-design eval regardless of the
    covariance-side ``knee_ell``.

    ``net`` / ``beam`` / ``eta`` may each be a scalar, broadcast across the channels
    ``n_det`` defines.

    Accumulate the per-channel inverse weight as ``b_l**2 / w_inv``, never as
    ``1 / (w_inv / b_l**2)``: the latter overflows to ``+inf`` for a large-beam
    channel on the delensing ell grid, and while the value is still correct the
    backward pass then evaluates ``0 * inf`` -> NaN.
    """
    n_chan = n_det.shape[0]
    net, beam, eta = (jnp.broadcast_to(jnp.asarray(x, dtype=float), (n_chan,))
                      for x in (net, beam, eta))
    inv = jnp.zeros_like(ells, dtype=float)
    for i in range(n_chan):
        # 1 / nl_i, formed without ever materializing nl_i = w_inv / b_l**2.
        w_inv_i = white_noise_power_continuous(
            net[i], n_det[i], eta[i], mission_years, f_sky)
        inv = inv + beam_bl(ells, beam[i]) ** 2 / w_inv_i
    return 1.0 / inv


def _delens_from_combined_bb(spectra: LensingSpectra,
                             nl_bb: jnp.ndarray,
                             ls: jnp.ndarray,
                             l_max_qe: int,
                             n_iter: int,
                             remat: bool = True,
                             fullsky: bool = True,
                             n_L_sample: int | str | None = AUTO_N_L_SAMPLE,
                             l_batch: int = 1) -> jnp.ndarray:
    """Residual lensing BB from the single combined polarization noise.

    In the design forward the inverse-variance-combined noise obeys the
    pol/temperature relation ``nl_ee = nl_bb`` and ``nl_tt = nl_bb / 2``, so
    the iterative-QE residual is a function of ``nl_bb`` alone. ``nl_bb`` is
    indexed on ``spectra.ells`` (the LensingSpectra grid).  Differentiable
    (``delens_residual_bb``, flat-sky or full-sky), so this composes into
    ``jax.grad``.
    """
    return delens_residual_bb(
        spectra, nl_bb / 2.0, nl_bb, nl_bb,
        ls=ls, L_max=l_max_qe, l_max_qe=l_max_qe, n_iter=n_iter,
        remat=remat, fullsky=fullsky, n_L_sample=n_L_sample,
        l_batch=l_batch)


@dataclass(frozen=True)
class DelensCoupling:
    """Design-dependent residual lensing BB from an iterative-QE solve.

    Build it with :meth:`build` at a reference design; call :meth:`residual` per
    design. The residual is a full solve at each evaluation, so its **value** is
    exact at every design, not an expansion about the reference.

    No linearized mode, deliberately: ``delens='linearized'`` in
    ``make_optimization_context`` was measured to give a 118% error (wrong sign) for
    a 5% change in detector count at a 3-band reference, because
    ``delens_residual_bb`` has a jump discontinuity in the noise at isolated
    multipoles that dominates the Jacobian contraction.

    That discontinuity also limits the design gradient here and in the analytic
    ``delens=`` path: the value is trustworthy, ``jax.grad`` through the QE solve
    is not.
    """

    spectra: LensingSpectra
    ls: jnp.ndarray            # ell grid of the residual
    ells: jnp.ndarray          # noise grid (spectra.ells)
    l_max_qe: int
    n_iter: int
    nl_bb0: jnp.ndarray        # reference combined white nl_bb (on ells)
    cl_bb_res0: jnp.ndarray    # reference residual (on ls)
    remat: bool = True         # gradient-checkpoint the QE scans
    #: ``remat`` must be read by BOTH :meth:`build` and :meth:`residual`; it is
    #: forward-transparent, so setting it on one side only passes every value
    #: test while leaving the other on the O(l_max_qe**2) tape.
    fullsky: bool = True       # full-sky Wigner-3j QE (JAX backend); False = flat-sky
    n_L_sample: int | str | None = AUTO_N_L_SAMPLE  # full-sky N_0 L grid; None = every L
    l_batch: int = 1           # full-sky only: L values vmapped per map step
    #: ``fullsky`` / ``n_L_sample`` / ``l_batch`` must enter BOTH solves: the
    #: reference residual and the per-design one have to be the same
    #: approximation, or the "reproduces cl_bb_res0 at the reference" contract
    #: breaks silently. This includes ``l_batch``, whose values are bit-identical
    #: today but need not stay so across XLA versions.

    @classmethod
    def build(
        cls,
        *,
        lensing_spectra: LensingSpectra,
        n_det,
        net,
        beam,
        eta,
        mission_years: float,
        f_sky: float,
        l_max_qe: int = 1000,
        n_iter: int = 5,
        ls: jnp.ndarray | None = None,
        remat: bool = True,
        fullsky: bool = True,
        n_L_sample: int | str | None = AUTO_N_L_SAMPLE,
        l_batch: int = 1,
    ) -> DelensCoupling:
        """Precompute the coupling at a reference design (one delensing solve).

        ``fullsky=True`` (default) runs the full-sky Wigner-3j QE on the sampled N_0 L
        grid ``n_L_sample`` (``"auto"`` = :func:`augr.delensing.default_n_L_sample`,
        ``None`` = every L); ``fullsky=False`` is the flat-sky Gauss-Legendre QE.

        ``n_det`` / ``net`` / ``beam`` / ``eta`` are the reference design's per-channel
        arrays (as :func:`design_to_channels` produces); ``f_sky`` / ``mission_years``
        set the noise normalization. :attr:`cl_bb_res0` is the residual there -- hand it
        to the forecast's ``SignalModel`` as ``delensed_bb`` so the model half of the
        coupling matches the sims at the reference.
        """
        ells = lensing_spectra.ells
        ls_arr = jnp.arange(2, 301, dtype=float) if ls is None else jnp.asarray(ls)
        nl_bb0 = _combined_white_nl_bb(
            jnp.asarray(n_det), jnp.asarray(net), jnp.asarray(beam), jnp.asarray(eta),
            ells, mission_years, f_sky)
        cl_res0 = _delens_from_combined_bb(
            lensing_spectra, nl_bb0, ls_arr, l_max_qe, n_iter, remat,
            fullsky=fullsky, n_L_sample=n_L_sample, l_batch=l_batch)
        return cls(
            spectra=lensing_spectra,
            ls=ls_arr,
            ells=ells,
            l_max_qe=int(l_max_qe),
            n_iter=int(n_iter),
            nl_bb0=nl_bb0,
            cl_bb_res0=cl_res0,
            remat=bool(remat),
            fullsky=bool(fullsky),
            n_L_sample=(n_L_sample if n_L_sample is None or isinstance(n_L_sample, str)
                        else int(n_L_sample)),
            l_batch=int(l_batch),
        )

    def residual(self, n_det, net, beam, eta, mission_years, f_sky):
        """Residual lensing ``C_ell^BB`` on :attr:`ls` for this design.

        Exact at every design (a full solve, not an expansion), and reproduces
        :attr:`cl_bb_res0` at the reference. Traceable, but see the class docstring
        before trusting ``jax.grad`` through it.
        """
        nl_bb = _combined_white_nl_bb(
            jnp.asarray(n_det), jnp.asarray(net), jnp.asarray(beam), jnp.asarray(eta),
            self.ells, mission_years, f_sky)
        return _delens_from_combined_bb(
            self.spectra, nl_bb, self.ls, self.l_max_qe, self.n_iter,
            self.remat, fullsky=self.fullsky, n_L_sample=self.n_L_sample,
            l_batch=self.l_batch)


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

    # Self-consistent delensing coupling (issue #45 Stage 2). All None when
    # delens is off -> the forward is byte-identical to the frozen-A_lens /
    # frozen-delensed_bb path. See make_optimization_context(delens=...).
    delens_mode: str | None = None            # None | 'recompute' | 'linearized'
    lensing_spectra: LensingSpectra | None = None
    delens_ls: jnp.ndarray | None = None      # ell grid of cl_bb_res
    delens_ells: jnp.ndarray | None = None    # noise grid for delensing (spectra.ells)
    delens_l_max_qe: int = 1000
    delens_n_iter: int = 5
    delens_cl_bb_res0: jnp.ndarray | None = None  # reference residual (on delens_ls)
    delens_nl_bb0: jnp.ndarray | None = None      # reference combined nl_bb (on delens_ells)
    delens_jac: jnp.ndarray | None = None         # d(cl_bb_res)/d(nl_bb), linearized mode
    delens_remat: bool = True                     # checkpoint the QE scans (see delensing._scan)
    delens_fullsky: bool = True                   # full-sky Wigner-3j QE (JAX); False = flat-sky
    delens_n_L_sample: int | str | None = AUTO_N_L_SAMPLE  # full-sky N_0 L grid; None = every L
    delens_l_batch: int = 1                       # full-sky only: L values vmapped per map step


def make_optimization_context(
    instrument: Instrument,
    foreground_model,
    cmb_spectra,
    fiducial_params: dict[str, float],
    priors: dict[str, float] | None = None,
    fixed_params: list[str] | None = None,
    *,
    delens: str | None = None,
    lensing_spectra: LensingSpectra | None = None,
    delens_l_max_qe: int = 1000,
    delens_n_iter: int = 5,
    delens_ls: jnp.ndarray | None = None,
    delens_remat: bool = True,
    delens_fullsky: bool = True,
    delens_n_L_sample: int | str | None = AUTO_N_L_SAMPLE,
    delens_l_batch: int = 1,
    **signal_kwargs,
) -> OptimizationContext:
    """One-time setup for differentiable sigma(r) optimization.

    Builds the SignalModel, pre-computes the Jacobian, assembles the prior
    structure, and extracts channel parameters as JAX arrays. The returned context
    is passed to sigma_r_from_channels or sigma_r_from_design.

    Args:
        instrument:      Reference instrument (defines frequencies, structure).
        foreground_model: ForegroundModel (Gaussian or Moment).
        cmb_spectra:     CMBSpectra instance.
        fiducial_params: Dict of fiducial parameter values.
        priors:          Dict mapping param name -> prior sigma.
        fixed_params:    List of params to hold fixed.
        delens:          None (off), 'recompute', or 'linearized' -- design-dependent
                         delensing in the forward; requires ``lensing_spectra`` and
                         owns ``delensed_bb``. 'recompute' re-runs the QE residual
                         every eval (exact, seconds per solve); 'linearized'
                         precomputes d(cl_bb_res)/d(nl_bb) once and applies it
                         linearly (near-free per eval, first-order accurate).
        lensing_spectra: LensingSpectra for the QE delensing (delens only).
        delens_l_max_qe: Max QE multipole for the delensing (delens only).
        delens_n_iter:   Delensing iterations (delens only).
        delens_remat:    Gradient-checkpoint the QE scans (delens only, default
                         True). Forward-transparent; required to keep the backward
                         tape bounded at production l_max_qe.
        delens_ls:       ell grid for the residual (delens only; default 2..300,
                         must span the SignalModel [ell_min, ell_max]).
        delens_fullsky:  True (default): full-sky Wigner-3j QE on the sampled N_0
                         grid -- exact low-L geometry, and the parallel path.
                         False: flat-sky Gauss-Legendre QE, whose l1 scan is serial.
        delens_n_L_sample: full-sky only; N_0 L-sample grid (``"auto"`` =
                         ``delensing.default_n_L_sample``, ``None`` = every L).
        delens_l_batch:  full-sky only; L values vmapped into each step of the
                         per-L map (default 1). Wall-clock knob only; trace-time
                         constant, like ``delens_remat``.
        **signal_kwargs: Passed to SignalModel (ell_min, ell_max, delta_ell,
                         ell_per_bin_below, delensed_bb, etc.)

    Returns:
        OptimizationContext for use with sigma_r_from_channels.
    """
    priors = priors or {}
    fixed_params = fixed_params or []

    # Channel parameters as JAX arrays (also the reference point for delensing).
    channels = instrument.channels
    n_det = jnp.array([float(ch.n_detectors) for ch in channels])
    net = jnp.array([ch.net_per_detector for ch in channels])
    beam = jnp.array([ch.beam_fwhm_arcmin for ch in channels])
    eta = jnp.array([ch.efficiency.total for ch in channels])
    freqs = tuple(ch.nu_ghz for ch in channels)

    # Reference delensing solve at the supplied instrument: it puts the
    # SignalModel in delensed mode (A_lens drops out; the residual is additive
    # in r, so the Jacobian stays structural) and serves as the linearization
    # point. Same white-noise combine as the per-eval path, so recompute at the
    # reference reproduces this residual when mission_years / f_sky match.
    if delens not in (None, "recompute", "linearized"):
        raise ValueError(
            f"delens must be None, 'recompute', or 'linearized'; got {delens!r}")
    delens_ls_arr = delens_nl_bb0 = delens_cl_bb_res0 = delens_jac = None
    delens_ells_arr = None
    if delens is not None:
        if lensing_spectra is None:
            raise ValueError(
                "delens requires lensing_spectra=load_lensing_spectra(...).")
        if "delensed_bb" in signal_kwargs:
            raise ValueError(
                "delens=... owns delensed_bb; do not also pass it in "
                "signal_kwargs.")
        delens_ells_arr = lensing_spectra.ells
        delens_ls_arr = (jnp.arange(2, 301, dtype=float)
                         if delens_ls is None else delens_ls)
        delens_nl_bb0 = _combined_white_nl_bb(
            n_det, net, beam, eta, delens_ells_arr,
            instrument.mission_duration_years, instrument.f_sky)
        delens_cl_bb_res0 = _delens_from_combined_bb(
            lensing_spectra, delens_nl_bb0, delens_ls_arr,
            delens_l_max_qe, delens_n_iter, delens_remat,
            fullsky=delens_fullsky, n_L_sample=delens_n_L_sample,
            l_batch=delens_l_batch)
        if delens == "linearized":
            # d(cl_bb_res)/d(nl_bb) at the reference; O(n_ls) solves, amortized
            # over many cheap evals.
            delens_jac = jax.jacrev(
                lambda nlbb: _delens_from_combined_bb(
                    lensing_spectra, nlbb, delens_ls_arr,
                    delens_l_max_qe, delens_n_iter,
                    delens_remat, fullsky=delens_fullsky,
                    n_L_sample=delens_n_L_sample,
                    l_batch=delens_l_batch))(delens_nl_bb0)
        # Put the SignalModel in delensed mode at the reference residual.
        signal_kwargs = dict(signal_kwargs)
        signal_kwargs["delensed_bb"] = delens_cl_bb_res0
        signal_kwargs["delensed_bb_ells"] = delens_ls_arr

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
        delens_mode=delens,
        lensing_spectra=lensing_spectra,
        delens_ls=delens_ls_arr,
        delens_ells=delens_ells_arr,
        delens_l_max_qe=delens_l_max_qe,
        delens_remat=delens_remat,
        delens_fullsky=delens_fullsky,
        delens_n_L_sample=delens_n_L_sample,
        delens_l_batch=delens_l_batch,
        delens_n_iter=delens_n_iter,
        delens_cl_bb_res0=delens_cl_bb_res0,
        delens_nl_bb0=delens_nl_bb0,
        delens_jac=delens_jac,
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
        Uses ``fisher._fisher_from_blocks``, the same primitive as
        ``FisherForecast.sigma``; the two paths agree to fp64 precision.
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

    # Residual lensing BB for this design, fed to the covariance as a
    # delensed_bb override. None when delens is off.
    delensed_override = None
    if ctx.delens_mode is not None:
        # Combined *white* pol noise on the delensing ell grid; nl_ee = nl_bb
        # and nl_tt = nl_bb/2 are applied inside _delens_from_combined_bb.
        nl_bb_del = _combined_white_nl_bb(
            n_det, net, beam_fwhm, eta_total, ctx.delens_ells,
            mission_years, f_sky)
        if ctx.delens_mode == "recompute":
            cl_res = _delens_from_combined_bb(
                ctx.lensing_spectra, nl_bb_del, ctx.delens_ls,
                ctx.delens_l_max_qe, ctx.delens_n_iter, ctx.delens_remat,
                fullsky=ctx.delens_fullsky, n_L_sample=ctx.delens_n_L_sample,
                l_batch=ctx.delens_l_batch)
        else:  # 'linearized': cl_bb_res0 + J (nl_bb - nl_bb0)
            cl_res = ctx.delens_cl_bb_res0 + ctx.delens_jac @ (
                nl_bb_del - ctx.delens_nl_bb0)
        # Interpolate onto the signal ell grid used by cmb_bb_unbinned.
        delensed_override = jnp.interp(ells, ctx.delens_ls, cl_res)

    # Covariance blocks: (n_bins, n_spec, n_spec)
    cov_blocks = bandpower_covariance_blocks_from_noise(
        ctx.signal_model, noise_nls, f_sky, ctx.fiducial_params,
        delensed_bb_override=delensed_override,
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

    jnp-returning analogue of ``FisherForecast(external_covariance=...).sigma("r")``:
    ``F = J^T C^-1 J`` via ``fisher._fisher_from_full``, Gaussian priors on the
    diagonal, inverted for ``sqrt((F^-1)_rr)``. No ``float()`` boundary, so it is
    differentiable in ``external_covariance``. The analytic block-diagonal
    counterpart is :func:`sigma_r_from_channels`.

    ``ctx.J`` is structural -- it depends on the cleaned-map ``SignalModel``, not on
    the covariance -- so it is fixed here and only the noise -> covariance path
    carries the design dependence.

    Args:
        external_covariance: full ``(n_data, n_data)`` bandpower covariance,
            ``n_data = n_spec x n_bins`` (just ``n_bins`` for a single cleaned map),
            on the same binning as ``ctx.signal_model``. E.g.
            ``mc_cutsky_bandpowers(...).covariance``.
        ctx: ``OptimizationContext`` built on the cleaned-map ``SignalModel``;
            supplies ``J``, ``prior_diag`` and ``r_idx``.

    Returns:
        Scalar sigma(r).
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
    extra_loading=None,
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
        extra_loading:   Optional jnp-traceable ``n_extra(nu_hz) -> occupation``
                         (e.g. ``config.galactic_extra_loading()``) added to
                         every band's photon-noise NET. Ignored when
                         ``net_override`` is supplied. Default ``None``
                         reproduces the no-Galactic-loading baseline.

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
                net_list.append(photon_noise_net_jax(nu, extra_loading=extra_loading))
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
    extra_loading=None,
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
        extra_loading:   Optional jnp-traceable ``n_extra(nu_hz) -> occupation``
                         (e.g. ``config.galactic_extra_loading()``) added to
                         every band's photon-noise NET. Default ``None``
                         reproduces the no-Galactic-loading baseline. Ignored
                         when ``net_override`` is supplied.

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
        extra_loading=extra_loading,
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
