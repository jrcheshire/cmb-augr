"""
fisher.py -- Fisher matrix computation and parameter constraints.

    F_{αβ} = Σ_b  J_b^T  Σ_b⁻¹  J_b   +   P

where J_b is the Jacobian slice for bin b, Σ_b is the (n_spec, n_spec)
Knox covariance block for that bin, and P is the diagonal Gaussian prior
matrix: P_{αα} = 1/σ_prior_α².

The covariance is block-diagonal across ℓ-bins (Knox approximation), so
rather than inverting the full (n_data, n_data) matrix, we ``solve``
each small (n_spec, n_spec) block independently.  Per-bin LU/solve is
O(n_bins × n_spec³) vs O(n_data³) for the full assembly, and validated
against mpmath ground truth on PICO-class instruments (cov_b cond ~10^28
at ell=2) -- LU/solve gives the correct Fisher to fp64 precision; an
``eigh + (s>0)`` clip biases F upward by 5-44% per bin by
face-valuing tiny positive eigenvalues that are fp64 rounding artifacts.

Fixed parameters are removed from the Fisher matrix entirely (their rows
and columns are dropped before inversion), so they do not contribute to
marginalization penalties.

Marginalized constraint:  σ_α = √(F⁻¹)_{αα}
Conditional constraint:   σ_α^{cond} = 1/√(F_{αα})
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from augr.covariance import (
    bandpower_covariance_blocks,
    bandpower_covariance_blocks_from_noise,
    bandpower_covariance_full_from_noise,
)
from augr.instrument import ARCMIN_TO_RAD, Instrument, white_noise_power
from augr.signal import SignalModel, flatten_params


@jax.jit
def _fisher_from_blocks(J_blocks: jnp.ndarray,
                        cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Compute F = sum_b J_b^T Sigma_b^{-1} J_b from per-bin blocks.

    Args:
        J_blocks:   (n_bins, n_spec, n_free) -- Jacobian reshaped per bin.
        cov_blocks: (n_bins, n_spec, n_spec) -- per-bin Knox covariance.

    Returns:
        Fisher matrix of shape (n_free, n_free).

    Each bin's contribution is computed via prewhitening (correlation-
    matrix trick: ``D_b = sqrt(diag(cov_b))``, then
    ``cov_w = D_b^{-1} cov_b D_b^{-1}``, ``J_w = D_b^{-1} J_b``) followed
    by ``jnp.linalg.solve(cov_w, J_w)``. Mathematically equivalent to
    ``J_b^T cov_b^{-1} J_b`` (the D_b's pair up algebraically): cov_b =
    D_b cov_w D_b ⇒ cov_b^{-1} = D_b^{-1} cov_w^{-1} D_b^{-1}, and the
    D_b's contract with the J_b's into J_w's. Forward F is unchanged
    within fp64 roundoff (no ε ridge; no bias).

    Why this matters numerically: cov_b at PICO 21-channel conditioning
    has condition number ~10^28 with 84/231 fp64-rounded-negative
    eigenvalues; ``jax.grad`` through ``solve(cov_b, J_b)`` amplifies
    those rounding artifacts in the backward pass, drifting per-axis
    gradients by 50-270% between jit and eager and giving random
    sign agreement with finite-difference references. After whitening
    by sqrt(diag(cov_b)), entries of cov_w are bounded by 1 in
    magnitude (Cauchy-Schwarz) and the condition number drops to
    ~10^15 at the same fixture -- still ill-conditioned, but well
    within fp64 LU's stable regime for both the forward solve and the
    autograd-traced backward solve. Empirically: post-fix per-axis
    jit-vs-eager gradient agreement is rtol ~1e-4, and
    ``cos(jax.grad, fd)`` is >0.99 at h=1e-2.

    A final ``0.5 * (F + F^T)`` symmetrization absorbs accumulated
    asymmetry from the per-bin scan; Fisher is mathematically symmetric
    so this is the correct closure.
    """
    def _one_bin(carry, inputs):
        J_b, cov_b = inputs
        d = jnp.sqrt(jnp.diag(cov_b))
        inv_d = 1.0 / d
        cov_w = cov_b * (inv_d[:, None] * inv_d[None, :])
        J_w = J_b * inv_d[:, None]
        SinvJ_w = jnp.linalg.solve(cov_w, J_w)
        return carry + J_w.T @ SinvJ_w, None

    n_free = J_blocks.shape[2]
    F, _ = jax.lax.scan(_one_bin, jnp.zeros((n_free, n_free)),
                         (J_blocks, cov_blocks))
    return 0.5 * (F + F.T)


@jax.jit
def _fisher_from_full(J: jnp.ndarray,
                      cov: jnp.ndarray) -> jnp.ndarray:
    """Compute F = J^T cov^-1 J for the full (n_data, n_data) covariance.

    Used for the BPWF-aware path where bins couple and the per-bin
    block-diagonal solve is unavailable. Same prewhiten + solve
    primitive as ``_fisher_from_blocks``: ``D = sqrt(diag(cov))``,
    ``cov_w = D^{-1} cov D^{-1}``, ``J_w = D^{-1} J``, then
    ``jnp.linalg.solve(cov_w, J_w)``. Symmetrised for the same reason.
    """
    d = jnp.sqrt(jnp.diag(cov))
    inv_d = 1.0 / d
    cov_w = cov * (inv_d[:, None] * inv_d[None, :])
    J_w = J * inv_d[:, None]
    F = J_w.T @ jnp.linalg.solve(cov_w, J_w)
    return 0.5 * (F + F.T)


class FisherForecast:
    """Fisher information matrix and parameter constraints.

    Args:
        signal_model:   SignalModel defining the data vector and Jacobian.
        instrument:     Instrument specification (for noise in covariance).
        fiducial_params: Dict of all parameter fiducial values.
        priors:         Dict mapping parameter name -> prior width sigma.
                        Adds 1/sigma^2 to the diagonal of F.
        fixed_params:   List of parameter names to hold fixed (not varied).
                        These are excluded from the Fisher matrix.
        external_noise_bb: Optional pre-computed noise N_ell^BB per channel,
                        shape (n_channels, n_ells) on signal_model.ells. When
                        provided, the analytic per-channel noise computation
                        is bypassed and this array is used directly in the
                        Knox covariance. Use for post-component-separation
                        forecasts where the "channel" is a single cleaned map
                        and the noise comes from a sim-based pipeline. The
                        default (None) keeps the existing analytic behavior.

                        **IMPORTANT: external_noise_bb must be beam-
                        deconvolved.** The signal side of the covariance
                        uses raw C_ell (no B_ell^2 factor), so a beam-
                        convolved noise array will make the Fisher
                        over-optimistic at every ell where B_ell^2 < 1
                        (factor of ~2 at ell=300 for a LiteBIRD-scale beam).
                        If the source pipeline returned a noise auto-
                        spectrum from a beam-smoothed map, divide by
                        B_ell^2 first (see augr.instrument.beam_bl).
    """

    def __init__(self,
                 signal_model: SignalModel,
                 instrument: Instrument,
                 fiducial_params: dict[str, float],
                 priors: dict[str, float] | None = None,
                 fixed_params: list[str] | None = None,
                 external_noise_bb: jnp.ndarray | None = None):
        self._signal = signal_model
        self._instrument = instrument
        self._fiducial = dict(fiducial_params)
        self._priors = priors or {}
        self._fixed = set(fixed_params or [])

        if external_noise_bb is not None:
            external_noise_bb = jnp.asarray(external_noise_bb)
            n_chan = len(instrument.channels)
            n_ells = len(signal_model.ells)
            if external_noise_bb.shape != (n_chan, n_ells):
                raise ValueError(
                    f"external_noise_bb has shape {external_noise_bb.shape}; "
                    f"expected ({n_chan}, {n_ells}) to match the instrument "
                    f"channel count and the SignalModel ell grid.")
        elif getattr(instrument, "requires_external_noise", False):
            # Presets like cleaned_map_instrument carry placeholder NET / beam
            # values and would silently produce a nonsensical analytic Fisher
            # if dropped into this code path without external_noise_bb.
            raise ValueError(
                "This instrument was constructed with "
                "requires_external_noise=True (e.g. cleaned_map_instrument); "
                "its Channel noise parameters are placeholders. Pass the "
                "post-component-separation noise spectrum via "
                "FisherForecast(external_noise_bb=...).")
        elif signal_model.has_measured_bpwf:
            # Measured BPWFs released by analysis pipelines have the beam
            # transfer function baked in (the BPWF maps underlying sky C_ℓ
            # to estimated bandpowers, not beam-convolved C_ℓ). The signal
            # side of M = S + N here is unbeamed, so a beam-convolved
            # analytic noise from instrument.noise_nl would be inconsistent
            # at every ℓ where B_ℓ² < 1. Force the user to supply the
            # beam-deconvolved noise that paired with the BPWF release.
            raise ValueError(
                "signal_model was constructed with a measured BPWF "
                "(has_measured_bpwf=True); BPWFs released by analysis "
                "pipelines have the beam baked in, so the noise spectrum "
                "that pairs with them must be beam-deconvolved. Pass it "
                "via FisherForecast(external_noise_bb=...). Use "
                "augr.instrument.deconvolve_noise_bb if your noise array "
                "is still beam-convolved.")
        self._external_noise_bb = external_noise_bb

        self._all_names = signal_model.parameter_names
        self._free_names = [n for n in self._all_names
                            if n not in self._fixed]
        self._free_idx = jnp.array(
            [self._all_names.index(n) for n in self._free_names])

        self._fisher_matrix: jnp.ndarray | None = None
        self._fisher_inverse: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def free_parameter_names(self) -> list[str]:
        return list(self._free_names)

    @property
    def n_free(self) -> int:
        return len(self._free_names)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> jnp.ndarray:
        """Compute the Fisher information matrix for free parameters.

        Uses per-bin LU/solve on the block-diagonal covariance, avoiding
        the need to invert the full (n_data, n_data) matrix.

        Returns:
            Fisher matrix of shape (n_free, n_free).
        """
        params = flatten_params(self._fiducial, self._all_names)

        # Full Jacobian: (n_data, n_all_params) where n_data = n_spec * n_bins
        J_full = self._signal.jacobian(params)

        # Select free parameter columns
        J = J_full[:, self._free_idx]   # (n_spec * n_bins, n_free)

        # Two covariance / Fisher solve paths.
        #
        # BPWF mode: bins couple through Σ_ℓ W_b(ℓ) W_{b'}(ℓ) (2ℓ+1), so
        # the per-bin block-diagonal solve is unavailable. Build the full
        # (n_data, n_data) covariance via the per-ℓ Knox sum and do one
        # eigh-based solve.
        if self._signal.has_measured_bpwf:
            # __init__ already enforced external_noise_bb is not None when
            # has_measured_bpwf is True (the BPWF/beam-deconvolution
            # contract).
            cov_full = bandpower_covariance_full_from_noise(
                self._signal, self._external_noise_bb,
                self._instrument.f_sky, params)
            F = _fisher_from_full(J, cov_full)
        else:
            # Per-bin covariance blocks: (n_bins, n_spec, n_spec).
            # When an external noise spectrum is provided, bypass the
            # analytic per-channel noise and feed the pre-computed N_ell
            # directly to the Knox covariance. The f_sky factor still
            # comes from the Instrument.
            if self._external_noise_bb is not None:
                cov_blocks = bandpower_covariance_blocks_from_noise(
                    self._signal, self._external_noise_bb,
                    self._instrument.f_sky, params)
            else:
                cov_blocks = bandpower_covariance_blocks(
                    self._signal, self._instrument, params)

            # Reshape J into per-bin blocks: (n_bins, n_spec, n_free)
            n_spec = len(self._signal.freq_pairs)
            n_bins = self._signal.n_bins
            # Data ordering is (spec, bin): data[s * n_bins + b]
            # Reshape to (n_spec, n_bins, n_free) then transpose to
            # (n_bins, n_spec, n_free)
            J_blocks = J.reshape(n_spec, n_bins, -1).transpose(1, 0, 2)

            # F = sum_b J_b^T Sigma_b^{-1} J_b via per-bin Cholesky
            F = _fisher_from_blocks(J_blocks, cov_blocks)

        # Add Gaussian priors
        for name, sigma_prior in self._priors.items():
            if name in self._fixed:
                continue
            if name in self._free_names:
                idx = self._free_names.index(name)
                F = F.at[idx, idx].add(1.0 / sigma_prior**2)

        self._fisher_matrix = F
        self._fisher_inverse = None
        return F

    @property
    def fisher_matrix(self) -> jnp.ndarray:
        if self._fisher_matrix is None:
            self.compute()
        return self._fisher_matrix

    @property
    def inverse(self) -> jnp.ndarray:
        if self._fisher_inverse is None:
            self._fisher_inverse = jnp.linalg.inv(self.fisher_matrix)
        return self._fisher_inverse

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def sigma(self, param: str = "r") -> float:
        """Marginalized 1-sigma constraint: sqrt((F^-1)_{aa})."""
        idx = self._free_names.index(param)
        return float(jnp.sqrt(self.inverse[idx, idx]))

    def sigma_conditional(self, param: str = "r") -> float:
        """Conditional 1-sigma constraint: 1/sqrt(F_{aa})."""
        idx = self._free_names.index(param)
        return float(1.0 / jnp.sqrt(self.fisher_matrix[idx, idx]))

    def marginalized_2d(self, param_i: str, param_j: str) -> dict:
        """2D marginalized sub-covariance and error ellipse parameters."""
        idx_i = self._free_names.index(param_i)
        idx_j = self._free_names.index(param_j)

        F_inv = self.inverse
        cov_2d = jnp.array([
            [F_inv[idx_i, idx_i], F_inv[idx_i, idx_j]],
            [F_inv[idx_j, idx_i], F_inv[idx_j, idx_j]],
        ])

        sigma_i = float(jnp.sqrt(cov_2d[0, 0]))
        sigma_j = float(jnp.sqrt(cov_2d[1, 1]))
        rho = float(cov_2d[0, 1] / (sigma_i * sigma_j))

        diff = float(cov_2d[0, 0] - cov_2d[1, 1])
        angle = 0.5 * float(jnp.arctan2(2.0 * cov_2d[0, 1], diff))

        return {
            "cov_2d": cov_2d,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "rho": rho,
            "angle_deg": float(jnp.degrees(angle)),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, name: str = "") -> str:
        """Human-readable summary of all assumptions and results.

        Includes instrument channels, efficiency factors, foreground model,
        fiducial parameters, priors, fixed parameters, ℓ-binning,
        and (if computed) sigma(r).
        """
        lines: list[str] = []
        sep = "-" * 70

        if name:
            lines.append(f"{'=' * 70}")
            lines.append(f"  {name}")
            lines.append(f"{'=' * 70}")
        else:
            lines.append(sep)

        inst = self._instrument
        sig = self._signal

        # Foreground model
        fg_name = type(sig.foreground_model).__name__
        lines.append(f"Foreground model:  {fg_name}")
        lines.append(f"Mission:           {inst.mission_duration_years:.1f} yr, "
                      f"f_sky = {inst.f_sky:.2f}")

        # Efficiency factors (from first channel — assumed same for all)
        eff = inst.channels[0].efficiency
        lines.append(f"Efficiency:        yield={eff.detector_yield:.2f}, "
                      f"obs_eff={eff.observing_efficiency:.2f}, "
                      f"data_cuts={eff.data_cut_fraction:.2f}, "
                      f"CR_dead={eff.cosmic_ray_deadtime:.2f}, "
                      f"pol_eff={eff.polarization_efficiency:.2f} "
                      f"(total={eff.total:.3f})")

        # ell-binning and Knox mode count per bin
        lines.append(f"ell range:         {sig.ells[0]:.0f} - {sig.ells[-1]:.0f}")
        lines.append(f"Bandpower bins:    {sig.n_bins}")
        lines.append(f"Cross-spectra:     {len(sig.freq_pairs)} "
                      f"({len(inst.channels)} channels)")

        # ν_b = effective number of independent modes per bin; below ~10
        # the Knox Gaussian likelihood breaks down and Fisher sigma(r) is
        # structurally narrower than a Hamimeche-Lewis or Wishart
        # posterior. For synthetic top-hat / Gaussian binning,
        # ν_b = f_sky × Σ_{ℓ in bin}(2ℓ+1). For measured BPWFs, the
        # coupling-aware analogue is f_sky × Σ_ℓ W_b(ℓ)² (2ℓ+1).
        if sig.has_measured_bpwf:
            ells_arr = np.asarray(sig.ells)
            # First-pair-representative for diagnostics. In per-spectrum
            # mode the BPWFs differ across cross-spectra; bin_centers
            # follows the same first-pair convention so the two
            # diagnostics agree.
            W_arr = np.asarray(sig.bin_matrix_per_spectrum[0])
            nu_b = inst.f_sky * (W_arr ** 2 * (2.0 * ells_arr + 1.0)
                                 ).sum(axis=1)
            mode_label = ("Knox modes/bin (BPWF, per-spec, first-pair):"
                          if sig.is_per_spectrum_bpwf
                          else "Knox modes/bin (BPWF):")
        else:
            nu_b = np.array([
                inst.f_sky * (hi - lo + 1) * (lo + hi + 1)
                for lo, hi in sig.bin_edges
            ])
            mode_label = "Knox modes/bin:   "
        nu_b_min = float(nu_b.min())
        nu_b_med = float(np.median(nu_b))

        def _fmt_nu(x: float) -> str:
            # Synthetic top-hats at f_sky~0.7 give ν_b ~ 10²-10⁴; measured
            # BPWFs at small f_sky (BK-style ~0.01) can give ν_b << 1 for
            # bins with low row-sums. Switch to scientific below 1 so the
            # number stays readable instead of collapsing to "0.0".
            if abs(x) >= 1.0:
                return f"{x:.1f}"
            return f"{x:.2e}"

        lines.append(f"{mode_label} min={_fmt_nu(nu_b_min)}, "
                     f"median={_fmt_nu(nu_b_med)}"
                     + (" -- WARNING: low-ell bins have < 10 modes; "
                        "Gaussian-likelihood approximation breaks down, "
                        "Fisher will be narrower than Hamimeche-Lewis / "
                        "Wishart posteriors"
                        if nu_b_min < 10 else ""))

        # Channel table
        lines.append("")
        lines.append("  Band     N_det    NET_det    FWHM    Map depth   "
                      "knee_ell  alpha")
        lines.append("  [GHz]             [uK√s]    [']     [uK-']      "
                      "                ")
        for ch in inst.channels:
            w_inv = white_noise_power(ch, inst.mission_duration_years,
                                       inst.f_sky)
            depth = math.sqrt(float(w_inv)) / float(ARCMIN_TO_RAD)
            lines.append(
                f"  {ch.nu_ghz:6.1f}  {ch.n_detectors:6d}  "
                f"{ch.net_per_detector:8.1f}  {ch.beam_fwhm_arcmin:6.1f}  "
                f"{depth:9.2f}   {ch.knee_ell:6.1f}  {ch.alpha_knee:.1f}")
        n_det_total = sum(ch.n_detectors for ch in inst.channels)
        lines.append(f"  {'Total':>6s}  {n_det_total:6d}")

        # Fiducial parameters
        lines.append("")
        lines.append("Fiducial parameters:")
        for name_p in self._all_names:
            val = self._fiducial[name_p]
            status = ""
            if name_p in self._fixed:
                status = "  [FIXED]"
            elif name_p in self._priors:
                status = f"  [prior σ={self._priors[name_p]:.4g}]"
            else:
                status = "  [free, no prior]"
            lines.append(f"  {name_p:20s} = {val:12.4g}{status}")

        # Results
        if self._fisher_matrix is not None:
            lines.append("")
            lines.append("Results:")
            lines.append(f"  Free parameters:  {self.n_free}")

            # Fisher condition number: cond(F) > ~1e14 indicates
            # near-degenerate parameter directions; the eigh solver
            # clips non-positive eigenvalues silently, so the sigmas
            # below may be dominated by numerical regularization rather
            # than data + priors.
            try:
                cond_F = float(jnp.linalg.cond(self._fisher_matrix))
                cond_line = f"  cond(F):          {cond_F:.2e}"
                if cond_F > 1e14:
                    cond_line += ("  -- WARNING: near-degenerate "
                                  "parameters; eigh clipping may "
                                  "dominate the reported sigmas")
                lines.append(cond_line)
            except Exception:
                pass

            for p in self._free_names:
                try:
                    s = self.sigma(p)
                    sc = self.sigma_conditional(p)
                    lines.append(f"  σ({p:15s}) = {s:.4e}  "
                                  f"(conditional: {sc:.4e})")
                except Exception:
                    lines.append(f"  σ({p:15s}) = [error]")

        lines.append(sep)
        return "\n".join(lines)
