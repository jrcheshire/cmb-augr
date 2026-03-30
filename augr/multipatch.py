"""
multipatch.py — Multi-patch Fisher forecasting with parameter sharing.

Runs independent single-patch Fisher forecasts on sky regions with different
foreground amplitudes and noise depths, then assembles the combined Fisher
matrix respecting shared (global) vs per-patch parameters.

The combined Fisher matrix has block structure:

    [ F_gg   F_g1   F_g2  ... ]     F_gg = Σ_p F_p[global,global]
    [ F_1g   F_11    0    ... ]     F_gk = F_k[global, per_patch_k]
    [ F_2g    0     F_22  ... ]     F_kk = F_k[per_patch_k, per_patch_k]

This correctly captures the fact that all patches constrain global parameters
(r, spectral indices), while per-patch parameters (foreground amplitudes)
are independently constrained by each patch's own data.
"""

from __future__ import annotations

import math
from itertools import combinations

import jax.numpy as jnp

from augr.instrument import Channel, Instrument
from augr.fisher import FisherForecast
from augr.signal import SignalModel
from augr.sky_patches import SkyPatch, SkyModel


# ---------------------------------------------------------------------------
# Parameter sharing
# ---------------------------------------------------------------------------

# Foreground parameters that are per-patch (amplitude-like)
_AMPLITUDE_PARAMS = {"A_dust", "A_sync", "Delta_dust"}
_MOMENT_AMPLITUDE_PARAMS = {
    "c_sync", "Delta_sync",
    "omega_d_beta", "omega_d_T", "omega_d_betaT",
    "omega_s_beta", "omega_s_c", "omega_s_betac",
}
_DUST_MOMENT_PARAMS = {"omega_d_beta", "omega_d_T", "omega_d_betaT"}
_SYNC_MOMENT_PARAMS = {"omega_s_beta", "omega_s_c", "omega_s_betac"}

def _is_per_patch(name: str) -> bool:
    """Check whether a foreground parameter is per-patch."""
    return name in _AMPLITUDE_PARAMS or name in _MOMENT_AMPLITUDE_PARAMS


# ---------------------------------------------------------------------------
# Per-patch instrument construction
# ---------------------------------------------------------------------------

def instrument_for_patch(base_instrument: Instrument,
                         patch: SkyPatch,
                         total_f_sky: float,
                         ) -> Instrument:
    """Create per-patch Instrument with correct noise and mode counting.

    The noise power per steradian in a patch is:
        w_inv_p = w_inv_uniform / noise_weight_p

    where w_inv_uniform uses the full survey f_sky_total.  To achieve
    this without modifying instrument.py, we scale each channel's NET:

        NET_eff = NET × sqrt(f_sky_total / (f_sky_p × noise_weight_p))

    and set f_sky = f_sky_p for mode counting in the Knox formula.
    """
    scale = math.sqrt(total_f_sky / (patch.f_sky * patch.noise_weight))

    new_channels = tuple(
        Channel(
            nu_ghz=ch.nu_ghz,
            n_detectors=ch.n_detectors,
            net_per_detector=ch.net_per_detector * scale,
            beam_fwhm_arcmin=ch.beam_fwhm_arcmin,
            knee_ell=ch.knee_ell,
            alpha_knee=ch.alpha_knee,
            efficiency=ch.efficiency,
        )
        for ch in base_instrument.channels
    )
    return Instrument(
        channels=new_channels,
        mission_duration_years=base_instrument.mission_duration_years,
        f_sky=patch.f_sky,
    )


# ---------------------------------------------------------------------------
# Per-patch fiducial parameters
# ---------------------------------------------------------------------------

def fiducial_for_patch(base_fiducial: dict[str, float],
                       patch: SkyPatch,
                       ) -> dict[str, float]:
    """Scale amplitude parameters for a sky patch.

    Dust amplitudes (A_dust, dust moment params) scale by A_dust_scale.
    Sync amplitudes (A_sync, sync moment params) scale by A_sync_scale.
    Global parameters (r, A_lens, spectral indices) are unchanged.
    """
    fid = dict(base_fiducial)
    # Dust amplitudes
    if "A_dust" in fid:
        fid["A_dust"] *= patch.A_dust_scale
    if "Delta_dust" in fid:
        fid["Delta_dust"] *= patch.A_dust_scale
    for key in _DUST_MOMENT_PARAMS:
        if key in fid:
            fid[key] *= patch.A_dust_scale
    # Sync amplitudes
    if "A_sync" in fid:
        fid["A_sync"] *= patch.A_sync_scale
    if "Delta_sync" in fid:
        fid["Delta_sync"] *= patch.A_sync_scale
    if "c_sync" in fid:
        fid["c_sync"] *= patch.A_sync_scale
    for key in _SYNC_MOMENT_PARAMS:
        if key in fid:
            fid[key] *= patch.A_sync_scale
    return fid


# ---------------------------------------------------------------------------
# Multi-patch Fisher
# ---------------------------------------------------------------------------

class MultiPatchFisher:
    """Multi-patch Fisher forecast with parameter sharing.

    Runs independent single-patch FisherForecasts, then assembles the
    combined Fisher matrix with shared global parameters and independent
    per-patch amplitude parameters.

    Args:
        base_instrument:  Instrument for the full survey.
        foreground_model: ForegroundModel instance.
        cmb_spectra:      CMBSpectra instance.
        sky_model:        SkyModel defining the patches.
        base_fiducial:    Fiducial parameter dict (for BK-field reference).
        priors:           Prior widths on global parameters.
        fixed_params:     Parameters to fix globally.
        signal_kwargs:    Extra kwargs for SignalModel (ell_bins, etc).
    """

    def __init__(self,
                 base_instrument: Instrument,
                 foreground_model,
                 cmb_spectra,
                 sky_model: SkyModel,
                 base_fiducial: dict[str, float],
                 priors: dict[str, float] | None = None,
                 fixed_params: list[str] | None = None,
                 signal_kwargs: dict | None = None):
        sky_model.validate()
        self._base_instrument = base_instrument
        self._fg_model = foreground_model
        self._cmb = cmb_spectra
        self._sky_model = sky_model
        self._base_fiducial = dict(base_fiducial)
        self._priors = priors or {}
        self._fixed = set(fixed_params or [])
        self._signal_kwargs = signal_kwargs or {}

        # Determine parameter classification
        all_names = (["r", "A_lens"]
                     + foreground_model.parameter_names)
        self._all_names = all_names
        self._global_free = [n for n in all_names
                             if n not in self._fixed and not _is_per_patch(n)]
        self._per_patch_free = [n for n in all_names
                                if n not in self._fixed and _is_per_patch(n)]
        self._n_global = len(self._global_free)
        self._n_per_patch = len(self._per_patch_free)
        self._n_patches = len(sky_model.patches)
        self._n_total = self._n_global + self._n_patches * self._n_per_patch

        # Build combined parameter name list
        self._combined_names: list[str] = list(self._global_free)
        for p_idx, patch in enumerate(sky_model.patches):
            for name in self._per_patch_free:
                self._combined_names.append(f"{name}_{patch.name}")

        # Per-patch Fisher forecasts (built lazily or in compute())
        self._patch_fishers: list[FisherForecast] = []
        self._combined_fisher: jnp.ndarray | None = None
        self._combined_inverse: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        return self._n_patches

    @property
    def n_total_params(self) -> int:
        return self._n_total

    @property
    def combined_parameter_names(self) -> list[str]:
        return list(self._combined_names)

    @property
    def patches(self) -> tuple[SkyPatch, ...]:
        return self._sky_model.patches

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _patch_idx_map(self, p_idx: int,
                       free_names: list[str]) -> list[int]:
        """Map per-patch Fisher indices to combined matrix indices."""
        combined_idx = []
        for name in free_names:
            if name in self._global_free:
                combined_idx.append(self._global_free.index(name))
            elif name in self._per_patch_free:
                local = self._per_patch_free.index(name)
                combined_idx.append(
                    self._n_global + p_idx * self._n_per_patch + local)
            else:
                raise ValueError(
                    f"Parameter '{name}' not in global or per-patch lists")
        return combined_idx

    def _build_patch_fisher(self, patch: SkyPatch) -> FisherForecast:
        """Build and compute a FisherForecast for one patch."""
        total_fsky = self._sky_model.total_f_sky
        inst_p = instrument_for_patch(
            self._base_instrument, patch, total_fsky)
        fid_p = fiducial_for_patch(self._base_fiducial, patch)
        sig_p = SignalModel(inst_p, self._fg_model, self._cmb,
                            **self._signal_kwargs)
        ff_p = FisherForecast(sig_p, inst_p, fid_p,
                              priors=self._priors,
                              fixed_params=list(self._fixed))
        ff_p.compute()
        return ff_p

    def compute(self) -> jnp.ndarray:
        """Compute the combined multi-patch Fisher matrix.

        Returns:
            Combined Fisher matrix of shape (n_total, n_total).
        """
        self._patch_fishers = []
        F_combined = jnp.zeros((self._n_total, self._n_total))

        for p_idx in range(self._n_patches):
            patch = self._sky_model.patches[p_idx]
            ff_p = self._build_patch_fisher(patch)
            self._patch_fishers.append(ff_p)
            F_p = ff_p.fisher_matrix
            free_names_p = ff_p.free_parameter_names

            # Map per-patch free params to combined matrix indices
            combined_idx = self._patch_idx_map(p_idx, free_names_p)

            # Scatter F_p into combined matrix
            for i_local in range(len(combined_idx)):
                for j_local in range(len(combined_idx)):
                    F_combined = F_combined.at[
                        combined_idx[i_local], combined_idx[j_local]
                    ].add(F_p[i_local, j_local])

        self._combined_fisher = F_combined
        self._combined_inverse = None
        return F_combined

    @property
    def fisher_matrix(self) -> jnp.ndarray:
        if self._combined_fisher is None:
            self.compute()
        return self._combined_fisher

    @property
    def inverse(self) -> jnp.ndarray:
        if self._combined_inverse is None:
            self._combined_inverse = jnp.linalg.inv(self.fisher_matrix)
        return self._combined_inverse

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def sigma(self, param: str = "r") -> float:
        """Marginalized 1-sigma constraint on a global parameter."""
        if param not in self._combined_names:
            raise ValueError(f"Parameter '{param}' not in combined names. "
                             f"Available: {self._combined_names}")
        idx = self._combined_names.index(param)
        return float(jnp.sqrt(self.inverse[idx, idx]))

    def sigma_conditional(self, param: str = "r") -> float:
        """Conditional 1-sigma constraint on a global parameter."""
        idx = self._combined_names.index(param)
        return float(1.0 / jnp.sqrt(self.fisher_matrix[idx, idx]))

    # ------------------------------------------------------------------
    # Optimal f_sky analysis
    # ------------------------------------------------------------------

    def sigma_vs_fsky_curve(self) -> list[dict]:
        """Sigma(r) as patches are added from cleanest to dustiest.

        Returns a list of dicts, one per step, with keys:
            f_sky:        cumulative sky fraction
            sigma_r:      marginalized sigma(r) including patches so far
            patches:      list of patch names included
            patch_sigmas: per-patch sigma(r) (single-patch, for reference)
        """
        # Ensure all patches are computed
        if not self._patch_fishers:
            self.compute()

        results = []
        for n_include in range(1, self._n_patches + 1):
            # Include first n_include patches (cleanest first)
            included = list(range(n_include))
            sr = self._sigma_for_subset(included)
            patch_names = [self._sky_model.patches[i].name
                           for i in included]
            f_sky_cum = sum(self._sky_model.patches[i].f_sky
                           for i in included)
            # Per-patch single-patch sigma(r)
            patch_srs = []
            for i in included:
                try:
                    patch_srs.append(self._patch_fishers[i].sigma("r"))
                except Exception:
                    patch_srs.append(float("nan"))
            results.append({
                "f_sky": f_sky_cum,
                "sigma_r": sr,
                "patches": patch_names,
                "patch_sigmas": patch_srs,
            })
        return results

    def optimal_subset(self) -> dict:
        """Find patch combination minimizing sigma(r).

        Tests all 2^N - 1 non-empty subsets.

        Returns dict with:
            best_sigma_r:  minimum sigma(r)
            best_patches:  list of patch names in optimal set
            best_f_sky:    total f_sky of optimal set
            all_subsets:   list of (patch_names, f_sky, sigma_r) for each
        """
        if not self._patch_fishers:
            self.compute()

        all_subsets = []
        best_sr = float("inf")
        best_patches = []
        best_fsky = 0.0

        for size in range(1, self._n_patches + 1):
            for subset in combinations(range(self._n_patches), size):
                included = list(subset)
                sr = self._sigma_for_subset(included)
                names = [self._sky_model.patches[i].name for i in included]
                fsky = sum(self._sky_model.patches[i].f_sky for i in included)
                all_subsets.append((names, fsky, sr))
                if sr < best_sr:
                    best_sr = sr
                    best_patches = names
                    best_fsky = fsky

        return {
            "best_sigma_r": best_sr,
            "best_patches": best_patches,
            "best_f_sky": best_fsky,
            "all_subsets": all_subsets,
        }

    def _sigma_for_subset(self, included: list[int]) -> float:
        """Compute sigma(r) using only the specified patch indices."""
        n_inc = len(included)
        n_sub = self._n_global + n_inc * self._n_per_patch
        F_sub = jnp.zeros((n_sub, n_sub))

        for sub_idx, p_idx in enumerate(included):
            ff_p = self._patch_fishers[p_idx]
            F_p = ff_p.fisher_matrix
            free_names_p = ff_p.free_parameter_names

            # Use sub_idx (not p_idx) for block position in subset matrix
            idx_map = self._patch_idx_map(sub_idx, free_names_p)

            for i_local in range(len(idx_map)):
                for j_local in range(len(idx_map)):
                    F_sub = F_sub.at[
                        idx_map[i_local], idx_map[j_local]
                    ].add(F_p[i_local, j_local])

        try:
            F_inv = jnp.linalg.inv(F_sub)
            r_idx = self._global_free.index("r")
            return float(jnp.sqrt(F_inv[r_idx, r_idx]))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, name: str = "") -> str:
        """Human-readable summary of multi-patch forecast."""
        lines: list[str] = []
        sep = "-" * 70

        if name:
            lines.append(f"{'=' * 70}")
            lines.append(f"  {name}")
            lines.append(f"{'=' * 70}")
        else:
            lines.append(sep)

        lines.append(f"Sky model:  {self._sky_model.description}  "
                      f"({self._n_patches} patches, "
                      f"f_sky_total = {self._sky_model.total_f_sky:.3f})")
        lines.append(f"Global params:    {self._n_global}")
        lines.append(f"Per-patch params: {self._n_per_patch} × "
                      f"{self._n_patches} patches")
        lines.append(f"Total free:       {self._n_total}")

        lines.append("")
        lines.append("  Patch          f_sky   A_dust×  A_sync×  "
                      "noise_wt  σ(r) [single]")
        for i, patch in enumerate(self._sky_model.patches):
            sr_p = "—"
            if self._patch_fishers:
                try:
                    sr_p = f"{self._patch_fishers[i].sigma('r'):.2e}"
                except Exception:
                    sr_p = "error"
            lines.append(
                f"  {patch.name:12s}  {patch.f_sky:.3f}   "
                f"{patch.A_dust_scale:6.1f}   {patch.A_sync_scale:5.1f}   "
                f"{patch.noise_weight:7.3f}   {sr_p}")

        if self._combined_fisher is not None:
            lines.append("")
            lines.append("Combined results:")
            try:
                sr = self.sigma("r")
                lines.append(f"  σ(r) = {sr:.4e}")
            except Exception:
                lines.append("  σ(r) = [error]")

        lines.append(sep)
        return "\n".join(lines)
