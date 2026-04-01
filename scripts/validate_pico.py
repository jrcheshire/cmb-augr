#!/usr/bin/env python3
"""
Rigorous validation of augr Fisher forecasts against PICO (arXiv:1902.10541).

Compares σ(r) from augr against PICO's published analytic forecast results,
carefully matching assumptions and documenting every discrepancy.

PICO analytic forecast method (Sec 2.7.2):
  - Code: Errard et al. 2016 (JCAP 03, p.052) [ref 225]
  - f_sky = 0.6 (cleanest 60% of sky)
  - ℓ = 2–300 (BB power spectrum)
  - r = 0 fiducial
  - Strictly white noise, Gaussian parameter likelihoods
  - Parametric ML FG: 8 params per 15°×15° patch
    (6 amplitudes for Q,U × {CMB, dust, sync} + 2 spectral indices)
  - Delensing: 73% baseline, 85% CBE
  - Result: σ(r) = 2×10⁻⁵ (Sec 2.7.3, described as "optimistic")
  - Requirement: σ(r) = 1×10⁻⁴ for 5σ detection of r = 5×10⁻⁴

Usage:
    python scripts/validate_pico.py [--no-plots]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from augr.instrument import (
    Channel, Instrument, ScalarEfficiency,
    white_noise_power, noise_nl, ARCMIN_TO_RAD, SECONDS_PER_YEAR,
)
from augr.foregrounds import GaussianForegroundModel, MomentExpansionModel
from augr.spectra import CMBSpectra
from augr.signal import SignalModel
from augr.fisher import FisherForecast
from augr.config import FIDUCIAL_BK15, FIDUCIAL_MOMENT
from augr.multipatch import MultiPatchFisher
from augr.sky_patches import SkyPatch, SkyModel


# =========================================================================
# Section 1: Reference data from arXiv:1902.10541
# =========================================================================

# Table 3.2 — All 21 PICO bands. Baseline polarization map depths (μK-arcmin)
# and beam FWHM. 5yr full-sky survey, 90% detector yield, 95% survey eff.
PICO_BASELINE_DEPTHS = [
    # (freq_GHz, depth_uKarcmin, fwhm_arcmin)
    ( 21.0,   23.9, 38.4),
    ( 25.0,   18.4, 32.0),
    ( 30.0,   12.4, 28.3),
    ( 36.0,    7.9, 23.6),
    ( 43.0,    7.9, 22.2),
    ( 52.0,    5.7, 18.4),
    ( 62.0,    5.4, 12.8),
    ( 75.0,    4.2, 10.7),
    ( 90.0,    2.8,  9.5),
    (108.0,    2.3,  7.9),
    (129.0,    2.1,  7.4),
    (155.0,    1.8,  6.2),
    (186.0,    4.0,  4.3),
    (223.0,    4.5,  3.6),
    (268.0,    3.1,  3.2),
    (321.0,    4.2,  2.6),
    (385.0,    4.5,  2.5),
    (462.0,    9.1,  2.1),
    (555.0,   45.8,  1.5),
    (666.0,  177.0,  1.3),
    (799.0, 1050.0,  1.1),
]

# Table 1.2 / Table 3.2 CBE column — Current Best Estimate depths.
# CBE has >40% margin over baseline (Sec 3.2.4).
PICO_CBE_DEPTHS = [
    ( 21.0,  16.9, 38.4),
    ( 25.0,  13.0, 32.0),
    ( 30.0,   8.7, 28.3),
    ( 36.0,   5.6, 23.6),
    ( 43.0,   5.6, 22.2),
    ( 52.0,   4.0, 18.4),
    ( 62.0,   3.8, 12.8),
    ( 75.0,   3.0, 10.7),
    ( 90.0,   2.0,  9.5),
    (108.0,   1.6,  7.9),
    (129.0,   1.5,  7.4),
    (155.0,   1.3,  6.2),
    (186.0,   2.8,  4.3),
    (223.0,   3.3,  3.6),
    (268.0,   2.2,  3.2),
    (321.0,   3.0,  2.6),
    (385.0,   3.2,  2.5),
    (462.0,   6.4,  2.1),
    (555.0,  32.5,  1.5),
    (666.0, 126.0,  1.3),
    (799.0, 744.0,  1.1),
]

PICO_FSKY = 0.6          # Sec 2.7.2: "cleanest 60% of the sky"
PICO_SIGMA_R = 2e-5      # Sec 2.7.3: analytic forecast result
PICO_REQUIREMENT = 1e-4  # SO1: 5σ detection of r = 5×10⁻⁴
PICO_COMBINED_DEPTH_BASELINE = 0.87   # Table 1.1, all 21 bands
PICO_COMBINED_DEPTH_CBE = 0.61        # Table 1.1, all 21 bands

EFF_UNITY = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)

# Assumption comparison table (PICO paper section → our approach)
ASSUMPTIONS = [
    ("f_sky",            "0.6 (Sec 2.7.2)",           "0.6",                "YES"),
    ("ℓ range",          "2–300 (Table 1.3 SO1)",     "2–300",              "YES"),
    ("Fiducial r",       "0 (Sec 2.7.2)",             "0",                  "YES"),
    ("Noise model",      "White (Sec 2.7.3)",         "White (knee=0)",     "YES"),
    ("FG approach",      "Parametric ML",             "Parametric Fisher",  "~YES"),
    ("FG per patch",     "8 params/15° patch",        "7 free, global",     "DIFFER"),
    ("Delens baseline",  "73% (Sec 2.2.1)",           "A_lens=0.27",        "YES"),
    ("Delens CBE",       "85% (Sec 2.2.1)",           "A_lens=0.15",        "YES"),
    ("Channels",         "21 bands (21–799 GHz)",     "21 bands (21–799)",  "YES"),
    ("Priors",           "Not specified",             "β_d:0.11, β_s:0.3",  "?"),
    ("Code reference",   "Errard+ 2016 (JCAP)",      "augr (this code)",   "--"),
]

KNOWN_DISCREPANCIES = [
    ("Per-patch vs global FG",
     "PICO fits 8 FG params per 15° patch; we use 7 free global params.\n"
     "  Our approach has fewer total FG DOF → slightly more optimistic σ(r)."),
    ("High-freq channels",
     "462–799 GHz bands are pure dust monitors (depth >9 μK-arcmin).\n"
     "  Included for completeness; negligible impact on σ(r)."),
    ("FG parameter differences",
     "We include α_dust, α_sync (ℓ-slopes) and ε (dust-sync correlation)\n"
     "  that PICO's 8-param model may not have. Partially offsets per-patch effect."),
    ("Combined depth",
     "Our combined depth should match PICO's published 0.87 μK-arcmin (baseline)\n"
     "  and 0.61 μK-arcmin (CBE) from Table 1.1."),
]


# =========================================================================
# Section 2: Instrument builders
# =========================================================================

def make_pico_from_depths(bands=PICO_BASELINE_DEPTHS, f_sky=PICO_FSKY):
    """Build PICO instrument from published map depths.

    Each channel is parameterised as N_det=1 with a synthetic NET chosen so
    that white_noise_power(ch, 1yr, f_sky) reproduces the published depth.
    """
    scale = np.sqrt(float(SECONDS_PER_YEAR) / (4 * np.pi * f_sky)) / np.sqrt(2)
    channels = []
    for nu, depth, fwhm in bands:
        net_det = depth * float(ARCMIN_TO_RAD) * scale
        channels.append(Channel(
            nu_ghz=nu, n_detectors=1, net_per_detector=net_det,
            beam_fwhm_arcmin=fwhm, efficiency=EFF_UNITY,
        ))
    return Instrument(channels=tuple(channels), mission_duration_years=1.0,
                      f_sky=f_sky)


def compute_combined_depth(inst, f_sky):
    """Compute inverse-variance-weighted combined polarization map depth.

    Returns (combined_uKarcmin, per_channel_depths).
    """
    per_channel = []
    total_w = 0.0
    for ch in inst.channels:
        w_inv = float(white_noise_power(ch, inst.mission_duration_years, f_sky))
        depth = np.sqrt(w_inv) / float(ARCMIN_TO_RAD)
        per_channel.append((ch.nu_ghz, depth))
        total_w += 1.0 / w_inv
    combined = np.sqrt(1.0 / total_w) / float(ARCMIN_TO_RAD)
    return combined, per_channel


# =========================================================================
# Section 3: Test configurations
# =========================================================================

@dataclass
class Case:
    name: str
    depth_set: str = "baseline"     # "baseline" or "cbe"
    A_lens: float = 0.27
    marginalize_fg: bool = True
    fg_model: str = "gaussian"      # "gaussian" or "moment"
    ell_min: int = 2
    ell_max: int = 300
    delta_ell: int = 30
    ell_per_bin_below: int = 10
    target: float | None = None     # expected σ(r), None = no target
    target_max: float | None = None # must be below this
    tolerance: float = 1.5          # pass if within this factor of target
    primary: bool = False
    description: str = ""


CASES = [
    # --- Raw sensitivity (no FG marginalization) ---
    Case("raw_no_delens", A_lens=1.0, marginalize_fg=False,
         description="Noise-only floor, no delensing"),
    Case("raw_73pct", A_lens=0.27, marginalize_fg=False,
         description="Noise-only, 73% delensing"),
    Case("raw_85pct_cbe", depth_set="cbe", A_lens=0.15, marginalize_fg=False,
         description="Noise-only CBE, 85% delensing"),

    # --- FG-marginalized, Gaussian model (main comparison) ---
    Case("gauss_no_delens", A_lens=1.0,
         description="Gaussian FG, no delensing"),
    Case("gauss_73pct", A_lens=0.27,
         target=PICO_SIGMA_R, tolerance=1.5, primary=True,
         description="PRIMARY: Gaussian FG, 73% delens → PICO σ(r)=2e-5"),
    Case("gauss_85pct", A_lens=0.15,
         description="Gaussian FG, 85% delensing"),
    Case("gauss_73pct_cbe", depth_set="cbe", A_lens=0.27,
         description="Gaussian FG, CBE depths, 73% delens"),
    Case("gauss_85pct_cbe", depth_set="cbe", A_lens=0.15,
         description="Gaussian FG, CBE depths, 85% delens"),
    Case("gauss_73pct_noReion", A_lens=0.27, ell_min=20,
         description="Gaussian FG, 73% delens, ℓ≥20 (no reion. bump)"),

    # --- FG-marginalized, Moment Expansion (stress test) ---
    Case("moment_73pct", A_lens=0.27, fg_model="moment",
         description="Moment FG (17 params), 73% delens"),
    Case("moment_85pct_cbe", depth_set="cbe", A_lens=0.15, fg_model="moment",
         description="Moment FG, CBE depths, 85% delens"),

    # --- Requirement check ---
    Case("SO1_requirement", A_lens=0.27,
         target_max=PICO_REQUIREMENT,
         description="Must satisfy σ(r) < 1e-4 (SO1 5σ of r=5e-4)"),
]


@dataclass
class Result:
    case: Case
    sigma_r: float
    combined_depth: float
    passed: bool | None = None  # None = no target to check
    ratio: float | None = None
    notes: str = ""


# =========================================================================
# Section 4: Multi-patch comparison (PICO per-patch FG model)
# =========================================================================

def pico_sky_model(n_patches: int = 6) -> SkyModel:
    """Create a sky model approximating PICO's f_sky=0.6 divided into patches.

    PICO uses ~110 independent 15°×15° patches over 60% of the sky.
    We approximate this with n_patches of equal f_sky, with dust amplitude
    scaling from clean (high galactic latitude) to dusty (lower latitude).
    Uniform noise weighting (no scan strategy weighting for this comparison).

    Dust amplitude scalings are approximate: the cleanest ~10% of sky has
    A_dust ~ 0.3× BK15, while moderate sky at |b|~30° has ~5-10×.
    """
    f_per = PICO_FSKY / n_patches

    if n_patches == 6:
        # 6 patches from cleanest to dustiest, uniform f_sky
        dust_scales = [0.3, 0.7, 1.5, 3.0, 7.0, 15.0]
        sync_scales = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    elif n_patches == 3:
        dust_scales = [0.5, 2.0, 10.0]
        sync_scales = [0.7, 1.2, 2.5]
    else:
        # Linear spacing
        dust_scales = np.linspace(0.3, 15.0, n_patches).tolist()
        sync_scales = np.linspace(0.5, 3.0, n_patches).tolist()

    patches = tuple(
        SkyPatch(
            name=f"patch_{i}",
            f_sky=f_per,
            A_dust_scale=d,
            A_sync_scale=s,
            noise_weight=1.0,
        )
        for i, (d, s) in enumerate(zip(dust_scales, sync_scales))
    )
    return SkyModel(patches=patches,
                    description=f"PICO-like {n_patches}-patch, f_sky={PICO_FSKY}")


def run_multipatch(n_patches: int = 6, A_lens: float = 0.27,
                   fg_model_type: str = "gaussian",
                   bands=None) -> float:
    """Run multi-patch Fisher forecast, return σ(r)."""
    if bands is None:
        bands = PICO_BASELINE_DEPTHS
    inst = make_pico_from_depths(bands, PICO_FSKY)
    sky = pico_sky_model(n_patches)

    if fg_model_type == "moment":
        fg_model = MomentExpansionModel()
        fid = dict(FIDUCIAL_MOMENT)
    else:
        fg_model = GaussianForegroundModel()
        fid = dict(FIDUCIAL_BK15)
    fid["A_lens"] = A_lens

    cmb = CMBSpectra()
    mp = MultiPatchFisher(
        base_instrument=inst,
        foreground_model=fg_model,
        cmb_spectra=cmb,
        sky_model=sky,
        base_fiducial=fid,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust", "Delta_dust"],
        signal_kwargs=dict(ell_min=2, ell_max=300, delta_ell=30,
                           ell_per_bin_below=10),
    )
    return mp.sigma("r")


def _multipatch_worker(args):
    """Worker for parallelized multi-patch runs."""
    n_patches, A_lens, fg_type = args
    return run_multipatch(n_patches, A_lens, fg_type)


# =========================================================================
# Section 5: Runner
# =========================================================================

def run_single(case: Case) -> Result:
    """Run a single validation case and return the result."""
    # Select depth set
    bands = PICO_BASELINE_DEPTHS if case.depth_set == "baseline" else PICO_CBE_DEPTHS
    inst = make_pico_from_depths(bands, PICO_FSKY)
    combined, _ = compute_combined_depth(inst, PICO_FSKY)

    # Select foreground model and fiducial
    if case.fg_model == "moment":
        fg_model = MomentExpansionModel()
        fid = dict(FIDUCIAL_MOMENT)
    else:
        fg_model = GaussianForegroundModel()
        fid = dict(FIDUCIAL_BK15)
    fid["A_lens"] = case.A_lens

    # Set up priors and fixed params
    if case.marginalize_fg:
        fixed = ["T_dust", "Delta_dust"]
        priors = {"beta_dust": 0.11, "beta_sync": 0.3}
    else:
        # Fix all FG params at their fiducial values (do NOT zero them —
        # changing the fiducial changes the covariance and makes the
        # comparison unfair). Also fix A_lens.
        fixed = list(fg_model.parameter_names) + ["A_lens"]
        priors = {}

    # Build signal model
    cmb = CMBSpectra()
    signal = SignalModel(
        inst, fg_model, cmb,
        ell_min=case.ell_min, ell_max=case.ell_max,
        delta_ell=case.delta_ell, ell_per_bin_below=case.ell_per_bin_below,
    )

    # Run Fisher
    ff = FisherForecast(signal, inst, fid, priors=priors, fixed_params=fixed)
    sigma_r = ff.sigma("r")

    # Check against target
    passed = None
    ratio = None
    notes = ""

    if case.target is not None:
        ratio = sigma_r / case.target
        lo = case.target / case.tolerance
        hi = case.target * case.tolerance
        passed = lo <= sigma_r <= hi
        if not passed:
            notes = f"OUTSIDE [{lo:.1e}, {hi:.1e}]"

    if case.target_max is not None:
        passed_max = sigma_r < case.target_max
        if passed is None:
            passed = passed_max
        else:
            passed = passed and passed_max
        if not passed_max:
            notes = f"EXCEEDS max {case.target_max:.1e}"

    return Result(case=case, sigma_r=sigma_r, combined_depth=combined,
                  passed=passed, ratio=ratio, notes=notes)


def run_consistency_checks(results: list[Result]) -> list[tuple[str, bool, str]]:
    """Post-hoc checks on relationships between results."""
    checks = []

    def get(name):
        for r in results:
            if r.case.name == name:
                return r
        return None

    # More delensing → lower σ(r)
    g_no = get("gauss_no_delens")
    g_73 = get("gauss_73pct")
    g_85 = get("gauss_85pct")
    if g_no and g_73 and g_85:
        ok = g_no.sigma_r > g_73.sigma_r > g_85.sigma_r
        checks.append(("More delensing → lower σ(r) [Gaussian]", ok,
                        f"{g_no.sigma_r:.2e} > {g_73.sigma_r:.2e} > {g_85.sigma_r:.2e}"))

    # CBE ≤ baseline (same delensing): better noise must give better σ(r)
    g_73_cbe = get("gauss_73pct_cbe")
    if g_73 and g_73_cbe:
        ok = g_73_cbe.sigma_r <= g_73.sigma_r
        checks.append(("CBE ≤ baseline [73% Gaussian]", ok,
                        f"{g_73_cbe.sigma_r:.2e} ≤ {g_73.sigma_r:.2e}"))

    # Raw sensitivity ≤ FG-marginalized (at same A_lens, same fiducial).
    # Note: the "raw" cases use A_dust~0 and fix A_lens, which changes
    # the covariance — so this is NOT a fair apples-to-apples comparison.
    # We flag it as informational, not a strict pass/fail.
    raw_73 = get("raw_73pct")
    if raw_73 and g_73:
        ok = raw_73.sigma_r <= g_73.sigma_r
        checks.append(("Raw ≤ FG-marg [73%] (diff. fiducial!)", ok,
                        f"{raw_73.sigma_r:.2e} vs {g_73.sigma_r:.2e}"))

    # Moment ≥ Gaussian (more FG params = worse)
    m_73 = get("moment_73pct")
    if g_73 and m_73:
        ok = m_73.sigma_r >= g_73.sigma_r
        checks.append(("Moment ≥ Gaussian [73%]", ok,
                        f"{m_73.sigma_r:.2e} ≥ {g_73.sigma_r:.2e}"))

    # Reionization bump helps
    g_no_reion = get("gauss_73pct_noReion")
    if g_73 and g_no_reion:
        ok = g_73.sigma_r < g_no_reion.sigma_r
        ratio = g_no_reion.sigma_r / g_73.sigma_r
        checks.append(("Reionization bump helps [73%]", ok,
                        f"ℓ≥2: {g_73.sigma_r:.2e}, ℓ≥20: {g_no_reion.sigma_r:.2e} ({ratio:.1f}×)"))

    return checks


# =========================================================================
# Section 5: Visualization
# =========================================================================

def make_plots(results: list[Result], outdir: str):
    """Generate all validation plots."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt  # noqa: F811

    os.makedirs(outdir, exist_ok=True)

    def savefig(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- Plot 1: σ(r) comparison bar chart ----
    plot_sigma_r_comparison(results, savefig)

    # ---- Plot 2: Noise spectrum ----
    plot_noise_spectrum(savefig)

    # ---- Plot 3: Delensing sweep ----
    plot_delensing_sweep(savefig)

    # ---- Plot 4: Reionization bump ----
    plot_reionization_bump(savefig)

    # ---- Plot 5: Assumption comparison table ----
    plot_assumption_table(savefig)


def plot_sigma_r_comparison(results: list[Result], savefig):
    """Bar chart of σ(r) across configurations."""
    import matplotlib.pyplot as plt

    # Group results for the bar chart
    groups = {
        "Raw\n(no FG)": ["raw_no_delens", "raw_73pct", "raw_85pct_cbe"],
        "Gaussian\nFG": ["gauss_no_delens", "gauss_73pct", "gauss_85pct",
                         "gauss_73pct_cbe", "gauss_85pct_cbe"],
        "Moment\nFG": ["moment_73pct", "moment_85pct_cbe"],
    }

    result_map = {r.case.name: r for r in results}

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        0.0: "#2196F3",   # no delens - blue (A_lens=1.0 mapped below)
        1.0: "#2196F3",   # no delens
        0.27: "#4CAF50",  # 73%
        0.15: "#FF9800",  # 85%
    }
    hatches = {"baseline": None, "cbe": "///"}

    x_pos = 0
    xticks = []
    xticklabels = []
    group_starts = []
    group_ends = []

    for case_names in groups.values():
        group_starts.append(x_pos)
        for name in case_names:
            r = result_map.get(name)
            if r is None:
                continue
            c = r.case
            color = colors.get(c.A_lens, "#9E9E9E")
            hatch = hatches.get(c.depth_set)
            ax.bar(x_pos, r.sigma_r, width=0.7, color=color,
                         hatch=hatch, edgecolor="black", linewidth=0.5,
                         alpha=0.85)
            xticks.append(x_pos)
            # Compact label: "0%", "73%", "85% CBE"
            delens_pct = int((1 - c.A_lens) * 100) if c.A_lens < 1.0 else 0
            lbl = f"{delens_pct}%"
            if c.depth_set == "cbe":
                lbl += " CBE"
            xticklabels.append(lbl)
            x_pos += 1
        group_ends.append(x_pos - 1)
        x_pos += 1.5  # wider gap between groups

    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=7, rotation=45, ha="right")

    # Group labels below x-axis
    for i, label in enumerate(groups.keys()):
        mid = (group_starts[i] + group_ends[i]) / 2
        ax.text(mid, -0.18, label.replace("\n", " "), ha="center", va="top",
                fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())

    # Reference lines
    ax.axhline(PICO_SIGMA_R, color="red", linestyle="--", linewidth=1.5,
               label=f"PICO analytic: σ(r)={PICO_SIGMA_R:.0e}")
    ax.axhline(PICO_REQUIREMENT, color="gray", linestyle=":", linewidth=1.5,
               label=f"PICO requirement: σ(r)={PICO_REQUIREMENT:.0e}")

    ax.set_ylabel("σ(r)", fontsize=12)
    ax.set_title("PICO Validation: σ(r) across configurations", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.5, x_pos)

    # Color legend for delensing
    from matplotlib.patches import Patch
    legend2 = [
        Patch(facecolor="#2196F3", label="No delensing"),
        Patch(facecolor="#4CAF50", label="73% delensing"),
        Patch(facecolor="#FF9800", label="85% delensing"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="CBE depths"),
    ]
    ax.legend(handles=legend2 + ax.get_legend_handles_labels()[0][:2],
              fontsize=8, loc="upper left", ncol=2)

    fig.tight_layout()
    savefig(fig, "sigma_r_comparison")


def plot_noise_spectrum(savefig):
    """Per-channel noise N_ℓ curves with PICO depth markers."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm

    inst = make_pico_from_depths(PICO_BASELINE_DEPTHS, PICO_FSKY)
    ells = np.arange(2, 301)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    freqs = [b[0] for b in PICO_BASELINE_DEPTHS]
    norm = mcolors.Normalize(min(freqs), max(freqs))
    cmap = mcm.get_cmap("viridis")

    # Plot N_ℓ (C_ℓ convention, no ℓ(ℓ+1)/2π scaling) so white noise is flat.
    for ch in inst.channels:
        nl = noise_nl(ch, ells, inst.mission_duration_years, PICO_FSKY)
        color = cmap(norm(ch.nu_ghz))
        ax1.semilogy(ells, np.array(nl), color=color, alpha=0.7, linewidth=1.0)
        # Mark white-noise level (flat line for N_ℓ)
        w_inv = float(white_noise_power(ch, inst.mission_duration_years, PICO_FSKY))
        ax1.axhline(w_inv, color=color, alpha=0.3, linewidth=0.5, linestyle=":")

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax1, label="Frequency [GHz]", pad=0.02)
    ax1.set_ylabel(r"$N_\ell^{BB}$ [$\mu$K$^2$ sr]", fontsize=11)
    ax1.set_xlabel(r"Multipole $\ell$", fontsize=11)
    ax1.set_title("PICO per-channel noise spectra (baseline)", fontsize=12)
    ax1.set_xlim(2, 300)

    # Lower panel: combined noise (C_ℓ convention)
    nl_combined = np.zeros(len(ells))
    for ch in inst.channels:
        nl_ch = np.array(noise_nl(ch, ells, inst.mission_duration_years, PICO_FSKY))
        nl_combined += 1.0 / nl_ch
    nl_combined = 1.0 / nl_combined
    ax2.semilogy(ells, nl_combined, "k-", linewidth=1.5)

    # Mark combined depth (white-noise level = depth² in C_ℓ units)
    combined, _ = compute_combined_depth(inst, PICO_FSKY)
    combined_cl = (combined * float(ARCMIN_TO_RAD))**2
    ax2.axhline(combined_cl, color="red", linestyle="--", linewidth=2.0,
                label=f"Combined depth: {combined:.2f} μK-arcmin\n"
                      f"(PICO 21-band: {PICO_COMBINED_DEPTH_BASELINE} μK-arcmin)")
    ax2.set_ylabel(r"Combined $N_\ell^{BB}$ [$\mu$K$^2$ sr]", fontsize=11)
    ax2.set_xlabel(r"Multipole $\ell$", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(2, 300)
    # Set y-limits to show the combined depth line clearly
    ax2.set_ylim(combined_cl * 0.5, np.max(nl_combined) * 2)

    fig.tight_layout()
    savefig(fig, "noise_spectrum")


def _delens_worker(args):
    """Worker for parallelized delensing sweep."""
    A_l, marg_fg, fg_type = args
    if fg_type == "moment":
        fg_model = MomentExpansionModel()
        fid = dict(FIDUCIAL_MOMENT)
    else:
        fg_model = GaussianForegroundModel()
        fid = dict(FIDUCIAL_BK15)
    fid["A_lens"] = float(A_l)
    if marg_fg:
        fixed = ["T_dust", "Delta_dust"]
        priors = {"beta_dust": 0.11, "beta_sync": 0.3}
    else:
        fixed = list(fg_model.parameter_names) + ["A_lens"]
        priors = {}
    inst = make_pico_from_depths(PICO_BASELINE_DEPTHS, PICO_FSKY)
    cmb = CMBSpectra()
    signal = SignalModel(inst, fg_model, cmb,
                         ell_min=2, ell_max=300, delta_ell=30,
                         ell_per_bin_below=10)
    ff = FisherForecast(signal, inst, fid, priors=priors, fixed_params=fixed)
    return ff.sigma("r")


def plot_delensing_sweep(savefig):
    """σ(r) vs delensing fraction for multiple FG models."""
    import matplotlib.pyplot as plt

    delens_fracs = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.73, 0.80, 0.85, 0.90, 0.95])
    A_lens_vals = 1.0 - delens_fracs

    configs = [
        ("No FG", False, "gaussian", "#2196F3", "--"),
        ("Gaussian FG", True, "gaussian", "#4CAF50", "-"),
        ("Moment FG", True, "moment", "#E91E63", "-"),
    ]

    # Build all jobs
    all_jobs = []
    job_map = {}  # (config_idx, A_idx) -> position in all_jobs
    for ci, (label, marg_fg, fg_type, color, ls) in enumerate(configs):
        for ai, A_l in enumerate(A_lens_vals):
            job_map[(ci, ai)] = len(all_jobs)
            all_jobs.append((float(A_l), marg_fg, fg_type))

    print(f"    delens sweep: {len(all_jobs)} jobs on 4 workers...")
    ctx = __import__("multiprocessing").get_context("spawn")
    with ctx.Pool(4) as pool:
        all_sigmas = pool.map(_delens_worker, all_jobs)

    fig, ax = plt.subplots(figsize=(9, 6))

    for ci, (label, marg_fg, fg_type, color, ls) in enumerate(configs):
        sigmas = [all_sigmas[job_map[(ci, ai)]] for ai in range(len(A_lens_vals))]
        ax.semilogy(delens_fracs * 100, sigmas, color=color, linestyle=ls,
                     linewidth=2, label=label, marker="o", markersize=4)

    # Mark PICO values
    ax.axvline(73, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(85, color="gray", linestyle=":", alpha=0.7)
    ax.text(73, ax.get_ylim()[1] * 0.7, "Baseline\n73%", ha="center",
            fontsize=9, color="gray")
    ax.text(85, ax.get_ylim()[1] * 0.7, "CBE\n85%", ha="center",
            fontsize=9, color="gray")

    # PICO published result
    ax.plot(73, PICO_SIGMA_R, "r*", markersize=15, zorder=5,
            label=f"PICO published: σ(r)={PICO_SIGMA_R:.0e}")

    ax.axhline(PICO_REQUIREMENT, color="red", linestyle=":", alpha=0.5,
               label=f"Requirement: {PICO_REQUIREMENT:.0e}")

    ax.set_xlabel("Delensing fraction [%]", fontsize=12)
    ax.set_ylabel("σ(r)", fontsize=12)
    ax.set_title("σ(r) vs delensing: PICO baseline depths", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 97)
    fig.tight_layout()
    savefig(fig, "delensing_sweep")


def plot_reionization_bump(savefig):
    """Impact of including low-ℓ modes (reionization bump).

    Left panel: σ(r) at ℓ_min=2 vs ℓ_min=20 (just two key numbers).
    Right panel: per-bin Fisher information F_b(r) showing where the
    constraining power actually lives — computed from a single run with
    fixed binning, avoiding the ℓ_min-sweep bin-boundary artifacts.
    """
    import matplotlib.pyplot as plt
    from augr.covariance import bandpower_covariance_blocks
    from augr.fisher import flatten_params

    inst = make_pico_from_depths(PICO_BASELINE_DEPTHS, PICO_FSKY)
    cmb = CMBSpectra()
    fg_model = GaussianForegroundModel()
    fid = dict(FIDUCIAL_BK15)
    fid["A_lens"] = 0.27
    fixed = ["T_dust", "Delta_dust"]
    priors = {"beta_dust": 0.11, "beta_sync": 0.3}

    # Full run with ℓ_min=2 (our standard binning)
    signal = SignalModel(inst, fg_model, cmb,
                         ell_min=2, ell_max=300, delta_ell=30,
                         ell_per_bin_below=10)
    ff_full = FisherForecast(signal, inst, fid, priors=priors,
                             fixed_params=fixed)
    sigma_2 = ff_full.sigma("r")

    # Run with ℓ_min=20 (recombination only, uniform Δℓ=30 bins)
    signal_20 = SignalModel(inst, fg_model, cmb,
                            ell_min=20, ell_max=300, delta_ell=30,
                            ell_per_bin_below=20)
    ff_20 = FisherForecast(signal_20, inst, fid, priors=priors,
                           fixed_params=fixed)
    sigma_20 = ff_20.sigma("r")

    ratio = sigma_20 / sigma_2
    print(f"    reion bump: ℓ≥2 σ(r)={sigma_2:.2e}, "
          f"ℓ≥20 σ(r)={sigma_20:.2e} ({ratio:.1f}×)")

    # --- Per-bin Fisher information on r ---
    # Extract per-bin F_b matrices from the ℓ_min=2 run, then compute
    # the marginal Fisher info on r from cumulative sum of F_b.
    all_names = list(signal.parameter_names)
    free_names = [n for n in all_names if n not in fixed]
    free_idx = jnp.array([all_names.index(n) for n in free_names])
    r_idx = free_names.index("r")

    params = flatten_params(fid, all_names)
    cov_blocks = bandpower_covariance_blocks(signal, inst, params)
    J_full = signal.jacobian(params)
    J = J_full[:, free_idx]
    n_spec = len(signal.freq_pairs)
    n_bins = signal.n_bins
    J_blocks = J.reshape(n_spec, n_bins, -1).transpose(1, 0, 2)

    # Cumulative Fisher: add bins from high-ℓ down to low-ℓ, inverting
    # at each step to get marginalized σ(r). This correctly accounts
    # for parameter degeneracies at each cumulation step.
    bin_centers = np.array(signal.bin_centers)
    F_cumul = jnp.zeros((len(free_names), len(free_names)))

    # Add prior contributions
    for name, sigma_prior in priors.items():
        if name in free_names:
            idx = free_names.index(name)
            F_cumul = F_cumul.at[idx, idx].add(1.0 / sigma_prior**2)

    # Accumulate from highest ℓ-bin down
    cumul_sigmas = np.zeros(n_bins)
    for b in range(n_bins - 1, -1, -1):
        cov_b = cov_blocks[b]
        J_b = J_blocks[b]
        s, U = jnp.linalg.eigh(cov_b)
        s_inv = jnp.where(s > 0.0, 1.0 / s, 0.0)
        UtJ = U.T @ J_b
        W = jnp.sqrt(s_inv)[:, None] * UtJ
        F_cumul = F_cumul + W.T @ W
        F_inv = jnp.linalg.inv(F_cumul)
        cumul_sigmas[b] = float(jnp.sqrt(F_inv[r_idx, r_idx]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cumulative σ(r) as bins are added from high-ℓ downward.
    # Zoom to ℓ < 60 where the action is (high-ℓ bins just flatten out).
    mask = np.array(bin_centers) < 60
    ax1.semilogy(np.array(bin_centers)[mask], cumul_sigmas[mask],
                 "k-", linewidth=2, marker="o", markersize=4)
    ax1.axvline(20, color="red", linestyle="--", alpha=0.7,
                label=r"$\ell = 20$")
    ax1.axhline(sigma_2, color="blue", linestyle=":", alpha=0.5,
                label=f"All bins: σ(r) = {sigma_2:.2e}")
    ax1.axhline(sigma_20, color="red", linestyle=":", alpha=0.5,
                label=f"ℓ ≥ 20 only: σ(r) = {sigma_20:.2e}")
    ax1.set_xlabel(r"$\ell_{\rm bin}$ (adding from right to left)", fontsize=12)
    ax1.set_ylabel(r"Cumulative $\sigma(r)$", fontsize=12)
    ax1.set_title(f"Reionization bump: {ratio:.1f}× improvement\n"
                  f"(Gaussian FG, 73% delensing, 21 bands)", fontsize=11)
    ax1.legend(fontsize=9)

    # Right: fractional Fisher information by bin (log scale).
    # F_total(r) = 1/σ²(r) from full run; per-bin fraction from
    # difference in cumulative 1/σ².
    fisher_cumul = 1.0 / cumul_sigmas**2
    fisher_per_bin = np.diff(fisher_cumul[::-1])[::-1]
    fisher_per_bin = np.append(fisher_per_bin, fisher_cumul[-1])
    fisher_frac = fisher_per_bin / fisher_cumul[0] * 100
    # Clamp tiny/negative values for log scale
    fisher_frac = np.maximum(fisher_frac, 1e-3)

    bin_widths = np.diff(np.append(bin_centers, bin_centers[-1] + 30))
    ax2.bar(bin_centers, fisher_frac, width=bin_widths * 0.8,
            color=["#E91E63" if c < 20 else "#2196F3" for c in bin_centers],
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\ell_{\rm bin}$", fontsize=12)
    ax2.set_ylabel("% of total Fisher info on r", fontsize=12)
    ax2.set_title("Per-bin constraining power", fontsize=11)
    # Label the reionization bins
    reion_pct = sum(f for c, f in zip(bin_centers, fisher_frac) if c < 20)
    ax2.text(0.95, 0.95, f"ℓ < 20: {reion_pct:.0f}% of info",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=10, color="#E91E63", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "reionization_bump")


def plot_assumption_table(savefig):
    """Visual summary of assumption comparison for slides."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    col_labels = ["Aspect", "PICO (1902.10541)", "augr", "Match"]
    cell_colors = []
    for row in ASSUMPTIONS:
        match = row[3]
        if match == "YES":
            cell_colors.append(["white", "white", "white", "#C8E6C9"])
        elif match == "DIFFER":
            cell_colors.append(["white", "white", "white", "#FFF9C4"])
        else:
            cell_colors.append(["white", "white", "white", "#E0E0E0"])

    table = ax.table(
        cellText=ASSUMPTIONS,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#BBDEFB"] * 4,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Bold header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")

    ax.set_title("Assumption Comparison: PICO vs augr",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    savefig(fig, "assumption_comparison")


# =========================================================================
# Section 6: Reporting and main
# =========================================================================

def print_header():
    print("=" * 72)
    print("PICO VALIDATION — arXiv:1902.10541")
    print("Analytic forecast code: Errard et al. 2016 (JCAP 03, p.052)")
    print("=" * 72)


def print_assumptions():
    print("\nASSUMPTION COMPARISON")
    print("-" * 72)
    print(f"  {'Aspect':<20s} {'PICO':<26s} {'augr':<22s} {'Match':>5s}")
    print("-" * 72)
    for aspect, pico, augr, match in ASSUMPTIONS:
        print(f"  {aspect:<20s} {pico:<26s} {augr:<22s} {match:>5s}")
    print("-" * 72)


def print_results(results: list[Result]):
    print("\nRESULTS")
    print("-" * 72)
    print(f"  {'Case':<28s} {'σ(r)':>10s} {'Target':>10s} "
          f"{'Ratio':>7s} {'Status':>7s}  Notes")
    print("-" * 72)

    for r in results:
        name = r.case.name
        if r.case.primary:
            name += " [PRIMARY]"

        target_str = "--"
        if r.case.target is not None:
            target_str = f"{r.case.target:.1e}"
        elif r.case.target_max is not None:
            target_str = f"<{r.case.target_max:.1e}"

        ratio_str = "--"
        if r.ratio is not None:
            ratio_str = f"{r.ratio:.2f}"

        status_str = "--"
        if r.passed is True:
            status_str = "PASS"
        elif r.passed is False:
            status_str = "FAIL"

        print(f"  {name:<28s} {r.sigma_r:>10.2e} {target_str:>10s} "
              f"{ratio_str:>7s} {status_str:>7s}  {r.notes}")

    print("-" * 72)


def print_consistency(checks: list[tuple[str, bool, str]]):
    print("\nCONSISTENCY CHECKS")
    print("-" * 72)
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {desc:<45s} {status:>5s}  {detail}")
    print("-" * 72)


def print_discrepancies():
    print("\nKNOWN DISCREPANCIES")
    print("-" * 72)
    for i, (title, explanation) in enumerate(KNOWN_DISCREPANCIES, 1):
        print(f"  {i}. {title}:")
        print(f"     {explanation}")
    print("-" * 72)


class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, path):
        self._stream = stream
        self._file = open(path, "w")
    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def close(self):
        self._file.close()


def main():
    parser = argparse.ArgumentParser(description="PICO validation")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    # Save text output alongside plots
    log_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "plots", "pico_validation"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "validation.log")
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee

    print_header()
    print_assumptions()

    # Verify combined depths
    print("\nDEPTH VERIFICATION")
    print("-" * 72)
    for label, bands, pico_ref in [
        ("Baseline (21 bands)", PICO_BASELINE_DEPTHS, PICO_COMBINED_DEPTH_BASELINE),
        ("CBE (21 bands)",      PICO_CBE_DEPTHS,      PICO_COMBINED_DEPTH_CBE),
    ]:
        inst = make_pico_from_depths(bands, PICO_FSKY)
        combined, _ = compute_combined_depth(inst, PICO_FSKY)
        print(f"  {label}: {combined:.3f} μK-arcmin "
              f"(PICO 21-band ref: {pico_ref} μK-arcmin)")
    print("  Note: small residual difference from rounding in published tables.")
    print("-" * 72)

    # Run all cases (parallelized with spawn context for JAX compatibility)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    N_WORKERS = min(4, len(CASES))
    print(f"\nRUNNING FORECASTS ({len(CASES)} cases on {N_WORKERS} workers)")
    print("-" * 72)
    t0 = time.time()
    with ctx.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, CASES)
    dt = time.time() - t0
    for r in results:
        print(f"  {r.case.name:<28s} σ(r) = {r.sigma_r:.2e}")
    print(f"  Total: {dt:.1f}s")

    # Print results
    print_results(results)

    # Consistency checks
    checks = run_consistency_checks(results)
    print_consistency(checks)

    # Multi-patch comparison: global vs per-patch FG
    print("\nMULTI-PATCH COMPARISON (global vs per-patch FG)")
    print("-" * 72)
    print("  PICO uses ~110 independent 15° patches with per-patch amplitudes.")
    print("  We approximate with 3 and 6 patches at f_sky=0.6.\n")

    import multiprocessing as mp2
    ctx2 = mp2.get_context("spawn")
    mp_jobs = [
        (3, 0.27, "gaussian"),
        (6, 0.27, "gaussian"),
        (3, 0.27, "moment"),
        (6, 0.27, "moment"),
    ]
    t0 = time.time()
    with ctx2.Pool(4) as pool:
        mp_results = pool.map(_multipatch_worker, mp_jobs)
    dt = time.time() - t0

    # Find single-patch results for comparison
    g_73 = next((r for r in results if r.case.name == "gauss_73pct"), None)
    m_73 = next((r for r in results if r.case.name == "moment_73pct"), None)

    print(f"  {'Config':<40s} {'σ(r)':>10s}  {'vs single':>10s}")
    print(f"  {'-'*62}")
    labels = [
        "Gaussian, 3-patch",
        "Gaussian, 6-patch",
        "Moment, 3-patch",
        "Moment, 6-patch",
    ]
    for label, (_, _, fg_type), sr in zip(labels, mp_jobs, mp_results):
        ref = g_73.sigma_r if fg_type == "gaussian" and g_73 else (
              m_73.sigma_r if fg_type == "moment" and m_73 else None)
        ratio_str = f"{sr/ref:.2f}×" if ref else "--"
        print(f"  {label:<40s} {sr:>10.2e}  {ratio_str:>10s}")

    if g_73:
        print(f"  {'Gaussian, single-patch (reference)':<40s} "
              f"{g_73.sigma_r:>10.2e}")
    if m_73:
        print(f"  {'Moment, single-patch (reference)':<40s} "
              f"{m_73.sigma_r:>10.2e}")
    print(f"  ({dt:.1f}s)")
    print("-" * 72)

    # Discrepancies
    print_discrepancies()

    # Plots
    if not args.no_plots:
        plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots",
                                "pico_validation")
        plot_dir = os.path.normpath(plot_dir)
        print(f"\nGENERATING PLOTS → {plot_dir}")
        print("-" * 72)
        make_plots(results, plot_dir)
        print("  Done.")

    # Summary
    n_targeted = sum(1 for r in results if r.passed is not None)
    n_passed = sum(1 for r in results if r.passed is True)
    n_failed = sum(1 for r in results if r.passed is False)
    n_checks = len(checks)
    n_checks_passed = sum(1 for _, p, _ in checks if p)

    print("\n" + "=" * 72)
    print(f"OVERALL: {n_passed}/{n_targeted} targeted cases PASSED, "
          f"{n_failed} FAILED")
    print(f"         {n_checks_passed}/{n_checks} consistency checks PASSED")
    print("=" * 72)

    # Restore stdout and close log
    sys.stdout = tee._stream
    tee.close()
    print(f"\nLog saved to {log_path}")

    if n_failed > 0 or n_checks_passed < n_checks:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
