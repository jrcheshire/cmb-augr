#!/usr/bin/env python3
"""
Validate Fisher σ(r) forecast against PICO (arXiv:1902.10541, 1908.07495).

PICO claims:
  - σ(r) ≈ 2×10⁻⁵ from analytic forecast (optimistic, parametric FG)
  - σ(r) = 1×10⁻⁴ requirement (5σ detection of r=5×10⁻⁴)
  - 73% delensing (baseline), 85% (CBE)
  - f_sky = 0.6 for r analysis
  - 21 bands, 12,996 TES at 0.1K, 5yr from L2

Uses PICO Table 3.2 baseline polarization map depths directly,
bypassing NET→depth conversion to isolate the Fisher formalism.
Channels ≥462 GHz are dropped (pure dust monitors, cond. number ~10^33).
"""

import numpy as np

from cmb_forecast.instrument import (
    Channel, Instrument, ScalarEfficiency,
    white_noise_power, ARCMIN_TO_RAD, SECONDS_PER_YEAR,
)
from cmb_forecast.foregrounds import GaussianForegroundModel
from cmb_forecast.spectra import CMBSpectra
from cmb_forecast.signal import SignalModel
from cmb_forecast.fisher import FisherForecast
from cmb_forecast.config import FIDUCIAL_BK15

# -----------------------------------------------------------------------
# PICO Table 3.2 baseline polarization map depths (μK-arcmin) and beams.
# Source: arXiv:1902.10541, Table 3.2, "Baseline polarization map depth"
# column. Full-sky survey, 5yr, 90% yield, 95% survey efficiency.
#
# NET values are CBE per-bolometer temperature NETs (code applies √2).
# N_bolo includes both polarizations per pixel.
# -----------------------------------------------------------------------

# (freq GHz, baseline pol. map depth [μK-arcmin], FWHM [arcmin])
PICO_BASELINE_DEPTHS = [
    ( 21.0, 23.9, 38.4),
    ( 25.0, 18.4, 32.0),
    ( 30.0, 12.4, 28.3),
    ( 36.0,  7.9, 23.6),
    ( 43.0,  7.9, 22.2),
    ( 52.0,  5.7, 18.4),
    ( 62.0,  5.4, 12.8),
    ( 75.0,  4.2, 10.7),
    ( 90.0,  2.8,  9.5),
    (108.0,  2.3,  7.9),
    (129.0,  2.1,  7.4),
    (155.0,  1.8,  6.2),
    (186.0,  4.0,  4.3),
    (223.0,  4.5,  3.6),
    (268.0,  3.1,  3.2),
    (321.0,  4.2,  2.6),
    (385.0,  4.5,  2.5),
    # 462-799 GHz dropped: pure dust monitors, create cond. number ~10^33
]

PICO_FSKYBB = 0.6
EFF_UNITY = ScalarEfficiency(1, 1, 1, 1, 1)


def make_pico_from_depths(bands=PICO_BASELINE_DEPTHS, f_sky=PICO_FSKYBB):
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


def run_forecast(inst, A_lens=1.0, marginalize_fg=True, label=""):
    """Run Fisher forecast and return σ(r)."""
    fg_model = GaussianForegroundModel()
    cmb = CMBSpectra()

    signal = SignalModel(
        inst, fg_model, cmb,
        ell_min=2, ell_max=300, delta_ell=30, ell_per_bin_below=10,
    )

    fid = dict(FIDUCIAL_BK15)
    fid["A_lens"] = A_lens

    if marginalize_fg:
        fixed = ["T_dust", "Delta_dust"]
        priors = {"beta_dust": 0.11, "beta_sync": 0.3}
    else:
        fixed = list(fg_model.parameter_names) + ["A_lens"]
        priors = {}
        fid["A_dust"] = 1e-6
        fid["A_sync"] = 1e-6
        fid["epsilon"] = 0.0

    ff = FisherForecast(signal, inst, fid, priors=priors, fixed_params=fixed)
    sr = ff.sigma("r")
    if label:
        print(f"  {label}: σ(r) = {sr:.2e}")
    return sr


if __name__ == "__main__":
    print("=" * 70)
    print("PICO FORECAST VALIDATION")
    print("=" * 70)

    inst = make_pico_from_depths()

    # Verify combined depth
    total_w = sum(
        1.0 / float(white_noise_power(ch, 1.0, PICO_FSKYBB))
        for ch in inst.channels
    )
    combined = np.sqrt(1.0 / total_w) / float(ARCMIN_TO_RAD)
    print(f"\nCombined depth (21-385 GHz): {combined:.3f} μK-arcmin")
    print(f"PICO baseline (all 21 bands): 0.87  μK-arcmin")

    # --- Raw sensitivity ---
    print("\n--- Raw sensitivity (no FG, A_lens=1 fixed) ---")
    run_forecast(inst, A_lens=1.0, marginalize_fg=False, label="ell=2-300")

    # --- FG marginalized ---
    print("\n--- FG marginalized ---")
    for A_lens, dl in [(1.0, "no delensing"),
                       (0.27, "73% delensing (baseline)"),
                       (0.15, "85% delensing (CBE)")]:
        run_forecast(inst, A_lens=A_lens, marginalize_fg=True,
                     label=f"A_lens={A_lens} ({dl})")

    # --- Also test ell≥20 only (recombination bump only) ---
    print("\n--- ell≥20 only (no reionization bump) ---")
    fg_model = GaussianForegroundModel()
    cmb = CMBSpectra()
    signal_20 = SignalModel(
        inst, fg_model, cmb,
        ell_min=20, ell_max=300, delta_ell=35, ell_per_bin_below=20,
    )
    for A_lens, dl in [(1.0, "no delens"), (0.27, "73% delens")]:
        fid = dict(FIDUCIAL_BK15)
        fid["A_lens"] = A_lens
        ff = FisherForecast(signal_20, inst, fid,
                            priors={"beta_dust": 0.11, "beta_sync": 0.3},
                            fixed_params=["T_dust", "Delta_dust"])
        print(f"  ell≥20, A_lens={A_lens} ({dl}): σ(r) = {ff.sigma('r'):.2e}")

    print("\n--- PICO targets ---")
    print("  Analytic forecast: σ(r) ≈ 2×10⁻⁵ (optimistic, 73% delens)")
    print("  Requirement:       σ(r) = 1×10⁻⁴ (5σ detection of r=5×10⁻⁴)")
    print("  f_sky = 0.6, ell = 2-300, 73% baseline delensing")
