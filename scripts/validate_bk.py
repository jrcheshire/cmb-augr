#!/usr/bin/env python3
"""
BK validation: BICEP/Keck σ(r) time evolution, 2013–2019.

Public-readable analog of Buza 2019 thesis Figure 7.9: σ(r) projection vs
calendar time as Keck Array and BICEP3 data accumulate, with two cuts:

  Top:    Map depths (μK-arcmin) by frequency
  Bottom: σ(r) — red dashed (raw) and gray solid (FG marginalized)

Red curve: raw sensitivity (no FG, A_lens=1 fixed, lensing in Knox variance).
Gray curve: 12+-band analysis (BK + Planck + WMAP) with BK15 priors, A_lens free.
Extends through 2018 including BICEP3 at 95 GHz.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.foregrounds import GaussianForegroundModel
from augr.spectra import CMBSpectra
from augr.signal import SignalModel
from augr.fisher import FisherForecast
from augr.config import FIDUCIAL_BK15

# ---------------------------------------------------------------------------
# Survey parameters
# ---------------------------------------------------------------------------

F_SKY = 400.0 / (4 * np.pi * (180 / np.pi) ** 2)
T_REF_YR = 1.0
T_REF_S = T_REF_YR * 365.25 * 86400.0
OMEGA_SKY_ARCMIN2 = F_SKY * 4 * np.pi * (180.0 * 60.0 / np.pi) ** 2

EFF_UNITY = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)

# Keck Array reference values (from BK15 endpoint)
BK15_DEPTH = {95: 5.2, 150: 2.9, 220: 26.0}
BK15_NEFF  = {95: 2048, 150: 8704, 220: 1024}
BEAM_KA    = {95: 43.0, 150: 30.0, 220: 21.0}

# BICEP3: 95 GHz, 24' beam, ~2500 detectors (8× Keck equivalent)
# From BK18 paper (arXiv:2110.00483): 2.8 μK-arcmin after 3 seasons
B3_DEPTH_3YR = 2.8
B3_BEAM = 24.0

# Planck/WMAP on BK field: full-sky depths × reobservation penalty
REOBS_DEPTH_PENALTY = 2.5

PLANCK_BANDS = {
    30: (195.0, 32.29), 44: (260.0, 27.0), 70: (200.0, 13.25),
    100: (118.0, 9.66), 143: (70.0, 7.27), 217: (105.0, 4.99),
    353: (439.0, 4.94),
}
WMAP_BANDS = {23: (1435.0, 52.8), 33: (1472.0, 39.6)}


def depth_from_neff(freq, n_eff):
    return BK15_DEPTH[freq] * np.sqrt(BK15_NEFF[freq] / n_eff)


def net_from_depth(depth, n_det):
    return depth * np.sqrt(n_det * T_REF_S / (2.0 * OMEGA_SKY_ARCMIN2))


# ---------------------------------------------------------------------------
# BK observation history
# ---------------------------------------------------------------------------

# BICEP2 3yr only as baseline (depth=5.2 μK-arcmin at 150 GHz)
# Keck data added as seasons starting from KA2012
BASELINE = {150: int(BK15_NEFF[150] * (BK15_DEPTH[150] / 5.2)**2)}  # 2707

# Per-receiver-year n_eff at each frequency:
#   150 GHz: 428 (from BK15: 14 Keck-RY + B2(=2707 n_eff) = 8704 total)
#   95 GHz:  512 (from BK15: 4 Keck-RY = 2048 total)
#   220 GHz: 512 (from BK15: 2 Keck-RY = 1024 total)
NEFF_PER_RY = {95: 512, 150: 428, 220: 512}

# Keck Array seasons — n_eff = n_receivers × NEFF_PER_RY[freq]
KECK_SEASONS = {
    2013: {150: 5*428 + 5*428},   # KA2012(5rx) + KA2013(5rx) folded together
    2014: {95: 2*512, 150: 3*428},               # 2×95, 3×150
    2015: {95: 2*512, 150: 1*428, 220: 2*512},   # 2×95, 1×150, 2×220
    2016: {150: 1*428, 220: 4*512},               # 1×150, 4×220
    2017: {220: 4*512},                           # 4×220
    2018: {220: 4*512},                           # 4×220
}

# BICEP3 at 95 GHz: first light 2016
B3_SEASONS = {2016: 1.0, 2017: 1.0, 2018: 1.0}

# ---------------------------------------------------------------------------
# Fiducials and priors
# ---------------------------------------------------------------------------

# Red curve: no FG, A_lens fixed at 1
FIDUCIAL_NO_FG = dict(FIDUCIAL_BK15)
FIDUCIAL_NO_FG["A_dust"] = 1e-6
FIDUCIAL_NO_FG["A_sync"] = 1e-6
FIDUCIAL_NO_FG["epsilon"] = 0.0
FIDUCIAL_NO_FG["A_lens"] = 1.0
FIXED_ALL = list(GaussianForegroundModel().parameter_names) + ["A_lens"]

# Gray curve priors (BK15 paper, arXiv:1810.05216)
GRAY_PRIORS = {
    "beta_dust":  0.11,
    "beta_sync":  0.3,
    "alpha_dust": 0.29,
    "alpha_sync": 0.29,
    "epsilon":    0.58,
}
GRAY_FIXED = ["T_dust", "Delta_dust"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cumul_at(t):
    """Return (keck_neff_dict, bicep3_years) accumulated at time t."""
    year = int(t)
    frac = t - year
    keck_acc = dict(BASELINE)
    b3_years = 0.0
    for y in range(2013, year):
        for freq, n in KECK_SEASONS.get(y, {}).items():
            keck_acc[freq] = keck_acc.get(freq, 0.0) + n
        b3_years += B3_SEASONS.get(y, 0.0)
    if frac > 0:
        for freq, n in KECK_SEASONS.get(year, {}).items():
            keck_acc[freq] = keck_acc.get(freq, 0.0) + n * frac
        b3_years += B3_SEASONS.get(year, 0.0) * frac
    return keck_acc, b3_years


def build_external_channels():
    """Build channels for Planck/WMAP reobserved through BICEP pipeline."""
    channels = []
    for freq, (depth_fs, beam) in {**PLANCK_BANDS, **WMAP_BANDS}.items():
        depth = depth_fs * REOBS_DEPTH_PENALTY
        n_det = 1000
        net = net_from_depth(depth, n_det)
        channels.append(Channel(
            nu_ghz=float(freq), n_detectors=n_det,
            net_per_detector=net, beam_fwhm_arcmin=beam,
            efficiency=EFF_UNITY,
        ))
    return channels


def build_bk_channels(keck_acc, b3_years):
    """Build BK channels from accumulated Keck n_eff and BICEP3 years."""
    channels = []
    # Keck channels
    for freq in sorted(keck_acc.keys()):
        n_tot = keck_acc[freq]
        if n_tot <= 0:
            continue
        depth = depth_from_neff(freq, n_tot)
        net = net_from_depth(depth, n_tot)
        channels.append(Channel(
            nu_ghz=float(freq), n_detectors=n_tot,
            net_per_detector=net, beam_fwhm_arcmin=BEAM_KA[freq],
            efficiency=EFF_UNITY,
        ))
    # BICEP3 at 95 GHz (separate channel — different beam than Keck 95)
    if b3_years > 0:
        b3_depth = B3_DEPTH_3YR * np.sqrt(3.0 / b3_years)
        b3_ndet = 1000.0
        b3_net = net_from_depth(b3_depth, b3_ndet)
        channels.append(Channel(
            nu_ghz=94.0,  # slight offset to avoid freq collision with Keck 95
            n_detectors=b3_ndet,
            net_per_detector=b3_net, beam_fwhm_arcmin=B3_BEAM,
            efficiency=EFF_UNITY,
        ))
    return channels


EXT_CHANNELS = build_external_channels()


def compute_sigmas(bk_channels):
    """Compute raw and gray σ(r) for a given set of BK channels."""
    # Raw: BK only, no FG, A_lens fixed
    inst_bk = Instrument(channels=tuple(bk_channels),
                         mission_duration_years=T_REF_YR, f_sky=F_SKY)
    signal_bk = SignalModel(
        inst_bk, GaussianForegroundModel(), CMBSpectra(),
        ell_min=20, ell_max=330, delta_ell=35, ell_per_bin_below=30,
    )
    f_raw = FisherForecast(signal_bk, inst_bk, FIDUCIAL_NO_FG,
                           priors=GRAY_PRIORS, fixed_params=FIXED_ALL)
    sr_raw = float(f_raw.sigma("r"))

    # Gray: BK + external, full FG marginalization, A_lens free
    all_channels = bk_channels + EXT_CHANNELS
    inst = Instrument(channels=tuple(all_channels),
                      mission_duration_years=T_REF_YR, f_sky=F_SKY)
    signal = SignalModel(
        inst, GaussianForegroundModel(), CMBSpectra(),
        ell_min=20, ell_max=330, delta_ell=35, ell_per_bin_below=30,
    )
    if len(bk_channels) >= 1:
        f_gray = FisherForecast(signal, inst, FIDUCIAL_BK15,
                                priors=GRAY_PRIORS, fixed_params=GRAY_FIXED)
        sr_gray = float(f_gray.sigma("r"))
    else:
        sr_gray = np.nan

    return sr_raw, sr_gray


# ---------------------------------------------------------------------------
# Compute on grid from 2013 to 2019 (end of 2018)
# ---------------------------------------------------------------------------

t_grid = np.linspace(2013.0, 2019.0, 37)

raw_vals   = []
gray_vals  = []
depths_k95 = []
depths_b3  = []
depths_150 = []
depths_220 = []

for t in t_grid:
    keck_acc, b3_years = cumul_at(t)
    bk_ch = build_bk_channels(keck_acc, b3_years)
    sr_raw, sr_gray = compute_sigmas(bk_ch)
    raw_vals.append(sr_raw)
    gray_vals.append(sr_gray if np.isfinite(sr_gray) else np.nan)

    # Track depths for top panel
    n95 = keck_acc.get(95, 0.0)
    depths_k95.append(depth_from_neff(95, n95) if n95 > 0 else np.nan)
    depths_b3.append(B3_DEPTH_3YR * np.sqrt(3.0 / b3_years) if b3_years > 0 else np.nan)
    n150 = keck_acc.get(150, 0.0)
    depths_150.append(depth_from_neff(150, n150) if n150 > 0 else np.nan)
    n220 = keck_acc.get(220, 0.0)
    depths_220.append(depth_from_neff(220, n220) if n220 > 0 else np.nan)

raw_vals   = np.array(raw_vals)
gray_vals  = np.array(gray_vals)
depths_k95 = np.array(depths_k95)
depths_b3  = np.array(depths_b3)
depths_150 = np.array(depths_150)
depths_220 = np.array(depths_220)

# Print table
print(f"{'t':>7}  {'dK95':>6} {'dB3':>6} {'d150':>6} {'d220':>6}  "
      f"{'σ_raw':>8} {'σ_gray':>8}")
print("-" * 65)
for i, t in enumerate(t_grid):
    dk95 = f"{depths_k95[i]:.1f}" if np.isfinite(depths_k95[i]) else "  -  "
    db3  = f"{depths_b3[i]:.1f}" if np.isfinite(depths_b3[i]) else "  -  "
    d150 = f"{depths_150[i]:.1f}" if np.isfinite(depths_150[i]) else "  -  "
    d220 = f"{depths_220[i]:.1f}" if np.isfinite(depths_220[i]) else "  -  "
    sgray = f"{gray_vals[i]:.5f}" if np.isfinite(gray_vals[i]) else "    nan"
    mark = ""
    if abs(t - 2014.0) < 0.01:
        mark = " ← BKP"
    elif abs(t - 2015.0) < 0.01:
        mark = " ← BK14"
    elif abs(t - 2016.0) < 0.01:
        mark = " ← BK15"
    elif abs(t - 2019.0) < 0.01:
        mark = " ← BK18"
    print(f"{t:7.3f}  {dk95:>6} {db3:>6} {d150:>6} {d220:>6}  "
          f"{raw_vals[i]:8.5f} {sgray:>8}{mark}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax_depth, ax_sr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1.2]})

# --- Top panel: map depths ---
m150 = np.isfinite(depths_150)
ax_depth.semilogy(t_grid[m150], depths_150[m150], "o-", color="C0", ms=4, lw=1.5,
                  label="150 GHz (Keck)")
mk95 = np.isfinite(depths_k95)
if mk95.any():
    ax_depth.semilogy(t_grid[mk95], depths_k95[mk95], "s-", color="C1", ms=4, lw=1.5,
                      label="95 GHz (Keck)")
mb3 = np.isfinite(depths_b3)
if mb3.any():
    ax_depth.semilogy(t_grid[mb3], depths_b3[mb3], "D-", color="C3", ms=4, lw=1.5,
                      label="95 GHz (BICEP3)")
m220 = np.isfinite(depths_220)
if m220.any():
    ax_depth.semilogy(t_grid[m220], depths_220[m220], "^-", color="C2", ms=4, lw=1.5,
                      label="220 GHz (Keck)")

ax_depth.set_ylabel("Map depth [μK-arcmin]", fontsize=11)
ax_depth.set_title("BICEP/Keck σ(r) time evolution\n"
                   f"(Knox formula, ext. reobs. penalty {REOBS_DEPTH_PENALTY:.1f}×)",
                   fontsize=10)
ax_depth.legend(fontsize=7.5, loc="lower left", ncol=2)
ax_depth.grid(True, alpha=0.3, which="both")
ax_depth.set_ylim(1, 100)

# --- Bottom panel: σ(r) ---
ax_sr.semilogy(t_grid, raw_vals, color="red", linestyle="--", linewidth=1.8,
               label=r"Raw sensitivity (no FG, $A_{\rm lens}=1$ fixed)")

mask = np.isfinite(gray_vals)
ax_sr.semilogy(t_grid[mask], gray_vals[mask], color="gray", linestyle="-", linewidth=1.8,
               label="With FG marg. (BK + ext, priors)")

# Published-analog targets as "x" markers (cf. Buza 2019 thesis Fig. 7.9).
# Raw sensitivity targets:
pub_raw_t = [2014.0, 2015.0, 2016.0]
pub_raw_sr = [0.007, 0.005, 0.004]
ax_sr.plot(pub_raw_t, pub_raw_sr, 'rx', ms=10, mew=2.0, zorder=5,
           label="Published raw targets")

# Gray curve published values:
pub_gray_t = [2015.0, 2016.0]
pub_gray_sr = [0.024, 0.020]
ax_sr.plot(pub_gray_t, pub_gray_sr, 'kx', ms=10, mew=2.0, zorder=5,
           label="Published gray targets")

# BK18 published result (full analysis, arXiv:2110.00483)
ax_sr.plot([2019.0], [0.009], 'kx', ms=10, mew=2.0, zorder=5)
ax_sr.annotate("BK18 published\nσ(r)=0.009",
               xy=(2019.0, 0.009), xytext=(2017.8, 0.025),
               fontsize=7.5, ha="center",
               arrowprops=dict(arrowstyle="->", color="k", lw=0.8))

# Epoch markers
for t_ep in [2014.0, 2015.0, 2016.0, 2019.0]:
    ax_sr.axvline(t_ep, color="k", linestyle=":", linewidth=0.5, alpha=0.3)

ax_sr.set_xlabel("Time (year)", fontsize=11)
ax_sr.set_ylabel(r"$\sigma(r)$", fontsize=12)
ax_sr.set_xlim(2013.0, 2019.5)
ax_sr.set_ylim(5e-4, 5e-2)
ax_sr.set_xticks([2013, 2014, 2015, 2016, 2017, 2018, 2019])
ax_sr.set_xticklabels(["2013\n(base)", "2014\n(BKP)", "2015\n(BK14)",
                        "2016\n(BK15)", "2017", "2018", "2019\n(BK18)"],
                       fontsize=8)
ax_sr.legend(fontsize=7.5, loc="lower left")
ax_sr.grid(True, alpha=0.3, which="both")

plt.tight_layout()
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
outpath = PLOT_DIR / "validate_bk.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved: {outpath}")
