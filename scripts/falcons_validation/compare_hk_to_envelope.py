#!/usr/bin/env python
"""Compare Falcons.jl h_k maps to the augr closed-form envelope prediction.

Agent's derivation (single-precession-cycle envelope, fast-spin-rate asymptotic):
    cos(A) = (cos beta - cos(theta_ecl) cos(alpha)) / (sin(theta_ecl) sin(alpha))
    h_k(theta) = cos[k (A - pi/2)]   on the observed band [|beta - alpha|, beta + alpha]

with agent (= Wallis / augr l2_hit_map) naming:
    alpha_agent = boresight-to-spin opening   = Falcons-beta
    beta_agent  = anti-sun-to-spin opening    = Falcons-alpha

Falcons convention (verified against Falcons.jl source: scanfields.jl, line 48):
    h(n, m, psi, phi) = exp(-i * (n*psi + m*phi))   (Takase / e^{-ik psi})

The envelope is purely real; year-averaging makes the two scan-circle
crossings at a given colatitude complex conjugates that sum to a cosine.
We therefore expect:
    Re(h_k_Falcons) ≈ cos[k(A - pi/2)]
    Im(h_k_Falcons) ≈ 0

Pass criterion (frozen pre-run): max-abs(|h_k_Falcons| - |h_k_pred|) < 0.02
across the observed sky for k ∈ {1,2,4}, ignoring pixels within 1° of the
band edges (where the envelope discontinuity lives).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

NSIDE = 128

# Falcons LiteBIRD config (must match validate_hk.jl)
ALPHA_FALCONS_DEG = 45.0  # precession opening (anti-sun to spin axis)
BETA_FALCONS_DEG = 50.0   # spin opening (spin axis to boresight)

# Agent / augr naming swap
ALPHA_AGENT_RAD = np.radians(BETA_FALCONS_DEG)
BETA_AGENT_RAD = np.radians(ALPHA_FALCONS_DEG)

# Observed band [|beta - alpha|, beta + alpha] in agent-naming
THETA_LO = abs(BETA_AGENT_RAD - ALPHA_AGENT_RAD)
THETA_HI = BETA_AGENT_RAD + ALPHA_AGENT_RAD

# Edge buffer for pass-criterion mask (1°)
EDGE_BUFFER_RAD = np.radians(1.0)

# Pass threshold (frozen)
PASS_THRESHOLD = 0.02

DATADIR = Path(__file__).parent.parent.parent / "data" / "falcons_validation"
PLOTDIR = DATADIR  # same dir; gitignored


def load_hk(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        return hdul[0].data + 1j * hdul[1].data


def envelope_h_k(theta_ecl: np.ndarray, k: int) -> np.ndarray:
    """Closed-form envelope prediction.

    Returns NaN outside the observed band [|beta - alpha|, beta + alpha].
    """
    cos_a = (np.cos(BETA_AGENT_RAD) - np.cos(theta_ecl) * np.cos(ALPHA_AGENT_RAD)) / (
        np.sin(theta_ecl) * np.sin(ALPHA_AGENT_RAD)
    )
    A = np.arccos(np.clip(cos_a, -1.0, 1.0))
    h = np.cos(k * (A - 0.5 * np.pi))
    out_of_band = (theta_ecl < THETA_LO) | (theta_ecl > THETA_HI)
    h = np.where(out_of_band, np.nan, h)
    return h


def report_stats(label: str, falcons: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """Return + print summary stats for a single k."""
    diff_abs = np.abs(np.abs(falcons[mask]) - np.abs(pred[mask]))
    diff_signed_real = (falcons[mask].real - pred[mask].real)  # pred is real
    imag_residual = falcons[mask].imag

    stats = {
        "label": label,
        "max_abs_err_amplitude": float(np.nanmax(diff_abs)),
        "rms_abs_err_amplitude": float(np.sqrt(np.nanmean(diff_abs**2))),
        "max_signed_real_err": float(np.nanmax(np.abs(diff_signed_real))),
        "rms_signed_real_err": float(np.sqrt(np.nanmean(diff_signed_real**2))),
        "max_abs_imag_residual": float(np.nanmax(np.abs(imag_residual))),
        "rms_imag_residual": float(np.sqrt(np.nanmean(imag_residual**2))),
        "n_compared": int(np.sum(np.isfinite(diff_abs))),
    }

    print(f"\n=== {label} ===")
    print(f"  pixels compared:           {stats['n_compared']}")
    print(f"  max  ||h|−|h_pred||:       {stats['max_abs_err_amplitude']:.4f}")
    print(f"  RMS  ||h|−|h_pred||:       {stats['rms_abs_err_amplitude']:.4f}")
    print(f"  max  |Re(h) − h_pred|:     {stats['max_signed_real_err']:.4f}")
    print(f"  RMS  |Re(h) − h_pred|:     {stats['rms_signed_real_err']:.4f}")
    print(f"  max  |Im(h)|:              {stats['max_abs_imag_residual']:.4f}")
    print(f"  RMS  |Im(h)|:              {stats['rms_imag_residual']:.4f}")
    return stats


def plot_panels(theta_ecl: np.ndarray, falcons: np.ndarray, pred: np.ndarray, mask: np.ndarray,
                k: int, outpath: Path) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # --- top row: 1D collapse over the FULL theta range (not just the formula band)
    ax_1d = fig.add_subplot(gs[0, :])
    nbins = 90
    edges = np.linspace(0, np.pi, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        observed = (np.zeros_like(falcons.real, dtype=bool) | (np.abs(falcons) >= 0))  # always true; np.abs gives nan-safe
        observed_full = np.isfinite(falcons.real)
        falcons_real_binmean = np.array([
            np.nanmean(falcons.real[observed_full & (theta_ecl >= lo) & (theta_ecl < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
        falcons_abs_binmean = np.array([
            np.nanmean(np.abs(falcons)[observed_full & (theta_ecl >= lo) & (theta_ecl < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
        pred_binmean = np.array([
            np.nanmean(pred[(theta_ecl >= lo) & (theta_ecl < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ])

    ax_1d.plot(np.degrees(mids), falcons_real_binmean, "o-", ms=3, color="C0",
               label="Falcons Re(h_k), binned over $\\varphi$", lw=1.5)
    ax_1d.plot(np.degrees(mids), falcons_abs_binmean, "s-", ms=3, color="C2",
               label="Falcons |h_k|, binned over $\\varphi$", lw=1.0, alpha=0.6)
    ax_1d.plot(np.degrees(mids), pred_binmean, "k--", lw=1.4,
               label=r"closed-form envelope $\cos[k(A-\pi/2)]$ (NaN outside band)")
    ax_1d.axvspan(0, np.degrees(THETA_LO), color="gray", alpha=0.12, label="outside formula band")
    ax_1d.axvspan(np.degrees(THETA_HI), 180, color="gray", alpha=0.12)
    ax_1d.axhline(0, color="k", lw=0.5)
    ax_1d.set_xlabel(r"ecliptic colatitude $\theta_\mathrm{ecl}$ [deg]")
    ax_1d.set_ylabel(r"$h_k$")
    ax_1d.set_xlim(0, 180)
    ax_1d.set_title(f"k={k}: 1D azimuthal collapse — Falcons (year-averaged) vs single-cycle envelope")
    ax_1d.legend(loc="upper right", fontsize=9)
    ax_1d.grid(alpha=0.3)

    # --- bottom row: full-sky maps with sensible scales
    abs_falcons_max = float(np.nanpercentile(np.abs(falcons.real), 99))
    abs_imag_max = float(np.nanpercentile(np.abs(falcons.imag), 99))

    plt.axes(fig.add_subplot(gs[1, 0]))
    hp.mollview(falcons.real, hold=True, cmap="RdBu_r",
                min=-abs_falcons_max, max=abs_falcons_max,
                title=f"Re(h_{k}) Falcons (year-averaged)", unit="", cbar=True, notext=True)

    pred_for_plot = np.where(np.isfinite(pred), pred, hp.UNSEEN)
    plt.axes(fig.add_subplot(gs[1, 1]))
    hp.mollview(pred_for_plot, hold=True, cmap="RdBu_r",
                min=-abs_falcons_max, max=abs_falcons_max,
                title=f"closed-form prediction (NaN = grey, formula band only)",
                unit="", cbar=True, notext=True)

    # Diff only where prediction is defined; show actual signed error.
    diff = falcons.real - pred
    diff_for_plot = np.where(np.isfinite(diff), diff, hp.UNSEEN)
    diff_max = float(np.nanpercentile(np.abs(diff), 99))
    plt.axes(fig.add_subplot(gs[1, 2]))
    hp.mollview(diff_for_plot, hold=True, cmap="RdBu_r",
                min=-diff_max, max=diff_max,
                title=f"Re(h_falcons) − h_pred (where pred defined)",
                unit="", cbar=True, notext=True)

    fig.suptitle(
        f"h_k validation, LiteBIRD ($\\alpha_F$={ALPHA_FALCONS_DEG}°, $\\beta_F$={BETA_FALCONS_DEG}°, 1 yr, nside={NSIDE})",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"  wrote {outpath}")


def main() -> int:
    hits_path = DATADIR / f"hitmap_litebird_nside{NSIDE}.fits"
    hits = fits.getdata(str(hits_path))
    theta_ecl, _ = hp.pix2ang(NSIDE, np.arange(len(hits)))

    # Pass-criterion mask: observed pixels, away from band edges
    in_band = (theta_ecl >= THETA_LO + EDGE_BUFFER_RAD) & (theta_ecl <= THETA_HI - EDGE_BUFFER_RAD)
    mask = (hits > 0) & in_band
    print(f"Observed pixels:               {(hits > 0).sum()} / {len(hits)}")
    print(f"In observed band (edge-buffered): {mask.sum()}")
    print(f"Pass threshold (frozen):       {PASS_THRESHOLD}")

    all_stats = []
    pass_overall = True
    for k in (1, 2, 4):
        h_falcons = load_hk(DATADIR / f"h{k}_litebird_nside{NSIDE}.fits")
        h_pred = envelope_h_k(theta_ecl, k)
        stats = report_stats(f"k={k}", h_falcons, h_pred, mask)
        passed = stats["max_abs_err_amplitude"] < PASS_THRESHOLD
        stats["passed"] = passed
        pass_overall &= passed
        all_stats.append(stats)

        plot_panels(theta_ecl, h_falcons, h_pred, mask, k, DATADIR / f"compare_h{k}.png")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if pass_overall else 'FAIL'}")
    print("=" * 60)
    for s in all_stats:
        flag = "PASS" if s["passed"] else "FAIL"
        print(f"  {s['label']:6s}  max-amp-err = {s['max_abs_err_amplitude']:.4f}  [{flag}]")

    return 0 if pass_overall else 1


if __name__ == "__main__":
    sys.exit(main())
