#!/usr/bin/env python
"""Monte Carlo ground truth for h_k vs ecliptic colatitude.

Samples (phi_y, phi_p, phi_s) uniformly on T^3, computes boresight position
and crossing angle psi in the ecliptic frame, bins by ecliptic colatitude,
averages exp(-ikψ) per bin. Compares to (a) Falcons.jl output and (b) my
1-D closed-form integral, to localize where the discrepancy lives.
"""
from __future__ import annotations

import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path

NSIDE = 128
ALPHA_F = np.radians(45.0)
BETA_F = np.radians(50.0)

DATADIR = Path(__file__).parent.parent.parent / "data" / "falcons_validation"


def boresight_and_psi(phi_p: np.ndarray, phi_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (theta_b, phi_b, psi) in ecliptic frame for given (phi_p, phi_s).

    phi_y is fixed at 0 (year-averaging is trivial by SO(2) sym, won't change theta_b or psi).
    """
    cos_a, sin_a = np.cos(ALPHA_F), np.sin(ALPHA_F)
    cos_b, sin_b = np.cos(BETA_F), np.sin(BETA_F)

    # Boresight in spin frame
    bx_s = sin_b * np.cos(phi_s)
    by_s = sin_b * np.sin(phi_s)
    bz_s = cos_b * np.ones_like(phi_s)

    # R_y(alpha) into spin-axis-tilted frame
    bx_a = bx_s * cos_a + bz_s * sin_a
    by_a = by_s
    bz_a = -bx_s * sin_a + bz_s * cos_a

    # R_z(phi_p) into antisun frame
    cp, sp = np.cos(phi_p), np.sin(phi_p)
    bx_aa = bx_a * cp - by_a * sp
    by_aa = bx_a * sp + by_a * cp
    bz_aa = bz_a

    # R_y(pi/2) into ecliptic frame: (x, y, z) -> (z, y, -x)
    bx_e = bz_aa
    by_e = by_aa
    bz_e = -bx_aa

    theta_b = np.arccos(np.clip(bz_e, -1, 1))
    phi_b = np.arctan2(by_e, bx_e)

    # Scan velocity v = db/dphi_s (spin direction). Same chain.
    dbx_s = -sin_b * np.sin(phi_s)
    dby_s = sin_b * np.cos(phi_s)
    dbz_s = np.zeros_like(phi_s)

    dbx_a = dbx_s * cos_a + dbz_s * sin_a
    dby_a = dby_s
    dbz_a = -dbx_s * sin_a + dbz_s * cos_a

    dbx_aa = dbx_a * cp - dby_a * sp
    dby_aa = dbx_a * sp + dby_a * cp
    dbz_aa = dbz_a

    vx_e = dbz_aa
    vy_e = dby_aa
    vz_e = -dbx_aa

    # Local north (toward decreasing theta) and east at boresight
    # ê_north = (-cos θ cos φ, -cos θ sin φ, sin θ)
    cos_tb = np.cos(theta_b)
    sin_tb = np.sin(theta_b)
    cos_pb = np.cos(phi_b)
    sin_pb = np.sin(phi_b)

    en_x = -cos_tb * cos_pb
    en_y = -cos_tb * sin_pb
    en_z = sin_tb

    ee_x = -sin_pb
    ee_y = cos_pb
    ee_z = np.zeros_like(phi_b)

    scan_n = vx_e * en_x + vy_e * en_y + vz_e * en_z
    scan_e = vx_e * ee_x + vy_e * ee_y + vz_e * ee_z

    psi = np.arctan2(scan_e, scan_n)

    return theta_b, phi_b, psi


def main():
    rng = np.random.default_rng(0)
    N = 5_000_000  # plenty for 90 colatitude bins
    phi_p = rng.uniform(0, 2 * np.pi, N)
    phi_s = rng.uniform(0, 2 * np.pi, N)

    print(f"Sampling N={N:,} (phi_p, phi_s) on T^2...")
    theta_b, phi_b, psi = boresight_and_psi(phi_p, phi_s)

    # Bin by theta_b
    n_bins = 90
    edges = np.linspace(0, np.pi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(theta_b, edges) - 1
    valid = (idx >= 0) & (idx < n_bins)

    h_mc = {}
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in (1, 2, 4):
        e_neg_ikpsi = np.exp(-1j * k * psi)
        sums = np.bincount(idx[valid], weights=e_neg_ikpsi[valid].real, minlength=n_bins) + \
               1j * np.bincount(idx[valid], weights=e_neg_ikpsi[valid].imag, minlength=n_bins)
        if k == 1:
            counts[:] = np.bincount(idx[valid], minlength=n_bins)
        h_mc[k] = sums / np.maximum(counts, 1)

    # Closed form (current 1-D integral)
    def closed_form(te, k, n=4001):
        ts_lo = max(np.pi/2 - ALPHA_F, abs(te - BETA_F))
        ts_hi = min(np.pi/2 + ALPHA_F, te + BETA_F)
        if ts_hi <= ts_lo:
            return np.nan + 0j
        eps = 1e-7
        ts = np.linspace(ts_lo + eps, ts_hi - eps, n)
        Da = np.sin(ALPHA_F)**2 - np.cos(ts)**2
        Db = (np.sin(ts)*np.sin(BETA_F))**2 - (np.cos(te) - np.cos(ts)*np.cos(BETA_F))**2
        valid = (Da > 0) & (Db > 0)
        w = np.zeros_like(ts)
        w[valid] = np.sin(ts[valid]) / np.sqrt(Da[valid] * Db[valid])
        C = np.where(valid, (np.cos(ts) - np.cos(te)*np.cos(BETA_F)) / (np.sin(te)*np.sin(BETA_F)), 0.0)
        A = np.arccos(np.clip(C, -1, 1))
        avg = np.trapezoid(w * np.cos(k * A), ts) / np.trapezoid(w, ts)
        return ((1j) ** k) * avg

    h_cf = {k: np.array([closed_form(t, k) for t in mids]) for k in (1, 2, 4)}

    # Falcons (band-binned per colatitude bin)
    pix_theta, _ = hp.pix2ang(NSIDE, np.arange(12 * NSIDE**2))
    h_falc = {}
    for k in (1, 2, 4):
        with fits.open(DATADIR / f"h{k}_litebird_nside{NSIDE}.fits") as h:
            hk = h[0].data + 1j * h[1].data
        binned = np.array([
            np.mean(hk[(pix_theta >= lo) & (pix_theta < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
        h_falc[k] = binned

    # Print table at a few colatitudes
    targets_deg = [10, 20, 33, 45, 60, 75, 90, 105, 120, 135, 150, 170]
    print(f"\n{'theta':>6}  {'k':>2}  {'MC':>22}  {'Falcons':>22}  {'closed form':>22}  {'|MC-Fal|':>9}  {'|MC-CF|':>8}")
    for tdeg in targets_deg:
        i = np.argmin(np.abs(np.degrees(mids) - tdeg))
        for k in (1, 2, 4):
            mc = h_mc[k][i]
            fa = h_falc[k][i]
            cf = h_cf[k][i]
            print(f"{tdeg:>6}  {k:>2}  {mc.real:+8.4f}{mc.imag:+8.4f}j  {fa.real:+8.4f}{fa.imag:+8.4f}j  {cf.real:+8.4f}{cf.imag:+8.4f}j  {abs(mc-fa):>9.4f}  {abs(mc-cf):>8.4f}")

    # Quantitative summary
    print("\nSummary across all bins (excluding boundary 5° and 175°):")
    bulk = (mids > np.radians(5)) & (mids < np.radians(175))
    for k in (1, 2, 4):
        mc_falc_diff = np.abs(h_mc[k][bulk] - h_falc[k][bulk])
        mc_cf_diff = np.abs(h_mc[k][bulk] - h_cf[k][bulk])
        print(f"  k={k}: max|MC-Falcons|={mc_falc_diff.max():.4f}  RMS={np.sqrt(np.mean(mc_falc_diff**2)):.4f}  ;  max|MC-CF|={np.nanmax(mc_cf_diff):.4f}  RMS={np.sqrt(np.nanmean(mc_cf_diff**2)):.4f}")

    return 0


if __name__ == "__main__":
    main()
