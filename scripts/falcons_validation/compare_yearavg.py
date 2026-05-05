#!/usr/bin/env python
"""Year-averaged closed form for h_k(theta_ecl), compared against Falcons.jl.

Geometry: ecliptic frame, anti-sun rotates around ecliptic, spin axis precesses
around anti-sun at opening alpha_F, boresight at angle beta_F from spin axis.

Year-averaging makes the map azimuthally symmetric around the ecliptic pole
(verified empirically). The remaining 1-D dependence on theta_ecl is:

    h_k(theta_ecl) = <cos[k (A(theta_S, theta_ecl) - pi/2)]>
                     averaged over theta_S with weight w(theta_S, theta_ecl)

where:
    cos A    = (cos theta_S - cos theta_ecl cos beta_F) / (sin theta_ecl sin beta_F)
    w        = sin theta_S / sqrt(D_alpha * D_beta)
    D_alpha  = sin^2 alpha_F - cos^2 theta_S    (precession time-density Jacobian)
    D_beta   = (sin theta_S sin beta_F)^2 - (cos theta_ecl - cos theta_S cos beta_F)^2
                                              (spin-circle to colatitude theta_ecl Jacobian)

Support: D_alpha > 0  AND  D_beta > 0  AND  spherical-triangle inequalities.

The single-precession-cycle envelope (agent's original) is the alpha_F -> 0
limit, where theta_S collapses to pi/2 and the integral is trivial.
"""
from __future__ import annotations

import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
from scipy.integrate import quad
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSIDE = 128
ALPHA_F = np.radians(45.0)  # precession opening
BETA_F  = np.radians(50.0)  # spin opening
DATADIR = Path(__file__).parent.parent.parent / "data" / "falcons_validation"


def yearavg_h_k(theta_ecl: float, k: int) -> complex:
    """Year-averaged h_k(theta_ecl) via 1-D adaptive quadrature over theta_S.

    Closed form:
        h_k = (i)^k * <cos(kA)>_w
    where A is the spherical-triangle vertex angle at the boresight,
        cos A = (cos theta_S - cos theta_ecl cos beta_F) / (sin theta_ecl sin beta_F)
    and the weight is the joint precession-x-spin Jacobian:
        w(theta_S) = sin theta_S / sqrt(D_alpha * D_beta)
        D_alpha    = sin^2(alpha_F) - cos^2(theta_S)        [precession Jacobian]
        D_beta     = (sin theta_S sin beta_F)^2
                     - (cos theta_ecl - cos theta_S cos beta_F)^2
                                                            [spin-circle to colat Jacobian]

    Both D_alpha and D_beta vanish linearly at the support endpoints, giving
    1/sqrt(boundary distance) integrable singularities. scipy.quad handles
    these correctly; trapezoid does not (by ~10% in worst cases) -- verified
    against direct (phi_p, phi_s) Monte Carlo on T^2.
    """
    # Allowed support for theta_S (intersection of precession band and
    # spherical-triangle inequality).
    theta_S_lo_prec = 0.5 * np.pi - ALPHA_F
    theta_S_hi_prec = 0.5 * np.pi + ALPHA_F
    theta_S_lo_tri = max(0.0, abs(theta_ecl - BETA_F))
    theta_S_hi_tri = min(np.pi, theta_ecl + BETA_F)
    lo = max(theta_S_lo_prec, theta_S_lo_tri)
    hi = min(theta_S_hi_prec, theta_S_hi_tri)
    if hi <= lo + 1e-9:
        return np.nan + 0j

    cos_te = np.cos(theta_ecl)
    sin_te = np.sin(theta_ecl)
    cos_b = np.cos(BETA_F)
    sin_b = np.sin(BETA_F)
    sin_a = np.sin(ALPHA_F)

    def w(ts):
        cos_ts = np.cos(ts)
        sin_ts = np.sin(ts)
        Da = sin_a * sin_a - cos_ts * cos_ts
        Db = (sin_ts * sin_b) ** 2 - (cos_te - cos_ts * cos_b) ** 2
        if Da <= 0 or Db <= 0:
            return 0.0
        return sin_ts / np.sqrt(Da * Db)

    def integrand(ts):
        cos_ts = np.cos(ts)
        Da = sin_a * sin_a - cos_ts * cos_ts
        Db = (np.sin(ts) * sin_b) ** 2 - (cos_te - cos_ts * cos_b) ** 2
        if Da <= 0 or Db <= 0:
            return 0.0
        C = (cos_ts - cos_te * cos_b) / (sin_te * sin_b)
        C = max(-1.0, min(1.0, C))
        return np.sin(ts) / np.sqrt(Da * Db) * np.cos(k * np.arccos(C))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # IntegrationWarning is OK; result is converged
        norm, _ = quad(w, lo, hi, limit=400)
        num, _ = quad(integrand, lo, hi, limit=400)
    if norm == 0:
        return np.nan + 0j
    avg = num / norm
    # Falcons psi convention is the conjugate of our derivation; (i)^k handles both.
    return ((1j) ** k) * avg


def main():
    hits = fits.getdata(str(DATADIR / f"hitmap_litebird_nside{NSIDE}.fits"))
    theta_ecl_pix, _ = hp.pix2ang(NSIDE, np.arange(len(hits)))

    # Bin Falcons over 90 colatitude bins for clean shape comparison
    edges = np.linspace(0, np.pi, 91)
    mids = 0.5 * (edges[:-1] + edges[1:])

    # Predict (complex) on the bin centers
    print(f"Evaluating year-averaged integral at {len(mids)} colatitudes...")
    pred = {k: np.array([yearavg_h_k(t, k) for t in mids]) for k in (1, 2, 4)}

    # Load Falcons and bin (Re and Im separately)
    falcons_re = {}
    falcons_im = {}
    for k in (1, 2, 4):
        with fits.open(DATADIR / f"h{k}_litebird_nside{NSIDE}.fits") as h:
            hk = h[0].data + 1j * h[1].data
        falcons_re[k] = np.array([
            np.mean(hk.real[(theta_ecl_pix >= lo) & (theta_ecl_pix < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
        falcons_im[k] = np.array([
            np.mean(hk.imag[(theta_ecl_pix >= lo) & (theta_ecl_pix < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ])

    # Stats: complex-valued comparison.
    print(f"\n{'k':>3} {'N(finite)':>10} {'max|Δh|':>10} {'RMS|Δh|':>10}")
    pass_threshold = 0.02
    pass_overall = True
    for k in (1, 2, 4):
        valid = np.isfinite(pred[k])
        falcons_complex = falcons_re[k] + 1j * falcons_im[k]
        diff = np.abs(falcons_complex[valid] - pred[k][valid])
        rms = np.sqrt(np.mean(diff ** 2))
        mxe = np.max(diff)
        flag = "PASS" if mxe < pass_threshold else "FAIL"
        pass_overall &= (mxe < pass_threshold)
        print(f"{k:>3} {valid.sum():>10} {mxe:>10.4f} {rms:>10.4f}  [{flag}]")
    print(f"\nOverall: {'PASS' if pass_overall else 'FAIL'} (threshold {pass_threshold} on |Δh|)")

    # Plot: 1D collapse for each k — show Re and Im separately
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, k in enumerate((1, 2, 4)):
        ax_re = axes[0, col]
        ax_re.plot(np.degrees(mids), falcons_re[k], "o-", ms=3, color="C0",
                   label="Falcons Re(h_k)")
        ax_re.plot(np.degrees(mids), pred[k].real, "k--", lw=1.4,
                   label="closed-form Re")
        ax_re.axhline(0, color="k", lw=0.4)
        ax_re.set_xlabel(r"ecliptic colatitude $\theta_\mathrm{ecl}$ [deg]")
        ax_re.set_ylabel(rf"Re($h_{k}$)")
        ax_re.set_title(f"k={k}: Re part")
        ax_re.legend(fontsize=9)
        ax_re.grid(alpha=0.3)
        ax_re.set_xlim(0, 180)

        ax_im = axes[1, col]
        ax_im.plot(np.degrees(mids), falcons_im[k], "o-", ms=3, color="C3",
                   label="Falcons Im(h_k)")
        ax_im.plot(np.degrees(mids), pred[k].imag, "k--", lw=1.4,
                   label="closed-form Im")
        ax_im.axhline(0, color="k", lw=0.4)
        ax_im.set_xlabel(r"ecliptic colatitude $\theta_\mathrm{ecl}$ [deg]")
        ax_im.set_ylabel(rf"Im($h_{k}$)")
        ax_im.set_title(f"k={k}: Im part")
        ax_im.legend(fontsize=9)
        ax_im.grid(alpha=0.3)
        ax_im.set_xlim(0, 180)
    fig.suptitle(
        f"Year-averaged 1D closed form vs Falcons (LiteBIRD: $\\alpha_F$={np.degrees(ALPHA_F):.0f}°, $\\beta_F$={np.degrees(BETA_F):.0f}°, 1 yr)",
        fontsize=11,
    )
    outpath = DATADIR / "compare_yearavg_1D.png"
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"\nWrote {outpath}")
    return 0 if pass_overall else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
