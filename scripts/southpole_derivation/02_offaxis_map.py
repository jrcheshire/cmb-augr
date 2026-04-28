"""Stop 4 visualization: |h_k|^2 and arg(h_k) vs declination at lat=-90.

The reveal: at lat = -90 deg, the off-axis correction in chi2alpha is
a uniform shift in alpha across all decks (chi2alpha's az calculation
does not depend on thetaref). Therefore for a single detector,

    |h_k|^2  is set by the deck schedule alone   (uniform in dec, RA).
    arg(h_k) carries the focal-plane / dec geometry (varies with dec).

This script plots both as a function of declination across the BICEP
CMB field, for the boresight and a representative off-axis detector,
making the amplitude flatness vs phase variation visible. The 2-D map
is RA-invariant, so a 1-D plot vs dec captures the full structure.

Run:
    conda run -n augr python 02_offaxis_map.py
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from augr.crosslinks_southpole import h_k_offaxis

OUT = Path(__file__).parent / "02_offaxis_map.png"

DEC_GRID = jnp.linspace(-73.0, -38.0, 71)  # 0.5 deg
DECKS = jnp.array([68.0, 113.0, 248.0, 293.0])
CHI = 11.0
THETA_FP = 45.0


def _hk(r_deg: float, k: int) -> np.ndarray:
    return np.array(h_k_offaxis(
        DEC_GRID, DECKS, r_deg=r_deg, theta_fp_deg=THETA_FP, chi_deg=CHI, k=k,
    ))


def main() -> None:
    h_bs = _hk(0.0, 2)        # r = 0
    h_off1 = _hk(1.0, 2)      # r = 1 deg
    h_off2 = _hk(2.0, 2)      # r = 2 deg
    h_off4 = _hk(4.0, 2)      # r = 4 deg (deliberately exaggerated)

    dec = np.array(DEC_GRID)
    series = [
        ("$r = 0$ (boresight)", h_bs, "C0"),
        ("$r = 1°$",            h_off1, "C1"),
        ("$r = 2°$",            h_off2, "C2"),
        ("$r = 4°$",            h_off4, "C3"),
    ]

    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for label, h, color in series:
        ax_a.plot(dec, np.abs(h) ** 2, color=color, label=label, lw=1.6)
        ax_p.plot(dec, np.rad2deg(np.angle(h)), color=color, label=label, lw=1.6)

    ax_a.set_xlabel("Dec (deg)")
    ax_a.set_ylabel("$|h_2|^2$")
    ax_a.set_title("Amplitude — uniform across Dec for any single detector\n"
                   "(determined by deck schedule alone)", fontsize=10.5)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="lower right", fontsize=9, frameon=False)

    ax_p.set_xlabel("Dec (deg)")
    ax_p.set_ylabel("$\\arg(h_2)$  (deg)")
    ax_p.set_title("Phase — varies with Dec for off-axis detectors\n"
                   "(boresight phase is constant by chi2alpha geometry)", fontsize=10.5)
    ax_p.grid(alpha=0.3)
    ax_p.legend(loc="best", fontsize=9, frameon=False)

    fig.suptitle(
        "South Pole single-detector $h_2$ vs declination — "
        "BK 4-deck schedule, $\\chi=11°$, $\\theta_{fp}=45°$  "
        "(map is RA-invariant)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
