"""Stop 1 geometry diagram: celestial sphere from the South Pole observer.

Polar projection: r = 90 deg + dec = altitude. The BICEP CMB field is
shown as a shaded arc. A representative diurnal circle (dec=-55 deg,
elevation=55 deg) is highlighted, with a boresight star and a deck-angle
rotation arc at the boresight.

Run from the augr pixi env:
    pixi run python scripts/southpole_derivation/01_geometry_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "01_pole_sky.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([15, 30, 45, 60, 75])
    ax.set_yticklabels(["alt 15°\nδ −75°", "30°\n−60°", "45°\n−45°",
                        "60°\n−30°", "75°\n−15°"], fontsize=8)
    ax.set_rlabel_position(135)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xticklabels(["RA 0h", "3h", "6h", "9h", "12h", "15h", "18h", "21h"])

    # BICEP CMB field: RA in [-60, +60] deg, Dec in [-73, -38] deg.
    n_arc = 200
    theta_arc = np.linspace(np.deg2rad(-60), np.deg2rad(60), n_arc)
    r_inner = 90 + (-73)  # = 17
    r_outer = 90 + (-38)  # = 52
    ax.fill_between(theta_arc, r_inner, r_outer, alpha=0.30, color="C0",
                    edgecolor="C0", label="BICEP CMB field")

    # Representative diurnal circle: dec=-55 deg -> alt=55 deg -> r=35.
    dec_obs = -55.0
    r_obs = 90.0 + dec_obs
    theta_full = np.linspace(0, 2 * np.pi, 360)
    ax.plot(theta_full, np.full_like(theta_full, r_obs), color="C3", lw=1.6,
            label=f"diurnal circle, δ = {dec_obs:.0f}°, alt = {-dec_obs:.0f}°")

    # Boresight star at one (RA, Dec) inside the field.
    boresight_ra = 30.0  # deg
    ax.plot(np.deg2rad(boresight_ra), r_obs, marker="*", color="C3",
            markersize=18, linestyle="", label="boresight pointing")

    # Deck-angle rotation arc near the boresight.
    arc_r = 6.0
    arc_center_theta = np.deg2rad(boresight_ra)
    arc_center_r = r_obs
    arc_phi = np.linspace(0, 1.5 * np.pi, 80)
    arc_x = arc_center_r * np.cos(arc_center_theta) + arc_r * np.cos(arc_phi)
    arc_y = arc_center_r * np.sin(arc_center_theta) + arc_r * np.sin(arc_phi)
    arc_r_polar = np.sqrt(arc_x ** 2 + arc_y ** 2)
    arc_theta_polar = np.arctan2(arc_y, arc_x)
    ax.plot(arc_theta_polar, arc_r_polar, color="k", lw=1.2)
    # Arrow head at the end of the arc.
    head_idx = -1
    tail_idx = -8
    ax.annotate(
        "",
        xy=(arc_theta_polar[head_idx], arc_r_polar[head_idx]),
        xytext=(arc_theta_polar[tail_idx], arc_r_polar[tail_idx]),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="k"),
    )
    ax.text(arc_center_theta + 0.12, arc_center_r + 9, "deck",
            fontsize=10, color="k")

    ax.set_title(
        "Sky from the South Pole observer\n"
        "polar projection: $r = 90° + \\delta$ = altitude",
        pad=18, fontsize=12,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.02), fontsize=9,
              frameon=False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
