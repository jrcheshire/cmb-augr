"""Stop 5 visualization: deck-schedule design landscape.

Bars of |h_k|^2 vs k for four representative deck distributions,
showing how schedule choice maps onto the survival pattern of spin
moments at the South Pole boresight:

* single deck (no diversity): |h_k| = 1 for all k.
* two decks at 0, 180 deg: nulls all odd k.
* BICEP 4-deck schedule (68, 113, 248, 293): nulls {1, 3, 4, 5, 7, ...}
  but leaves |h_2| = 1/sqrt(2). h_2 (-> differential gain) survives.
* BICEP Array 8-deck schedule (multiples of 45 deg + 23 deg offset):
  nulls k = 1..7, leaves |h_8| = 1.

Run:
    conda run -n augr python 03_validation.py
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from augr.crosslinks_southpole import BA_DECK_ANGLES_8, h_k_boresight

OUT = Path(__file__).parent / "03_validation.png"

K_VALUES = np.arange(1, 11)
SCHEDULES = {
    "1 deck (no diversity)":         (jnp.array([0.0]),                                "C0"),
    "2 decks {0, 180}":              (jnp.array([0.0, 180.0]),                         "C1"),
    "BICEP 4-deck {68,113,248,293}": (jnp.array([68.0, 113.0, 248.0, 293.0]),          "C2"),
    "BICEP Array 8-deck (45° step)": (jnp.asarray(BA_DECK_ANGLES_8),                   "C3"),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bar_w = 0.2
    offsets = np.linspace(-1.5 * bar_w, 1.5 * bar_w, len(SCHEDULES))

    for off, (label, (decks, color)) in zip(offsets, SCHEDULES.items()):
        amps = np.array([
            abs(complex(h_k_boresight(decks, chi_deg=0.0, k=int(k)))) ** 2
            for k in K_VALUES
        ])
        ax.bar(K_VALUES + off, amps, bar_w, label=label, color=color,
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("spin order $k$")
    ax.set_ylabel("$|h_k|^2$ at boresight")
    ax.set_title(
        "Deck-schedule design landscape — South Pole boresight\n"
        "BA's 8-deck cycle null-suppresses every spin moment Wallis 2017 "
        "cares about ($k = 1, 2, 4$)",
        fontsize=11,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    # Annotate the Wallis 2017 contamination orders.
    for k_target in (1, 2, 4):
        ax.axvspan(k_target - 0.45, k_target + 0.45, alpha=0.07, color="red",
                   zorder=0)
    ax.text(2.5, 1.0, "shaded: Wallis 2017 contamination orders\n"
                       "(differential pointing $h_1$, gain $h_2$, ellipticity $h_4$)",
            fontsize=8.5, ha="center", va="top", color="darkred",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none",
                      alpha=0.85))

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
