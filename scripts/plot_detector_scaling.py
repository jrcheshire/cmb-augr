"""plot_detector_scaling.py -- EIG vs cost along the detector-count axis.

Reads the iso-cost grid json and slices it at the aperture column nearest the
fiducial (1.5 m): a clean 1-D "what detectors buy" curve -- Gaussian EIG
against mission cost as the total detector count scales, everything else
fixed. Every point shares the same held-out CRN ensemble, so the curve is
pure design response (unlike cross-ensemble value scatters, which are
noise-dominated; measured R^2 = 0.15 on the 96-design sample).

Run:  pixi run python scripts/plot_detector_scaling.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#0072B2"
INK = "#1a1a2e"
MUTED = "#5a5a6e"
GRID = "#d8d8e0"

APERTURE_FID_M = 1.5


def make_figure(summary: dict) -> plt.Figure:
    ap = np.asarray(summary["aperture_m"])
    eig = np.asarray(summary["eig"])
    cost = np.asarray(summary["cost_musd"])
    nd = np.asarray(summary["n_det_total"])
    i = int(np.argmin(np.abs(ap - APERTURE_FID_M)))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.6), layout="constrained")

    ax.plot(cost[i], eig[i], "o-", color=ACCENT, lw=1.8, ms=7, zorder=3)
    for j in range(0, nd.shape[1], 2):
        ax.annotate(
            f"{nd[i, j] / 1000:.0f}k",
            (cost[i, j], eig[i, j]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=9.5,
            color=MUTED,
        )

    ax.set_xlabel("mission cost ($M)")
    ax.set_ylabel("expected information gain on $r$ (nats)")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).parent.parent
        / "data/eig_runs/isocost_d1s1_n128_949652.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "plots",
    )
    args = p.parse_args()

    summary = json.loads(args.json.read_text())
    fig = make_figure(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "detector_scaling_949652"
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
