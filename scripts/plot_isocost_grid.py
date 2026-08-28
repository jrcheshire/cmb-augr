"""plot_isocost_grid.py -- poster/paper iso-cost figure from an iso-cost grid run.

Reads the ``.json`` written by ``scripts/active_subspace_hl_eig.py --grid-aperture
--grid-fpd`` and renders the substitution-question figure: the Gaussian-EIG
surface in the (aperture, total detector count) plane with iso-cost contours
overlaid, and the budget-constrained optimum marked at a few budgets.

Styling matches scripts/plot_active_subspace.py (the companion poster figure).

Run:  pixi run python scripts/plot_isocost_grid.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#1a1a2e"
MUTED = "#5a5a6e"

BUDGET_MARKS_MUSD = (1000.0, 1500.0, 2000.0)


def make_figure(summary: dict) -> plt.Figure:
    ap = np.asarray(summary["aperture_m"])
    eig = np.asarray(summary["eig"])  # (n_ap, n_fp)
    cost = np.asarray(summary["cost_musd"])
    nd = np.asarray(summary["n_det_total"])
    n_det_axis = nd[0, :]  # detector count is set by fp_diameter alone

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
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.0), layout="constrained")

    cf = ax.contourf(ap, n_det_axis, eig.T, levels=14, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("expected information gain on $r$ (nats)")

    levels = [750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0]
    widths = [2.2 if v in (1000.0, 2000.0) else 1.0 for v in levels]
    cs = ax.contour(ap, n_det_axis, cost.T, levels=levels, colors="w", linewidths=widths)
    ax.clabel(
        cs,
        fmt=lambda v: f"${v / 1000:g}B" if v >= 1000 else f"${v:.0f}M",
        fontsize=9.5,
    )

    # Budget-constrained optimum: best grid point with cost <= B.
    for b in BUDGET_MARKS_MUSD:
        masked = np.where(cost <= b, eig, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        ax.plot(
            ap[i],
            n_det_axis[j],
            marker="*",
            ms=16,
            mec="w",
            mew=1.2,
            color="#D55E00",
            clip_on=False,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ap[::2])
    ax.set_xticklabels([f"{a:.1f}" for a in ap[::2]])
    ax.set_yticks(n_det_axis[::2])
    ax.set_yticklabels([f"{n / 1000:.0f}k" for n in n_det_axis[::2]])
    ax.minorticks_off()
    ax.set_xlabel("aperture (m)")
    ax.set_ylabel("total detector count")
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).parent.parent
        / "data/eig_runs/isocost_d1s1_n128_949652.json",
        help="grid json written by active_subspace_hl_eig.py --grid-aperture/--grid-fpd",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "plots",
        help="output directory (png + pdf)",
    )
    args = p.parse_args()

    summary = json.loads(args.json.read_text())
    fig = make_figure(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / args.json.stem
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
