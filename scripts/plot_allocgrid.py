"""plot_allocgrid.py -- poster/paper figure for the focal-plane allocation plane.

Reads the ``.json`` written by ``scripts/active_subspace_hl_eig.py --grid-knob1
--grid-knob2`` over the two allocation knobs and renders the EIG surface in TRUE
area-fraction coordinates (the softmax couples the two logits, so the fraction
mesh is curvilinear), with the equal-area fiducial and the grid optimum marked.

Styling matches the other poster figures (plot_active_subspace.py).

Run:  pixi run python scripts/plot_allocgrid.py --json <run json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#1a1a2e"
MUTED = "#5a5a6e"


def _fraction_label(knob: str) -> str:
    band = knob.split("@")[1].replace("+", "/") if "@" in knob else knob
    return f"focal-plane area fraction @ {band} GHz"


def make_figure(summary: dict) -> plt.Figure:
    eig = np.asarray(summary["eig"])  # (n1, n2)
    fracs = np.asarray(summary["area_fractions"])  # (n1, n2, n_groups)
    labels = summary["knob_labels"]
    k1, k2 = summary["knob1"], summary["knob2"]
    # An alloc knob's group index: free-group g's logit is knob position among alloc
    # labels; area_fractions is indexed by group, with the gauge-fixed ref group first.
    free_groups = summary["free_groups"]
    g1 = free_groups[labels.index(k1)]
    g2 = free_groups[labels.index(k2)]
    f1 = fracs[..., g1]  # curvilinear coordinate meshes
    f2 = fracs[..., g2]

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
    fig, ax = plt.subplots(figsize=(6.8, 5.2), layout="constrained")

    cf = ax.contourf(f1, f2, eig, levels=14, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("expected information gain on $r$ (nats)")

    # Iso-cost overlay, drawn only when the plane has meaningful cost variation
    # (the mid-band pair is cost-flat to ~±6%; the monitor pair varies ~±15%).
    cost = np.asarray(summary["cost_musd"])
    rel = (cost.max() - cost.min()) / cost.mean()
    if rel > 0.10:
        lo = np.ceil(cost.min() / 100) * 100
        hi = np.floor(cost.max() / 100) * 100
        levels = np.arange(lo, hi + 1, 100.0)
        cs = ax.contour(f1, f2, cost, levels=levels, colors="w", linewidths=1.0)
        ax.clabel(cs, fmt=lambda v: f"${v / 1000:g}B" if v >= 1000 else f"${v:.0f}M",
                  fontsize=9)

    # Equal-area fiducial and the sampled optimum.
    n_groups = fracs.shape[-1]
    ax.plot(
        1.0 / n_groups, 1.0 / n_groups, marker="o", ms=9, mfc="none",
        mec="w", mew=1.8, ls="none",
    )
    i, j = np.unravel_index(np.argmax(eig), eig.shape)
    ax.plot(
        f1[i, j], f2[i, j], marker="*", ms=16, mec="w", mew=1.2,
        color="#D55E00", ls="none", clip_on=False, zorder=5,
    )

    ax.set_xlabel(_fraction_label(k1))
    ax.set_ylabel(_fraction_label(k2))
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True, help="z-pair grid json")
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
