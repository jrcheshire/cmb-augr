"""plot_active_subspace.py -- poster/paper figure for an active_subspace_hl_eig run.

Reads the ``.json`` summary written by ``scripts/active_subspace_hl_eig.py`` and
renders a three-panel figure of the **Gaussian-EIG** design active subspace:

  (a) eigenvalue spectrum of ``C = E_xi[grad u grad u^T]`` with bootstrap
      16-84 percentile bars,
  (b) per-knob activity scores (how much each physical design knob
      participates in the active subspace),
  (c) the Gaussian-EIG profile along the leading active direction,
      relative to the fiducial design.

The HL-EIG scan in the same json is deliberately NOT plotted: at the
production prior the HL estimator's r-grid cannot resolve the posterior and
the reported values are suspected resolution-saturated (see issue #64).

Run:  pixi run python scripts/plot_active_subspace.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#0072B2"  # Okabe-Ito blue, CVD-safe on white
INK = "#1a1a2e"
MUTED = "#5a5a6e"
GRID = "#d8d8e0"

KNOB_LABELS = {
    "alloc@35": "FP area @ 35 GHz",
    "alloc@80+115": "FP area @ 80/115 GHz",
    "alloc@160+225": "FP area @ 160/225 GHz",
    "alloc@315+440": "FP area @ 315/440 GHz",
    "alloc@615": "FP area @ 615 GHz",
    "aperture": "aperture",
    "f_number": "f-number",
    "mission_years": "mission length",
}


def _score_label(v: float) -> str:
    if v >= 0.01:
        return f"{v:.2f}"
    if v >= 0.001:
        return f"{v:.3f}"
    exp = int(np.floor(np.log10(v)))
    return f"{v / 10**exp:.0f}×10$^{{{exp}}}$"


def make_figure(summary: dict, *, with_scan: bool = False) -> plt.Figure:
    """Two panels (spectrum + activity scores) by default -- the poster form.

    ``with_scan=True`` appends the Gaussian-EIG profile along the leading
    direction as a third panel: a validation view for the paper, hard to read
    cold on a poster.
    """
    eig = np.asarray(summary["eigenvalues"])
    p16 = np.asarray(summary["bootstrap"]["eig_p16"])
    p84 = np.asarray(summary["bootstrap"]["eig_p84"])
    scores = np.asarray(summary["activity_scores"])
    labels = [KNOB_LABELS.get(k, k) for k in summary["knob_labels"]]
    ts = np.asarray(summary["scan_t"])
    gauss = np.asarray(summary["scan_gauss_eig"])

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
    if with_scan:
        fig, (ax_a, ax_b, ax_c) = plt.subplots(
            1, 3, figsize=(11.5, 3.6), layout="constrained", width_ratios=[1.0, 1.25, 1.0]
        )
    else:
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(8.2, 3.6), layout="constrained", width_ratios=[1.0, 1.25]
        )
        ax_c = None

    # (a) eigenvalue spectrum with bootstrap 16-84 bars
    idx = np.arange(1, eig.size + 1)
    ax_a.errorbar(
        idx,
        eig,
        yerr=[eig - p16, p84 - eig],
        fmt="o-",
        color=ACCENT,
        ecolor=MUTED,
        elinewidth=1.2,
        capsize=3,
        lw=1.6,
        ms=6,
    )
    ax_a.set_yscale("log")
    ax_a.set_xticks(idx)
    ax_a.set_xlabel("active-subspace direction")
    ax_a.set_ylabel(r"eigenvalue of $C = \mathbb{E}[\nabla u\,\nabla u^{\top}]$")
    ax_a.grid(True, which="major", axis="y", color=GRID, lw=0.6)
    ax_a.set_axisbelow(True)

    # (b) activity scores, sorted, horizontal
    order = np.argsort(scores)
    ax_b.barh(
        np.arange(scores.size),
        scores[order],
        color=ACCENT,
        height=0.62,
        edgecolor="none",
    )
    ax_b.set_yticks(np.arange(scores.size))
    ax_b.set_yticklabels([labels[i] for i in order])
    ax_b.set_xlabel("activity score")
    ax_b.set_xlim(0, 0.72)
    for y, i in enumerate(order):
        ax_b.text(
            scores[i] + 0.012,
            y,
            _score_label(scores[i]),
            va="center",
            ha="left",
            fontsize=9.5,
            color=MUTED,
        )
    ax_b.grid(True, axis="x", color=GRID, lw=0.6)
    ax_b.set_axisbelow(True)

    # (c) Gaussian-EIG profile along the leading direction (paper/validation view)
    if ax_c is None:
        return fig
    ax_c.axvline(0.0, color=GRID, lw=1.0)
    ax_c.plot(
        ts,
        gauss - gauss[len(ts) // 2],
        "o-",
        color=ACCENT,
        lw=1.8,
        ms=6,
    )
    ax_c.set_xlabel("displacement along\ndirection 1 (dex)")
    ax_c.set_ylabel(r"$\Delta$ EIG from fiducial (nats)")
    ax_c.grid(True, axis="y", color=GRID, lw=0.6)
    ax_c.set_axisbelow(True)

    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).parent.parent
        / "data/eig_runs/subspace_d1s1_n128_947183.json",
        help="summary json written by active_subspace_hl_eig.py",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "plots",
        help="output directory (png + pdf)",
    )
    p.add_argument(
        "--with-scan",
        action="store_true",
        help="append the direction-1 Gaussian-EIG profile as a third panel "
        "(paper/validation view; output gets a _scan suffix)",
    )
    args = p.parse_args()

    summary = json.loads(args.json.read_text())
    fig = make_figure(summary, with_scan=args.with_scan)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / (args.json.stem + ("_scan" if args.with_scan else ""))
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
