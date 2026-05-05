#!/usr/bin/env python
"""Mollweide hit-map visualizations for the LiteBIRD and Planck configs.

Reads the FITS hitmaps produced by validate_hk.jl and validate_hk_planck.jl
and writes a 2-panel comparison PNG to data/falcons_validation/hitmaps_mollview.png.
Both maps are in the ecliptic frame (Falcons coord='E').
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATADIR = Path(__file__).parent.parent.parent / "data" / "falcons_validation"

CONFIGS = [
    {
        "tag": "litebird",
        "title": (
            r"LiteBIRD: $\alpha_F=45°$, $\beta_F=50°$, "
            r"$T_\mathrm{prec}=192\,\mathrm{min}$, $T_\mathrm{spin}=20\,\mathrm{min}$, 1 yr"
        ),
    },
    {
        "tag": "planck",
        "title": (
            r"Planck: $\alpha_F=7.5°$, $\beta_F=85°$, "
            r"$T_\mathrm{prec}=6\,\mathrm{months}$, $T_\mathrm{spin}=1\,\mathrm{min}$, 3 yr"
        ),
    },
]


def make_one_panel(cfg: dict) -> Path:
    """Render one config's Mollweide to its own PNG. Returns path."""
    path = DATADIR / f"hitmap_{cfg['tag']}_nside128.fits"
    hits = fits.getdata(str(path)).astype(float)
    observed = hits > 0

    n_obs = int(observed.sum())
    f_obs = n_obs / len(hits)
    h_min = float(hits[observed].min()) if n_obs > 0 else 0.0
    h_max = float(hits.max())
    h_median = float(np.median(hits[observed])) if n_obs > 0 else 0.0
    h_p99 = float(np.percentile(hits[observed], 99)) if n_obs > 0 else h_max

    display = np.where(observed, hits, hp.UNSEEN)

    sub_title = (
        f"observed {f_obs:.0%} of sky · "
        f"hits/pix: min {h_min:.0f}, median {h_median:.0f}, max {h_max:.0f}"
    )

    fig = plt.figure(figsize=(10, 5.5))
    hp.mollview(
        display,
        fig=fig.number,
        cmap="viridis",
        min=h_min,
        max=h_p99,
        title=f"{cfg['title']}\n{sub_title}",
        unit="hits per pixel",
        cbar=True,
        notext=True,
    )
    hp.graticule(dpar=30, dmer=60, color="white", lw=0.4, alpha=0.5)

    outpath = DATADIR / f"hitmap_mollview_{cfg['tag']}.png"
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> int:
    paths = [make_one_panel(cfg) for cfg in CONFIGS]
    for p in paths:
        print(f"wrote {p}")

    # Combined 2-panel figure by stitching the per-config PNGs.
    from matplotlib.image import imread
    fig, axes = plt.subplots(len(paths), 1, figsize=(10, 5.0 * len(paths)))
    if len(paths) == 1:
        axes = [axes]
    for ax, p in zip(axes, paths):
        ax.imshow(imread(p))
        ax.set_axis_off()
    fig.suptitle(
        "L2 hit-map comparison (ecliptic frame, 1 Hz boresight, Falcons.jl)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    outpath = DATADIR / "hitmaps_mollview.png"
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return 0


if __name__ == "__main__":
    main()
