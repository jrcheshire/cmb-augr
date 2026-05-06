"""compare.py - Compare augr.delensing N_0 against the plancklens reference.

Loads the NPZ produced by ``run_plancklens.py``, runs
``augr.delensing.compute_n0_*`` on the *same* nl_tt / nl_ee / nl_bb /
LensingSpectra inputs in both flat-sky and full-sky modes, plots the
ratio, and prints max-relative-error per estimator and per L band so
the tolerances locked into ``tests/test_delensing.py::TestN0AgainstPlancklens``
can be set with eyes on the actual numbers.

Run in the augr pixi env:

    pixi run python scripts/n0_validation/compare.py [--ref REF.npz] [--no-fullsky]

Diagnostic figures go to ``scripts/n0_validation/figures/`` (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ESTIMATORS = ("tt", "te", "ee", "eb", "tb", "mv")

L_BANDS = (
    ("low",  2,    9),
    ("bulk", 10,   2000),
    ("high", 2001, 3000),
)


def load_reference(path: Path) -> dict:
    """Load the plancklens reference NPZ, return as a plain dict."""
    npz = np.load(path, allow_pickle=True)
    out = {k: npz[k] for k in npz.files}
    return out


def compute_augr_n0(ref: dict, fullsky: bool) -> dict:
    """Run augr.delensing.compute_n0_* on the reference inputs."""
    import jax.numpy as jnp

    from augr.delensing import (
        LensingSpectra,
        compute_n0_eb,
        compute_n0_ee,
        compute_n0_mv,
        compute_n0_tb,
        compute_n0_te,
        compute_n0_tt,
    )

    spectra = LensingSpectra(
        ells=jnp.asarray(ref["ells"]),
        cl_tt_unl=jnp.asarray(ref["cl_tt_unl"]),
        cl_ee_unl=jnp.asarray(ref["cl_ee_unl"]),
        cl_bb_unl=jnp.asarray(ref["cl_bb_unl"]),
        cl_te_unl=jnp.asarray(ref["cl_te_unl"]),
        cl_tt_len=jnp.asarray(ref["cl_tt_len"]),
        cl_ee_len=jnp.asarray(ref["cl_ee_len"]),
        cl_bb_len=jnp.asarray(ref["cl_bb_len"]),
        cl_te_len=jnp.asarray(ref["cl_te_len"]),
        cl_pp=jnp.asarray(ref["cl_pp"]),
    )
    nl_tt = jnp.asarray(ref["nl_tt"])
    nl_ee = jnp.asarray(ref["nl_ee"])
    nl_bb = jnp.asarray(ref["nl_bb"])
    Ls = jnp.asarray(ref["Ls"], dtype=float)

    common = dict(l_min=2, l_max=3000, fullsky=fullsky)
    out = {
        "tt": compute_n0_tt(Ls, spectra, nl_tt, **common),
        "te": compute_n0_te(Ls, spectra, nl_tt, nl_ee, **common),
        "ee": compute_n0_ee(Ls, spectra, nl_ee, **common),
        "eb": compute_n0_eb(Ls, spectra, nl_ee, nl_bb, **common),
        "tb": compute_n0_tb(Ls, spectra, nl_tt, nl_bb, **common),
        "mv": compute_n0_mv(Ls, spectra, nl_tt, nl_ee, nl_bb, **common),
    }
    return {k: np.asarray(v) for k, v in out.items()}


def relative_error(augr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|augr - ref| / |ref|, with safe handling at ref=0/inf."""
    finite = np.isfinite(augr) & np.isfinite(ref) & (ref != 0)
    out = np.full_like(augr, np.nan, dtype=float)
    out[finite] = np.abs(augr[finite] - ref[finite]) / np.abs(ref[finite])
    return out


def report_band(rel_err: np.ndarray, Ls: np.ndarray) -> None:
    """Print max rel-err per L band."""
    for label, lo, hi in L_BANDS:
        mask = (Ls >= lo) & (Ls <= hi)
        if not mask.any():
            continue
        sub = rel_err[mask]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            print(f"        {label:5s} L in [{lo}, {hi}]: no finite values")
            continue
        print(f"        {label:5s} L in [{lo:>4}, {hi:>4}]: "
              f"max_rel_err = {sub.max():.4f}  median = {np.median(sub):.4f}  n = {sub.size}")


def plot_ratios(ref: dict, augr_flat: dict, augr_full: dict | None,
                out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    Ls = ref["Ls"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, est in zip(axes.flat, ESTIMATORS, strict=True):
        ratio_flat = augr_flat[est] / ref[f"n0_{est}"]
        ax.semilogx(Ls, ratio_flat, label="augr flat-sky / plancklens",
                    color="C0")
        if augr_full is not None:
            ratio_full = augr_full[est] / ref[f"n0_{est}"]
            ax.semilogx(Ls, ratio_full, label="augr full-sky / plancklens",
                        color="C1", linestyle="--")
        ax.axhline(1.0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_title(f"N_0^{est.upper()}")
        ax.set_xlabel("L")
        ax.set_ylabel("ratio")
        ax.set_ylim(0.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "n0_ratios.png"
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ref",
        default=str(Path(__file__).parent / "n0_reference_litebird.npz"),
        help="plancklens reference NPZ produced by run_plancklens.py.",
    )
    parser.add_argument("--no-fullsky", action="store_true",
                        help="Skip the full-sky augr comparison (slow, ~minutes).")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        print(f"Reference NPZ not found at {ref_path}. "
              "Run run_plancklens.py first.", file=sys.stderr)
        return 1

    print(f"Loading reference: {ref_path}")
    ref = load_reference(ref_path)
    Ls = ref["Ls"]

    print(f"Running augr flat-sky N_0 at {Ls.size} L values ...", flush=True)
    augr_flat = compute_augr_n0(ref, fullsky=False)

    augr_full = None
    if not args.no_fullsky:
        print("Running augr full-sky N_0 (this can take ~10 minutes) ...",
              flush=True)
        augr_full = compute_augr_n0(ref, fullsky=True)

    print("\n=== max relative error per L band ===")
    for est in ESTIMATORS:
        ref_arr = ref[f"n0_{est}"]
        print(f"\n  {est.upper()}:")
        print("    flat-sky:")
        report_band(relative_error(augr_flat[est], ref_arr), Ls)
        if augr_full is not None:
            print("    full-sky:")
            report_band(relative_error(augr_full[est], ref_arr), Ls)

    plot_ratios(ref, augr_flat, augr_full,
                out_dir=Path(__file__).parent / "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
