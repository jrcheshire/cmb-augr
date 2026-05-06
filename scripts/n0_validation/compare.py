"""compare.py - Compare augr.delensing N_0 against the plancklens reference.

Loads the NPZ produced by ``run_plancklens.py``, runs
``augr.delensing.compute_n0_*`` on the *same* nl_tt / nl_ee / nl_bb /
LensingSpectra inputs in full-sky mode, plots the ratio, and prints
max-relative-error per L band.

The plancklens reference covers three QE keys:

  - ``ptt``    -> ``n0_tt``      compared to ``augr.compute_n0_tt(fullsky=True)``
  - ``p_p``    -> ``n0_pp``      compared to ``MV(augr.compute_n0_ee, augr.compute_n0_eb, fullsky=True)``
  - ``p``      -> ``n0_mv``      compared to ``augr.compute_n0_mv(fullsky=True)``

**Only TT is an apples-to-apples comparison.** plancklens's ``p_p``
and ``p`` keys include the inter-estimator cross-correlations (joint
GMV combination), while augr's ``compute_n0_mv`` and the manual MV
of EE/EB use the diagonal HO02 Eq. 22
``1 / Sum_alpha (1/N_alpha)``. The diagonal MV is *strictly* larger
than the joint GMV at scales where multiple estimators contribute
significantly (the cross-correlations help). PP and MV ratios are
reported here for diagnostic eyes-on, NOT as a tolerance test.

Flat-sky augr is NOT a meaningful comparison against plancklens
(which is full-sky natively). The flat-sky-vs-full-sky geometric
factor of ``(L+1)^2 / L^2`` at low L plus the C(l) shape interaction
at all L will dominate any flat-sky-vs-full-sky ratio. Flat-sky augr
is validated against an analytic closed form by
``controlled_input_test.py``; this script only validates full-sky.

Run in the augr pixi env:

    pixi run python scripts/n0_validation/compare.py [--ref REF.npz]

Diagnostic figures go to ``scripts/n0_validation/figures/`` (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Estimators that the plancklens reference NPZ exposes (see
# run_plancklens.py): n0_tt, n0_pp, n0_mv.
REF_ESTIMATORS = ("tt", "pp", "mv")

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


def compute_augr_n0(ref: dict) -> dict:
    """Run augr.delensing.compute_n0_* full-sky on the reference inputs.

    Returns a dict with the same three keys as the reference NPZ:
    ``tt``, ``pp`` (= diagonal MV of EE/EB), ``mv`` (= full diagonal MV).
    """
    import jax.numpy as jnp

    from augr.delensing import (
        LensingSpectra,
        compute_n0_eb,
        compute_n0_ee,
        compute_n0_mv,
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

    n0_tt = np.asarray(compute_n0_tt(Ls, spectra, nl_tt, fullsky=True))
    n0_ee = np.asarray(compute_n0_ee(Ls, spectra, nl_ee, fullsky=True))
    n0_eb = np.asarray(compute_n0_eb(Ls, spectra, nl_ee, nl_bb, fullsky=True))
    n0_mv = np.asarray(compute_n0_mv(Ls, spectra, nl_tt, nl_ee, nl_bb,
                                     fullsky=True))

    n0_pp = 1.0 / (1.0 / n0_ee + 1.0 / n0_eb)  # diagonal MV(EE, EB)

    return {"tt": n0_tt, "pp": n0_pp, "mv": n0_mv}


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
              f"max_rel_err = {sub.max():.4e}  median = {np.median(sub):.4e}  "
              f"n = {sub.size}")


def plot_ratios(ref: dict, augr_full: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    Ls = ref["Ls"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, est in zip(axes, REF_ESTIMATORS, strict=True):
        ratio_full = augr_full[est] / ref[f"n0_{est}"]
        ax.semilogx(Ls, ratio_full, label="augr full-sky / plancklens",
                    color="C1")
        ax.axhline(1.0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_title(f"N_0^{est.upper()}")
        ax.set_xlabel("L")
        ax.set_ylabel("ratio")
        ax.set_ylim(0.5, 1.5) if est == "tt" else ax.set_ylim(0.0, 5.0)
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
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        print(f"Reference NPZ not found at {ref_path}. "
              "Run run_plancklens.py first.", file=sys.stderr)
        return 1

    print(f"Loading reference: {ref_path}")
    ref = load_reference(ref_path)
    Ls = ref["Ls"]

    print(f"Running augr full-sky N_0 at {Ls.size} L values "
          "(this can take ~10 minutes) ...", flush=True)
    augr_full = compute_augr_n0(ref)

    print("\n=== max relative error per L band (augr full-sky / plancklens) ===")
    for est in REF_ESTIMATORS:
        ref_arr = ref[f"n0_{est}"]
        print(f"\n  {est.upper()}:")
        report_band(relative_error(augr_full[est], ref_arr), Ls)

    print()
    print("Notes:")
    print("  - TT is the apples-to-apples comparison; expect <1% in bulk.")
    print("  - PP/MV: augr uses the diagonal HO02 Eq.22 MV;")
    print("    plancklens 'p_p'/'p' include inter-estimator cross-corrs")
    print("    (joint GMV). Diagonal MV is strictly >= joint GMV; the")
    print("    ratio is bounded > 1 wherever multiple estimators")
    print("    contribute significantly. Diagnostic only.")

    plot_ratios(ref, augr_full,
                out_dir=Path(__file__).parent / "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
