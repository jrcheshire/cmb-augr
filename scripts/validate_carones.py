"""Consume BROOM-produced residual-template spectra via augr Fisher.

Loads the (ell_center, C_ell^BB) npy pairs written by
`broom_residual_template.py`, plugs the post-NILC noise spectrum into
FisherForecast via external_noise_bb, and the Eq. 3.7 debiased
residual-template spectrum into SignalModel as a known shape with a
nuisance amplitude A_res.

Two Fisher runs are performed:

  (1) Carones-faithful: flat A_res prior (no entry in priors dict).
  (2) augr default:     Gaussian prior sigma(A_res) = 0.3.

Diagnostics printed:
  - sigma(r) for baseline (no template), (1), and (2).
  - 2x2 (r, A_res) Fisher-submatrix condition number, as a
    sanity check that the (r, A_res) sub-block is well-conditioned.

Usage:
    pixi run python scripts/validate_carones.py \\
        --tag litebird_ptep_d1s1_nilc_gal60_020sims
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from augr.config import DEFAULT_PRIORS_POST_COMPSEP
from augr.forecast import forecast_from_spectra

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE.parent / "data" / "broom_outputs"


# ---------------------------------------------------------------------------
# Analysis setup (defaults; any override via CLI)
# ---------------------------------------------------------------------------

F_SKY_DEFAULT = 0.6       # matches GAL60
ELL_MIN = 2
ELL_MAX = 180
DELTA_ELL = 5
ELL_PER_BIN_BELOW = 30
A_RES_PRIOR_DEFAULT = DEFAULT_PRIORS_POST_COMPSEP["A_res"]  # augr default


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_npy_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (ell_center, C_ell^BB) from a BROOM-produced npy."""
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Expected (n_bins, 2) array at {path}, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# Fisher runs
# ---------------------------------------------------------------------------

def run_fisher_variants(tres_ells: np.ndarray, tres_bb: np.ndarray,
                        nl_ells: np.ndarray, nl_bb: np.ndarray,
                        f_sky: float,
                        a_res_prior: float,
                        delensed_bb: np.ndarray | None = None,
                        delensed_bb_ells: np.ndarray | None = None) -> dict:
    """Three Fisher runs: baseline (no template) and with-template under
    flat and Gaussian A_res priors. Returns a dict of printable scalars.

    With ``delensed_bb`` / ``delensed_bb_ells`` provided, both
    SignalModel instances replace ``A_lens * cl_bb_lensed`` with the
    supplied residual BB (typically the output of
    ``iterate_delensing``); ``A_lens`` is then dropped from the
    parameter vector and the lensing-B variance contribution to
    ``sigma(r)`` shrinks accordingly. The supplied range must span
    ``[ELL_MIN, ELL_MAX]``; ``iterate_delensing``'s default
    ``ls=[2, 300]`` covers ELL_MAX = 180.

    Thin wrapper over :func:`augr.forecast.forecast_from_spectra` (the shared
    post-separation forecast core): the noise lives on ``nl_ells`` and the residual
    template on ``tres_ells`` (different bandpower grids). Returns the original
    (smaller) key set this script's report and ``plot_broom_showcase`` expect.
    """
    res = forecast_from_spectra(
        nl_ells=nl_ells,
        nl_post=nl_bb,
        template_ells=tres_ells,
        template_cl=tres_bb,
        f_sky=f_sky,
        r_fid=0.0,
        ell_min=ELL_MIN,
        ell_max=ELL_MAX,
        delta_ell=DELTA_ELL,
        ell_per_bin_below=ELL_PER_BIN_BELOW,
        a_res_prior=a_res_prior,
        delensed_bb=delensed_bb,
        delensed_bb_ells=delensed_bb_ells,
    )
    return {
        "sigma_r_baseline": res.sigma_r_baseline,
        "sigma_r_flat": res.sigma_r_flat,
        "sigma_r_gauss": res.sigma_r_gauss,
        "sigma_A_res_flat": res.sigma_A_res_flat,
        "sigma_A_res_gauss": res.sigma_A_res_gauss,
        "cond_r_Ares": res.cond_r_Ares,
        "a_res_prior": res.a_res_prior,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(tag: str, f_sky: float, nl_range: tuple[float, float],
                  tres_range: tuple[float, float], result: dict) -> None:
    nl_lo, nl_hi = nl_range
    tres_lo, tres_hi = tres_range
    lines = [
        "=" * 70,
        f"  validate_carones  tag: {tag}",
        "=" * 70,
        f"  f_sky:                           {f_sky:.2f}",
        f"  ell range:                       [{ELL_MIN}, {ELL_MAX}]",
        f"  post-NILC N_ell^BB range:        [{nl_lo:.2e}, {nl_hi:.2e}] uK^2",
        f"  residual-template C_ell range:   [{tres_lo:.2e}, {tres_hi:.2e}] uK^2",
        "-" * 70,
        f"  sigma(r) [baseline: no template]          = {result['sigma_r_baseline']:.3e}",
        f"  sigma(r) [template, A_res flat prior]     = {result['sigma_r_flat']:.3e}",
        (f"  sigma(r) [template, A_res Gauss "
         f"sigma={result['a_res_prior']:.2g}]  = {result['sigma_r_gauss']:.3e}"),
        f"  sigma(A_res) [flat prior]                 = {result['sigma_A_res_flat']:.3e}",
        f"  sigma(A_res) [Gaussian prior]             = {result['sigma_A_res_gauss']:.3e}",
        f"  (r, A_res) marginalized cond number       = {result['cond_r_Ares']:.2e}",
        "=" * 70,
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", type=str, required=True,
                   help="File tag written by broom_residual_template.py, "
                        "e.g. 'litebird_ptep_d1s1_nilc_gal60_020sims'.")
    p.add_argument("--fsky", type=float, default=F_SKY_DEFAULT,
                   help="Effective f_sky for the Knox mode count; should "
                        "match the mask used when producing the inputs "
                        "(default: 0.6 for GAL60).")
    p.add_argument("--a-res-prior", type=float, default=A_RES_PRIOR_DEFAULT,
                   help="Gaussian prior width sigma(A_res) for the "
                        "augr-default Fisher run (default: 0.3).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    nl_path = OUTPUTS_DIR / f"{args.tag}_nl_bb.npy"
    tres_path = OUTPUTS_DIR / f"{args.tag}_tres_bb.npy"
    for p in (nl_path, tres_path):
        if not p.exists():
            sys.exit(f"error: missing input {p}. Run "
                     f"broom_residual_template.py first.")

    nl_ells, nl_bb = _load_npy_pair(nl_path)
    tres_ells, tres_bb = _load_npy_pair(tres_path)

    if not np.allclose(nl_ells, tres_ells):
        sys.exit(f"error: nl_ells and tres_ells disagree "
                 f"(shapes {nl_ells.shape} vs {tres_ells.shape}).")

    result = run_fisher_variants(
        tres_ells=tres_ells, tres_bb=tres_bb,
        nl_ells=nl_ells, nl_bb=nl_bb,
        f_sky=args.fsky,
        a_res_prior=args.a_res_prior,
    )
    _print_report(
        tag=args.tag,
        f_sky=args.fsky,
        nl_range=(float(nl_bb.min()), float(nl_bb.max())),
        tres_range=(float(tres_bb.min()), float(tres_bb.max())),
        result=result,
    )


if __name__ == "__main__":
    main()
