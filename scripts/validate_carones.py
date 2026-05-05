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
  - 2x2 (r, A_res) Fisher-submatrix condition number (acceptance
    criterion 3 of the plan).

Usage:
    pixi run python scripts/validate_carones.py \\
        --tag litebird_ptep_d1s1_nilc_gal60_020sims
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from augr.config import cleaned_map_instrument
from augr.fisher import FisherForecast
from augr.foregrounds import NullForegroundModel
from augr.signal import SignalModel
from augr.spectra import CMBSpectra


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
A_RES_PRIOR_DEFAULT = 0.3  # augr's default Gaussian prior width


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
                        a_res_prior: float) -> dict:
    """Three Fisher runs: baseline (no template) and with-template under
    flat and Gaussian A_res priors. Returns a dict of printable scalars."""
    inst = cleaned_map_instrument(f_sky=f_sky)
    cmb = CMBSpectra()

    # Baseline SignalModel -- no residual template, just CMB + noise.
    baseline = SignalModel(
        instrument=inst,
        foreground_model=NullForegroundModel(),
        cmb_spectra=cmb,
        ell_min=ELL_MIN, ell_max=ELL_MAX,
        delta_ell=DELTA_ELL, ell_per_bin_below=ELL_PER_BIN_BELOW,
    )

    # Nearest-neighbour extrapolation (jnp.interp default) outside the
    # BROOM bandpower range -- zero-extrapolation at ell < first bin
    # center would silently drop noise at the reionization bump where
    # sigma(r) is most sensitive.
    nl_interp = jnp.interp(baseline.ells,
                           jnp.asarray(nl_ells),
                           jnp.asarray(nl_bb))
    external_noise_bb = nl_interp[None, :]  # (n_channels=1, n_ells)

    fiducial_base = {"r": 0.0, "A_lens": 1.0}
    fisher_baseline = FisherForecast(
        baseline, inst, fiducial_base,
        priors={}, fixed_params=[],
        external_noise_bb=external_noise_bb,
    )
    fisher_baseline.compute()

    # With residual template.
    signal = SignalModel(
        instrument=inst,
        foreground_model=NullForegroundModel(),
        cmb_spectra=cmb,
        ell_min=ELL_MIN, ell_max=ELL_MAX,
        delta_ell=DELTA_ELL, ell_per_bin_below=ELL_PER_BIN_BELOW,
        residual_template_cl=tres_bb,
        residual_template_ells=tres_ells,
    )
    fiducial = {**fiducial_base, "A_res": 1.0}

    fisher_flat = FisherForecast(
        signal, inst, fiducial,
        priors={}, fixed_params=[],
        external_noise_bb=external_noise_bb,
    )
    fisher_flat.compute()

    fisher_gauss = FisherForecast(
        signal, inst, fiducial,
        priors={"A_res": a_res_prior}, fixed_params=[],
        external_noise_bb=external_noise_bb,
    )
    fisher_gauss.compute()

    # (r, A_res) marginalized condition-number diagnostic.
    # Taking F[[r,A_res]][:, [r,A_res]] (the 2x2 sub-block of F) would
    # give the *conditional* Fisher on (r, A_res) with A_lens held fixed
    # at fiducial, and would under-flag the degeneracy that matters
    # physically -- which is the (r, A_res) constraint *after*
    # marginalizing over A_lens. The covariance C = F^-1 naturally
    # marginalizes out non-selected parameters, and cond(C_sub) equals
    # cond(F_marg_sub) (cond is invariant under inversion).
    F = np.asarray(fisher_flat.fisher_matrix)
    names = fisher_flat.free_parameter_names
    r_idx = names.index("r")
    a_idx = names.index("A_res")
    C = np.linalg.inv(F)
    ix = np.array([r_idx, a_idx])
    C_sub = C[np.ix_(ix, ix)]
    cond = np.linalg.cond(C_sub)

    return {
        "sigma_r_baseline": fisher_baseline.sigma("r"),
        "sigma_r_flat": fisher_flat.sigma("r"),
        "sigma_r_gauss": fisher_gauss.sigma("r"),
        "sigma_A_res_flat": fisher_flat.sigma("A_res"),
        "sigma_A_res_gauss": fisher_gauss.sigma("A_res"),
        "cond_r_Ares": cond,
        "a_res_prior": a_res_prior,
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
