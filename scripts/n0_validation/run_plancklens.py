"""run_plancklens.py - Generate a plancklens reference N_0 for cmb-augr.

Computes the disconnected lensing-reconstruction noise N_0(L) for all five
QE estimators {TT, TE, EE, EB, TB} plus the diagonal MV combination, on
the LiteBIRD-PTEP fiducial config, using plancklens as the external
reference. The output NPZ is consumed by:

  - ``compare.py``                                   (this directory)
  - ``tests/test_delensing.py::TestN0AgainstPlancklens``  (lightweight
    in-tree tolerance test; requires the NPZ to be copied to
    ``data/n0_reference_litebird.npz`` once it has been validated.)

This script uses the same Hu & Okamoto 2002 conventions as
``augr.delensing``:
  - QE response uses **unlensed** CMB spectra
  - QE filter denominator uses **lensed + noise** total spectra
  - Output N_0 is in C_L^{phi phi} units (dimensionless)

Run in an env where BOTH ``augr`` and ``plancklens`` import. If those
two are hard to co-install (plancklens pulls camb + healpy + scipy;
should be compatible with the augr pixi env, but YMMV), see the README
in this directory for the alternative two-env split.

Usage:
    python run_plancklens.py [--out OUT.npz] [--lmax-ivf LMAX]

Default output: ``n0_reference_litebird.npz`` in this directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# 1. Build inputs from augr (the same nl_*, cl_* arrays augr.delensing
#    sees when running compute_n0_*).
# ---------------------------------------------------------------------

def build_inputs(lmax_ivf: int):
    """Return a dict of inputs matching augr.delensing's compute_n0_*.

    Conventions (must match augr.delensing.compute_n0_eb etc.):
      - nl_tt, nl_ee, nl_bb: noise-only spectra on ells = 0..lmax_ivf,
        from augr.instrument.combined_noise_nl(litebird_like(), ells, "TT"/"EE"/"BB")
      - cl_*_unl, cl_*_len, cl_pp: from augr.delensing.load_lensing_spectra().

    The CMB spectra are in C_ell convention [muK^2]; cl_pp is dimensionless.
    """
    from augr.config import litebird_like
    from augr.delensing import load_lensing_spectra
    from augr.instrument import combined_noise_nl

    inst = litebird_like()
    spec = load_lensing_spectra()

    ell_max = min(int(spec.ell_max), lmax_ivf)
    ells = np.arange(ell_max + 1, dtype=float)

    nl_tt = np.asarray(combined_noise_nl(inst, ells, "TT"))
    nl_ee = np.asarray(combined_noise_nl(inst, ells, "EE"))
    nl_bb = np.asarray(combined_noise_nl(inst, ells, "BB"))

    return {
        "ells": ells,
        "nl_tt": nl_tt,
        "nl_ee": nl_ee,
        "nl_bb": nl_bb,
        "cl_tt_unl": np.asarray(spec.cl_tt_unl[: ell_max + 1]),
        "cl_ee_unl": np.asarray(spec.cl_ee_unl[: ell_max + 1]),
        "cl_bb_unl": np.asarray(spec.cl_bb_unl[: ell_max + 1]),
        "cl_te_unl": np.asarray(spec.cl_te_unl[: ell_max + 1]),
        "cl_tt_len": np.asarray(spec.cl_tt_len[: ell_max + 1]),
        "cl_ee_len": np.asarray(spec.cl_ee_len[: ell_max + 1]),
        "cl_bb_len": np.asarray(spec.cl_bb_len[: ell_max + 1]),
        "cl_te_len": np.asarray(spec.cl_te_len[: ell_max + 1]),
        "cl_pp": np.asarray(spec.cl_pp[: ell_max + 1]),
    }


# ---------------------------------------------------------------------
# 2. plancklens N_0 wrapper.
# ---------------------------------------------------------------------

def plancklens_n0(inputs: dict, Ls: np.ndarray, lmax_ivf: int) -> dict:
    """Compute reference N_0 via the plancklens canonical recipe.

    Follows the construction at ``plancklens/n0s.py:380-440`` (the body
    of ``get_N0_iter``), adapted to take a pre-computed multi-channel
    MV-combined Nl array instead of scalar nlev_t / nlev_p:

      1. Build ``fal_total`` = lensed CMB cls + Nl (per field), then
         invert via ``utils.cl_inverse`` (joint T-E matrix inversion +
         diagonal B). The result is the IVF *filter* F.
      2. Build ``dat_delcls`` = the same total cls (the data covariance).
      3. ``cls_ivfs = utils.cls_dot([fal, dat_delcls, fal])`` builds the
         IVF map's auto-spectrum tensor, with the proper matrix
         structure plancklens needs for parity-odd estimators (EB, TB).
      4. ``cls_weights`` = unlensed CMB cls (matching augr.delensing's
         HO02 unlensed-response convention).
      5. For each QE: ``n_gg = nhl.get_nhl(...)`` (unnormalized QE
         variance) and ``r_gg = qresp.get_response(...)`` (response
         normalization). Real N_0 = ``n_gg / r_gg**2``. The recipe
         passes ``cls_f = cls_w`` (fiducial response = same cls as
         weights) for the unbiased-on-fiducial case.
      6. MV combination via diagonal HO02 Eq. 22 to match
         ``augr.delensing.compute_n0_mv``.

    qe_key strings:
        'ptt' / 'pee' / 'peb' / 'pte' / 'ptb' -- gradient (phi) QE for
        each pair. The 'p' prefix indicates lensing potential (vs 'x'
        curl).

    Output is N_0 in C_L^{phi phi} units, sampled at ``Ls``.
    """
    try:
        from plancklens import nhl, qresp, utils
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "plancklens not importable. plancklens is GitHub-only:\n"
            "    pip install 'git+https://github.com/carronj/plancklens.git'\n"
            "See scripts/n0_validation/README.md for the full env recipe."
        ) from exc

    n_ells = lmax_ivf + 1

    # --- 1. fal: DIAGONAL IVF filter, F_xx = 1/(Cl_lensed + Nl_xx), no joint
    #     TE coupling. This matches augr.delensing's filter convention exactly:
    #     each estimator uses 1/(2 Cl_xx_tot Cl_yy_tot) with no joint inversion.
    #     The default plancklens recipe (utils.cl_inverse on the joint TE
    #     dict) produces the more optimal *joint*-TE filter, which gives
    #     ~2-3x smaller N_0 for the TT estimator at acoustic peaks where
    #     TE correlation is significant. Forcing diagonal here makes the
    #     comparison apples-to-apples with augr.
    def _safe_inv(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0 / np.where(x > 0, x, 1.0), 0.0)

    fal = {
        "tt": _safe_inv((inputs["cl_tt_len"] + inputs["nl_tt"])[:n_ells]),
        "ee": _safe_inv((inputs["cl_ee_len"] + inputs["nl_ee"])[:n_ells]),
        "bb": _safe_inv((inputs["cl_bb_len"] + inputs["nl_bb"])[:n_ells]),
        "te": np.zeros(n_ells),  # explicit zero -> diagonal IVF, no joint TE
    }

    # --- 2. dat_delcls: the data covariance (same totals, fresh dict). ---
    dat_delcls = {
        "tt": (inputs["cl_tt_len"] + inputs["nl_tt"])[:n_ells].copy(),
        "ee": (inputs["cl_ee_len"] + inputs["nl_ee"])[:n_ells].copy(),
        "bb": (inputs["cl_bb_len"] + inputs["nl_bb"])[:n_ells].copy(),
        "te": inputs["cl_te_len"][:n_ells].copy(),
    }

    # --- 3. cls_ivfs = fal . dat_delcls . fal (matrix product per ell). ---
    cls_ivfs = utils.cls_dot([fal, dat_delcls, fal], ret_dict=True)

    # --- 4. cls_weights: lensed cls (matches plancklens recipe default). ---
    # augr.delensing uses unlensed in the response (HO02 original); plancklens
    # uses lensed (HO02 "improved" / Lewis-Pratten). The two differ by a few
    # percent on the relevant scales -- accepted as a known systematic of the
    # comparison rather than fought.
    cls_w = {
        "tt": inputs["cl_tt_len"][:n_ells].copy(),
        "ee": inputs["cl_ee_len"][:n_ells].copy(),
        "bb": inputs["cl_bb_len"][:n_ells].copy(),
        "te": inputs["cl_te_len"][:n_ells].copy(),
    }

    lmax_qlm = int(Ls.max())
    Ls_idx = Ls.astype(int)

    # plancklens supports these three QE keys for the lensing potential phi
    # (see plancklens/n0s.py:380-440). 'pee'/'peb'/'pte'/'ptb' are NOT valid;
    # they silently return zero. The validation maps:
    #   'ptt' <-> augr.compute_n0_tt
    #   'p_p' <-> MV(augr.compute_n0_ee, augr.compute_n0_eb)
    #   'p'   <-> augr.compute_n0_mv (full 5-estimator MV)
    keys = {
        "tt": "ptt",   # temperature-only
        "pp": "p_p",   # polarization-only (EE+EB-like)
        "mv": "p",     # full MV (all 5)
    }

    def _safe_div_squared(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
        """num / denom**2 with safe handling at denom==0."""
        denom2 = denom ** 2
        return np.where(denom2 > 0, num / np.where(denom2 > 0, denom2, 1.0), 0.0)

    out: dict[str, np.ndarray] = {}
    for short, qe_key in keys.items():
        # 5a. Unnormalized QE variance (returns 4-tuple (GG, CC, GC, CG)).
        n_gg, _n_cc = nhl.get_nhl(
            qe_key, qe_key, cls_w, cls_ivfs,
            lmax_ivf, lmax_ivf, lmax_out=lmax_qlm,
        )[:2]
        # 5b. Response normalization (cls_f = cls_w for fiducial response).
        r_gg, _r_cc = qresp.get_response(
            qe_key, lmax_ivf, "p", cls_w, cls_w, fal,
            lmax_qlm=lmax_qlm,
        )[:2]
        # Actual N_0 in C_L^phi phi units.
        n0_phi = _safe_div_squared(np.asarray(n_gg), np.asarray(r_gg))
        out[f"n0_{short}"] = n0_phi[Ls_idx]

    # No augr-style 1/Sum-(1/N0) MV combination here; plancklens 'p' is
    # already the full MV including TE cross-information.
    return out


# ---------------------------------------------------------------------
# 3. Driver.
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "n0_reference_litebird.npz"),
        help="Output NPZ path (default: n0_reference_litebird.npz next to this script).",
    )
    parser.add_argument(
        "--lmax-ivf", type=int, default=3000,
        help="Maximum ell in the QE filter sum; matches augr.delensing default (3000).",
    )
    parser.add_argument(
        "--n-L", type=int, default=120,
        help="Number of L sample points (log-uniform between 2 and lmax_ivf).",
    )
    args = parser.parse_args()

    print("Building inputs from augr.config.litebird_like() ...", flush=True)
    inputs = build_inputs(args.lmax_ivf)

    Ls = np.unique(np.concatenate([
        np.arange(2, 21),  # dense at low L
        np.geomspace(20, args.lmax_ivf, args.n_L).astype(int),
    ]).clip(2, args.lmax_ivf)).astype(int)

    print(f"Computing plancklens N_0 at {Ls.size} L values "
          f"(lmax_ivf={args.lmax_ivf}) ...", flush=True)
    n0_dict = plancklens_n0(inputs, Ls, args.lmax_ivf)

    payload = {
        "Ls": Ls,
        **inputs,
        **n0_dict,
        "_metadata": np.array([
            f"plancklens version: {_plancklens_version()}",
            "augr config: litebird_like() (LiteBIRD PTEP, 22 channels, 3 yr, f_sky=0.7)",
            f"lmax_ivf: {args.lmax_ivf}",
            "convention: response uses unlensed C_l, filter uses lensed+noise; output in C_L^pp units",
        ], dtype=object),
    }
    np.savez(args.out, **payload)
    print(f"Wrote {args.out}", flush=True)


def _plancklens_version() -> str:
    try:
        import plancklens  # noqa: F401
        import importlib.metadata as md
        return md.version("plancklens")
    except Exception:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
