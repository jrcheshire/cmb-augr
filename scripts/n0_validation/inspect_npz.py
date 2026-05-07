"""Quick diagnostic: print magnitudes of plancklens vs augr N_0.

Run from repo root:

    python scripts/n0_validation/inspect_npz.py

Compares plancklens's three QE keys against the matching augr-side
combinations:
  - plancklens 'ptt' (n0_tt)  vs  augr.compute_n0_tt
  - plancklens 'p_p' (n0_pp)  vs  MV(compute_n0_ee, compute_n0_eb)
  - plancklens 'p'   (n0_mv)  vs  compute_n0_mv (full 5-estimator MV)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "n0_reference_litebird.npz"


def main() -> int:
    if not REF.exists():
        print(f"NPZ not found at {REF}", file=sys.stderr)
        return 1
    ref = np.load(REF, allow_pickle=True)
    Ls = ref["Ls"]

    sample_idx = [0, 5, 20, 60, 100]
    sample_idx = [i for i in sample_idx if i < len(Ls)]
    sample_Ls = Ls[sample_idx]
    print(f"sample Ls: {sample_Ls}")

    print("\n--- plancklens reference values ---")
    for est in ("tt", "pp", "mv"):
        arr = np.asarray(ref[f"n0_{est}"])
        n_finite = np.sum(np.isfinite(arr))
        n_nonzero = np.sum(arr != 0)
        print(f"  n0_{est}: n_finite={n_finite}/{len(arr)}, "
              f"n_nonzero={n_nonzero}/{len(arr)}")
        print(f"           sampled = {arr[sample_idx]}")

    print("\n--- augr values at the same Ls ---")
    import jax.numpy as jnp

    from augr.delensing import (
        LensingSpectra,
        compute_n0_eb,
        compute_n0_ee,
        compute_n0_mv,
        compute_n0_tt,
    )

    spec = LensingSpectra(
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
    Ls_sample = jnp.asarray(sample_Ls, dtype=float)
    nl_tt = jnp.asarray(ref["nl_tt"])
    nl_ee = jnp.asarray(ref["nl_ee"])
    nl_bb = jnp.asarray(ref["nl_bb"])

    # augr matches:
    #   plancklens 'ptt'  -> compute_n0_tt
    #   plancklens 'p_p'  -> MV(compute_n0_ee, compute_n0_eb)
    #   plancklens 'p'    -> compute_n0_mv (full 5)
    augr_tt = np.asarray(compute_n0_tt(Ls_sample, spec, nl_tt))
    augr_ee = np.asarray(compute_n0_ee(Ls_sample, spec, nl_ee))
    augr_eb = np.asarray(compute_n0_eb(Ls_sample, spec, nl_ee, nl_bb))
    augr_mv = np.asarray(compute_n0_mv(Ls_sample, spec, nl_tt, nl_ee, nl_bb))

    augr_pp = 1.0 / (1.0 / augr_ee + 1.0 / augr_eb)

    print(f"  augr n0_tt:        {augr_tt}")
    print(f"  augr n0_pp (EE+EB): {augr_pp}")
    print(f"  augr n0_mv (full):  {augr_mv}")

    print("\n--- ratio plancklens/augr at sample Ls ---")
    for est, augr_arr in (("tt", augr_tt), ("pp", augr_pp), ("mv", augr_mv)):
        ref_arr = np.asarray(ref[f"n0_{est}"])[sample_idx]
        ratio = ref_arr / augr_arr
        print(f"  {est}: {ratio}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
