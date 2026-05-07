"""diagnose_fullsky_L2.py - Probe augr full-sky compute_n0_tt at L=2 on LiteBIRD.

Why? The controlled-input test showed augr full-sky agrees with the
constant-C closed form to ~0.08% at L=2, but compare.py reports
``augr_full / plancklens`` ratio of 8410 at L=2 on realistic LiteBIRD
spectra. So the bug is non-constant-spectrum + low-L specific. This
script samples each piece of the integrand independently to localize
which component diverges.

Usage:
    conda run -n n0val python scripts/n0_validation/diagnose_fullsky_L2.py
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

    # Recreate augr's _compute_n0_tt_fullsky internals at L=2.
    cl_tt_unl = np.asarray(ref["cl_tt_unl"])
    cl_tt_tot = np.asarray(ref["cl_tt_len"]) + np.asarray(ref["nl_tt"])
    print("len cls:", len(cl_tt_unl))
    print("cl_tt_unl[0:5] =", cl_tt_unl[:5])
    print("cl_tt_unl[2:6] =", cl_tt_unl[2:6])
    print("cl_tt_tot[0:5] =", cl_tt_tot[:5])

    # Spot-check: at L=2 the triangle gives l2 in [|l1-2|, l1+2].
    # For l1 = 2: l2 in [0, 4]. For l1 = 3: l2 in [1, 5]. etc.
    # cl_tt_unl[0] and cl_tt_unl[1] are the monopole/dipole entries
    # (typically zero from CAMB), and 1/cl_tt_tot[0,1] is also zero
    # (or zero-padded). Inspect.

    L = 2
    l_min = 2
    l_max = 2000

    from augr.wigner import wigner3j_000_vectorized
    l1_arr = np.arange(l_min, l_max + 1, dtype=int)
    l2_grid, w000 = wigner3j_000_vectorized(L, l1_arr, l2_min=l_min, l2_max=l_max)
    print(f"\nAt L={L}:")
    print(f"  l1_arr.shape = {l1_arr.shape}")
    print(f"  l2_grid.shape = {l2_grid.shape}")
    print(f"  w000.shape    = {w000.shape}")
    print(f"  l2_grid[:10]  = {l2_grid[:10]}")
    # For L=2, the (l1, l2, L; 0, 0, 0) symbol is non-zero only when
    # l1 + l2 + L is even, |l1 - 2| <= l2 <= l1 + 2.
    # So the non-trivial l2 values for each l1 are very restricted.

    L_LL = L * (L + 1)
    l1_ll1 = l1_arr * (l1_arr + 1)
    l2_ll2 = l2_grid * (l2_grid + 1)
    alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :]) / 2.0
    alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None]) / 2.0

    pf = np.sqrt((2 * l1_arr + 1)[:, None] * (2 * l2_grid + 1)[None, :] *
                 (2 * L + 1) / (4.0 * np.pi))

    tt_l1 = cl_tt_unl[l1_arr]
    tt_l2 = np.zeros(len(l2_grid))
    valid = (l2_grid >= 0) & (l2_grid < len(cl_tt_unl))
    tt_l2[valid] = cl_tt_unl[l2_grid[valid]]

    f_response = tt_l1[:, None] * alpha1 + tt_l2[None, :] * alpha2
    f_sq = (f_response) ** 2 * pf**2 * w000**2

    inv_tt_l1 = np.where(cl_tt_tot[l1_arr] > 0, 1.0 / cl_tt_tot[l1_arr], 0.0)
    # Mimic _fullsky_inv_spectrum:
    l2_int = l2_grid.astype(int)
    valid_l2 = (l2_int >= 0) & (l2_int < len(cl_tt_tot))
    inv_tt_l2 = np.zeros(len(l2_grid))
    inv_tt_l2[valid_l2] = np.where(cl_tt_tot[l2_int[valid_l2]] > 0,
                                   1.0 / cl_tt_tot[l2_int[valid_l2]], 0.0)

    integrand = f_sq * inv_tt_l1[:, None] * inv_tt_l2[None, :] / 2.0
    n0_inv = np.sum(integrand) / (2 * L + 1)
    n0 = 1.0 / n0_inv if n0_inv > 0 else float("inf")
    print(f"\n  augr full-sky N_0(L={L}) = {n0:.6e}")
    print(f"  cumulative: 1/N_0 = {n0_inv:.6e}")

    # Find which (l1, l2) cells dominate the integrand.
    flat = integrand.ravel()
    idx_top = np.argsort(flat)[-15:][::-1]
    print("\n  Top 15 (l1, l2) cells:")
    print(f"  {'l1':>4} {'l2':>4} {'cl_unl(l1)':>14} {'cl_unl(l2)':>14} "
          f"{'cl_tot(l1)':>14} {'cl_tot(l2)':>14} {'integrand':>14}")
    for k in idx_top:
        i, j = np.unravel_index(k, integrand.shape)
        l1_v = int(l1_arr[i])
        l2_v = int(l2_grid[j])
        cu_l1 = cl_tt_unl[l1_v] if l1_v < len(cl_tt_unl) else float("nan")
        cu_l2 = cl_tt_unl[l2_v] if 0 <= l2_v < len(cl_tt_unl) else float("nan")
        ct_l1 = cl_tt_tot[l1_v] if l1_v < len(cl_tt_tot) else float("nan")
        ct_l2 = cl_tt_tot[l2_v] if 0 <= l2_v < len(cl_tt_tot) else float("nan")
        print(f"  {l1_v:>4} {l2_v:>4} {cu_l1:>14.4e} {cu_l2:>14.4e} "
              f"{ct_l1:>14.4e} {ct_l2:>14.4e} {flat[k]:>14.4e}")

    # Also probe whether there's a w000-dropping bug at low l.
    print(f"\n  w000 stats: nonzero = {np.count_nonzero(w000)}, "
          f"max abs = {np.max(np.abs(w000)):.4e}")
    print(f"  w000^2 sum = {np.sum(w000**2):.4e}")

    # Compare against plancklens reference at L=2.
    Ls_ref = np.asarray(ref["Ls"])
    n0_pl_at_L = ref["n0_tt"][Ls_ref == L]
    print(f"\n  plancklens N_0(L={L})    = {float(n0_pl_at_L[0]):.6e}")
    print(f"  augr / plancklens         = {n0 / float(n0_pl_at_L[0]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
