"""controlled_input_test.py - Constant-C, no-noise sanity check on augr's TT N_0.

Background
----------
``run_plancklens.py`` + ``compare.py`` show a residual ~2x discrepancy
between augr's ``compute_n0_tt`` and ``plancklens.nhl.get_nhl(qe_key='ptt')``
at LiteBIRD-PTEP that does NOT go away after ruling out the obvious
causes (estimator family, lensed-vs-unlensed cls, joint-vs-diagonal TE
filter). To localize whether the residual is in augr's TT formula or
in plancklens conventions, this script feeds both codes a controlled
input where the analytic answer is known.

Setup
-----
We set ``C_TT(l) = C0`` (constant) for all l, ``N_l = 0``, and
``cl_*_unl = cl_*_len`` (no lensing-induced shape difference). All
other spectra are zero. The TT QE integrand simplifies because
``L.l1 + L.l2 = L.(l1 + l2) = L.L = L^2`` for any (l1, l2) with
``l2 = L - l1``:

    f(L, l1, l2) = C0 (L.l1) + C0 (L.l2) = C0 L^2
    denom         = 2 C0^2
    F             = f / denom = L^2 / (2 C0)
    f * F         = L^4 / 2     (independent of phi, l1, C0)

Augr's discrete-sum-over-l1 integration measure
``contrib(l1) = sum_phi(f * F * w_phi) * l1 / (2 pi)^2`` then gives,
with ``sum_phi w_phi = 2 pi``:

    total(L) = (L^4 / (4 pi)) * S,  S = sum_{l1=lmin}^{lmax} l1

so

    N_0^TT(L) = 4 pi / (L^4 * S)

and ``S = lmax(lmax+1)/2 - lmin(lmin-1)/2``.

What this tests
---------------
- the ``L.l1 + L.l2`` response shape on the TT estimator;
- the factor of 2 in the same-field denominator;
- the ``l1 / (2 pi)^2`` integration measure;
- the phi integral via Gauss-Legendre nodes weighted to sum to 2 pi.

If augr matches this closed form, the TT QE formula is internally
self-consistent and the ~2x discrepancy with plancklens lives in
plancklens or in the conventions plumbing between them.

Full-sky companion check
------------------------
Augr's ``_compute_n0_tt_fullsky`` uses Wigner-3j coupling with
``alpha_i = [L(L+1) + l_i(l_i+1) - l_j(l_j+1)] / 2`` (full-sky analog
of ``L.l_i``). Because ``alpha1 + alpha2 = L(L+1)``, constant C also
gives a closed form on the full sky:

    1/N_0^{TT,full}(L)
        = (1/(2L+1)) sum_{l1, l2} f^2 / (2 C0^2)
        ~ L^2 (L+1)^2 / (8 pi) * S',  S' = sum_{l1=lmin}^{lmax} (2 l1 + 1)
                                          = (l_max+1)^2 - l_min^2

(using the Wigner-3j orthogonality
``sum_{l2} (2 l2 + 1) w_000(l1, l2, L)^2 = 1``, exact in the
``l_max -> infinity`` limit). So

    N_0^{TT,full}(L) ~ 8 pi / [L^2 (L+1)^2 * S'].

The flat/full ratio is ``2 (L+1)^2 / L^2 * S / S' ~ (L+1)^2 / L^2``,
which is geometric (not a bug): 2.25 at L=2, 1.21 at L=10, 1.02 at
L=100. So **the flat vs full-sky split at low L is partly expected
geometry, partly potentially-buggy**; the test asks how close augr's
full-sky path gets to the closed form above.

plancklens (--plancklens flag)
------------------------------
plancklens.nhl.get_nhl uses full-sky Wigner-3j coupling, so it should
hit the same closed form ``N_0 ~ 8 pi / [L^2 (L+1)^2 * S']`` on the
constant-C input. If it does, the LiteBIRD-PTEP discrepancy lives in
the spectra-shape interaction (cls_ivfs construction, lensed-vs-
unlensed cls in response, etc.); if it does NOT, plancklens has a
formula bug under our calling conventions.

The ``--plancklens`` flag requires plancklens to be importable, which
the augr pixi env does not have. Run from the dedicated env:

    conda run -n n0val python scripts/n0_validation/controlled_input_test.py --plancklens

Why TT only (for now)
---------------------
EE / EB / TB carry a ``cos(2 phi_12)`` or ``sin(2 phi_12)`` spin-2
factor whose phi-average against the triangle geometry does NOT
collapse to a constant when C is constant. A controlled-input test
for those estimators needs a high-precision numerical reference, not
a closed form, and is a separate exercise from this one.

Usage
-----
    pixi run python scripts/n0_validation/controlled_input_test.py
    conda run -n n0val python scripts/n0_validation/controlled_input_test.py --plancklens
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from augr.delensing import LensingSpectra, compute_n0_tt  # noqa: E402


def make_constant_spectra(C0: float, ell_max: int) -> LensingSpectra:
    """Build a stub LensingSpectra with C_TT = C0 (constant) and others zero.

    The TT array is non-zero from l=0 to l=ell_max so that the QE
    integration's ``l2 = |L - l1|`` lookups never fall onto the
    ``_interp_at`` zero-extrapolation region (which would silently
    break the constant assumption at the integration boundary).
    """
    ells = jnp.arange(ell_max + 1, dtype=float)
    cl_const = jnp.full(ell_max + 1, float(C0))
    zero = jnp.zeros(ell_max + 1)
    return LensingSpectra(
        ells=ells,
        cl_tt_unl=cl_const,
        cl_ee_unl=zero,
        cl_bb_unl=zero,
        cl_te_unl=zero,
        cl_tt_len=cl_const,  # match unlensed -> no lensed/unlensed split
        cl_ee_len=zero,
        cl_bb_len=zero,
        cl_te_len=zero,
        cl_pp=zero,
    )


def analytic_n0_tt_flatsky(L: np.ndarray, l_min: int, l_max: int) -> np.ndarray:
    """Flat-sky closed-form N_0^TT under constant C, no noise.

    N_0(L) = 4 pi / (L^4 * S),  S = sum_{l=l_min}^{l_max} l.
    """
    S = 0.5 * (l_max * (l_max + 1) - l_min * (l_min - 1))
    L = np.asarray(L, dtype=float)
    return 4.0 * np.pi / (L**4 * S)


def analytic_n0_tt_fullsky(L: np.ndarray, l_min: int, l_max: int) -> np.ndarray:
    """Full-sky closed-form N_0^TT under constant C, no noise.

    N_0(L) ~ 8 pi / [L^2 (L+1)^2 * S'],  S' = (l_max+1)^2 - l_min^2.

    Approximate to leading order in 1 / l_max via the Wigner-3j
    orthogonality ``sum_{l2} (2 l2 + 1) w_000^2 = 1`` (exact only as
    l_max -> infinity). Boundary effects from finite l_max enter at
    O(L / l_max).
    """
    Sp = (l_max + 1) ** 2 - l_min ** 2
    L = np.asarray(L, dtype=float)
    return 8.0 * np.pi / (L ** 2 * (L + 1) ** 2 * Sp)


def plancklens_n0_tt_constant(C0: float, l_min: int, l_max: int,
                              Ls: np.ndarray) -> np.ndarray:
    """plancklens.nhl + qresp under constant C, no noise (TT-only).

    Builds cls / fal dicts in the same plancklens conventions as
    ``run_plancklens.py`` but with constant C in TT only and zeros
    everywhere else. The TT entries are non-zero only on
    ``[l_min, l_max]`` so the QE filter sums respect the same
    integration range as augr.
    """
    from plancklens import nhl, qresp

    # Length must reach at least l_max + max(L) for plancklens's internal
    # 3j tables; the cls and fal arrays only need to be non-zero on
    # [l_min, l_max] for the controlled input.
    lmax_ivf = int(l_max)
    n = lmax_ivf + 1

    cl_const = np.zeros(n)
    cl_const[l_min:l_max + 1] = C0
    fal_const = np.zeros(n)
    fal_const[l_min:l_max + 1] = 1.0 / C0

    fal = {"tt": fal_const,
           "ee": np.zeros(n),
           "bb": np.zeros(n),
           "te": np.zeros(n)}
    # cls_ivfs = fal . dat . fal: for diagonal TT-only with no noise this is
    # just (1/C0) * C0 * (1/C0) = 1/C0 on [l_min, l_max].
    cls_ivfs = {"tt": fal_const.copy(),
                "ee": np.zeros(n),
                "bb": np.zeros(n),
                "te": np.zeros(n)}
    cls_w = {"tt": cl_const.copy(),
             "ee": np.zeros(n),
             "bb": np.zeros(n),
             "te": np.zeros(n)}

    lmax_qlm = int(np.max(Ls))
    n_gg, _ = nhl.get_nhl(
        "ptt", "ptt", cls_w, cls_ivfs,
        lmax_ivf, lmax_ivf, lmax_out=lmax_qlm,
    )[:2]
    r_gg, _ = qresp.get_response(
        "ptt", lmax_ivf, "p", cls_w, cls_w, fal,
        lmax_qlm=lmax_qlm,
    )[:2]
    n_gg = np.asarray(n_gg)
    r_gg = np.asarray(r_gg)
    n0 = np.where(r_gg ** 2 > 0, n_gg / np.where(r_gg ** 2 > 0, r_gg ** 2, 1.0), 0.0)
    Ls_idx = np.asarray(Ls).astype(int)
    return n0[Ls_idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C0", type=float, default=1.0,
                        help="Constant TT power level. Cancels in the ratio; "
                             "vary only as a sanity check.")
    parser.add_argument("--l-min", type=int, default=2,
                        help="Lower limit of the QE l1 sum (passed to compute_n0_tt).")
    parser.add_argument("--l-max", type=int, default=2000,
                        help="Upper limit of the QE l1 sum.")
    parser.add_argument("--n-phi", type=int, default=128,
                        help="GL nodes for the phi integral.")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Pass tolerance on the augr flat / analytic ratio.")
    parser.add_argument("--Ls", type=int, nargs="+",
                        default=[2, 5, 10, 30, 100, 300, 600],
                        help="L values at which to evaluate N_0^TT.")
    parser.add_argument("--fullsky", action="store_true",
                        help="Also run augr's full-sky compute_n0_tt path "
                             "and compare against the full-sky closed form.")
    parser.add_argument("--plancklens", action="store_true",
                        help="Also call plancklens (nhl + qresp) with the "
                             "same controlled input. Requires plancklens "
                             "(not in the augr pixi env; use the n0val "
                             "conda env).")
    args = parser.parse_args()

    # The cl arrays must be long enough that |l2| = |L - l1| stays inside
    # the constant region for every (L, l1, phi) probed. Worst case is
    # l1 = l_max, phi = pi: l2 = L + l_max. Pad to that.
    pad_max = args.l_max + max(args.Ls) + 8
    spec = make_constant_spectra(args.C0, pad_max)

    Ls = jnp.array(args.Ls, dtype=float)
    nl_tt = jnp.zeros_like(spec.cl_tt_unl)

    n0_flat = np.asarray(compute_n0_tt(
        Ls, spec, nl_tt,
        l_min=args.l_min, l_max=args.l_max, n_phi=args.n_phi,
    ))
    Ls_np = np.asarray(args.Ls)
    n0_flat_ana = analytic_n0_tt_flatsky(Ls_np, args.l_min, args.l_max)
    n0_full_ana = analytic_n0_tt_fullsky(Ls_np, args.l_min, args.l_max)

    print(f"Controlled-input TT test: C_TT = {args.C0}, nl = 0, "
          f"l_min = {args.l_min}, l_max = {args.l_max}, n_phi = {args.n_phi}")
    Sflat = 0.5 * (args.l_max * (args.l_max + 1) - args.l_min * (args.l_min - 1))
    Sfull = (args.l_max + 1) ** 2 - args.l_min ** 2
    print(f"S_flat = {Sflat:.6e},  S_full = {Sfull:.6e}")
    print()

    rows = [n0_flat, n0_flat_ana, n0_flat / n0_flat_ana]
    headers = ["augr flat", "flat ana", "flat ratio"]

    # Full-sky augr (optional, slow due to wigner3j).
    n0_full = None
    if args.fullsky:
        n0_full = np.asarray(compute_n0_tt(
            Ls, spec, nl_tt,
            l_min=args.l_min, l_max=args.l_max, fullsky=True,
        ))
        rows += [n0_full, n0_full_ana, n0_full / n0_full_ana]
        headers += ["augr full", "full ana", "full ratio"]

    n0_pl = None
    if args.plancklens:
        n0_pl = plancklens_n0_tt_constant(args.C0, args.l_min, args.l_max, Ls_np)
        rows += [n0_pl, n0_pl / n0_full_ana, n0_pl / n0_flat_ana]
        headers += ["plancklens", "pl/full ana", "pl/flat ana"]

    width = 14
    head_line = f"{'L':>5}  " + "  ".join(f"{h:>{width}}" for h in headers)
    print(head_line)
    for i, L_ in enumerate(Ls_np):
        cells = [f"{r[i]:{width}.6e}" if isinstance(r[i], (float, np.floating))
                 else f"{r[i]:{width}}"
                 for r in rows]
        print(f"{int(L_):>5d}  " + "  ".join(cells))

    print()
    max_flat_err = float(np.max(np.abs(n0_flat / n0_flat_ana - 1.0)))
    print(f"flat:       max |ratio - 1| = {max_flat_err:.3e}  "
          f"(tolerance: {args.rtol:.0e})")
    if n0_full is not None:
        max_full_err = float(np.max(np.abs(n0_full / n0_full_ana - 1.0)))
        print(f"full-sky:   max |ratio - 1| = {max_full_err:.3e}  "
              "(boundary effects ~ L/l_max expected)")
    if n0_pl is not None:
        rel_full = n0_pl / n0_full_ana - 1.0
        max_pl_full = float(np.max(np.abs(rel_full)))
        print(f"plancklens: max |pl/full_ana - 1| = {max_pl_full:.3e}")
        if max_pl_full > 0.05:
            print()
            print("INTERESTING: plancklens disagrees with the constant-C closed form")
            print("by more than 5%. Either (a) the closed-form approximation breaks")
            print("(check L/l_max), or (b) plancklens uses a different convention")
            print("under our calling pattern. Inspect L = ", Ls_np.tolist())

    if max_flat_err > args.rtol:
        print("FAIL: augr flat-sky TT does NOT match the closed form.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
