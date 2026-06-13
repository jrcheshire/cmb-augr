"""masking_diagnostics.py — cut-sky B-mode estimator: purity null, transfer, fidelity.

Demonstrates the masked-Wiener B-mode estimator (:mod:`augr.masking`, built on
jht) on single-cleaned-map validation sims:

* the **purity null** — inject realistic *lensed EE* with B=0 and show the
  recovered B power (the E→B leakage) is ≪ the lensing-BB floor, reported as a
  **Δr-equivalent** (leakage as a fraction of the tensor r=1 template) so it can
  be compared to a typical σ(r) ~ 10⁻³;
* the **transfer function** F_b (from B-only sims; absorbs the beam B_ℓ² and the
  Wiener gain) and the **fidelity** of the debiased recovery of an injected
  r=r_fid BB.

This is the estimator-level check behind the gated tests in
``tests/test_masking.py``; the production wiring (per-band foregrounds, MC
covariance, instrument sweep) lands separately.

Usage:
    pixi run python scripts/masking_diagnostics.py --nside 256 --f-sky 0.6 --nsims 4
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from augr import masking as mk
from augr.compsep_sims import cmb_b_alm, cmb_e_alm, cmb_eb_qu
from augr.delensing import load_lensing_spectra
from augr.instrument import beam_bl
from augr.signal import _build_bin_matrix, _make_bin_edges
from augr.spectra import CMBSpectra


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--lmax", type=int, default=300)
    p.add_argument("--fwhm", type=float, default=30.0, help="beam FWHM [arcmin]")
    p.add_argument("--f-sky", type=float, default=0.6)
    p.add_argument("--nsims", type=int, default=4)
    p.add_argument("--r-fid", type=float, default=0.01)
    p.add_argument("--var-pix", type=float, default=1e-5, help="per-pixel noise var [μK²]")
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=200)
    p.add_argument("--delta-ell", type=int, default=20)
    args = p.parse_args()

    nside, lmax = args.nside, args.lmax
    ell = np.arange(lmax + 1)
    bl = np.asarray(beam_bl(jnp.asarray(ell, dtype=float), args.fwhm))

    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb_len = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    cl_ee_p = cl_ee * bl**2  # beamed priors describe the (beamed) data covariance
    cl_bb_p = cl_bb_len * bl**2

    spec = CMBSpectra()
    cl_bb_true = jnp.clip(spec.cl_bb(jnp.asarray(ell, dtype=float), args.r_fid), 0.0, None)
    cl_tensor_r1 = jnp.clip(spec.cl_tensor_r1(jnp.asarray(ell, dtype=float)), 0.0, None)

    edges = _make_bin_edges(args.ell_min, args.ell_max, 2, args.delta_ell)
    bm, centers = _build_bin_matrix(np.arange(args.ell_min, args.ell_max + 1), edges, "tophat")
    centers = np.asarray(centers)

    import healpy as hp

    npix = hp.nside2npix(nside)
    mask = mk.galactic_mask(nside, args.f_sky)
    invn = mk.inv_noise_map(jnp.ones(npix), args.var_pix, mask=mask)
    print(
        f"nside={nside} lmax={lmax} fwhm={args.fwhm}' f_sky={mk.f_sky_of(mask):.3f} "
        f"nsims={args.nsims} r_fid={args.r_fid}"
    )

    def estimate(qu, seed):
        noise = jnp.asarray(
            np.random.default_rng(seed).standard_normal((2, npix)) * np.sqrt(args.var_pix)
        )
        cl = mk.masked_wiener_bb(qu + noise, invn, cl_ee_p, cl_bb_p, nside=nside, lmax=lmax)
        return mk.bin_spectrum(cl, bm, args.ell_min)

    true_b = mk.bin_spectrum(cl_bb_true, bm, args.ell_min)
    lensing_floor = mk.bin_spectrum(cl_bb_len, bm, args.ell_min)
    tensor_r1_b = mk.bin_spectrum(cl_tensor_r1, bm, args.ell_min)

    # transfer (B-only) + leakage (E-only)
    rec_b = jnp.stack(
        [
            estimate(_b_only(spec, args.r_fid, lmax, nside, args.fwhm, 100 + s), s)
            for s in range(args.nsims)
        ]
    )
    F = mk.transfer_function(rec_b, true_b)
    rec_e = jnp.stack(
        [
            estimate(_e_only(cl_ee, lmax, nside, args.fwhm, 300 + s), s + 50)
            for s in range(args.nsims)
        ]
    )
    leak = mk.leakage_template(rec_e)

    # purity null: leakage as Δr-equivalent (leak corrected by the transfer, mapped
    # through the tensor template). Δr_b = (leak_b / F_b) / C_b^{tensor,r=1}.
    dr_equiv = np.asarray((leak / F) / tensor_r1_b)

    # fidelity: mean debiased recovery over held-out B-only sims
    deb = jnp.stack(
        [
            mk.debias_bandpower(
                estimate(_b_only(spec, args.r_fid, lmax, nside, args.fwhm, 500 + s), 700 + s),
                F,
                leak,
            )
            for s in range(args.nsims)
        ]
    )
    fid = np.asarray(jnp.mean(deb, axis=0)) / np.asarray(true_b)

    leak_ratio = np.asarray(leak) / np.asarray(lensing_floor)
    print("\n  ℓ_c    F_b    leak/lensBB   Δr-equiv    fidelity(mean/true)")
    for b in range(len(centers)):
        print(
            f"  {centers[b]:5.0f}  {float(F[b]):5.3f}   {leak_ratio[b]:9.2e}   "
            f"{dr_equiv[b]:9.2e}   {fid[b]:6.3f}"
        )
    print(
        f"\nmax |Δr-equiv| over bins: {np.max(np.abs(dr_equiv)):.2e}  "
        f"(cf. typical σ(r) ~ 1e-3 → leakage is sub-dominant)"
    )
    print(f"leakage/lensing-BB: max {np.max(leak_ratio):.2e}")


def _b_only(spec, r, lmax, nside, fwhm, seed):
    b = cmb_b_alm(spec, r, lmax, seed=seed)
    return cmb_eb_qu(jnp.zeros_like(b), b, fwhm, lmax, nside)


def _e_only(cl_ee, lmax, nside, fwhm, seed):
    e = cmb_e_alm(cl_ee, lmax, seed=seed)
    return cmb_eb_qu(e, jnp.zeros_like(e), fwhm, lmax, nside)


if __name__ == "__main__":
    main()
