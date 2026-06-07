"""cmilc_diagnostics.py — constrained-moment ILC (cMILC) vs blind NILC.

Builds a small multi-band sim and cleans it two ways — blind **NILC**
(:func:`augr.nilc.nilc_clean`) and **cMILC** (:func:`augr.cmilc.cmilc_clean`, default
cMILC08) — then compares the foreground residual (true ``fg_qu`` through each cleaner's
weights) per ℓ-band and runs the ``A_res`` Fisher with each, printing σ(r) (baseline /
A_res flat / A_res Gaussian) and the FG-leakage bias Δr.

cMILC deprojects the dust + sync moment SEDs explicitly, so on a spatially-varying-β sky
(``--fg-model d10s5`` / ``d10s6``) it nulls the moment that blind NILC leaves as residual.
The per-band ``cMILC ÷ NILC`` residual ratio shows *which* ℓ-scales (and, by comparing
``--moments cmilc06`` vs ``cmilc08``, which moments) drive the surviving foreground.

Usage:
    pixi run python scripts/cmilc_diagnostics.py --fg-model d10s5 --nside 64
    pixi run python scripts/cmilc_diagnostics.py --fg-model d10s5 --moments cmilc06
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from augr.cmilc import CMILC06_MOMENTS, CMILC08_MOMENTS, cmilc_clean
from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.gnilc import alm2cl
from augr.nilc import nilc_clean
from augr.nilc_forecast import nilc_forecast, nilc_spectra
from augr.spectra import CMBSpectra

# Illustrative wide-band space-mission-like config (not a study design). 8 bands so cMILC08
# (6 constraints) keeps >=2 DoF for the variance minimization at every needlet band.
FREQS = (30.0, 45.0, 95.0, 150.0, 220.0, 280.0, 353.0, 402.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 14.0, 12.0, 10.0, 9.0)  # arcmin
W_INV = jnp.array([2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 8.0e-5, 1.5e-4, 4.0e-4, 6.0e-4])  # μK²·sr

_MOMENT_SETS = {"cmilc08": CMILC08_MOMENTS, "cmilc06": CMILC06_MOMENTS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", default="d10s5", choices=["d1s1", "d10s5", "d10s6"])
    p.add_argument("--moments", default="cmilc08", choices=list(_MOMENT_SETS))
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--r-in", type=float, default=0.0)
    p.add_argument("--ell-max", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--localization-fwhm",
        type=float,
        default=None,
        help="Gaussian localization FWHM [arcmin] for both cleaners (default: global weights).",
    )
    args = p.parse_args()
    moments = _MOMENT_SETS[args.moments]

    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=args.r_in,
        nside=args.nside,
        lmax=args.lmax,
        fg_model=args.fg_model,
        cmb_seed=args.seed,
    )
    hit = jnp.ones(12 * args.nside**2)
    total = assemble_band_maps(sky, W_INV, hit, noise_key=jax.random.PRNGKey(args.seed))
    noise = total - sky.cmb_qu - sky.fg_qu

    loc = args.localization_fwhm
    res_nilc = nilc_clean(
        total, BEAMS, lmax=args.lmax, nside=args.nside, localization_fwhm_arcmin=loc
    )
    res_cmilc, info = cmilc_clean(
        total,
        BEAMS,
        FREQS,
        lmax=args.lmax,
        nside=args.nside,
        moments=moments,
        localization_fwhm_arcmin=loc,
        return_diagnostics=True,
    )

    spec_kw = dict(total_qu=total, noise_qu=noise, fg_qu=sky.fg_qu, cmb_qu=sky.cmb_qu)
    spec_nilc = nilc_spectra(res_nilc, **spec_kw)
    spec_cmilc = nilc_spectra(res_cmilc, **spec_kw)

    fg_nilc = np.asarray(alm2cl(res_nilc.project(sky.fg_qu), args.lmax))
    fg_cmilc = np.asarray(alm2cl(res_cmilc.project(sky.fg_qu), args.lmax))
    ell = np.arange(args.lmax + 1)

    fc_kw = dict(f_sky=1.0, r_fid=args.r_in, ell_min=2, ell_max=args.ell_max, delta_ell=10)
    out_nilc = nilc_forecast(spec_nilc, **fc_kw)
    out_cmilc = nilc_forecast(spec_cmilc, **fc_kw)

    print("=" * 70)
    print(
        f"  cMILC diagnostics  fg={args.fg_model}  moments={args.moments}  "
        f"nside={args.nside}  lmax={args.lmax}"
    )
    print("=" * 70)
    print(f"  deprojected moments:   {info['moments']}")
    print(f"  constraints (CMB + moments): {info['n_constraints']}")
    print(f"  retained columns / needlet band: {info['retained_columns_per_band']}")
    print("  " + "-" * 66)
    print("  FG residual auto-spectrum, ℓ-band mean   [cMILC ÷ NILC]:")
    for lo, hi in [(4, 10), (10, 30), (30, 60), (60, 100), (100, args.lmax + 1)]:
        b = (ell >= lo) & (ell < hi)
        rn, rc = fg_nilc[b].mean(), fg_cmilc[b].mean()
        print(f"      ℓ {lo:3d}-{hi:3d}:  NILC {rn:.3e}   cMILC {rc:.3e}   ratio {rc / rn:6.3f}")
    print("  " + "-" * 66)
    print(f"  {'metric':<26}{'NILC':>16}{'cMILC':>16}")
    for key, label in [
        ("sigma_r_baseline", "σ(r) baseline"),
        ("sigma_r_flat", "σ(r) A_res flat"),
        ("sigma_r_gauss", "σ(r) A_res Gaussian"),
        ("delta_r", "Δr (debias-OFF)"),
    ]:
        print(f"  {label:<26}{out_nilc[key]:>16.3e}{out_cmilc[key]:>16.3e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
