"""cutsky_headline.py — σ(r): scalar-1/f_sky Knox vs cut-sky masked-Wiener MC.

The headline for the cut-sky B-mode realism work: run the *same* forecast spine
twice — once with the full-sky analytic Knox covariance and its scalar ``1/f_sky``
correction (``SpectrumSource.FULLSKY_SCALAR``), once with a Monte-Carlo covariance
from a spin-2 Q/U cleaner through the masked-Wiener cut-sky estimator
(``SpectrumSource.CUTSKY_MC``) — and report the σ(r) ratio. Both arms share the
residual-foreground template and the Fisher Jacobian, so the difference isolates what
the scalar-``1/f_sky`` approximation costs: the mask-dependent E→B leakage variance
(and, with foregrounds on, the FG-residual variance).

This drives the *pipeline* (:func:`augr.pipeline.run_forecast`) — no bespoke Fisher /
SignalModel wiring. The MC ensemble is the cost driver (one spin-2 clean + three
masked-Wiener solves per sim); use ``--workers`` once the cleaner is picklable, else
run serial.

Usage:
    pixi run python scripts/cutsky_headline.py --nside 128 --f-sky 0.6 --nsims-cov 80
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.hit_maps import l2_hit_map
from augr.pipeline import ForecastConfig, ResidualTemplateSource, SpectrumSource, run_forecast

# Illustrative wide-band space-mission-like configuration (LiteBIRD-PTEP-ish; not a
# study design — matches scripts/nilc_diagnostics.py so the two scripts are comparable).
FREQS = (30.0, 44.0, 95.0, 150.0, 280.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 12.0)  # arcmin
W_INV = (2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 1.5e-4)  # uK^2 sr per band


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", default="d1s1", choices=["d1s1", "d10s5", "none"])
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--lmax", type=int, default=192)
    p.add_argument("--f-sky", type=float, default=0.6)
    p.add_argument("--r-in", type=float, default=0.0)
    p.add_argument("--ell-max", type=int, default=150)
    p.add_argument("--delta-ell", type=int, default=15)
    p.add_argument("--ell-per-bin-below", type=int, default=10)
    p.add_argument("--nsims-cov", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--uniform-hits",
        action="store_true",
        help="uniform hit map instead of the default L2 anisotropic scan depth",
    )
    p.add_argument(
        "--residual",
        default="oracle",
        choices=["oracle", "gnilc"],
        help="residual-FG template source (shared by both arms)",
    )
    args = p.parse_args()

    fg_model = None if args.fg_model == "none" else args.fg_model
    npix = 12 * args.nside**2
    hit = jnp.ones(npix) if args.uniform_hits else jnp.asarray(l2_hit_map(args.nside, coord="G"))
    mask = mk.galactic_mask(args.nside, args.f_sky)
    residual = ResidualTemplateSource[args.residual.upper()]

    base = ForecastConfig(
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=W_INV,
        cleaner=nilc_cleaner(clean_e=True),  # spin-2 cleaner (CUTSKY needs cleaned_qu)
        nside=args.nside,
        lmax=args.lmax,
        fg_model=fg_model,
        r_in=args.r_in,
        seed=args.seed,
        hit_map=hit,
        residual_source=residual,
        f_sky=args.f_sky,
        r_fid=args.r_in,
        ell_min=2,
        ell_max=args.ell_max,
        delta_ell=args.delta_ell,
        ell_per_bin_below=args.ell_per_bin_below,
    )

    knox = run_forecast(replace(base, spectrum_source=SpectrumSource.FULLSKY_SCALAR))
    mc = run_forecast(
        replace(
            base,
            spectrum_source=SpectrumSource.CUTSKY_MC,
            mask=mask,
            n_sims_mc=args.nsims_cov,
            base_seed_mc=args.seed,
            mc_workers=args.workers,
        )
    )

    fsky_real = mk.f_sky_of(mask)
    print("=" * 68)
    print(
        f"  cut-sky σ(r) headline   fg={args.fg_model}  nside={args.nside}  "
        f"lmax={args.lmax}  f_sky={fsky_real:.3f}  nsims={args.nsims_cov}"
    )
    print("=" * 68)
    print(f"  {'':<20}{'Knox (1/f_sky)':>16}{'MC (masked-W)':>16}{'MC/Knox':>10}")
    for label, key in (
        ("σ(r) baseline", "sigma_r_baseline"),
        ("σ(r) A_res flat", "sigma_r_flat"),
        ("σ(r) A_res Gauss", "sigma_r_gauss"),
    ):
        k = getattr(knox, key)
        m = getattr(mc, key)
        ratio = m / k if k else float("nan")
        print(f"  {label:<20}{k:>16.3e}{m:>16.3e}{ratio:>10.2f}")
    print(f"  {'Δr (debias-OFF)':<20}{knox.delta_r:>16.3e}{mc.delta_r:>16.3e}")
    print("=" * 68)
    print(
        "  The MC covariance adds what the scalar-1/f_sky Knox omits: the mask-dependent\n"
        "  E→B leakage variance and the bin-bin correlations. The σ(r) ratio quantifies\n"
        "  that. The baseline (single-parameter) ratio is the cleanest read; the A_res-\n"
        "  marginalized ratios also fold in the covariance structure and need enough\n"
        "  sims/bins to be stable (n_sims ≫ n_bins)."
    )


if __name__ == "__main__":
    main()
