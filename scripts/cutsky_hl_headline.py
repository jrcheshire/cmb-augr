"""cutsky_hl_headline.py — honest non-Gaussian σ(r) for the cut-sky map-based forecast.

The Bayesian companion to ``scripts/cutsky_headline.py``: take the *same* cut-sky
masked-Wiener Monte-Carlo ensemble (one spin-2 Q/U clean + three masked-Wiener solves
per sim) and, instead of reading σ(r) off a Gaussian-Fisher on the MC covariance, feed
the ensemble into augr's inference layer to get the **Hamimeche-Lewis posterior σ(r)** —
retiring the documented Knox/Gaussian few-to-ten-percent optimism at the reionization
bump (``reference_knox_gaussian_likelihood_bias``).

It reports σ(r) four ways — Gaussian-Fisher (the existing Knox baseline), Gaussian-NUTS
(the parity check), HL-NUTS (the headline), HL-profile (the sampling-free cross-check) —
plus the :func:`augr.likelihood.bandpower_ks` calibration verdict on the realized
cleaned-bandpower ensemble, which *decides* whether the analytic HL is adequate or the
ensemble-calibrated form is needed.

The MC ensemble is the cost driver; it needs the ``[compsep]`` extra (ducc0; pysm3 for
``--fg-model``) and the ``[sampling]`` extra (blackjax + optax) for the NUTS / MLE /
profile paths. Use ``--workers`` once the cleaner is picklable, else run serial.

Usage:
    pixi run python scripts/cutsky_hl_headline.py --nside 64 --f-sky 0.6 --nsims-cov 80 --fg-model d1s1
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import jax.numpy as jnp

from augr import masking as mk
from augr.cleaning import nilc_cleaner
from augr.hit_maps import l2_hit_map
from augr.likelihood import hl_forecast_from_cutsky_mc
from augr.pipeline import (
    ForecastConfig,
    ResidualTemplateSource,
    SpectrumSource,
    clean_sky,
    cutsky_mc_bandpowers,
)

# Same illustrative wide-band configuration as scripts/cutsky_headline.py so the two
# headlines are directly comparable (LiteBIRD-PTEP-ish; not a study design).
FREQS = (30.0, 44.0, 95.0, 150.0, 280.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 12.0)  # arcmin
W_INV = (2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 1.5e-4)  # uK^2 sr per band


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", default="d1s1", choices=["d1s1", "d10s5", "none"])
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--f-sky", type=float, default=0.6)
    p.add_argument("--r-in", type=float, default=0.0)
    p.add_argument("--ell-max", type=int, default=120)
    p.add_argument("--delta-ell", type=int, default=15)
    p.add_argument("--ell-per-bin-below", type=int, default=10)
    p.add_argument("--nsims-cov", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--num-warmup", type=int, default=800)
    p.add_argument("--num-samples", type=int, default=1500)
    p.add_argument("--no-profile", action="store_true", help="skip the HL profile-σ cross-check")
    p.add_argument(
        "--mc-calibrated",
        action="store_true",
        help="also sample the ensemble-calibrated offset-lognormal cross-check",
    )
    p.add_argument(
        "--uniform-hits", action="store_true", help="uniform hits instead of L2 scan depth"
    )
    p.add_argument("--residual", default="oracle", choices=["oracle", "gnilc"])
    args = p.parse_args()

    fg_model = None if args.fg_model == "none" else args.fg_model
    npix = 12 * args.nside**2
    hit = jnp.ones(npix) if args.uniform_hits else jnp.asarray(l2_hit_map(args.nside, coord="G"))
    mask = mk.galactic_mask(args.nside, args.f_sky)
    fsky_real = mk.f_sky_of(mask)

    cfg = ForecastConfig(
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
        residual_source=ResidualTemplateSource[args.residual.upper()],
        r_fid=args.r_in,
        ell_min=2,
        ell_max=args.ell_max,
        delta_ell=args.delta_ell,
        ell_per_bin_below=args.ell_per_bin_below,
        spectrum_source=SpectrumSource.CUTSKY_MC,
        mask=mask,
        n_sims_mc=args.nsims_cov,
        base_seed_mc=args.seed,
        mc_workers=args.workers,
    )
    # f_sky is the realized <mask> (matches pipeline.run_forecast's CUTSKY_MC branch).
    cfg = replace(cfg, f_sky=fsky_real)

    # One representative clean for the residual template + the MC ensemble.
    cleaned = clean_sky(cfg)
    mc, template_ells, template_cl = cutsky_mc_bandpowers(cleaned, cfg)

    res = hl_forecast_from_cutsky_mc(
        mc,
        template_ells=template_ells,
        template_cl=template_cl,
        f_sky=cfg.f_sky,
        r_fid=cfg.r_fid,
        ell_min=cfg.ell_min,
        ell_max=cfg.ell_max,
        delta_ell=cfg.delta_ell,
        ell_per_bin_below=cfg.ell_per_bin_below,
        n_chains=args.n_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        profile=not args.no_profile,
        run_mc_calibrated=args.mc_calibrated,
        seed=args.seed,
    )

    print("=" * 70)
    print(
        f"  cut-sky HL σ(r) headline   fg={args.fg_model}  nside={args.nside}  "
        f"lmax={args.lmax}  f_sky={fsky_real:.3f}  nsims={args.nsims_cov}"
    )
    print("=" * 70)
    print(f"  {'Gaussian-Fisher (Knox baseline)':<34}{res.sigma_r_gauss_fisher:>12.3e}")
    print(
        f"  {'Gaussian-NUTS  (parity check)':<34}{res.sigma_r_gauss_nuts:>12.3e}"
        f"   x{res.sigma_r_gauss_nuts / res.sigma_r_gauss_fisher:5.3f} vs Fisher"
    )
    print(
        f"  {'HL-NUTS        (headline)':<34}{res.sigma_r_hl_nuts:>12.3e}"
        f"   x{res.hl_widening:5.3f} vs Fisher"
    )
    if not args.no_profile:
        print(f"  {'HL-profile     (sampling-free)':<34}{res.sigma_r_hl_profile:>12.3e}")
    if res.sigma_r_mc_calibrated_nuts is not None:
        print(f"  {'MC-calibrated  (cross-check)':<34}{res.sigma_r_mc_calibrated_nuts:>12.3e}")
    print("-" * 70)
    ks = res.ks
    print(
        f"  KS verdict: recommend={ks['recommend']!r}  "
        f"(gauss_rejected_bump={ks['gauss_rejected_bump']}, "
        f"chi2_rejected_bump={ks['chi2_rejected_bump']}, "
        f"hl_better_bump={ks['hl_better_bump']})"
    )
    print(
        f"  converged: gaussian={res.converged_gauss}  hl={res.converged_hl}   "
        f"free={list(res.free_names)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
