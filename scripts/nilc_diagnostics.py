"""nilc_diagnostics.py — intrinsic validation figure for the differentiable NILC.

Runs the empirical needlet ILC on a small multi-band PySM sim and produces the
four intrinsic diagnostics (no external truth required):

  1. Post-NILC noise N_ell^BB vs the analytic inverse-variance ILC floor.
  2. Residual-foreground C_ell^BB vs the input foreground (suppression factor).
  3. CMB transfer T_ell = cleaned-CMB x common-res-CMB / auto  (signal loss).
  4. Leakage correlation rho_ell between the residual and the input foreground.

Also prints sigma(r) (baseline / A_res flat / A_res Gaussian) and the FG-leakage
bias Delta r from the Fisher wiring. The band list / beams here are illustrative;
the aperture sweep that turns these into D_min is the JPL-side Stage-6 driver.

Usage:
    pixi run python scripts/nilc_diagnostics.py --fg-model d1s1 --out nilc_diag.png
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.hit_maps import l2_hit_map
from augr.nilc import common_resolution_b_alm, nilc_clean
from augr.nilc_forecast import (
    analytic_mv_noise_floor,
    cl_bb,
    nilc_forecast,
    nilc_leakage_correlation,
    nilc_spectra,
)
from augr.spectra import CMBSpectra

# Illustrative wide-band space-mission-like configuration (not a study design).
FREQS = (30.0, 44.0, 95.0, 150.0, 280.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 12.0)  # arcmin
W_INV = (2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 1.5e-4)  # uK^2 sr per band


def _build(fg_model: str, r_in: float, nside: int, lmax: int, seed: int, hit: jnp.ndarray):
    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=r_in,
        nside=nside,
        lmax=lmax,
        fg_model=fg_model,
        cmb_seed=seed,
    )
    total = assemble_band_maps(sky, jnp.asarray(W_INV), hit, noise_key=jax.random.PRNGKey(seed))
    noise = total - sky.cmb_qu - sky.fg_qu
    return total, noise, sky.fg_qu, sky.cmb_qu


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", default="d1s1", choices=["d1s1", "d10s5"])
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--r-in", type=float, default=0.0)
    p.add_argument("--ell-max", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--uniform-hits",
        action="store_true",
        help="use a uniform hit map instead of the default L2 anisotropic scan depth",
    )
    p.add_argument("--out", default="nilc_diagnostics.png")
    args = p.parse_args()

    plt.switch_backend("Agg")  # headless render

    npix = 12 * args.nside**2
    hit = jnp.ones(npix) if args.uniform_hits else jnp.asarray(l2_hit_map(args.nside, coord="G"))
    total, noise, fg, cmb = _build(args.fg_model, args.r_in, args.nside, args.lmax, args.seed, hit)
    res = nilc_clean(total, BEAMS, lmax=args.lmax, nside=args.nside)
    spec = nilc_spectra(res, total_qu=total, noise_qu=noise, fg_qu=fg, cmb_qu=cmb, f_sky=1.0)

    ells = spec.ells
    floor = analytic_mv_noise_floor(W_INV, BEAMS, args.lmax)
    fg_common, _ = common_resolution_b_alm(
        fg, BEAMS, lmax=args.lmax, nside=args.nside, n_iter=res.n_iter
    )
    cl_fg_input = cl_bb(fg_common[0], args.lmax)
    _, rho = nilc_leakage_correlation(res, fg)

    out = nilc_forecast(
        spec,
        f_sky=1.0,
        r_fid=args.r_in,
        ell_min=2,
        ell_max=args.ell_max,
        delta_ell=10,
        ell_per_bin_below=30,
    )

    sel = ells >= 2
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"NILC intrinsic diagnostics ({args.fg_model}, nside={args.nside})")

    ax[0, 0].loglog(ells[sel], spec.nl_post[sel], label="post-NILC noise")
    ax[0, 0].loglog(ells[sel], floor[sel], "--", label="analytic MV floor")
    ax[0, 0].set(xlabel=r"$\ell$", ylabel=r"$N_\ell^{BB}$ [$\mu$K$^2$]", title="post-NILC noise")
    ax[0, 0].legend()

    ax[0, 1].loglog(ells[sel], cl_fg_input[sel], label="input FG (lowest band)")
    ax[0, 1].loglog(ells[sel], np.abs(spec.cl_residual_fg[sel]), label="residual FG")
    ax[0, 1].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{BB}$ [$\mu$K$^2$]", title="FG suppression")
    ax[0, 1].legend()

    ax[1, 0].semilogx(ells[sel], spec.transfer[sel])
    ax[1, 0].axhline(1.0, color="k", lw=0.5)
    ax[1, 0].set(xlabel=r"$\ell$", ylabel=r"$T_\ell$", title="CMB transfer", ylim=(0.9, 1.1))

    ax[1, 1].semilogx(ells[sel], rho[sel])
    ax[1, 1].axhline(0.0, color="k", lw=0.5)
    ax[1, 1].set(xlabel=r"$\ell$", ylabel=r"$\rho_\ell$", title="residual x input-FG", ylim=(-1, 1))

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)

    print("=" * 60)
    print(f"  NILC diagnostics  fg={args.fg_model}  nside={args.nside}  lmax={args.lmax}")
    print("=" * 60)
    band = (ells >= 2) & (ells <= 30)
    print(
        f"  FG suppression (low-l):     {np.mean(cl_fg_input[band]) / np.mean(np.abs(spec.cl_residual_fg[band])):.1f}x"
    )
    print(f"  transfer (mean over fit):   {out['transfer_mean']:.4f}")
    print(f"  sigma(r) baseline:          {out['sigma_r_baseline']:.3e}")
    print(f"  sigma(r) A_res flat:        {out['sigma_r_flat']:.3e}")
    print(f"  sigma(r) A_res Gaussian:    {out['sigma_r_gauss']:.3e}")
    print(f"  Delta r (debias-OFF):       {out['delta_r']:.3e}")
    print(f"  wrote figure -> {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
