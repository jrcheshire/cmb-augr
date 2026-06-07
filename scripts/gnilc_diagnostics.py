"""gnilc_diagnostics.py — GNILC residual template vs the oracle, and σ(r)/Δr.

Builds a small multi-band sim, forms the foreground-residual template two ways —
(a) the *oracle* (true ``fg_qu`` through the NILC weights, what
``augr.nilc_forecast`` uses) and (b) the data-driven in-house **GNILC**
(``augr.gnilc.gnilc_residual_template``) — and runs the ``A_res`` Fisher with each,
printing σ(r) (baseline / A_res flat / A_res Gaussian) and the FG-leakage bias Δr.

The GNILC template is data-driven (no knowledge of the true foregrounds), so it is the
BROOM-parity replacement for the oracle. With ``m_bias=1`` (Carones 2025) the global
GNILC template tracks the oracle to O(1) across ℓ; pure AIC (``--m-bias 0``) under-
selects the foreground subspace and the template decouples in shape (see
``augr.gnilc`` module docstring).

Usage:
    pixi run python scripts/gnilc_diagnostics.py --fg-model d1s1 --nside 64
"""

from __future__ import annotations

import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.gnilc import alm2cl, gnilc_residual_template
from augr.instrument import beam_bl
from augr.nilc import nilc_clean
from augr.nilc_forecast import nilc_forecast, nilc_spectra
from augr.spectra import CMBSpectra

# Illustrative wide-band space-mission-like configuration (not a study design).
FREQS = (30.0, 44.0, 95.0, 150.0, 280.0, 353.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 12.0, 10.0)  # arcmin
W_INV = jnp.array([2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 1.5e-4, 4.0e-4])  # μK²·sr per band


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", default="d1s1", choices=["d1s1", "d10s5", "d10s6"])
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--r-in", type=float, default=0.0)
    p.add_argument("--ell-max", type=int, default=100)
    p.add_argument("--m-bias", type=int, default=1, help="GNILC AIC offset (Carones uses 1)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

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

    # NILC clean + spectra (post-NILC noise + the *oracle* FG residual template).
    res = nilc_clean(total, BEAMS, lmax=args.lmax, nside=args.nside)
    spec = nilc_spectra(res, total_qu=total, noise_qu=noise, fg_qu=sky.fg_qu, cmb_qu=sky.cmb_qu)

    # In-house GNILC residual template (data-driven), on the same ℓ grid + beam.
    ells_g, cl_g, gres = gnilc_residual_template(
        total,
        sky.cmb_qu,
        noise,
        BEAMS,
        lmax=args.lmax,
        nside=args.nside,
        m_bias=args.m_bias,
        return_result=True,
    )
    spec_gnilc = dataclasses.replace(spec, cl_residual_fg=np.asarray(cl_g))

    fc_kw = dict(
        f_sky=1.0,
        r_fid=args.r_in,
        ell_min=2,
        ell_max=args.ell_max,
        delta_ell=10,
        ell_per_bin_below=30,
    )
    out_oracle = nilc_forecast(spec, **fc_kw)
    out_gnilc = nilc_forecast(spec_gnilc, **fc_kw)

    # Template shape agreement vs the oracle (amplitude is absorbed by A_res).
    ell = np.asarray(ells_g)
    bl2 = np.maximum(np.asarray(beam_bl(ell.astype(float), res.common_fwhm_arcmin)) ** 2, 1e-8)
    oracle = np.asarray(alm2cl(gres.nilc_clean_alm(sky.fg_qu), args.lmax)) / bl2
    gnilc = np.asarray(cl_g)

    print("=" * 66)
    print(
        f"  GNILC diagnostics  fg={args.fg_model}  nside={args.nside}  "
        f"lmax={args.lmax}  m_bias={args.m_bias}"
    )
    print("=" * 66)
    print(f"  GNILC m per needlet band:   {tuple(int(x) for x in gres.m_per_band)}")
    print("  template/oracle ratio by ℓ-band [GNILC ÷ oracle]:")
    for lo, hi in [(4, 10), (10, 20), (20, 40), (40, 70), (70, 110)]:
        b = (ell >= lo) & (ell < hi)
        print(f"      ℓ {lo:3d}-{hi:3d}:  {gnilc[b].mean() / oracle[b].mean():6.3f}")
    print("  " + "-" * 62)
    print(f"  {'metric':<26}{'oracle':>16}{'GNILC':>16}")
    for key, label in [
        ("sigma_r_baseline", "σ(r) baseline"),
        ("sigma_r_flat", "σ(r) A_res flat"),
        ("sigma_r_gauss", "σ(r) A_res Gaussian"),
        ("delta_r", "Δr (debias-OFF)"),
    ]:
        print(f"  {label:<26}{out_oracle[key]:>16.3e}{out_gnilc[key]:>16.3e}")
    print("=" * 66)


if __name__ == "__main__":
    main()
