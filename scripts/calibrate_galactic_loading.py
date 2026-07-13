"""Calibrate the Galactic dust + synchrotron optical-loading normalizations.

One-time, offline. Computes the science-region (GAL070, f_sky ≈ 0.70) mean of
the Planck component-separated Galactic emission products and prints a
config-ready block of constants for ``augr.config.GALACTIC_LOADING``. The
runtime ``extra_loading`` (``augr.loading.make_galactic_extra_loading``) then
reads those baked constants only — it never touches a map.

Sources (all peer-reviewed Planck products; the loading uses TOTAL-INTENSITY
brightness, distinct from the polarized C_ℓ foreground amplitudes):

- **Dust** — Planck GNILC dust model (the product PySM d10/d11/d12 are built
  on): τ_353 (Opacity), T_d (Temperature), β_d (Spectral-Index) maps, NSIDE
  2048, RING. τ_353 is dimensionless at NU_REF = 353 GHz.
- **Synchrotron** — Planck 2015 Commander synchrotron amplitude I_ML [µK_RJ]
  at NU_REF = 408 MHz, NSIDE 256, NESTED. Extrapolated to band with
  β_s = −3.1 (a long lever arm, but the term is negligible in-band).
- **Region** — Planck HFI galactic-plane mask, GAL070 column (f_sky 0.7005),
  matching the probe-study fiducial f_sky = 0.7.

The GNILC Temperature + Spectral-Index maps are NOT shipped with the study;
fetch them once from the Planck Legacy Archive (see ``--help`` for URLs) and
pass their paths.

Usage:
    pixi run python scripts/calibrate_galactic_loading.py \\
        --tau  <...>/planck/COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits \\
        --tdust <...>/COM_CompMap_Dust-GNILC-Model-Temperature_2048_R2.01.fits \\
        --beta  <...>/COM_CompMap_Dust-GNILC-Model-Spectral-Index_2048_R2.01.fits \\
        --sync  <...>/COM_CompMap_Synchrotron-commander_0256_R2.00.fits \\
        --mask  <...>/HFI_Mask_GalPlane-apo0_2048_R2.00.fits
"""

from __future__ import annotations

import argparse

import healpy as hp
import numpy as np
from astropy.io import fits

# Verified local paths (see the map-inspection in the issue-#43 work).
DEF_TAU = (
    "/Users/jamie/spherex/SPHEREx-L4-Cosmology-Pipeline/data/dustmaps/planck/"
    "COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits"
)
DEF_SYNC = (
    "/Users/jamie/spherex/astrophysical_templates/cache/"
    "COM_CompMap_Synchrotron-commander_0256_R2.00.fits"
)
DEF_MASK = "/Users/jamie/Downloads/HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
# PLA (must be downloaded once):
#   https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=<filename>
DEF_TDUST = "COM_CompMap_Dust-GNILC-Model-Temperature_2048_R2.01.fits"
DEF_BETA = "COM_CompMap_Dust-GNILC-Model-Spectral-Index_2048_R2.01.fits"

BAD = -1.6375e30  # Planck BAD_DATA sentinel


def _read_field(path: str, candidates: tuple[str, ...], nest_out: bool):
    """Read the first matching named column, else column 0; return RING/NESTED
    as requested. Prints the resolved field name for provenance."""
    names = [c.name.upper() for c in fits.open(path)[1].columns]
    field = 0
    resolved = names[0]
    for cand in candidates:
        if cand.upper() in names:
            field = names.index(cand.upper())
            resolved = cand.upper()
            break
    m = hp.read_map(path, field=field, nest=nest_out)
    print(f"  {path.split('/')[-1]}: field '{resolved}' (cols={names})")
    return np.asarray(m)


def _region_mean(vals: np.ndarray, mask: np.ndarray) -> float:
    """Mean over mask==1 pixels, dropping BAD_DATA / non-finite."""
    keep = (mask > 0) & np.isfinite(vals) & (vals > BAD * 0.5)
    return float(np.mean(vals[keep]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tau", default=DEF_TAU)
    p.add_argument("--tdust", default=DEF_TDUST)
    p.add_argument("--beta", default=DEF_BETA)
    p.add_argument("--sync", default=DEF_SYNC)
    p.add_argument("--mask", default=DEF_MASK)
    p.add_argument(
        "--gal-col", type=int, default=3, help="Mask column index; 3 = GAL070 (f_sky 0.70)."
    )
    p.add_argument("--sync-beta", type=float, default=-3.1)
    args = p.parse_args()

    print("Reading maps (field resolution shown for provenance):")
    # GNILC dust products: RING, NSIDE 2048.
    tau = _read_field(args.tau, ("TAU353",), nest_out=False)
    tdust = _read_field(args.tdust, ("TEMP", "TEMPERATURE"), nest_out=False)
    beta = _read_field(args.beta, ("BETA", "SPECTRAL-INDEX"), nest_out=False)
    nside_dust = hp.npix2nside(tau.size)

    # GAL mask (NESTED) -> RING to match the dust maps.
    mask_nest = hp.read_map(args.mask, field=args.gal_col, nest=True)
    mask_ring = hp.reorder(mask_nest, n2r=True)
    if hp.npix2nside(mask_ring.size) != nside_dust:
        mask_ring = hp.ud_grade(mask_ring, nside_dust, order_in="RING", order_out="RING")
    mask_ring = (mask_ring > 0.5).astype(float)
    fsky = float(np.mean(mask_ring))

    # Commander synchrotron I_ML [µK_RJ], NESTED NSIDE 256 -> RING, mask ud_grade.
    sync_uK = _read_field(args.sync, ("I_ML", "I_MEAN"), nest_out=False)
    nside_sync = hp.npix2nside(sync_uK.size)
    mask_sync = hp.ud_grade(mask_ring, nside_sync, order_in="RING", order_out="RING")
    mask_sync = (mask_sync > 0.5).astype(float)

    tau_353 = _region_mean(tau, mask_ring)
    dust_beta = _region_mean(beta, mask_ring)
    dust_T = _region_mean(tdust, mask_ring)
    sync_T_ref_K = _region_mean(sync_uK, mask_sync) * 1e-6  # µK -> K

    # Provenance / cross-checks.
    print(f"\nRegion: GAL{args.gal_col}-col, f_sky = {fsky:.4f}")
    print(
        f"  tau_353  full-sky mean/median = "
        f"{np.mean(tau[np.isfinite(tau)]):.3e} / "
        f"{np.median(tau[np.isfinite(tau)]):.3e}"
    )
    print(f"  tau_353  GAL070 mean          = {tau_353:.3e}")
    print("  (Planck Int. XLVIII 2016 high-lat diffuse τ_353 ~ 1e-6..1e-5)")

    print("\n# --- paste into augr/config.py: GALACTIC_LOADING ---")
    print(f"TAU_353         = {tau_353:.4e}   # GNILC Opacity, GAL070 mean")
    print(f"DUST_BETA       = {dust_beta:.4f}     # GNILC Spectral-Index, GAL070 mean")
    print(f"DUST_T_K        = {dust_T:.4f}    # GNILC Temperature [K], GAL070 mean")
    print(f"SYNC_T_REF_K    = {sync_T_ref_K:.4e}   # Commander I_ML [K_RJ], GAL070 mean")
    print("SYNC_NU_REF_GHZ = 0.408      # Commander NU_REF = 408 MHz")
    print(f"SYNC_BETA       = {args.sync_beta}       # RJ index (FIDUCIAL_BK15 beta_sync)")


if __name__ == "__main__":
    main()
