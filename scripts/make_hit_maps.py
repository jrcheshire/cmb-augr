"""make_hit_maps.py - generate per-channel L2 hit-map FITS files for BROOM.

Writes one HEALPix FITS file per channel of the requested BROOM
experiment, named `{prefix}_{channel_tag}.fits` to match BROOM's
`path_hits_maps` convention. All channels share the same analytic
L2 hit map (envelope model from `augr.hit_maps.l2_hit_map`); per-
channel feedhorn offsets are not modeled in v1.

Usage:
    conda run -n augr python scripts/make_hit_maps.py \\
        --prefix data/hit_maps/litebird_ptep_l2 \\
        --experiment LiteBIRD_PTEP \\
        --nside 64

Optional:
    --spin-angle-deg 50.0       # alpha
    --precession-angle-deg 45.0 # beta
    --coord G                   # or E, C

Downstream: pass the prefix to `broom_residual_template.py
--hits-prefix ...` and BROOM will pick up the per-channel files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import broom
import healpy as hp
import numpy as np
from broom import Configs

from augr.hit_maps import l2_hit_map


BROOM_EXPERIMENTS_YAML = (
    Path(broom.__file__).parent / "utils" / "experiments.yaml"
)


def _channel_tags(experiment: str, nside: int) -> list[str]:
    """Channel tags from BROOM's InstrumentConfig for the given experiment.

    Uses BROOM's own tag-generation so names match what
    `broom_residual_template.py` and the compsep pipeline expect
    (handles duplicate frequencies via `aGHz`/`bGHz` suffixes).
    """
    cfg = Configs(config={
        "experiment": experiment,
        "experiments_file": str(BROOM_EXPERIMENTS_YAML),
        "nside": nside, "nside_in": nside,
        "lmax": 2 * nside, "lmax_in": 2 * nside,
        "foreground_models": ["d1"], "data_type": "alms",
        "units": "uK_CMB", "coordinates": "G", "nsims": 1,
        "generate_input_data": False, "generate_input_cmb": False,
        "generate_input_foregrounds": False, "generate_input_noise": False,
        "bandpass_integrate": False,
    })
    return list(cfg.instrument.channels_tags)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prefix", type=str, required=True,
                   help="Output path prefix; files written as "
                        "<prefix>_<channel_tag>.fits")
    p.add_argument("--experiment", type=str, default="LiteBIRD_PTEP",
                   help="Name of the BROOM experiment (key in "
                        "broom/utils/experiments.yaml). Default: "
                        "LiteBIRD_PTEP")
    p.add_argument("--nside", type=int, default=64,
                   help="HEALPix nside. Default: 64")
    p.add_argument("--spin-angle-deg", type=float, default=50.0,
                   help="Boresight-to-spin-axis angle alpha, in deg. "
                        "Default: 50 (LiteBIRD-like envelope)")
    p.add_argument("--precession-angle-deg", type=float, default=45.0,
                   help="Spin-axis-to-antisun angle beta, in deg. "
                        "Default: 45 (LiteBIRD-like envelope)")
    p.add_argument("--coord", type=str, default="G", choices=["G", "E", "C"],
                   help="Output coordinate frame. Default: G (galactic, "
                        "to match BROOM's default coordinates='G')")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing FITS files.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    tags = _channel_tags(args.experiment, args.nside)
    print(f"experiment={args.experiment}: {len(tags)} channels")

    hits = l2_hit_map(
        nside=args.nside,
        spin_angle_deg=args.spin_angle_deg,
        precession_angle_deg=args.precession_angle_deg,
        coord=args.coord,
    )
    surveyed_frac = float((hits > 0).mean())
    print(f"l2_hit_map: nside={args.nside} coord={args.coord} "
          f"alpha={args.spin_angle_deg} beta={args.precession_angle_deg}")
    print(f"  surveyed f_sky={surveyed_frac:.4f}, "
          f"min={hits[hits > 0].min():.3g}, max={hits.max():.3g}")

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        out = prefix.with_name(prefix.name + f"_{tag}.fits")
        hp.write_map(
            str(out), hits, overwrite=args.overwrite,
            column_names=["HITS"], dtype=np.float32,
        )
    print(f"wrote {len(tags)} files to {prefix.parent}/")


if __name__ == "__main__":
    main()
