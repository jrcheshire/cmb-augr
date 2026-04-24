"""Produce MC-averaged post-NILC noise and Eq. 3.7 residual-template
spectra via the BROOM pipeline, ready for consumption by augr.

Pipeline (per simulation):
    get_input_data -> component_separation (NILC + GNILC)
                   -> estimate_residuals (GNILC maps through NILC weights)
                   -> _compute_spectra (anafast + mask)

After the MC loop, the per-realization Cls are averaged and the Carones 2025
Eq. 3.7 debiasing is applied:

    C_ell^{f,res} = <C_ell^{f_tilde,res}> - <C_ell^{n_tilde}>

Two npy products are written to data/broom_outputs/, each as an (n_bins, 2)
array whose columns are (ell_center, C_ell^BB) in uK_CMB^2:

    {tag}_nl_bb.npy    -- post-NILC noise (C_ell of 'noise_residuals')
    {tag}_tres_bb.npy  -- Eq. 3.7 debiased residual template
    {tag}_fgds_bb.npy  -- ground-truth fgds residual (diagnostic)

where `tag` encodes experiment / fg_model / compsep / mask / nsims.

Usage:
    conda run -n augr python scripts/broom_residual_template.py --nsims 50

Note: first run populates scripts/_broom_scratch/inputs/ with PySM3-generated
sims; several minutes at nside=64.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import broom
import healpy as hp
import numpy as np
from broom import (
    Configs,
    _compute_spectra,
    component_separation,
    estimate_residuals,
    get_input_data,
)
from broom.routines import _format_nsim


# ---------------------------------------------------------------------------
# Fixed configuration (constants; override via CLI where sensible)
# ---------------------------------------------------------------------------

NSIDE = 64
LMAX = 180
FWHM_OUT = 30.0
FOREGROUND_MODELS = ("d1", "s1")
EXPERIMENT = "LiteBIRD_PTEP"
MASK_TYPE_DEFAULT = "GAL60"

NEEDLET_CONFIG = [
    {"needlet_windows": "mexican"},
    {"width": 1.3},
    {"merging_needlets": [0, 14, 17, 19, 40]},
]
NEEDLET_WIDTH = NEEDLET_CONFIG[1]["width"]
NEEDLET_BANDS = NEEDLET_CONFIG[2]["merging_needlets"]

ILC_BIAS_NILC = 0.001
ILC_BIAS_GNILC = 0.01

FG_TAG = "".join(FOREGROUND_MODELS)

BROOM_ROOT = Path(broom.__file__).parent

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "_broom_scratch"
OUTPUTS_DIR = HERE.parent / "data" / "broom_outputs"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _needlet_merge_tag() -> str:
    """BROOM's needlet-band tag: e.g. 'j0j13_j14j16_j17j18_j19j39'."""
    bands = NEEDLET_BANDS
    return "_".join(
        f"j{lo}j{hi - 1}" for lo, hi in zip(bands[:-1], bands[1:])
    )


def _nilc_path() -> str:
    merge = _needlet_merge_tag()
    return (f"ilc_needlet_bias{ILC_BIAS_NILC}/mexican_B{NEEDLET_WIDTH}_{merge}"
            f"/cmb_reconstruction")


def _gnilc_root_dir() -> Path:
    """Discover the GNILC top-level directory dynamically.

    BROOM appends hyperparameter tags (e.g. '_m+1_nls1-2-3') to the
    directory name, so we find the unique gilc_* subdirectory after
    GNILC has run.
    """
    base = SCRATCH / "outputs" / EXPERIMENT / FG_TAG
    candidates = sorted(p for p in base.iterdir() if p.is_dir()
                        and p.name.startswith("gilc_"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one gilc_* directory under {base}, "
            f"found {[p.name for p in candidates]}")
    return candidates[0]


def _gnilc_path() -> str:
    merge = _needlet_merge_tag()
    return f"{_gnilc_root_dir().name}/mexican_B{NEEDLET_WIDTH}_{merge}"


def _fgres_template_subpath() -> str:
    """Residual template derived from GNILC total output (FG + noise)."""
    merge = _needlet_merge_tag()
    return (f"fgres_templates/{_gnilc_root_dir().name}_"
            f"mexican_B{NEEDLET_WIDTH}_{merge}")


def _fgres_template_noise_subpath() -> str:
    """Noise-only residual template (GNILC noise-residuals through NILC).

    The Eq. 3.7 subtrahend.
    """
    merge = _needlet_merge_tag()
    return (f"fgres_templates_noise/{_gnilc_root_dir().name}_"
            f"mexican_B{NEEDLET_WIDTH}_{merge}")


def _output_tag(mask_type: str, nsims: int) -> str:
    """Filename tag encoding the instrument/FG/method/mask/nsims."""
    # Lowercase, no punctuation problematic for file systems.
    return (f"{EXPERIMENT.lower()}_{FG_TAG}_nilc_"
            f"{mask_type.lower()}_{nsims:03d}sims")


# ---------------------------------------------------------------------------
# BROOM config assembly
# ---------------------------------------------------------------------------

def _base_config(nsims: int, mask_type: str) -> dict:
    sims = SCRATCH / "inputs" / EXPERIMENT
    return {
        # General
        "lmin": 2,
        "lmin_in": 2,
        "nside": NSIDE,
        "nside_in": NSIDE,
        "lmax": LMAX,
        "lmax_in": LMAX,
        "data_type": "alms",
        "verbose": True,
        "nsim_start": 0,
        "nsims": nsims,
        "foreground_models": list(FOREGROUND_MODELS),
        "experiments_file": str(BROOM_ROOT / "utils/experiments.yaml"),
        "experiment": EXPERIMENT,
        "units": "uK_CMB",
        "coordinates": "G",

        # Input sims
        "generate_input_foregrounds": True,
        "return_fgd_components": False,
        "bandpass_integrate": False,
        "generate_input_noise": True,
        "seed_noise": None,
        "data_splits": False,
        "only_splits": False,
        "generate_input_cmb": True,
        "seed_cmb": None,
        "cls_cmb_path": str(BROOM_ROOT / "utils/Cls_Planck2018_lensed_r0.fits"),
        "cls_cmb_new_ordered": True,
        "generate_input_data": True,
        "save_inputs": True,
        "pixel_window_in": False,
        "data_path": str(sims / "total" / f"total_alms_ns{NSIDE}_lmax{LMAX}"),
        "fgds_path": str(
            sims / "foregrounds" / FG_TAG
            / f"foregrounds_alms_ns{NSIDE}_lmax{LMAX}"
        ),
        "cmb_path": str(sims / "cmb" / f"cmb_alms_ns{NSIDE}_lmax{LMAX}"),
        "noise_path": str(sims / "noise" / f"noise_alms_ns{NSIDE}_lmax{LMAX}"),

        # Common compsep
        "fwhm_out": FWHM_OUT,
        "bring_to_common_resolution": True,
        "pixel_window_out": False,
        "mask_observations": None,
        "mask_covariance": None,
        "leakage_correction": None,
        "save_compsep_products": True,
        "return_compsep_products": False,
        "path_outputs": str(SCRATCH / "outputs" / EXPERIMENT / FG_TAG) + "/",
        "field_in": "TEB",
        "field_out": "B",

        # Spectra
        "delta_ell": 5,
        "ell_min_bpws": None,
        "spectra_comp": "anafast",
        "return_Dell": False,
        "field_cls_out": ["BB"],
        "return_spectra": False,
        "save_spectra": True,
        "save_mask": False,

        # The active compsep / residual / spectra blocks are attached in
        # main() at the right phase (needed to resolve the GNILC path
        # suffix after GNILC has actually run).
        "compsep": _compsep_blocks(),
        "compsep_residuals": [],
        "compute_spectra": [],
    }


def _compsep_blocks() -> list[dict]:
    """Two compsep blocks: NILC on scalar B, and GNILC on QU for FG maps.

    GNILC hyperparameters follow Carones 2025 Sec. 3.2 (m(n_hat) + 1 modes,
    full CMB deprojection except at the lowest needlet band).
    """
    return [
        {
            "method": "ilc",
            "domain": "needlet",
            "needlet_config": NEEDLET_CONFIG,
            "ilc_bias": ILC_BIAS_NILC,
            "reduce_ilc_bias": False,
            "b_squared": False,
            "adapt_nside": True,
            "save_needlets": True,
            "save_weights": True,
            "cov_noise_debias": [0.0, 0.0, 0.0, 0.0],
            "load_noise_covariance": False,
            "component_out": "cmb",
            "minimize_variance": False,
        },
        {
            "method": "gilc",
            "domain": "needlet",
            "needlet_config": NEEDLET_CONFIG,
            "ilc_bias": ILC_BIAS_GNILC,
            "reduce_ilc_bias": False,
            "b_squared": False,
            "adapt_nside": True,
            "save_needlets": True,
            "nuisance": ["noise", "cmb"],
            "load_nuisance_covariance": False,
            "depro_cmb": [None, 0.0, 0.0, 0.0],
            "m_bias": [0, 1, 1, 1],
            "cov_noise_debias": [0.0, 0.0, 0.0, 0.0],
            "load_noise_covariance": False,
        },
    ]


def _compsep_residuals_block() -> list[dict]:
    return [
        {
            "gilc_path": _gnilc_path(),
            "gilc_components": ["output_total", "noise_residuals"],
            "compsep_path": _nilc_path(),
            "field_in": "B",
            "adapt_nside": True,
        },
    ]


def _spectra_block(mask_type: str) -> list[dict]:
    return [
        {
            "path_method": _nilc_path(),
            "components_for_cls": [
                "output_total",
                "noise_residuals",
                "fgds_residuals",
                "output_cmb",
                _fgres_template_subpath(),
                _fgres_template_noise_subpath(),
            ],
            "mask_type": mask_type,
            "mask_path": "",
            "field_out": "B",
            "apodize_mask": None,
            "apodize_scale": 0.0,
            "smooth_tracer": 3.0,
            "fsky": 0.6 if mask_type == "GAL60" else 0.4,
            "nmt_purify_B": False,
            "nmt_purify_E": False,
        },
    ]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def run_all_sims(config: Configs) -> None:
    # TODO(multiprocess): this loop is embarrassingly parallel -- each
    # iteration is a self-contained (get_input_data -> component_separation
    # -> estimate_residuals) chain with per-sim filenames so there is no
    # write contention. BROOM's own `parallelize` config flag is read in
    # broom.configurations but otherwise unreferenced ("Not implemented yet"
    # per the demo config), so we would need to wrap it ourselves with
    # multiprocessing.Pool.
    # Gotchas for a future implementation:
    #   - Pin BLAS to one thread per worker (OMP_NUM_THREADS=1 /
    #     MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 set before Pool
    #     spawns) -- otherwise BLAS threads fight each other. Currently
    #     nsims=20 saturates ~3 cores; pinning + Pool(n=8..12) should
    #     give near-linear speedup.
    #   - The `config.compsep_residuals = _compsep_residuals_block()`
    #     line below mutates shared state as a BROOM workaround, so
    #     each worker needs its own Configs instance (fork-safe copy
    #     or per-worker construction).
    #   - compute_all_spectra / _compute_spectra below also loop over
    #     sims internally; parallelize there too for consistency.
    # Deferred until we move beyond ~20-sim exploratory runs; at nsims=100
    # (Carones baseline) this is the difference between ~25 min serial
    # and ~3 min parallel.
    for nsim in range(config.nsim_start, config.nsim_start + config.nsims):
        tag = _format_nsim(nsim)
        print(f"\n=== sim {tag} ===")
        print(f"[{tag}] get_input_data")
        data = get_input_data(config, nsim=tag)
        print(f"[{tag}] component_separation (NILC + GNILC)")
        component_separation(config, data, nsim=tag)
        # estimate_residuals mutates compsep_residuals in-place; rebuild
        # fresh each iteration so nsim_weights does not leak across sims.
        config.compsep_residuals = _compsep_residuals_block()
        print(f"[{tag}] estimate_residuals")
        estimate_residuals(config, nsim=tag)


def compute_all_spectra(config: Configs) -> None:
    print("\n=== compute_spectra across MC ===")
    _compute_spectra(config)


# ---------------------------------------------------------------------------
# Spectrum loading
# ---------------------------------------------------------------------------

def _spectrum_file(component_subpath: str, component_label: str,
                   nsim: int, mask_type: str) -> Path:
    tag = _format_nsim(nsim)
    base = (SCRATCH / "outputs" / EXPERIMENT / FG_TAG / _nilc_path()
            / "spectra" / mask_type / component_subpath)
    return (base / f"Cls_BB_{component_label}_{FWHM_OUT}acm_ns{NSIDE}"
                   f"_lmax{LMAX}_{tag}.fits")


def load_mc_spectra(component_subpath: str, component_label: str,
                    nsims: int, mask_type: str) -> np.ndarray:
    """Load per-realization BB spectra, stack shape (nsims, nbins)."""
    arrs = []
    for nsim in range(nsims):
        path = _spectrum_file(component_subpath, component_label, nsim,
                              mask_type)
        arrs.append(np.asarray(hp.read_cl(str(path))))
    return np.stack(arrs)


def _bandpower_ell_centres(n_bins: int, bin_width: int = 5,
                           lmin: int = 2) -> np.ndarray:
    """Reconstruct BROOM's bandpower centres from the binning convention."""
    ell_lo = np.arange(n_bins) * bin_width + lmin
    return ell_lo + (bin_width - 1) / 2.0


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------

def _save_product(out_path: Path, ell_centres: np.ndarray,
                  cl_bb: np.ndarray) -> None:
    """Write a 2-column (ell_center, C_ell^BB) array as .npy."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.column_stack([ell_centres, cl_bb])
    np.save(out_path, stacked)
    print(f"  wrote {out_path}  shape={stacked.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nsims", type=int, default=20,
                   help="Number of simulations (Carones 2025 uses 100; "
                        "default: 20)")
    p.add_argument("--mask", type=str, default=MASK_TYPE_DEFAULT,
                   choices=["GAL40", "GAL60", "GAL70", "GAL80"],
                   help="Planck galactic mask to use for spectra "
                        "(default: GAL60)")
    p.add_argument("--skip-compsep", action="store_true",
                   help="Skip sim generation + NILC/GNILC runs; reuse "
                        "existing products on disk (spectra will still "
                        "be recomputed).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)

    config = Configs(config=_base_config(nsims=args.nsims, mask_type=args.mask))

    if not args.skip_compsep:
        print(f"Phase 1: NILC + GNILC + estimate_residuals, nsims={args.nsims}")
        run_all_sims(config)

    print("\nResolving GNILC path and attaching spectra block ...")
    config.compute_spectra = _spectra_block(args.mask)

    print("\nPhase 2: compute spectra across MC")
    compute_all_spectra(config)

    print("\nPhase 3: MC-average + Eq. 3.7 debiasing")
    tres_raw = load_mc_spectra(
        _fgres_template_subpath(), "fgres_templates", args.nsims, args.mask)
    tres_noise = load_mc_spectra(
        _fgres_template_noise_subpath(), "fgres_templates_noise",
        args.nsims, args.mask)
    nl_post = load_mc_spectra(
        "noise_residuals", "noise_residuals", args.nsims, args.mask)
    fgds_res = load_mc_spectra(
        "fgds_residuals", "fgds_residuals", args.nsims, args.mask)

    tres_mean = tres_raw.mean(0) - tres_noise.mean(0)
    nl_mean = nl_post.mean(0)
    fgds_mean = fgds_res.mean(0)

    n_bins = tres_mean.shape[-1]
    ell_centres = _bandpower_ell_centres(n_bins)

    print("\nPhase 4: write output artifacts")
    tag = _output_tag(args.mask, args.nsims)
    _save_product(OUTPUTS_DIR / f"{tag}_nl_bb.npy", ell_centres, nl_mean)
    _save_product(OUTPUTS_DIR / f"{tag}_tres_bb.npy", ell_centres, tres_mean)
    _save_product(OUTPUTS_DIR / f"{tag}_fgds_bb.npy", ell_centres, fgds_mean)

    print("\nDone.")


if __name__ == "__main__":
    main()
