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
import json
import string
import sys
from pathlib import Path

import broom
import healpy as hp
import numpy as np
import yaml
from broom import (
    Configs,
    _compute_spectra,
    component_separation,
    estimate_residuals,
    get_input_data,
)
from broom.routines import _format_nsim

from augr.hit_maps import mean_pixel_rescale_factor


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


def _output_tag(mask_type: str, nsims: int, hits_prefix: str | None = None,
                knee_config: str | None = None,
                cov_noise_debias_factor: float = 0.0) -> str:
    """Filename tag encoding the instrument/FG/method/mask/noise/nsims.

    White-noise + debias=0 tag is unchanged from the pre-noise-realism
    behavior.  Anisotropic noise contributes `_l2`; 1/f contributes
    `_1f`; non-zero covariance-noise debias contributes `_debX.X`.
    """
    parts = [EXPERIMENT.lower(), FG_TAG, "nilc", mask_type.lower()]
    if hits_prefix is not None:
        parts.append("l2")
    if knee_config is not None:
        parts.append("1f")
    if cov_noise_debias_factor > 0.0:
        parts.append(f"deb{cov_noise_debias_factor:g}")
    parts.append(f"{nsims:03d}sims")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Instrument override assembly (hit maps + 1/f noise)
# ---------------------------------------------------------------------------

def _experiment_channel_tags(experiment: str,
                             frequencies: list[float]) -> list[str]:
    """Reproduce BROOM's InstrumentConfig._generate_channel_tags."""
    freqs = np.asarray(frequencies)
    unique, counts = np.unique(freqs, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    label_pool = {u: list(string.ascii_lowercase[:c])
                  for u, c in zip(unique.tolist(), counts.tolist())}
    tags: list[str] = []
    for f in freqs:
        f = float(f)
        if count_map[f] == 1:
            tags.append(f"{f}GHz")
        else:
            tags.append(f"{f}{label_pool[f].pop(0)}GHz")
    return tags


def _load_knee_config(path: Path, channel_tags: list[str]) -> tuple[list[float], list[float]]:
    """Load per-channel 1/f config from JSON aligned to channel order.

    JSON schema: {channel_tag: {"ell_knee": float, "alpha_knee": float}}.
    Must cover every channel -- raises ValueError on gaps.  Convention:
    alpha_knee < 0 for 1/f that rises at low ell; BROOM applies
    N_ell *= 1 + (ell / ell_knee)^alpha_knee.
    """
    raw = json.loads(Path(path).read_text())
    missing = [t for t in channel_tags if t not in raw]
    if missing:
        shown = ", ".join(missing[:3])
        extra = " ..." if len(missing) > 3 else ""
        raise ValueError(
            f"knee-config {path} missing entries for channels: "
            f"[{shown}{extra}]. Provide ell_knee/alpha_knee for every "
            "channel, or drop --knee-config entirely for white noise."
        )
    ell_knee = [float(raw[t]["ell_knee"]) for t in channel_tags]
    alpha_knee = [float(raw[t]["alpha_knee"]) for t in channel_tags]
    return ell_knee, alpha_knee


def _build_instrument_override(
    experiment: str,
    hits_prefix: str | None = None,
    knee_config_path: str | None = None,
) -> tuple[dict, list[str]] | tuple[None, list[str]]:
    """Build an instrument-dict override with hit-map + 1/f knobs applied.

    Returns (None, channel_tags) when neither override is set -- the
    caller should stay on the experiment-YAML path.  Returns
    (instrument_dict, channel_tags) otherwise; the dict is ready to
    drop into `config["instrument"]`.

    Hit-map normalization: `depth_I` and `depth_P` are divided by
    `augr.hit_maps.mean_pixel_rescale_factor(hits)` so the sky-averaged
    pixel noise variance equals the YAML-spec value.  BROOM's
    internal max=1 normalization then leaves the ecliptic poles
    deeper than spec and the equator shallower.
    """
    yaml_path = BROOM_ROOT / "utils" / "experiments.yaml"
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)
    if experiment not in yaml_data:
        raise ValueError(f"experiment {experiment!r} not in {yaml_path}")
    inst = dict(yaml_data[experiment])

    tags = _experiment_channel_tags(experiment, inst["frequency"])

    if hits_prefix is None and knee_config_path is None:
        return None, tags

    if hits_prefix is not None:
        first_fits = Path(f"{hits_prefix}_{tags[0]}.fits")
        if not first_fits.exists():
            raise FileNotFoundError(
                f"hit map not found: {first_fits}. "
                "Generate first via scripts/make_hit_maps.py."
            )
        hits = hp.read_map(str(first_fits))
        k = mean_pixel_rescale_factor(hits)
        print(f"  hit-map rescale factor k = {k:.4f} "
              f"(depth_I/P divided by k for sky-average normalization)")
        inst["depth_I"] = [float(d) / k for d in inst["depth_I"]]
        inst["depth_P"] = [float(d) / k for d in inst["depth_P"]]
        inst["path_hits_maps"] = hits_prefix

    if knee_config_path is not None:
        ell_knee, alpha_knee = _load_knee_config(Path(knee_config_path), tags)
        inst["ell_knee"] = ell_knee
        inst["alpha_knee"] = alpha_knee

    return inst, tags


# ---------------------------------------------------------------------------
# BROOM config assembly
# ---------------------------------------------------------------------------

def _input_cache_tag(has_hits: bool, has_knee: bool) -> str:
    """Suffix for cached noise/total alms paths.

    Keeps "alms" as the default so the white-noise cache directory
    built by prior runs stays valid; hit-map and 1/f runs go to
    separate subpaths so noise regenerates correctly.
    """
    parts = []
    if has_hits:
        parts.append("l2")
    if has_knee:
        parts.append("1f")
    return "_".join(parts) if parts else "alms"


def _base_config(nsims: int, mask_type: str,
                 instrument_override: dict | None = None,
                 input_cache_tag: str = "alms",
                 cov_noise_debias_factor: float = 0.0) -> dict:
    sims = SCRATCH / "inputs" / EXPERIMENT
    cfg = {
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
        "data_path": str(sims / "total"
                         / f"total_{input_cache_tag}_ns{NSIDE}_lmax{LMAX}"),
        "fgds_path": str(
            sims / "foregrounds" / FG_TAG
            / f"foregrounds_alms_ns{NSIDE}_lmax{LMAX}"
        ),
        "cmb_path": str(sims / "cmb" / f"cmb_alms_ns{NSIDE}_lmax{LMAX}"),
        "noise_path": str(sims / "noise"
                          / f"noise_{input_cache_tag}_ns{NSIDE}_lmax{LMAX}"),

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
        "compsep": _compsep_blocks(cov_noise_debias_factor),
        "compsep_residuals": [],
        "compute_spectra": [],
    }
    if instrument_override is not None:
        cfg["instrument"] = instrument_override
    return cfg


def _compsep_blocks(debias_factor: float = 0.0) -> list[dict]:
    """Two compsep blocks: NILC on scalar B, and GNILC on QU for FG maps.

    GNILC hyperparameters follow Carones 2025 Sec. 3.2 (m(n_hat) + 1 modes,
    full CMB deprojection except at the lowest needlet band).

    `debias_factor` broadcasts to all needlet bands for both ILC and
    GILC `cov_noise_debias`.  Default 0.0 matches prior behavior.
    """
    debias_list = [debias_factor] * (len(NEEDLET_BANDS) - 1)
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
            "cov_noise_debias": debias_list,
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
            "cov_noise_debias": debias_list,
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
    p.add_argument("--hits-prefix", type=str, default=None,
                   help="If set, BROOM consumes per-channel hit map "
                        "FITS at <prefix>_<channel_tag>.fits and "
                        "per-channel depth_I/depth_P are rescaled for "
                        "sky-average-matched normalization. Generate "
                        "the hit maps first via scripts/make_hit_maps.py.")
    p.add_argument("--knee-config", type=str, default=None,
                   help="Path to a JSON file mapping channel_tag -> "
                        "{ell_knee, alpha_knee} for per-channel 1/f "
                        "noise. Must cover every channel. Example "
                        "value: ell_knee=30, alpha_knee=-2 gives the "
                        "standard knee shape rising at low ell.")
    p.add_argument("--cov-noise-debias", type=float, default=0.0,
                   help="Factor in [0, 1] for NILC+GNILC noise-covariance "
                        "debias; broadcast to all needlet bands. 0.0 "
                        "(default) = no debias. Non-zero is useful for "
                        "checking whether debias helps under anisotropic / "
                        "1/f noise where the noise covariance is no longer "
                        "a simple diagonal.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)

    instrument_override, _ = _build_instrument_override(
        EXPERIMENT,
        hits_prefix=args.hits_prefix,
        knee_config_path=args.knee_config,
    )
    if args.hits_prefix:
        print(f"Anisotropic noise: hit maps at {args.hits_prefix}_<tag>.fits")
    if args.knee_config:
        print(f"1/f noise: per-channel knee config from {args.knee_config}")

    cache_tag = _input_cache_tag(
        has_hits=args.hits_prefix is not None,
        has_knee=args.knee_config is not None,
    )
    config = Configs(config=_base_config(
        nsims=args.nsims, mask_type=args.mask,
        instrument_override=instrument_override,
        input_cache_tag=cache_tag,
        cov_noise_debias_factor=args.cov_noise_debias,
    ))

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
    tag = _output_tag(args.mask, args.nsims,
                      hits_prefix=args.hits_prefix,
                      knee_config=args.knee_config,
                      cov_noise_debias_factor=args.cov_noise_debias)
    _save_product(OUTPUTS_DIR / f"{tag}_nl_bb.npy", ell_centres, nl_mean)
    _save_product(OUTPUTS_DIR / f"{tag}_tres_bb.npy", ell_centres, tres_mean)
    _save_product(OUTPUTS_DIR / f"{tag}_fgds_bb.npy", ell_centres, fgds_mean)

    print("\nDone.")


if __name__ == "__main__":
    main()
