# falcons_validation

Validation of `augr.crosslinks` against
[Falcons.jl](https://github.com/yusuke-takase/Falcons.jl) (Takase 2025,
arXiv:2503.03176), and the literature trail backing the
"plausibly novel" tag on the year-averaged closed form.

This directory is the load-bearing evidence for two claims made in the
package:

  1. The closed-form `h_k(theta_ecl)` quadrature implemented in
     `augr/crosslinks.py` reproduces the Falcons.jl ergodic-limit
     result to within 0.008 (absolute) for k in {1, 2, 4} at
     LiteBIRD-standard parameters, and matches the bulk at
     Planck-extreme parameters with the pole-region disagreement
     traced to non-ergodicity of the integer-minute Falcons preset.
  2. The 1-D quadrature with the precession * spin Jacobian split
     does not appear in the published CMB scan-strategy literature
     surveyed (Wallis et al. 2017, McCallum et al. 2021, Takase 2025
     thesis, Falcons.jl source, and the pseudo-Cl asymmetric-beam
     stream).

The lit-search write-up plus closed-form derivation lives in
[`derivation_and_lit_search.md`](derivation_and_lit_search.md).

## Layout

Python (analysis + comparison plots, run with the augr pixi env):

  - `compare_hk_to_envelope.py`  — `h_k` from `augr.crosslinks` vs the
    Falcons.jl FITS for k in {1, 2, 4}; emits `compare_h{k}.png` to
    `data/falcons_validation/`.
  - `compare_yearavg.py`  — bulk 1-D residual across `theta_ecl`;
    emits `compare_yearavg_1D.png`.
  - `mc_truth.py`  — independent Monte-Carlo realization of the
    closed form, used as a fixed-time-series cross-check.
  - `plot_hitmaps.py`  — Mollview plots of the LiteBIRD and
    Planck-preset hit maps (envelope from `augr.hit_maps`).

Julia (Falcons.jl driver, run with the local Julia project):

  - `validate_hk.jl`         — Falcons.jl run at LiteBIRD-standard
    parameters; writes `h{k}_litebird_nside128.fits` and
    `hitmap_litebird_nside128.fits`.
  - `validate_hk_planck.jl`  — same at Planck-default parameters.
  - `Project.toml` / `Manifest.toml` — Julia environment lockfile
    (Falcons.jl + deps).

## Reproducing the validation outputs

Outputs live in `data/falcons_validation/`. The PNGs and `metadata*.txt`
are tracked; the regenerable FITS files (~27 MB) are gitignored.

```bash
# 1. Generate the Falcons.jl reference FITS (Julia)
cd scripts/falcons_validation
julia --project=. validate_hk.jl
julia --project=. validate_hk_planck.jl

# 2. Compare augr.crosslinks against Falcons.jl, regenerate PNGs
cd ../..
pixi run python scripts/falcons_validation/compare_hk_to_envelope.py
pixi run python scripts/falcons_validation/compare_yearavg.py
pixi run python scripts/falcons_validation/plot_hitmaps.py
```

Julia is not part of the pixi environment; install separately from
[julialang.org](https://julialang.org/downloads/) (1.10+ recommended)
and resolve the Julia env once with
`julia --project=. -e 'using Pkg; Pkg.instantiate()'` from this
directory before running the `.jl` scripts.

The first FITS-generation step takes O(10) minutes per preset at
nside=128. Subsequent comparison-plot regenerations take seconds.
