# augr

![](./assets/logo.png)

Fisher-matrix forecasting for CMB B-mode polarization experiments, targeting the tensor-to-scalar ratio *r*.

`augr` translates physical instrument specifications (aperture, focal plane geometry, detector counts, NETs, beams) into a marginalized Fisher constraint on *r*, accounting for Galactic foregrounds, gravitational lensing, and frequency-by-frequency cross-spectrum information. The full pipeline is JAX-differentiable end-to-end, so instrument design parameters can be optimized via `jax.grad`.

## What it does

Given an instrument specification (frequency bands, detector counts, noise levels, beam sizes, integration time), `augr` computes the marginalized Fisher constraint on *r* after accounting for:

- **Foreground contamination** from polarized dust and synchrotron, modeled as either a simple Gaussian (BK15-style, 9 parameters) or a moment expansion (17 parameters) that captures SED spatial variation and frequency decorrelation
- **Gravitational lensing** B-modes, either parameterized by A_lens or self-consistently delensed via iterative quadratic-estimator lensing reconstruction (flat-sky or full-sky Wigner 3j)
- **Priors** on foreground spectral indices from Planck/WMAP
- **Bandpower covariance** via the Knox formula across all frequency cross-spectra

The telescope design module derives detector counts and photon-noise-limited NETs from physical specifications (aperture, f-number, focal plane size, feedhorn packing), enabling systematic optimization of band layout and focal plane area allocation.

## Quick start

The project uses [pixi](https://pixi.sh/) to manage a reproducible
conda + pypi environment pinned via `pixi.lock`:

```bash
pixi install         # solve + install the locked environment
pixi run test        # run the fast pytest subset
pixi run test-all    # full suite (includes opt-in slow tests)
pixi run validate-pico   # PICO sigma(r) cross-check
pixi run nb          # launch jupyter lab on notebooks/
```

For a guided tour of the API, see [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).

```python
from augr.telescope import probe_design, to_instrument
from augr.fisher import FisherForecast
from augr.foregrounds import MomentExpansionModel
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.config import FIDUCIAL_MOMENT, DEFAULT_PRIORS_MOMENT, DEFAULT_FIXED_MOMENT

# Build instrument from physical telescope specs
inst = to_instrument(probe_design())

# Set up the signal model
signal = SignalModel(
    inst,
    MomentExpansionModel(),
    CMBSpectra(),
    ell_min=2, ell_max=1000, delta_ell=30,
)

# Run the Fisher forecast -- two options for lensing:

# Option 1: fixed A_lens (fast, approximate)
fiducial = {**FIDUCIAL_MOMENT, "A_lens": 0.27}  # 73% delensing
ff = FisherForecast(
    signal, inst, fiducial,
    priors=DEFAULT_PRIORS_MOMENT,
    fixed_params=DEFAULT_FIXED_MOMENT,
)
print(f"sigma(r) = {ff.sigma('r'):.2e}")

# Option 2: self-consistent delensing (full-sky QE reconstruction)
from augr.delensing import load_lensing_spectra, iterate_delensing
from augr.instrument import combined_noise_nl
spec = load_lensing_spectra()
nl_bb = combined_noise_nl(inst, spec.ells, "BB")
result = iterate_delensing(spec, combined_noise_nl(inst, spec.ells, "TT"),
                           nl_bb, nl_bb, fullsky=True, n_iter=5)
signal_d = SignalModel(inst, MomentExpansionModel(), CMBSpectra(),
                       delensed_bb=result.cl_bb_res, delensed_bb_ells=result.ls)
ff_d = FisherForecast(signal_d, inst,
                      {k: v for k, v in FIDUCIAL_MOMENT.items() if k != "A_lens"},
                      priors=DEFAULT_PRIORS_MOMENT, fixed_params=DEFAULT_FIXED_MOMENT)
print(f"sigma(r) [delensed] = {ff_d.sigma('r'):.2e}")
```

## Package structure

```
augr/
  config.py        Fiducial parameters, priors, and instrument presets
                   (simple_probe, pico_like, litebird_like, so_like, cmbs4_like)
  instrument.py    Channel, Instrument, ScalarEfficiency dataclasses;
                   noise power spectrum N_ell from NET, beam, and 1/f
  telescope.py     Physical telescope model: derives beams, detector counts,
                   and photon-noise NETs from aperture, focal plane, and
                   feedhorn geometry; supports dichroic pixel groups
  foregrounds.py   GaussianForegroundModel (9 params, BK15-style) and
                   MomentExpansionModel (17 params, Chluba+ 2017)
  spectra.py       CMB BB power spectra from CAMB templates (tensor + lensing)
  signal.py        SignalModel: assembles the binned cross-frequency data
                   vector and computes the Jacobian via jax.jacfwd
  covariance.py    Bandpower covariance matrix (Knox formula)
  fisher.py        Fisher information matrix, marginalized and conditional
                   constraints; Cholesky solver with eigendecomposition fallback
  delensing.py     Iterative QE lensing reconstruction: all 5 estimators
                   (TT, TE, EE, EB, TB) with MV combination, residual BB
                   via lensing kernel, flat-sky and full-sky (Wigner 3j) modes
  wigner.py        Wigner 3j symbols: closed-form (0,0,0) via log-gamma,
                   Schulten-Gordon backward recursion for spin-2, vectorized
                   over l1 for fixed L
  optimize.py      Differentiable sigma(r) for gradient-based instrument
                   optimization: channel-level (Tier 1) and telescope
                   design-level (Tier 2) via jax.grad
  units.py         Physical constants, RJ/CMB unit conversions, dust and
                   synchrotron SEDs and their log-derivatives
  multipatch.py    Multi-patch Fisher with shared spectral indices,
                   per-patch amplitudes, L2 scan strategy model
  sky_patches.py   Sky patch definitions and scan strategy

scripts/
  validate_pico.py             Validation against PICO published sigma(r) targets
  validate_carones.py          Validation against Carones 2025 post-CompSep
                               residual-template forecast (LiteBIRD-PTEP)
  validate_bk.py               BK sigma(r) time evolution; analog of
                               Buza 2019 thesis Fig. 7.9
  broom_residual_template.py   End-to-end BROOM driver: NILC + GNILC +
                               residual-template MC for an external
                               component-separation forecast
  make_hit_maps.py             Per-channel L2 hit map FITS writer for BROOM
  generate_camb_templates.py   Regenerate the CAMB spectra under data/
  southpole_derivation/        Pedagogical walkthrough of the South Pole
                               h_k closed form

notebooks/
  quickstart.ipynb     Guided tour of the API

tests/              Full pytest suite covering every module
data/               CAMB template spectra (tensor r=1, lensing, unlensed TT/EE/TE/BB, phi-phi)
plots/              Output directory (gitignored)
```

## Design principles

- **JAX throughout** for exact autodiff (Jacobians via `jax.jacfwd`), JIT compilation, and differentiable instrument optimization via `jax.grad`.
- **Physics-based noise** from first principles (photon NEP, optical loading, feedhorn packing). Adding a mode to rescale from achieved performance is a potential future item.
- **Extensible foreground models** via a structural `Protocol` type. Any class with `parameter_names` and `cl_bb(nu_i, nu_j, ells, params)` works.
- **Frozen dataclasses** for all specifications (immutable, hashable, safe to pass across threads).
- **Realistic telescope and survey efficiency factors**: detector yield, survey efficiency, data loss, and more. For the telescope module, floor-based pixel counting, packing efficiency, and optical efficiency. Defaults are conservative, but optimistic "idealized" presets are available for comparison.

## Performance

All times on a single machine (Ryzen 9 5900X, 32 GB). First call includes JAX JIT compilation; subsequent calls reuse cached traces.

| Operation | First call | Cached |
|-----------|-----------|--------|
| FisherForecast (probe, 6-band Gaussian) | ~4 s | **70 ms** |
| FisherForecast (PICO, 17-band Gaussian) | ~15 s | **1.1 s** |
| FisherForecast (probe, 6-band Moment 17-param) | ~5 s | **130 ms** |
| MultiPatchFisher (probe, 3-patch Gaussian) | — | **7 s** |
| MultiPatchFisher (probe, 3-patch Moment) | — | **16 s** |
| `iterate_delensing` (flat-sky, 5 iter, l_max=3000) | ~2 min | ~25 s |
| `iterate_delensing` (full-sky Wigner 3j, 5 iter) | — | ~10 min |
| `sigma_r_from_channels` forward pass | ~4 s | **90 ms** |
| `jax.grad(sigma_r)` w.r.t. (n_det, NET, beam) | ~20 s | **470 ms** |

Scaling: Fisher cost grows as O(n_chan^2) in the Jacobian (n_chan^2 cross-spectra) and O(n_spec^3) per ell-bin in the covariance eigendecomposition. Going from 6 to 17 bands increases the number of cross-spectra from 21 to 153, accounting for the ~15x increase. Multi-patch scales linearly in the number of patches (independent per-patch Fishers). The gradient adds ~5x overhead vs the forward pass.

## Telescope design module

The `telescope.py` module derives a complete `Instrument` from physical specifications:

| Input | Default (probe) | Default (flagship) |
|---|---|---|
| Aperture | 1.5 m | 3.0 m |
| Focal ratio | f/2 | f/2 |
| Focal plane diameter | 0.4 m | 0.6 m |
| Telescope temperature | 4 K | 4 K |
| Optical efficiency | 0.35 | 0.35 |
| Pixel pitch | 2 F lambda (feedhorn) | 2 F lambda (feedhorn) |
| Packing efficiency | 80% | 80% |

"Idealized" variants (`probe_idealized`, `flagship_idealized`) use PICO-like assumptions (f/1.42, eta=0.50, 95% observing efficiency) for direct comparison, while retaining the feedhorn pixel pitch.

## Foreground models

**Gaussian (BK15-style):** Dust modified blackbody + synchrotron power law, with amplitudes, spectral indices, ell-dependence slopes, dust-sync correlation, and dust frequency decorrelation. 9 free parameters.

**Moment expansion (Chluba+ 2017):** Extends the Gaussian model with second-order terms capturing spatial variation of spectral parameters (variance of beta_d, T_d, beta_s, c_s, and their cross-moments). 17 free parameters. Reduces exactly to the Gaussian model when all moment amplitudes are zero.

## Delensing

The `delensing.py` module computes self-consistent iterative QE delensing, replacing the external A_lens parameter with a derived residual lensing spectrum:

1. Compute the minimum-variance QE reconstruction noise N_0(L) from all 5 estimators (TT, TE, EE, EB, TB)
2. Compute the Wiener-filtered residual lensing potential: C_L^{phi,res} = C_L^{phi} N_0 / (C_L^{phi} + N_0)
3. Compute the residual BB via the lensing kernel: C_l^{BB,res} = K(l,L) @ C_L^{phi,res}
4. Update the BB in the EB/TB filter denominators and iterate until converged

Two modes are available:

- **Flat-sky** (`fullsky=False`): Gauss-Legendre quadrature over the azimuthal angle. Fast (~2 min for 5 iterations at l_max=3000).
- **Full-sky** (`fullsky=True`) [EXPERIMENTAL]: Wigner 3j coupling via Schulten-Gordon backward recursion, vectorized over l1 for fixed L with log-spaced L sampling.

```python
from augr.delensing import load_lensing_spectra, iterate_delensing
from augr.instrument import combined_noise_nl

spec = load_lensing_spectra()
nl_bb = combined_noise_nl(inst, spec.ells, "BB")
nl_ee, nl_tt = nl_bb, combined_noise_nl(inst, spec.ells, "TT")

result = iterate_delensing(spec, nl_tt, nl_ee, nl_bb, fullsky=True, n_iter=5)
# result.A_lens_eff ~ 0.29 for probe-class, result.cl_bb_res for Fisher input
```

## Gradient-based instrument optimization

The `optimize.py` module provides a fully differentiable path from instrument parameters to σ(r), enabling gradient-based optimization via `jax.grad`:

```python
import jax
from augr.optimize import make_optimization_context, sigma_r_from_channels
from augr.telescope import probe_design, to_instrument
from augr.foregrounds import GaussianForegroundModel
from augr.spectra import CMBSpectra
from augr.config import FIDUCIAL_BK15

inst = to_instrument(probe_design())
ctx = make_optimization_context(
    inst, GaussianForegroundModel(), CMBSpectra(), dict(FIDUCIAL_BK15),
    priors={"beta_dust": 0.11, "beta_sync": 0.3},
    fixed_params=["T_dust", "Delta_dust"],
)

# Gradient of sigma(r) w.r.t. detector counts per channel
grad_fn = jax.grad(sigma_r_from_channels, argnums=0)
d_sigma_d_ndet = grad_fn(ctx.n_det, ctx.net, ctx.beam, ctx.eta, ctx)
# All negative: more detectors in any channel reduces sigma(r)
```

Two tiers are available:

- **Tier 1** (`sigma_r_from_channels`): optimize detector counts, NETs, and beam sizes directly as continuous floats.
- **Tier 2** (`sigma_r_from_design`): optimize telescope geometry (aperture, f-number, focal plane diameter, area fractions) and derive channel parameters via the physics.

## TODO

- **Scale-dependent moment expansion**: make omega parameters functions of ell to capture the angular-scale dependence of foreground SED variation.
- **Achieved-performance noise mode**: option to rescale from measured detector performance rather than computing from first principles.
- **Full-sky N_0 cross-validation**: compare against plancklens/lenspyx for absolute normalization of the lensing reconstruction noise.

## References

- Buza 2019, PhD thesis (Harvard) -- Fisher formalism, BICEP/Keck forecasting
- BICEP2/Keck 2018 (arXiv:1810.05216) -- BK15 foreground model and parameters
- Chluba et al. 2017 (arXiv:1701.00274) -- Moment expansion for foreground complexity
- Hanany et al. 2019 (arXiv:1902.10541) -- PICO probe study report
- PanEx Group et al. 2025 (arXiv:2502.20452) -- PanEx PySM3 foreground models
- Bianchini et al. 2025 (ApJ 993:105) -- Foreground pipeline comparison (from CMB-S4 effort)
- Hu & Okamoto 2002 (arXiv:astro-ph/0111606) -- Quadratic estimator lensing reconstruction
- Okamoto & Hu 2003 (PRD 67, 083002) -- Full-sky QE formalism
- Smith et al. 2012 (arXiv:1010.0048) -- Residual BB after delensing
- Maniyar et al. 2021 (arXiv:2101.12193) -- Full-sky N_0 formulas
- Trendafilova, Meyers et al. 2023 (arXiv:2312.02954) -- CLASS_delens iterative delensing
