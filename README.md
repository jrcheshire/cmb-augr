# augr

![](./assets/logo.png)

Fisher-matrix forecasting for CMB B-mode polarization experiments, targeting the tensor-to-scalar ratio *r*.

Built for the JPL CMB Probe 2026 study to explore how instrument design choices affect sensitivity to primordial gravitational waves in the presence of realistic Galactic foregrounds.

## What it does

Given an instrument specification (frequency bands, detector counts, noise levels, beam sizes, integration time), `augr` computes the marginalized Fisher constraint on *r* after accounting for:

- **Foreground contamination** from polarized dust and synchrotron, modeled as either a simple Gaussian (BK15-style, 9 parameters) or a moment expansion (17 parameters) that captures SED spatial variation and frequency decorrelation
- **Gravitational lensing** B-modes (parameterized by A_lens)
- **Priors** on foreground spectral indices from Planck/WMAP
- **Bandpower covariance** via the Knox formula across all frequency cross-spectra

The telescope design module derives detector counts and photon-noise-limited NETs from physical specifications (aperture, f-number, focal plane size, feedhorn packing), enabling systematic optimization of band layout and focal plane area allocation.

## Quick start

An "`augr`" `conda` environment is included with the needed dependencies.

```bash
make install   # create conda env + pip install -e .
make test      # run 169 tests
```

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

# Run the Fisher forecast
fiducial = {**FIDUCIAL_MOMENT, "A_lens": 0.27}  # 73% delensing
ff = FisherForecast(
    signal, inst, fiducial,
    priors=DEFAULT_PRIORS_MOMENT,
    fixed_params=DEFAULT_FIXED_MOMENT,
)
print(f"sigma(r) = {ff.sigma('r'):.2e}")
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
  units.py         Physical constants, RJ/CMB unit conversions, dust and
                   synchrotron SEDs and their log-derivatives

scripts/
  explore_designs.py   Band optimization: density scan, frequency range scan,
                       area allocation, experiment comparison (parallelized)
  validate_pico.py     Validation against PICO published sigma(r) targets
  plot_figure5.py      Reproduction of BICEP/Keck Figure 5 time evolution

tests/              169 tests covering all modules
data/               CAMB template spectra (tensor r=1, lensing)
plots/              Output from explore_designs.py
```

## Design principles

- **JAX throughout** for exact autodiff (Jacobians via `jax.jacfwd`) and JIT compilation.
- **Physics-based noise** from first principles (photon NEP, optical loading, feedhorn packing). Adding a mode to rescale from achieved performance is a potential future item.
- **Extensible foreground models** via a structural `Protocol` type. Any class with `parameter_names` and `cl_bb(nu_i, nu_j, ells, params)` works.
- **Frozen dataclasses** for all specifications (immutable, hashable, safe to pass across threads -- see example in `scripts/explore_designs.py`).
- **Realistic telescope and survey efficiency factors**: detector yield, survey efficiency, data loss, and more. For the telescope module, floor-based pixel counting, packing efficiency, and optical efficiency. Defaults are conservative, but optimistic "idealized" presets are available for comparison.

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

## Key findings so far

Results from systematic exploration of the probe/flagship design space:

- **Band count matters more than aperture for foregrounds.** Under the moment expansion, sigma(r) improves steadily out to 15+ bands. With only 6 bands, the moment model gives 3-5x worse sigma(r) than the Gaussian model. By ~15 bands, the penalty drops to ~2x.
- **Feedhorn coupling limits band count.** At 2Flambda pitch, 30 GHz horns are 40mm diameter. A 0.4m focal plane with 12 bands and equal allocation gives only 24 detectors at 30 GHz. Past ~15 bands on a probe FP, low-freq channels are starved of detectors.
- **Area allocation and band layout are coupled.** With only 3 bands above 200 GHz, the moment expansion's 8 dust parameters are underdetermined, so adding more *detectors* in that regime doesn't help. You need more *bands* in the dust regime, not more *area*. Optimizing one while the other is fixed can mislead.
- **PICO's advantage is technology, not design.** PICO's sinuous antennas at Flambda pitch give 4x the pixel density of feedhorns. Their ~70% optical efficiency vs our 35% conservative assumption compounds this. A feedhorn-based probe with PICO-level optimism (f/1.42, eta=0.50, 95% obs efficiency) gets within ~2x of PICO's sigma(r).
- **Delensing is critical but has diminishing returns for foregrounds.** Going from no delensing to 73% helps sigma(r) by 2-3x. Going from 73% to 95% helps another 2-3x for the Gaussian model, but only ~2x for the moment model, because the foreground floor dominates.
- **Frequency range below 30 GHz barely helps.** Synchrotron is subdominant to dust for B-mode foregrounds. Extending above 400 GHz helps the moment model slightly but runs into dust SED degeneracies.

## Future work

### Self-consistent delensing

Currently A_lens is a fixed external parameter controlling the residual lensing B-mode amplitude. A more complete treatment would:

1. Derive the lensing reconstruction noise N_L^{dd} from the instrument's high-ell temperature and polarization sensitivity (iterative EB estimator)
2. Compute the delensed BB residual power spectrum as a function of N_L^{dd}
3. Replace the fixed A_lens with a self-consistently derived delensing level

This would properly capture the value of larger apertures: better high-ell resolution enables better lensing reconstruction, which lowers A_lens, which improves sigma(r). Currently our scans show aperture has little effect because the foreground information content doesn't depend on beam size, but the delensing benefit is real and unaccounted for.

References: Carron et al. (2017) for iterative EB delensing; PICO Sec. 2.3.2; Bianchini et al. (2025) Sec. 5; S4 Science Book Sec. 8.10.

### Differentiable instrument optimization

The Fisher pipeline is built on JAX, so sigma(r) is in principle differentiable with respect to continuous instrument parameters. By relaxing discrete quantities (e.g. dropping `floor()` in detector counting), one could compute exact gradients d(sigma(r))/d(theta) for:

- Focal plane area fractions per dichroic pair (replace the coarse 3-group simplex scan with gradient descent over 6+ area fractions)
- Optical efficiency, mission duration, f_sky (sensitivity trade studies)
- Per-channel depths (for matching achieved/measured performance rather than photon noise)

Requires rewriting `photon_noise_net()` in JAX (currently numpy), relaxing `count_pixels()` to continuous, and composing the full chain: design params -> instrument -> Fisher -> sigma(r). Not needed for the current exploration scans (discrete band layouts), but powerful for fine-tuning a specific design once the broad parameter space is understood.

### Scale-dependent foreground complexity

The moment expansion currently assumes frequency decorrelation is scale-independent. In reality, foreground SED variation has a characteristic angular scale: at small scales (high ell), each pixel's SED is more uniform, so decorrelation is weaker. A scale-dependent variant would:

1. Make the moment amplitude parameters (omega_d_beta, etc.) functions of ell
2. Model as e.g. omega(ell) = omega_0 * (ell / ell_pivot)^gamma with a tilt parameter
3. Propagate through the Jacobian (straightforward since JAX handles the extra parameters)

This would couple the foreground and ell-range questions: scale-dependent decorrelation would make high-ell data more valuable for foregrounds, since cross-frequency coherence is better preserved there. Lower priority than self-consistent delensing, but relevant for understanding the true foreground floor.

### Achieved-performance noise mode

Currently all noise is computed from first principles (photon NEP). An alternative mode that rescales from measured/achieved detector performance would be useful for:

- Validating against published experiment sensitivities
- Forecasting for partially-built experiments where detector yield and noise are measured
- Comparing "what the physics says" against "what we actually get" to understand systematic noise penalties

### Assumption tracking and reproducibility

Fisher forecasts are notoriously sensitive to unstated assumptions (efficiency factors, fiducial values, which parameters are fixed, ell range, bin width). Every published sigma(r) number should carry a complete provenance record. The `FisherForecast.summary()` method dumps all assumptions for a given forecast. Future work: structured machine-readable output (JSON/YAML) for automated comparison across studies.

### Joint probe + ground analysis

A space probe doesn't operate in isolation. Combining with ground-based data (SO, CMB-S4) would change optimal band allocation: the ground experiments provide deep CMB-band data, so the space mission could focus more on frequency leverage (low and high freq). This requires modeling the joint covariance across experiments with different sky coverage and ell ranges.

## References

- Buza 2019, PhD thesis (Harvard) -- Fisher formalism, BICEP/Keck forecasting
- BICEP2/Keck 2018 (arXiv:1810.05216) -- BK15 foreground model and parameters
- Chluba et al. 2017 (arXiv:1701.00274) -- Moment expansion for foreground complexity
- Hanany et al. 2019 (arXiv:1902.10541) -- PICO probe study report
- Puglisi et al. 2025 (arXiv:2502.20452) -- PanEx PySM3 foreground models
- Bianchini et al. 2025 (ApJ 993:105) -- CMB-S4 foreground pipeline comparison
