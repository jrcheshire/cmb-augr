# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

`cmb-augr` (import name `augr`) is a JAX-native Fisher-forecasting pipeline
that goes from **physical telescope specs → per-channel noise → binned
cross-frequency BB data vector → Fisher matrix → marginalized σ(r)**, with
analytic Jacobians via `jax.jacfwd` and a differentiable σ(r) via
`jax.grad` for instrument optimization. The full chain is differentiable
end-to-end, so design parameters can be optimized directly.

```bash
pixi install                                # solve + install the locked conda + pypi env
pixi run test                               # fast pytest subset (slow tests deselected)
pixi run test-all                           # full pytest suite (includes slow)
pixi run validate-pico                      # reproduce the PICO sigma(r) cross-check
pixi run pytest tests/test_fisher.py -v     # single file
```

## Architecture

Knowing how the modules chain together matters more than any one file:

1. **`telescope.py` → `instrument.py`.** `probe_design()` /
   `flagship_design()` return a physical `TelescopeDesign` (aperture,
   f/#, focal plane, feedhorn packing, temperature, optical efficiency).
   `to_instrument(design)` derives a frozen `Instrument` (list of
   `Channel`s: frequency, NET, beam, n_det, efficiency).
   `combined_noise_nl(inst, ells, "BB")` yields the MV-combined noise
   spectrum. The default photon-noise calculation models only CMB +
   telescope graybody emission (L2 baseline); per-band extra loading
   (galactic foregrounds at high ν, atmosphere for ground/balloon
   repurposings) attaches via the `extra_loading` callable on each
   `BandSpec` and is threaded through `to_instrument` automatically.
   Presets in `config.py`: `simple_probe`, `pico_like`,
   `litebird_like` (22 sub-array Channels per the LiteBIRD PTEP
   channel-specification table — each physical sub-array at a shared
   frequency is its own Channel so Fisher MV-combines them
   correctly), `so_like`, `cmbs4_like`, plus `_idealized` variants
   with PICO-style optics. A
   `deconvolve_noise_bb(noise, ells, fwhm_arcmin)` helper is available
   for users with beam-convolved noise who need the beam-deconvolved
   form required by the external-noise Fisher path.

2. **`foregrounds.py`.** Pluggable via structural `Protocol` — any
   class with `parameter_names` and `cl_bb(nu_i, nu_j, ells, params)`
   qualifies. Two built-ins: `GaussianForegroundModel` (BK15-style, 9
   params) and `MomentExpansionModel` (Chluba+ 2017, 17 params,
   reduces to Gaussian when moment amplitudes are zero). A
   `NullForegroundModel` is a no-op for the post-component-separation
   Fisher path. Default fiducials/priors/fixed-params sets live in
   `config.py` (`FIDUCIAL_BK15`, `FIDUCIAL_MOMENT`, `DEFAULT_PRIORS_*`,
   `DEFAULT_FIXED_*`).

3. **`spectra.py` + `signal.py`.** `CMBSpectra` loads CAMB templates
   from `data/` (tensor r=1, lensing BB, unlensed TT/EE/TE/BB, φφ).
   `SignalModel(inst, fg_model, spectra, ell_min, ell_max, delta_ell,
   delensed_bb=None)` assembles the binned cross-frequency data vector
   D(params) and its Jacobian. Pass `delensed_bb=result.cl_bb_res` to
   use a self-consistent residual lensing spectrum instead of an
   `A_lens` multiplier. Pass `bandpower_window=W,
   bandpower_window_ells=...` to substitute a measured BPWF for the
   synthetic top-hat / Gaussian binning -- either a shared 2-D matrix
   (Phase 1) or a `{(i_ch, j_ch): W}` per-spectrum dict (Phase 2). See
   "Measured bandpower window functions" below.

4. **`covariance.py` → `fisher.py`.** `BandpowerCovariance` is the Knox
   formula across all frequency cross-spectra.
   `FisherForecast(signal, inst, fiducial, priors=..., fixed_params=...)`
   builds F = Jᵀ C⁻¹ J per bin, adds priors on the diagonal, and
   exposes `sigma(param)` (marginalized) and conditional constraints.
   Two solver paths: a per-bin block-diagonal solve (default; valid for
   the synthetic top-hat / Gaussian binning), and a full
   `(n_data, n_data)` solve dispatched automatically when
   `signal_model.has_measured_bpwf` is True (BPWFs typically overlap so
   the per-bin block structure breaks). Both use eigendecomposition
   with non-positive-eigenvalue clipping for robustness against
   near-singular covariances.

5. **`delensing.py` + `wigner.py`.** Optional self-consistent
   iterative QE delensing replacing the `A_lens` parameter.
   `load_lensing_spectra()` + `iterate_delensing(spec, nl_tt, nl_ee,
   nl_bb, fullsky=..., n_iter=...)` runs all 5 estimators (TT, TE, EE,
   EB, TB), MV-combines, Wiener-filters to get C_L^{φ,res}, applies
   the lensing kernel for residual C_ℓ^{BB}, and iterates. Flat-sky
   uses Gauss-Legendre quadrature (~2 min / 5 iter); full-sky uses
   `wigner.py` Schulten-Gordon recursion (experimental, ~10 min).

6. **`optimize.py`.** End-to-end differentiable σ(r).
   `make_optimization_context(...)` packs the static pieces;
   `sigma_r_from_channels(n_det, net, beam, eta, ctx)` (Tier 1) and
   `sigma_r_from_design(...)` (Tier 2, physical geometry) are both
   `jax.grad`-compatible.

7. **`multipatch.py` + `sky_patches.py`.** `MultiPatchFisher` runs
   independent per-patch Fishers with shared spectral indices and
   per-patch amplitudes, then combines. Costs scale linearly in
   patches. Only `A_dust` and `A_sync` scale per patch; SED-shape
   params (`beta_*`, `T_dust`), decorrelation strengths
   (`Delta_dust`, `Delta_sync`), `c_sync`, and the moment-expansion
   variance parameters `omega_d_*` / `omega_s_*` are all global. The
   `omega_*` classification follows Chluba+ 2017 (arXiv:1701.00274)
   Eq. 8: `ω_{ij} = ⟨[p_i(r) - p̄_i][p_j(r) - p̄_j]⟩` is a pure
   sky-level central moment of spectral parameters, not bundled with
   amplitude. `MultiPatchFisher` builds a throwaway `SignalModel` in
   `__init__` to derive the parameter list, so delensed and
   residual-template modes are handled correctly.

8. **`hit_maps.py`.** `l2_hit_map(nside, alpha, beta, coord)` lifts
   the `sky_patches.l2_scan_depth` 1-D envelope onto HEALPix via
   `hp.Rotator`; input for BROOM's `path_hits_maps`.
   `mean_pixel_rescale_factor(hits)` returns
   `sqrt(max(H) · mean_surveyed(1/H))` — divide `depth_P` by this so
   the sky-averaged pixel noise variance matches spec (BROOM
   renormalizes hits to max=1 internally; without rescale, spec would
   describe the best pixel). For the defaults (α=50°, β=45°)
   `k ≈ 3`. Envelope-only: no sharp Deep Field ring, no
   per-feedhorn offsets, and the `θ_ecl < |β − α|` zero-fill is a
   single-precession-cycle artifact that doesn't reflect the year-long
   anti-sun sweep — don't use the model at extreme α ≫ β (e.g. Planck
   nominal β=0).

9. **`crosslinks.py`.**
   `h_k_map(nside, spin_angle_deg, precession_angle_deg, k, coord)`
   and `yearavg_h_k_1d(theta_ecl, ...)` give the year-averaged ergodic
   spin coefficients `h_k = ⟨e^{−ikψ}⟩` for an L2 scan. Closed form
   `h_k = (i)^k ⟨cos kA⟩_w` via 1-D adaptive quadrature over spin-axis
   colatitude `θ_S`, with Chebyshev substitution absorbing the
   integrable `1/√(boundary-distance)` singularities at the
   precession turning points and spin-circle tangencies.
   JAX-differentiable. **Conventions match `hit_maps.l2_hit_map`
   (Wallis: `spin_angle_deg` = boresight-to-spin half-angle), opposite
   of Takase / Falcons.jl** — internal vars use `spin`/`prec` words to
   sidestep. Phase prefactor `(i)^k` matches Falcons.jl's `ψ` sign
   convention; for direct east-of-north take complex conjugate.
   Validated against Falcons.jl at LiteBIRD-standard to within 0.008
   absolute everywhere bulk for k ∈ {1, 2, 4}; Planck-extreme bulk
   passes too, with pole-region disagreement traced to non-ergodicity
   of the integer-minute Falcons Planck preset (the closed form is the
   ergodic phase-space limit). Out of scope here: bias propagation
   `h_k → ΔC_ℓ^BB` (no `systematics.py` yet — the standard h_k
   factorization breaks for asymmetric-sidelobe × non-uniform FG; a
   Leloup-style end-to-end framework is the right tool there).

10. **`crosslinks_southpole.py` + `_chi2alpha.py`.** South Pole /
    BICEP-Keck companion to `crosslinks.py`. At lat = −90° the closed
    form reduces to a *finite weighted sum over the deck distribution*:
    no orbital phase-space integral, no Chebyshev quadrature. Public
    API: `h_k_boresight(deck_deg, weights, chi_deg, k)`,
    `h_k_offaxis(dec_deg, deck_deg, weights, r_deg, theta_fp_deg, chi_deg, k)`,
    `h_k_map_southpole(ra_grid, dec_grid, ...)`,
    `southpole_field_mask(...)`, `BA_DECK_ANGLES_8` constant.
    `_chi2alpha.py` is a JAX port of the BICEP/Keck `chi2alpha.m`
    polarization-angle routine, validated bit-exact against MATLAB at
    4 spot points (~1e-13). Two non-obvious results, locked in by
    tests: (a) for a *single detector* `|h_k|²` is invariant under
    `(r, θ_fp, dec)` — the off-axis correction enters only as a global
    phase; amplitude is set by the deck schedule alone. (b) BICEP
    Array's 8-deck cycle (45° step) null-suppresses h_k for k = 1..7
    cleanly — every spin moment in Wallis 2017's contamination list.
    (c) lat = −89.99° MAPO offset is `O(ε²) ~ 1e-7` because symmetric
    HA averaging kills the leading-order term; no `lat_deg` knob is
    exposed publicly. Pedagogical walkthrough in
    `scripts/southpole_derivation/`.

**Parameter-vector convention** (canonical list in `config.py`):

`FIDUCIAL_BK15` (12 entries): `r, A_lens, A_dust, beta_dust, alpha_dust,
T_dust, A_sync, beta_sync, alpha_sync, epsilon, Delta_dust, A_res`.

`FIDUCIAL_MOMENT` extends `FIDUCIAL_BK15` with 8 more entries:
`c_sync, Delta_sync, omega_d_beta, omega_d_T, omega_d_betaT,
omega_s_beta, omega_s_c, omega_s_betac` (20 entries total).

`A_res` is the post-component-separation residual-template amplitude;
it only enters the Fisher when `SignalModel` is constructed with
`residual_template_cl=...` (otherwise silently ignored). `A_lens`
similarly drops out in delensed mode. The `omega_*` parameters follow
the Chluba+ 2017 (arXiv:1701.00274) Eq. 8 sky-level central-moment
convention; naming is `omega_<species>_<quantity>` where species is
`d` (dust) or `s` (sync) and quantity is the spectral parameter
(`beta`, `T`, `c`) or pair (`betaT`, `betac`).

## Conventions

- **Frozen dataclasses everywhere** (`Instrument`, `Channel`,
  `TelescopeDesign`, `CMBSpectra`) — immutable, hashable, safe to pass
  across processes. Don't mutate; build new ones.
- **JAX 64-bit is enabled at import.** Keep arrays as `jnp`, not `np`,
  inside the Fisher/Jacobian path so JIT traces reuse. Numpy is fine
  for I/O and plotting.
- **Priors are 1σ Gaussian values**, added to F diagonal; fixed params
  are removed from the parameter vector before inversion. Both are
  specified by name (e.g. `priors={"beta_dust": 0.11}`).
- **Per-multipole vs binned.** `delta_ell` in `SignalModel` sets
  binning; set `delta_ell=1` for per-ℓ. The reionization bump (ℓ≲10)
  matters for space missions — lower `ell_min` to 2.
- **Validation discipline.** Prioritize careful validation over feature
  velocity. Do not relax test tolerances to make a test pass without
  explicit discussion; investigate the discrepancy.
  `scripts/validate_pico.py` is the canonical external check.

## Invariants worth preserving

- **`SignalModel` public surface.** Typed properties: `bin_matrix`,
  `bin_edges` (`list[tuple[int, int]]`), `frequencies`,
  `foreground_model`. `covariance.py` / `fisher.py` use these; do not
  introduce new `signal_model._*` reaches.
- **`delensed_bb` range is enforced.** `SignalModel` raises
  `ValueError` if `delensed_bb_ells` does not span
  `[ell_min, ell_max]`. `iterate_delensing`'s default `ls=[2, 300]`
  matches the default `SignalModel` range; a user extending `ell_max`
  past 300 must also widen `ls`.
- **Self-consistent delensing uses exact Smith+ 2012 Eq. 12.**
  `iterate_delensing` threads `nl_ee` through `residual_cl_bb`, which
  builds both the standard kernel `K` and a `W_EE(ℓ_E)`-weighted
  `K_WEE` and returns
  `K @ [C_φφ(1 - W_φφ)] + (K - K_WEE) @ [C_φφ W_φφ]`. Fullsky
  iterations are ~2× the pre-W_EE runtime (15-25 min for n_iter=5).
  At ℓ > ls[-1], `cl_bb_current` falls back to the full lensed BB —
  no flat-constant extrapolation.
- **`FisherForecast.summary()` diagnostics.** Always reports Knox
  modes per bin and `cond(F)`. Emits WARNING lines when
  `min(ν_b) < 10` (Gaussian-likelihood breakdown at the reionization
  bump) or `cond(F) > 1e14` (near-degenerate parameters; eigh-clipping
  may dominate the reported σ's).
- **Beam-deconvolved-noise contract.**
  `bandpower_covariance_blocks_from_noise` /
  `FisherForecast(external_noise_bb=...)` require beam-deconvolved
  noise spectra. NILC/GNILC outputs already are; raw `anafast`
  auto-spectra are not. Use
  `instrument.deconvolve_noise_bb(noise, ells, fwhm_arcmin)` if you
  need to convert.
- **Measured BPWFs imply beam-deconvolved external noise.**
  `FisherForecast` raises `ValueError` if `signal_model.has_measured_bpwf`
  is True and `external_noise_bb` is not supplied -- BPWFs released by
  real-data pipelines have the beam baked in, so analytic, beam-
  convolved noise would be inconsistent at every ℓ where `B_ℓ² < 1`.
  This is the same family of footgun as `requires_external_noise=True`
  on `cleaned_map_instrument`, with the trigger sitting on the signal
  side rather than the instrument side.
- **`optimize.sigma_r_from_*` vs `FisherForecast.sigma`.** `optimize`
  uses a gradient-smooth `solve`-based inversion;
  `FisherForecast.sigma` uses `eigh` with silent clipping of
  non-positive eigenvalues. The two can disagree by a few percent in
  degenerate-cov regimes. For reporting absolute numbers prefer
  `FisherForecast.sigma`; for autodiff use `optimize.*`.

## Measured bandpower window functions

The synthetic top-hat / Gaussian binning is replaced by a user-supplied
BPWF when `SignalModel` is constructed with `bandpower_window=...` and
`bandpower_window_ells=...`. The BPWF kernel encodes mask-mode coupling,
transfer-function corrections, and beam smoothing -- the kernel mapping
the underlying sky `C_ℓ` to the bandpower estimator from a real-data
analysis pipeline (BICEP/Keck releases, NaMaster / bk-jax outputs).

Two flavours of `bandpower_window`:

* **Shared** (Phase 1): a 2-D array of shape `(n_bins, n_ells)` applied
  identically to every cross-frequency spectrum.
* **Per-spectrum** (Phase 2): a `dict` `{(i_ch, j_ch): W}` carrying one
  `(n_bins, n_ells)` BPWF per cross-spectrum. Captures per-channel
  mask, transfer-function, and beam differences that BICEP/Keck-style
  multi-frequency releases (and bk-jax) produce. Either ordering
  `(i, j)` or `(j, i)` is accepted; entries are canonicalised to
  `(min, max)`. Every pair in `freq_pairs` must be supplied.

What changes when BPWFs are active (`signal_model.has_measured_bpwf =
True`):

- `bin_centers` derives from `Σ_ℓ ℓ W_b(ℓ) / Σ_ℓ W_b(ℓ)`. `bin_edges`
  becomes `None` (the (lo, hi) interval description is meaningless for
  an arbitrary BPWF). User-supplied W is **not** re-normalised --
  pipeline calibration is preserved.
- In **shared** mode, `bin_matrix` is the 2-D shared W. In
  **per-spectrum** mode, `bin_matrix` raises with a pointer to
  `bin_matrix_per_spectrum` (3-D tensor, shape
  `(n_pairs, n_bins, n_ells)`, ordered to match `freq_pairs`) and to
  the mode-agnostic accessor `bandpower_window_for(i_ch, j_ch)`.
  `is_per_spectrum_bpwf` is the public bool flag for downstream
  dispatch.
- The per-bin block-diagonal Knox approximation breaks: bins couple
  through `Σ_ℓ W_b(ℓ) W_{b'}(ℓ) (2ℓ+1)`. `bandpower_covariance` and
  `FisherForecast.compute()` automatically dispatch to the full-path
  Knox sum (`bandpower_covariance_full*`); the
  `bandpower_covariance_blocks*` entry points raise
  `NotImplementedError` in BPWF mode. The `_knox_full` einsum has two
  branches (rank-2 shared / rank-3 per-spectrum) so JAX sees the
  contraction structure; numerics are bit-identical to Phase 1 when a
  per-spectrum dict has all rows equal.
- `FisherForecast` requires `external_noise_bb` whenever
  `has_measured_bpwf` is True (BPWFs released by analysis pipelines
  have the beam baked in; pairing them with augr's analytic,
  beam-convolved noise would be inconsistent). Use
  `instrument.deconvolve_noise_bb` if the available noise array is
  still beam-convolved.
- `summary()` reports the BPWF Knox-mode count
  `Σ_ℓ W_b(ℓ)² (2ℓ+1) f_sky` from the first cross-spectrum's window in
  per-spectrum mode (matching the `bin_centers` first-pair convention)
  and tags the line `(BPWF, per-spec, first-pair)`. Sub-unity values
  are formatted in scientific notation so they don't collapse to
  "0.0" at small `f_sky`.

Loaders (`augr.bandpower_windows`):

- `load_bandpower_window(path)` -- single 2-D BPWF, shared use. Returns
  `(ells, W)`. Sniffs:
  - `.npy`: 2-D array `(n_ells, 1 + n_bins)`, column 0 = ℓ.
  - `.npz`: arrays `ells` (`(n_ells,)`) and `window` (`(n_bins, n_ells)`).
  - `.csv` / `.dat` / `.txt`: whitespace- or comma-delimited table,
    same layout as `.npy`. Lines starting with `#` are comments.
- `load_bandpower_window_set(spec)` -- per-spectrum dict for Phase 2.
  Returns `(ells, dict)`. Two input shapes:
  - **Directory / glob** of files named `bpwf_{i}_{j}.{ext}`
    (regex `bpwf_(\d+)_(\d+)\.<ext>`); each file is parsed with the
    single-file loader, channel indices come from the filename, and
    pair ordering is canonicalised. Duplicate canonicals (e.g. both
    `bpwf_0_1` and `bpwf_1_0`) raise.
  - **Single `.npz`** with arrays `ells` (`(n_ells,)`),
    `window` (`(n_pairs, n_bins, n_ells)`), and `freq_pairs`
    (`(n_pairs, 2)`). bk-jax's eventual cross-pair output uses this
    layout.

Phase 2.5 (deferred): asymmetric beams × non-uniform foregrounds. The
standard BPWF formalism factorises only for isotropic beams; the right
tool there is a Leloup-style end-to-end TOD framework, not a BPWF.

Motivating use case: linking augr's Fisher forecast to bk-jax's
real-data bandpower outputs (`~/bicepkeck/bk-jax/`), so the same
forecast machinery runs on BK24 / BK28 bandpowers.


## Post-component-separation forecasts (BROOM consumer mode)

For forecasts that consume an external component-separation pipeline
(NILC/GNILC via [BROOM](https://github.com/alecarones/broom) instead
of augr's built-in multifrequency FG model), two scripts live under
`scripts/`:

- `broom_residual_template.py` — BROOM driver. Runs NILC + GNILC +
  `estimate_residuals` + per-sim `anafast` across an MC loop, applies
  the Carones 2025 (arXiv:2510.20785) Eq. 3.7 noise debiasing, and
  writes three
  `(n_bins, 2)` npy pairs (ell_center, C_ell^BB) to
  `data/broom_outputs/`: `{tag}_nl_bb.npy` (post-NILC noise),
  `{tag}_tres_bb.npy` (debiased residual template), `{tag}_fgds_bb.npy`
  (ground-truth fgds residual, diagnostic). CLI knobs: `--nsims`,
  `--mask`, `--skip-compsep`, `--fg-model {d1s1,d10s5}`,
  `--hits-prefix`, `--knee-config`, `--cov-noise-debias`.
- `validate_carones.py` — augr-only consumer. Loads a tag's npy pair,
  builds a single-channel `cleaned_map_instrument` +
  `NullForegroundModel` + residual-template `SignalModel`, runs Fisher
  variants (baseline / `A_res` flat / `A_res` Gaussian), prints
  σ(r) and the 2×2 (r, A_res) Fisher condition-number diagnostic.

The `augr` API extensions supporting this mode:

- `foregrounds.NullForegroundModel` — Protocol-satisfying no-op
  foreground model (empty `parameter_names`, zero `cl_bb`). Drop-in
  replacement for the Gaussian / Moment models when the foregrounds
  have already been removed.
- `signal.SignalModel(..., residual_template_cl=..., residual_template_ells=...)`
  — appends `A_res` to the parameter vector and adds
  `A_res * T_res(ell)` to the data vector on i==j auto blocks only.
  `SignalModel.residual_bb_unbinned(params)` exposes the same
  `A_res * T_res` on the ℓ grid; `covariance._build_M` and
  `bandpower_covariance_blocks_from_noise` consume it so the Knox
  covariance `M = S + N` picks up the template on `i==j` blocks too.
  Template interpolation onto augr's ℓ grid uses nearest-neighbour
  extrapolation at ℓ below the first BROOM bin centre — zero-extrapolation
  there silently nulls the reionization-bump constraint.
- `fisher.FisherForecast(..., external_noise_bb=...)` — opt-in (default
  `None`), shape `(n_channels, n_ells)` on the SignalModel ell grid.
  When provided, routes covariance through
  `bandpower_covariance_blocks_from_noise`; the analytic path is
  unchanged.
- `config.cleaned_map_instrument(f_sky, mission_years=3.0, nu_ghz=150.0)`
  — single-channel placeholder `Instrument`. The only field that
  meaningfully enters Fisher in this mode is `f_sky` (Knox mode count);
  noise parameters are dummies expected to be overridden via
  `external_noise_bb`. The preset sets
  `Instrument.requires_external_noise=True` and `FisherForecast` raises
  `ValueError` if that flag is set and `external_noise_bb` is not
  supplied — catches the silent "analytic path with dummy NET=1 µK√s"
  footgun.
- `FIDUCIAL_BK15` / `FIDUCIAL_MOMENT` include `A_res=1.0`;
  `DEFAULT_PRIORS` / `DEFAULT_PRIORS_MOMENT` include `A_res=0.3`
  (conservative Gaussian default; drop from the priors dict to
  reproduce a flat prior).
