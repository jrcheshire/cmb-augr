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
   the per-bin block structure breaks). Both use ``jnp.linalg.solve``
   (LU with partial pivoting) and a closing ``0.5 * (F + F^T)``
   symmetrization. ``optimize.sigma_r_from_channels`` and
   ``FisherForecast.sigma`` route through the same primitive, so they
   agree to fp64 precision at any allocation.

5. **`delensing.py` + `wigner.py`.** Optional self-consistent
   iterative QE delensing replacing the `A_lens` parameter.
   `load_lensing_spectra()` + `iterate_delensing(spec, nl_tt, nl_ee,
   nl_bb, fullsky=..., n_iter=...)` runs all 5 estimators (TT, TE, EE,
   EB, TB), MV-combines, Wiener-filters to get C_L^{φ,res}, applies
   the lensing kernel for residual C_ℓ^{BB}, and iterates. Flat-sky
   uses Gauss-Legendre quadrature (~2 min / 5 iter); full-sky uses
   `wigner.py` Schulten-Gordon recursion (~10 min / 5 iter).

   **N₀ validation status (2026-05-07).** Validated against `plancklens`
   at the LiteBIRD-PTEP fiducial in `scripts/n0_validation/`:
   - **TT flat-sky**: machine-precision against the constant-Cℓ closed
     form (`controlled_input_test.py`).
   - **TT full-sky**: machine-precision against `plancklens 'ptt'`
     (max |ratio−1| ≈ 5e-8 in bulk-L, 9e-5 at L > 2000). Locked in by
     `tests/test_delensing.py::TestN0AgainstPlancklens`.
   - **EE/EB flat-sky**: implicitly validated (EE matches plancklens at
     high L to <1%; geometric flat-vs-full factor (L+1)²/L² explains
     the low-L gap).
   - **EE/EB/TB full-sky**: validated against `plancklens
     'pee' / 'p_eb' / 'p_tb'` (symmetrized parity-odd variants for
     EB / TB) to <1e-3 in bulk-L. Locked in by
     `tests/test_delensing.py::TestN0{EE,EB,TB}AgainstPlancklens`.
   - **TE full-sky**: validated against `plancklens 'p_te'`
     (symmetrized) to **<6e-2** in bulk-L = (10, 1800). Deliberately
     looser than the <1e-3 gate above for a documented structural
     reason: `_compute_n0_te_fullsky` implements OkaHu 2003 Table I's
     *single-projection* spin-mixed response (spin-2 on E leg via
     `wigner3j_vectorized(m1=-2, m2=0)`, spin-0 on T leg via
     `wigner3j_000_vectorized`, summed and squared together as
     `(f_2 + f_0)^2`), with the spin-2 leg carrying the parity-even
     mask. Plancklens `'p_te'` is the symmetric estimator `g_pte +
     g_pet` whose variance carries an additional cross-Wick term
     `2 Cov(pte, pet)` that the single-projection form does not
     reproduce. With `fal['te']=0` the cross term is non-zero
     (`cls_ivfs[te] = cl_te / (C_TT_tot * C_EE_tot)` is non-zero),
     contributing the ~5% structural floor across mid-L. The C_TE
     zero-crossings near l~1850 amplify this to 10-20% relative
     residual where the response amplitude vanishes — the bulk-L
     band stops at L=1800 to keep the test gate informative.
     Production `compute_n0_te(fullsky=True)` keeps HO02 Eq. 13's
     diagonal-approximation filter `1/(C_TT*C_EE + C_TE^2)`; the
     test calls with `te_filter='strict_diagonal'` to align with
     plancklens's `fal['te']=0` apples-to-apples. Per
     `compute_n0_te`'s docstring TE contributes ~1-2% to N_0^MV at
     space-experiment noise levels, so the 5% TE residual propagates
     as <0.1% on N_0^MV and <1% on A_L for realistic delensing
     efficiencies — below the level where it would shift any sigma(r)
     decision, so **the full-sky path is production-grade for space-
     mission applications** (where the reionization bump dominates
     the sigma(r) constraint and the `(L+1)^2/L^2` flat-vs-full
     geometric correction matters at low L). Flat-sky remains the
     `iterate_delensing` default for runtime (~5x faster) but is no
     longer the math/physics preference. Path to closure (recovering
     <1e-3 like the other estimators): port `plancklens.nhl._get_nhl`'s
     leg-pair Wick logic to harmonic space; the leg-construction half
     is already in `augr/_qe.py` (43 tests bit-exact vs plancklens).
     Deferred — pairs naturally with future GMV / iterative-N_0 work.
     Full diagnosis in `scripts/n0_validation/derivation.md` "TE
     structural residual" section.

     The earlier "5-20x off in bulk-L" residual was a sign error in
     `augr.wigner._sg_b` (Schulten-Gordon recursion coefficient): the
     m_3 term had the wrong sign per SG 1975 Eq. 5, so the recursion
     produced wrong values for any (m_1, m_2) with m_3 = -(m_1+m_2)
     != 0. Affected every full-sky polarization path (EE, EB, TE,
     TB N_0; the lensing kernel). TT was unaffected because it uses
     `wigner3j_000_vectorized` (closed-form Racah path).
     `tests/test_wigner.py` locks in the sympy-truth regression so
     this can't reappear silently. The earlier "missing
     spin-lowering branch" / "two-branch fix" diagnosis in
     `scripts/n0_validation/derivation.md` was a misdiagnosis;
     `derivation.md` rewritten with the actual fix and traceback.

     **`iterate_delensing` defaults to `fullsky=False` (flat-sky
     path is unaffected and used in production forecasts); the bug
     only affected opt-in `fullsky=True` polarization N_0 and the
     full-sky lensing kernel's per-cell values, neither of which any
     production forecast consumed. The bug fix doesn't invalidate
     any prior sigma(r) numbers.**

     Foundation for any future plancklens-style GMV / iterative-N_0
     work: numpy-only port of plancklens's QE-leg machinery
     (`qeleg`, `qe`, `get_qes`, `qe_simplify`, `qe_proj`,
     `get_resp_legs`, `get_covresp`, `spin_cls`, `get_spin_raise/lower`)
     lives at `augr/_qe.py`, validated bit-for-bit by
     `tests/test_qe.py` (43 tests, all passing under
     PYTHONPATH=~/cmb/plancklens). Not consumed by any production
     code today; dormant infrastructure for the GMV reach.

   **plancklens-wrapper convention gotcha.** `run_plancklens.py` must
   apply plancklens's standard `fal[s][:l_min] = 0` and
   `cls_w[s][:l_min] = 0` (cf. `plancklens/n0s.py:137`); without it,
   `nl_tt[0:2] = 1.5e-7` floor → `fal[0:2] = 6.5e6` pollutes
   plancklens's QE response with monopole/dipole modes (inflates
   `r_gg` by ~1000× at low L, gives spuriously small N_0). Apparent
   "8000× discrepancy at L=2" earlier in the validation was this
   wrapper bug, not augr.

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
- **`optimize.sigma_r_from_*` vs `FisherForecast.sigma` agree to fp64
  precision.** Both route through ``fisher._fisher_from_blocks``
  (``jnp.linalg.solve`` per bin, then a ``0.5 * (F + F^T)``
  symmetrisation). The previous "few-percent disagreement" caveat was
  inverted: at PICO-class conditioning (cov_b cond ~10^28 at ell=2),
  the legacy ``eigh + (s>0)`` clip biased F upward by 5-44% per bin
  -- it face-valued tiny positive eigenvalues that were fp64 rounding
  artifacts, contributing fictitious Fisher info via ``s_inv = 1/s``
  for ``s ~ 1e-14``. ``jnp.linalg.solve`` is essentially exact at fp64
  (validated against mpmath @ 30 dps on bins 0/1/12/25/37 of the PICO
  21-channel moment-FG fixture: rel error 1e-13 to 6e-3 across the ell
  range, with bin 0 setting the ~0.6% floor). See
  ``tests/test_fisher_stability.py`` for the locked-in regression.

- **Open numerical follow-ups (PICO conditioning).** Two failure modes
  remain after the unification, both tracked as xfail in the stability
  test file:
  - **JIT vs eager drift.** Wrapping ``sigma_r_from_channels`` in an
    outer ``jax.jit`` reorders XLA fusion of the covariance build and
    drifts σ(r) by ~10% from eager evaluation. The inner Fisher
    primitive is unchanged; the drift is upstream in the cov_b
    assembly under fusion.
  - **Gradient stability for L-BFGS-B.** Forward solve is essentially
    exact, but ``d/dθ A^{-1} J`` at cond~10^28 amplifies roundoff in
    the backward pass through two ``solve`` calls per bin. L-BFGS-B
    line search aborts with status=ABNORMAL even though the function
    values do decrease through the trial steps. Likely fixable with a
    prewhitening pass for the backward direction only; not addressed
    in the unification PR.

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
