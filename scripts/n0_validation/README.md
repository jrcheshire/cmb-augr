# n0_validation

Cross-validation of `augr.delensing` lensing-reconstruction noise N_0
against [`plancklens`](https://github.com/carronj/plancklens) at the
LiteBIRD-PTEP fiducial config. The result is a small reference NPZ
checked into `data/n0_reference_litebird.npz` and a tolerance test in
`tests/test_delensing.py::TestN0AgainstPlancklens`.

This directory mirrors the layout of `scripts/falcons_validation/`:
heavy regen pipeline lives here (gitignored), the only file under git is
this `README.md`. The reference NPZ produced by the pipeline lives at
`data/n0_reference_litebird.npz`, next to the existing CAMB templates,
so the lightweight test can run in CI without the optional `plancklens`
dep.

## Conventions

`augr.delensing.compute_n0_*` follows Hu & Okamoto 2002:

  - QE response uses **unlensed** CMB spectra
  - QE filter denominator uses **lensed + noise** total spectra
  - Output `N_0(L)` is in `C_L^{phi phi}` units (dimensionless)
  - MV is the diagonal `1 / Sum 1/N_0_alpha` (HO02 Eq. 22), no
    cross-correlation between estimators

`plancklens.nhl.get_nhl` must be invoked under the same conventions
(`cls_weight = unlensed`, `cls_ivfs = lensed + noise`) for the
comparison to be apples-to-apples. `run_plancklens.py` sets these up
automatically.

## Layout

  - `run_plancklens.py`   - heavy driver. Builds the LiteBIRD-PTEP
    `nl_*` and `cl_*` arrays from `augr` + `augr.delensing.load_lensing_spectra`,
    calls `plancklens.nhl.get_nhl` for `{TT, TE, EE, EB, TB}`, MV-combines,
    writes a self-describing NPZ.
  - `compare.py`         - loads the NPZ, runs `augr.delensing.compute_n0_*`
    on the same inputs in flat-sky and (optionally) full-sky modes,
    prints max relative error per L band, plots ratio curves to
    `figures/`.
  - `figures/`           - diagnostic plots (gitignored).
  - `n0_reference_litebird.npz` - regen output (gitignored here; the
    accepted version lives at `../../data/n0_reference_litebird.npz`).

## Reproducing the validation

`plancklens` is not part of the augr pixi env. Easiest path:

`plancklens` is GitHub-only (Julien Carron's repo
[carronj/plancklens](https://github.com/carronj/plancklens), not on
PyPI). It depends on `numpy`, `scipy`, `healpy`, `camb`, and `pyfftw`.

```bash
# 1. Set up an env with both plancklens and augr (one option):
conda create -n n0val python=3.12
conda activate n0val
cd ~/cmb/cmb-augr
pip install -e .                 # editable augr
pip install "git+https://github.com/carronj/plancklens.git"
# (or clone + pip install -e . if you want to poke at the source)

# 2. Generate the plancklens reference:
python scripts/n0_validation/run_plancklens.py
# -> writes scripts/n0_validation/n0_reference_litebird.npz

# 3. Compare against augr (run in the augr pixi env, OR same env as above):
pixi run python scripts/n0_validation/compare.py
# -> prints max-rel-err per estimator per L band
# -> writes scripts/n0_validation/figures/n0_ratios.png

# 4. Once the comparison looks healthy and tolerances are decided, copy
#    the reference into the in-tree data directory:
cp scripts/n0_validation/n0_reference_litebird.npz data/

# 5. Run the lightweight tolerance test:
pixi run pytest tests/test_delensing.py::TestN0AgainstPlancklens -v
```

If `plancklens` and `augr` cannot co-install in the same env, split
into two NPZs:

  1. In the augr env, dump the inputs (`nl_*`, `cl_*`, `Ls`) to
     `inputs.npz` (small patch to `run_plancklens.py:build_inputs`).
  2. In a plancklens-only env, load `inputs.npz`, call
     `plancklens.nhl.get_nhl`, write `n0_plancklens.npz`.
  3. Back in the augr env, run `compare.py` against
     `n0_plancklens.npz`.

## plancklens version compatibility

The `qe_key` strings (`'ptt'`, `'pee'`, `'peb'`, `'pte'`, `'ptb'`) and
the dict layouts for `cls_weight` / `cls_ivfs` have moved a couple times
between plancklens releases. If `run_plancklens.py` errors on a key:

  - Check `plancklens.utils.qe_keys()` (or grep
    `~/<env>/lib/python*/site-packages/plancklens/` for `qe_key`).
  - Newer versions sometimes split `'p_p'` into separate `'pee'` /
    `'peb'`; the script targets the post-split spelling.

The version pin and date of last regen are recorded in the reference
NPZ's `_metadata` field, so you can always trace back what was used to
produce the in-tree reference.

## Last regen

  - Date: 2026-05-06 (initial pipeline build; comparison NOT YET locked in)
  - plancklens version: HEAD of github.com/carronj/plancklens (editable)
  - Author: jc

## Open thread: TT discrepancy (2026-05-06)

The pipeline runs end-to-end and the polarization MV agrees with augr to
~5-25%. **TT shows a residual ~2x discrepancy at high L that is not
yet explained.** Status table:

| Comparison                        | Ratio plancklens/augr (L=2..658) | Status            |
|-----------------------------------|----------------------------------|-------------------|
| `'p_p'` vs MV(EE,EB)              | 0.64 -> 1.00                     | OK (<25%)         |
| `'p'`   vs full MV                | 0.06 -> 1.13                     | low-L mystery     |
| `'ptt'` vs `compute_n0_tt`        | 0.05 -> 0.36                     | **2.6x off open** |

What the investigation has ruled out:

  - augr's flat-sky N_0 formulas are structurally correct per Maniyar
    et al. 2021 (arXiv:2101.12193) Eq 11-12. The (1+delta_XY) factor of
    2 in the same-field denominator matches exactly, the integration
    measure matches Eq 4, and the response form matches Table I.
  - Different-estimator hypothesis (HO02 vs OH03 vs GMV vs SQE):
    Maniyar Fig 1 shows HO02-vs-OH03 agree to <0.5%, and Fig 2 shows
    HO02/SQE/GMV all within ~10% of each other. None of these can
    account for a 2.6x discrepancy.
  - lensed-vs-unlensed cls in the QE response: at most a few percent
    effect at relevant scales (and goes the wrong direction).
  - Joint-vs-diagonal TE filter in `fal`: tested by setting
    `fal['te'] = 0` in `run_plancklens.py`. Accounts for ~20% of the
    TT discrepancy at most, not the dominant ~2x.

What's still open:

  - **Hand-derivation of plancklens's `_get_nhl` + `get_response`
    coupling at a single (L, l1, l2) point**, comparing the resulting
    integrand against augr's `compute_n0_tt` at the same point. This
    would localize whether the residual is in the response
    normalization, the cls_ivfs construction, or the QE coupling
    coefficient.
  - **Controlled-input test**: set Cl = constant, Nl = 0 (signal-only,
    no noise) where the analytic answer is known, and compare both
    codes against the analytic value. Distinguishes formula bugs
    from input-shape effects.
  - **augr full-sky paths show low-L flat-vs-full inconsistency**
    (TT full/flat = 432 at L=2, EE 10x, EB 1e-4) for all three
    estimators in different directions. The 4pi/16pi prefactor
    inconsistency between TT (1/(4pi)) and EE/EB (1/(16pi)) was
    investigated -- changing TT to 1/(16pi) made things worse, so
    that is NOT the right fix. Unclear what the actual full-sky bug
    is. Convergence at very high L (~1) is the right limit, so the
    issue is low-L specific.

This investigation should resume by: (1) running
`run_plancklens.py` to produce a reference NPZ in this directory, (2)
running `compare.py` and `inspect_npz.py` to reproduce the diagnostic
table above, then (3) doing the hand-derivation step. The pipeline is
ready; the open work is the math.

(Update the section above this one once the comparison is locked in.)
