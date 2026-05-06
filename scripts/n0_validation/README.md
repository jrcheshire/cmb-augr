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

## Status (2026-05-06, after controlled-input test)

The controlled-input test (`controlled_input_test.py`) feeds both codes
``C_TT(l) = C0`` (constant), ``N_l = 0``, ``cl_*_unl = cl_*_len``. In
that limit ``f = C0 L^2`` and ``f * F = L^4 / 2``, giving closed forms

    N_0^{TT, flat}(L) = 4 pi / (L^4 * S),  S = sum_{l=lmin}^{lmax} l
    N_0^{TT, full}(L) ~ 8 pi / (L^2 (L+1)^2 * S'),
        S' = (lmax+1)^2 - lmin^2

(the flat / full ratio at low L is ``2 (L+1)^2 / L^2 * S / S'``, or
``(L+1)^2 / L^2`` for ``l_max >> L`` — geometry, not a bug).

Results at ``l_min = 2``, ``l_max = 2000``:

| L  | augr flat / flat ana | augr full / full ana | plancklens / full ana |
|----|----------------------|----------------------|-----------------------|
| 2  | 1.000000             | 1.000751             | 1.000708              |
| 10 | 1.000000             | 1.003341             | 1.003339              |
| 30 | 1.000000             | 1.144304             | 1.009795              |
| 100| 1.000000             | 1.126336             | 1.033026              |
| 300| 1.000000             | 1.241                | 1.106                 |
| 600| 1.000000             | 1.235                | 1.235                 |

What this resolves:

  - **augr's flat-sky ``compute_n0_tt`` is correct to machine
    precision** on the controlled input. The TT response shape
    (``L.l1 + L.l2 -> L^2``), the factor of 2 in the same-field
    denominator, the ``l1 / (2 pi)^2`` integration measure, and the GL
    phi quadrature all reproduce the closed form. The earlier "~2.6x
    plancklens / augr_flat discrepancy at LiteBIRD-PTEP" is **NOT a
    bug in augr's TT formula**; it is largely the
    flat-sky-vs-full-sky geometric factor (plus interaction with the
    LiteBIRD spectrum shape — see below).
  - **plancklens is computing the correct full-sky N_0** to <1% on
    the constant-C input at low L. Boundary effects from the finite
    upper limit of the ``l1, l2`` sum dominate the residual at L >> a
    few hundred, scaling like ``L / l_max``.
  - The controlled-input test obsoletes the earlier "open" hypotheses
    (HO02 vs OH03 vs GMV, lensed-vs-unlensed in response,
    joint-vs-diagonal TE) for the flat-sky formula. None of those was
    the dominant cause; the dominant cause is that the flat- and
    full-sky formulas differ by a clean geometric factor that does
    NOT vanish at low L.

New open thread: augr's full-sky path at intermediate L
-------------------------------------------------------

On the **constant-C** input augr's ``_compute_n0_tt_fullsky`` is fine
at low L (matches plancklens to ~3 sig figs at L=2..30) but disagrees
with the closed form by ~12-24% at L = 30..300 — and at L = 600 augr
full and plancklens both run away in lockstep at ~+24%, which is the
expected ``L / l_max`` boundary truncation. The augr-vs-plancklens
disagreement at L = 30..300 is therefore real, not boundary, and
is on **augr**'s side (plancklens stays close to the closed form
through that range).

On the **realistic LiteBIRD-PTEP** spectra ``compare.py`` reports
``augr full-sky / plancklens`` max rel-err = **8410 at L in [2, 9]**
and **494 at L in [10, 2000]**, converging to <2% only at L > 2000.
This is wildly bigger than the constant-C residual; something about
how ``_compute_n0_tt_fullsky`` handles non-constant spectra at low L
is broken. The augr **flat-sky** path on the same realistic input
shows max rel-err = 20 at low L, 14 at bulk L, 0.01 at high L — the
flat-vs-full geometric factor accounts for some of the low-L gap,
not all of it.

The ``compute_n0_ee`` / ``compute_n0_eb`` polarization estimators
were not reproduced at the same level by this test (their controlled-
input answer needs a numerical reference because the
``cos(2 phi_12)`` / ``sin(2 phi_12)`` factors do not collapse for
constant C). The polarization MV ratio "0.64 -> 1.00" reported in the
earlier ratio table is consistent with the same flat-vs-full
geometric factor, so it is likely OK; running the analogous numerical-
reference test for EE/EB is a small follow-up.

Next steps to chase the augr full-sky bug:

  1. Add an EE/EB controlled-input test using a high-precision
     numerical reference (no closed form needed -- just integrate the
     same flat-sky formula at very high quadrature against augr's
     ``compute_n0_ee`` / ``compute_n0_eb``). Compare the polarization
     MV against plancklens ``p_p``.
  2. Diagnose ``_compute_n0_tt_fullsky``: the wigner3j_000 evaluation,
     the alpha_1 / alpha_2 (geometric analog of L.l), the (2L+1)
     normalization, and the spectrum lookup at l2 (note the
     ``_fullsky_inv_spectrum`` interpolation, which uses
     ``np.interp`` -- if the realistic LiteBIRD spectra have sharp
     features at low l, this is a candidate for catastrophic
     interpolation error).
  3. Cross-check on the realistic LiteBIRD inputs whether augr full
     stays close to ``plancklens / augr_flat -> (L+1)^2 / L^2 * 1``
     once both bugs are fixed.

How to reproduce
----------------

```bash
# Augr-only flat-sky controlled-input test (passes in pixi env):
pixi run python scripts/n0_validation/controlled_input_test.py

# Full augr + plancklens comparison (needs the n0val conda env):
conda run -n n0val python scripts/n0_validation/controlled_input_test.py \
    --fullsky --plancklens
```
