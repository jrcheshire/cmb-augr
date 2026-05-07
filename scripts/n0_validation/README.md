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

RESOLVED (2026-05-06, end of session): the discrepancy is a wrapper bug
-----------------------------------------------------------------------

What looked like a "catastrophic 8410x discrepancy at L=2 on realistic
LiteBIRD spectra" between augr's full-sky path and plancklens was a
bug in ``run_plancklens.py``'s ``fal`` construction. Both codes are
correct.

**The wrapper bug.** ``run_plancklens.py:plancklens_n0`` builds

    fal['tt'] = _safe_inv(cl_tt_len + nl_tt)

without zeroing the result below ``l_min``. ``augr.combined_noise_nl``
returns ``nl_tt[0:2] = 1.5e-7`` (a small but non-zero floor at the
monopole / dipole), and ``cl_tt_len[0:2] = 0`` from CAMB, so
``fal[0:2] = 1 / 1.5e-7 = 6.5e6`` -- enormous. plancklens's
``get_response`` then includes the l=0,1 modes in the QE response
calculation, inflating ``r_gg`` at low L by ~1000x and giving a
correspondingly tiny N_0 (since N_0 = n_gg / r_gg^2 = 1 / r_gg when
the unbiasedness condition holds).

augr's ``compute_n0_*`` enforces ``l_min=2`` via the lower bound of
its ``l1`` sum and the ``l2_min=l_min`` argument to
``wigner3j_000_vectorized``, so the l=0,1 modes never enter the augr
calculation. The two codes were silently using different effective
``l_min`` for the same input.

The fix is one line: after building ``fal``, apply the plancklens
convention ``fal[s][:max(1, lmin_ivf)] = 0`` (cf.
``plancklens/n0s.py:137``). ``cls_weight`` should be similarly
zeroed below ``lmin``, though in our case ``cl_unl[0:2]=0`` already
masks it on that side.

**Verification.** With the lmin filter applied to plancklens at the
LiteBIRD-PTEP fiducial config and dense input Ls covering [2, 1000]:

| L    | augr full   | plancklens (fixed) | ratio    |
|------|-------------|--------------------|----------|
| 2    | 1.6718e-07  | 1.6718e-07         | 0.999999 |
| 5    | 5.8098e-09  | 5.8098e-09         | 1.000000 |
| 10   | 4.2274e-10  | 4.2274e-10         | 1.000000 |
| 20   | 2.8998e-11  | 2.8998e-11         | 1.000000 |
| 30   | 5.9957e-12  | 5.9702e-12         | 1.004273 |
| 1000 | 5.9014e-17  | 5.9014e-17         | 1.000000 |

Max ``|ratio - 1|`` over all dense Ls in [2, 1000]: **7.2e-3 (<1%).**
augr full-sky and plancklens agree everywhere.

**Conclusions on each piece:**

  - augr **flat-sky** TT formula: correct to machine precision on
    closed-form input (controlled-input test).
  - augr **full-sky** TT formula: correct. The Smith-2012 substitution
    ``L.l1 -> alpha_1 = [L(L+1) + l1(l1+1) - l2(l2+1)] / 2`` is
    mathematically equivalent to plancklens's spin-raised /
    Wigner-convolved formulation; both reproduce the controlled-input
    closed forms and agree on realistic LiteBIRD-PTEP inputs to <1%.
  - plancklens TT QE: correct.

**Bug B (real but mild) remains:** ``_fullsky_L_samples`` uses
``n_sample = min(len(Ls), max(50, L_max // 20))`` log-spaced samples
followed by log-interp. With sparse input ``Ls`` (e.g. 7 points), the
sample grid is sparse and log-interp adds ~10-20% error at
intermediate L. The fix is to make ``n_sample`` ignore ``len(Ls)``
(always use a reasonable internal grid). All references to "augr full
~14% high at L=30..300 on the constant-C controlled input" in
earlier session notes were this artefact.

Status (2026-05-06, after expanding to EE/EB)
---------------------------------------------

The TT validation is locked in to machine precision; the EE / EB
validation revealed real bugs in augr's full-sky polarization paths.
The test suite now tracks all three estimators with honest expected
status:

| component        | status                                  |
|------------------|-----------------------------------------|
| TT full-sky      | PASS at <1e-3 vs plancklens 'ptt'       |
| EE full-sky      | XFAIL: 5-20x larger than plancklens 'pee' |
| EB full-sky vs flat at low L | XFAIL: 10000x smaller at L=2 |

Final agreement on **TT**, L in [2, 3000]:

| L band         | max ``\|ratio - 1\|`` |
|----------------|-----------------------|
| 2 - 9          | 1.08e-6               |
| 10 - 200       | 5.02e-8               |
| 200 - 2000     | 2.52e-10              |
| 2000 - 3000    | 9.42e-5               |

Pipeline fixes from the previous session, all closed:

  1. ``run_plancklens.py`` now applies the standard plancklens
     ``fal[s][:l_min] = 0`` and ``cls_w[s][:l_min] = 0`` convention
     (cf. ``plancklens/n0s.py:137``). New ``--l-min`` flag (default 2).
  2. ``run_plancklens.py`` now passes ``cls_w = unlensed C_l`` to
     match augr's HO02 response convention exactly.
  3. ``augr/delensing.py:_fullsky_L_samples`` drops the ``len(Ls)``
     cap and includes input Ls directly. ``_compute_n0_eb_fullsky``
     also now uses this helper (was inline before).
  4. NPZ regenerated with ``n0_pee`` added; copied to
     ``data/n0_reference_litebird.npz``.

EE / EB full-sky bugs (the new open thread)
-------------------------------------------

**Diagnostic, augr full-sky vs plancklens at LiteBIRD-PTEP, L=2..1000:**

| L    | augr_ee_full | plancklens_pee | ratio | augr_eb_full | augr_eb_flat | full/flat |
|------|--------------|----------------|-------|--------------|--------------|-----------|
| 2    | 2.34e-06     | 1.14e-07       | 20.5  | 4.62e-12     | 4.04e-08     | 0.0001    |
| 10   | 2.24e-09     | 3.06e-10       | 7.3   | 8.71e-14     | 6.49e-11     | 0.0013    |
| 100  | 3.47e-13     | 5.07e-14       | 6.8   | 8.95e-16     | 6.25e-15     | 0.143     |
| 300  | 3.88e-14     | 1.16e-14       | 3.3   | 8.02e-17     | 7.83e-17     | 1.025     |
| 1000 | 1.40e-15     | 2.92e-16       | 4.8   | 1.08e-17     | 1.02e-17     | 1.054     |

EE full-sky is **5-20x too large** at all L; EB full-sky is **0.0001x
flat-sky at L=2** but converges to flat-sky at L >= 300.

**Why we missed this earlier.** The constant-C controlled-input test
(``controlled_input_test.py``) only validated the SUM
``alpha_1 + alpha_2 = L(L+1)``, not the individual alpha factors.
With constant C, ``(alpha_1 C + alpha_2 C) = (alpha_1 + alpha_2) C =
L(L+1) C`` regardless of which form of alpha is used. The bug is
invisible in that test by construction; it shows up only when C is
non-constant.

**Hypothesis on the bug.** augr's full-sky uses the substitution
``L.l_i -> alpha_i = [L(L+1) + l_i(l_i+1) - l_j(l_j+1)] / 2`` for ALL
spins. This is correct for spin-0 (TT) where the gradient eigenvalue
is sqrt(l(l+1)). For spin-2 (EE, EB), the gradient on the polarization
tensor uses spin-raising / spin-lowering eigenvalues
``sqrt((l-2)(l+3))`` (raise) and ``sqrt((l+2)(l-1))`` (lower), which
differ from spin-0's sqrt(l(l+1)) by terms of order ``2/l``. The
correct full-sky alpha for spin-2 likely uses ``(l+2)(l-1)`` and/or
``(l-2)(l+3)`` factors instead of ``l(l+1)``. Reference: plancklens
``utils_spin.get_spin_raise/lower``.

**Why the lensing kernel still works.** The forward direction
(C_phi -> C_BB via parity-odd spin-2 coupling) is validated against
CAMB at <1% (``TestFullSkyKernel::test_matches_camb_low_ell``). The
kernel is dominated by the geometric ``geom = -l_B(l_B+1) +
l_E(l_E+1) + L(L+1)`` factor and survives even with the wrong-spin
alpha because the SUM of the alphas is right (L(L+1) terms cancel
correctly in the net). Only the QE inverse, weighted by per-cell
``C_EE_l_E^2``, sensitive to the individual alpha factors, exposes
the bug.

Action items (next session)
---------------------------

  1. Re-derive the full-sky polarization QE response from first
     principles or from a clean reference. Best candidates:
     Smith-Hanson-Lewis 2012 (arXiv:1205.0474) for the EB pathway,
     Hu & Okamoto 2002 (astro-ph/0111606) Appendix A, Lewis-Pratten
     2020 / carronj papers. Goal: an explicit closed-form expression
     for the spin-2 alpha analog that augr can drop in.
  2. Implement and test against ``plancklens 'pee'`` (already
     wired up via the regenerated NPZ). The xfail tests in
     ``TestN0EEAgainstPlancklens`` should start passing once the fix
     lands.
  3. The EB bug at low L may have the same root cause; once the
     spin-2 alpha is correct, retest EB full-sky vs flat-sky at low L.
     If still off, investigate further.
  4. After both fixes, re-evaluate whether to remove "EXPERIMENTAL"
     from full-sky delensing in the main ``README.md``.

The investigation did NOT find a bug in either augr's flat-sky path
or in plancklens. The EE/EB full-sky bugs are localized to augr's
spin-2 substitution in ``_compute_n0_ee_fullsky`` and
``_compute_n0_eb_fullsky``.

How to reproduce
----------------

```bash
# Augr-only flat-sky controlled-input test (passes in pixi env):
pixi run python scripts/n0_validation/controlled_input_test.py

# Full augr + plancklens comparison (needs the n0val conda env):
conda run -n n0val python scripts/n0_validation/controlled_input_test.py \
    --fullsky --plancklens
```
