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

Closed out (2026-05-06)
-----------------------

All four action items from the previous session were knocked out:

  1. ``run_plancklens.py`` now applies the standard plancklens
     convention ``fal[s][:l_min] = 0`` and ``cls_w[s][:l_min] = 0``
     (cf. ``plancklens/n0s.py:137``). New CLI flag ``--l-min``
     defaulting to 2.
  2. ``run_plancklens.py`` now passes ``cls_w = unlensed C_l`` to
     ``qresp.get_response``, matching augr's HO02 response convention
     exactly. Removes the ~5-10% "lensed-vs-unlensed in response"
     systematic that the previous comment had accepted.
  3. ``augr/delensing.py:_fullsky_L_samples`` drops the
     ``len(Ls)`` cap on the internal sample grid and now always
     includes the input ``Ls`` directly, eliminating the
     sparse-input log-interp error.
  4. NPZ regenerated with the fixes above and copied to
     ``data/n0_reference_litebird.npz``.
     ``tests/test_delensing.py::TestN0AgainstPlancklens`` is now an
     active slow test gated on TT (the only apples-to-apples
     comparison; PP and MV differ by the diagonal-vs-joint MV
     formulation, which is real physics, not a tolerance bug).

Final agreement on TT, L in [2, 3000]:

| L band         | max ``\|ratio - 1\|`` |
|----------------|-----------------------|
| 2 - 9          | 1.08e-6               |
| 10 - 200       | 5.02e-8               |
| 200 - 2000     | 2.52e-10              |
| 2000 - 3000    | 9.42e-5               |

Bulk and high-L test tolerances locked at 1e-3 (~4 orders of
magnitude headroom on the actual numbers, just to catch regressions
without being noisy).

The investigation did NOT find a bug in either augr or plancklens'
QE math. All discrepancies traced back to mismatched conventions
between the wrapper and plancklens's official recipe.

How to reproduce
----------------

```bash
# Augr-only flat-sky controlled-input test (passes in pixi env):
pixi run python scripts/n0_validation/controlled_input_test.py

# Full augr + plancklens comparison (needs the n0val conda env):
conda run -n n0val python scripts/n0_validation/controlled_input_test.py \
    --fullsky --plancklens
```
