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

New open thread: augr's full-sky path on non-constant inputs
------------------------------------------------------------

Two distinct issues:

**Bug 1 (mild): sparse input Ls + log-interp in ``_fullsky_L_samples``.**
``_compute_n0_*_fullsky`` computes only at ``L_samples`` (a log-spaced
subset, ``n_sample = min(len(Ls), max(50, L_max // 20))``) and then
``np.interp`` in log space onto the requested ``Ls``. With sparse
input ``Ls`` the geomspace gridding is sparse too, and log-interp
adds ~10-20% error between samples. The earlier table's "~14% spike
at L=30" was this artefact: re-running the controlled-input test with
a denser ``--Ls`` arg gives a smooth ratio rising from 1.001 at L=2
to 1.87 at L=1500 (the latter is just the ``L / l_max`` boundary).
plancklens with the same Ls and l_max=2000 sits ~1% below that
through L=300 and converges to augr full at L=600+.

**Bug 2 (severe, structural): non-constant cls give the constant-C
answer at low L.** On realistic LiteBIRD-PTEP spectra at L=2:

| input                                          | augr flat   | augr full   | plancklens |
|------------------------------------------------|-------------|-------------|------------|
| C_unl=real, C_len=real, nl=real (LB-PTEP)      | 3.87e-10    | 1.67e-07    | 1.99e-11   |
| C_unl=real, C_len=real, nl=0                   | 4.16e-08    | 2.17e-08    | --         |
| C_unl=10x,  C_len=real, nl=0   (scale unl)     | 4.16e-10    | 2.17e-10    | --         |
| C_unl=real, C_len=10x,  nl=0   (scale tot)     | 4.16e-06    | 2.17e-06    | --         |
| C0=1, nl=0                                     | 3.93e-07    | 1.74e-07    | 1.74e-07   |
| C0=1, nl=1                                     | --          | 6.98e-07    | 6.98e-07   |
| C0=1, nl=10                                    | --          | 2.11e-05    | 2.11e-05   |

Augr full-sky agrees with plancklens to **4 significant figures** on
**any constant** ``(C0, nl)`` pair (rows 5-7), but on realistic
LiteBIRD inputs (row 1) is **8410x larger** than plancklens at L=2
and roughly equal to the constant-C answer (1.67e-07 vs 1.74e-07).
Augr full also responds correctly to **uniform scaling** of the
realistic spectra (rows 2-4: C cancellation behaves predictably).
Only when the spectra are *non-constant in l* does augr full
diverge from plancklens.

Extending ``l_max=2000 -> 3000`` to match plancklens's lmax_ivf gives
**identical** augr-full numbers (high-l region is killed by huge nl,
not by the cutoff). Forcing ``l_max=3000`` does not fix it.

**Hypothesis (under investigation, 2026-05-06):** augr's full-sky
formula uses the Smith et al. 2012 substitution
``L.l1 -> alpha_1 = [L(L+1) + l1(l1+1) - l2(l2+1)] / 2`` to lift the
flat-sky response into full-sky. plancklens decomposes the same QE
in terms of spin-raising / spin-lowering operators on the legs
(``sqrt(l(l+1))`` factors) convolved through ``uspin.wignerc``. On
constant inputs these are mathematically equivalent (both pass).
On non-constant inputs they may not be, if augr's "alpha
substitution" double-counts or mis-couples the spectrum across legs
in a way that's invisible when Cl is constant. Possible angles:
the (alpha_1 + alpha_2) cancellation leaves the ``f^2 / C_tot^2``
ratio geometric and ~spectrum-independent at low L — which is what
augr full appears to be doing — while the proper spin-raised
formulation retains a non-cancelling spectrum-shape dependence.

Next steps:

  1. Re-derive the full-sky TT QE response from first principles in
     both formulations and check whether
     ``f^TT_alpha = (Cl1 alpha1 + Cl2 alpha2) * w000 * pf`` is in
     fact the right substitution for non-constant Cl, or whether the
     correct full-sky form is a different (non-additive) combination
     of spin-raised legs and the Wigner-3j coupling. References:
     Hu & Okamoto 2002 (astro-ph/0111606) Eq. A14, A18; Smith,
     Hanson, Challinor 2012 (arXiv:1205.0474) for the spin-2 case;
     plancklens ``qresp.get_qes`` + ``uspin.wignerc`` for the
     reference implementation.
  2. Test on a *smooth-but-non-constant* C(l) (e.g. C(l) = 1 + 0.1 l)
     to see at what amount of non-constancy augr full starts to
     diverge — the constant-C agreement is a *measure-zero* check
     and a slope sweep will tell us whether the bug enters at first
     order in dC/dl or only at higher.
  3. Once the augr full bug is understood, check whether the same
     issue affects ``_compute_n0_ee_fullsky`` and
     ``_compute_n0_eb_fullsky`` (the EE/EB versions use the same
     alpha substitution).
  4. Separately, fix Bug 1 (sparse-Ls log-interp) regardless of
     Bug 2's resolution: it is a small but real ~10% systematic on
     internally-sampled L bins that should not exist.

Pre-existing flat-sky path is correct and unaffected by either bug;
no production-path changes are required while this is open. The
"~2.6x discrepancy at LiteBIRD-PTEP" between flat-sky augr and
plancklens is partially geometric (flat-vs-full factor
``(L+1)^2 / L^2``) and partially the same Bug 2 manifestation
when looking at the post-flat-vs-full residual.

The ``compute_n0_ee`` / ``compute_n0_eb`` polarization estimators
were not reproduced at the same level by this test (their controlled-
input answer needs a numerical reference because the
``cos(2 phi_12)`` / ``sin(2 phi_12)`` factors do not collapse for
constant C). The polarization MV ratio "0.64 -> 1.00" reported in the
earlier ratio table is consistent with the same flat-vs-full
geometric factor, so it is likely OK; running the analogous numerical-
reference test for EE/EB is a small follow-up.

How to reproduce
----------------

```bash
# Augr-only flat-sky controlled-input test (passes in pixi env):
pixi run python scripts/n0_validation/controlled_input_test.py

# Full augr + plancklens comparison (needs the n0val conda env):
conda run -n n0val python scripts/n0_validation/controlled_input_test.py \
    --fullsky --plancklens
```
