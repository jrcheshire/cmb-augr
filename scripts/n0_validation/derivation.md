# Full-sky polarization N_0: derivation and bug localization

Status: 2026-05-07 -- TT / EE / EB / TB full-sky N_0 match plancklens
at the LiteBIRD-PTEP fiducial to <1e-3 in bulk-L (matching the existing
TT tolerance). TE is locked at a deliberately looser 6e-2 bulk-L
gate in (10, 1800); see "TE structural residual" section at bottom.

## TL;DR

The "5-20x off" full-sky EE residual was a **sign error in
`augr.wigner._sg_b`** (the diagonal coefficient of the
Schulten-Gordon Wigner-3j recursion). The original implementation
flipped the sign of the m_3 term relative to Schulten & Gordon 1975
Eq. 5, so the recursion produced wrong values for any (m_1, m_2)
with m_3 = -(m_1+m_2) != 0. TT was unaffected because it uses the
closed-form Racah path (`wigner3j_000_vectorized`) instead of the
recursion. Every other full-sky path (EE, EB, TE, TB N_0; the
lensing kernel) silently used the buggy 3j and gave wrong answers.

The earlier hypothesis that augr was missing the "spin-lowering
branch" of the lensing source action was a misdiagnosis -- the
single-bracket Hu-Okamoto Eq. 14 form was correct all along, given
correct 3j inputs.

## Diagnosis trail

### 1. Initial framing (wrong)

Before the bug was found, the residual was reasoned about by
inspecting plancklens's `qresp.get_qes('pee', ...)` qe list. That
list has 8 qes after `qe_simplify+qe_proj` in two families:

| Family | qes | legb.spin_ou | legb eigenvalue        | Operator branch |
|--------|-----|--------------|------------------------|-----------------|
| A      | 0-3 | 3            | sqrt((l-2)(l+3))       | spin-raise (eth)|
| B      | 4-7 | -1           | sqrt((l+2)(l-1))       | spin-lower      |

This led to a hypothesis that augr's single-bracket Hu-Okamoto Eq. 14
implementation was missing the spin-lowering branch of the lensing
source action

    delta(_2 X) = -1/2 [alpha_+1 eth(_2 X) + alpha_-1 ethbar(_2 X)]

and that the "two-branch fix" required augr to mirror plancklens's
`nhl._get_nhl` formula explicitly. Several phases of work on that
plan happened (port of qe-leg machinery to `augr/_qe.py`, sketches
of a wignerc 3j-table reduction). Then the actual bug was found.

### 2. Actual bug (right)

Schulten-Gordon 1975 Eq. 5 gives the three-term recursion for the
3j as a function of j_3 (with j_1, j_2, m_1, m_2, m_3 fixed):

    A(j+1) f(j+1) + B(j) f(j) + A(j) f(j-1) = 0

with

    A(j) = sqrt[(j^2 - (j_1-j_2)^2)((j_1+j_2+1)^2 - j^2)(j^2 - m_3^2)]
    B(j) = (2j+1) [j(j+1)(m_2 - m_1) - m_3 (j_1(j_1+1) - j_2(j_2+1))]

The augr recursion divides through by j(j+1) (so the coefficient on
f(j-1) is `a_j = E(j)/j` and the coefficient on f(j) is `b_j` with a
1/(j(j+1)) factor folded in). After that division, the augr
implementation of `b_j` reads (`augr/wigner.py`):

```python
return ((2.0 * j_f + 1.0)
        * (m3 * (j1 * (j1 + 1) - j2 * (j2 + 1))     # WRONG: should be -m3
           - (m1 - m2) * j_f * (j_f + 1.0))
        / denom)
```

Compared to the SG form (after the j(j+1) division), the m_3 term
has the **wrong sign**. For m_3 = 0 the bug is invisible. For
m_3 != 0 the recursion produces wrong values.

Numerical impact at the (j_1=2, j_2=1, m_1=0, m_2=-1, m_3=1) test
case:

    sympy reference:  3j(2,1,1; 0,-1, 1) = 0.18257
                      3j(2,1,2; 0,-1, 1) = 0.31623
                      3j(2,1,3; 0,-1, 1) = 0.23905

    augr buggy:       (-0.31623, 0.18257, 0.27603)
    augr fixed:       ( 0.18257, 0.31623, 0.23905)  matches sympy to 1e-17

Same magnitude of error (~50% absolute) for the production
(m_1=-2, m_2=0, m_3=2) signature used by every fullsky polarization
N_0 path.

### 3. Why TT was unaffected

`_compute_n0_tt_fullsky` calls `wigner3j_000_vectorized` which uses
the closed-form Racah formula (in `augr.wigner.wigner3j_000`) and
never touches the recursion. EE/EB/TE/TB call the recursion-based
`wigner3j_vectorized`. The TT validation against plancklens
(<5e-8 in bulk) was orthogonal to the bug.

### 4. Why the kernel still matched CAMB

`_lensing_kernel_fullsky` also uses the buggy recursion (m_1=-2,
m_2=0, m_3=2). Pre-fix, `tests/test_delensing.py::TestFullSkyKernel
::test_matches_camb_low_ell` was passing at the 1.5% tolerance --
which seems implausible given the recursion was systematically off
by ~50% in individual 3j cells. The explanation is in the test
docstring:

> Residual ~0.8% error is from the first-order gradient
> approximation (CAMB uses the full resummed lensing calculation).

The pre-fix kernel was wrong by ~50% per cell BUT the (l_E, l_B)
sum that produces the BB lensing prediction is dominated by the
(2 l_E + 1) Racah-orthogonality identity that survives the per-cell
error in a delicate way. The post-fix kernel still passes the same
test, with a slightly different (and now physically correct)
residual. Investigated by running the kernel test pre-fix and
post-fix; both pass at the 1.5% tolerance.

## Fix

```diff
 def _sg_b(j, j1, j2, m1, m2, m3):
     ...
     return ((2.0 * j_f + 1.0)
-            * (m3 * (j1 * (j1 + 1) - j2 * (j2 + 1))
+            * (-m3 * (j1 * (j1 + 1) - j2 * (j2 + 1))
                - (m1 - m2) * j_f * (j_f + 1.0))
             / denom)
```

Same change to `_sg_b_vec`. Plus two related fixes in
`wigner3j_recurse` and `wigner3j_vectorized` to honor the
|m_1| <= j_1 / |m_2| <= L magnetic-quantum constraints (previously
the n=1 closed-form path returned 1/sqrt(2j_max+1) regardless of
whether the magnetic-quantum constraint was satisfied).

Locked in by `tests/test_wigner.py` (sympy-truth at small-n with
m_3 != 0 cases including the production signature). Pre-fix this
test would have flagged ~0.5 absolute errors immediately.

## Practical impact on production forecasts

`iterate_delensing` defaults to `fullsky=False`, so all production
sigma(r) forecasts use the flat-sky path (validated separately by
`controlled_input_test.py` against the closed-form Hu-Okamoto 2002
N_0). The bug only affected the opt-in `fullsky=True` path and the
full-sky lensing kernel's per-cell values; the kernel summed-up
prediction matched CAMB pre-fix coincidentally (see "Why the kernel
still matched CAMB" above).

No production forecasts need to be re-run.

## Validation

Post-fix at the LiteBIRD-PTEP fiducial config, lmax_ivf=3000,
augr full-sky N_0 vs plancklens (in `scripts/n0_validation/compare.py`):

| Estimator       | bulk-L max rel-err            | bulk-L band      |
|-----------------|-------------------------------|------------------|
| TT              | 5.0e-8                        | (10, 2000)       |
| EE              | <1e-3                         | (10, 2000)       |
| EB              | <1e-3                         | (10, 2000)       |
| TB              | <1e-3                         | (10, 2000)       |
| TE              | <6e-2 (structural floor)      | (10, 1800)       |
| PP (= EE + EB)  | 5.6e-7                        | (10, 2000)       |

`tests/test_delensing.py` has `TestN0{TT,EE,EB,TB}AgainstPlancklens`
locked at <1e-3 bulk tolerance and `TestN0TEAgainstPlancklens` at
6e-2 in (10, 1800). The previous EE / EB xfail markers and the EB
flat-vs-full proxy test are removed.

## TE structural residual

The TE single-projection structural fix (2026-05-07) takes the
residual from 51x (the previous spin-0-on-both-legs form) down to
~5% across mid-L. The remaining 5% is structural, not numerical:
plancklens's `'p_te'` is the *symmetric* estimator and augr's
`_compute_n0_te_fullsky` is OkaHu Table I's *single-projection* form.

### The fix

Per OkaHu 2003 Table I, TE is spin-mixed:

    f^TE(l1, l2, L) = C^TE(l1) * _2F_{l2 L l1} * eps_TE
                    + C^TE(l2) * _0F_{l1 L l2}

(spin-2 on the E leg, spin-0 on the T leg). The previous
`_compute_n0_te_fullsky` used the spin-0 (m=0,0,0) Wigner-3j on
*both* legs and squared the entire response together as
`(C(l1)*alpha1 + C(l2)*alpha2)^2 * w000^2`. The fix:

* Compute both `w000` (spin-0) and `w2F` (spin-2, m=-2,0,2) on a
  shared (l1, l2) grid. The two `wigner3j_*` functions agree on
  l2-grid extent when called with matching `l_min/l_max` for `l_min
  >= 2` (the spin-2 path internally clamps `l2_min = max(l_min,
  |m_3|)` = `max(l_min, 2)`).
* Build two response terms with the spin-2 leg carrying the
  parity-even mask (since `w2F` does not vanish for L+l1+l2 odd
  whereas `w000` does):

      f_2 = C(l1) * alpha1 * pf * w2F * even_mask
      f_0 = C(l2) * alpha2 * pf * w000

* Form `(f_2 + f_0)**2` and integrate. The cross term `2*f_2*f_0`
  is essential at low L; without it (variant `f_2**2 + f_0**2`)
  the L=10 residual blows up by ~80x. With it, the bulk-L
  agreement vs plancklens is at the few-percent structural floor.

Bracket / prefactor convention: spin-0 form (`alpha = bracket/2`,
`pf = sqrt(...(2L+1)/(4 pi))`) used uniformly on both terms.
`pf * alpha` is numerically identical in the spin-0 and spin-2
conventions of OkaHu Eq. 14 (cf. `_compute_n0_ee_fullsky` lines
761-770).

### What the 5% is

Plancklens `p_te` is not a single-projection QE -- it is the
symmetric estimator `g_p_te = g_pte + g_pet` (per
`plancklens/qresp.py` and `augr/_qe.py:316-369` which is bit-for-bit
validated against plancklens). Its variance is

    Var(g_p_te) = Var(pte) + Var(pet) + 2 Cov(pte, pet)
                = 2 Var(pte) + 2 Cov(pte, pet)        (by symmetry)

`Cov(pte, pet)` is a Wick contraction between *different* QEs --
the T leg of `pte` is paired with the T leg of `pet` (which sits
at l2 not l1 in `pet`), and similarly for E legs. In plancklens
`nhl._get_nhl`, the double loop `for qe1 in qes1: for qe2 in qes2`
walks over all such cross-pairings, with the leg-pair contraction
mediated by

    cls_ivfs[te] = fal_tt . cl_te . fal_ee = cl_te / (C_TT_tot * C_EE_tot)

In the strict-diagonal filter (`fal['te']=0`), this is the only
non-zero off-diagonal element of `cls_ivfs`. It is non-zero, so
`Cov(pte, pet)` does not vanish, and contributes a few-percent
correction to `N_0^p_te` at all L.

augr's `_compute_n0_te_fullsky` does the harmonic-space sum
`sum_{l1, l2} (f_pte)^2 / denom` which captures `Var(pte)` only --
it does NOT carry the cross-Wick contraction. This is the origin
of the 5% bulk-L residual. EE / EB / TB do not have this issue
because EE is single-projection (no symmetrization) and EB / TB
are parity-odd whose Wick-pair structure is different.

### The C_TE zero-crossing spikes

At L ~ 1887 / 1969, the residual jumps to ~10-20% (as opposed to
~5% in mid-L). These L's are near the C_TE zero-crossings around
l ~ 1850. The response amplitude `f^TE(l1, l2, L)` vanishes when
`C_TE(l1)*alpha1 + C_TE(l2)*alpha2 ~ 0`, and any structural
percent-level difference between augr's form and plancklens's
blows up to large *relative* error there. The bulk-L test band
`L_BULK_TE = (10, 1800)` excludes this region so the test gate
reports the structural floor cleanly.

### Filter and apples-to-apples convention

Plancklens forces `fal['te']=0` (strict-diagonal filter
`1/(C_TT*C_EE)`); augr's production filter is HO02 Eq. 13's
diagonal approximation `1/(C_TT*C_EE + C_TE^2)`. These differ by
~few percent at acoustic peaks. To remove this from the validation
arm, the test calls `compute_n0_te(..., te_filter='strict_diagonal')`
which selects `1/(C_TT*C_EE)` in the full-sky path only. Production
defaults are unchanged.

### Why this matters (or doesn't)

Per `compute_n0_te`'s own docstring, TE contributes ~1-2% to
`N_0^MV` at space-experiment noise levels. A 5% TE residual
therefore propagates as <0.1% on `N_0^MV`. With `A_L = N_0^MV /
(C_phi + N_0^MV)`, `dA_L / A_L = (1 - A_L) * dN_0^MV / N_0^MV`,
so a 0.1% shift on `N_0^MV` is at most a 1% relative shift on
`A_L` even for aggressively-delensed cases (`A_L ~ 0.1`); for
typical `A_L ~ 0.5` it's 0.05% relative. **Below the level where
it would shift any sigma(r) decision, so the full-sky path is
production-grade for space-mission applications.** The reionization
bump (`l <~ 10`) is where space-mission sigma(r) gets its
constraint, and full-sky is the right tool there because the
`(L+1)^2 / L^2` flat-sky-vs-full-sky geometric correction
matters most at low L. Flat-sky remains the `iterate_delensing`
default for runtime (~5x faster) -- a runtime-convenience choice,
not a math/physics preference.

### Path to closure (deferred)

Reaching <1e-3 like the other estimators requires porting
`plancklens.nhl._get_nhl`'s cross-Wick contraction to harmonic
space. The leg-construction is already ported in `augr/_qe.py`
(43 tests bit-exact vs plancklens under `PYTHONPATH=~/cmb/plancklens`).
The remaining work is the variance machinery itself: implement
`get_qes('p_te', ...)` -> sum-over-(qe1, qe2)-pairs of
`Wigner3j-spin-coupled (l1, l2) integrand`, mirroring `nhl.py:65-97`.
This naturally also fixes any related residual on the multifrequency
QE / iterative-N_0 path, so it pairs well with future GMV work.

## References

- Schulten & Gordon 1975, J. Math. Phys. 16, 1961 (the recursion
  whose Eq. 5 has the correct sign convention).
- Edmonds 1957, Angular Momentum in Quantum Mechanics (Racah
  formula used by the closed-form path).
- Hu & Okamoto 2002 (`papers/0111606_HuOkamoto_2002_polarization_QE.pdf`)
  -- flat-sky polarization QE.
- Okamoto & Hu 2003 (`papers/0301031_OkamotoHu_2003_fullsky_lensing_QE.pdf`)
  -- full-sky generalization. Eq. 14 building block; Table I per-
  estimator response forms; Eq. 22 parity factors.
- plancklens `utils_qe.py`, `qresp.py`, `nhl.py`,
  `utils_spin.py` -- the qe-leg machinery; ported (numpy-only) to
  `augr/_qe.py` as a foundation for any future plancklens-style
  GMV / iterative-N_0 work, validated bit-for-bit by
  `tests/test_qe.py`. Not consumed by any production code today;
  the `_sg_b` fix alone resolves the immediate validation, so the
  qe-leg port is dormant infrastructure.
