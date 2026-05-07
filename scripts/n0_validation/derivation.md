# Full-sky polarization N₀: derivation and bug localization

Status: 2026-05-06 (no code changes yet; this is the from-first-principles
re-derivation that will guide the fix to `_compute_n0_ee_fullsky` and friends).

## What we know

- TT full-sky validates against plancklens to <1e-7 in bulk-L
  (`TestN0AgainstPlancklens`).
- EE full-sky vs plancklens 'pee' is 5–20× too large at all L
  (`scripts/n0_validation/compare.py`, README "EE / EB full-sky bugs"
  table). Currently `pytest.mark.xfail(strict=True)`.
- EB "full-sky vs flat-sky at low L" is 0.0001× at L=2, converging to 1
  at L≥300. Currently `pytest.mark.xfail(strict=True)`. **This is not
  yet a comparison against a trusted reference**; flat-sky at L=2 is
  itself unreliable due to (L+1)²/L² curvature corrections, so the
  apparent "EB bug" may be an artifact of the test target.

## The README's hypothesis was wrong

The earlier hypothesis — that augr's full-sky polarization paths are
broken because they use the spin-0 raising eigenvalue `√(ℓ(ℓ+1))`
where spin-2 should use `√((ℓ∓2)(ℓ±3))` — is **not what Okamoto & Hu
2003 actually says**. The paper's Eq. 14 (the canonical full-sky QE
building block) reads

    _{±s}F_{ℓLℓ'} = [L(L+1) + ℓ'(ℓ'+1) − ℓ(ℓ+1)]
                    · √[ (2L+1)(2ℓ+1)(2ℓ'+1) / (16π) ]
                    · ( ℓ  L  ℓ'  ;  ±s  0  ∓s ).

The bracket factor `[L(L+1) + ℓ'(ℓ'+1) − ℓ(ℓ+1)]` is **identical for
all spins**. Spin enters only through the magnetic quantum numbers in
the Wigner-3j: `(0, 0, 0)` for TT/TE legs, `(±2, 0, ∓2)` for the
polarization legs.

The `get_spin_raise/lower` factors `√((ℓ∓s)(ℓ±s+1))` do appear in
plancklens (`utils_spin.py:97-114`), but at a different level: in the
**leg-builder** formulation of the QE response (`qresp.get_resp_legs`),
which factors the response of `_sX → _sX − ½α(ð̄ _sX) − ½α(ð _sX)` into
`leg_a(ℓ)` × `leg_b(ℓ)` arrays before the Gaunt sum. After the Gaunt
sum is performed, the result reduces to OkaHu's Eq. 14 form with the
spin-independent bracket and the spin-dependent Wigner symbol. The
two formulations are equivalent — plancklens has chosen the leg
factorization because it parallelizes neatly across estimators, not
because the bracket itself depends on spin.

## OkaHu 2003 — direct full-sky responses

Translating Table I of OkaHu 2003 (substituting Eq. 14 in for `_sF`):

| QE | f^α(ℓ₁, ℓ₂, L) — parity factor implicit |
|----|------|
| TT | C^TT_{ℓ₁} _0F_{ℓ₂Lℓ₁} + C^TT_{ℓ₂} _0F_{ℓ₁Lℓ₂}   (even) |
| TE | C^TE_{ℓ₁} _2F_{ℓ₂Lℓ₁} + C^TE_{ℓ₂} _0F_{ℓ₁Lℓ₂}   (even) |
| EE | C^EE_{ℓ₁} _2F_{ℓ₂Lℓ₁} + C^EE_{ℓ₂} _2F_{ℓ₁Lℓ₂}   (even) |
| TB | i C^TE_{ℓ₁} _2F_{ℓ₂Lℓ₁}                          (odd)  |
| EB | i [C^EE_{ℓ₁} _2F_{ℓ₂Lℓ₁} − C^BB_{ℓ₂} _2F_{ℓ₁Lℓ₂}] (odd)  |
| BB | C^BB_{ℓ₁} _2F_{ℓ₂Lℓ₁} + C^BB_{ℓ₂} _2F_{ℓ₁Lℓ₂}   (even) |

Note the index ordering inside `_sF_{ℓᵢLℓⱼ}`: the **third** index
ℓⱼ is the one that contributes `+ℓⱼ(ℓⱼ+1)` to the bracket; the **first**
index ℓᵢ contributes `−ℓᵢ(ℓᵢ+1)`. So in the C^EE_{ℓ₁} term, the bracket
is `L(L+1) + ℓ₁(ℓ₁+1) − ℓ₂(ℓ₂+1)`. In the C^EE_{ℓ₂} term it's
`L(L+1) + ℓ₂(ℓ₂+1) − ℓ₁(ℓ₁+1)`.

For parity-even pairs both Wigner symbols `(ℓ₂ L ℓ₁; ±2 0 ∓2)` and
`(ℓ₁ L ℓ₂; ±2 0 ∓2)` are equal up to sign-of-(−1)^{ℓ₁+L+ℓ₂} = +1, so
the two F-terms share a common Wigner factor and the EE response
collapses to

    f^EE(ℓ₁, ℓ₂, L) = C^EE_{ℓ₁} · α(ℓ₁; L, ℓ₂)
                    + C^EE_{ℓ₂} · α(ℓ₂; L, ℓ₁)
                    × √[(2L+1)(2ℓ₁+1)(2ℓ₂+1)/(16π)]
                    × (ℓ₂ L ℓ₁; +2 0 −2)

with **`α(ℓᵢ; L, ℓⱼ) := L(L+1) + ℓᵢ(ℓᵢ+1) − ℓⱼ(ℓⱼ+1)`**.

Crucially: NO division by 2. Every flat-sky-to-full-sky reference I
checked (OkaHu 2003 Eq. 14; Hanson-Challinor-Lewis 2010 review; the
plancklens leg-product expansion when reduced to closed form) uses
this bracket without a /2.

## Where the /2 in augr came from, and why TT is fine

augr's code (`_compute_n0_*_fullsky`) systematically uses

    α₁ := [L(L+1) + ℓ₁(ℓ₁+1) − ℓ₂(ℓ₂+1)] / 2
    α₂ := [L(L+1) + ℓ₂(ℓ₂+1) − ℓ₁(ℓ₁+1)] / 2

with a **prefactor that depends on whether the leg is spin-0 or spin-2**:

| QE     | augr prefactor `pf`              |
|--------|---------------------------------|
| TT, TE | `√[(2ℓ₁+1)(2ℓ₂+1)(2L+1)/(4π)]`  |
| EE     | `√[(2ℓ₁+1)(2ℓ₂+1)(2L+1)/(16π)]` |
| EB, TB | `√[(2ℓ₁+1)(2ℓ₂+1)(2L+1)/(16π)]` |

The `/2` originates from the flat-sky identity
`L · ℓᵢ = ½(|L|² + |ℓᵢ|² − |ℓⱼ|²)` (with `L = ℓ₁ + ℓ₂`), so each `L · ℓᵢ`
in the flat-sky response form gets replaced by a half-bracket. In
flat-sky, `L · ℓᵢ` *itself* is the full response — there is no separate
`(2ℓ+1)/(4π)`-type prefactor.

When the same flat-sky form is "promoted" to full-sky by replacing the
`d²ℓ/(2π)²` measure with a Wigner-3j sum, the canonical replacement is

    (L · ℓᵢ) cos(2(φ₁−φ₂))    →   _2F_{ℓⱼLℓᵢ}
                                = bracket_i · √[…/16π] · w_2

i.e. the **whole** `(L · ℓᵢ) cos(2φ)` factor — not just `L · ℓᵢ` — gets
absorbed into the `_sF` building block. There's no separate
"/2" left over once that absorption is done.

For TT augr's choice produces the right answer because of an
arithmetic coincidence:

    (1/2) · bracket · √[…/(4π)] · w₀₀₀
        = bracket · √[…/(16π)] · w₀₀₀
        = OkaHu's _0F.

The `/2` cancels against `√(16π/4π) = 2`. This is why the TT
controlled-input test matches to machine precision and why the
realistic LiteBIRD-PTEP TT vs plancklens passes <1e-7.

For EE the cancellation does NOT happen because the prefactor is
already 16π:

    augr's f^EE = (α₁ + α₂)·C·pf·w_2  with pf=√[…/16π]
                = ½ · 2L(L+1) · C · √[…/16π] · w_2     (since α₁+α₂ = L(L+1))
                = L(L+1) · C · √[…/16π] · w_2

vs OkaHu's f^EE under the same constant-C / equal-spectra simplification:

    OkaHu f^EE = (bracket₁ + bracket₂)·C · √[…/16π] · w_2
                = 2L(L+1) · C · √[…/16π] · w_2.

Ratio: `f_augr / f_OkaHu = ½` ⇒ `|f|² ratio = ¼` ⇒ `1/N₀ ratio = ¼`
⇒ **augr N₀^EE ~ 4× too large**. This is the leading piece of the 5–20×
discrepancy seen in `compare.py`.

## What about EB?

`_compute_n0_eb_fullsky` is structured differently: it builds a single
combined `geom = L(L+1) + ℓ_E(ℓ_E+1) − ℓ_B(ℓ_B+1)` (note: NOT divided
by 2) and applies the 16π prefactor. Under the unlensed-B
approximation `C^BB ≈ 0`, OkaHu's EB row reduces to

    f^EB = i · C^EE_{ℓ_E} · _2F_{ℓ_BLℓ_E}
         = i · C^EE_{ℓ_E} · [L(L+1) + ℓ_E(ℓ_E+1) − ℓ_B(ℓ_B+1)]
                          · √[(2L+1)(2ℓ_E+1)(2ℓ_B+1)/(16π)]
                          · (ℓ_B L ℓ_E ; +2 0 −2),

which **matches augr's `_compute_n0_eb_fullsky` exactly** (up to a
column-permutation `(−1)^{ℓ_E+L+ℓ_B}` on the Wigner symbol that
squares away in `|f|²`).

Conclusion: **augr's EB full-sky formula is structurally correct.**

The xfail "10000× too small at L=2 vs flat-sky" is comparing to a
flat-sky baseline that has its own L=2 problems (curvature correction
~(L+1)²/L² = 9/4 at L=2). The discrepancy is largely the EXPECTED
curvature correction, not an EB bug. To know if augr's EB is actually
correct, we need a true reference: either plancklens 'peb' (or the
'p_eb' symmetrized variant) under matched conventions, or an MC of
QE-applied-to-Gaussian-sims.

## The fix for EE — Option A applied 2026-05-06

**Drop the /2 on α inside `_compute_n0_ee_fullsky`.** Keep the 16π
prefactor; keep the parity-even mask.

```diff
-alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :]) / 2.0
-alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None]) / 2.0
+alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :])
+alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None])
```

This is principled — it matches OkaHu 2003 Eq. 14 exactly. Empirical
result vs plancklens 'pee' on LiteBIRD-PTEP / l_min=2 / l_max=3000:

| L band     | augr / plancklens (fixed)        |
|------------|----------------------------------|
| L = 2      | 5.12 (was 20.5 before fix)       |
| L = 5      | 2.17 (was ~10)                   |
| L = 10     | 1.83 (was 7.3)                   |
| L = 30     | 1.73                             |
| L ~ 30-200 | 1.7 plateau                      |
| L = 300    | 0.74                             |
| L = 1000   | 1.20                             |
| L = 2876   | 0.83                             |

The fix improves the agreement uniformly by a factor of 4 across L
(consistent with the (1/2)→1 change in the response squaring), but
**leaves a non-constant L-dependent residual**: a ~1.7× plateau in
mid-L plus undershoot/overshoot at high L. This residual is **not**
explained by the simple OkaHu Eq. 14 reading.

## Empirical follow-up tests (2026-05-06): what the residual is NOT

After applying the /2 fix, ran several diagnostic substitutions to
narrow down the remaining factor of ~1.7. **None match plancklens.**

### Eigenvalue substitution (the README's original hypothesis)

The README hypothesised that the bracket should use spin-2 raising or
lowering eigenvalues `(ℓ−2)(ℓ+3)` or `(ℓ+2)(ℓ−1)` instead of the
spin-0 `ℓ(ℓ+1)`. This is **falsified by direct numerical test**:

| eigenvalue substitution | N₀^EE(L=2) / plancklens | … (L=10) | … (L=300) |
|------------------------|------------------------:|---------:|----------:|
| `ℓ(ℓ+1)`               | 5.12                    | 1.83     | 0.74      |
| `(ℓ−2)(ℓ+3)`           | 5.12                    | 1.83     | 0.74      |
| `(ℓ+2)(ℓ−1)`           | 5.12                    | 1.83     | 0.74      |
| `ℓ(ℓ+1) − 2`           | 5.12                    | 1.83     | 0.74      |
| `ℓ(ℓ+1) − 4`           | 5.12                    | 1.83     | 0.74      |

All five give bit-identical results because the eigenvalue difference
between candidates is a CONSTANT shift (e.g.
`(ℓ−2)(ℓ+3) − ℓ(ℓ+1) = −6`), and the response form
`C(ℓ₁)·α₁ + C(ℓ₂)·α₂` only sees `α₁ + α₂ = 2L(L+1)` and
`α₁ − α₂ = 2(eig(ℓ₁) − eig(ℓ₂))`. The first is independent of `eig(·)`;
the second is invariant under constant shifts of `eig(ℓ)`. The README's
"spin-2 vs spin-0 eigenvalue" framing **cannot be the bug** because
those substitutions are all algebraically equivalent in this response
form.

### Drop the parity-even mask

Hypothesis: plancklens's gradient-mode 'pee' QE sums Wigner-3j over
both parities, while augr's mask restricts to L+ℓ₁+ℓ₂ even. Tested by
removing `even_mask`:

| L      | augr (no mask) / plancklens |
|--------|-----------------------------:|
| L=2    | 3.71                         |
| L=10   | 1.07                         |
| L=30   | **1.003**                    |
| L=100  | ~0.95                        |
| L=300  | 0.40 (worse than masked)     |
| L=1000 | ~0.5                         |

Better in mid-L (1.7 → 1.0 at L=30), much worse at high L (drops to
~0.4). At constant-Cℓ the no-mask form matched plancklens exactly at
all L≤30, but at L=300 even constant-Cℓ shows the no-mask form
undershooting by 0.69. So the no-mask form is empirically wrong too —
just at a different L band than the masked form. **Reverted.**

### What this means

The OkaHu Eq. 14 form (with /2 dropped, with even mask) and the
plancklens 'pee' QE differ by a non-constant, L-dependent factor that
no obvious local substitution captures. At constant-Cℓ the ratio is a
clean `2.0` everywhere (which is why "drop even_mask" reproduces
plancklens for that input), but at realistic Cℓ the ratio acquires a
spectrum-shape-dependent component that scales roughly like `(ℓ_E /
ℓ_B)^?` × `(C(ℓ₁) − C(ℓ₂))^?`.

This points to plancklens's leg-product representation NOT being a
simple permutation of OkaHu's `_2F + _2F` form. Plancklens builds 4
separate (s_left, sin) leg combinations
(`{(±2, ±2), (±2, ∓2)}`) under `qresp.get_qes('p_p', ...)` and projects
onto E×E via `qe_proj('e', 'e')`. Each leg carries spin-raise/lower
factors `±0.5·√((ℓ∓s)(ℓ±s+1))` that interact nontrivially when the
4-term sum is reduced. The Gaunt sum of these leg products yields a
form that, while still producing a single N₀ for the gradient-mode QE,
has **per-cell weights that do not factorise as `C(ℓ₁)α₁ + C(ℓ₂)α₂`
with any single bracket form**. The OkaHu Eq. 14 (single-`_2F`-per-leg)
is an approximation that is exact only in some limit (perhaps the
flat-sky or constant-Cℓ limit).

## Resolution — the missing piece is the spin-lowering branch

After tracing through `plancklens.qresp.get_qes('pee', ...)` and
inspecting the resulting qe-list (8 qes after `qe_simplify` +
`qe_proj`), the missing structural piece is now identified.

**Plancklens's 'pee' QE has TWO families of qes:**

| Family | qes | legb.spin_ou | legb eigenvalue        | Operator branch |
|--------|-----|--------------|------------------------|-----------------|
| A      | 0–3 | 3            | `√((ℓ−2)(ℓ+3))`        | spin-raise (ð)  |
| B      | 4–7 | −1           | `√((ℓ+2)(ℓ−1))`        | spin-lower (ð̄) |

Each family has 4 (lega.spin_in, legb.spin_in) ∈ {−2,2}²
combinations corresponding to the (E,E) projection of the joint
polarization estimator. The total response uses BOTH branches,
corresponding to the two terms in the lensing-source action on a
spin-2 field

    δ(_2X) = − ½ [ α_+1 · ð(_2X) + α_−1 · ð̄(_2X) ].

These are the two halves of plancklens's `get_resp_legs('p', lmax)`
return — `prR = -0.5·get_spin_lower(s)` and `mrR = -0.5·get_spin_raise(s)` —
which feed into get_covresp's tuple (s_qe, prR*coupl, mrR*coupl, cL).
get_qes uses ONLY mrR (the raising branch) to construct legs, but
the lowering branch is recovered implicitly in `nhl._get_nhl` via the
"sign-flipped" R_msmtuv term:

    GG_N0 += 0.5 · R_sutv.real           # raising branch (matched signs)
    GG_N0 += 0.5 · (-1)^(to+so) · R_msmtuv.real   # lowering branch (sign-flipped spins)

For TT (spin-0), get_spin_raise(0) = get_spin_lower(0) =
`√(ℓ(ℓ+1))`, so prR and mrR are identical in magnitude. The single
qe with `legb.cl = -√(ℓ(ℓ+1))·C^TT` captures both branches up to
trivial signs — which is why augr's TT formula (single Wigner-3j
sum, single bracket) reproduces plancklens 'ptt' to <1e-7.

For EE/EB/TE/TB (spin-2 legs), prR and mrR have DIFFERENT magnitudes
(`√((ℓ+2)(ℓ−1))` vs `√((ℓ−2)(ℓ+3))`), so the two branches contribute
INEQUIVALENTLY. A single Wigner-3j sum with any one bracket (the
test I ran with 5 different bracket forms) cannot capture the
two-branch structure — those substitutions are all algebraically
equivalent on a single bracket, but the actual response is the SUM
of two branches each with its own bracket-and-Wigner combination.

### The proper fix for `_compute_n0_ee_fullsky` (and EB/TE/TB)

Structural rewrite. Approximate sketch:

1. Build TWO Wigner-3j tables:
   - `w_raise = (l1, L, l2; m1=−2, m2=0, m3=2)` — raising-branch leg coupling
   - `w_lower = (l1, L, l2; m1=+2, m2=0, m3=−2)` — lowering-branch leg coupling
   `|w_raise|² = |w_lower|²` for parity-even pairs; for parity-odd
   they differ by `(−1)^{l1+L+l2}`.

2. Build TWO response forms:
   - `f_raise = [C(l1)·α_R(l1; l2, L) + C(l2)·α_R(l2; l1, L)] · pf · w_raise`
     where `α_R(ℓ_i; …) = L(L+1) + ε_R(ℓ_i) − ε_R(ℓ_j)` with raising
     eigenvalue `ε_R(ℓ) = √((ℓ−2)(ℓ+3))` (or its squared-eigenvalue
     analog `(ℓ−2)(ℓ+3)` if appearing inside a difference, depending
     on the exact Hu-Okamoto reduction).
   - `f_lower = [C(l1)·α_L(l1; l2, L) + C(l2)·α_L(l2; l1, L)] · pf · w_lower`
     with lowering eigenvalue `ε_L(ℓ) = √((ℓ+2)(ℓ−1))`.

3. Combine per nhl._get_nhl:
   - `1/N_0(L) = (1/(2L+1)) Σ_{l1,l2} ½·{|f_raise|² + (−1)^(to+so)·|f_lower'|²} / (2 C C)`
   where `f_lower'` is the cross-spin-flipped variant (R_msmtuv
   contribution; the to+so phase is +1 here for the EE qe spins).
   The exact combination needs to be derived by reducing the 64-term
   sum over (qe1, qe2) cross-products in `_get_nhl`.

4. Sanity-check on constant-Cℓ: the closed-form analog of TT's
   `8π/(L²(L+1)²S')` should emerge once the two-branch sum is
   correctly assembled.

Note: the simpler picture "augr matches OkaHu Eq. 14" is only
asymptotically correct (high ℓ where raise and lower eigenvalues
both `→ ℓ+½`). At low ℓ the two branches genuinely differ and the
full reduction is needed. OkaHu's f^EE in Table I implicitly
assumes this asymptotic limit or has a separate treatment of the
discrepancy that I haven't yet identified in the paper.

### Implementation cost

Probably 1–2 sessions. Steps:

1. Implement two-branch `_compute_n0_ee_fullsky` (with raise + lower).
2. Validate against plancklens 'pee' on constant-Cℓ first; expect
   exact agreement (within Wigner-3j numerical precision and
   l_max-truncation effects).
3. Validate against plancklens 'pee' on LiteBIRD-PTEP realistic
   spectra; expect <1% in bulk-L.
4. Mirror the fix in `_compute_n0_eb_fullsky` (where ð and ð̄ act on
   different fields E and B in a parity-odd combination).
5. Mirror in `_compute_n0_te_fullsky` and `_compute_n0_tb_fullsky`
   for the spin-2 leg specifically. The spin-0 (T) leg keeps its
   current single-branch form.
6. Remove xfails on EE/EB tests; tighten tolerance to <1e-2 in bulk-L.

## Sequencing — revised

1. ✓ EE Fix 1 applied (drop /2; keep mask). Median residual ~1.7× in
   bulk-L; principled but incomplete.
2. ✓ Diagnosis complete: missing the spin-lowering branch in the
   response sum.
3. **Open**: implement the two-branch response. Multi-session work.
4. After (3): mirror fixes for EB/TE/TB; tighten tolerances.

## Acceptance criteria — current status (post-Fix 1 only)

- TT: PASS (unchanged) — `TestN0AgainstPlancklens` <1e-3 in bulk.
- EE: still FAIL — ~1.7× in bulk, drift at high L. xfail kept;
  diagnosis in xfail message updated to point at the spin-lowering
  branch fix.
- EB: untested vs plancklens; still xfail-against-flat-sky.
- TE/TB: untested.

## Practical-impact note

augr's default `iterate_delensing` calls `compute_n0_*` with
`fullsky=False`, i.e. the flat-sky path is used for all production
forecasts. The flat-sky path is validated against the constant-Cℓ
closed form (`controlled_input_test.py`) and reproduces standard
flat-sky lensing-reconstruction noise formulas. The full-sky path
is opt-in (`fullsky=True`) and currently still labelled
"EXPERIMENTAL" in the README. The bug above does NOT affect σ(r)
forecasts that use the default flat-sky delensing — only forecasts
that explicitly set `fullsky=True` and use the polarization
estimators are affected. Recommend leaving `fullsky=False` as the
forecast default until the EE/EB full-sky fix lands.

## Open questions before writing code

1. **Does TE need fixing?** OkaHu Table I says ΘE has _2F on the
   "C^ΘE_{ℓ₁}" term and _0F on the "C^ΘE_{ℓ₂}" term — i.e. it's
   spin-MIXED. augr's `_compute_n0_te_fullsky` uses spin-0 (`w000`)
   for both legs, with the 4π prefactor and α/2 form. Under the
   spin-0 form both legs are TT-style and both /2's cancel against
   √(16π/4π) → OK numerically for the spin-0 part, but it's missing
   the spin-2 contribution to the response.

   plancklens 'pte' would be the reference. compare.py's npz already
   exposes only TT/EE/PP/MV — TE isn't computed separately. Would
   need a one-line addition to `run_plancklens.py` to surface 'pte'
   and a complementary xfail-or-pass test before declaring TE done.

2. **Same for TB.** OkaHu Table I has TB = `i C^TE_{ℓ₁} _2F_{ℓ₂Lℓ₁}`
   (odd), one term, spin-2. augr's `_compute_n0_tb_fullsky` uses the
   spin-2 coupling helper `_fullsky_spin2_coupling` (with `(2,-2,0)`
   m's) so that part looks right. Does the spectrum-weight construction
   (`l1_weight = C^TE² / C^TT`) handle the parity-odd-only restriction
   correctly? Needs a plancklens 'ptb' check.

3. **Is the EB result actually correct, or coincidentally close?**
   The "EB structurally matches OkaHu Eq. 14" argument above is
   correct for the |f|² piece, but the full N_0 also depends on (a)
   the parity mask, (b) the C_EE-squared / C_EE-tot leg-weight
   construction, (c) the (2L+1) normalization. Need plancklens 'p_eb'
   (or 'peb') in the npz so we have an apples-to-apples reference.
   Predicted result if augr-EB is right: ratio = 1.0 (well within
   bulk-L tolerance).

4. **TE spin-2 prefactor sanity check.** If TE needs the spin-2 term
   added, what's the right prefactor? If we add a `_2F_{ℓ₂Lℓ₁}` term
   to TE's `f`, that piece should use `√[…/16π]` with bracket (no /2),
   while the existing `_0F_{ℓ₁Lℓ₂}` term keeps `√[…/4π]` with
   bracket/2. This gives a heterogeneous response sum that's harder
   to keep numerically clean than the TT/EE cases.

5. **MV combination.** Once EE is fixed, augr's `compute_n0_mv`
   diagonal MV will get tighter at low L (since EE was overestimated).
   compare.py's MV ratio is currently a diagnostic only because
   plancklens 'p' is the joint GMV (not the diagonal MV); a tighter
   augr diagonal MV brings it closer to plancklens but still strictly
   ≥ from above.

## Acceptance criteria for the fix

- TT continues to pass: `TestN0AgainstPlancklens` stays <1e-3 in
  bulk-L, <1e-7 at L=2..200.
- EE: new test `TestN0EEAgainstPlancklens` passes <5% in bulk-L,
  <20% at L<10, replacing the current xfail.
- EB: extend `run_plancklens.py` to expose 'p_eb', then new test
  `TestN0EBAgainstPlancklens` passes <5% in bulk-L. If it doesn't,
  return to step 1 with a real EB diagnostic.
- TE/TB: add 'pte' and 'ptb' to the npz; if either is broken, file
  follow-up issue and add xfails. Don't conflate with the EE fix.
- The constant-Cℓ controlled-input test continues to pass for TT.
  For EE, document the closed form (or accept that constant-Cℓ EE
  doesn't have a clean closed form because of the cos(2φ)² factor —
  see `controlled_input_test.py` "Why TT only" note).

## Sequencing recommendation

1. Fix EE (`_compute_n0_ee_fullsky`): drop /2 on α.
2. Add 'p_eb' to `run_plancklens.py` and validate EB; lock it in or
   open follow-up.
3. Add 'pte' and 'ptb' similarly. Fix or xfail.
4. Drop README hypothesis text that blames spin-raising eigenvalues
   (that's not what's wrong) and replace with the corrected
   diagnosis above.

## References

- Hu & Okamoto 2002 (`papers/0111606_HuOkamoto_2002_polarization_QE.pdf`)
  — flat-sky polarization QE; the canonical
  `f = [C(ℓ₁) L·ℓ₁ + C(ℓ₂) L·ℓ₂] cos(2φ)` forms used as augr's
  flat-sky compute_n0_*.
- Okamoto & Hu 2003 (`papers/0301031_OkamotoHu_2003_fullsky_lensing_QE.pdf`)
  — full-sky generalization. **Eq. 14** is the building block;
  **Eq. 28** + **Table I** define f^α for each estimator;
  **Eq. 22** has the parity factors ε, β; **Eqs. 34–39** define the
  N₀ from g^α.
- Hanson, Challinor, Lewis 2010 review (`papers/0911.0612_*.pdf`)
  — collects the full-sky QE formulas in modern notation; useful as
  a sanity cross-check on OkaHu's conventions.
- plancklens (`/Users/jamie/cmb/plancklens/`):
  - `utils_spin.py:97-114` — `get_spin_raise/lower` definitions.
  - `qresp.py:104-133` — `get_resp_legs` defines the lensing response
    for spin-s maps via `−½α(ð̄ _sX) − ½α(ð _sX)`.
  - `qresp.py:50-101` — `get_qes` builds the leg-product representation
    of the QE; reduces to OkaHu Eq. 14 after the Gaunt sum.
  - `nhl.py:45-97` — `_get_nhl` performs the Gaunt sum via
    `wignerc` (Gauss-Legendre quadrature in cos(θ)).
