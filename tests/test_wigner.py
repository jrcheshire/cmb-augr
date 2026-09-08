"""Tests for augr.wigner Wigner-3j primitives.

Sympy's ``wigner_3j`` is the truth here. Covers:
  - the closed-form ``wigner3j_000`` path (m1=m2=m3=0);
  - the recursion ``wigner3j_recurse`` for small n with m_3 != 0, where a sign
    error on the ``_sg_b`` m_3 term once failed silently;
  - the vectorized table ``wigner3j_vectorized`` for several (m_1, m_2)
    signatures including the production (m_1=-2, m_2=0).
"""

from __future__ import annotations

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
from sympy.physics.wigner import wigner_3j as sym_3j  # noqa: E402

from augr.wigner import (  # noqa: E402
    wigner3j_000,
    wigner3j_000_vectorized,
    wigner3j_recurse,
    wigner3j_vectorized,
)


def _sym(j1, j2, j3, m1, m2, m3):
    return float(sym_3j(j1, j2, j3, m1, m2, m3))


# ---------------------------------------------------------------------
# Closed-form (m1=m2=m3=0) path: should be machine-precision.
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "j1,j2,L",
    [
        (2, 2, 2),
        (3, 5, 4),
        (10, 8, 6),
        (50, 40, 30),
        (100, 100, 50),
    ],
)
def test_wigner3j_000_closed_form(j1, j2, L):
    expected = _sym(j1, j2, L, 0, 0, 0)
    actual = wigner3j_000(j1, j2, L)
    assert abs(actual - expected) < 1e-12


def test_wigner3j_000_vectorized_matches_sympy():
    L = 5
    j1_arr = np.arange(0, 10, dtype=int)
    l2_grid, w = wigner3j_000_vectorized(L, j1_arr, l2_min=0, l2_max=14)
    for i, j1 in enumerate(j1_arr):
        for k, l2 in enumerate(l2_grid):
            expected = _sym(int(j1), L, int(l2), 0, 0, 0)
            assert abs(w[i, k] - expected) < 1e-10, (
                f"j1={j1}, L={L}, l2={l2}: got {w[i, k]:.6f}, "
                f"expected {expected:.6f}"
            )


# ---------------------------------------------------------------------
# Recursion path: m_3 != 0 cases. These previously failed silently
# because of the sign bug in _sg_b.
# ---------------------------------------------------------------------

# Pairs (j1, j2, m1, m2) covering small n (where bidirectional matching
# was previously unstable) and a range of m_3 values.
RECURSE_CASES = [
    # m_3 = 0 (m1 = -m2): used to work, must keep working.
    (3, 2, 2, -2),
    (5, 4, 2, -2),
    (10, 8, 2, -2),
    # m_3 = 1: failed previously.
    (2, 1, 0, -1),
    (3, 2, 0, -1),
    (4, 3, 0, -1),
    (10, 8, 0, -1),
    # m_3 = 2 (production case in delensing.py: m1=-2, m2=0).
    (2, 1, -2, 0),
    (3, 2, -2, 0),
    (5, 4, -2, 0),
    (10, 8, -2, 0),
    (50, 40, -2, 0),
    (100, 80, -2, 0),
    # m_3 = -2 (sign-flipped variant).
    (3, 2, 2, 0),
    (10, 8, 2, 0),
    # Mixed.
    (5, 4, 2, -1),
    (10, 8, -2, 1),
]


@pytest.mark.parametrize("j1,j2,m1,m2", RECURSE_CASES)
def test_wigner3j_recurse_matches_sympy(j1, j2, m1, m2):
    j_grid, w_augr = wigner3j_recurse(j1=j1, j2=j2, m1=m1, m2=m2)
    m3 = -(m1 + m2)
    for j, wa in zip(j_grid, w_augr, strict=True):
        expected = _sym(j1, j2, int(j), m1, m2, m3)
        assert abs(wa - expected) < 1e-10, (
            f"(j1={j1}, j2={j2}, j3={j}, m1={m1}, m2={m2}, m3={m3}): "
            f"augr={wa:.6e}, sympy={expected:.6e}"
        )


@pytest.mark.parametrize(
    "L,j1_max,m1,m2",
    [
        (1, 3, 0, -1),
        (1, 3, -2, 0),
        (5, 8, -2, 0),
        (5, 8, 2, -2),
        (10, 12, 2, 0),
        (20, 25, -2, 0),
        (50, 60, -2, 0),
    ],
)
def test_wigner3j_vectorized_matches_sympy(L, j1_max, m1, m2):
    j1_arr = np.arange(0, j1_max + 1, dtype=float)
    l2_grid, w = wigner3j_vectorized(L, j1_arr, m1=m1, m2=m2)
    m3 = -(m1 + m2)
    for i, j1 in enumerate(j1_arr.astype(int)):
        for k, l2 in enumerate(l2_grid.astype(int)):
            expected = _sym(int(j1), int(L), int(l2), m1, m2, m3)
            actual = w[i, k]
            # Triangle violations and |m_2| > L cases give w=0 in
            # sympy; recursion may produce small non-zero spurious
            # values which we treat as zero up to abs tol.
            if abs(expected) < 1e-12:
                assert abs(actual) < 1e-10, (
                    f"L={L}, j1={j1}, l2={l2}: expected 0, "
                    f"got {actual:.3e}"
                )
            else:
                assert abs(actual - expected) < 1e-10 * (1 + abs(expected)), (
                    f"L={L}, j1={j1}, l2={l2}, m1={m1}, m2={m2}, m3={m3}: "
                    f"augr={actual:.6e}, sympy={expected:.6e}"
                )


# -----------------------------------------------------------------------
# JAX port (augr.wigner_jax) -- validated against the sympy-locked numpy
# version (issue #45 Stage 3). jax == numpy suffices since numpy == sympy
# is locked above.
# -----------------------------------------------------------------------

import functools  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from augr.wigner_jax import (  # noqa: E402
    _spin2_body_sg,
    spin0_body,
    spin2_body,
    wigner3j_000_vectorized_jax,
    wigner3j_vectorized_jax,
)


@pytest.mark.parametrize(
    "L,j1_max,m1,m2",
    [
        (1, 3, 0, -1),
        (1, 3, -2, 0),
        (5, 8, -2, 0),
        (5, 8, 2, -2),
        (10, 12, 2, 0),
        (20, 25, -2, 0),
        (50, 60, -2, 0),
        (100, 120, -2, 0),
    ],
)
def test_wigner3j_vectorized_jax_matches_numpy(L, j1_max, m1, m2):
    j1 = np.arange(0, j1_max + 1, dtype=float)
    l2max = int(j1.max()) + L
    _, w_np = wigner3j_vectorized(L, j1, m1=m1, m2=m2, l2_max_global=l2max)
    _, w_j = wigner3j_vectorized_jax(L, j1, m1=m1, m2=m2, l2_max_global=l2max)
    w_j = np.asarray(w_j)
    # fp64: max abs diff is machine-eps; meaningful entries match to <1e-8.
    np.testing.assert_allclose(w_j, w_np, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("L,j1_max", [(2, 14), (5, 20), (37, 60), (100, 120)])
def test_wigner3j_000_vectorized_jax_matches_numpy(L, j1_max):
    j1 = np.arange(0, j1_max + 1, dtype=float)
    l2max = int(j1.max()) + L
    _, w_np = wigner3j_000_vectorized(L, j1, l2_max=l2max)
    _, w_j = wigner3j_000_vectorized_jax(L, j1, l2_max=l2max)
    np.testing.assert_allclose(np.asarray(w_j), w_np, rtol=1e-8, atol=1e-11)


def test_wigner_jax_jit_compiles():
    """Both paths compile under jax.jit (L and l2 bounds static).

    ``l1`` is traced here, so the closed-form table size must be given via
    ``l1_max`` (it is ``max(l1) + l2_max + 1``; see ``wigner_closed.p_max_for``).
    """
    j1 = np.arange(2, 60, dtype=float)
    f2 = jax.jit(functools.partial(
        wigner3j_vectorized_jax, 37, m1=-2, m2=0, l2_max_global=100, l1_max=59))
    _, w2 = wigner3j_vectorized(37, j1, m1=-2, m2=0, l2_max_global=100)
    np.testing.assert_allclose(np.asarray(f2(j1)[1]), w2, rtol=1e-8, atol=1e-11)
    f0 = jax.jit(functools.partial(wigner3j_000_vectorized_jax, 37, l2_max=100,
                                   l1_max=59))
    _, w0 = wigner3j_000_vectorized(37, j1, l2_max=100)
    np.testing.assert_allclose(np.asarray(f0(j1)[1]), w0, rtol=1e-8, atol=1e-11)


# -----------------------------------------------------------------------
# spin2_body: the traced core's own contract. The tests above all go
# through wigner3j_vectorized_jax, which guards the edge cases before
# calling the core -- but lax.map callers (delensing_fullsky_jax, and the
# MASTER coupling matrix) call spin2_body DIRECTLY, so the core has to
# hold the same contract on its own.
# -----------------------------------------------------------------------


@pytest.mark.parametrize("L", [0, 1])
def test_spin2_body_zeroes_L_below_abs_m2(L):
    """|m2| > L must give exactly zero, as the public wrapper already returns.

    The recursion constrains m1 row-wise (``m1_ok``) and m3 through the l2 lower
    bound, but nothing tests m2 against L. Without an explicit guard the seed and
    the ``sum_j (2j+1) w^2 = 1`` normalization hand back a unit-norm table where
    the symbol is identically zero -- measured |w|max = 0.447 at L=0 and 0.526 at
    L=1 for (m1, m2) = (2, -2). ``wigner3j_vectorized_jax`` short-circuits this
    for concrete L; the core must too.

    This has never bitten because every pre-existing caller
    (``delensing_fullsky_jax``) uses m2=0, where |m2| <= L is free. The MASTER
    coupling matrix streams l2 from 0 at (m1, m2) = (2, -2) and hits it directly.
    """
    l1 = jnp.arange(0, 6, dtype=float)
    w = np.asarray(spin2_body(float(L), l1, 2, -2, 0, 0, 8))
    assert np.all(w == 0.0), f"L={L}: |w|max = {np.abs(w).max():.4f}, expected 0"


@pytest.mark.parametrize("L", [0, 1, 2, 3, 7])
def test_spin2_body_matches_numpy_at_spin2_m(L):
    """Direct-core parity with the sympy-locked numpy (SG) version at (m1, m2) = (2, -2).

    Covers both sides of the guard: L < 2 (must be zero) and L >= 2 (must match).
    The l2 grid reaches ``max(l1) + L`` so the SG reference's sum-rule
    normalization sees every row's full triangle -- cut at 8 it renormalized
    the ``l1 = 5, L = 7`` row by 40% while the closed form was unaffected.
    """
    l1 = np.arange(0, 6, dtype=float)
    l2_max = 5 + L
    _, w_np = wigner3j_vectorized(L, l1, m1=2, m2=-2, l2_min_global=0,
                                  l2_max_global=l2_max)
    w_j = np.asarray(spin2_body(float(L), jnp.asarray(l1), 2, -2, 0, 0, l2_max))
    np.testing.assert_allclose(w_j, w_np, rtol=1e-8, atol=1e-11)


def test_spin2_body_normalization_requires_full_l2_grid():
    """The SG recursion needs the l2 grid to reach l1 + L for every row.

    ``_spin2_body_sg`` normalizes by ``sum_j (2j+1) w^2 = 1`` over whatever grid
    it is handed, so a row whose support runs past ``l2_max`` is renormalized
    against a partial sum. Rows that fit are untouched (bit-identical below);
    clipped rows come back wrong by 73-162% of their own scale. This is the
    reason ``pseudo_cl_jax`` allocates its l3 axis out to ``2 * lmax``; the
    closed form that now backs ``spin2_body`` has no such dependence (see
    :func:`test_spin2_body_is_truncation_independent`), but the grid must still
    cover the physical support of whatever sum consumes the table.
    """
    L, n = 12, 19
    l1 = jnp.arange(0, 13, dtype=float)
    full = np.asarray(_spin2_body_sg(float(L), l1, 2, -2, 0, 0, 24))[:, :n]
    trunc = np.asarray(_spin2_body_sg(float(L), l1, 2, -2, 0, 0, 18))[:, :n]

    # rows with support [|l1-L|, l1+L] inside the truncated grid: bit-identical
    fits = slice(0, 7)  # l1 <= 6 -> l1 + L <= 18
    np.testing.assert_array_equal(trunc[fits], full[fits])

    # clipped rows: wrong by a large fraction of their own scale
    for row in (8, 10, 12):
        scale = np.abs(full[row]).max()
        rel = np.abs(trunc[row] - full[row]).max() / scale
        assert rel > 0.5, f"l1={row}: truncation changed the row by only {rel:.3f}"


@pytest.mark.parametrize("m1,m2,m3", [(2, -2, 0), (-2, 0, 2)])
def test_spin2_body_is_truncation_independent(m1, m2, m3):
    """Closed-form entries do not depend on the l2 grid extent (bit-for-bit).

    Each cell is evaluated on its own, with no normalization over the grid, so
    truncating l2 leaves every remaining column untouched -- including rows
    whose triangle runs past ``l2_max``, exactly where the SG recursion above
    goes wrong.
    """
    L, n = 12, 19
    l1 = jnp.arange(0, 13, dtype=float)
    full = np.asarray(spin2_body(float(L), l1, m1, m2, m3, 0, 24))[:, :n]
    trunc = np.asarray(spin2_body(float(L), l1, m1, m2, m3, 0, 18))[:, :n]
    np.testing.assert_array_equal(trunc, full)


@pytest.mark.parametrize("m1,m2,m3", [(2, -2, 0), (-2, 0, 2)])
def test_spin2_body_gradient_is_finite(m1, m2, m3):
    """Reverse mode through the recursion must not produce NaN.

    ``_sg_a_jax`` uses the double-where idiom, substituting the argument before the
    sqrt: ``sqrt(maximum(arg, 0))`` is correct in value but NaN-poisons the reverse
    pass, since d/dx sqrt(x) is infinite at x=0 and the clamped branch contributes a
    zero, giving inf * 0.

    Nothing in augr differentiates with respect to a multipole index, so this guards
    a property rather than a caller -- worth keeping in a module whose purpose is
    differentiability.
    """
    l1 = jnp.arange(0, 6, dtype=float)
    g = jax.grad(jax.jit(lambda j2: spin2_body(j2, l1, m1, m2, m3, 0, 8).sum()))
    grads = np.array([float(g(float(j2))) for j2 in (0, 1, 2, 3, 5)])
    assert np.all(np.isfinite(grads)), f"non-finite gradients: {grads}"


def test_spin0_body_gradient_is_finite():
    """Companion to the spin-2 case; the table gathers carry no gradient, the sqrt is guarded."""
    l1 = jnp.arange(0, 6, dtype=float)
    g = jax.grad(jax.jit(lambda j2: spin0_body(j2, l1, 0, 8).sum()))
    grads = np.array([float(g(float(j2))) for j2 in (0, 1, 3, 5)])
    assert np.all(np.isfinite(grads)), f"non-finite gradients: {grads}"


# -----------------------------------------------------------------------
# Closed-form tables (Kiddier & Gratton 2026; issue #48): augr.wigner_closed
# -----------------------------------------------------------------------

from augr.wigner_closed import (  # noqa: E402
    canonical_slots,
    g_table,
    j000,
    j000_table,
    p_max_for,
    spin2_canonical,
    spin2_table,
)
from augr.wigner_jax import _spin0_body_gammaln  # noqa: E402

_JMAX = 12
_G = g_table(p_max_for(_JMAX + 2, _JMAX + 2))
_XP = [pytest.param(np, id="numpy"), pytest.param(jnp, id="jax")]


def _triples(jmax=_JMAX):
    """Every (j1, j2, j3) in [0, jmax]^3 -- on and off the triangle, both parities."""
    j = np.arange(jmax + 1, dtype=float)
    return [x.ravel() for x in np.meshgrid(j, j, j, indexing="ij")]


def test_g_table_matches_exact_rationals():
    """g(p) = C(2p, p) / 4^p, correctly rounded, for a p range the tests reach."""
    from fractions import Fraction
    from math import comb
    g = g_table(200)
    for p in range(201):
        assert g[p] == float(Fraction(comb(2 * p, p), 4 ** p))


@pytest.mark.parametrize("xp", _XP)
def test_j000_matches_sympy_exhaustive(xp):
    """Signed (j1 j2 j3; 0 0 0) over [0, 12]^3: 1e-14 everywhere, exact selection-rule zeros.

    Measured worst 5.6e-17 on this grid; the gate is 1e-14.
    """
    j1, j2, j3 = _triples()
    got = np.asarray(j000(xp.asarray(j1), xp.asarray(j2), xp.asarray(j3),
                          xp.asarray(_G), xp=xp))
    truth = np.array([_sym(int(a), int(b), int(c), 0, 0, 0)
                      for a, b, c in zip(j1, j2, j3, strict=True)])
    assert np.max(np.abs(got - truth)) < 1e-14
    rule = ((j1 + j2 + j3) % 2 == 1) | (j3 < np.abs(j1 - j2)) | (j3 > j1 + j2)
    np.testing.assert_array_equal(got[rule], 0.0)


@pytest.mark.parametrize("parity", ["even", "odd"])
@pytest.mark.parametrize("xp", _XP)
def test_spin2_canonical_matches_sympy_exhaustive(parity, xp):
    """Signed (j1 j2 j3; 0 -2 2) over [0, 12]^3, one gate per J parity.

    Even J is the paper's Eq. 15; odd J is derived from its B1, B8 and B18 and
    is not in the paper's final form or its C code, so it carries its own gate.
    Includes every triangle edge (j3 = |j1-j2|, j1+j2) and the |m| <= j zeros
    (j2 < 2 or j3 < 2). Measured worst 2.8e-16 on this grid; the gate is 1e-14.
    """
    j1, j2, j3 = _triples()
    sel = ((j1 + j2 + j3) % 2 == 0) if parity == "even" else ((j1 + j2 + j3) % 2 == 1)
    j1, j2, j3 = j1[sel], j2[sel], j3[sel]
    got = np.asarray(spin2_canonical(xp.asarray(j1), xp.asarray(j2), xp.asarray(j3),
                                     xp.asarray(_G), xp=xp))
    truth = np.array([_sym(int(a), int(b), int(c), 0, -2, 2)
                      for a, b, c in zip(j1, j2, j3, strict=True)])
    assert np.sum(truth != 0.0) > 100
    assert np.max(np.abs(got - truth)) < 1e-14
    # Selection-rule zeros are exact zeros (the mask, not a cancellation).
    # Accidental zeros on valid cells -- e.g. (2,3,3), (4,4,6), (9,4,8) -- come
    # out at ~1e-17 from the closed form and are covered by the bound above.
    rule = (j2 < 2) | (j3 < 2) | (j3 < np.abs(j1 - j2)) | (j3 > j1 + j2)
    np.testing.assert_array_equal(got[rule], 0.0)


@pytest.mark.parametrize("xp", _XP)
def test_spin2_canonical_symmetric_in_j2_j3(xp):
    """(j1 j2 j3; 0 -2 2) = (j1 j3 j2; 0 2 -2) = (j1 j3 j2; 0 -2 2) by (-1)^J twice.

    The closed form is not manifestly symmetric under j2 <-> j3, so this is a
    free consistency check on the alpha / beta / bracket algebra.
    """
    j1, j2, j3 = _triples()
    g = xp.asarray(_G)
    a = np.asarray(spin2_canonical(xp.asarray(j1), xp.asarray(j2), xp.asarray(j3), g, xp=xp))
    b = np.asarray(spin2_canonical(xp.asarray(j1), xp.asarray(j3), xp.asarray(j2), g, xp=xp))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)


_SIX = [(-2, 0, 2), (2, 0, -2), (2, -2, 0), (-2, 2, 0), (0, 2, -2), (0, -2, 2)]


@pytest.mark.parametrize("m1,m2,m3", _SIX)
def test_spin2_table_configs_match_sympy(m1, m2, m3):
    """All six slot assignments of (0, -2, 2) through spin2_table, both backends.

    Covers the odd-permutation (-1)^J sign and the |m| <= j masks per slot.
    """
    l1 = np.arange(0, 9, dtype=float)
    l2_min, l2_max = 0, 16
    for j2 in range(0, 7):
        truth = np.array([[_sym(int(a), j2, c, m1, m2, m3)
                           for c in range(l2_min, l2_max + 1)] for a in l1.astype(int)])
        for xp in (np, jnp):
            got = np.asarray(spin2_table(float(j2), xp.asarray(l1), m1, m2, m3,
                                         l2_min, l2_max, xp.asarray(_G), xp=xp))
            np.testing.assert_allclose(got, truth, rtol=0, atol=1e-14)


@pytest.mark.parametrize("ms", [(0, -1, 1), (0, 0, 0), (2, -2, 1), (1, -1, 0), (-2, -2, 4)])
def test_canonical_slots_rejects_other_configs(ms):
    with pytest.raises(ValueError):
        canonical_slots(*ms)


def test_canonical_slots_permutation_parity():
    """Config A (-2,0,2) and B (2,-2,0) are odd permutations; cyclic ones are even."""
    assert canonical_slots(-2, 0, 2) == ((1, 0, 2), True)
    assert canonical_slots(2, -2, 0) == ((2, 1, 0), True)
    assert canonical_slots(0, -2, 2) == ((0, 1, 2), False)
    assert canonical_slots(2, 0, -2) == ((1, 2, 0), False)


def test_spin2_body_falls_back_to_sg_for_other_configs():
    """(0, -1, 1) is outside the closed form: spin2_body must be the SG table, bit-for-bit."""
    l1 = jnp.arange(0, 10, dtype=float)
    a = np.asarray(spin2_body(5.0, l1, 0, -1, 1, 1, 20))
    b = np.asarray(_spin2_body_sg(5.0, l1, 0, -1, 1, 1, 20))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("m1,m2,m3", [(-2, 0, 2), (2, -2, 0)])
def test_spin2_body_traced_in_lax_map_matches_concrete(m1, m2, m3):
    """The lax.map (traced j2) path equals per-L concrete calls, including j2 < 2 rows.

    Measured max abs difference 1.1e-16 (XLA fusion in the mapped body); the
    gate is 1e-15 absolute.
    """
    l1 = jnp.arange(0, 13, dtype=float)

    def body(j2):
        return spin2_body(j2, l1, m1, m2, m3, 0, 40)

    mapped = np.asarray(jax.lax.map(body, jnp.arange(0.0, 21.0)))
    concrete = np.stack([np.asarray(body(float(j2))) for j2 in range(21)])
    np.testing.assert_allclose(mapped, concrete, rtol=0, atol=1e-15)


def test_spin0_body_matches_gammaln_reference():
    """Closed-form (0 0 0) table vs the pre-#48 gammaln body, row-max-normalized.

    Measured 5.7e-12 .. 8.2e-12 at l1 = 2..1500, l2 = 2..2999 (the gammaln side
    is the inexact one -- against pywigxjpf the closed form is ~1e-15 and the
    gammaln form ~1e-11); this smaller shape measured the same order. Gate 1e-10.
    """
    l1 = jnp.arange(2, 301, dtype=float)
    for L in (2, 37, 150, 299):
        a = np.asarray(spin0_body(float(L), l1, 2, 600))
        b = np.asarray(_spin0_body_gammaln(float(L), l1, 2, 600))
        scale = np.abs(b).max(axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        assert np.max(np.abs(a - b) / scale) < 1e-10, L


def test_wigner3j_000_vectorized_numpy_uses_closed_form():
    """The numpy (0 0 0) table is the closed form (same table, same grid bookkeeping)."""
    l1 = np.arange(0, 40)
    l2_grid, w = wigner3j_000_vectorized(17, l1, l2_min=3, l2_max=60)
    g = g_table(p_max_for(39, 60))
    ref = j000_table(17.0, l1.astype(float), 3, 60, g, xp=np)
    np.testing.assert_array_equal(l2_grid, np.arange(3, 61))
    np.testing.assert_array_equal(w, ref)


@pytest.mark.slow
@pytest.mark.parametrize("m1,m2,m3,l2_min", [(-2, 0, 2, 2), (2, -2, 0, 0)])
def test_spin2_body_matches_sg_at_production_shape(m1, m2, m3, l2_min):
    """Closed form vs the SG scan on the delensing / MASTER grids, row-max-normalized.

    l1 = 2..1500, l2 up to 2999, L in {2, 50, 500, 1499}. Measured 6.2e-16 ..
    8.4e-15 (config A) and 1.3e-15 .. 8.2e-15 (config B on its l2_min=0
    production grid); gate 1e-13. Config B needs l2_min=0: with the m=0 slot
    on l2 the triangle reaches l2 < 2, and a grid floored at 2 truncates the
    SG normalization on rows l1 in {L-1, L, L+1} (58% wrong at L=2, l1=2 --
    the closed form was arbitrated correct against pywigxjpf there).
    """
    l1 = jnp.arange(2, 1501, dtype=float)
    for L in (2, 50, 500, 1499):
        a = np.asarray(spin2_body(float(L), l1, m1, m2, m3, l2_min, 2999))
        b = np.asarray(_spin2_body_sg(float(L), l1, m1, m2, m3, l2_min, 2999))
        scale = np.abs(b).max(axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        assert np.max(np.abs(a - b) / scale) < 1e-13, L

