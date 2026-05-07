"""Tests for augr.wigner Wigner-3j primitives.

Locks in the Schulten-Gordon recursion against sympy. The
``_sg_b`` sign-on-m_3 bug surfaced by n0_validation work
(2026-05-06) is the motivating regression: previously,
``wigner3j_recurse`` and the vectorized variant produced wrong
values for any (m_1, m_2) with m_3 = -(m_1+m_2) != 0, because
``_sg_b`` had the wrong sign on the m_3 term.

Sympy's ``wigner_3j`` is the truth here. We test:
  - the closed-form ``wigner3j_000`` path (m1=m2=m3=0).
  - the recursion path ``wigner3j_recurse`` for small n with
    m_3 != 0 (the corner case that failed silently).
  - the vectorized table path ``wigner3j_vectorized`` for several
    (m_1, m_2) signatures including the production case
    (m_1=-2, m_2=0).
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
