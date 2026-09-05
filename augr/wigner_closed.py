"""
wigner_closed.py -- closed-form Wigner-3j tables via lookup (Kiddier & Gratton 2026).

Replaces the Schulten-Gordon recursion for the two symbol families augr's
full-sky QE delensing and MASTER coupling matrices need:

  * ``(j1 j2 j3; 0 0 0)`` -- Eq. 10-12 of Kiddier & Gratton (arXiv:2602.15605):
    the squared symbol is ``g(p1) g(p2) g(p3) / (g(p) (J+1))`` with
    ``g(p) = (2p)! / (2^{2p} (p!)^2)``, ``J = j1+j2+j3``, ``p = J/2`` and
    ``p_i`` the halved triangle combinations. ``g`` is a table of ``O(lmax)``
    entries built once, so every symbol is four gathers, three multiplies, a
    divide and a square root -- no factorials, no ``gammaln``.
  * ``(j1 j2 j3; 0 -2 2)`` -- written in terms of two ``(0 0 0)`` symbols.
    Even ``J`` is the paper's Eq. 15 / B14-B16. **Odd ``J`` is not in the
    paper's final form and not in its ``threej_cosmo`` C code, which only
    steps through even ``J``**; it follows from the paper's B1, B8 and B18
    and reduces to a single shifted ``(0 0 0)`` symbol. Both branches are
    gated against sympy over every triangle edge in ``tests/test_wigner.py``
    and against pywigxjpf to l ~ 3000 in ``tests/test_delensing.py``.

Everything here is elementwise, so a whole ``(n_l1, n_l2)`` table is one fused
kernel instead of an ``n_l2``-step sequential recursion. The functions take
an array namespace ``xp`` (``numpy`` or ``jax.numpy``) so the numpy and JAX
callers share one implementation; the truth for the formulas is sympy /
pywigxjpf, never the other backend.

Wigner symbols carry no dependence on any design parameter, so nothing
differentiable ever passes through these tables. The double-``where``
guards are still needed: several radicands are exactly zero on *valid*
triangle-edge cells (``beta`` at ``j3 = j1 + j2``), and a single ``where``
around ``sqrt`` there leaves an ``inf * 0`` in the reverse pass that the
gradient-finite tests in ``tests/test_wigner.py`` would catch.

Conventions match ``augr.wigner`` / ``augr.wigner_jax``: tables are indexed
``[l1, l2]`` for the symbol ``(l1, j2, l2; m1, m2, m3)`` with ``j2`` the fixed
middle slot (traced inside ``lax.map`` callers), signed, zero wherever the
triangle or a ``|m| <= j`` constraint fails.
"""

from __future__ import annotations

import functools

import numpy as np

__all__ = [
    "canonical_slots",
    "g_table",
    "j000",
    "j000_table",
    "p_max_for",
    "spin2_canonical",
    "spin2_table",
]


# -----------------------------------------------------------------------
# g(p) lookup table
# -----------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def g_table(p_max: int) -> np.ndarray:
    """``g(p) = C(2p, p) / 4^p`` for ``p = 0 .. p_max``, correctly rounded.

    Built by the exact integer recurrence ``C(2p,p) = C(2p-2,p-1) * 2(2p-1) / p``
    and one correctly-rounded big-int division per entry (0 ulp against mpmath
    at 30 digits for p <= 9000; ~30 ms at p_max = 9000). A ``gammaln``-based
    table is 4e-11 off at these arguments and was measured to cost 5e-10 on
    the spin-2 symbols -- do not substitute one. ``g`` decays as
    ``(pi p)^(-1/2)``, so no under/overflow anywhere in range.
    """
    p_max = int(p_max)
    if p_max < 0:
        raise ValueError(f"p_max must be >= 0, got {p_max}")
    out = np.empty(p_max + 1)
    out[0] = 1.0
    c = 1  # C(0, 0)
    for p in range(1, p_max + 1):
        c = c * 2 * (2 * p - 1) // p        # exact: the quotient is an integer
        out[p] = c / (1 << (2 * p))         # int / int -> correctly rounded float
    return out


def p_max_for(l1_max: int, l2_max: int) -> int:
    """Table size covering every gather a ``(l1, j2, l2)`` table can issue.

    On valid cells the triangle gives ``j2 <= l1 + l2``, so
    ``J <= 2 (l1_max + l2_max)``; the spin-2 branches read ``(0 0 0)`` at
    ``j3 + 2`` (even J) and ``j3 + 1`` (odd J), i.e. at ``p + 1`` -- hence the
    ``+ 1``. The bound needs ``l1_max``, not ``j2``, so a traced ``j2`` is fine.
    Off-triangle cells may index anywhere; the gathers clip and the mask zeroes
    them.
    """
    return int(l1_max) + int(l2_max) + 1


def canonical_slots(m1: int, m2: int, m3: int) -> tuple[tuple[int, int, int], bool]:
    """Map ``(m1, m2, m3)`` onto the canonical ``(0, -2, 2)`` ordering.

    Returns ``((slot0, slot_minus, slot_plus), odd_permutation)``: which of the
    three table slots ``(l1, j2, l2)`` carries ``m = 0``, ``-2`` and ``+2``, and
    whether that column permutation is odd (in which case the symbol picks up
    ``(-1)^J``; Edmonds 3.7.4-3.7.5). Raises ``ValueError`` for any other
    magnetic configuration -- callers fall back to the Schulten-Gordon path.
    """
    ms = (int(m1), int(m2), int(m3))
    if sorted(ms) != [-2, 0, 2]:
        raise ValueError(
            f"closed-form spin-2 tables need (m1, m2, m3) to be a permutation "
            f"of (0, -2, 2); got {ms}")
    perm = (ms.index(0), ms.index(-2), ms.index(2))
    inversions = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b])
    return perm, bool(inversions % 2)


# -----------------------------------------------------------------------
# Elementwise helpers (xp = numpy or jax.numpy)
# -----------------------------------------------------------------------

def _int(x, xp):
    return xp.round(x).astype(xp.int64)


def _parity_sign(x, xp):
    """``(-1)^x`` for integer-valued ``x``, as +-1.0."""
    return xp.where(_int(x, xp) % 2 == 0, 1.0, -1.0)


def _safe_sqrt(arg, ok, xp):
    """``sqrt(arg)`` where ``ok``, else 0 -- argument substituted before the sqrt."""
    return xp.where(ok, xp.sqrt(xp.where(ok, arg, 1.0)), 0.0)


def _safe_div(num, den, ok, xp):
    """``num / den`` where ``ok``, else 0 -- denominator substituted first."""
    return xp.where(ok, num / xp.where(ok, den, 1.0), 0.0)


def _gather(g, idx, xp):
    """``g[idx]`` with the (possibly traced, possibly out-of-range) index clipped."""
    n = g.shape[0] - 1
    return xp.take(g, xp.clip(_int(idx, xp), 0, n), mode="clip")


def _triangle(j1, j2, j3, xp):
    return ((j3 >= xp.abs(j1 - j2)) & (j3 <= j1 + j2)
            & (j1 >= 0) & (j2 >= 0) & (j3 >= 0))


# -----------------------------------------------------------------------
# (j1 j2 j3; 0 0 0)
# -----------------------------------------------------------------------

def j000(j1, j2, j3, g, *, xp):
    """Signed ``(j1 j2 j3; 0 0 0)``, elementwise with broadcasting.

    ``(-1)^p sqrt(g(p1) g(p2) g(p3) / (g(p) (J+1)))`` (Kiddier & Gratton Eq. 10
    with Edmonds' ``(-1)^{J/2}`` sign); zero off the triangle or for odd ``J``.
    ``g`` is :func:`g_table` sized by :func:`p_max_for`.
    """
    J = j1 + j2 + j3
    ok = _triangle(j1, j2, j3, xp) & (_int(J, xp) % 2 == 0)
    p = xp.floor(J / 2.0)
    p1 = xp.floor((-j1 + j2 + j3) / 2.0)
    p2 = xp.floor((j1 - j2 + j3) / 2.0)
    p3 = xp.floor((j1 + j2 - j3) / 2.0)
    sq = (_gather(g, p1, xp) * _gather(g, p2, xp) * _gather(g, p3, xp)
          / (_gather(g, p, xp) * (xp.abs(J) + 1.0)))
    return _parity_sign(p, xp) * _safe_sqrt(sq, ok, xp)


# -----------------------------------------------------------------------
# (j1 j2 j3; 0 -2 2)
# -----------------------------------------------------------------------

def spin2_canonical(j1, j2, j3, g, *, xp):
    """Signed ``S(j1, j2, j3) = (j1 j2 j3; 0 -2 2)``, elementwise with broadcasting.

    Even ``J`` (paper Eq. 15, B14-B16)::

        S = [alpha X + beta Y] / eta,   X = (j1 j2 j3; 000),  Y = (j1 j2 j3+2; 000)

    Odd ``J`` (from the paper's B1, B8, B18; the ``(000)`` symbol at ``j3``
    vanishes and the two ``(0 -1 1)`` terms collapse onto one shifted symbol)::

        S = -(Lambda Z / eta) sqrt((j3+2) / (j2 (j3+1)))
            * [ 1/sqrt(j2+1) + sqrt(j2+1) (1 - (J+3)(J_mpp+2) / (2 (j2+1)(j3+2))) ]
        Z = (j1 j2 j3+1; 000),  Lambda = sqrt((J+2)(J_mpp+1)(J_pmp+1) J_ppm)

    with ``lambda = sqrt(j2(j2+1)(j3+1)(j3+2))``, ``eta = sqrt((j2-1)(j2+2)(j3-1)j3)``
    and ``J_mpp = -j1+j2+j3`` etc. Zero when ``j2 < 2`` or ``j3 < 2`` (the
    ``|m| <= j`` constraints) or off the triangle. Symmetric under ``j2 <-> j3``.
    """
    J = j1 + j2 + j3
    Jmpp = -j1 + j2 + j3
    Jpmp = j1 - j2 + j3
    Jppm = j1 + j2 - j3
    ok = _triangle(j1, j2, j3, xp) & (j2 >= 2) & (j3 >= 2)
    odd = _int(J, xp) % 2 == 1

    lam = _safe_sqrt(j2 * (j2 + 1.0) * (j3 + 1.0) * (j3 + 2.0), ok, xp)
    inv_eta = _safe_div(1.0, _safe_sqrt((j2 - 1.0) * (j2 + 2.0) * (j3 - 1.0) * j3, ok, xp),
                        ok, xp)

    # --- even J ---
    X = j000(j1, j2, j3, g, xp=xp)
    Y = j000(j1, j2, j3 + 2.0, g, xp=xp)
    alpha = lam + _safe_div(2.0 * lam, j2, ok, xp) * (
        1.0 - _safe_div((J + 2.0) * (Jmpp + 1.0), 2.0 * (j2 + 1.0) * (j3 + 1.0), ok, xp))
    beta_arg = ((J + 2.0) * (Jmpp + 1.0) * (Jpmp + 1.0) * Jppm
                * (J + 3.0) * (Jmpp + 2.0) * (Jpmp + 2.0) * (Jppm - 1.0))
    beta = _safe_div(_safe_sqrt(beta_arg, ok & (beta_arg > 0), xp), 2.0 * lam, ok, xp)
    s_even = inv_eta * (alpha * X + beta * Y)

    # --- odd J ---
    Z = j000(j1, j2, j3 + 1.0, g, xp=xp)
    lam_arg = (J + 2.0) * (Jmpp + 1.0) * (Jpmp + 1.0) * Jppm
    Lam = _safe_sqrt(lam_arg, ok & (lam_arg > 0), xp)
    root_j2p1 = _safe_sqrt(j2 + 1.0, ok, xp)
    bracket = (_safe_div(1.0, root_j2p1, ok, xp)
               + root_j2p1 * (1.0 - _safe_div((J + 3.0) * (Jmpp + 2.0),
                                                2.0 * (j2 + 1.0) * (j3 + 2.0), ok, xp)))
    root = _safe_sqrt(_safe_div(j3 + 2.0, j2 * (j3 + 1.0), ok, xp), ok, xp)
    s_odd = -Lam * Z * inv_eta * root * bracket

    return xp.where(ok, xp.where(odd, s_odd, s_even), 0.0)


# -----------------------------------------------------------------------
# (n_l1, n_l2) tables in augr's slot convention
# -----------------------------------------------------------------------

def j000_table(L, l1, l2_min: int, l2_max: int, g, *, xp):
    """``w[i, j] = (l1[i], l2_grid[j], L; 0 0 0)``; ``L`` may be traced.

    The ``(0 0 0)`` symbol is fully permutation-symmetric, so the slot order is
    immaterial; this matches ``wigner.wigner3j_000_vectorized``.
    """
    l1c = xp.asarray(l1, dtype=float)[:, None]
    l2c = xp.arange(l2_min, l2_max + 1, dtype=float)[None, :]
    return j000(l1c, l2c, L, g, xp=xp)


def spin2_table(j2, l1, m1: int, m2: int, m3: int, l2_min: int, l2_max: int, g, *, xp):
    """``w[i, j] = (l1[i], j2, l2_grid[j]; m1, m2, m3)``; ``j2`` may be traced.

    ``(m1, m2, m3)`` must be a permutation of ``(0, -2, 2)`` (see
    :func:`canonical_slots`); the slots are reordered onto the canonical
    ``(0, -2, 2)`` symbol and an odd column permutation contributes ``(-1)^J``.
    The ``|m| <= j`` constraints are enforced by the canonical form
    (``j2 >= 2`` and ``j3 >= 2`` on the ``+-2`` slots).
    """
    (slot0, slot_m, slot_p), odd_perm = canonical_slots(m1, m2, m3)
    l1c = xp.asarray(l1, dtype=float)[:, None]
    l2c = xp.arange(l2_min, l2_max + 1, dtype=float)[None, :]
    slots = (l1c, j2, l2c)
    s = spin2_canonical(slots[slot0], slots[slot_m], slots[slot_p], g, xp=xp)
    if odd_perm:
        s = s * _parity_sign(l1c + j2 + l2c, xp)
    return s
