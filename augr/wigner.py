"""
wigner.py — Wigner 3j symbol computation for full-sky CMB lensing coupling.

Provides two computation methods:
  1. Closed-form via log-gamma for (l1 l2 L; 0 0 0) — exact, stable at
     all ell. Used for scalar (TT, TE) QE estimators.
  2. Schulten-Gordon three-term recursion for general (j1 j2 j; m1 m2 m3).
     Used for spin-2 (EB, TB, EE) QE estimators and the lensing kernel.

The vectorized recursion processes all l1 values simultaneously for a
fixed L, enabling efficient computation of the full coupling matrix
needed for the N_0 and kernel sums. The recursion is bidirectional
(forward from j_min, backward from j_max) with median-ratio matching
at the midpoint for numerical stability across the full j range.

The delensing module calls this with two m-value configurations:
  - m1=-2, m2=0 (m3=2): computes (l_E, L, l_B; -2, 0, 2), which
    equals Smith et al.'s (l_B, l_E, L; 2, -2, 0) by cyclic symmetry.
    Used for the parity-odd EB/TB coupling and the lensing kernel.
  - m1=m2=0 (m3=0): computes (l1, L, l2; 0, 0, 0) via the closed-form
    path. Used for TT and TE.

References:
  - Schulten & Gordon 1975, J. Math. Phys. 16, 1961
  - Luscombe & Luban 1998, Phys. Rev. E 57, 7274 (numerical stability)
  - Edmonds 1957, Angular Momentum in Quantum Mechanics (Racah formula)
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

# -----------------------------------------------------------------------
# (l1 l2 L; 0 0 0) — closed-form via log-gamma
# -----------------------------------------------------------------------

def wigner3j_000(l1: int, l2: int, L: int) -> float:
    """Compute (l1 l2 L; 0 0 0) using the closed-form Racah formula.

    Uses log-gamma for numerical stability at large ell. Returns 0 when
    l1+l2+L is odd or the triangle condition fails.

    Formula (Edmonds, p. 125):
        (j1 j2 j3; 0 0 0) = (-1)^g * g! / [(g-j1)!(g-j2)!(g-j3)!]
                              * sqrt[(2g-2j1)!(2g-2j2)!(2g-2j3)! / (2g+1)!]
    where g = (j1+j2+j3)/2.
    """
    s_sum = l1 + l2 + L
    if s_sum % 2 != 0:
        return 0.0
    if abs(l1 - l2) > L or l1 + l2 < L:
        return 0.0

    g = s_sum // 2
    a = g - l1  # = (l2 + L - l1) / 2
    b = g - l2  # = (l1 + L - l2) / 2
    c = g - L   # = (l1 + l2 - L) / 2

    # log|w| = gammaln(g+1) - gammaln(a+1) - gammaln(b+1) - gammaln(c+1)
    #          + 0.5*(gammaln(2a+1) + gammaln(2b+1) + gammaln(2c+1) - gammaln(2g+2))
    log_w = (gammaln(g + 1)
             - gammaln(a + 1) - gammaln(b + 1) - gammaln(c + 1)
             + 0.5 * (gammaln(2 * a + 1) + gammaln(2 * b + 1)
                      + gammaln(2 * c + 1) - gammaln(2 * g + 2)))
    sign = (-1) ** g
    return sign * np.exp(log_w)


def wigner3j_000_vectorized(L: int, l1_arr: np.ndarray,
                            l2_min: int = 0,
                            l2_max: int | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Compute (l1, l2, L; 0 0 0) for all l1 and valid l2, vectorized.

    Returns (l2_grid, w3j) where w3j[i, j] = (l1_arr[i], l2_grid[j], L; 0 0 0).
    Zero where triangle fails or l1+l2+L is odd.
    """
    l1 = np.asarray(l1_arr, dtype=int)
    n_l1 = len(l1)
    if l2_max is None:
        l2_max = int(np.max(l1)) + L
    l2_grid = np.arange(l2_min, l2_max + 1, dtype=int)
    n_l2 = len(l2_grid)
    w = np.zeros((n_l1, n_l2))

    for j in range(n_l2):
        l2 = l2_grid[j]
        # Triangle: |l1-l2| <= L <= l1+l2 and parity: l1+l2+L even
        tri_ok = (np.abs(l1 - l2) <= L) & (l1 + l2 >= L)
        parity_ok = ((l1 + l2 + L) % 2 == 0)
        valid = tri_ok & parity_ok
        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        l1_v = l1[idx]
        s = (l1_v + l2 + L) // 2
        a = s - l1_v
        b = s - l2
        c = s - L
        sign = (-1.0) ** s
        log_w = (gammaln(s + 1)
                 - gammaln(a + 1) - gammaln(b + 1) - gammaln(c + 1)
                 + 0.5 * (gammaln(2*a + 1) + gammaln(2*b + 1)
                          + gammaln(2*c + 1) - gammaln(2*s + 2)))
        w[idx, j] = sign * np.exp(log_w)

    return l2_grid, w


# -----------------------------------------------------------------------
# Schulten-Gordon recursion coefficients
# -----------------------------------------------------------------------

def _sg_a(j, j1, j2, m3):
    """Scalar a(j) coefficient for the three-term recursion."""
    j_f = float(j)
    if j_f == 0:
        return 0.0
    arg = ((j_f**2 - (j1 - j2)**2)
           * ((j1 + j2 + 1)**2 - j_f**2)
           * (j_f**2 - m3**2))
    return np.sqrt(max(arg, 0.0)) / j_f


def _sg_b(j, j1, j2, m1, m2, m3):
    """Scalar b(j) coefficient for the three-term recursion."""
    j_f = float(j)
    denom = j_f * (j_f + 1.0)
    if abs(denom) < 1e-30:
        return 0.0
    return ((2.0 * j_f + 1.0)
            * (m3 * (j1 * (j1 + 1) - j2 * (j2 + 1))
               - (m1 - m2) * j_f * (j_f + 1.0))
            / denom)


def _sg_a_vec(j, j1_arr, j2, m3):
    """Vectorized a(j) over j1 array. j and j2 are scalars."""
    j_f = float(j)
    j2_f = float(j2)
    if j_f == 0:
        return np.zeros_like(j1_arr)
    arg = ((j_f**2 - (j1_arr - j2_f)**2)
           * ((j1_arr + j2_f + 1)**2 - j_f**2)
           * (j_f**2 - m3**2))
    return np.sqrt(np.maximum(arg, 0.0)) / j_f


def _sg_b_vec(j, j1_arr, j2, m1, m2, m3):
    """Vectorized b(j) over j1 array. j and j2 are scalars."""
    j_f = float(j)
    j2_f = float(j2)
    denom = j_f * (j_f + 1.0)
    if abs(denom) < 1e-30:
        return np.zeros_like(j1_arr)
    return ((2.0 * j_f + 1.0)
            * (m3 * (j1_arr * (j1_arr + 1) - j2_f * (j2_f + 1))
               - (m1 - m2) * j_f * (j_f + 1.0))
            / denom)


# -----------------------------------------------------------------------
# Scalar recursion for a single (j1, j2) pair
# -----------------------------------------------------------------------

def wigner3j_recurse(j1: int, j2: int, m1: int, m2: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute (j1, j2, j; m1, m2, m3) for all valid j.

    Uses backward Schulten-Gordon recursion (stable for 3j minimal solution),
    normalized with the sum rule and sign-fixed at j_max.

    Args:
        j1, j2: Fixed angular momenta.
        m1, m2: Magnetic quantum numbers. m3 = -(m1+m2).

    Returns:
        (j_array, w3j_array): j values and corresponding 3j symbols.
    """
    m3 = -(m1 + m2)
    j_min = max(abs(j1 - j2), abs(m3))
    j_max = j1 + j2
    n = j_max - j_min + 1

    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    j_vals = np.arange(j_min, j_max + 1)

    if n == 1:
        w = np.array([1.0 / np.sqrt(2.0 * j_min + 1.0)])
        if (-1) ** (j1 - j2 - m3) < 0:
            w = -w
        return j_vals, w

    # --- Bidirectional Schulten-Gordon recursion ---
    # Forward from j_min: stable near j_min, unstable far from it.
    # Backward from j_max: stable near j_max, unstable far from it.
    # Match at midpoint for robust result at all j.

    # Backward recursion
    w_back = np.zeros(n)
    w_back[-1] = 1.0
    j = j_max
    a_j = _sg_a(j, j1, j2, m3)
    b_j = _sg_b(j, j1, j2, m1, m2, m3)
    if abs(a_j) > 1e-30 and n >= 2:
        w_back[-2] = -b_j / a_j
    for idx in range(n - 3, -1, -1):
        j = j_vals[idx + 1]
        a_j = _sg_a(j, j1, j2, m3)
        b_j = _sg_b(j, j1, j2, m1, m2, m3)
        a_jp1 = _sg_a(j + 1, j1, j2, m3)
        if abs(a_j) > 1e-30:
            w_back[idx] = -(b_j * w_back[idx + 1] + a_jp1 * w_back[idx + 2]) / a_j

    # Forward recursion
    w_fwd = np.zeros(n)
    w_fwd[0] = 1.0
    if n >= 2:
        j = j_min
        b_j = _sg_b(j, j1, j2, m1, m2, m3)
        a_jp1 = _sg_a(j + 1, j1, j2, m3)
        if abs(a_jp1) > 1e-30:
            w_fwd[1] = -b_j / a_jp1
    for idx in range(2, n):
        j = j_vals[idx - 1]
        a_j = _sg_a(j, j1, j2, m3)
        b_j = _sg_b(j, j1, j2, m1, m2, m3)
        a_jp1 = _sg_a(j + 1, j1, j2, m3)
        if abs(a_jp1) > 1e-30:
            w_fwd[idx] = -(a_j * w_fwd[idx - 2] + b_j * w_fwd[idx - 1]) / a_jp1

    # --- Match forward and backward via median ratio ---
    # In the "classical region" (middle of the j range), both w_fwd and
    # w_back are proportional to the true 3j, so their ratio is constant.
    # Outside this region, one grows exponentially (contaminated by the
    # dominant solution). The median ratio filters out the contaminated points.
    ratios = []
    for k in range(max(1, n // 4), min(n - 1, 3 * n // 4)):
        if abs(w_back[k]) > 1e-30 and abs(w_fwd[k]) > 1e-30:
            ratios.append(w_back[k] / w_fwd[k])

    if len(ratios) >= 3:
        scale = float(np.median(ratios))
        # Use forward (scaled) for first half, backward for second half
        mid = n // 2
        w = np.empty(n)
        w[:mid] = w_fwd[:mid] * scale
        w[mid:] = w_back[mid:]
    elif len(ratios) > 0:
        scale = ratios[0]
        mid = n // 2
        w = np.empty(n)
        w[:mid] = w_fwd[:mid] * scale
        w[mid:] = w_back[mid:]
    else:
        w = w_back.copy()

    # --- Normalize: sum_j (2j+1)*w^2 = 1 ---
    norm_sq = np.sum((2.0 * j_vals + 1.0) * w**2)
    if norm_sq > 0:
        w /= np.sqrt(norm_sq)

    # --- Fix sign: w(j_max) has sign (-1)^{j1-j2-m3} ---
    target_sign = (-1) ** (j1 - j2 - m3)
    if w[-1] * target_sign < 0:
        w = -w

    return j_vals, w


# -----------------------------------------------------------------------
# Vectorized recursion: all l1 simultaneously for fixed L
# -----------------------------------------------------------------------

def wigner3j_vectorized(L: int, l1_array: np.ndarray,
                        m1: int = 2, m2: int = -2,
                        l2_min_global: int = 0,
                        l2_max_global: int | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Compute (l1, L, l2; m1, m2, m3) for all l1 and valid l2 simultaneously.

    The Schulten-Gordon three-term recursion varies the THIRD angular
    momentum (here l2), with l1 and L fixed per recursion step.
    All l1 values are processed in parallel (vectorized backward sweep).

    The 3j symbol ordering is (j1=l1, j2=L, j3=l2) with magnetic quantum
    numbers (m1, m2, m3) where m3 = -(m1+m2).  Column permutations of
    the 3j symbol differ only by a sign (-1)^{l1+L+l2}, so |w3j|^2 is
    independent of column ordering.

    Args:
        L:         Fixed second angular momentum (j2 in the recursion).
        l1_array:  Array of first angular momentum values (j1).
        m1, m2:    Magnetic quantum numbers on l1 and L respectively.
                   m3 = -(m1+m2) is assigned to l2.
        l2_min_global: Minimum l2 in output grid.
        l2_max_global: Maximum l2 in output grid. Default: max(l1) + L.

    Returns:
        l2_grid: 1-D int array of l2 values, shape (n_l2,).
        w3j:     2-D array of shape (n_l1, n_l2).
               w3j[i, j] = (l1_array[i], L, l2_grid[j]; m1, m2, m3).
               Zero where triangle condition |l1-L| ≤ l2 ≤ l1+L fails.
    """
    # Recursion on j (=l2) of (j1=l1, j2=L, j=l2; m1, m2, m3=-m1-m2)
    m3 = -(m1 + m2)
    l1 = np.asarray(l1_array, dtype=float)
    n_l1 = len(l1)
    L_f = float(L)

    l2_min = max(l2_min_global, abs(m3))
    l2_max = int(np.max(l1)) + L if l2_max_global is None else l2_max_global
    n_l2 = l2_max - l2_min + 1

    if n_l2 <= 0:
        return np.arange(l2_min, l2_max + 1, dtype=int), np.zeros((n_l1, 0))

    l2_grid = np.arange(l2_min, l2_max + 1, dtype=float)

    # Per-l1 triangle: l2 in [|l1-L|, l1+L] and l2 >= |m3|
    l2_lo = np.maximum(np.abs(l1 - L_f), abs(m3))
    l2_hi = l1 + L_f

    mask = ((l2_grid[None, :] >= l2_lo[:, None])
            & (l2_grid[None, :] <= l2_hi[:, None]))

    # --- Backward recursion ---
    # Recursion: a(j)*w(j-1) + b(j)*w(j) + a(j+1)*w(j+1) = 0
    # where j=l2, j1=l1[:], j2=L.
    # Vectorized: l1 is an array; L, j are scalars at each step.

    w = np.zeros((n_l1, n_l2))
    jmax_idx = np.clip((l2_hi - l2_min).astype(int), 0, n_l2 - 1)

    # Seed at each l1's j_max
    w[np.arange(n_l1), jmax_idx] = 1.0

    # First backward step: w(j_max-1) = -b(j_max)/a(j_max)
    jmax_m1_idx = np.clip(jmax_idx - 1, 0, n_l2 - 1)
    for i in range(n_l1):
        jm = int(l2_hi[i])
        # (j1=l1[i], j2=L, j=jm; m1, m2, m3)
        a_val = _sg_a(jm, l1[i], L_f, m3)
        b_val = _sg_b(jm, l1[i], L_f, m1, m2, m3)
        if abs(a_val) > 1e-30 and jmax_idx[i] > 0:
            w[i, jmax_m1_idx[i]] = -b_val / a_val

    # General backward sweep
    for idx in range(n_l2 - 3, -1, -1):
        j = l2_grid[idx + 1]
        active = (l2_grid[idx] <= l2_hi - 2) & mask[:, idx]

        if not np.any(active):
            continue

        # j1=l1[:] (array), j2=L (scalar), j=l2_grid[idx+1] (scalar)
        a_j = _sg_a_vec(j, l1, L_f, m3)
        b_j = _sg_b_vec(j, l1, L_f, m1, m2, m3)
        a_jp1 = _sg_a_vec(j + 1.0, l1, L_f, m3)

        safe_a = np.where(np.abs(a_j) > 1e-30, a_j, 1.0)
        new_val = -(b_j * w[:, idx + 1] + a_jp1 * w[:, idx + 2]) / safe_a
        new_val = np.where(np.abs(a_j) > 1e-30, new_val, 0.0)

        w[:, idx] = np.where(active, new_val, w[:, idx])

    # --- Normalize: sum_j (2j+1)*w^2 = 1 ---
    wt = 2.0 * l2_grid + 1.0
    norm_sq = np.sum(wt[None, :] * w**2, axis=1)
    safe_norm = np.where(norm_sq > 1e-30, np.sqrt(norm_sq), 1.0)
    w /= safe_norm[:, None]

    # --- Fix sign: w at j_max has sign (-1)^{j1-j2-m3} = (-1)^{l1-L-m3} ---
    target_sign = (-1.0) ** (l1 - L - m3)
    current_val = w[np.arange(n_l1), jmax_idx]
    needs_flip = (current_val * target_sign) < 0
    w[needs_flip] = -w[needs_flip]

    return l2_grid.astype(int), w
