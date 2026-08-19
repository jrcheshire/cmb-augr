"""
wigner_jax.py -- JAX port of augr.wigner for differentiable full-sky delensing.

Mirrors the two computation paths of the numpy ``augr.wigner`` module so the
full-sky QE / lensing-kernel drivers can run inside ``jax.jit`` / ``jax.grad``
(issue #45 Stage 3):

  1. ``wigner3j_000_vectorized_jax`` -- (l1 l2 L; 0 0 0) closed-form Racah via
     ``jax.scipy.special.gammaln``. Fully vectorized, no recursion.
  2. ``wigner3j_vectorized_jax`` -- spin-2 Schulten-Gordon three-term
     recursion. The numpy version is a backward-only sweep over l2 with
     per-l1 seeds; here that sweep is a ``lax.scan`` carrying the two most
     recent columns. All l1 rows are processed in parallel.

Conventions, coefficient signs (including the ``_sg_b`` m_3 term), the sum-rule
normalization ``sum_j (2j+1) w^2 = 1``, and the ``(-1)^{l1-L-m3}`` sign fix are
identical to ``augr.wigner``; ``tests/test_wigner.py`` locks the numpy version
to sympy truth, and the ``wigner_jax`` port is validated bit-for-bit against it.

Static-shape contract: ``L`` and the l2 grid bounds are Python ints (they set
array shapes), so these functions trace with static shapes -- l1 values are the
only array input.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import gammaln

_TINY = 1e-30


def _parity_sign(x: jnp.ndarray) -> jnp.ndarray:
    """(-1)^x for integer-valued x, as +-1.0 (avoids complex jnp.power)."""
    return jnp.where(jnp.round(x).astype(jnp.int64) % 2 == 0, 1.0, -1.0)


# -----------------------------------------------------------------------
# (l1 l2 L; 0 0 0) -- closed-form via log-gamma
# -----------------------------------------------------------------------

def spin0_body(L_f, l1, l2_min: int, l2_max: int) -> jnp.ndarray:
    """(l1 l2 L; 0 0 0) closed-form table for a single L; ``L`` may be traced.

    ``L_f`` scalar value (concrete or tracer); ``l1`` jnp array; ``l2_min,
    l2_max`` static ints. ``lax.map``-friendly core of
    :func:`wigner3j_000_vectorized_jax`. Returns w of shape
    ``(len(l1), l2_max - l2_min + 1)``.
    """
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)   # (n_l2,)
    l1c = l1[:, None]                 # (n_l1, 1)
    l2c = l2_grid[None, :]            # (1, n_l2)
    ssum = l1c + l2c + L_f            # (n_l1, n_l2)

    tri = (jnp.abs(l1c - l2c) <= L_f) & (l1c + l2c >= L_f)
    parity = (jnp.round(ssum).astype(jnp.int64) % 2) == 0
    valid = tri & parity

    s = jnp.floor(ssum / 2.0)
    # Clamp the arguments on invalid entries so gammaln stays finite (those
    # entries are masked to 0 below); on valid entries a, b, c >= 0.
    a = jnp.maximum(s - l1c, 0.0)
    b = jnp.maximum(s - l2c, 0.0)
    c = jnp.maximum(s - L_f, 0.0)
    s_safe = jnp.maximum(s, 0.0)

    log_w = (gammaln(s_safe + 1.0)
             - gammaln(a + 1.0) - gammaln(b + 1.0) - gammaln(c + 1.0)
             + 0.5 * (gammaln(2.0 * a + 1.0) + gammaln(2.0 * b + 1.0)
                      + gammaln(2.0 * c + 1.0) - gammaln(2.0 * s_safe + 2.0)))
    sign = _parity_sign(s_safe)
    return jnp.where(valid, sign * jnp.exp(log_w), 0.0)


def wigner3j_000_vectorized_jax(L: int, l1_arr,
                                l2_min: int = 0,
                                l2_max: int | None = None
                                ) -> tuple[np.ndarray, jnp.ndarray]:
    """JAX port of ``wigner.wigner3j_000_vectorized`` (concrete ``L``).

    Returns (l2_grid, w3j) with w3j[i, j] = (l1_arr[i], l2_grid[j], L; 0 0 0),
    zero where the triangle fails or l1+l2+L is odd. The ``lax.map``-friendly
    traced-``L`` core is :func:`spin0_body`.
    """
    l1 = jnp.asarray(l1_arr, dtype=float)
    if l2_max is None:
        l2_max = int(np.max(np.asarray(l1_arr))) + int(L)
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    w = spin0_body(float(L), l1, l2_min, l2_max)
    return l2_grid.astype(int), w


# -----------------------------------------------------------------------
# Schulten-Gordon recursion coefficients (elementwise jnp)
# -----------------------------------------------------------------------

def _sg_a_jax(j, j1, j2, m3):
    """a(j) coefficient, elementwise over (j, j1); j2, m3 broadcast."""
    arg = ((j ** 2 - (j1 - j2) ** 2)
           * ((j1 + j2 + 1.0) ** 2 - j ** 2)
           * (j ** 2 - m3 ** 2))
    # Double-where. ``sqrt(maximum(arg, 0))`` is right in value but its reverse
    # pass is NaN wherever arg <= 0: d/dx sqrt(x) is infinite at x=0 and the
    # clamp contributes a zero, so the cotangent is inf * 0. Substitute the
    # argument BEFORE the sqrt so the untaken branch never evaluates one.
    ok = (arg > 0.0) & (j != 0.0)
    safe_arg = jnp.where(ok, arg, 1.0)
    safe_j = jnp.where(j == 0.0, 1.0, j)
    return jnp.where(ok, jnp.sqrt(safe_arg) / safe_j, 0.0)


def _sg_b_jax(j, j1, j2, m1, m2, m3):
    """b(j) coefficient (Schulten-Gordon 1975 Eq. 5), elementwise."""
    denom = j * (j + 1.0)
    safe_denom = jnp.where(jnp.abs(denom) < _TINY, 1.0, denom)
    b = ((2.0 * j + 1.0)
         * (-m3 * (j1 * (j1 + 1.0) - j2 * (j2 + 1.0))
            - (m1 - m2) * j * (j + 1.0))
         / safe_denom)
    return jnp.where(jnp.abs(denom) < _TINY, 0.0, b)


# -----------------------------------------------------------------------
# Vectorized spin-2 recursion: all l1 simultaneously for fixed L
# -----------------------------------------------------------------------

def spin2_body(j2, l1, m1: int, m2: int, m3: int,
               l2_min: int, l2_max: int) -> jnp.ndarray:
    """Spin-2 Schulten-Gordon table for a single j2, ``j2`` may be traced.

    ``j2`` is a scalar value (concrete or a JAX tracer); ``l1`` is a jnp
    array; ``m1, m2, m3, l2_min, l2_max`` are static Python ints (they set
    array shapes and carry no control flow on ``j2``). This is the
    ``lax.map``-friendly core; the public ``wigner3j_vectorized_jax`` wraps
    it with the concrete-``j2`` grid bookkeeping and edge cases. Returns
    w_full of shape ``(len(l1), l2_max - l2_min + 1)``.
    """
    n_l1 = l1.shape[0]
    n_l2 = l2_max - l2_min + 1
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)   # (n_l2,)
    rows = jnp.arange(n_l1)

    l2_hi = l1 + j2
    m1_ok = jnp.abs(m1) <= l1
    mask = ((l2_grid[None, :] >= jnp.maximum(jnp.abs(l1 - j2), abs(m3))[:, None])
            & (l2_grid[None, :] <= l2_hi[:, None])
            & m1_ok[:, None])                              # (n_l1, n_l2)

    jmax_idx = jnp.clip((l2_hi - l2_min).astype(int), 0, n_l2 - 1)

    # --- Seeds. Order matters: scatter the first backward step (at
    #     jmax-1), then the j_max seed (=1), so that when jmax_idx == 0 the
    #     unit seed wins (matching the numpy jmax_idx > 0 guard). ---
    a_jmax = _sg_a_jax(l2_hi, l1, j2, m3)                  # j = l1 + j2, per row
    b_jmax = _sg_b_jax(l2_hi, l1, j2, m1, m2, m3)
    safe_a_jmax = jnp.where(jnp.abs(a_jmax) > _TINY, a_jmax, 1.0)
    first = jnp.where((jnp.abs(a_jmax) > _TINY) & m1_ok,
                      -b_jmax / safe_a_jmax, 0.0)
    jmax_m1_idx = jnp.clip(jmax_idx - 1, 0, n_l2 - 1)

    w_seed = jnp.zeros((n_l1, n_l2))
    w_seed = w_seed.at[rows, jmax_m1_idx].set(first)
    w_seed = w_seed.at[rows, jmax_idx].set(jnp.where(m1_ok, 1.0, 0.0))

    # --- Backward scan over idx = n_l2-3 .. 0. Carry = (w_{idx+1}, w_{idx+2}).
    if n_l2 >= 3:
        idxs = jnp.arange(n_l2 - 3, -1, -1)                # descending (static)
        j_step = l2_grid[idxs + 1]                         # j at each step
        l2_at = l2_grid[idxs]
        mask_cols = mask[:, idxs].T                        # (n_steps, n_l1)
        wseed_cols = w_seed[:, idxs].T                     # (n_steps, n_l1)

        def body(carry, x):
            w1, w2 = carry                                 # w_{idx+1}, w_{idx+2}
            j, l2i, mcol, wscol = x
            a_j = _sg_a_jax(j, l1, j2, m3)
            b_j = _sg_b_jax(j, l1, j2, m1, m2, m3)
            a_jp1 = _sg_a_jax(j + 1.0, l1, j2, m3)
            safe_a = jnp.where(jnp.abs(a_j) > _TINY, a_j, 1.0)
            rec = -(b_j * w1 + a_jp1 * w2) / safe_a
            rec = jnp.where(jnp.abs(a_j) > _TINY, rec, 0.0)
            active = (l2i <= (l2_hi - 2.0)) & mcol
            w_idx = jnp.where(active, rec, wscol)
            return (w_idx, w1), w_idx

        init = (w_seed[:, n_l2 - 2], w_seed[:, n_l2 - 1])
        _, w_cols = lax.scan(body, init, (j_step, l2_at, mask_cols, wseed_cols))
        # w_cols is idx-descending (idx = n_l2-3 .. 0); reverse to ascending.
        w_low = w_cols[::-1].T                             # (n_l1, n_l2-2)
        w_full = jnp.concatenate([w_low, w_seed[:, n_l2 - 2:n_l2]], axis=1)
    else:
        w_full = w_seed

    # --- Normalize: sum_j (2j+1) w^2 = 1 per row. ---
    wt = 2.0 * l2_grid + 1.0
    norm_sq = jnp.sum(wt[None, :] * w_full ** 2, axis=1)
    # NOTE: this single-where has the same latent NaN as _sg_a_jax had -- rows
    # with |m1| > l1 are identically zero, so norm_sq == 0 and sqrt's reverse
    # pass is infinite there. Deliberately left alone because it is unreachable:
    # both inputs (j2, l1) are integer multipole indices, so nothing
    # differentiable sits upstream of w_full and the cotangent never arrives.
    # Verified -- mutating this line back fails no test, while mutating
    # _sg_a_jax fails two. If a caller ever makes the table depend on a
    # continuous parameter, apply the _sg_a_jax double-where here too.
    safe_norm = jnp.where(norm_sq > _TINY, jnp.sqrt(norm_sq), 1.0)
    w_full = w_full / safe_norm[:, None]

    # --- Sign fix: w at j_max has sign (-1)^{l1-j2-m3}. ---
    target_sign = _parity_sign(l1 - j2 - m3)
    current_val = w_full[rows, jmax_idx]
    needs_flip = (current_val * target_sign) < 0
    w_full = jnp.where(needs_flip[:, None], -w_full, w_full)

    # --- Guard |m2| <= j2. The symbol vanishes identically when a magnetic
    #     quantum number exceeds its own angular momentum, but the recursion
    #     above only ever constrains m1 (row-wise, via ``m1_ok``) and m3 (via
    #     the l2 lower bound) -- nothing tests m2 against j2, so the seed and
    #     normalization would hand back a unit-norm but meaningless table.
    #     ``wigner3j_vectorized_jax`` short-circuits this case for concrete j2;
    #     the traced core must too, because ``lax.map`` callers reach it
    #     directly. No-op for the m2=0 delensing callers.
    return jnp.where(jnp.abs(m2) <= j2, w_full, 0.0)


def wigner3j_vectorized_jax(j2: int, l1_array,
                            m1: int = 2, m2: int = -2,
                            l2_min_global: int = 0,
                            l2_max_global: int | None = None
                            ) -> tuple[np.ndarray, jnp.ndarray]:
    """JAX port of ``wigner.wigner3j_vectorized`` (backward SG sweep as scan).

    Computes (l1, j2, l2; m1, m2, m3) for all l1 and valid l2. Same seeds,
    coefficient signs, normalization, and sign fix as the numpy version.
    ``j2`` is a concrete int here; the ``lax.map``-friendly traced-``j2`` core
    is :func:`spin2_body`.
    """
    m3 = -(m1 + m2)
    l1 = jnp.asarray(l1_array, dtype=float)
    n_l1 = l1.shape[0]

    l2_min = max(l2_min_global, abs(m3))
    l2_max = (int(np.max(np.asarray(l1_array))) + int(j2)
              if l2_max_global is None else l2_max_global)
    n_l2 = l2_max - l2_min + 1

    if n_l2 <= 0 or abs(m2) > j2:
        return (np.arange(l2_min, l2_max + 1, dtype=int),
                jnp.zeros((n_l1, max(n_l2, 0))))

    w_full = spin2_body(float(j2), l1, m1, m2, m3, l2_min, l2_max)
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    return l2_grid.astype(int), w_full
