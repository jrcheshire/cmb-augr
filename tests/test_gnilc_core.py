"""Gate for the public geometry-agnostic GNILC core (augr.gnilc, no [compsep] extra).

The core — ``gnilc_fg_estimator`` / ``aic_dimension`` / ``matsqrt_pair`` — consumes only
channel-space covariance matrices, so unlike test_gnilc.py this file is NOT gated on
ducc0: it locks that other-geometry consumers (flat-sky patch pipelines) can import and
drive the core without the SHT stack installed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from augr.gnilc import aic_dimension, gnilc_fg_estimator, matsqrt_pair


def _fg_covariances(seed: int, n: int = 5, k: int = 2):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov_n = a @ a.T + n * np.eye(n)  # random PD nuisance (CMB+noise)
    fdirs = rng.normal(size=(n, k))
    cov_t = cov_n + fdirs @ np.diag([200.0, 80.0]) @ fdirs.T  # + k strong FG modes
    return jnp.asarray(cov_t), jnp.asarray(cov_n), fdirs


def test_public_estimator_recovers_fg_dim_and_projects() -> None:
    """AIC recovers the injected FG dimension; W is the C_n^{1/2}-projector onto it."""
    cov_t, cov_n, fdirs = _fg_covariances(0)
    w, m = gnilc_fg_estimator(cov_t, cov_n, m_bias=0)
    w = np.asarray(w)
    assert int(m) == fdirs.shape[1]
    np.testing.assert_allclose(w @ w, w, atol=1e-8)  # idempotent (oblique projector)
    f = fdirs[:, 0]
    np.testing.assert_allclose(w @ f, f, rtol=1e-6, atol=1e-8)


def test_public_estimator_batches_over_leading_dim() -> None:
    cov_t0, cov_n0, _ = _fg_covariances(0)
    cov_t1, cov_n1, _ = _fg_covariances(3)
    w_b, m_b = gnilc_fg_estimator(
        jnp.stack([cov_t0, cov_t1]), jnp.stack([cov_n0, cov_n1]), m_bias=0
    )
    w0, m0 = gnilc_fg_estimator(cov_t0, cov_n0, m_bias=0)
    assert w_b.shape == (2, 5, 5) and m_b.shape == (2,)
    np.testing.assert_allclose(np.asarray(w_b[0]), np.asarray(w0), atol=1e-12)
    assert int(m_b[0]) == int(m0)


def test_aic_dimension_counts_super_unity_eigenvalues() -> None:
    """Nuisance eigenvalues sit at 1 (zero AIC cost); strong FG modes are counted."""
    lam_desc = jnp.asarray([50.0, 20.0, 1.0, 1.0, 1.0])
    assert int(aic_dimension(lam_desc)) == 2


def test_aic_dimension_ignores_sub_unity_eigenvalues() -> None:
    """λ < 1 = nuisance over-modeled in that direction (e.g. a regularization floor
    against a rank-deficient noiseless total): no foreground content — must not be
    penalized into the subspace via −ln λ (pre-clamp this saturated m = n and W
    degenerated to the identity)."""
    lam_desc = jnp.asarray([50.0, 20.0, 1.0, 1e-8, 1e-12])
    assert int(aic_dimension(lam_desc)) == 2


def test_estimator_rank_deficient_total_does_not_saturate() -> None:
    """k FG modes + rank-1 nuisance + floor, total rank < n: m stays k, W != I."""
    rng = np.random.default_rng(5)
    n, k = 4, 2
    ones = np.ones((n, 1))
    cov_n = 10.0 * (ones @ ones.T) + 1e-5 * np.eye(n)  # rank-1 CMB + tiny floor
    fdirs = rng.normal(size=(n, k))
    cov_t = 10.0 * (ones @ ones.T) + fdirs @ np.diag([200.0, 80.0]) @ fdirs.T  # rank 3 of 4
    w, m = gnilc_fg_estimator(jnp.asarray(cov_t), jnp.asarray(cov_n), m_bias=0)
    assert int(m) == k
    assert not np.allclose(np.asarray(w), np.eye(n), atol=1e-3)


def test_matsqrt_pair_roundtrip() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 4))
    cov = jnp.asarray(a @ a.T + 4 * np.eye(4))
    half, ihalf = matsqrt_pair(cov)
    np.testing.assert_allclose(np.asarray(half @ half), np.asarray(cov), atol=1e-10)
    np.testing.assert_allclose(np.asarray(half @ ihalf), np.eye(4), atol=1e-10)
