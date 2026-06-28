"""Gate for augr.active_subspace -- the Constantine design subspace.

Fast, deterministic tests on a toy quadratic with a *known* gradient covariance: for
``f(z) = 1/2 z^T A z`` the gradient is ``A z``, so with ``z ~ N(0, I)`` the gradient
covariance is ``C = E[A z z^T A] = A^2`` -- eigenvectors of ``A``, eigenvalues ``lambda(A)^2``.
This pins the eigensolve, the activity scores, and the bootstrap robustness with no map forward
(and no jht), so it runs anywhere. The realistic end-to-end subspace (through the cut-sky MC
forward) is exercised by the driver / a slow test elsewhere.
"""

from __future__ import annotations

import numpy as np

from augr.active_subspace import (
    DesignSpec,
    active_subspace,
    activity_scores,
    bootstrap_eiguncertainty,
    collect_gradients,
    sample_designs,
    subspace_alignment,
)


def _spd_with_spectrum(eigs, seed=0):
    """A symmetric matrix with the given eigenvalues and a random orthonormal basis."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((len(eigs), len(eigs))))
    return q @ np.diag(eigs) @ q.T, q


def _toy_vg(a):
    """value_and_grad of f(z) = 1/2 z^T A z: (value, grad=A z). ctx ignored (noise-free)."""

    def vg(z, _ctx):
        z = np.asarray(z)
        return 0.5 * z @ a @ z, a @ z

    return vg


def test_recovers_known_subspace_eigvecs_and_spectrum():
    """5a: C = A^2 -> eigenvectors of A, eigenvalues lambda(A)^2 (large M)."""
    eigs = np.array([10.0, 3.0, 1.0, 0.3, 0.1])
    a, q = _spd_with_spectrum(eigs, seed=1)
    z = sample_designs(20000, len(eigs), sigma=1.0, method="gaussian", seed=2)
    gs = collect_gradients(_toy_vg(a), z, lambda _i: None, n_crn=1)
    sub = active_subspace(gs.grads)
    # leading eigenvector aligns with A's top eigenvector (A and A^2 share eigenvectors).
    assert subspace_alignment(sub.eigenvectors[:, 0], q[:, np.argmax(eigs)]) > 0.999
    # energy spectrum matches lambda(A)^2 / sum.
    expected = np.sort(eigs**2)[::-1]
    expected /= expected.sum()
    np.testing.assert_allclose(sub.energy, expected, rtol=0.03)


def test_convergence_to_C_squared_in_M():
    """5a: the recovered C -> A^2 as M grows (assert the trend, tighter at larger M)."""
    eigs = np.array([8.0, 2.0, 0.5])
    a, _q = _spd_with_spectrum(eigs, seed=3)
    target = a @ a
    errs = []
    for m in (500, 8000):
        z = sample_designs(m, len(eigs), sigma=1.0, method="gaussian", seed=4)
        gs = collect_gradients(_toy_vg(a), z, lambda _i: None, n_crn=1)
        errs.append(np.linalg.norm(gs.grads.T @ gs.grads / m - target))
    assert errs[1] < errs[0]


def test_bootstrap_leading_eig_clears_bulk():
    """5b: the leading-eigenvalue bootstrap interval excludes the second eigenvalue."""
    eigs = np.array([10.0, 1.0, 0.5, 0.2])
    a, _q = _spd_with_spectrum(eigs, seed=5)
    z = sample_designs(4000, len(eigs), sigma=1.0, method="gaussian", seed=6)
    gs = collect_gradients(_toy_vg(a), z, lambda _i: None, n_crn=1)
    boot = bootstrap_eiguncertainty(gs.grads, n_boot=300, n_active=1, seed=7)
    # 16th-percentile of lambda_0 sits well above the 84th-percentile of lambda_1.
    assert boot["eig_p16"][0] > boot["eig_p84"][1]
    # the 1-D active subspace is stable across resamples.
    assert boot["subspace_distance_p84"] < 0.1


def test_activity_scores_rank_active_knobs():
    """5c: with an axis-aligned A, activity scores rank the high-eigenvalue knobs first."""
    eigs = np.array([20.0, 5.0, 1.0, 0.2])
    a = np.diag(eigs)  # axis-aligned: knob k carries eigenvalue eigs[k]
    z = sample_designs(8000, len(eigs), sigma=1.0, method="gaussian", seed=8)
    gs = collect_gradients(_toy_vg(a), z, lambda _i: None, n_crn=1)
    sub = active_subspace(gs.grads)
    scores = activity_scores(sub, n_active=2)
    assert np.argmax(scores) == 0  # knob 0 (eigenvalue 20) most active
    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_n_active_reads_off_gap():
    """5b: n_active(threshold) reflects the spectral gap (one dominant direction here)."""
    eigs = np.array([100.0, 1.0, 0.5, 0.3])
    a, _q = _spd_with_spectrum(eigs, seed=9)
    z = sample_designs(8000, len(eigs), sigma=1.0, method="gaussian", seed=10)
    gs = collect_gradients(_toy_vg(a), z, lambda _i: None, n_crn=1)
    sub = active_subspace(gs.grads)
    assert sub.n_active(threshold=0.95) == 1


def test_designspec_log_roundtrip_and_zspace_gradient():
    """DesignSpec: z<->xi round-trip, and the z-space chain-rule factor (log mode)."""
    import jax

    fid = {"n_det": np.array([200.0, 400.0]), "mission_years": np.array(4.0)}
    spec = DesignSpec.from_pytree(fid, ("n_det[0]", "n_det[1]", "mission_years"), mode="log")
    # round-trip at the fiducial -> z = 0 -> xi = xi_fid.
    np.testing.assert_allclose(spec.standardize(spec.xi_fid_flat), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(spec.unstandardize(np.zeros(3))), spec.xi_fid_flat)

    # jax.grad of a raw-space quadratic composed with unstandardize == grad already in z-space
    # (log mode: d xi_k / d z_k = xi_k, so grad_z_k = grad_xi_k * xi_k).
    w = np.array([1.0, 2.0, 3.0])

    def loss_z(z):
        xi = spec.xi_fid_flat * jax.numpy.exp(z)
        return 0.5 * jax.numpy.sum(w * xi**2)

    g = np.asarray(jax.grad(loss_z)(np.zeros(3)))
    expected = w * spec.xi_fid_flat**2  # grad_xi = w*xi; *xi (chain) = w*xi^2 at z=0
    np.testing.assert_allclose(g, expected, rtol=1e-10)
