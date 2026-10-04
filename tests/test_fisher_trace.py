"""Channel-space (trace-form) per-bin Fisher: ``fisher._fisher_from_M_blocks``.

The per-bin Knox Fisher J_bᵀ Σ_b⁻¹ J_b equals (ν_b/2) Tr[M_b⁻¹ ∂M_b M_b⁻¹ ∂M_b],
and cond(Σ_b) = cond(M_b)². These tests lock in:

  1. The identity: trace form == Σ_b route on a well-conditioned instrument
     (Fisher and the bias right-hand side).
  2. Accuracy where the Σ_b route fails: against an mpmath (50 dps) evaluation
     of the Σ_b route itself on an ill-conditioned synthetic M.
  3. PICO-class: F is positive definite at the reionization bump, and the
     ℓ = 2 bin matches mpmath (slow).

Thresholds are set from values measured in these exact configurations
(recorded per test), with margin; do not loosen without re-measuring.
"""

from __future__ import annotations

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from augr.config import (
    DEFAULT_FIXED_MOMENT,
    DEFAULT_PRIORS_MOMENT,
    FIDUCIAL_BK15,
    FIDUCIAL_MOMENT,
    pico_like,
    simple_probe,
)
from augr.covariance import (
    bandpower_covariance_blocks,
    bandpower_M_blocks,
    bin_mode_counts,
)
from augr.fisher import (
    FisherForecast,
    _cinv_d_blocks,
    _fisher_from_blocks,
    _fisher_from_M_blocks,
    _jt_cinv_d_from_M_blocks,
    spectra_to_channel_matrix,
)
from augr.foregrounds import GaussianForegroundModel, MomentExpansionModel
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra


def _max_rel(A, B):
    """Max |A - B| normalized by sqrt(|B_ii B_jj|) (scale-free per entry)."""
    n = np.sqrt(np.outer(np.abs(np.diag(B)), np.abs(np.diag(B))))
    return float(np.max(np.abs(A - B) / n))


def _blocks(sig, inst, params):
    """Per-bin (J_b, Σ_b, ∂M_b, M_b, ν_b) for a SignalModel at ``params``."""
    J = sig.jacobian(params)
    n_spec, n_bins, n_chan = sig.n_spectra, sig.n_bins, len(sig.frequencies)
    J_blocks = J.reshape(n_spec, n_bins, -1).transpose(1, 0, 2)
    dM = spectra_to_channel_matrix(J_blocks.transpose(0, 2, 1), sig.freq_pairs, n_chan)
    cov = bandpower_covariance_blocks(sig, inst, params)
    M = bandpower_M_blocks(sig, inst, params)
    nu = inst.f_sky * jnp.asarray(bin_mode_counts(sig))
    return J_blocks, cov, dM, M, nu


@pytest.fixture(scope="module")
def well_conditioned():
    """simple_probe + Gaussian FG at ℓ 30-300: max cond(M_b) ~ 5e5."""
    inst = simple_probe()
    sig = SignalModel(inst, GaussianForegroundModel(), CMBSpectra(),
                      ell_min=30, ell_max=300, delta_ell=30)
    params = flatten_params(dict(FIDUCIAL_BK15), sig.parameter_names)
    return sig, inst, params


def test_spectra_to_channel_matrix_layout():
    pairs = [(0, 0), (0, 1), (1, 1)]
    X = np.asarray(spectra_to_channel_matrix(jnp.array([[1.0, 2.0, 3.0]]), pairs, 2))
    np.testing.assert_array_equal(X[0], [[1.0, 2.0], [2.0, 3.0]])


def test_trace_form_matches_knox_route_well_conditioned(well_conditioned):
    """Identity check. Measured 2.1e-12 max rel; gate 1e-9."""
    sig, inst, params = well_conditioned
    J_blocks, cov, dM, M, nu = _blocks(sig, inst, params)
    F_knox = np.asarray(_fisher_from_blocks(J_blocks, cov))
    F_trace = np.asarray(_fisher_from_M_blocks(dM, M, nu))
    assert _max_rel(F_trace, F_knox) < 1e-9


def test_bias_rhs_matches_knox_route_well_conditioned(well_conditioned):
    """Σ_b J_bᵀ Σ_b⁻¹ ΔD_b in both forms. Measured 2.1e-11 rel; gate 1e-8."""
    sig, inst, params = well_conditioned
    J_blocks, cov, dM, M, nu = _blocks(sig, inst, params)
    rng = np.random.default_rng(0)
    dd = rng.normal(size=sig.n_data) * np.abs(np.asarray(sig.data_vector(params))) * 1e-2
    dd_blocks = jnp.asarray(dd).reshape(sig.n_spectra, sig.n_bins).T
    u_knox = np.einsum("bsf,bs->f", np.asarray(J_blocks),
                       np.asarray(_cinv_d_blocks(cov, dd_blocks)))
    u_trace = np.asarray(_jt_cinv_d_from_M_blocks(
        dM, spectra_to_channel_matrix(dd_blocks, sig.freq_pairs, len(sig.frequencies)),
        M, nu))
    assert np.max(np.abs(u_trace - u_knox)) / np.max(np.abs(u_knox)) < 1e-8


def test_trace_form_accurate_where_knox_route_fails():
    """Synthetic 5-channel M = (rank-2 signal) + 1e-10·I, cond(M) = 3.0e10.

    Ground truth is the Σ_b route itself in mpmath (50 dps), independent of
    the trace identity. Measured: Σ_b route in fp64 1.0 max rel (fails),
    trace form 1.7e-6. Gates: trace < 1e-4, Σ_b route > 0.1 (so the test
    discriminates).
    """
    mp.mp.dps = 50
    rng = np.random.default_rng(1)
    n, nu = 5, 7.0
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    f = rng.normal(size=(n, 2))
    M = f @ f.T + np.diag(np.full(n, 1e-10))
    X = rng.normal(size=(3, n, n))
    dM = X + X.transpose(0, 2, 1)
    J = np.stack([dM[:, i, j] for (i, j) in pairs])

    Mm = mp.matrix(M.tolist())
    C = mp.matrix(len(pairs), len(pairs))
    for a, (i, j) in enumerate(pairs):
        for b, (k, m) in enumerate(pairs):
            C[a, b] = (Mm[i, k] * Mm[j, m] + Mm[i, m] * Mm[j, k]) / nu
    Jm = mp.matrix(J.tolist())
    cols = [mp.lu_solve(C, Jm.column(c)) for c in range(Jm.cols)]
    F_mp = np.array([[float(sum(Jm[s, a] * cols[c][s] for s in range(Jm.rows)))
                      for c in range(Jm.cols)] for a in range(Jm.cols)])

    cov = np.array(C.tolist(), dtype=float)
    F_knox = np.asarray(_fisher_from_blocks(jnp.asarray(J)[None], jnp.asarray(cov)[None]))
    F_trace = np.asarray(_fisher_from_M_blocks(
        jnp.asarray(dM)[None], jnp.asarray(M)[None], jnp.asarray([nu])))
    assert np.linalg.cond(M) > 1e10
    assert _max_rel(F_trace, F_mp) < 1e-4
    assert _max_rel(F_knox, F_mp) > 0.1


@pytest.mark.slow
def test_pico_reionization_bump_fisher_is_positive_definite():
    """PICO 21-band, moment FG, per-ℓ bins ℓ = 2-8 (cond(M_b) ~ 4e14 at ℓ = 2).

    Measured: smallest eigenvalue of the correlation-normalized F +2.8e-5
    (the Σ_b route gives -6.7e-8 here without priors, and -0.09 over
    ℓ = 2-300 with delensing); ℓ = 2 bin vs mpmath trace form 5.1e-7.
    Gates: > 0 and < 1e-5.
    """
    inst = pico_like()
    sig = SignalModel(inst, MomentExpansionModel(), CMBSpectra(),
                      ell_min=2, ell_max=8, delta_ell=30)
    ff = FisherForecast(sig, inst, dict(FIDUCIAL_MOMENT),
                        priors=DEFAULT_PRIORS_MOMENT, fixed_params=DEFAULT_FIXED_MOMENT)
    assert ff.min_correlation_eigenvalue() > 0

    params = flatten_params(dict(FIDUCIAL_MOMENT), sig.parameter_names)
    free_idx = [sig.parameter_names.index(n) for n in ff.free_parameter_names]
    J = sig.jacobian(params)[:, jnp.array(free_idx)]
    n_chan = len(sig.frequencies)
    J_blocks = J.reshape(sig.n_spectra, sig.n_bins, -1).transpose(1, 0, 2)
    dM = np.asarray(spectra_to_channel_matrix(
        J_blocks.transpose(0, 2, 1), sig.freq_pairs, n_chan))[0]
    M = np.asarray(bandpower_M_blocks(sig, inst, params))[0]
    nu = inst.f_sky * bin_mode_counts(sig)[0]

    mp.mp.dps = 50
    Mi = mp.matrix(M.tolist()) ** -1
    P = [Mi * mp.matrix(x.tolist()) for x in dM]
    F_mp = np.zeros((len(P), len(P)))
    for a in range(len(P)):
        for c in range(a, len(P)):
            F_mp[a, c] = F_mp[c, a] = float(
                0.5 * nu * sum((P[a] * P[c])[k, k] for k in range(n_chan)))
    F_trace = np.asarray(_fisher_from_M_blocks(
        jnp.asarray(dM)[None], jnp.asarray(M)[None], jnp.asarray([nu])))
    assert _max_rel(F_trace, F_mp) < 1e-5
