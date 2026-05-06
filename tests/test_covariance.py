"""Tests for covariance.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from augr.covariance import (
    _build_M,
    _nu_b,
    bandpower_covariance,
    bandpower_covariance_blocks,
    bandpower_covariance_blocks_from_noise,
    bandpower_covariance_full,
    bandpower_covariance_full_from_noise,
)
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_chan_instrument():
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    return Instrument(channels=(
        Channel(90.0,  500, 400.0, 30.0, efficiency=eff),
        Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)


@pytest.fixture(scope="module")
def signal_model(two_chan_instrument):
    return SignalModel(
        two_chan_instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
    )


FIDUCIAL = {
    "r": 0.0, "A_lens": 1.0,
    "A_dust": 4.7, "beta_dust": 1.6, "alpha_dust": -0.58, "T_dust": 19.6,
    "A_sync": 1.5, "beta_sync": -3.1, "alpha_sync": -0.6,
    "epsilon": 0.0, "Delta_dust": 0.0,
}


# -----------------------------------------------------------------------
# nu_b (effective modes)
# -----------------------------------------------------------------------

def test_nu_b_per_ell():
    """Per-ℓ bin at ℓ: ν_b = f_sky × (2ℓ+1)."""
    edges = [(10, 10), (11, 11), (50, 50)]
    nu = _nu_b(edges, f_sky=1.0)
    assert abs(float(nu[0]) - 21.0) < 1e-10
    assert abs(float(nu[1]) - 23.0) < 1e-10
    assert abs(float(nu[2]) - 101.0) < 1e-10


def test_nu_b_wide_bin():
    """Wide bin [30, 64]: ν_b = f_sky × (64-30+1) × (30+64+1) = 35 × 95 = 3325."""
    edges = [(30, 64)]
    nu = _nu_b(edges, f_sky=1.0)
    assert abs(float(nu[0]) - 35 * 95) < 1e-10


def test_nu_b_fsky_scaling():
    """ν_b scales linearly with f_sky."""
    edges = [(50, 84)]
    nu1 = float(_nu_b(edges, f_sky=0.5)[0])
    nu2 = float(_nu_b(edges, f_sky=1.0)[0])
    assert abs(nu2 / nu1 - 2.0) < 1e-10


# -----------------------------------------------------------------------
# Covariance matrix structure
# -----------------------------------------------------------------------

def test_covariance_shape(signal_model, two_chan_instrument):
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    n = signal_model.n_data
    assert cov.shape == (n, n)


def test_covariance_symmetric(signal_model, two_chan_instrument):
    """Covariance matrix is symmetric."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    assert jnp.allclose(cov, cov.T, rtol=1e-6)


def test_covariance_positive_diagonal(signal_model, two_chan_instrument):
    """Diagonal entries (variances) are positive."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    assert jnp.all(jnp.diag(cov) > 0)


def test_covariance_block_diagonal(signal_model, two_chan_instrument):
    """Covariance is block-diagonal: different bins are uncorrelated."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    n_spec = signal_model.n_spectra
    n_bins = signal_model.n_bins

    # Check a few off-diagonal bin pairs are zero
    for s1 in range(n_spec):
        for s2 in range(n_spec):
            for b1 in range(min(n_bins, 3)):
                for b2 in range(min(n_bins, 3)):
                    if b1 != b2:
                        val = float(cov[s1 * n_bins + b1, s2 * n_bins + b2])
                        assert val == 0.0, f"Non-zero off-bin covariance at s1={s1},b1={b1},s2={s2},b2={b2}"


def test_covariance_positive_semidefinite(signal_model, two_chan_instrument):
    """Covariance matrix has non-negative eigenvalues."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    eigvals = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigvals >= -1e-10 * jnp.max(eigvals))


# -----------------------------------------------------------------------
# Knox formula values
# -----------------------------------------------------------------------

def test_auto_spectrum_variance(signal_model, two_chan_instrument):
    """Variance of auto-spectrum (i,i) at bin b equals 2*M[i,i,b]^2 / nu_b.

    For an auto-spectrum (i,j) = (i,i), the Knox formula gives:
        Cov = (M_ii * M_ii + M_ii * M_ii) / nu_b = 2 * M_ii^2 / nu_b
    """
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov = bandpower_covariance(signal_model, two_chan_instrument, params)
    M = _build_M(signal_model, two_chan_instrument, params)

    from augr.covariance import _nu_b
    nu = _nu_b(signal_model._bin_edges, two_chan_instrument.f_sky)

    n_bins = signal_model.n_bins
    # Spectrum (0,0) is the first spectrum (index 0)
    # Its variance at bin b is at index [0 * n_bins + b, 0 * n_bins + b]
    for b in range(min(5, n_bins)):
        expected = 2.0 * float(M[0, 0, b]) ** 2 / float(nu[b])
        actual = float(cov[b, b])
        assert abs(actual - expected) / expected < 1e-6, \
            f"Auto-spectrum variance wrong at bin {b}"


def test_noise_increases_variance(signal_model, two_chan_instrument):
    """Higher noise → larger diagonal covariance entries."""
    params = flatten_params(FIDUCIAL, signal_model.parameter_names)
    cov_normal = bandpower_covariance(signal_model, two_chan_instrument, params)

    # Create instrument with 10× worse NET
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    noisy_inst = Instrument(channels=(
        Channel(90.0,  500, 4000.0, 30.0, efficiency=eff),
        Channel(150.0, 1000, 3000.0, 20.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)
    cov_noisy = bandpower_covariance(signal_model, noisy_inst, params)

    # Auto-spectrum variance should be larger
    assert float(jnp.diag(cov_noisy).mean()) > float(jnp.diag(cov_normal).mean())


# -----------------------------------------------------------------------
# Residual template contributes to M = S + N on auto-blocks only
# -----------------------------------------------------------------------

# A distinctive flat template amplitude makes the expected delta easy to
# read off: adding A_res × T_res to an auto bandpower shifts W @ (...) by
# exactly A_res × T_res (flat templates survive the binning unchanged).
_RES_T_AMPLITUDE = 1e-4      # uK^2


@pytest.fixture(scope="module")
def signal_model_with_template(two_chan_instrument):
    """Signal model with a flat residual template, 2-channel / same binning."""
    ells = np.arange(2, 400, dtype=float)
    cl = np.full_like(ells, _RES_T_AMPLITUDE)
    return SignalModel(
        two_chan_instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        residual_template_cl=cl, residual_template_ells=ells,
    )


def _fid_with_a_res(names, a_res):
    return flatten_params({**FIDUCIAL, "A_res": a_res}, names)


def test_build_M_adds_template_on_autos(signal_model_with_template,
                                         two_chan_instrument):
    """A_res×T_res enters M on i==j only; cross-blocks are unchanged.

    This is the regression test for a bug where cmb_bb_unbinned omitted
    A_res×T_res, so _build_M under-counted signal variance on the
    post-CompSep auto-spectrum and produced an over-optimistic Fisher.
    """
    names = signal_model_with_template.parameter_names
    p0 = _fid_with_a_res(names, 0.0)
    p1 = _fid_with_a_res(names, 1.0)

    M0 = _build_M(signal_model_with_template, two_chan_instrument, p0)
    M1 = _build_M(signal_model_with_template, two_chan_instrument, p1)
    dM = M1 - M0

    n_chan = len(two_chan_instrument.channels)
    for i in range(n_chan):
        for j in range(n_chan):
            block = dM[i, j, :]
            if i == j:
                # Flat template with amplitude _RES_T_AMPLITUDE, binned
                # by a row-sum-1 tophat, equals _RES_T_AMPLITUDE per bin.
                assert jnp.allclose(block, _RES_T_AMPLITUDE, rtol=1e-8), \
                    f"Auto-block ({i},{j}) missing residual template"
            else:
                assert jnp.allclose(block, 0.0, atol=1e-14), \
                    f"Cross-block ({i},{j}) moved with A_res (should not)"


def test_covariance_auto_variance_scales_with_a_res(signal_model_with_template,
                                                     two_chan_instrument):
    """Auto-variance grows when A_res > 0 (template enters M = S + N).

    Knox: Var(C_b^{ii}) = 2 (M_ii)^2 / nu_b. Adding A_res×T_res to M_ii
    must raise the auto-spectrum variance, confirming the template
    contributes to the covariance and not just the data vector.
    """
    names = signal_model_with_template.parameter_names
    p0 = _fid_with_a_res(names, 0.0)
    p1 = _fid_with_a_res(names, 1.0)

    cov0 = bandpower_covariance(signal_model_with_template,
                                two_chan_instrument, p0)
    cov1 = bandpower_covariance(signal_model_with_template,
                                two_chan_instrument, p1)

    n_bins = signal_model_with_template.n_bins
    # Spectrum 0 is (0,0), the 90×90 auto: its diagonal lives at indices [0, n_bins).
    auto_var0 = jnp.diag(cov0)[:n_bins]
    auto_var1 = jnp.diag(cov1)[:n_bins]
    assert jnp.all(auto_var1 > auto_var0), \
        "Auto-spectrum variance did not increase with A_res"


def test_covariance_blocks_from_noise_adds_template_on_autos(
        signal_model_with_template, two_chan_instrument):
    """External-noise covariance path also picks up the residual template."""
    names = signal_model_with_template.parameter_names
    p0 = _fid_with_a_res(names, 0.0)
    p1 = _fid_with_a_res(names, 1.0)

    ells = signal_model_with_template.ells
    n_chan = len(two_chan_instrument.channels)
    # Use constant external noise so the only difference between cov0/cov1
    # is the residual template on the diagonal.
    nl = jnp.ones((n_chan, len(ells))) * 1e-5
    f_sky = two_chan_instrument.f_sky

    cov0 = bandpower_covariance_blocks_from_noise(
        signal_model_with_template, nl, f_sky, p0)
    cov1 = bandpower_covariance_blocks_from_noise(
        signal_model_with_template, nl, f_sky, p1)
    # Auto (0,0) block at spectrum index 0 (pairs are (0,0), (0,1), (1,1))
    auto_var0 = cov0[:, 0, 0]
    auto_var1 = cov1[:, 0, 0]
    assert jnp.all(auto_var1 > auto_var0), \
        "External-noise covariance did not pick up residual template"

    # Cross-spectrum (0,1) is spectrum index 1: residual template must
    # not leak into the cross-block variance.
    cross_var0 = cov0[:, 1, 1]
    cross_var1 = cov1[:, 1, 1]
    assert jnp.allclose(cross_var0, cross_var1, rtol=1e-10), \
        "Cross-spectrum variance changed with A_res (should not)"


# -----------------------------------------------------------------------
# Measured BPWF covariance path
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def per_ell_signal_models(two_chan_instrument):
    """Two SignalModels at delta_ell=1: one default, one BPWF=identity.

    These should produce numerically identical bandpower covariances --
    a per-ℓ disjoint-delta BPWF reduces the BPWF-aware Knox sum to the
    same per-bin formula the fast path uses.
    """
    ell_min, ell_max = 30, 80
    sm_fast = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        delta_ell=1, ell_per_bin_below=ell_min,
    )
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.eye(ells_in.size)
    sm_bpwf = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    return sm_fast, sm_bpwf


def test_full_covariance_per_ell_identity_matches_fast_path(
        per_ell_signal_models, two_chan_instrument):
    """Identity BPWF on per-ℓ bins reproduces the fast-path covariance."""
    sm_fast, sm_bpwf = per_ell_signal_models
    params = flatten_params(FIDUCIAL, sm_fast.parameter_names)

    cov_fast = bandpower_covariance(sm_fast, two_chan_instrument, params)
    cov_full = bandpower_covariance_full(sm_bpwf, two_chan_instrument, params)
    assert cov_fast.shape == cov_full.shape
    assert jnp.allclose(cov_fast, cov_full, rtol=1e-10, atol=1e-14)


def test_full_covariance_dispatch_via_bandpower_covariance(
        per_ell_signal_models, two_chan_instrument):
    """bandpower_covariance() autodetects BPWF mode and routes to full path."""
    _sm_fast, sm_bpwf = per_ell_signal_models
    params = flatten_params(FIDUCIAL, sm_bpwf.parameter_names)
    cov_dispatched = bandpower_covariance(sm_bpwf, two_chan_instrument, params)
    cov_full = bandpower_covariance_full(sm_bpwf, two_chan_instrument, params)
    assert jnp.allclose(cov_dispatched, cov_full, rtol=1e-12)


def test_full_covariance_symmetric_and_psd(two_chan_instrument):
    """Full covariance from overlapping Gaussian BPWFs is symmetric and PSD."""
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    centers = [40.0, 70.0, 110.0, 160.0]
    W = np.array([
        np.exp(-(ells_in - c) ** 2 / (2.0 * 15.0 ** 2)) for c in centers
    ])
    W /= W.sum(axis=1, keepdims=True)

    sm = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL, sm.parameter_names)
    cov = bandpower_covariance_full(sm, two_chan_instrument, params)

    n_data = sm.n_data
    assert cov.shape == (n_data, n_data)
    assert jnp.allclose(cov, cov.T, rtol=1e-8)
    # PSD: smallest eigenvalue > -tol × largest.
    eigvals = jnp.linalg.eigvalsh(cov)
    assert float(eigvals.min()) > -1e-10 * float(eigvals.max())


def test_full_covariance_overlapping_bins_couple(two_chan_instrument):
    """Overlapping Gaussian BPWFs produce non-zero off-diagonal bin coupling."""
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    centers = [60.0, 80.0, 110.0]   # tightly packed → overlap is non-trivial
    W = np.array([
        np.exp(-(ells_in - c) ** 2 / (2.0 * 25.0 ** 2)) for c in centers
    ])
    W /= W.sum(axis=1, keepdims=True)

    sm = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL, sm.parameter_names)
    cov = bandpower_covariance_full(sm, two_chan_instrument, params)

    n_bins = sm.n_bins  # 3
    # Auto (0,0) sub-block of the covariance: spectrum index 0, 0..n_bins.
    auto_block = cov[:n_bins, :n_bins]
    # Diagonal entries are positive variances.
    assert jnp.all(jnp.diag(auto_block) > 0)
    # Off-diagonal entries (different bins) are non-zero — bins couple.
    off_diag = auto_block - jnp.diag(jnp.diag(auto_block))
    assert float(jnp.max(jnp.abs(off_diag))) > 0.01 * float(
        jnp.max(jnp.diag(auto_block))), \
        "Overlapping BPWFs should produce non-trivial bin-bin coupling"


def test_full_covariance_from_noise_matches_instrument_path(
        two_chan_instrument):
    """*_full_from_noise with analytic noise matches *_full at the same fiducial."""
    from augr.instrument import noise_nl

    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.array([
        np.exp(-(ells_in - c) ** 2 / (2.0 * 15.0 ** 2))
        for c in [50.0, 100.0, 150.0]
    ])
    W /= W.sum(axis=1, keepdims=True)

    sm = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL, sm.parameter_names)

    cov_inst = bandpower_covariance_full(sm, two_chan_instrument, params)

    # Reconstruct the analytic per-channel noise on sm.ells.
    nl = jnp.array([
        noise_nl(ch, sm.ells,
                 two_chan_instrument.mission_duration_years,
                 two_chan_instrument.f_sky)
        for ch in two_chan_instrument.channels
    ])
    cov_ext = bandpower_covariance_full_from_noise(
        sm, nl, two_chan_instrument.f_sky, params)
    assert jnp.allclose(cov_inst, cov_ext, rtol=1e-10, atol=1e-14)


def test_blocks_path_raises_on_bpwf(two_chan_instrument):
    """The block path raises NotImplementedError for measured BPWFs."""
    ell_min, ell_max = 20, 100
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.eye(ells_in.size)
    sm = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL, sm.parameter_names)
    with pytest.raises(NotImplementedError, match="measured BPWFs"):
        bandpower_covariance_blocks(sm, two_chan_instrument, params)
    n_chan = len(two_chan_instrument.channels)
    nl = jnp.ones((n_chan, sm.ells.shape[0])) * 1e-5
    with pytest.raises(NotImplementedError, match="measured BPWFs"):
        bandpower_covariance_blocks_from_noise(
            sm, nl, two_chan_instrument.f_sky, params)


# -----------------------------------------------------------------------
# Per-spectrum BPWFs (Phase 2)
# -----------------------------------------------------------------------

def _gaussian_bpwf_rows(centers, sigmas, ells):
    rows = []
    for c, s in zip(centers, sigmas, strict=True):
        row = np.exp(-(ells - c) ** 2 / (2.0 * s ** 2))
        rows.append(row / row.sum())
    return np.array(rows)


def test_per_spec_covariance_identical_rows_matches_shared(two_chan_instrument):
    """Per-spectrum dict with all rows equal == shared-BPWF covariance.

    Bit-for-bit equality validates that the new 3-D Knox einsum reduces
    correctly to the Phase 1 2-D path when every cross-spectrum carries
    the same BPWF.
    """
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = _gaussian_bpwf_rows([50.0, 100.0, 150.0],
                             [15.0, 15.0, 15.0], ells_in)

    sm_shared = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    bpwf_dict = {p: W for p in sm_shared.freq_pairs}
    sm_per = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=bpwf_dict, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL, sm_shared.parameter_names)

    cov_shared = bandpower_covariance_full(sm_shared, two_chan_instrument,
                                            params)
    cov_per = bandpower_covariance_full(sm_per, two_chan_instrument,
                                         params)
    np.testing.assert_allclose(np.asarray(cov_per),
                                np.asarray(cov_shared),
                                rtol=1e-10, atol=1e-14)


def test_per_spec_covariance_uses_per_pair_window(two_chan_instrument):
    """Distinct per-pair BPWFs feed into the covariance per pair.

    Construction: keep two of the three cross-spectrum BPWFs identical
    to a baseline ``W0``; replace the third pair's BPWF with a scaled
    version of ``W0``. The covariance auto-block for the modified pair
    must change while the auto-block for an unchanged pair must not.
    """
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W0 = _gaussian_bpwf_rows([50.0, 100.0, 150.0],
                              [15.0, 15.0, 15.0], ells_in)

    sm_probe = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W0, bandpower_window_ells=ells_in,
    )
    pairs = sm_probe.freq_pairs   # [(0,0), (0,1), (1,1)]

    target_pair = (1, 1)
    untouched_pair = (0, 0)
    bpwf_a = {p: W0 for p in pairs}
    bpwf_b = {p: W0 for p in pairs}
    # Boost the (1, 1) BPWF amplitude in run B; covariance scales as W^4
    # via the einsum (W on row, W on col, both squared in auto Knox).
    bpwf_b[target_pair] = 2.0 * W0

    sm_a = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=bpwf_a, bandpower_window_ells=ells_in,
    )
    sm_b = SignalModel(
        two_chan_instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=bpwf_b, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL, sm_a.parameter_names)

    cov_a = bandpower_covariance_full(sm_a, two_chan_instrument, params)
    cov_b = bandpower_covariance_full(sm_b, two_chan_instrument, params)

    n_bins = sm_a.n_bins
    s_target = pairs.index(target_pair)
    s_other = pairs.index(untouched_pair)

    target_block_a = cov_a[s_target * n_bins:(s_target + 1) * n_bins,
                            s_target * n_bins:(s_target + 1) * n_bins]
    target_block_b = cov_b[s_target * n_bins:(s_target + 1) * n_bins,
                            s_target * n_bins:(s_target + 1) * n_bins]
    other_block_a = cov_a[s_other * n_bins:(s_other + 1) * n_bins,
                           s_other * n_bins:(s_other + 1) * n_bins]
    other_block_b = cov_b[s_other * n_bins:(s_other + 1) * n_bins,
                           s_other * n_bins:(s_other + 1) * n_bins]

    # The (1,1) auto-block must move (atol=0 so the ratio test isn't
    # masked by the variances being O(1e-12)); the (0,0) auto-block
    # must not.
    assert not jnp.allclose(target_block_a, target_block_b,
                             rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(np.asarray(other_block_a),
                                np.asarray(other_block_b),
                                rtol=1e-12, atol=1e-14)
    # Direction sanity-check: doubling W on (1,1) auto multiplies its
    # variance by 2 × 2 = 4. The Knox einsum has two W factors (one
    # row, one col); M itself carries no W, so the scaling is W², not
    # W⁴.
    np.testing.assert_allclose(np.asarray(jnp.diag(target_block_b)),
                                4.0 * np.asarray(
                                    jnp.diag(target_block_a)),
                                rtol=1e-10)
