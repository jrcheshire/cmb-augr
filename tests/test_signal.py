"""Tests for signal.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from augr.foregrounds import GaussianForegroundModel, NullForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.signal import (
    SignalModel,
    _build_bin_matrix,
    _make_bin_edges,
    flatten_params,
    unflatten_params,
)
from augr.spectra import CMBSpectra

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_instrument():
    """A 3-channel instrument for testing."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    channels = (
        Channel(nu_ghz=90.0, n_detectors=500,
                net_per_detector=400.0, beam_fwhm_arcmin=30.0,
                efficiency=eff),
        Channel(nu_ghz=150.0, n_detectors=1000,
                net_per_detector=300.0, beam_fwhm_arcmin=20.0,
                efficiency=eff),
        Channel(nu_ghz=220.0, n_detectors=500,
                net_per_detector=500.0, beam_fwhm_arcmin=15.0,
                efficiency=eff),
    )
    return Instrument(channels=channels, mission_duration_years=5.0, f_sky=0.7)


@pytest.fixture(scope="module")
def fg_model():
    return GaussianForegroundModel()


@pytest.fixture(scope="module")
def cmb_spectra():
    return CMBSpectra()


@pytest.fixture(scope="module")
def signal_model(simple_instrument, fg_model, cmb_spectra):
    """SignalModel with coarse binning for fast tests."""
    return SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200, delta_ell=35,
        ell_per_bin_below=30, window="tophat",
    )


# Fiducial parameter dict
FIDUCIAL_DICT = {
    "r": 0.0,
    "A_lens": 1.0,
    "A_dust": 4.7,
    "beta_dust": 1.6,
    "alpha_dust": -0.58,
    "T_dust": 19.6,
    "A_sync": 1.5,
    "beta_sync": -3.1,
    "alpha_sync": -0.6,
    "epsilon": 0.0,
    "Delta_dust": 0.0,
}


# -----------------------------------------------------------------------
# Binning
# -----------------------------------------------------------------------

def test_bin_edges_per_ell():
    """Per-ℓ bins for ℓ < 30: each bin is a single multipole."""
    edges = _make_bin_edges(2, 100, ell_per_bin_below=30, delta_ell=35)
    # First 28 bins should be (2,2), (3,3), ..., (29,29)
    for i, (lo, hi) in enumerate(edges[:28]):
        assert lo == hi == i + 2


def test_bin_edges_uniform_above():
    """Uniform Δℓ=35 bins above the per-ℓ threshold."""
    edges = _make_bin_edges(2, 100, ell_per_bin_below=30, delta_ell=35)
    # Bins above 30: (30, 64), (65, 99), (100, 100)
    uniform = edges[28:]
    assert uniform[0] == (30, 64)
    assert uniform[1] == (65, 99)
    assert uniform[2] == (100, 100)  # last bin truncated at ell_max


def test_bin_edges_custom():
    """Custom bin edges override auto-binning."""
    custom = np.array([20, 50, 100, 200])
    edges = _make_bin_edges(20, 200, 30, 35, ell_bins=custom)
    assert edges == [(20, 49), (50, 99), (100, 199)]


def test_bin_matrix_rows_normalized():
    """Each row of the bin matrix sums to 1."""
    ells = np.arange(20, 101, dtype=float)
    edges = _make_bin_edges(20, 100, 30, 35)
    W, _centers = _build_bin_matrix(ells, edges, "tophat")
    row_sums = jnp.sum(W, axis=1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-6)


def test_bin_matrix_gaussian_rows_normalized():
    """Gaussian window rows also sum to 1."""
    ells = np.arange(20, 101, dtype=float)
    edges = _make_bin_edges(20, 100, 30, 35)
    W, _centers = _build_bin_matrix(ells, edges, "gaussian")
    row_sums = jnp.sum(W, axis=1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-6)


def test_per_ell_bin_is_delta():
    """Per-ℓ bin picks out exactly one multipole."""
    ells = np.arange(2, 50, dtype=float)
    edges = _make_bin_edges(2, 49, 30, 35)
    W, _ = _build_bin_matrix(ells, edges, "tophat")
    # First bin (ℓ=2): row should be [1, 0, 0, ...]
    assert float(W[0, 0]) == 1.0
    assert float(jnp.sum(W[0, 1:])) == 0.0


# -----------------------------------------------------------------------
# Flatten / unflatten
# -----------------------------------------------------------------------

def test_flatten_unflatten_roundtrip(signal_model):
    """flatten → unflatten is identity."""
    names = signal_model.parameter_names
    arr = flatten_params(FIDUCIAL_DICT, names)
    result = unflatten_params(arr, names)
    for n in names:
        assert abs(result[n] - FIDUCIAL_DICT[n]) < 1e-6


def test_flatten_order(signal_model):
    """Flat array has r first, A_lens second."""
    names = signal_model.parameter_names
    arr = flatten_params(FIDUCIAL_DICT, names)
    assert float(arr[0]) == 0.0   # r
    assert float(arr[1]) == 1.0   # A_lens


# -----------------------------------------------------------------------
# Signal model structure
# -----------------------------------------------------------------------

def test_parameter_names(signal_model, fg_model):
    """Parameter names start with r, A_lens, then foreground params."""
    names = signal_model.parameter_names
    assert names[0] == "r"
    assert names[1] == "A_lens"
    assert names[2:] == fg_model.parameter_names


def test_n_spectra_3chan(signal_model):
    """3 channels → 6 unique cross-spectra."""
    # (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    assert signal_model.n_spectra == 6


def test_data_vector_length(signal_model):
    """Data vector length = n_spectra × n_bins."""
    names = signal_model.parameter_names
    params = flatten_params(FIDUCIAL_DICT, names)
    mu = signal_model.data_vector(params)
    assert mu.shape == (signal_model.n_data,)
    assert signal_model.n_data == signal_model.n_spectra * signal_model.n_bins


def test_data_vector_positive(signal_model):
    """All bandpowers are positive at fiducial (r=0, with lensing + foregrounds)."""
    names = signal_model.parameter_names
    params = flatten_params(FIDUCIAL_DICT, names)
    mu = signal_model.data_vector(params)
    assert jnp.all(mu > 0)


# -----------------------------------------------------------------------
# Physics: CMB enters identically at all freq pairs
# -----------------------------------------------------------------------

def test_cmb_same_all_pairs(simple_instrument, cmb_spectra):
    """With zero foregrounds, all cross-spectra are identical (pure CMB)."""
    # Use a foreground model with all amplitudes zero
    fg = GaussianForegroundModel()
    model = SignalModel(simple_instrument, fg, cmb_spectra,
                        ell_min=20, ell_max=200, delta_ell=35,
                        ell_per_bin_below=30)
    params_dict = dict(FIDUCIAL_DICT)
    params_dict["A_dust"] = 0.0
    params_dict["A_sync"] = 0.0
    params = flatten_params(params_dict, model.parameter_names)
    mu = model.data_vector(params)

    # Extract each spectrum and compare
    n_bins = model.n_bins
    first_spec = mu[:n_bins]
    for s in range(1, model.n_spectra):
        this_spec = mu[s * n_bins: (s + 1) * n_bins]
        assert jnp.allclose(first_spec, this_spec, rtol=1e-5), \
            f"Spectrum {s} differs from spectrum 0 with zero foregrounds"


# -----------------------------------------------------------------------
# Jacobian
# -----------------------------------------------------------------------

def test_jacobian_shape(signal_model):
    """Jacobian has shape (n_data, n_params)."""
    names = signal_model.parameter_names
    params = flatten_params(FIDUCIAL_DICT, names)
    J = signal_model.jacobian(params)
    assert J.shape == (signal_model.n_data, signal_model.n_params)


def test_jacobian_r_nonzero(signal_model):
    """Derivative w.r.t. r is nonzero (tensor spectrum contributes)."""
    names = signal_model.parameter_names
    params = flatten_params(FIDUCIAL_DICT, names)
    J = signal_model.jacobian(params)
    r_idx = names.index("r")
    dmu_dr = J[:, r_idx]
    # Tensor BB is nonzero, so derivative w.r.t. r should be nonzero
    assert jnp.any(jnp.abs(dmu_dr) > 1e-10)


def test_jacobian_r_same_all_pairs(signal_model):
    """∂μ/∂r is identical across all freq pairs (CMB is freq-independent)."""
    names = signal_model.parameter_names
    params = flatten_params(FIDUCIAL_DICT, names)
    J = signal_model.jacobian(params)
    r_idx = names.index("r")
    dmu_dr = J[:, r_idx]
    n_bins = signal_model.n_bins
    first = dmu_dr[:n_bins]
    for s in range(1, signal_model.n_spectra):
        this = dmu_dr[s * n_bins: (s + 1) * n_bins]
        assert jnp.allclose(first, this, rtol=1e-5)


def test_jacobian_matches_finite_diff(signal_model):
    """JAX Jacobian matches numerical central finite difference (cross-check)."""
    names = signal_model.parameter_names
    params = flatten_params(FIDUCIAL_DICT, names)
    J_auto = signal_model.jacobian(params)

    # Numerical finite difference for a few parameters
    eps = 1e-5
    for p_idx in [0, 1, 2, 4]:  # r, A_lens, A_dust, A_sync
        params_p = params.at[p_idx].add(eps)
        params_m = params.at[p_idx].add(-eps)
        dmu_fd = (signal_model.data_vector(params_p)
                  - signal_model.data_vector(params_m)) / (2 * eps)
        # Use generous tolerance: finite diff is approximate
        assert jnp.allclose(J_auto[:, p_idx], dmu_fd, rtol=1e-2, atol=1e-10), \
            f"Jacobian mismatch for param {names[p_idx]}"


# -----------------------------------------------------------------------
# Spectrum slice helper
# -----------------------------------------------------------------------

def test_spectrum_slice(signal_model):
    """spectrum_slice returns the correct range."""
    sl = signal_model.spectrum_slice(0, 1)
    assert sl.start == signal_model.n_bins  # second spectrum
    assert sl.stop == 2 * signal_model.n_bins


# -----------------------------------------------------------------------
# Null foreground model plumbed through SignalModel
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def null_signal_model(simple_instrument, cmb_spectra):
    """SignalModel with NullForegroundModel (post-CompSep-style use)."""
    return SignalModel(
        simple_instrument, NullForegroundModel(), cmb_spectra,
        ell_min=20, ell_max=200, delta_ell=35,
        ell_per_bin_below=30, window="tophat",
    )


def test_null_model_parameter_names(null_signal_model):
    """With NullForegroundModel, parameters are exactly [r, A_lens]."""
    assert null_signal_model.parameter_names == ["r", "A_lens"]


def test_null_model_fg_contribution_zero(null_signal_model,
                                         simple_instrument, cmb_spectra):
    """Data vector equals the CMB-only binned bandpower (foregrounds are zero)."""
    params = jnp.array([0.01, 1.0])  # r, A_lens
    mu = null_signal_model.data_vector(params)
    # CMB-only bandpower: all cross-spectra must be identical
    n_bins = null_signal_model.n_bins
    ref = mu[:n_bins]
    for s in range(1, null_signal_model.n_spectra):
        assert jnp.allclose(ref, mu[s * n_bins:(s + 1) * n_bins], rtol=1e-8)


# -----------------------------------------------------------------------
# Residual template (A_res) plumbing
# -----------------------------------------------------------------------

# Flat residual template, distinctive amplitude so it's easy to see.
RESIDUAL_TEMPLATE_AMPLITUDE = 1e-4     # uK^2


@pytest.fixture(scope="module")
def residual_template():
    """(ells, cl) pair for a flat residual-template spectrum."""
    ells = np.arange(2, 400, dtype=float)
    cl = np.full_like(ells, RESIDUAL_TEMPLATE_AMPLITUDE)
    return ells, cl


@pytest.fixture(scope="module")
def signal_model_with_template(simple_instrument, fg_model, cmb_spectra,
                               residual_template):
    """SignalModel with a residual-template amplitude A_res appended."""
    ells, cl = residual_template
    return SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200, delta_ell=35,
        ell_per_bin_below=30, window="tophat",
        residual_template_cl=cl, residual_template_ells=ells,
    )


def _fiducial_with_a_res(names, a_res_value):
    """Build a flat param array from FIDUCIAL_DICT extended with A_res."""
    extended = dict(FIDUCIAL_DICT)
    extended["A_res"] = a_res_value
    return flatten_params(extended, names)


def test_a_res_appended_to_parameter_names(signal_model_with_template,
                                           fg_model):
    """Residual template appends A_res after the fg params."""
    names = signal_model_with_template.parameter_names
    expected = ["r", "A_lens", *list(fg_model.parameter_names), "A_res"]
    assert names == expected


def test_fg_params_from_excludes_a_res(signal_model_with_template, fg_model):
    """fg_params_from returns exactly the fg block, not A_res at the tail."""
    names = signal_model_with_template.parameter_names
    params = _fiducial_with_a_res(names, 1.0)
    fg = signal_model_with_template.fg_params_from(params)
    assert fg.shape == (len(fg_model.parameter_names),)


def test_a_res_zero_matches_no_template(signal_model, signal_model_with_template):
    """A_res = 0 reproduces the no-template data vector exactly."""
    names_plain = signal_model.parameter_names
    names_tmpl = signal_model_with_template.parameter_names
    mu_plain = signal_model.data_vector(
        flatten_params(FIDUCIAL_DICT, names_plain))
    mu_tmpl = signal_model_with_template.data_vector(
        _fiducial_with_a_res(names_tmpl, 0.0))
    assert jnp.allclose(mu_plain, mu_tmpl, rtol=1e-10, atol=1e-14)


def test_a_res_affects_only_auto_spectra(signal_model_with_template):
    """Increasing A_res moves only the i==j (auto) bandpowers."""
    names = signal_model_with_template.parameter_names
    mu_0 = signal_model_with_template.data_vector(
        _fiducial_with_a_res(names, 0.0))
    mu_1 = signal_model_with_template.data_vector(
        _fiducial_with_a_res(names, 1.0))
    delta = mu_1 - mu_0

    n_bins = signal_model_with_template.n_bins
    for idx, (i_ch, j_ch) in enumerate(signal_model_with_template.freq_pairs):
        block = delta[idx * n_bins:(idx + 1) * n_bins]
        if i_ch == j_ch:
            # Auto: template was added. Shape matches the flat amplitude.
            expected = RESIDUAL_TEMPLATE_AMPLITUDE * jnp.ones(n_bins)
            assert jnp.allclose(block, expected, rtol=1e-8)
        else:
            # Cross: residual absent, block should not have moved.
            assert jnp.allclose(block, 0.0, atol=1e-14)


def test_jacobian_a_res_column(signal_model_with_template):
    """dD/dA_res equals the binned template on auto blocks, zero elsewhere."""
    names = signal_model_with_template.parameter_names
    params = _fiducial_with_a_res(names, 1.0)
    J = signal_model_with_template.jacobian(params)
    a_res_idx = names.index("A_res")
    col = J[:, a_res_idx]

    n_bins = signal_model_with_template.n_bins
    for idx, (i_ch, j_ch) in enumerate(signal_model_with_template.freq_pairs):
        block = col[idx * n_bins:(idx + 1) * n_bins]
        if i_ch == j_ch:
            expected = RESIDUAL_TEMPLATE_AMPLITUDE * jnp.ones(n_bins)
            assert jnp.allclose(block, expected, rtol=1e-8)
        else:
            assert jnp.allclose(block, 0.0, atol=1e-14)


def test_residual_template_requires_ells(simple_instrument, fg_model, cmb_spectra):
    """Passing residual_template_cl without ells should error loudly."""
    cl = np.full(100, 1e-4)
    with pytest.raises(ValueError, match="residual_template_ells"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
            residual_template_cl=cl,
        )


def test_residual_template_rejects_empty(simple_instrument, fg_model, cmb_spectra):
    """An empty residual_template_cl / ells should fail loudly at init,
    not silently register a zero-jacobian A_res column that explodes
    inside the Fisher solve."""
    with pytest.raises(ValueError, match="length"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
            residual_template_cl=np.array([]),
            residual_template_ells=np.array([]),
        )


def test_residual_template_rejects_mismatched_shapes(
        simple_instrument, fg_model, cmb_spectra):
    """Mismatched ells/cl lengths fail at init with a clear message."""
    with pytest.raises(ValueError, match="shape"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
            residual_template_cl=np.ones(10),
            residual_template_ells=np.arange(20),
        )


def test_residual_template_extrapolates_nearest_neighbour(
        simple_instrument, fg_model, cmb_spectra):
    """Template provided with ell_min > SignalModel.ell_min must extrapolate
    to fp[0] (not zero) at low ells.

    BROOM bandpowers start at the first bin center (~ell=4 for delta_ell=5).
    Zero-extrapolation at the reionization bump would silently null the
    A_res constraint where sigma(r) is most sensitive for a space mission.
    """
    # Template provided only at ell >= 50 with a distinctive amplitude
    ells_in = np.arange(50, 200, dtype=float)
    amp = 7e-4
    cl_in = np.full_like(ells_in, amp)

    sm = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=2, ell_max=180, delta_ell=5, ell_per_bin_below=30,
        residual_template_cl=cl_in, residual_template_ells=ells_in,
    )
    # Internal interpolated template on SignalModel's ell grid
    tmpl = np.asarray(sm._residual_template_cl)
    # Below the input range: nearest-neighbour = fp[0] = amp
    assert np.all(tmpl == pytest.approx(amp, rel=1e-12)), \
        "Template should extrapolate flat (nearest-neighbour), not zero"


def test_delensed_bb_range_must_cover_signal_model(
        simple_instrument, fg_model, cmb_spectra):
    """delensed_bb_ells that does not span [ell_min, ell_max] must raise.

    Pre-fix this silently zero-extrapolated, nulling the reionization
    bump or the high-ell tail where sigma(r) is most sensitive.
    """
    # Input covers ell=20..200 but SignalModel wants ell=2..200.
    ells_in = np.arange(20, 201, dtype=float)
    cl_in = np.full_like(ells_in, 1e-6)
    with pytest.raises(ValueError, match="must cover the SignalModel ell range"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=2, ell_max=200, delta_ell=35, ell_per_bin_below=30,
            delensed_bb=cl_in, delensed_bb_ells=ells_in,
        )


def test_delensed_bb_requires_ells(simple_instrument, fg_model, cmb_spectra):
    """Passing delensed_bb without delensed_bb_ells should error loudly."""
    cl = np.full(200, 1e-6)
    with pytest.raises(ValueError, match="delensed_bb_ells"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=2, ell_max=200, delta_ell=35, ell_per_bin_below=30,
            delensed_bb=cl,
        )


# -----------------------------------------------------------------------
# Measured bandpower window functions
# -----------------------------------------------------------------------

def _gaussian_bpwf(centers, sigmas, ell_grid):
    """Synthetic overlapping Gaussian BPWFs, row-normalised to sum to 1."""
    rows = []
    for c, s in zip(centers, sigmas):
        row = np.exp(-(ell_grid - c) ** 2 / (2.0 * s ** 2))
        rows.append(row / row.sum())
    return np.array(rows)


def test_measured_bpwf_basic(simple_instrument, fg_model, cmb_spectra):
    """SignalModel accepts a (n_bins, n_ells) BPWF and exposes the right shape."""
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    centers = [40.0, 80.0, 130.0]
    W = _gaussian_bpwf(centers, [10.0, 10.0, 12.0], ells_in)

    sm = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )

    assert sm.has_measured_bpwf
    assert sm.bin_edges is None
    assert sm.n_bins == 3
    assert sm.bin_matrix.shape == (3, ells_in.size)
    # Bin centres reflect the supplied Gaussian centres to within the grid step.
    for c_target, c_actual in zip(centers, sm.bin_centers):
        assert abs(float(c_actual) - c_target) < 1.0


def test_measured_bpwf_xor_kwargs(simple_instrument, fg_model, cmb_spectra):
    """Supplying only one of bandpower_window / bandpower_window_ells errors."""
    W = np.eye(10)
    with pytest.raises(ValueError, match="must be supplied together"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=29, bandpower_window=W,
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=29,
            bandpower_window_ells=np.arange(20, 30, dtype=float),
        )


def test_measured_bpwf_range_check(simple_instrument, fg_model, cmb_spectra):
    """BPWF whose ell range does not span [ell_min, ell_max] must raise."""
    ells_in = np.arange(50, 150, dtype=float)
    W = np.eye(ells_in.size)
    with pytest.raises(ValueError, match="must cover the SignalModel ell range"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=200,
            bandpower_window=W, bandpower_window_ells=ells_in,
        )


def test_measured_bpwf_shape_validation(simple_instrument, fg_model, cmb_spectra):
    """Mismatched ells/W or wrong-rank inputs raise at construction."""
    ells_in = np.arange(20, 201, dtype=float)
    W_wrong_cols = np.eye(50)  # n_ells = 50, doesn't match ells_in
    with pytest.raises(ValueError, match="ell columns"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=200,
            bandpower_window=W_wrong_cols, bandpower_window_ells=ells_in,
        )

    W_one_d = np.zeros(ells_in.size)  # 1-D — rejected
    with pytest.raises(ValueError, match="must be 2-D"):
        SignalModel(
            simple_instrument, fg_model, cmb_spectra,
            ell_min=20, ell_max=200,
            bandpower_window=W_one_d, bandpower_window_ells=ells_in,
        )


def test_measured_bpwf_normalization_preserved(simple_instrument, fg_model,
                                                cmb_spectra):
    """User-supplied BPWFs are NOT auto-normalised (rows do not get
    forced to sum to 1).

    BICEP/Keck-style pipelines release windows already-normalised so
    that <C_b> = Σ W_bℓ C_ℓ matches the bandpower estimator. Forcing
    row-sum=1 would corrupt that calibration.
    """
    ells_in = np.arange(20, 201, dtype=float)
    # Two Gaussian BPWFs scaled by 2 so the rows sum to 2, not 1.
    rows = _gaussian_bpwf([60.0, 120.0], [15.0, 15.0], ells_in)
    W = 2.0 * rows

    sm = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    rowsums = np.asarray(sm.bin_matrix).sum(axis=1)
    assert np.allclose(rowsums, 2.0, rtol=1e-8), \
        "bin_matrix rows should not be re-normalised"


def test_measured_bpwf_data_vector_matches_W_at_Cl(
        simple_instrument, cmb_spectra):
    """With NullForegroundModel + Gaussian BPWFs, data_vector reproduces
    W @ C_ℓ^CMB exactly on every cross-spectrum (CMB is freq-independent)."""
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = _gaussian_bpwf([50.0, 100.0, 150.0], [12.0, 12.0, 12.0], ells_in)

    sm = SignalModel(
        simple_instrument, NullForegroundModel(), cmb_spectra,
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    params = jnp.array([0.01, 1.0])  # r, A_lens (no foreground params)
    mu = sm.data_vector(params)

    # All cross-spectra must be identical (pure CMB).
    n_bins = sm.n_bins
    ref = mu[:n_bins]
    for s in range(1, sm.n_spectra):
        assert jnp.allclose(ref, mu[s * n_bins:(s + 1) * n_bins], rtol=1e-8)
    # And the reference matches W @ C_ℓ^BB on the SignalModel ell grid.
    cl = cmb_spectra.cl_bb(sm.ells, 0.01, 1.0)
    expected = jnp.asarray(W) @ cl
    assert jnp.allclose(ref, expected, rtol=1e-8)


def test_measured_bpwf_per_ell_identity_matches_default(
        simple_instrument, fg_model, cmb_spectra):
    """An identity BPWF on per-ℓ bins matches a delta_ell=1 default model."""
    ell_min, ell_max = 20, 60
    sm_default = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=ell_min, ell_max=ell_max,
        delta_ell=1, ell_per_bin_below=ell_min,
    )

    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.eye(ells_in.size)
    sm_bpwf = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )

    params = flatten_params(FIDUCIAL_DICT, sm_default.parameter_names)
    mu_default = sm_default.data_vector(params)
    mu_bpwf = sm_bpwf.data_vector(params)
    assert jnp.allclose(mu_default, mu_bpwf, rtol=1e-10, atol=1e-14)


def test_measured_bpwf_jacobian_shape(simple_instrument, fg_model, cmb_spectra):
    """Jacobian of a BPWF SignalModel has shape (n_data, n_params)."""
    ells_in = np.arange(20, 201, dtype=float)
    W = _gaussian_bpwf([60.0, 120.0], [15.0, 15.0], ells_in)
    sm = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    params = flatten_params(FIDUCIAL_DICT, sm.parameter_names)
    J = sm.jacobian(params)
    assert J.shape == (sm.n_data, sm.n_params)
    # ∂μ/∂r is nonzero (tensor BB contributes through W).
    r_idx = sm.parameter_names.index("r")
    assert jnp.any(jnp.abs(J[:, r_idx]) > 1e-12)


# -----------------------------------------------------------------------
# Per-spectrum BPWFs (Phase 2)
# -----------------------------------------------------------------------

def _per_spec_bpwf_dict(freq_pairs, ells, scale=None):
    """Build a {(i,j): W} dict of distinct Gaussian BPWFs per pair.

    ``scale`` lets a caller bias different pairs by a known multiplier
    so tests can detect that the per-pair window was actually used.
    """
    out = {}
    for s, (i, j) in enumerate(freq_pairs):
        # Slightly different center / width per pair to make the rows
        # distinguishable.
        centers = [40.0 + 5.0 * s, 100.0 + 3.0 * s]
        sigmas = [12.0 + 0.5 * s, 12.0 + 0.5 * s]
        rows = _gaussian_bpwf(centers, sigmas, ells)
        if scale is not None:
            rows = scale[s] * rows
        out[(i, j)] = rows
    return out


def test_per_spectrum_bpwf_basic(simple_instrument, fg_model, cmb_spectra):
    """Dict input populates the 3-D tensor and the per-spectrum flags."""
    ells_in = np.arange(20, 201, dtype=float)
    sm_probe = SignalModel(   # only used to read freq_pairs
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
    )
    bpwf_dict = _per_spec_bpwf_dict(sm_probe.freq_pairs, ells_in)
    sm = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200,
        bandpower_window=bpwf_dict, bandpower_window_ells=ells_in,
    )
    assert sm.has_measured_bpwf
    assert sm.is_per_spectrum_bpwf
    assert sm.bin_matrix_per_spectrum.shape == (
        len(sm.freq_pairs), 2, ells_in.size)
    # Per-pair accessor returns the right slice with canonicalisation.
    for s, (i, j) in enumerate(sm.freq_pairs):
        np.testing.assert_allclose(
            np.asarray(sm.bandpower_window_for(i, j)),
            np.asarray(sm.bin_matrix_per_spectrum[s]), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(sm.bandpower_window_for(j, i)),
            np.asarray(sm.bin_matrix_per_spectrum[s]), rtol=1e-12)


def test_per_spectrum_bin_matrix_raises(simple_instrument, fg_model,
                                         cmb_spectra):
    """The 2-D bin_matrix property is undefined in per-spectrum mode."""
    sm_probe = SignalModel(simple_instrument, fg_model, cmb_spectra,
                           ell_min=20, ell_max=200,
                           delta_ell=35, ell_per_bin_below=30)
    ells_in = np.arange(20, 201, dtype=float)
    sm = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200,
        bandpower_window=_per_spec_bpwf_dict(sm_probe.freq_pairs, ells_in),
        bandpower_window_ells=ells_in,
    )
    with pytest.raises(ValueError, match="per-spectrum BPWF mode"):
        _ = sm.bin_matrix


def test_per_spectrum_key_canonicalisation(simple_instrument, fg_model,
                                            cmb_spectra):
    """Either (i, j) or (j, i) ordering is accepted; results agree."""
    sm_probe = SignalModel(simple_instrument, fg_model, cmb_spectra,
                           ell_min=20, ell_max=200,
                           delta_ell=35, ell_per_bin_below=30)
    ells_in = np.arange(20, 201, dtype=float)
    canonical = _per_spec_bpwf_dict(sm_probe.freq_pairs, ells_in)
    # Flip every (i, j) with i != j to (j, i) ordering.
    swapped = {(j, i) if i != j else (i, j): W
               for (i, j), W in canonical.items()}
    sm_a = SignalModel(simple_instrument, fg_model, cmb_spectra,
                       ell_min=20, ell_max=200,
                       bandpower_window=canonical,
                       bandpower_window_ells=ells_in)
    sm_b = SignalModel(simple_instrument, fg_model, cmb_spectra,
                       ell_min=20, ell_max=200,
                       bandpower_window=swapped,
                       bandpower_window_ells=ells_in)
    np.testing.assert_allclose(
        np.asarray(sm_a.bin_matrix_per_spectrum),
        np.asarray(sm_b.bin_matrix_per_spectrum), rtol=1e-12)


def test_per_spectrum_validation_errors(simple_instrument, fg_model,
                                         cmb_spectra):
    """Missing / extra / malformed dict entries fail loudly."""
    sm_probe = SignalModel(simple_instrument, fg_model, cmb_spectra,
                           ell_min=20, ell_max=200,
                           delta_ell=35, ell_per_bin_below=30)
    ells_in = np.arange(20, 201, dtype=float)
    canonical = _per_spec_bpwf_dict(sm_probe.freq_pairs, ells_in)

    missing_one = {k: v for k, v in canonical.items()
                   if k != sm_probe.freq_pairs[0]}
    with pytest.raises(ValueError, match="missing entries"):
        SignalModel(simple_instrument, fg_model, cmb_spectra,
                    ell_min=20, ell_max=200,
                    bandpower_window=missing_one,
                    bandpower_window_ells=ells_in)

    extra = dict(canonical)
    extra[(99, 99)] = canonical[sm_probe.freq_pairs[0]]
    with pytest.raises(ValueError, match="unknown cross-spectra"):
        SignalModel(simple_instrument, fg_model, cmb_spectra,
                    ell_min=20, ell_max=200,
                    bandpower_window=extra,
                    bandpower_window_ells=ells_in)

    bad_key = dict(canonical)
    bad_key["not_a_tuple"] = canonical[sm_probe.freq_pairs[0]]
    with pytest.raises(ValueError, match="2-tuples of channel indices"):
        SignalModel(simple_instrument, fg_model, cmb_spectra,
                    ell_min=20, ell_max=200,
                    bandpower_window=bad_key,
                    bandpower_window_ells=ells_in)


def test_per_spectrum_inconsistent_n_bins(simple_instrument, fg_model,
                                           cmb_spectra):
    """Mixing (n_bins=2, n_bins=3) entries fails at construction."""
    sm_probe = SignalModel(simple_instrument, fg_model, cmb_spectra,
                           ell_min=20, ell_max=200,
                           delta_ell=35, ell_per_bin_below=30)
    ells_in = np.arange(20, 201, dtype=float)
    base = _per_spec_bpwf_dict(sm_probe.freq_pairs, ells_in)
    # Replace the first pair's BPWF with a 3-bin version.
    pair0 = sm_probe.freq_pairs[0]
    base[pair0] = _gaussian_bpwf([40.0, 100.0, 160.0],
                                  [12.0, 12.0, 12.0], ells_in)
    with pytest.raises(ValueError, match="inconsistent n_bins"):
        SignalModel(simple_instrument, fg_model, cmb_spectra,
                    ell_min=20, ell_max=200,
                    bandpower_window=base,
                    bandpower_window_ells=ells_in)


def test_per_spectrum_data_vector_differentiates_pairs(
        simple_instrument, cmb_spectra):
    """Distinct per-pair BPWFs produce distinct bandpowers per pair.

    With NullForegroundModel + identical CMB across pairs, the only
    thing varying across the data vector is the per-pair BPWF; if the
    SignalModel ignored ``s`` and applied a shared W, all pairs would
    still come out identical. They must not. We use a single base
    Gaussian shape and a per-pair scale factor, which makes the
    bandpowers in pair s simply ``scales[s]`` times the bandpowers in
    pair 0 -- a quantitative check that the right window was applied.
    """
    sm_probe = SignalModel(simple_instrument, NullForegroundModel(),
                           cmb_spectra, ell_min=20, ell_max=200,
                           delta_ell=35, ell_per_bin_below=30)
    ells_in = np.arange(20, 201, dtype=float)
    base_W = _gaussian_bpwf([55.0, 110.0], [15.0, 15.0], ells_in)
    scales = [1.0 + 0.3 * s for s in range(len(sm_probe.freq_pairs))]
    bpwf_dict = {p: scales[s] * base_W
                 for s, p in enumerate(sm_probe.freq_pairs)}
    sm = SignalModel(simple_instrument, NullForegroundModel(),
                     cmb_spectra, ell_min=20, ell_max=200,
                     bandpower_window=bpwf_dict,
                     bandpower_window_ells=ells_in)
    params = jnp.array([0.01, 1.0])  # r, A_lens
    mu = sm.data_vector(params)
    n_bins = sm.n_bins
    base = mu[:n_bins] / scales[0]
    for s in range(1, sm.n_spectra):
        rec = mu[s * n_bins:(s + 1) * n_bins] / scales[s]
        np.testing.assert_allclose(np.asarray(rec), np.asarray(base),
                                    rtol=1e-10)


def test_per_spectrum_identical_rows_matches_shared(
        simple_instrument, fg_model, cmb_spectra):
    """A per-spectrum dict with all rows equal reproduces the shared path.

    Both the public data vector and the underlying 3-D tensor must match
    a Phase 1 shared-BPWF SignalModel built from the same window.
    """
    ells_in = np.arange(20, 201, dtype=float)
    W = _gaussian_bpwf([55.0, 110.0], [15.0, 15.0], ells_in)
    sm_shared = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    bpwf_dict = {p: W for p in sm_shared.freq_pairs}
    sm_per = SignalModel(
        simple_instrument, fg_model, cmb_spectra,
        ell_min=20, ell_max=200,
        bandpower_window=bpwf_dict, bandpower_window_ells=ells_in,
    )
    assert sm_shared.is_per_spectrum_bpwf is False
    assert sm_per.is_per_spectrum_bpwf is True

    params = flatten_params(FIDUCIAL_DICT, sm_shared.parameter_names)
    mu_shared = sm_shared.data_vector(params)
    mu_per = sm_per.data_vector(params)
    np.testing.assert_allclose(np.asarray(mu_shared),
                                np.asarray(mu_per),
                                rtol=1e-10, atol=1e-14)
