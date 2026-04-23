"""Tests for signal.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.signal import (
    SignalModel,
    flatten_params,
    unflatten_params,
    _make_bin_edges,
    _build_bin_matrix,
)
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.foregrounds import GaussianForegroundModel, NullForegroundModel
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
    W, centers = _build_bin_matrix(ells, edges, "tophat")
    row_sums = jnp.sum(W, axis=1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-6)


def test_bin_matrix_gaussian_rows_normalized():
    """Gaussian window rows also sum to 1."""
    ells = np.arange(20, 101, dtype=float)
    edges = _make_bin_edges(20, 100, 30, 35)
    W, centers = _build_bin_matrix(ells, edges, "gaussian")
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
    expected = ["r", "A_lens"] + list(fg_model.parameter_names) + ["A_res"]
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
