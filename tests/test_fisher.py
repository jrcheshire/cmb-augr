"""Tests for fisher.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency, noise_nl
from augr.signal import SignalModel
from augr.spectra import CMBSpectra

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

FIDUCIAL = {
    "r": 0.0, "A_lens": 1.0,
    "A_dust": 4.7, "beta_dust": 1.6, "alpha_dust": -0.58, "T_dust": 19.6,
    "A_sync": 1.5, "beta_sync": -3.1, "alpha_sync": -0.6,
    "epsilon": 0.0, "Delta_dust": 0.0,
}


@pytest.fixture(scope="module")
def instrument():
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    return Instrument(channels=(
        Channel(90.0,  500, 400.0, 30.0, efficiency=eff),
        Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
        Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)


@pytest.fixture(scope="module")
def signal_model(instrument):
    return SignalModel(
        instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
    )


@pytest.fixture(scope="module")
def fisher(signal_model, instrument):
    return FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )


# -----------------------------------------------------------------------
# Fisher matrix structure
# -----------------------------------------------------------------------

def test_fisher_matrix_shape(fisher):
    """Fisher matrix shape = (n_free, n_free). T_dust is fixed → 10 free params."""
    F = fisher.fisher_matrix
    n = fisher.n_free
    assert F.shape == (n, n)
    assert n == 10  # 11 total - 1 fixed


def test_fisher_matrix_symmetric(fisher):
    """Fisher matrix is symmetric."""
    F = fisher.fisher_matrix
    assert jnp.allclose(F, F.T, rtol=1e-5)


def test_fisher_matrix_positive_diagonal(fisher):
    """Diagonal entries of Fisher matrix are positive."""
    F = fisher.fisher_matrix
    assert jnp.all(jnp.diag(F) > 0)


def test_fisher_matrix_positive_definite(fisher):
    """Fisher matrix is positive definite (all eigenvalues > 0)."""
    F = fisher.fisher_matrix
    eigvals = jnp.linalg.eigvalsh(F)
    assert jnp.all(eigvals > 0), f"Non-positive eigenvalue: {float(eigvals.min())}"


def test_free_params_exclude_fixed(fisher):
    """T_dust is not in the free parameter list."""
    assert "T_dust" not in fisher.free_parameter_names
    assert "r" in fisher.free_parameter_names
    assert "A_dust" in fisher.free_parameter_names


# -----------------------------------------------------------------------
# Constraints
# -----------------------------------------------------------------------

def test_sigma_r_positive(fisher):
    """σ(r) is positive and finite."""
    sr = fisher.sigma("r")
    assert sr > 0
    assert np.isfinite(sr)


def test_sigma_conditional_leq_marginalised(fisher):
    """Conditional σ ≤ marginalised σ (marginalization adds uncertainty)."""
    for param in fisher.free_parameter_names:
        sc = fisher.sigma_conditional(param)
        sm = fisher.sigma(param)
        assert sc <= sm * (1 + 1e-6), \
            f"Conditional σ({param})={sc:.4e} > marginalised {sm:.4e}"


def test_sigma_r_improves_with_more_detectors(signal_model, instrument):
    """Doubling detector count should improve (lower) σ(r)."""
    fisher_base = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    sr_base = fisher_base.sigma("r")

    # More detectors
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    inst2 = Instrument(channels=(
        Channel(90.0,  1000, 400.0, 30.0, efficiency=eff),
        Channel(150.0, 2000, 300.0, 20.0, efficiency=eff),
        Channel(220.0, 1000, 500.0, 15.0, efficiency=eff),
    ), mission_duration_years=5.0, f_sky=0.7)
    model2 = SignalModel(inst2, GaussianForegroundModel(), CMBSpectra(),
                         ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30)
    fisher2 = FisherForecast(
        model2, inst2, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    sr_more = fisher2.sigma("r")
    assert sr_more < sr_base, f"More detectors didn't help: {sr_more:.4e} >= {sr_base:.4e}"


def test_fixing_params_improves_sigma_r(signal_model, instrument):
    """Fixing more foreground params → fewer free params → lower σ(r)."""
    fisher_many_free = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    fisher_more_fixed = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust", "alpha_dust", "alpha_sync", "epsilon", "Delta_dust"],
    )
    sr_many = fisher_many_free.sigma("r")
    sr_fewer = fisher_more_fixed.sigma("r")
    assert sr_fewer < sr_many, \
        f"Fixing params didn't help: {sr_fewer:.4e} >= {sr_many:.4e}"


# -----------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------

def test_prior_tightens_constraint(signal_model, instrument):
    """Adding a tight prior on a degenerate parameter improves σ(r)."""
    fisher_no_prior = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={},
        fixed_params=["T_dust"],
    )
    fisher_with_prior = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors={"beta_dust": 0.01},  # very tight prior
        fixed_params=["T_dust"],
    )
    sr_no = fisher_no_prior.sigma("r")
    sr_with = fisher_with_prior.sigma("r")
    assert sr_with <= sr_no * (1 + 1e-6)


# -----------------------------------------------------------------------
# 2D marginalized
# -----------------------------------------------------------------------

def test_marginalized_2d_structure(fisher):
    """marginalized_2d returns correct keys and consistent values."""
    result = fisher.marginalized_2d("r", "A_dust")
    assert "cov_2d" in result
    assert result["cov_2d"].shape == (2, 2)
    assert result["sigma_i"] == pytest.approx(fisher.sigma("r"), rel=1e-5)
    assert result["sigma_j"] == pytest.approx(fisher.sigma("A_dust"), rel=1e-5)
    assert -1.0 <= result["rho"] <= 1.0


# -----------------------------------------------------------------------
# Inverse consistency
# -----------------------------------------------------------------------

def test_fisher_times_inverse_is_identity(fisher):
    """F × F⁻¹ ≈ I."""
    F = fisher.fisher_matrix
    F_inv = fisher.inverse
    product = F @ F_inv
    assert jnp.allclose(product, jnp.eye(fisher.n_free), atol=1e-6)


# -----------------------------------------------------------------------
# External noise path (opt-in)
# -----------------------------------------------------------------------

def _analytic_noise_array(signal_model, instrument):
    """Per-channel analytic N_ell^BB on the SignalModel ell grid, stacked."""
    ells = signal_model.ells
    return jnp.stack([
        noise_nl(ch, ells, instrument.mission_duration_years, instrument.f_sky)
        for ch in instrument.channels
    ])


def test_external_noise_matches_analytic(signal_model, instrument):
    """Passing the analytic noise array via external_noise_bb matches the
    analytic path to numerical precision."""
    priors = {"beta_dust": 0.11, "beta_sync": 0.3}
    fixed = ["T_dust"]

    fisher_analytic = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors=priors, fixed_params=fixed,
    )
    sigma_r_analytic = fisher_analytic.sigma("r")

    nl = _analytic_noise_array(signal_model, instrument)
    fisher_external = FisherForecast(
        signal_model, instrument, FIDUCIAL,
        priors=priors, fixed_params=fixed,
        external_noise_bb=nl,
    )
    sigma_r_external = fisher_external.sigma("r")

    assert sigma_r_external == pytest.approx(sigma_r_analytic, rel=1e-6)


def test_external_noise_bb_shape_validation(signal_model, instrument):
    """Wrong-shape external_noise_bb raises a ValueError."""
    nl_bad = jnp.zeros((99, 99))
    with pytest.raises(ValueError, match="external_noise_bb"):
        FisherForecast(
            signal_model, instrument, FIDUCIAL,
            priors={}, fixed_params=["T_dust"],
            external_noise_bb=nl_bad,
        )


def test_cleaned_map_instrument_requires_external_noise():
    """cleaned_map_instrument without external_noise_bb must raise, not
    silently fall through to the analytic path using dummy NET=1 uK√s."""
    from augr.config import cleaned_map_instrument
    from augr.foregrounds import NullForegroundModel

    inst = cleaned_map_instrument(f_sky=0.6)
    sm = SignalModel(
        inst, NullForegroundModel(), CMBSpectra(),
        ell_min=2, ell_max=50, delta_ell=5, ell_per_bin_below=30,
    )
    fiducial = {"r": 0.0, "A_lens": 1.0}
    with pytest.raises(ValueError, match="requires_external_noise"):
        FisherForecast(sm, inst, fiducial, priors={}, fixed_params=[])


# -----------------------------------------------------------------------
# Residual-template amplitude (A_res)
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def signal_model_with_template(instrument):
    """SignalModel carrying a flat residual template."""
    ells = np.arange(2, 400, dtype=float)
    cl = np.full_like(ells, 1e-4)
    return SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        residual_template_cl=cl, residual_template_ells=ells,
    )


_FIDUCIAL_WITH_A_RES = {**FIDUCIAL, "A_res": 1.0}


def test_a_res_in_free_params_without_prior(signal_model_with_template,
                                            instrument):
    """With no prior on A_res, it's a free parameter and sigma(A_res) is finite."""
    fisher = FisherForecast(
        signal_model_with_template, instrument, _FIDUCIAL_WITH_A_RES,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
    )
    assert "A_res" in fisher.free_parameter_names
    s = fisher.sigma("A_res")
    assert s > 0 and np.isfinite(s)


def test_a_res_gaussian_prior_tightens_sigma_r(signal_model_with_template,
                                               instrument):
    """Adding a Gaussian prior on A_res tightens (or preserves) sigma(r)."""
    priors_base = {"beta_dust": 0.11, "beta_sync": 0.3}
    fixed = ["T_dust"]
    fisher_flat = FisherForecast(
        signal_model_with_template, instrument, _FIDUCIAL_WITH_A_RES,
        priors=priors_base, fixed_params=fixed,
    )
    fisher_tight = FisherForecast(
        signal_model_with_template, instrument, _FIDUCIAL_WITH_A_RES,
        priors={**priors_base, "A_res": 0.01},
        fixed_params=fixed,
    )
    assert fisher_tight.sigma("r") <= fisher_flat.sigma("r") * (1 + 1e-8)


def test_a_res_prior_adds_to_fisher_diagonal(signal_model_with_template,
                                             instrument):
    """Prior sigma on A_res adds exactly 1/sigma^2 to its Fisher diagonal."""
    priors_base = {"beta_dust": 0.11, "beta_sync": 0.3}
    fixed = ["T_dust"]
    sigma_prior = 0.3

    fisher_no = FisherForecast(
        signal_model_with_template, instrument, _FIDUCIAL_WITH_A_RES,
        priors=priors_base, fixed_params=fixed,
    )
    fisher_with = FisherForecast(
        signal_model_with_template, instrument, _FIDUCIAL_WITH_A_RES,
        priors={**priors_base, "A_res": sigma_prior},
        fixed_params=fixed,
    )
    F_no = fisher_no.fisher_matrix
    F_with = fisher_with.fisher_matrix
    idx = fisher_no.free_parameter_names.index("A_res")

    off = F_with - F_no
    assert off[idx, idx] == pytest.approx(1.0 / sigma_prior**2, rel=1e-8)
    # Only the (A_res, A_res) diagonal should change.
    mask = jnp.ones_like(off).at[idx, idx].set(0.0)
    assert jnp.allclose(off * mask, 0.0, atol=1e-10)


def test_a_res_fixed_excluded_from_free_params(signal_model_with_template,
                                               instrument):
    """Fixing A_res removes it from the Fisher matrix entirely."""
    fisher = FisherForecast(
        signal_model_with_template, instrument, _FIDUCIAL_WITH_A_RES,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust", "A_res"],
    )
    assert "A_res" not in fisher.free_parameter_names
    assert fisher.fisher_matrix.shape == (fisher.n_free, fisher.n_free)


def test_summary_flags_low_mode_count_bins(instrument):
    """When any bin has fewer than ~10 Knox modes (f_sky × (2ℓ+1) × Δℓ),
    the Gaussian-likelihood approximation breaks down and Fisher sigma(r)
    is structurally narrower than a Wishart posterior.  The summary
    must flag this so absolute sigma(r) quotes are not taken at face
    value."""
    # ell_min=2, delta_ell=1 at the reionization bump: at ell=2 with
    # f_sky=0.7, nu_b = 0.7 * 5 = 3.5 -- well below 10.
    sig = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=2, ell_max=50, delta_ell=10, ell_per_bin_below=5,
    )
    ff = FisherForecast(sig, instrument, FIDUCIAL,
                        priors={"beta_dust": 0.11, "beta_sync": 0.3},
                        fixed_params=["T_dust"])
    ff.compute()
    text = ff.summary()
    assert "Knox modes/bin" in text
    assert "WARNING" in text
    assert "Gaussian-likelihood approximation" in text


def test_summary_no_warning_when_all_bins_well_sampled(instrument):
    """No warning when every bin has >> 10 modes."""
    sig = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=100, ell_max=300, delta_ell=50, ell_per_bin_below=100,
    )
    ff = FisherForecast(sig, instrument, FIDUCIAL,
                        priors={"beta_dust": 0.11, "beta_sync": 0.3},
                        fixed_params=["T_dust"])
    ff.compute()
    text = ff.summary()
    assert "Knox modes/bin" in text
    assert "WARNING" not in text


def test_summary_reports_condition_number(fisher):
    """Summary should always include a cond(F) line for reference."""
    fisher.compute()
    text = fisher.summary()
    assert "cond(F)" in text


def test_summary_flags_degenerate_fisher():
    """A single-channel Fisher with all FG parameters free is genuinely
    degenerate: one frequency cannot simultaneously constrain multiple
    SEDs.  cond(F) is effectively infinite and the summary must flag it."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    inst = Instrument(
        channels=(Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),),
        mission_duration_years=5.0, f_sky=0.7,
    )
    sig = SignalModel(
        inst, GaussianForegroundModel(), CMBSpectra(),
        ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
    )
    ff = FisherForecast(sig, inst, FIDUCIAL, priors={}, fixed_params=["T_dust"])
    ff.compute()
    cond_val = float(jnp.linalg.cond(ff.fisher_matrix))
    assert cond_val > 1e14, (
        f"fixture should be degenerate but cond(F) = {cond_val:.2e}; "
        "the warning path is not actually being exercised.")
    text = ff.summary()
    assert "cond(F)" in text
    assert "WARNING: near-degenerate" in text


# -----------------------------------------------------------------------
# Measured BPWF Fisher path
# -----------------------------------------------------------------------

def _per_ell_bpwf_signal_model(instrument, ell_min=30, ell_max=80):
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.eye(ells_in.size)
    return SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )


def test_fisher_measured_bpwf_requires_external_noise(instrument):
    """FisherForecast(BPWF SignalModel) without external_noise_bb must raise."""
    sm = _per_ell_bpwf_signal_model(instrument)
    with pytest.raises(ValueError, match="has_measured_bpwf"):
        FisherForecast(sm, instrument, FIDUCIAL, fixed_params=["T_dust"])


def test_fisher_measured_bpwf_per_ell_matches_baseline(instrument):
    """Per-ℓ identity BPWF Fisher matches the analytic delta_ell=1 Fisher.

    With the same external noise array on both sides the two paths
    should agree to machine precision, demonstrating that the BPWF-aware
    full-covariance solve reduces correctly to the per-bin block solve
    for non-overlapping single-ℓ delta windows.
    """
    ell_min, ell_max = 30, 80
    sm_baseline = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        delta_ell=1, ell_per_bin_below=ell_min,
    )
    sm_bpwf = _per_ell_bpwf_signal_model(instrument, ell_min, ell_max)

    nl = jnp.array([
        noise_nl(ch, sm_baseline.ells,
                 instrument.mission_duration_years,
                 instrument.f_sky)
        for ch in instrument.channels
    ])

    ff_baseline = FisherForecast(
        sm_baseline, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
        external_noise_bb=nl,
    )
    ff_bpwf = FisherForecast(
        sm_bpwf, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
        external_noise_bb=nl,
    )
    F_baseline = ff_baseline.compute()
    F_bpwf = ff_bpwf.compute()
    assert jnp.allclose(F_baseline, F_bpwf, rtol=1e-8, atol=1e-12)


def test_fisher_measured_bpwf_overlapping_runs(instrument):
    """Overlapping Gaussian BPWFs produce a finite, positive σ(r)."""
    ell_min, ell_max = 20, 200
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    centers = [40.0, 80.0, 130.0, 180.0]
    W = np.array([
        np.exp(-(ells_in - c) ** 2 / (2.0 * 18.0 ** 2)) for c in centers
    ])
    W /= W.sum(axis=1, keepdims=True)
    sm = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    nl = jnp.array([
        noise_nl(ch, sm.ells,
                 instrument.mission_duration_years,
                 instrument.f_sky)
        for ch in instrument.channels
    ])
    ff = FisherForecast(
        sm, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
        external_noise_bb=nl,
    )
    ff.compute()
    sigma_r = ff.sigma("r")
    assert np.isfinite(sigma_r) and sigma_r > 0


def test_fisher_summary_handles_bpwf_mode(instrument):
    """summary() works in BPWF mode and reports the BPWF Knox-mode label."""
    sm = _per_ell_bpwf_signal_model(instrument, ell_min=50, ell_max=120)
    nl = jnp.array([
        noise_nl(ch, sm.ells,
                 instrument.mission_duration_years,
                 instrument.f_sky)
        for ch in instrument.channels
    ])
    ff = FisherForecast(
        sm, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"],
        external_noise_bb=nl,
    )
    ff.compute()
    text = ff.summary()
    assert "Knox modes/bin (BPWF)" in text
    assert "cond(F)" in text


# -----------------------------------------------------------------------
# Per-spectrum BPWFs (Phase 2)
# -----------------------------------------------------------------------

def test_fisher_per_spectrum_identical_rows_matches_shared(instrument):
    """All-identical-rows dict reproduces the shared-BPWF Fisher exactly."""
    ell_min, ell_max = 30, 80
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.eye(ells_in.size)   # per-ℓ disjoint windows

    sm_shared = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    bpwf_dict = {p: W for p in sm_shared.freq_pairs}
    sm_per = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=bpwf_dict, bandpower_window_ells=ells_in,
    )

    nl = jnp.array([
        noise_nl(ch, sm_shared.ells,
                 instrument.mission_duration_years,
                 instrument.f_sky)
        for ch in instrument.channels
    ])
    ff_shared = FisherForecast(
        sm_shared, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"], external_noise_bb=nl,
    )
    ff_per = FisherForecast(
        sm_per, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"], external_noise_bb=nl,
    )
    F_shared = ff_shared.compute()
    F_per = ff_per.compute()
    assert jnp.allclose(F_shared, F_per, rtol=1e-8, atol=1e-12)


def test_fisher_summary_per_spectrum_label(instrument):
    """summary() tags the Knox-modes line as per-spec when applicable."""
    ell_min, ell_max = 30, 80
    ells_in = np.arange(ell_min, ell_max + 1, dtype=float)
    W = np.eye(ells_in.size)
    sm_probe = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=W, bandpower_window_ells=ells_in,
    )
    bpwf_dict = {p: W for p in sm_probe.freq_pairs}
    sm = SignalModel(
        instrument, GaussianForegroundModel(), CMBSpectra(),
        ell_min=ell_min, ell_max=ell_max,
        bandpower_window=bpwf_dict, bandpower_window_ells=ells_in,
    )
    nl = jnp.array([
        noise_nl(ch, sm.ells,
                 instrument.mission_duration_years,
                 instrument.f_sky)
        for ch in instrument.channels
    ])
    ff = FisherForecast(
        sm, instrument, FIDUCIAL,
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust"], external_noise_bb=nl,
    )
    ff.compute()
    text = ff.summary()
    assert "per-spec, first-pair" in text


# -----------------------------------------------------------------------
# Parameter bias
# -----------------------------------------------------------------------

# Moment-expansion superset of the test FIDUCIAL. Eight extra entries
# beyond the 11-key BK15-like dict at the top of this file; values set
# to zero so the moment fiducial reduces to Gaussian at the same SED.
FIDUCIAL_MOMENT_TEST = {
    **FIDUCIAL,
    "c_sync":        0.0,
    "Delta_sync":    0.0,
    "omega_d_beta":  0.0,
    "omega_d_T":     0.0,
    "omega_d_betaT": 0.0,
    "omega_s_beta":  0.0,
    "omega_s_c":     0.0,
    "omega_s_betac": 0.0,
}


class TestParameterBias:
    """parameter_bias / bias_from_truth_model linear bias formula."""

    @pytest.fixture(scope="class")
    def fisher_no_prior(self, signal_model, instrument):
        return FisherForecast(
            signal_model, instrument, FIDUCIAL,
            fixed_params=["T_dust"],
        )

    @pytest.fixture(scope="class")
    def fisher_priors(self, signal_model, instrument):
        return FisherForecast(
            signal_model, instrument, FIDUCIAL,
            priors={"beta_dust": 0.11, "beta_sync": 0.3},
            fixed_params=["T_dust"],
        )

    # ----- ΔD primitive ----------------------------------------------------

    def test_zero_delta_d_gives_zero_bias(self, fisher_priors, signal_model):
        bias = fisher_priors.parameter_bias(jnp.zeros(signal_model.n_data))
        assert set(bias) == set(fisher_priors.free_parameter_names)
        for name, val in bias.items():
            assert val == 0.0, f"non-zero bias on {name}: {val!r}"

    def test_returns_dict_keyed_by_free_names(self, fisher_priors, signal_model):
        rng = np.random.default_rng(0)
        dd = jnp.asarray(rng.standard_normal(signal_model.n_data) * 1e-4)
        bias = fisher_priors.parameter_bias(dd)
        assert set(bias) == set(fisher_priors.free_parameter_names)
        # T_dust is fixed in the fixture → must not appear.
        assert "T_dust" not in bias

    def test_linearity_in_delta_d(self, fisher_priors, signal_model):
        rng = np.random.default_rng(1)
        dd = jnp.asarray(rng.standard_normal(signal_model.n_data) * 1e-4)
        bias_1 = fisher_priors.parameter_bias(dd)
        bias_2 = fisher_priors.parameter_bias(2.0 * dd)
        for name in bias_1:
            # Allow a tiny absolute slack for fp64 roundoff at small values.
            assert abs(bias_2[name] - 2.0 * bias_1[name]) < \
                max(1e-14, 1e-10 * abs(bias_1[name]))

    def test_rejects_wrong_shape(self, fisher_priors, signal_model):
        with pytest.raises(ValueError, match="delta_data_vector has shape"):
            fisher_priors.parameter_bias(jnp.zeros(signal_model.n_data + 1))

    def test_recovers_known_r_shift_no_prior(self,
                                              fisher_no_prior,
                                              signal_model):
        """With no priors, parameter_bias must recover an input r shift
        exactly: the BB model is linear in r, so the Jacobian
        linearization has no quadratic remainder, and (F⁻¹·F)_{r,r} = 1
        with zero leakage into other params.
        """
        from augr.signal import flatten_params
        delta_r = 0.01
        fid_shifted = {**FIDUCIAL, "r": delta_r}
        p_fid = flatten_params(FIDUCIAL, signal_model.parameter_names)
        p_shifted = flatten_params(fid_shifted, signal_model.parameter_names)
        dd = signal_model.data_vector(p_shifted) - \
            signal_model.data_vector(p_fid)
        bias = fisher_no_prior.parameter_bias(dd)
        assert abs(bias["r"] - delta_r) < 1e-9 * abs(delta_r)
        for n in fisher_no_prior.free_parameter_names:
            if n != "r":
                assert abs(bias[n]) < 1e-8 * abs(delta_r), \
                    f"leakage into {n}: {bias[n]:.4e}"

    @pytest.mark.slow
    def test_matches_scipy_nonlinear_mle(self,
                                          fisher_no_prior,
                                          signal_model,
                                          instrument):
        """Brute-force MAP via scipy.minimize agrees with the linear
        bias formula at a small truth shift (in-manifold).
        """
        from scipy.optimize import minimize

        from augr.covariance import bandpower_covariance_blocks
        from augr.signal import flatten_params

        sigma_r = fisher_no_prior.sigma("r")
        # 5% of σ_r — well within the linear regime.
        delta_r = 0.05 * sigma_r
        fid_shifted = {**FIDUCIAL, "r": delta_r}
        all_names = signal_model.parameter_names
        p_fid = flatten_params(FIDUCIAL, all_names)
        p_shifted = flatten_params(fid_shifted, all_names)
        d_target = np.asarray(signal_model.data_vector(p_shifted))
        dd = jnp.asarray(d_target - np.asarray(
            signal_model.data_vector(p_fid)))

        bias_linear = fisher_no_prior.parameter_bias(dd)

        # Brute-force MAP. Pre-invert cov blocks once.
        cov_blocks = np.asarray(bandpower_covariance_blocks(
            signal_model, instrument, p_fid))
        cov_inv_blocks = np.linalg.inv(cov_blocks)
        n_spec = signal_model.n_spectra
        n_bins = signal_model.n_bins
        free_names = fisher_no_prior.free_parameter_names
        free_idx = [all_names.index(n) for n in free_names]
        p_fid_np = np.asarray(p_fid)

        def neg_log_lik(free_arr):
            full = p_fid_np.copy()
            full[free_idx] = free_arr
            d_model = np.asarray(signal_model.data_vector(jnp.asarray(full)))
            resid = d_target - d_model
            r_blocks = resid.reshape(n_spec, n_bins).T   # (n_bins, n_spec)
            chi2 = float(np.einsum('bs,bst,bt->',
                                    r_blocks, cov_inv_blocks, r_blocks))
            return 0.5 * chi2

        free_fid_arr = np.array([FIDUCIAL[n] for n in free_names])
        res = minimize(
            neg_log_lik, free_fid_arr, method='Nelder-Mead',
            options={'xatol': 1e-9, 'fatol': 1e-16,
                     'maxiter': 50000, 'adaptive': True},
        )
        delta_scipy = {n: float(res.x[i] - FIDUCIAL[n])
                       for i, n in enumerate(free_names)}

        for name in free_names:
            sigma_n = fisher_no_prior.sigma(name)
            diff = abs(bias_linear[name] - delta_scipy[name])
            # 1% of σ allows for Nelder-Mead convergence noise; the
            # underlying agreement is much tighter on r itself.
            assert diff < 0.01 * sigma_n, (
                f"bias mismatch on {name}: "
                f"linear={bias_linear[name]:.4e}, "
                f"scipy={delta_scipy[name]:.4e}, "
                f"σ={sigma_n:.4e}, diff={diff:.4e}"
            )

    # ----- bias_from_truth_model -------------------------------------------

    def test_zero_when_truth_equals_fit(self, fisher_priors, instrument):
        """Same FG model + same fiducials ⇒ ΔD = 0 ⇒ zero bias."""
        from augr.foregrounds import GaussianForegroundModel as G
        sm_truth = SignalModel(
            instrument, G(), CMBSpectra(),
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        )
        bias = fisher_priors.bias_from_truth_model(sm_truth, FIDUCIAL)
        for name, val in bias.items():
            assert abs(val) < 1e-12, \
                f"non-zero bias on {name}: {val:.4e}"

    def test_moment_truth_with_zero_omega_matches_gaussian(self,
                                                            fisher_priors,
                                                            instrument):
        """MomentExpansion(ω=0) ≡ Gaussian, so the bias must be ~zero."""
        from augr.foregrounds import MomentExpansionModel
        sm_truth = SignalModel(
            instrument, MomentExpansionModel(), CMBSpectra(),
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        )
        bias = fisher_priors.bias_from_truth_model(
            sm_truth, FIDUCIAL_MOMENT_TEST)
        sigma_r = fisher_priors.sigma("r")
        # The two FG models give identical C_ell at ω = 0; residual is
        # pure roundoff in the moment-expansion arithmetic.
        assert abs(bias["r"]) < 1e-6 * sigma_r, \
            f"ω=0 moment truth gave non-trivial bias['r']: {bias['r']:.4e}"

    def test_moment_truth_with_nonzero_omega_biases_r(self,
                                                       fisher_priors,
                                                       instrument):
        """A non-zero ω_d_beta produces a measurable Δr when the fit
        uses a Gaussian (no-moment) FG model."""
        from augr.foregrounds import MomentExpansionModel
        sm_truth = SignalModel(
            instrument, MomentExpansionModel(), CMBSpectra(),
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        )
        fid = {**FIDUCIAL_MOMENT_TEST, "omega_d_beta": 1e-3}
        bias = fisher_priors.bias_from_truth_model(sm_truth, fid)
        sigma_r = fisher_priors.sigma("r")
        # Effect should be well above any roundoff floor.
        assert abs(bias["r"]) > 1e-3 * sigma_r, \
            f"ω-truth gave suspiciously small bias['r']: {bias['r']:.4e}"

    def test_bias_linear_in_omega(self, fisher_priors, instrument):
        """The moment correction to C_ell is linear in ω at leading
        order, so doubling ω should double |Δr| to ~10% in the linear
        regime.
        """
        from augr.foregrounds import MomentExpansionModel
        sm_truth = SignalModel(
            instrument, MomentExpansionModel(), CMBSpectra(),
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        )
        fid_1 = {**FIDUCIAL_MOMENT_TEST, "omega_d_beta": 1e-3}
        fid_2 = {**FIDUCIAL_MOMENT_TEST, "omega_d_beta": 2e-3}
        b_1 = fisher_priors.bias_from_truth_model(sm_truth, fid_1)["r"]
        b_2 = fisher_priors.bias_from_truth_model(sm_truth, fid_2)["r"]
        ratio = b_2 / b_1
        assert abs(ratio - 2.0) < 0.1, \
            f"non-linear scaling: b(2ω)/b(ω) = {ratio:.3f}, expected ~2.0"

    def test_validates_frequency_mismatch(self, fisher_priors, instrument):
        eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
        inst_other = Instrument(channels=(
            Channel(95.0,  500, 400.0, 30.0, efficiency=eff),
            Channel(155.0, 1000, 300.0, 20.0, efficiency=eff),
            Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
        ), mission_duration_years=5.0, f_sky=0.7)
        sm_other = SignalModel(
            inst_other, GaussianForegroundModel(), CMBSpectra(),
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        )
        with pytest.raises(ValueError, match="frequencies"):
            fisher_priors.bias_from_truth_model(sm_other, FIDUCIAL)

    def test_validates_ell_grid_mismatch(self, fisher_priors, instrument):
        sm_other = SignalModel(
            instrument, GaussianForegroundModel(), CMBSpectra(),
            ell_min=20, ell_max=300, delta_ell=35, ell_per_bin_below=30,
        )
        with pytest.raises(ValueError, match="ells"):
            fisher_priors.bias_from_truth_model(sm_other, FIDUCIAL)


class TestParameterBiasIterative:
    """Gauss-Newton iteration of the linear bias formula."""

    @pytest.fixture(scope="class")
    def fisher_no_prior(self, signal_model, instrument):
        return FisherForecast(
            signal_model, instrument, FIDUCIAL,
            fixed_params=["T_dust"],
        )

    @pytest.fixture(scope="class")
    def fisher_priors(self, signal_model, instrument):
        return FisherForecast(
            signal_model, instrument, FIDUCIAL,
            priors={"beta_dust": 0.11, "beta_sync": 0.3},
            fixed_params=["T_dust"],
        )

    def test_first_step_matches_linear_formula(self, fisher_priors,
                                                 signal_model):
        """With max_iter=1, the iterative step equals the linear bias
        by construction (modulo a single non-convergence warning).
        """
        rng = np.random.default_rng(42)
        dd = jnp.asarray(rng.standard_normal(signal_model.n_data) * 1e-4)
        linear = fisher_priors.parameter_bias(dd)
        with pytest.warns(UserWarning,
                          match="did not converge in 1 iterations"):
            iterative = fisher_priors.parameter_bias_iterative(
                dd, max_iter=1, tol=0.0)
        for name in linear:
            assert abs(iterative[name] - linear[name]) < \
                max(1e-14, 1e-10 * abs(linear[name])), \
                f"first-step mismatch on {name}"

    def test_zero_delta_d_converges_in_one_iter(self, fisher_priors,
                                                  signal_model):
        biases, diag = fisher_priors.parameter_bias_iterative(
            jnp.zeros(signal_model.n_data), return_diagnostics=True)
        assert diag["converged"]
        assert diag["n_iter"] == 1
        assert diag["step_history"] == [0.0]
        for val in biases.values():
            assert val == 0.0

    def test_linear_regime_matches_linear_after_few_iters(self,
                                                            fisher_priors,
                                                            signal_model):
        """For ΔD built from a small in-manifold parameter shift, the
        iterative MAP converges to the linear formula within fp64.

        Use a tiny r-shift: r enters the BB model linearly so D(θ) is
        exactly affine in r, the Gauss-Newton fixed point coincides
        with the linear formula, and convergence is instant.
        """
        from augr.signal import flatten_params
        fid_shifted = {**FIDUCIAL, "r": 1e-6}
        p_fid = flatten_params(FIDUCIAL, signal_model.parameter_names)
        p_shifted = flatten_params(fid_shifted,
                                     signal_model.parameter_names)
        dd = signal_model.data_vector(p_shifted) - \
            signal_model.data_vector(p_fid)
        linear = fisher_priors.parameter_bias(dd)
        biases, diag = fisher_priors.parameter_bias_iterative(
            dd, tol=1e-8, return_diagnostics=True)
        assert diag["converged"]
        assert diag["n_iter"] <= 3, \
            f"unexpected slow convergence: {diag}"
        for name in linear:
            sigma_n = fisher_priors.sigma(name)
            assert abs(biases[name] - linear[name]) < 1e-6 * sigma_n, \
                f"linear-regime drift on {name}: " \
                f"iter={biases[name]:.4e}, linear={linear[name]:.4e}"

    def test_recovers_in_manifold_shift_no_prior(self, fisher_no_prior,
                                                   signal_model):
        """A substantial in-manifold shift of a nonlinear FG parameter
        (beta_dust ↦ beta_dust + 0.1) is recovered to high precision by
        Gauss-Newton iteration; the linear formula alone misses by
        O(δθ²) because beta_dust enters the FG model nonlinearly.
        """
        from augr.signal import flatten_params
        shift = 0.1
        fid_shifted = {**FIDUCIAL,
                        "beta_dust": FIDUCIAL["beta_dust"] + shift}
        p_fid = flatten_params(FIDUCIAL, signal_model.parameter_names)
        p_shifted = flatten_params(fid_shifted,
                                     signal_model.parameter_names)
        dd = signal_model.data_vector(p_shifted) - \
            signal_model.data_vector(p_fid)
        biases, diag = fisher_no_prior.parameter_bias_iterative(
            dd, max_iter=30, tol=1e-8, return_diagnostics=True)
        assert diag["converged"], \
            f"failed to converge on in-manifold shift: {diag}"
        # Recovered shift on the perturbed parameter.
        assert abs(biases["beta_dust"] - shift) < 1e-6 * abs(shift), \
            f"failed to recover beta_dust shift: " \
            f"got {biases['beta_dust']:.6e}, expected {shift:.6e}"
        # Other free params: near-zero (no degeneracy mixing at MAP
        # since the truth is exactly representable).
        for name in fisher_no_prior.free_parameter_names:
            if name == "beta_dust":
                continue
            sigma_n = fisher_no_prior.sigma(name)
            assert abs(biases[name]) < 1e-4 * sigma_n, \
                f"unexpected leakage into {name}: {biases[name]:.4e} " \
                f"(σ={sigma_n:.4e})"

    def test_linear_overshoots_in_nonlinear_regime(self, fisher_no_prior,
                                                     signal_model):
        """Linear bias of an in-manifold β_dust shift is off by O(δθ²);
        iterative cleans that up. Confirms the iterative refinement is
        actually doing work and not just returning the linear formula.
        """
        from augr.signal import flatten_params
        shift = 0.1
        fid_shifted = {**FIDUCIAL,
                        "beta_dust": FIDUCIAL["beta_dust"] + shift}
        p_fid = flatten_params(FIDUCIAL, signal_model.parameter_names)
        p_shifted = flatten_params(fid_shifted,
                                     signal_model.parameter_names)
        dd = signal_model.data_vector(p_shifted) - \
            signal_model.data_vector(p_fid)
        linear = fisher_no_prior.parameter_bias(dd)
        iterative = fisher_no_prior.parameter_bias_iterative(
            dd, max_iter=30, tol=1e-8)
        # Linear should miss the true shift by visibly more than fp64
        # noise (β_dust is nonlinear in the FG SED, so O(δθ²) bites);
        # iterative should land within ~1e-5 relative of it.
        assert abs(linear["beta_dust"] - shift) > 1e-3 * abs(shift), \
            "shift too small to exercise nonlinearity"
        assert abs(iterative["beta_dust"] - shift) < 1e-5 * abs(shift)

    @pytest.mark.slow
    def test_matches_scipy_nonlinear_mle(self, fisher_no_prior,
                                          signal_model, instrument):
        """Iterative bias agrees with scipy.minimize MAP on a
        nonlinearly-shifted in-manifold truth."""
        from scipy.optimize import minimize

        from augr.covariance import bandpower_covariance_blocks
        from augr.signal import flatten_params

        shift = 0.15
        fid_shifted = {**FIDUCIAL,
                        "beta_dust": FIDUCIAL["beta_dust"] + shift}
        all_names = signal_model.parameter_names
        p_fid = flatten_params(FIDUCIAL, all_names)
        p_shifted = flatten_params(fid_shifted, all_names)
        d_target = np.asarray(signal_model.data_vector(p_shifted))
        dd = jnp.asarray(d_target -
                          np.asarray(signal_model.data_vector(p_fid)))

        biases_iter, diag = fisher_no_prior.parameter_bias_iterative(
            dd, max_iter=50, tol=1e-8, return_diagnostics=True)
        assert diag["converged"]

        # Brute-force MAP via scipy.minimize.
        cov_blocks = np.asarray(bandpower_covariance_blocks(
            signal_model, instrument, p_fid))
        cov_inv_blocks = np.linalg.inv(cov_blocks)
        n_spec, n_bins = (signal_model.n_spectra,
                           signal_model.n_bins)
        free_names = fisher_no_prior.free_parameter_names
        free_idx = [all_names.index(n) for n in free_names]
        p_fid_np = np.asarray(p_fid)

        def neg_log_lik(free_arr):
            full = p_fid_np.copy()
            full[free_idx] = free_arr
            d_model = np.asarray(signal_model.data_vector(
                jnp.asarray(full)))
            r_b = (d_target - d_model).reshape(n_spec, n_bins).T
            return 0.5 * float(np.einsum('bs,bst,bt->',
                                          r_b, cov_inv_blocks, r_b))

        free_fid_arr = np.array([FIDUCIAL[n] for n in free_names])
        res = minimize(
            neg_log_lik, free_fid_arr, method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-16,
                     'maxiter': 50000, 'adaptive': True},
        )
        delta_scipy = {n: float(res.x[i] - FIDUCIAL[n])
                       for i, n in enumerate(free_names)}

        for name in free_names:
            sigma_n = fisher_no_prior.sigma(name)
            diff = abs(biases_iter[name] - delta_scipy[name])
            # 1% of σ tolerates Nelder-Mead convergence noise.
            assert diff < 0.01 * sigma_n, (
                f"iterative vs scipy mismatch on {name}: "
                f"iter={biases_iter[name]:.4e}, "
                f"scipy={delta_scipy[name]:.4e}, "
                f"σ={sigma_n:.4e}, diff={diff:.4e}"
            )

    def test_diagnostics_dict_shape(self, fisher_priors, signal_model):
        from augr.signal import flatten_params
        # Use a small in-manifold shift so the iteration is well-behaved
        # and we exercise the success path of the diagnostics output.
        fid_shifted = {**FIDUCIAL, "r": 1e-5}
        p_fid = flatten_params(FIDUCIAL, signal_model.parameter_names)
        p_shifted = flatten_params(fid_shifted,
                                     signal_model.parameter_names)
        dd = signal_model.data_vector(p_shifted) - \
            signal_model.data_vector(p_fid)
        result = fisher_priors.parameter_bias_iterative(
            dd, return_diagnostics=True)
        assert isinstance(result, tuple) and len(result) == 2
        biases, diag = result
        assert set(diag) == {"converged", "n_iter", "step_history"}
        assert isinstance(diag["converged"], bool)
        assert isinstance(diag["n_iter"], int)
        assert isinstance(diag["step_history"], list)
        assert len(diag["step_history"]) == diag["n_iter"]
        assert set(biases) == set(fisher_priors.free_parameter_names)

    def test_warns_when_not_converged(self, fisher_priors, signal_model):
        rng = np.random.default_rng(3)
        dd = jnp.asarray(rng.standard_normal(signal_model.n_data) * 1e-3)
        with pytest.warns(UserWarning, match="did not converge"):
            biases, diag = fisher_priors.parameter_bias_iterative(
                dd, max_iter=1, tol=0.0, return_diagnostics=True)
        assert diag["converged"] is False
        assert diag["n_iter"] == 1
        # Biases should still be the latest iterate (the linear step).
        for name in biases:
            assert np.isfinite(biases[name])

    def test_bias_from_truth_iterative_matches_direct_dd(self,
                                                          fisher_no_prior,
                                                          instrument):
        """The truth-model wrapper produces the same iterate as feeding
        the manually-built ΔD to parameter_bias_iterative."""
        from augr.foregrounds import MomentExpansionModel
        from augr.signal import flatten_params
        sm_truth = SignalModel(
            instrument, MomentExpansionModel(), CMBSpectra(),
            ell_min=20, ell_max=200, delta_ell=35, ell_per_bin_below=30,
        )
        fid_truth = {**FIDUCIAL_MOMENT_TEST, "omega_d_beta": 5e-3}
        via_wrapper = fisher_no_prior.bias_from_truth_model_iterative(
            sm_truth, fid_truth, max_iter=30, tol=1e-8)

        # Build ΔD by hand and call the primitive directly.
        truth_params = flatten_params(fid_truth,
                                       sm_truth.parameter_names)
        d_truth = sm_truth.data_vector(truth_params)
        p_fid = flatten_params(FIDUCIAL,
                                fisher_no_prior._signal.parameter_names)
        dd = d_truth - fisher_no_prior._signal.data_vector(p_fid)
        direct = fisher_no_prior.parameter_bias_iterative(
            dd, max_iter=30, tol=1e-8)

        for name in via_wrapper:
            assert via_wrapper[name] == direct[name], \
                f"mismatch on {name}: " \
                f"wrapper={via_wrapper[name]}, direct={direct[name]}"
