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
