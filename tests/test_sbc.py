"""Tests for augr.sbc — the simulation-based-calibration / coverage core.

The fast tests validate the core for an *exactly correct* likelihood with no MC pipeline:
data drawn from N(mu, Sigma), Gaussian likelihood with that Sigma -> coverage must be
nominal (and a deliberately too-narrow Sigma must under-cover, proving the test has power).
This is the baseline that must hold before any MC verdict from the driver is trusted. The
slow test is a tiny nside=16 end-to-end wiring smoke of the MC driver.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from augr import sbc
from augr.likelihood.from_cutsky import build_cutsky_signal_model
from augr.likelihood.gaussian import GaussianLikelihood
from augr.likelihood.hl import HLLikelihood
from augr.likelihood.ordering import SpectrumLayout
from augr.signal import flatten_params

_R_TRUE = 0.01


def _gaussian_case(cov_scale_for_likelihood: float = 1.0):
    """Single-field linear model + Gaussian draws; return (core, draws, r_grid, n_trials).

    The likelihood covariance is ``cov_scale_for_likelihood * Sigma``; the draws always
    use the true ``Sigma``. Scale 1.0 = correctly specified (nominal coverage); scale < 1
    = too-narrow likelihood (under-coverage).
    """
    ells = np.arange(2, 31, dtype=float)
    sm, _inst = build_cutsky_signal_model(
        ells, np.zeros_like(ells), 0.5, ell_min=2, ell_max=30, delta_ell=10, ell_per_bin_below=2
    )
    names = list(sm.parameter_names)
    fid_vec = flatten_params({"r": _R_TRUE, "A_lens": 1.0, "A_res": 1.0}, names)
    layout = SpectrumLayout.from_freq_pairs(sm.freq_pairs, sm.n_bins)

    signal_fid = sbc.data_vector_at(sm, names, _R_TRUE, 1.0, 1.0)
    n_b = 0.3 * signal_fid  # a noise floor
    c_total = signal_fid + n_b
    sigma_diag = (0.3 * c_total) ** 2  # true per-bin variance (diagonal)

    gauss0 = GaussianLikelihood.from_external(sm, fid_vec, np.diag(cov_scale_for_likelihood * sigma_diag))
    hl0 = HLLikelihood.from_external(sm, fid_vec, c_total, np.diag(cov_scale_for_likelihood * sigma_diag))

    floated: set[str] = set()  # conditional: isolate the likelihood-shape effect on r
    prior_sig = {"A_lens": 0.25, "A_res": 0.3}
    nuis = sbc.NuisanceGrid.build(floated=floated, prior_sig=prior_sig, n_nuis_grid=21, n_sigma_nuis=5.0)
    base, t_r, t_l, t_res = sbc.linear_basis(sm, names)
    sigma_r = sbc.marginal_sigma_r(
        t_r=t_r, t_l=t_l, t_res=t_res, cov_diag=sigma_diag, floated=floated, prior_sig=prior_sig
    )
    r_grid = np.linspace(_R_TRUE - 12 * sigma_r, _R_TRUE + 12 * sigma_r, 400)
    pred_flat = sbc.build_pred_grid(base, t_r, t_l, t_res, r_grid=r_grid, al_axis=nuis.al_axis, ares_axis=nuis.ares_axis)
    core = sbc.make_marginal_logpost(
        gauss0=gauss0, hl0=hl0, noise_floor=n_b, layout=layout, pred_flat=pred_flat,
        logprior_grid=nuis.logprior_grid, n_grid=400, n_al=nuis.n_al, n_ar=nuis.n_ar,
    )

    n_trials = 2000
    rng = np.random.default_rng(0)
    draws = c_total[None, :] + np.sqrt(sigma_diag)[None, :] * rng.standard_normal((n_trials, c_total.size))
    return core, draws, r_grid, n_trials


def _coverage(rows, label, key):
    return dict(rows)[label][key][0]


def test_synthetic_gaussian_coverage_is_nominal():
    """Correctly-specified Gaussian: coverage matches the nominal level (the calibration baseline)."""
    core, draws, r_grid, n_trials = _gaussian_case(cov_scale_for_likelihood=1.0)
    result = sbc.run_coverage(core, draws, r_true=_R_TRUE, r_grid=r_grid, n_trials=n_trials)
    rows = sbc.coverage_table(result)
    assert result.edge_hits["gauss"] == 0  # grid wide enough
    assert abs(_coverage(rows, "two-sided 0.68", "gauss") - 0.68) < 0.04
    assert abs(_coverage(rows, "two-sided 0.95", "gauss") - 0.95) < 0.025
    # PIT ~ Uniform(0,1): mean ~ 0.5.
    assert abs(result.pit["gauss"].mean() - 0.5) < 0.03


def test_too_narrow_cov_undercovers():
    """A 4x-too-narrow likelihood covariance must under-cover (the test has power)."""
    core, draws, r_grid, n_trials = _gaussian_case(cov_scale_for_likelihood=0.25)
    result = sbc.run_coverage(core, draws, r_true=_R_TRUE, r_grid=r_grid, n_trials=n_trials)
    rows = sbc.coverage_table(result)
    assert _coverage(rows, "two-sided 0.68", "gauss") < 0.55  # nominal 0.68, clearly under


def test_nuisance_grid_conditional_is_singleton():
    nuis = sbc.NuisanceGrid.build(floated=set(), prior_sig={"A_lens": 0.25, "A_res": 0.3}, n_nuis_grid=21, n_sigma_nuis=5.0)
    assert nuis.n_al == 1 and nuis.n_ar == 1
    nuis2 = sbc.NuisanceGrid.build(floated={"A_res"}, prior_sig={"A_lens": 0.25, "A_res": 0.3}, n_nuis_grid=21, n_sigma_nuis=5.0)
    assert nuis2.n_al == 1 and nuis2.n_ar == 21


@pytest.mark.slow
def test_mc_coverage_smoke():
    """End-to-end wiring smoke of the MC driver at nside=16 (fg=none, no network)."""
    pytest.importorskip("ducc0")
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_hl_coverage_mc.py"
    cmd = [
        sys.executable, str(script),
        "--fg-model", "none", "--float", "A_lens",
        "--nside", "16", "--lmax", "24", "--ell-max", "24",
        "--delta-ell", "8", "--ell-per-bin-below", "2",
        "--n-train", "12", "--n-test", "12", "--n-grid", "80", "--workers", "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "MC-ensemble HL coverage" in proc.stdout
    assert "cross-debias (headline)" in proc.stdout
    assert "KS verdict" in proc.stdout
