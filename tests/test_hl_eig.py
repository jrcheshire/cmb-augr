"""Gate for the Stage-2 HL-EIG estimator (augr.eig.hl_eig_from_external_cov).

Fast tests run on a synthetic SPD covariance + a real cleaned-map linear signal model (no
map forward), so they exercise the nested-MC + grid-quadrature estimator directly:

* 4a-i  -- the Gaussian column reproduces the exact linear-Gaussian r-marginal mutual
  information ``log(sigma_prior_r / sigma_post)`` to MC error (validates the whole
  draw / grid / KL machinery, no HL approximation involved).
* 4a-ii -- in the small-residual (Gaussian) limit the HL column reduces to the Gaussian
  column (CRN-paired), validating that HL -> Gaussian.
* 4c    -- convergence: the MC standard error falls as ``1/sqrt(n_outer)`` and the estimate
  is stable under doubling the r-grid; the r-grid is wide enough (small edge mass).

Importing ``augr.eig`` pulls in the map-based forward (spectrum_stages -> jht / ducc0), so
these are skipped without the [masking] extra even though the math needs no map.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jht")
pytest.importorskip("ducc0")

import jax

from augr.eig import HLEIGContext, hl_eig_from_external_cov
from augr.signal import flatten_params

PRIOR_SIG = {"A_lens": 0.25, "A_res": 0.3}


def _analytic_sigma_post(ctx, cov, sigma_prior_r):
    """Linear-Gaussian r-marginal posterior sigma ``sqrt((F^-1)_00)``.

    ``F = J^T C^-1 J + diag(prior_prec)`` with ``J = [t_r, t_l, t_res]`` (floated columns
    only). ``sigma_prior_r = inf`` gives the data-only marginal width (flat r prior).
    """
    cols = [np.asarray(ctx.t_r)]
    prec = [1.0 / sigma_prior_r**2]
    if "A_lens" in ctx.floated:
        cols.append(np.asarray(ctx.t_l))
        prec.append(1.0 / ctx.prior_sig["A_lens"] ** 2)
    if "A_res" in ctx.floated:
        cols.append(np.asarray(ctx.t_res))
        prec.append(1.0 / ctx.prior_sig["A_res"] ** 2)
    j = np.stack(cols, axis=1)  # (n_bins, n_param)
    fisher = j.T @ np.linalg.solve(np.asarray(cov), j) + np.diag(prec)
    return float(np.sqrt(np.linalg.inv(fisher)[0, 0]))


def _analytic_marginal_mi(ctx, cov, sigma_prior_r):
    """Exact linear-Gaussian r-marginal MI ``log(sigma_prior_r / sigma_post)``."""
    return np.log(sigma_prior_r) - np.log(_analytic_sigma_post(ctx, cov, sigma_prior_r))


def _build(
    *,
    frac,
    ratio=5.0,
    n_grid=800,
    n_nuis_grid=51,
    ell_max=60,
    floated=frozenset({"A_lens", "A_res"}),
):
    """HLEIGContext + synthetic diagonal cov + mean_bandpower, sized for an O(1) EIG.

    ``frac`` sets the per-bin fluctuation level ``sqrt(diag(cov)) = frac * mean_bandpower``
    (small ``frac`` -> Gaussian limit). ``sigma_prior_r`` is set to ``ratio`` x the data-only
    marginal sigma(r) so the EIG ~ log(ratio) is O(1) (the positive-information regime where
    the nested-MC estimator is well-conditioned), independent of the tensor template scale.
    """
    kw = dict(
        template_ells=np.arange(2, ell_max + 1, dtype=float),
        template_cl=(np.arange(2, ell_max + 1, dtype=float) / 5.0) ** -2.4,
        f_sky=0.6,
        r_fid=0.0,
        floated=floated,
        prior_sig=PRIOR_SIG,
        n_nuis_grid=n_nuis_grid,
        ell_max=ell_max,
        delta_ell=10,
        ell_per_bin_below=15,
    )
    probe = HLEIGContext.build(sigma_prior_r=1.0, n_grid=8, **kw)
    sig_fid = np.asarray(probe.signal_model.data_vector(probe.fid_vec))
    noise_floor = 0.5 * np.mean(np.abs(sig_fid)) * np.ones_like(sig_fid)
    mean_bp = sig_fid + noise_floor
    cov = np.diag((frac * mean_bp) ** 2)
    sigma_prior_r = ratio * _analytic_sigma_post(probe, cov, np.inf)
    ctx = HLEIGContext.build(sigma_prior_r=sigma_prior_r, n_grid=n_grid, **kw)
    return ctx, cov, mean_bp, sigma_prior_r


def test_gaussian_column_matches_analytic_mi():
    """4a-i: the Gaussian-column EIG reproduces the exact linear-Gaussian marginal MI.

    Floats one nuisance (A_res) at a well-resolved grid -- validates the on-grid nuisance
    marginalization, not just the r quadrature. (Two nuisances at this resolution is an
    n_nuis_grid**2 grid and too slow for the fast gate; the convergence test covers the
    resolution dependence.)
    """
    ctx, cov, mean_bp, sigma_prior_r = _build(
        frac=0.05, n_grid=600, n_nuis_grid=61, floated=frozenset({"A_res"})
    )
    res = hl_eig_from_external_cov(
        cov, mean_bp, ctx, key=jax.random.PRNGKey(0), n_outer=2000, return_diagnostics=True
    )
    mi = _analytic_marginal_mi(ctx, cov, sigma_prior_r)
    # Gate on the *computed* MC standard-error band (no hand rtol); a small grid-quadrature
    # allowance covers the residual nuisance-grid bias at this resolution (see 4c below).
    band = 4.0 * res.stderr_gauss + 0.02 * abs(mi)
    assert abs(res.eig_gauss - mi) < band, (res.eig_gauss, mi, res.stderr_gauss)
    assert res.edge_frac < 1e-3  # r-grid wide enough that the KL is not truncated


def test_hl_reduces_to_gaussian_small_residual():
    """4a-ii: in the small-residual limit HL -> Gaussian (the g() transform -> identity).

    No nuisances floated, so this isolates the per-bin HL transform vs Gaussian (the nuisance
    grid is a separate concern, gated in 4a-i / the convergence test).
    """
    ctx, cov, mean_bp, _sigma_prior_r = _build(frac=0.03, n_grid=600, floated=frozenset())
    res = hl_eig_from_external_cov(
        cov, mean_bp, ctx, key=jax.random.PRNGKey(1), n_outer=2000, return_diagnostics=True
    )
    # The two columns see the *same* draws (CRN), so the only difference is the g()
    # nonlinearity, which vanishes as the residuals shrink.
    band = 4.0 * (res.stderr_hl + res.stderr_gauss)
    assert abs(res.eig_hl - res.eig_gauss) < band + 0.02 * abs(res.eig_gauss), (
        res.eig_hl,
        res.eig_gauss,
    )


def test_nuisance_grid_converges():
    """4c: the nuisance-grid bias shrinks to within MC error as n_nuis_grid grows.

    A tightly-constrained nuisance (A_res posterior ~3x narrower than its prior here) is
    badly under-resolved by a coarse grid; doubling n_nuis_grid must drive the Gaussian-column
    EIG to the analytic MI. This operationalizes the resolution discipline in
    HLEIGContext.build's docstring.
    """
    floated = frozenset({"A_res"})
    diffs = []
    for nn in (21, 41, 81):
        ctx, cov, mean_bp, sigma_prior_r = _build(
            frac=0.05, n_grid=400, n_nuis_grid=nn, floated=floated
        )
        res = hl_eig_from_external_cov(
            cov, mean_bp, ctx, key=jax.random.PRNGKey(7), n_outer=1500, return_diagnostics=True
        )
        mi = _analytic_marginal_mi(ctx, cov, sigma_prior_r)
        diffs.append((abs(res.eig_gauss - mi), res.stderr_gauss))
    # monotone improvement, and the finest grid is within the MC band.
    assert diffs[0][0] > diffs[1][0] > diffs[2][0], diffs
    assert diffs[2][0] < 4.0 * diffs[2][1] + 0.02, diffs


def test_stderr_scales_with_n_outer():
    """4c: the MC standard error falls as 1/sqrt(n_outer)."""
    ctx, cov, mean_bp, _sp = _build(frac=0.08, n_grid=400, floated=frozenset())
    se = []
    for n_outer in (500, 2000):
        res = hl_eig_from_external_cov(
            cov, mean_bp, ctx, key=jax.random.PRNGKey(2), n_outer=n_outer, return_diagnostics=True
        )
        se.append(res.stderr_hl)
    ratio = se[0] / se[1]
    np.testing.assert_allclose(ratio, np.sqrt(2000 / 500), rtol=0.2)


def test_grid_resolution_stable():
    """4c: the HL-EIG estimate is stable (within MC error) under doubling the r-grid."""
    se = None
    vals = []
    for n_grid in (400, 800):
        ctx, cov, mean_bp, _sp = _build(frac=0.08, n_grid=n_grid, floated=frozenset())
        res = hl_eig_from_external_cov(
            cov, mean_bp, ctx, key=jax.random.PRNGKey(3), n_outer=2000, return_diagnostics=True
        )
        vals.append(res.eig_hl)
        se = res.stderr_hl
    assert abs(vals[0] - vals[1]) < 4.0 * se + 0.01 * abs(vals[1]), vals


def test_flatten_params_smoke():
    """Sanity: the context fiducial round-trips the (r, A_lens, A_res) layout."""
    ctx, _cov, _mean, _sp = _build(frac=0.05, n_grid=8, floated=frozenset())
    names = list(ctx.signal_model.parameter_names)
    fid = flatten_params({"r": 0.0, "A_lens": 1.0, "A_res": 1.0}, names)
    np.testing.assert_allclose(np.asarray(ctx.fid_vec), np.asarray(fid))
