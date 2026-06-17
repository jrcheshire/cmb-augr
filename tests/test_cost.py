"""Tests for the convex cost / budget primitive (augr.cost)."""

from __future__ import annotations

import jax
import numpy as np
from scipy.optimize import minimize_scalar

from augr.cost import CostModel, aperture_from_fwhm, budget_penalty
from augr.telescope import beam_fwhm_arcmin


def test_total_cost_matches_pico_anchor():
    """The placeholder defaults are anchored so a PICO-scale point is ~$1B."""
    cost = CostModel()
    c = float(cost.total_cost(aperture_m=1.4, n_det_total=12000.0, mission_years=5.0))
    # optics 204*1.96 + detectors 0.0292*12000 + ops 50*5 = 1000.24 $M.
    assert abs(c - 1000.0) < 1.0


def test_total_cost_grad_finite_and_fd_matched():
    """jax.grad in each design argument is finite and matches finite differences."""
    cost = CostModel()
    args = (1.5, 12000.0, 5.0)

    def f(d, n, y):
        return cost.total_cost(d, n, y)

    g = jax.grad(f, argnums=(0, 1, 2))(*args)
    g = np.array([float(gi) for gi in g])
    assert np.all(np.isfinite(g))

    # Analytic partials: a*p*D**(p-1), b_detector, c_year.
    assert np.isclose(g[0], cost.a_aperture * cost.p_aperture * args[0] ** (cost.p_aperture - 1))
    assert np.isclose(g[1], cost.b_detector)
    assert np.isclose(g[2], cost.c_year)

    # Central finite differences.
    for i, h in ((0, 1e-4), (1, 1.0), (2, 1e-4)):
        a_hi = list(args)
        a_lo = list(args)
        a_hi[i] += h
        a_lo[i] -= h
        fd = (float(f(*a_hi)) - float(f(*a_lo))) / (2 * h)
        assert np.isclose(g[i], fd, rtol=1e-5)


def test_total_cost_monotone_increasing():
    """Cost strictly increases in each design argument (more/bigger costs more)."""
    cost = CostModel()
    base = float(cost.total_cost(1.5, 10000.0, 4.0))
    assert float(cost.total_cost(1.6, 10000.0, 4.0)) > base
    assert float(cost.total_cost(1.5, 11000.0, 4.0)) > base
    assert float(cost.total_cost(1.5, 10000.0, 5.0)) > base


def test_total_cost_convex_in_aperture():
    """Convex in aperture: a smooth (non-constant) midpoint check on the D**p term."""
    cost = CostModel()
    # Second difference f(D-d) - 2 f(D) + f(D+d) >= 0 across a range (not measure-zero:
    # the aperture is swept, so the convex D**p term, not just an average, is exercised).
    n0, y0, d = 10000.0, 4.0, 0.25
    for big_d in (1.0, 1.5, 2.0, 3.0):
        lo = float(cost.total_cost(big_d - d, n0, y0))
        mid = float(cost.total_cost(big_d, n0, y0))
        hi = float(cost.total_cost(big_d + d, n0, y0))
        assert lo - 2.0 * mid + hi > 0.0


def test_budget_penalty_one_sided():
    """Penalty is zero under/at budget, positive above; gradient one-sided."""
    budget = 1000.0
    assert float(budget_penalty(800.0, budget)) == 0.0
    assert float(budget_penalty(budget, budget)) == 0.0
    assert float(budget_penalty(1200.0, budget)) > 0.0

    grad = jax.grad(lambda c: budget_penalty(c, budget, weight=1e-3))
    assert float(grad(800.0)) == 0.0  # affordable design is not pushed
    over_grad = float(grad(1200.0))
    assert over_grad > 0.0
    assert np.isclose(over_grad, 2 * 1e-3 * (1200.0 - budget))  # 2*w*(cost-budget)


def test_aperture_from_fwhm_inverts_beam():
    """aperture_from_fwhm round-trips augr.telescope.beam_fwhm_arcmin."""
    for nu in (90.0, 150.0, 220.0):
        for d in (0.8, 1.4, 2.5):
            fwhm = beam_fwhm_arcmin(nu, d)
            d_back = float(aperture_from_fwhm(fwhm, nu))
            assert np.isclose(d_back, d, rtol=1e-10)

    # Differentiable in FWHM (d aperture / d FWHM = -K(nu)/FWHM**2 < 0: finer beam ->
    # bigger dish).
    g = float(jax.grad(lambda f: aperture_from_fwhm(f, 150.0))(20.0))
    assert np.isfinite(g) and g < 0.0


def test_budget_creates_interior_optimum():
    """A budget converts a run-to-the-boundary descent into an interior optimum.

    sigma_proxy(D) = k/D decreases without bound in aperture, so unconstrained the
    optimum is D -> inf. Adding the budget penalty stops it at the budget wall: the
    minimizer lands in the interior of the bracket, at cost ~= budget.
    """
    cost = CostModel()
    n0, y0 = 12000.0, 5.0
    d_budget = 2.0
    budget = float(cost.total_cost(d_budget, n0, y0))  # cost exhausted at D = 2.0 m

    def objective(d):
        sigma_proxy = 1e-2 / d  # tiny resolution pull toward larger D
        pen = budget_penalty(cost.total_cost(d, n0, y0), budget, weight=1.0)
        return float(sigma_proxy + pen)

    res = minimize_scalar(objective, bounds=(0.5, 6.0), method="bounded")
    d_star = float(res.x)

    # Interior: the budget stopped the descent before either bracket bound (without it,
    # sigma_proxy alone would run to the upper bound).
    assert 0.5 + 1e-3 < d_star < 6.0 - 1e-3
    # The budget binds: the optimum sits at the budget wall (aperture ~= d_budget, with
    # cost ~= budget to within the optimizer tolerance).
    assert abs(d_star - d_budget) < 0.05
    assert np.isclose(float(cost.total_cost(d_star, n0, y0)), budget, rtol=1e-3)
