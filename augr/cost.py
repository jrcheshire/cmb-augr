"""
cost.py -- A differentiable convex cost / budget primitive for design optimization.

The differentiable map-based design levers (``optimize_mapbased.sigma_r_from_noise_design``
/ ``sigma_r_from_beam_design``) minimize sigma(r) as a function of the instrument
design. Unconstrained, those descents are ill-posed: more detectors, lower NET, and a
finer beam all only ever *help* sigma(r), so a free design knob runs to its boundary
(``n_det -> inf``, ``NET -> 0``, ``FWHM -> 0``). A real design lives under a **budget**;
this module supplies the convex cost surface that closes the optimization so it has an
interior optimum.

This is the *primitive only* -- the prerequisite for Bayesian-optimal experimental
design (EIG/BOED through the differentiable forward). Wiring the cost into a constrained
sigma(r) / EIG objective (penalty term or budget-normalized reparametrization) is the
design-optimization work that consumes it; see ``allocation.grouped_allocation`` for the
existing fixed-budget *softmax reparametrization* pattern (which conserves a detector /
focal-plane-area budget) that this generalizes to a dollar budget across aperture,
detectors, and mission time.

Everything is ``jnp`` and differentiable in the design arguments.

Usage::

    from augr.cost import CostModel, budget_penalty, aperture_from_fwhm

    cost = CostModel()                              # placeholder, ~PICO-scale defaults
    c = cost.total_cost(aperture_m=1.5, n_det_total=12000.0, mission_years=5.0)
    pen = budget_penalty(c, budget=1000.0, weight=1e-3)   # add to a sigma(r) objective

    # Connect the beam-design knob to an aperture cost:
    aperture_m = aperture_from_fwhm(fwhm_arcmin=20.0, nu_ghz=220.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from augr.telescope import beam_fwhm_arcmin


@dataclass(frozen=True)
class CostModel:
    r"""Convex placeholder cost model [\$M] over the differentiable design knobs.

    .. math::

        \mathrm{cost} = \mathrm{base}
            + a_\mathrm{ap}\, D^{\,p_\mathrm{ap}}
            + b_\mathrm{det}\, N_\mathrm{det}
            + c_\mathrm{yr}\, T_\mathrm{mission}

    where :math:`D` is the aperture [m], :math:`N_\mathrm{det}` the total detector
    count, and :math:`T_\mathrm{mission}` the mission duration [yr]. The function is
    jointly **convex** in all three arguments for ``p_aperture >= 1`` and non-negative
    coefficients, so a budget-constrained design optimization has an interior optimum
    (a free FWHM / NET / detector count no longer runs to the boundary).

    The defaults are an **illustrative placeholder** loosely anchored to a PICO-scale
    ~\$1B probe (aperture ~1.4 m, ~12000 detectors, ~5 yr): optics ~\$400M (``D**2``,
    i.e. collecting-area scaling), detectors/readout ~\$350M, ops ~\$250M. They set
    only the *scale* so the interior optimum lands in a sane region -- replace them with
    the JPL study's cost relation when it is available. A steeper optics exponent
    (``p_aperture`` 2.5-3, common for deployable/space optics) stays convex.
    """

    a_aperture: float = 204.0  # $M per m**p_aperture  (~$400M at D=1.4 m, p=2)
    p_aperture: float = 2.0  # aperture cost exponent; >= 1 for convexity
    b_detector: float = 2.92e-2  # $M per detector  (~$29k/detector, ~$350M at 12000)
    c_year: float = 50.0  # $M per year  (~$250M at 5 yr)
    base: float = 0.0  # fixed bus / integration cost [$M]

    def total_cost(self, aperture_m, n_det_total, mission_years):
        """Total mission cost [$M] from the design vector. Differentiable in all args."""
        aperture_m = jnp.asarray(aperture_m)
        n_det_total = jnp.asarray(n_det_total)
        mission_years = jnp.asarray(mission_years)
        return (
            self.base
            + self.a_aperture * aperture_m**self.p_aperture
            + self.b_detector * n_det_total
            + self.c_year * mission_years
        )


def budget_penalty(cost, budget, weight: float = 1.0):
    """One-sided soft budget penalty ``weight * max(cost - budget, 0)**2``.

    Zero strictly under budget, smoothly (C1) rising above it -- add it to a sigma(r)
    objective (or subtract from an EIG objective) so gradient descent stays on the
    affordable side of the budget surface. The gradient is ``0`` below budget and
    ``2 * weight * (cost - budget)`` above, so it never pushes a design that is already
    affordable. Convex in ``cost``.

    Args:
        cost:   Realized cost [$M] (e.g. ``CostModel.total_cost(...)``).
        budget: Budget cap [$M].
        weight: Penalty stiffness. Tune against the sigma(r) scale so the wall is firm
                but does not dominate the curvature of the objective.

    Returns:
        Scalar penalty [same units as ``weight``].
    """
    over = jnp.maximum(jnp.asarray(cost) - budget, 0.0)
    return weight * over**2


def total_error_r(sigma_r, delta_r, *, bias_w: float = 1.0):
    """Effective error on r with the bias folded in: ``sqrt(sigma^2 + bias_w*Delta r^2)``.

    The r-marginal EIG is ``log(sigma_prior/sigma(r))`` -- it carries **no bias term at
    all**, so on its own it will buy a design with a small ``sigma(r)`` and a large
    ``Delta r`` and nothing in the fit flags it. That is not hypothetical: the
    mode-disjoint ILC campaign measured foreground-destruction biases of -0.68 / -1.81 /
    -5.22 sigma while the residual amplitude ``A_res`` was itself unmeasurable.

    This is the standard quadrature error budget -- statistical and systematic added in
    quadrature -- and it is deliberately preferred to a one-sided "wall" on the bias in
    units of sigma(r), i.e. ``max(|Delta r|/sigma - eps, 0)**2``. That form has a
    perverse stationary point: it *decreases* as sigma(r) grows, so a design can satisfy
    it by making the error bar worse rather than the bias smaller. Against a utility of
    ``-log sigma`` the balance sits at ``2w*u(u - eps) = 1`` for ``u = |Delta r|/sigma``
    -- a local minimum at ``u = 1`` for ``w = 1, eps = 0.5``, reachable by inflating
    sigma. Nothing here decreases in sigma, so that mechanism is absent:
    ``d/dsigma = w1*sigma/sqrt(.) > 0`` and ``d/dDelta r = w2*Delta r/sqrt(.) > 0``
    always. Smaller is better on both axes.

    **Only the ratio of the two weights matters.** A general ``w1*sigma^2 + w2*dr^2``
    factorizes as ``w1*(sigma^2 + (w2/w1)*dr^2)``, and consumers take a logarithm, so
    ``w1`` is a design-independent additive constant with zero gradient. ``bias_w`` is
    that ratio: ``1.0`` weights a unit of bias exactly like a unit of statistical error
    (plain total error), larger values weight bias more.

    Both terms are in ``r^2``, so ``bias_w`` is dimensionless and O(1) -- unlike a
    penalty added to an EIG in nats, which needs a stiffness of ~1e8 at a PICO-class
    sigma(r) before it influences anything.

    Args:
        sigma_r:  Marginalized sigma(r) at this design.
        delta_r:  Bias on r at this design (sign carried; only the square is used).
        bias_w:   Weight on the bias term relative to the variance term.

    Returns:
        Effective 1-sigma error on r, in the same units as ``sigma_r``.
    """
    sigma = jnp.asarray(sigma_r)
    dr = jnp.asarray(delta_r)
    return jnp.sqrt(sigma**2 + bias_w * dr**2)


def aperture_from_fwhm(fwhm_arcmin, nu_ghz: float, illumination_factor: float = 1.22):
    """Aperture diameter [m] implied by a diffraction-limited beam FWHM.

    Inverts :func:`augr.telescope.beam_fwhm_arcmin` (``FWHM = K(nu) / D``): evaluating
    the public relation at ``D = 1 m`` returns ``K(nu)``, so ``D = K(nu) / FWHM``. This
    connects the per-band beam-design knob (``sigma_r_from_beam_design``'s free FWHM) to
    an aperture cost; differentiable in ``fwhm_arcmin``.

    Note a single dish has *one* physical aperture, while the beam optimizer treats the
    per-band FWHMs as independent (optics-agnostic) stand-ins. Reconciling per-band free
    FWHMs with a single aperture (e.g. cost off the tightest-beam / highest-frequency
    band, or carry an explicit aperture design scalar) is a modeling choice deferred to
    the design-optimization (B.3) work; this helper is the single-band connector.

    Args:
        fwhm_arcmin: Beam FWHM [arcmin] (scalar or array).
        nu_ghz:      Band frequency [GHz].
        illumination_factor: Multiplier on lambda/D (1.22 for an Airy disk), matching
                     :func:`augr.telescope.beam_fwhm_arcmin`.

    Returns:
        Aperture diameter [m], same shape as ``fwhm_arcmin``.
    """
    k = beam_fwhm_arcmin(nu_ghz, 1.0, illumination_factor)  # = FWHM at D = 1 m
    return k / jnp.asarray(fwhm_arcmin)
