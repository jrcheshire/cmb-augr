"""
eig.py -- Expected Information Gain (EIG) objectives for Bayesian-optimal design.

The capstone of the differentiable map-based forward: choose the instrument design xi
to maximize the information a forecast carries about r, under a cost budget. This
module supplies the EIG utilities and the cost-constrained design objective; the
non-Gaussian Hamimeche-Lewis EIG (the science novelty) is designed here but deferred
to Stage 2 (see :func:`design_objective` ``objective="hl_eig"`` and the recipe below).

**Gaussian EIG, two flavors, off one posterior Fisher.** Every utility reads a scalar
off ``F_post = J^T Sigma_hat(xi)^-1 J + diag(prior)`` -- the same posterior Fisher
that :func:`augr.optimize.sigma_r_from_external_cov` builds (via
``fisher._fisher_from_full``), differentiable in the covariance:

* **r-marginal EIG** (:func:`marginal_eig_r_from_external_cov`) --
  ``log(sigma_prior_r / sigma_post(r))``. Since ``sigma_post(r) = sqrt((F_post^-1)_rr)``
  is exactly what ``sigma_r_from_external_cov`` returns, the r-marginal Gaussian EIG is
  ``-log sigma(r) + const`` -- maximizing it is *identical* to minimizing sigma(r). The
  ``sigma_prior_r`` only sets the additive offset, **not** the optimum (its design
  gradient is zero), so the EIG-optimal design equals the sigma(r)-optimal design. This
  is the primary objective: the science QoI is r.
* **D-optimal EIG** (:func:`gaussian_eig_from_external_cov`) --
  ``0.5 * logdet(F_post)``, the joint information on the full parameter vector
  (r + nuisances). Exposed as a secondary/general BOED objective; it rewards nuisance
  information (e.g. pinning ``A_res``), so it is *not* focused on r.

**The cost-constrained loop is the new capability.** The prior sigma(r) optimization had
no budget and ran a free FWHM/NET to the boundary; :func:`design_objective` adds the
convex cost (:mod:`augr.cost`) as a soft budget wall so the optimum is interior.

**Structural gift (makes Stage-2 HL-EIG tractable).** ``Sigma_hat(xi)`` is
design-dependent but cosmology-parameter-independent, so one map forward per design
yields ``(Sigma_hat, N_b)`` and the cheap analytic ``data_vector(theta)`` carries the
parameter dependence -- there is **no map forward inside any EIG/sampling inner loop**.

**Stage-2 HL-EIG (``objective="hl_eig"``) -- value-only.** The non-Gaussian
Hamimeche-Lewis r-marginal EIG, estimated by nested Monte Carlo over the cheap analytic
HL likelihood (no map forward in the inner loop). Per design xi, one map forward gives
``(Sigma_hat, mean_bandpower)``; :func:`hl_eig_from_external_cov` then estimates the
r-marginal mutual information

    ``EIG_r = E_{d ~ p(d)} [ KL( P_HL(r | d) || prior(r) ) ]``

(the Rao-Blackwellized form of the recipe's ``E[log p(d|r) - log p_marg(d)]`` with the
nuisances ``eta=(A_lens, A_res)`` marginalized). The nuisance + r grid quadrature reuses
:mod:`augr.sbc` (``make_marginal_logpost`` gives the ``eta``-marginalized per-bin HL/Gaussian
log-likelihood on the grid); the outer data draws are reparametrized
``d = mean(theta) + chol(Sigma_hat) @ z`` so they track ``Sigma_hat(xi)``. The Gaussian
column of the same draws gives a CRN-paired Gaussian EIG for free (the validation handle).

This path is **value-only**: ``HLLikelihood.from_external`` inverts the covariance on host
(``np.linalg.inv``), so ``jax.grad`` through it is not supported. The design subspace
(:mod:`augr.active_subspace`) is built from the cheap Gaussian-EIG gradient and only
*evaluates* HL-EIG along the reduced axes, so no HL-EIG design gradient is needed. The
differentiable HL-EIG path (a traced marginalizer + compounded-MC gradient variance) is
deferred. **Caveat:** the HL ``log_prob`` carries no per-theta ``X_g`` Jacobian / ``logdet``
term, so the estimator is exact only in the high-mode limit (an information *proxy* off it) --
it is the same likelihood object used everywhere else for inference; gate 4a bounds the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from augr.cleaning import Cleaner
from augr.cost import CostModel, aperture_from_fwhm, budget_penalty
from augr.fisher import _fisher_from_full
from augr.optimize import OptimizationContext, design_to_channels
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.spectrum_stages import CutskyMCContext, mc_cutsky_cov_traced

_OBJECTIVES = ("marginal_eig_r", "d_optimal", "sigma_r", "hl_eig")


def posterior_fisher_from_external_cov(cov, ctx: OptimizationContext):
    """Posterior Fisher ``F = J^T cov^-1 J + diag(prior)`` (jnp, differentiable in ``cov``).

    The same prewhitened dense solve as
    :func:`augr.optimize.sigma_r_from_external_cov` /
    ``FisherForecast(external_covariance=...).compute()``; factored out so the EIG
    utilities share it. ``ctx`` supplies the structural Jacobian ``J`` and the prior
    diagonal ``1/sigma_prior^2``.
    """
    F = _fisher_from_full(ctx.J, jnp.asarray(cov))
    return F + jnp.diag(ctx.prior_diag)


def marginal_eig_r_from_external_cov(
    cov, ctx: OptimizationContext, *, sigma_prior_r: float = 1.0
):
    """r-marginal Gaussian EIG ``log(sigma_prior_r / sigma_post(r))`` -- the primary objective.

    Differentiable in ``cov``; consistent by construction with
    :func:`augr.optimize.sigma_r_from_external_cov` (same ``F_post``, same
    ``(F^-1)_rr``). ``sigma_prior_r`` sets only the additive offset -- its design
    gradient is zero, so the EIG-optimal design equals the sigma(r)-optimal design.
    """
    F = posterior_fisher_from_external_cov(cov, ctx)
    sigma_r = jnp.sqrt(jnp.linalg.inv(F)[ctx.r_idx, ctx.r_idx])
    return jnp.log(sigma_prior_r) - jnp.log(sigma_r)


def gaussian_eig_from_external_cov(
    cov, ctx: OptimizationContext, *, prior_fisher_logdet=None
):
    """D-optimal Gaussian EIG ``0.5 * logdet(F_post)`` -- the secondary/general objective.

    Joint information on the full parameter vector. With ``prior_fisher_logdet`` supplied
    (proper priors), returns the calibrated ``0.5 * (logdet F_post - logdet F_prior)``;
    with improper priors (some ``prior_diag`` are zero) ``logdet F_prior`` is undefined,
    but the design-dependent part ``0.5 * logdet(F_post)`` -- and hence the optimization
    gradient -- is well-defined, so the absolute EIG is offset-only.
    """
    F = posterior_fisher_from_external_cov(cov, ctx)
    _sign, logdet = jnp.linalg.slogdet(F)
    eig = 0.5 * logdet
    if prior_fisher_logdet is not None:
        eig = eig - 0.5 * prior_fisher_logdet
    return eig


def _trapezoid_log_weights(grid: np.ndarray) -> np.ndarray:
    """Log of the trapezoid quadrature weights for a 1-D ``grid`` (numpy)."""
    diffs = np.diff(grid)
    w = np.empty_like(grid, dtype=float)
    w[0] = diffs[0] / 2.0
    w[-1] = diffs[-1] / 2.0
    w[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    return np.log(w)


@dataclass(frozen=True)
class HLEIGContext:
    """Static grid + linear-basis pieces for the HL-EIG estimator (one per signal model).

    Design-independent: built once from the cleaned-map residual-template signal model and
    the (r, A_lens, A_res) prior, then reused across every design in a scan. The
    design-dependent ``(Sigma_hat, mean_bandpower)`` enter :func:`hl_eig_from_external_cov`
    at call time, not here. Mirrors how :class:`augr.optimize.OptimizationContext` packs the
    static forecast pieces.
    """

    signal_model: object
    fid_vec: jax.Array  # (n_data,) fiducial param vector, parameter_names order
    layout: object  # SpectrumLayout
    nuis: object  # augr.sbc.NuisanceGrid
    pred_flat: jax.Array  # (n_grid * n_al * n_ar, n_bins) linear prediction grid
    base: jax.Array  # (n_bins,) data_vector(r=0, A_lens=0, A_res=0)
    t_r: jax.Array  # (n_bins,) d data_vector / d r
    t_l: jax.Array  # (n_bins,) d data_vector / d A_lens
    t_res: jax.Array  # (n_bins,) d data_vector / d A_res
    r_grid: np.ndarray  # (n_grid,)
    logprior_r: jax.Array  # (n_grid,) Gaussian r log-prior (unnormalized)
    log_w: jax.Array  # (n_grid,) trapezoid log quadrature weights
    sigma_prior_r: float
    r_fid: float
    floated: frozenset
    prior_sig: dict

    @property
    def n_grid(self) -> int:
        return int(self.r_grid.size)

    @classmethod
    def build(
        cls,
        *,
        template_ells,
        template_cl,
        f_sky: float,
        r_fid: float = 0.0,
        floated=frozenset({"A_lens", "A_res"}),
        prior_sig=None,
        sigma_prior_r: float = 1.0,
        n_grid: int = 400,
        n_sigma_grid: float = 6.0,
        n_nuis_grid: int = 41,
        n_sigma_nuis: float = 5.0,
        ell_min: int = 2,
        ell_max: int = 180,
        delta_ell: int = 5,
        ell_per_bin_below: int = 30,
    ) -> HLEIGContext:
        """Pack the sbc marginalization grid + linear basis for the HL-EIG estimator.

        ``sigma_prior_r`` is the Gaussian r-prior 1sigma (sets the data distribution the
        outer MC averages over and the KL reference). The r-grid spans
        ``r_fid +/- n_sigma_grid * sigma_prior_r`` (so it contains the prior draws); use a
        weak prior (``sigma_prior_r`` a few x the expected sigma(r)) so the grid resolves the
        posterior. ``prior_sig`` defaults to ``{"A_lens": 0.25, "A_res": 0.3}`` (augr's
        nuisance priors).

        **Resolution discipline.** The EIG magnitude (a KL, unlike coverage's PIT) is
        sensitive to grid resolution: the nuisance grid spacing must be ``<<`` the nuisance
        *posterior* width, or the marginal ``p(d|r)`` -- and hence the EIG -- is biased. The
        uniform ``+/- n_sigma_nuis * prior`` grid resolves a nuisance only when its posterior
        is not much narrower than its prior; a tightly-constrained nuisance (e.g. ``A_lens``
        in non-delensed mode) needs a large ``n_nuis_grid``. Prefer delensed mode (``A_lens``
        removed) + floating ``A_res`` (degenerate with r -> wide posterior). Always confirm
        convergence by doubling ``n_nuis_grid`` (the gate in ``tests/test_hl_eig.py``).
        """
        from augr import sbc
        from augr.likelihood.from_cutsky import build_cutsky_signal_model
        from augr.likelihood.ordering import SpectrumLayout
        from augr.signal import flatten_params

        if prior_sig is None:
            prior_sig = {"A_lens": 0.25, "A_res": 0.3}
        floated = frozenset(floated)

        signal_model = build_cutsky_signal_model(
            template_ells,
            template_cl,
            f_sky,
            ell_min=ell_min,
            ell_max=ell_max,
            delta_ell=delta_ell,
            ell_per_bin_below=ell_per_bin_below,
        )[0]
        names = list(signal_model.parameter_names)
        fid_vec = flatten_params({"r": r_fid, "A_lens": 1.0, "A_res": 1.0}, names)
        layout = SpectrumLayout.from_freq_pairs(
            signal_model.freq_pairs, signal_model.n_bins
        )
        base, t_r, t_l, t_res = (
            jnp.asarray(x) for x in sbc.linear_basis(signal_model, names)
        )

        nuis = sbc.NuisanceGrid.build(
            floated=floated,
            prior_sig=prior_sig,
            n_nuis_grid=n_nuis_grid,
            n_sigma_nuis=n_sigma_nuis,
        )
        r_grid = np.linspace(
            r_fid - n_sigma_grid * sigma_prior_r,
            r_fid + n_sigma_grid * sigma_prior_r,
            n_grid,
        )
        pred_flat = sbc.build_pred_grid(
            np.asarray(base),
            np.asarray(t_r),
            np.asarray(t_l),
            np.asarray(t_res),
            r_grid=r_grid,
            al_axis=nuis.al_axis,
            ares_axis=nuis.ares_axis,
        )
        logprior_r = jnp.asarray(-0.5 * ((r_grid - r_fid) / sigma_prior_r) ** 2)
        log_w = jnp.asarray(_trapezoid_log_weights(r_grid))
        return cls(
            signal_model=signal_model,
            fid_vec=fid_vec,
            layout=layout,
            nuis=nuis,
            pred_flat=pred_flat,
            base=base,
            t_r=t_r,
            t_l=t_l,
            t_res=t_res,
            r_grid=r_grid,
            logprior_r=logprior_r,
            log_w=log_w,
            sigma_prior_r=float(sigma_prior_r),
            r_fid=float(r_fid),
            floated=floated,
            prior_sig=dict(prior_sig),
        )


@dataclass(frozen=True)
class HLEIGResult:
    """HL-EIG estimate + the CRN-paired Gaussian EIG and MC/grid diagnostics."""

    eig_hl: float
    stderr_hl: float
    eig_gauss: float
    stderr_gauss: float
    edge_frac: (
        float  # mean posterior mass in the outermost grid points (grid-width check)
    )


def _grid_eig(marg_col, logprior_r, log_w):
    """``E_n[ KL(P(r|d_n) || prior(r)) ]`` from per-draw grid log-likelihoods.

    ``marg_col`` ``(n_outer, n_grid)`` is the eta-marginalized ``log p(d_n | r_i)``;
    ``logprior_r`` / ``log_w`` are the ``(n_grid,)`` Gaussian r log-prior and trapezoid
    log-weights. Returns ``(eig_mean, eig_stderr, kl_per_draw)``. All logsumexp-stable.
    """
    la = marg_col + logprior_r[None, :]  # log unnormalized r-posterior per draw
    log_z_post = logsumexp(la + log_w[None, :], axis=1)  # (n_outer,)
    log_p = la - log_z_post[:, None]
    log_z_prior = logsumexp(logprior_r + log_w)
    log_q = logprior_r - log_z_prior  # (n_grid,) normalized log-prior
    kl = jnp.sum(jnp.exp(log_w[None, :] + log_p) * (log_p - log_q[None, :]), axis=1)
    return jnp.mean(kl), jnp.std(kl) / jnp.sqrt(kl.shape[0]), kl


def hl_eig_from_external_cov(
    cov,
    mean_bandpower,
    ctx: HLEIGContext,
    *,
    key,
    n_outer: int = 256,
    return_diagnostics: bool = False,
):
    """r-marginal HL EIG ``E_d[ KL(P_HL(r|d) || prior_r) ]`` -- nested MC + grid quadrature.

    Value-only (``cov`` must be concrete; ``HLLikelihood.from_external`` inverts on host).
    Builds the HL + Gaussian likelihoods at ``mean_bandpower``/``cov``, draws ``n_outer``
    reparametrized totals ``d = mean(theta) + chol(cov) @ z`` with ``theta ~ prior``, runs
    the sbc nuisance-marginalizer per draw, and KL-reduces the HL r-posterior against the
    r-prior. Returns the scalar HL EIG (to be MAXIMIZED), or an :class:`HLEIGResult` with the
    CRN-paired Gaussian EIG + diagnostics when ``return_diagnostics``.
    """
    from augr import sbc
    from augr.likelihood.gaussian import GaussianLikelihood
    from augr.likelihood.hl import HLLikelihood

    cov = jnp.asarray(cov)
    mean_bp = jnp.asarray(mean_bandpower)
    sm = ctx.signal_model
    n_b = mean_bp - sm.data_vector(ctx.fid_vec)

    hl0 = HLLikelihood.from_external(sm, ctx.fid_vec, mean_bp, cov)
    gauss0 = GaussianLikelihood.from_external(sm, ctx.fid_vec, cov)
    core = sbc.make_marginal_logpost(
        gauss0=gauss0,
        hl0=hl0,
        noise_floor=n_b,
        layout=ctx.layout,
        pred_flat=ctx.pred_flat,
        logprior_grid=ctx.nuis.logprior_grid,
        n_grid=ctx.n_grid,
        n_al=ctx.nuis.n_al,
        n_ar=ctx.nuis.n_ar,
    )

    # Outer MC: theta ~ prior (centred at fiducial), d = mean(theta) + chol(cov) z. Because
    # the model is linear, mean(theta) = mean_bandpower + dr t_r + d_al t_l + d_ar t_res with
    # the offsets dr = r - r_fid etc. drawn from the (zero-centred) prior.
    k_r, k_al, k_ar, k_z = jax.random.split(key, 4)
    n_bins = mean_bp.shape[0]
    dr = ctx.sigma_prior_r * jax.random.normal(k_r, (n_outer,))
    d_al = (
        ctx.prior_sig["A_lens"] * jax.random.normal(k_al, (n_outer,))
        if "A_lens" in ctx.floated
        else jnp.zeros((n_outer,))
    )
    d_ar = (
        ctx.prior_sig["A_res"] * jax.random.normal(k_ar, (n_outer,))
        if "A_res" in ctx.floated
        else jnp.zeros((n_outer,))
    )
    mean_n = (
        mean_bp[None, :]
        + dr[:, None] * ctx.t_r[None, :]
        + d_al[:, None] * ctx.t_l[None, :]
        + d_ar[:, None] * ctx.t_res[None, :]
    )
    chol = jnp.linalg.cholesky(cov)
    z = jax.random.normal(k_z, (n_outer, n_bins))
    draws = mean_n + z @ chol.T

    # lax.map (not vmap): scan the draws sequentially at one-draw memory -- vmap would
    # materialize the full (n_grid * n_al * n_ar) nuisance grid for every draw at once.
    marg = jax.lax.map(core, draws)  # (n_outer, n_grid, 2): columns [Gaussian, HL]
    eig_hl, se_hl, _ = _grid_eig(marg[:, :, 1], ctx.logprior_r, ctx.log_w)
    if not return_diagnostics:
        return eig_hl

    eig_g, se_g, _ = _grid_eig(marg[:, :, 0], ctx.logprior_r, ctx.log_w)
    # Posterior mass piled at the grid edges => the r-grid is too narrow (KL truncated).
    la = marg[:, :, 1] + ctx.logprior_r[None, :]
    log_p = la - logsumexp(la + ctx.log_w[None, :], axis=1)[:, None]
    p_mass = jnp.exp(ctx.log_w[None, :] + log_p)
    edge_frac = float(jnp.mean(p_mass[:, 0] + p_mass[:, -1]))
    return HLEIGResult(
        eig_hl=float(eig_hl),
        stderr_hl=float(se_hl),
        eig_gauss=float(eig_g),
        stderr_gauss=float(se_g),
        edge_frac=edge_frac,
    )


def _utility(
    cov,
    ctx,
    objective,
    sigma_prior_r,
    *,
    mean_bandpower=None,
    hl_eig_ctx=None,
    eig_key=None,
    n_outer: int = 256,
):
    """EIG utility (to be MAXIMIZED) for the named ``objective``."""
    if objective == "marginal_eig_r":
        return marginal_eig_r_from_external_cov(cov, ctx, sigma_prior_r=sigma_prior_r)
    if objective == "d_optimal":
        return gaussian_eig_from_external_cov(cov, ctx)
    if objective == "sigma_r":
        # Utility = -sigma(r), so maximizing utility == minimizing sigma(r). Lets the
        # design driver reuse one code path to reproduce the sigma(r) optimum.
        F = posterior_fisher_from_external_cov(cov, ctx)
        return -jnp.sqrt(jnp.linalg.inv(F)[ctx.r_idx, ctx.r_idx])
    if objective == "hl_eig":
        if hl_eig_ctx is None or mean_bandpower is None or eig_key is None:
            raise ValueError(
                "objective='hl_eig' requires hl_eig_ctx, mean_bandpower, and eig_key "
                "(value-only; design gradient unsupported -- see the augr.eig docstring)."
            )
        return hl_eig_from_external_cov(
            cov, mean_bandpower, hl_eig_ctx, key=eig_key, n_outer=n_outer
        )
    raise ValueError(f"unknown objective {objective!r}; expected one of {_OBJECTIVES}.")


def design_cost(n_det, beam_fwhm, mission_years, *, cost_model: CostModel, freqs_ghz):
    """Mission cost [$M] from the design knobs (no map forward) -- differentiable.

    The aperture is taken from the **tightest-beam (highest-frequency) band**, the one
    that sets the diffraction limit of the single dish (``aperture_from_fwhm``); the
    detector cost from ``sum(n_det)``; the ops cost from ``mission_years``. Reconciling
    per-band free FWHMs with a single physical aperture is a modeling choice deferred to
    a real optics model (see :func:`augr.cost.aperture_from_fwhm`).
    """
    freqs = tuple(float(f) for f in freqs_ghz)
    i_tight = int(np.argmax(freqs))  # static: freqs are concrete
    aperture = aperture_from_fwhm(beam_fwhm[i_tight], freqs[i_tight])
    n_det_total = jnp.sum(jnp.asarray(n_det))
    return cost_model.total_cost(aperture, n_det_total, mission_years)


def design_objective(
    n_det,
    net,
    eta_total,
    mission_years,
    beam_fwhm,
    beam_p,
    *,
    mc_ctx: CutskyMCContext,
    opt_ctx: OptimizationContext,
    cleaner: Cleaner,
    cost_model: CostModel,
    budget: float,
    freqs_ghz,
    penalty_weight: float = 1.0,
    objective: str = "marginal_eig_r",
    sigma_prior_r: float = 1.0,
    hl_eig_ctx: HLEIGContext | None = None,
    eig_key=None,
    n_outer: int = 256,
):
    """Cost-constrained Bayesian-design objective to MINIMIZE.

    Composes the noise + beam design into ``w_inv`` and per-band beams, runs the cut-sky
    masked-Wiener Monte-Carlo compsep forward
    (:func:`augr.spectrum_stages.mc_cutsky_cov_traced`) to ``Sigma_hat``, evaluates the
    EIG ``objective`` (default r-marginal), and adds the convex budget penalty
    (:func:`augr.cost.budget_penalty`).

    For the Gaussian objectives (``marginal_eig_r`` / ``d_optimal`` / ``sigma_r``) this is
    ``jax.grad``-able via ``jax.grad(design_objective, argnums=(0, 1, 2, 3, 4, 5))`` -> the
    joint noise+beam design gradient (``mc_ctx`` / ``opt_ctx`` / ``cleaner`` / ``cost_model``
    static). ``objective="hl_eig"`` is **value-only** (requires ``hl_eig_ctx`` + ``eig_key``;
    the HL covariance inverse is computed on host, so the trace breaks under ``jax.grad``) --
    the design subspace is built from the cheap Gaussian gradient and only evaluates HL-EIG.

    Returns ``-utility + budget_penalty`` (a minimization scalar): descent maximizes EIG
    while staying on the affordable side of the budget surface.
    """
    w_inv = w_inv_from_noise_design(n_det, net, eta_total, mission_years, mc_ctx.f_sky)
    traced = mc_cutsky_cov_traced(
        w_inv, mc_ctx, cleaner, beam_fwhm=beam_fwhm, beam_p=beam_p
    )
    util = _utility(
        traced.covariance,
        opt_ctx,
        objective,
        sigma_prior_r,
        mean_bandpower=traced.mean_bandpower,
        hl_eig_ctx=hl_eig_ctx,
        eig_key=eig_key,
        n_outer=n_outer,
    )
    cost = design_cost(
        n_det, beam_fwhm, mission_years, cost_model=cost_model, freqs_ghz=freqs_ghz
    )
    return -util + budget_penalty(cost, budget, penalty_weight)


def physical_design_objective(
    design,
    *,
    freqs_per_group,
    fp_diameter_m,
    mc_ctx: CutskyMCContext,
    opt_ctx: OptimizationContext,
    cleaner: Cleaner,
    cost_model: CostModel,
    budget: float,
    eta_total=0.5,
    beam_p=None,
    illumination_factor: float = 1.22,
    packing_efficiency: float = 0.80,
    net_override=None,
    penalty_weight: float = 1.0,
    objective: str = "marginal_eig_r",
    sigma_prior_r: float = 1.0,
    hl_eig_ctx: HLEIGContext | None = None,
    eig_key=None,
    n_outer: int = 256,
):
    """Cost-constrained EIG objective from the physical horn-packing design knobs.

    ``design`` is the dict produced by
    :meth:`augr.design_packing.PackingDesignSpec.design_pytree`:
    ``{aperture_m, f_number, area_fractions, mission_years}``. Derives per-channel
    ``(n_det, net, beam)`` via :func:`augr.optimize.design_to_channels` (focal-plane
    packing, NET from photon noise, beam from the single physical aperture) and forwards
    to :func:`design_objective` -- the same cut-sky MC forward + EIG utility + budget
    penalty.

    Returns ``-utility + budget_penalty`` (a minimization scalar). For the Gaussian
    objectives, ``jax.grad`` w.r.t. the standardized ``z`` flows
    ``z -> design_pytree -> design_to_channels -> design_objective``;
    ``objective="hl_eig"`` is value-only (see the module docstring).
    """
    n_det, net, beam = design_to_channels(
        design["aperture_m"],
        design["f_number"],
        fp_diameter_m,
        design["area_fractions"],
        freqs_per_group,
        net_override=net_override,
        illumination_factor=illumination_factor,
        packing_efficiency=packing_efficiency,
    )
    freqs_flat = tuple(float(f) for grp in freqs_per_group for f in grp)
    n_chan = len(freqs_flat)
    if beam_p is None:
        beam_p = jnp.ones(n_chan)
    eta_arr = (
        jnp.full((n_chan,), eta_total)
        if jnp.ndim(eta_total) == 0
        else jnp.asarray(eta_total)
    )
    return design_objective(
        n_det,
        net,
        eta_arr,
        design["mission_years"],
        beam,
        beam_p,
        mc_ctx=mc_ctx,
        opt_ctx=opt_ctx,
        cleaner=cleaner,
        cost_model=cost_model,
        budget=budget,
        freqs_ghz=freqs_flat,
        penalty_weight=penalty_weight,
        objective=objective,
        sigma_prior_r=sigma_prior_r,
        hl_eig_ctx=hl_eig_ctx,
        eig_key=eig_key,
        n_outer=n_outer,
    )
