"""eig_design_demo.py -- cost-constrained Bayesian design optimization (Gaussian EIG).

A *methods demonstration* on the tiny CMB-only config: maximize the r-marginal Expected
Information Gain (``EIG_r = -log sigma(r) + const``) over a noise + beam design under a
$ budget, via ``augr.eig.design_objective`` + ``augr.cost``. It shows the three things
the scaffolding delivers:

* **The budget turns a run-to-the-boundary descent into an interior optimum.** Without a
  cost (huge budget) the design drives detectors / mission / aperture up without limit
  (sigma(r) keeps shrinking); with a real budget the optimum sits at the budget wall.
* **EIG_r improves** vs the fiducial design under the same budget.
* **The EIG_r optimum coincides with the sigma(r) optimum** (the Gaussian equivalence:
  r-marginal EIG is a monotone reparametrization of sigma(r)).

The design is parametrized by three log-scales applied to the fiducial design --
detector count, mission years, and beam FWHM -- the cost-bearing knobs (NET / efficiency
have no cost term, so they are held fixed: optimizing them is the runaway the budget does
*not* fix). Tiny (nside=16, CMB-only, no PySM) so it runs on a laptop CPU in a few
minutes; pass ``--backend jht`` to run the transforms on a GPU.

Run:  pixi run python scripts/eig_design_demo.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from augr import masking as mk
from augr import sht
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.cost import CostModel
from augr.delensing import load_lensing_spectra
from augr.eig import design_cost, design_objective, marginal_eig_r_from_external_cov
from augr.foregrounds import NullForegroundModel
from augr.optimize import make_optimization_context
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context, mc_cutsky_cov_traced

# Fiducial 3-band design (cost-free NET / efficiency held fixed).
FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)
N_DET = (200.0, 400.0, 200.0)
NET = (60.0, 50.0, 80.0)
ETA = (0.5, 0.5, 0.5)
MISSION_YEARS = 4.0
F_SKY = 0.6


def build_contexts(n_sims, *, nside, lmax):
    """mc_ctx + opt_ctx + cleaner for the tiny CMB-only config."""
    ell_max, delta_ell, ell_per_bin_below = lmax, 8, 2
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=F_SKY),
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    bm = jnp.asarray(sm.bin_matrix)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
    )
    cleaner = nilc_cleaner(clean_e=True)
    w_inv_fid = np.asarray(
        w_inv_from_noise_design(
            jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, F_SKY
        )
    )
    mc_ctx = make_cutsky_mc_context(
        cleaner=cleaner,
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=w_inv_fid,
        nside=nside,
        lmax=lmax,
        mask=mk.galactic_mask(nside, F_SKY),
        cl_ee=cl_ee,
        cl_bb_prior_unbeamed=cl_bb,
        bin_matrix=bm,
        ell_min=2,
        true_bb_binned=true_b,
        n_sims=n_sims,
        base_seed=0,
        fg_model=None,
        r_in=0.0,
    )
    opt_ctx = make_optimization_context(
        cleaned_map_instrument(f_sky=F_SKY),
        NullForegroundModel(),
        CMBSpectra(),
        {"r": 0.0, "A_lens": 1.0},
        priors={},
        fixed_params=[],
        ell_min=2,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    return mc_ctx, opt_ctx, cleaner


def _design_from_theta(theta):
    """Three log-scales -> (n_det, net, eta, mission_years, beam_fwhm, beam_p)."""
    s_ndet, s_mission, s_beam = jnp.exp(theta)
    return (
        jnp.asarray(N_DET) * s_ndet,
        jnp.asarray(NET),
        jnp.asarray(ETA),
        MISSION_YEARS * s_mission,
        jnp.asarray(BEAMS) * s_beam,
        jnp.ones(len(BEAMS)),
    )


def _metrics(theta, mc_ctx, opt_ctx, cleaner, cost_model):
    """(EIG_r, sigma(r), cost) at a design — the readable diagnostics, not the objective."""
    n_det, net, eta, yr, bf, bp = _design_from_theta(theta)
    w_inv = w_inv_from_noise_design(n_det, net, eta, yr, mc_ctx.f_sky)
    cov = mc_cutsky_cov_traced(w_inv, mc_ctx, cleaner, beam_fwhm=bf, beam_p=bp).covariance
    eig = float(marginal_eig_r_from_external_cov(cov, opt_ctx, sigma_prior_r=1.0))
    cost = float(design_cost(n_det, bf, yr, cost_model=cost_model, freqs_ghz=FREQS))
    return eig, float(np.exp(-eig)), cost  # sigma(r) = exp(-EIG_r) since sigma_prior=1


def _optimize(mc_ctx, opt_ctx, cleaner, cost_model, budget, *, objective, penalty_weight, maxiter):
    def loss(theta):
        n_det, net, eta, yr, bf, bp = _design_from_theta(jnp.asarray(theta))
        return design_objective(
            n_det,
            net,
            eta,
            yr,
            bf,
            bp,
            mc_ctx=mc_ctx,
            opt_ctx=opt_ctx,
            cleaner=cleaner,
            cost_model=cost_model,
            budget=budget,
            freqs_ghz=FREQS,
            penalty_weight=penalty_weight,
            objective=objective,
            sigma_prior_r=1.0,
        )

    vg = jax.value_and_grad(loss)

    def scipy_vg(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(
        scipy_vg,
        np.zeros(3),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxiter, "ftol": 1e-9, "gtol": 1e-7},
    )
    return res.x


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--n-sims",
        type=int,
        default=8,
        help="MC sims for the bandpower covariance; must exceed (#bandpower bins + 2) "
        "for the Hartlap correction (default binning has 3 bins, so >=6).",
    )
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--lmax", type=int, default=24)
    p.add_argument("--maxiter", type=int, default=12)
    p.add_argument(
        "--budget-factor",
        type=float,
        default=1.5,
        help="budget = budget_factor * cost(fiducial design)",
    )
    p.add_argument("--penalty-weight", type=float, default=1.0)
    p.add_argument("--backend", choices=["ducc", "jht"], default="ducc")
    args = p.parse_args()

    sht.set_sht_backend(args.backend)
    print(f"SHT backend: {sht.get_sht_backend()}")

    t0 = time.time()
    mc_ctx, opt_ctx, cleaner = build_contexts(args.n_sims, nside=args.nside, lmax=args.lmax)
    cost_model = CostModel()
    theta0 = np.zeros(3)

    eig0, sig0, cost0 = _metrics(theta0, mc_ctx, opt_ctx, cleaner, cost_model)
    budget = args.budget_factor * cost0
    print(f"\nfiducial design [{time.time() - t0:.0f}s]:")
    print(f"  EIG_r={eig0:.4f}   sigma(r)={sig0:.4e}   cost={cost0:.1f} $M")
    print(f"  budget = {args.budget_factor} x fiducial cost = {budget:.1f} $M")

    print("\noptimizing EIG_r under budget ...")
    theta_eig = _optimize(
        mc_ctx,
        opt_ctx,
        cleaner,
        cost_model,
        budget,
        objective="marginal_eig_r",
        penalty_weight=args.penalty_weight,
        maxiter=args.maxiter,
    )
    eig1, sig1, cost1 = _metrics(theta_eig, mc_ctx, opt_ctx, cleaner, cost_model)
    s_ndet, s_mission, s_beam = np.exp(theta_eig)
    print(f"  optimized: EIG_r={eig1:.4f}   sigma(r)={sig1:.4e}   cost={cost1:.1f} $M")
    print(f"  scales: n_det x{s_ndet:.2f}  mission x{s_mission:.2f}  beam x{s_beam:.2f}")

    # The sigma(r) objective under the same budget should land at the same design.
    print("\noptimizing sigma(r) under the same budget (equivalence check) ...")
    theta_sig = _optimize(
        mc_ctx,
        opt_ctx,
        cleaner,
        cost_model,
        budget,
        objective="sigma_r",
        penalty_weight=args.penalty_weight,
        maxiter=args.maxiter,
    )

    print("\n=== verdict ===")
    print(f"  EIG_r improved: {eig0:.4f} -> {eig1:.4f}  (sigma(r) {sig0:.4e} -> {sig1:.4e})")
    binds = cost1 > 0.95 * budget
    print(f"  budget binds (cost ~ budget): {binds}  ({cost1:.1f} / {budget:.1f} $M)")
    dtheta = float(np.linalg.norm(theta_eig - theta_sig))
    print(f"  EIG_r-optimum == sigma(r)-optimum: |dtheta|={dtheta:.3f} (small => same design)")
    if eig1 > eig0 and binds:
        print("  VERDICT: cost-constrained EIG optimization found an interior optimum at the")
        print("  budget wall, improving EIG_r over the fiducial design.")
    else:
        print("  VERDICT: inconclusive at this maxiter/n_sims -- raise --maxiter or --n-sims.")


if __name__ == "__main__":
    main()
