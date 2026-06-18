"""validate_hl_coverage.py -- frequentist coverage / SBC test of the HL sigma(r) interval.

The companion to the CDF/KS bandpower check. The KS plot validates the *ingredient*
(the bandpower ensemble is chi^2-skewed, not Gaussian); this validates the *product*
(the interval on r is calibrated). It answers "how do we know to trust the HL error bar"
with an operational, frequentist statement -- a calibrated 68% interval contains the truth
68% of the time -- against an **analytic oracle**, with no external pipeline.

Oracle. In the idealized single-field limit (known f_sky, one cleaned BB field, no
off-diagonal mode coupling) the bandpower estimator is *exactly* a scaled chi^2:

    Chat_b = C_b * chi2(nu_b) / nu_b,   nu_b = sum_{l in bin} (2l+1) * f_sky,

mean C_b, Knox variance 2 C_b^2 / nu_b. That is the exact single-field bandpower
likelihood (Wishart) -- textbook, not augr code. HL was *constructed* to reproduce it;
the Gaussian/Knox likelihood approximates the chi^2 by a matched-variance Gaussian.

Two regimes (`--float`):
  * conditional (default, no nuisances floated): scan r with A_lens / A_res held at truth.
    Finding: r is linear, so rhat = sum_b w_b Chat_b is a weighted sum over bandpowers ->
    Knox gets its variance exactly right and the sum central-limits back to ~Gaussian, so
    BOTH likelihoods cover at nominal -- the bump chi^2 skew does NOT show up as
    r-interval under-coverage here.
  * marginalized (`--float A_lens,A_res` with `--res-to-tensor > 0`): a non-zero red
    residual template makes A_res degenerate with r at low l (the realistic
    FG-residual-vs-tensor degeneracy that drives augr's documented HL widening). The
    nuisances are marginalized on a grid (Gaussian priors). This is where HL is expected
    to matter -- the apples-to-apples match to the cut-sky HL-NUTS-vs-Fisher comparison.

Procedure (simulation-based calibration):
  1. Build the real augr HL and Gaussian likelihoods (`*.from_external`).
  2. Draw many bandpower realizations from the exact scaled-chi^2 oracle at fixed truth.
  3. For each, evaluate the r-posterior on a grid (marginalizing floated nuisances), and
     record the probability-integral transform PIT = F_post(r_true).
  4. Calibrated <=> PIT ~ Uniform(0,1). Two-sided central-L coverage = fraction with
     |PIT - 0.5| <= L/2; one-sided upper-limit coverage at level u (the relevant quantity
     for an r upper limit; run with `--r-true 0`) = fraction with PIT <= u.

The same trials drive both likelihoods, paired (common random numbers). Caveats: the HL
prep (C_fl, M_f) and the covariance are built at the fiducial = truth (correctly-specified,
fixed-cov); nuisance priors default to augr's (A_res sigma=0.3); the realistic leg on the
actual masked-Wiener MC ensemble (with a disjoint train/test split) is a separate follow-on.

Usage::

    pixi run python scripts/validate_hl_coverage.py --n-trials 2000                      # conditional
    pixi run python scripts/validate_hl_coverage.py --float A_lens,A_res --res-to-tensor 3
    pixi run python scripts/validate_hl_coverage.py --float A_lens,A_res --res-to-tensor 3 --r-true 0
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from augr import sbc
from augr.likelihood.from_cutsky import build_cutsky_signal_model
from augr.likelihood.gaussian import GaussianLikelihood
from augr.likelihood.hl import HLLikelihood
from augr.likelihood.ordering import SpectrumLayout
from augr.signal import flatten_params

_R_REF = 0.01  # reference r for normalizing the residual template (so it survives r_true=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f-sky", type=float, default=0.6)
    p.add_argument("--r-true", type=float, default=0.01, help="injected (and fiducial) r")
    p.add_argument("--r-fid", type=float, default=None, help="likelihood-prep fiducial r (default = r-true)")
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=120)
    p.add_argument("--delta-ell", type=int, default=5)
    p.add_argument("--ell-per-bin-below", type=int, default=30)
    p.add_argument("--noise-cl", type=float, default=0.0, help="flat post-sep noise C_l (signal units)")
    p.add_argument("--float", dest="floated", type=str, default="",
                   help="comma list of nuisances to marginalize: A_lens,A_res (default none)")
    p.add_argument("--res-to-tensor", type=float, default=0.0,
                   help="A_res=1 residual / tensor(r=0.01) power ratio over the bump (0 = no template)")
    p.add_argument("--res-slope", type=float, default=-2.4, help="residual template C_l ~ (l/pivot)^slope")
    p.add_argument("--res-pivot", type=float, default=5.0)
    p.add_argument("--a-res-prior", type=float, default=0.3, help="Gaussian sigma on A_res (augr default)")
    p.add_argument("--a-lens-prior", type=float, default=0.25, help="Gaussian sigma on A_lens")
    p.add_argument("--n-nuis-grid", type=int, default=21)
    p.add_argument("--n-sigma-nuis", type=float, default=5.0)
    p.add_argument("--n-trials", type=int, default=2000)
    p.add_argument("--n-grid", type=int, default=600, help="r-grid points")
    p.add_argument("--n-sigma-grid", type=float, default=12.0, help="r-grid half-width in marginal sigma")
    p.add_argument("--upper-level", type=float, default=0.95, help="one-sided upper-limit level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args()

    r_fid = args.r_true if args.r_fid is None else args.r_fid
    floated = {s.strip() for s in args.floated.split(",") if s.strip()}
    if floated - {"A_lens", "A_res"}:
        raise ValueError(f"--float may only name A_lens / A_res, got {floated}")
    if "A_res" in floated and args.res_to_tensor <= 0:
        raise ValueError("floating A_res is meaningless with a zero residual template; pass --res-to-tensor > 0")

    # --- residual template: raw red shape, then normalized so A_res=1 ~ res_to_tensor x tensor(R_ref) ---
    ells_tmpl = np.arange(args.ell_min, args.ell_max + 1, dtype=float)
    cl_res_raw = (ells_tmpl / args.res_pivot) ** args.res_slope if args.res_to_tensor > 0 else np.zeros_like(ells_tmpl)

    def build(cl_tmpl):
        return build_cutsky_signal_model(
            ells_tmpl, cl_tmpl, args.f_sky,
            ell_min=args.ell_min, ell_max=args.ell_max,
            delta_ell=args.delta_ell, ell_per_bin_below=args.ell_per_bin_below,
        )[0]

    probe = build(cl_res_raw)
    names = list(probe.parameter_names)

    def dv_of(sm, r, al, ares):
        return np.asarray(sm.data_vector(flatten_params({"r": r, "A_lens": al, "A_res": ares}, names)))

    if args.res_to_tensor > 0:
        bump = np.asarray(probe.bin_centers) < 30.0
        t_r0 = dv_of(probe, 1.0, 0.0, 0.0) - dv_of(probe, 0.0, 0.0, 0.0)
        t_res0 = dv_of(probe, 0.0, 0.0, 1.0) - dv_of(probe, 0.0, 0.0, 0.0)
        kappa = args.res_to_tensor * _R_REF * t_r0[bump].sum() / t_res0[bump].sum()
        cl_res = kappa * cl_res_raw
    else:
        cl_res = cl_res_raw
    signal_model = build(cl_res)
    layout = SpectrumLayout.from_freq_pairs(signal_model.freq_pairs, signal_model.n_bins)
    n_bins = signal_model.n_bins

    # --- linear basis templates (model is linear in r, A_lens, A_res) ---
    base, t_r, t_L, t_res = sbc.linear_basis(signal_model, names)

    # --- exact-chi^2 oracle: total mean per bin at the truth (A_lens=A_res=1) ---
    n_b = np.full(n_bins, args.noise_cl)
    c_total = dv_of(signal_model, args.r_true, 1.0, 1.0) + n_b
    if np.any(c_total <= 0):
        raise ValueError("non-positive total bandpower; check r-true / noise-cl / template.")
    nu_b = sbc.mode_counts(signal_model.bin_edges, args.f_sky)
    cov_diag = 2.0 * c_total**2 / nu_b
    knox_cov = jnp.asarray(np.diag(cov_diag))

    # --- real augr likelihoods, built once at the fiducial ---
    fid_vec = flatten_params({"r": r_fid, "A_lens": 1.0, "A_res": 1.0}, names)
    total_fid = jnp.asarray(dv_of(signal_model, r_fid, 1.0, 1.0) + n_b)
    hl0 = HLLikelihood.from_external(signal_model, fid_vec, total_fid, knox_cov)
    gauss0 = GaussianLikelihood.from_external(signal_model, fid_vec, knox_cov)

    # --- nuisance grid + r-grid sized from the marginal Gaussian sigma ---
    prior_sig = {"A_lens": args.a_lens_prior, "A_res": args.a_res_prior}
    nuis = sbc.NuisanceGrid.build(
        floated=floated, prior_sig=prior_sig, n_nuis_grid=args.n_nuis_grid, n_sigma_nuis=args.n_sigma_nuis
    )
    sigma_r = sbc.marginal_sigma_r(
        t_r=t_r, t_l=t_L, t_res=t_res, cov_diag=cov_diag, floated=floated, prior_sig=prior_sig
    )
    r_grid = np.linspace(args.r_true - args.n_sigma_grid * sigma_r,
                         args.r_true + args.n_sigma_grid * sigma_r, args.n_grid)
    pred_flat = sbc.build_pred_grid(
        base, t_r, t_L, t_res, r_grid=r_grid, al_axis=nuis.al_axis, ares_axis=nuis.ares_axis
    )
    core = sbc.make_marginal_logpost(
        gauss0=gauss0, hl0=hl0, noise_floor=n_b, layout=layout, pred_flat=pred_flat,
        logprior_grid=nuis.logprior_grid, n_grid=args.n_grid, n_al=nuis.n_al, n_ar=nuis.n_ar,
    )

    # --- draw the exact scaled-chi^2 oracle (CRN: same draws for both likelihoods) ---
    rng = np.random.default_rng(args.seed)
    draws = c_total[None, :] * rng.gamma(shape=nu_b[None, :] / 2.0, scale=2.0,
                                         size=(args.n_trials, n_bins)) / nu_b[None, :]
    result = sbc.run_coverage(core, draws, r_true=args.r_true, r_grid=r_grid, n_trials=args.n_trials)

    # --- report ---
    print("\nHL coverage / SBC test  (analytic scaled-chi^2 oracle, single field)")
    print(f"  f_sky={args.f_sky}  r_true={args.r_true}  ell=[{args.ell_min},{args.ell_max}]  "
          f"noise_cl={args.noise_cl}  n_trials={args.n_trials}")
    print(f"  floated={sorted(floated) or 'none'}  res_to_tensor={args.res_to_tensor}  "
          f"marginal sigma(r)={sigma_r:.3e}  nu_b[0:4]={np.round(nu_b[:4], 1)}")
    sbc.print_coverage_table(sbc.coverage_table(result, upper_level=args.upper_level))
    print("\n  (target = the quoted level; under-coverage => interval too narrow / over-optimistic)")
    if result.edge_hits["gauss"] or result.edge_hits["hl"]:
        print(f"  note: grid-edge PIT hits gauss={result.edge_hits['gauss']} hl={result.edge_hits['hl']} "
              f"(widen --n-sigma-grid if non-negligible)")

    if args.plot:
        marg = f"marginalized over {', '.join(sorted(floated))}" if floated else "nuisances fixed"
        title = (f"σ(r) error-bar calibration — analytic χ² oracle ({marg})   "
                 f"f_sky={args.f_sky}, r={args.r_true}, ℓ≤{args.ell_max}, n_trials={args.n_trials}")
        sbc.make_coverage_plot(args.plot, result, title=title)
        print(f"\n  wrote {args.plot}")


if __name__ == "__main__":
    main()
