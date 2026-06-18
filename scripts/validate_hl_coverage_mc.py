"""validate_hl_coverage_mc.py -- SBC coverage test on the REAL cut-sky masked-Wiener MC.

The realistic companion to ``scripts/validate_hl_coverage.py``. That script ran the
frequentist coverage / simulation-based-calibration (SBC) test against an *analytic*
scaled-chi^2 oracle and found the Knox/Gaussian r interval well-calibrated at the bump --
i.e. the bump chi^2 skew alone does NOT make the Gaussian under-cover. But that cannot
settle the *real* pipeline, whose non-Gaussianity (if any) comes from the **masking + ILC**
machinery (mask-mode coupling, E->B leakage variance, ILC chance correlations), not analytic
chi^2. This script draws the data realizations from a real cut-sky masked-Wiener Monte-Carlo
ensemble and asks the same question: does the Gaussian interval under-cover (vindicating the
Hamimeche-Lewis / non-Gaussian likelihood), or is it calibrated (so the documented HL
widening is a posterior-width artifact, not a calibration fix)?

Design (disjoint train/test SBC):
  * TRAIN ensemble (base_seed) -> covariance, fiducial bandpower, residual template ->
    the Gaussian + HL likelihoods (the "analysis", frozen).
  * TEST ensemble (base_seed + SEED_STRIDE, disjoint) -> fresh "observed" realizations.
  * **Self-debias** (each test sim debiased with the TEST ensemble's own transfer/leakage) is
    the HEADLINE / calibration measure: each ensemble is centered at truth, so this isolates
    the likelihood-shape calibration. **Cross-debias** (test raw recs through TRAIN's frozen
    transfer/leakage) is a DIAGNOSTIC ONLY -- it is NOT a valid coverage construction: one
    train calibration is reused across all test realizations, imprinting a shared offset
    (random sign per draw; does not shrink with n_train) that under-covers regardless of the
    likelihood. Reported to expose the finite-sim transfer/leakage effect, not to read
    calibration off.
  * For each test realization: marginal r posterior (floating A_lens/A_res on a grid) ->
    PIT = F_post(r_true=r_in). Coverage = how often the interval covers the truth.

Reports two-sided central coverage + the one-sided upper-limit coverage (the relevant
quantity for an r limit; run with the default --r-in 0), for Gaussian vs HL, plus the
augr bandpower_ks calibration verdict on the ensemble. Decision: Gaussian under-covers &
HL ~nominal => HL vindicated; both ~nominal => Knox was fine, re-scope the widening claim;
both under-cover => the masking/ILC non-Gaussianity exceeds the HL idealization (mc_calibrated).

Needs the ``[compsep]`` extra (ducc0; pysm3 for --fg-model). The coverage core itself is
sampling-free (no NUTS), so no ``[sampling]`` extra. Each test realization is one full
masked-Wiener clean (~20 s/sim at nside=64), so n_test dominates cost; pass ``--workers N``
to parallelize the ensemble over a spawn pool (the NILC cleaner is picklable). Workers pin
BLAS to 1 thread; this driver sets ``DUCC0_NUM_THREADS = cpu_count // workers`` so
``workers x ducc-threads ~= cpu_count`` (no oversubscription -- see
``reference_ducc_omp_threading``).

Usage:
    pixi run python scripts/validate_hl_coverage_mc.py --nside 64 --n-train 100 --n-test 300 \
        --fg-model d1s1 --r-in 0 --float A_lens,A_res --workers 8 --plot cov_mc.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from augr import masking as mk
from augr import sbc
from augr.cleaning import nilc_cleaner
from augr.hit_maps import l2_hit_map
from augr.likelihood import bandpower_ks
from augr.likelihood.from_cutsky import build_cutsky_signal_model, build_likelihood
from augr.likelihood.ordering import SpectrumLayout
from augr.parallel import cpu_count
from augr.pipeline import (
    ForecastConfig,
    ResidualTemplateSource,
    SpectrumSource,
    clean_sky,
    cutsky_mc_bandpowers,
)
from augr.signal import flatten_params

# Same illustrative wide-band config as the cut-sky headlines (LiteBIRD-PTEP-ish).
FREQS = (30.0, 44.0, 95.0, 150.0, 280.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 12.0)  # arcmin
W_INV = (2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 1.5e-4)  # uK^2 sr per band
SEED_STRIDE = 100_000  # disjoint train/test seed blocks (matches augr.design_opt)


def _build_config(args, mask, hit, *, n_sims, base_seed):
    cfg = ForecastConfig(
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=W_INV,
        cleaner=nilc_cleaner(clean_e=True),  # picklable (functools.partial) -> mc_workers>1 OK
        nside=args.nside,
        lmax=args.lmax,
        fg_model=None if args.fg_model == "none" else args.fg_model,
        r_in=args.r_in,
        seed=base_seed,
        hit_map=hit,
        residual_source=ResidualTemplateSource[args.residual.upper()],
        r_fid=args.r_in,
        ell_min=2,
        ell_max=args.ell_max,
        delta_ell=args.delta_ell,
        ell_per_bin_below=args.ell_per_bin_below,
        spectrum_source=SpectrumSource.CUTSKY_MC,
        mask=mask,
        n_sims_mc=n_sims,
        base_seed_mc=base_seed,
        mc_workers=args.workers,
    )
    return replace(cfg, f_sky=mk.f_sky_of(mask))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", default="d1s1", choices=["d1s1", "d10s5", "none"])
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--f-sky", type=float, default=0.6)
    p.add_argument("--r-in", type=float, default=0.0, help="input r of the sims (= r_true)")
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=120)
    p.add_argument("--delta-ell", type=int, default=15)
    p.add_argument("--ell-per-bin-below", type=int, default=10)
    p.add_argument("--n-train", type=int, default=100, help="train sims (covariance + template)")
    p.add_argument("--n-test", type=int, default=300, help="test sims (coverage realizations)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--residual", default="oracle", choices=["oracle", "gnilc"])
    p.add_argument("--uniform-hits", action="store_true", help="uniform hits instead of L2 scan depth")
    p.add_argument("--float", dest="floated", type=str, default="A_lens,A_res",
                   help="comma list of nuisances to marginalize (default A_lens,A_res)")
    p.add_argument("--a-res-prior", type=float, default=0.3)
    p.add_argument("--a-lens-prior", type=float, default=0.25)
    p.add_argument("--n-nuis-grid", type=int, default=21)
    p.add_argument("--n-sigma-nuis", type=float, default=5.0)
    p.add_argument("--n-grid", type=int, default=400, help="r-grid points")
    p.add_argument("--n-sigma-grid", type=float, default=12.0)
    p.add_argument("--upper-level", type=float, default=0.95)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args()

    floated = {s.strip() for s in args.floated.split(",") if s.strip()}
    if floated - {"A_lens", "A_res"}:
        raise ValueError(f"--float may only name A_lens / A_res, got {floated}")
    if args.base_seed + args.n_train + 1 >= args.base_seed + SEED_STRIDE:
        raise ValueError("train block overlaps the test block; reduce n_train or raise SEED_STRIDE")
    if args.workers > 1:
        # ducc reads DUCC0_NUM_THREADS first; set it so workers x ducc-threads ~= cpu_count
        # (spawn children inherit the parent env). Overrides any pre-set high value.
        os.environ["DUCC0_NUM_THREADS"] = str(max(1, cpu_count() // args.workers))

    npix = 12 * args.nside**2
    hit = jnp.ones(npix) if args.uniform_hits else jnp.asarray(l2_hit_map(args.nside, coord="G"))
    mask = mk.galactic_mask(args.nside, args.f_sky)
    fsky_real = mk.f_sky_of(mask)

    # --- TRAIN ensemble + residual template (the frozen "analysis") ---
    cfg = _build_config(args, mask, hit, n_sims=args.n_train, base_seed=args.base_seed)
    cleaned = clean_sky(cfg)
    mc_train, template_ells, template_cl = cutsky_mc_bandpowers(cleaned, cfg)

    # --- TEST ensemble (disjoint seeds); reuse `cleaned` for its template, ignore it ---
    cfg_test = _build_config(args, mask, hit, n_sims=args.n_test, base_seed=args.base_seed + SEED_STRIDE)
    mc_test, _ell_t, _cl_t = cutsky_mc_bandpowers(cleaned, cfg_test)
    if mc_test.rec_full is None:
        raise RuntimeError("mc_test.rec_full is None; needs the CutskyMC.rec_full forward field")

    # Self-debias = HEADLINE / calibration measure (each ensemble centered at truth).
    data_self = np.asarray(mc_test.debiased_bandpowers)
    # Cross-debias = DIAGNOSTIC ONLY (test raw recs through TRAIN's frozen transfer/leakage):
    # a shared offset is reused across all test sims -> under-covers regardless of likelihood;
    # exposes the finite-sim transfer/leakage effect, not a calibration measure.
    data_cross = np.asarray(mk.debias_bandpower(mc_test.rec_full, mc_train.transfer, mc_train.leakage))

    # --- likelihoods from TRAIN ---
    signal_model, _inst = build_cutsky_signal_model(
        template_ells, template_cl, cfg.f_sky,
        ell_min=cfg.ell_min, ell_max=cfg.ell_max,
        delta_ell=cfg.delta_ell, ell_per_bin_below=cfg.ell_per_bin_below,
    )
    names = list(signal_model.parameter_names)
    fid_vec = flatten_params({"r": args.r_in, "A_lens": 1.0, "A_res": 1.0}, names)
    gauss0 = build_likelihood("gaussian", signal_model, fid_vec, mc_train)
    hl0 = build_likelihood("hl", signal_model, fid_vec, mc_train)
    noise_floor = np.asarray(mc_train.mean_bandpower) - np.asarray(signal_model.data_vector(fid_vec))
    layout = SpectrumLayout.from_freq_pairs(signal_model.freq_pairs, signal_model.n_bins)

    # --- grid + core (from train cov) ---
    base, t_r, t_L, t_res = sbc.linear_basis(signal_model, names)
    cov_diag = np.diag(np.asarray(mc_train.covariance))
    prior_sig = {"A_lens": args.a_lens_prior, "A_res": args.a_res_prior}
    nuis = sbc.NuisanceGrid.build(
        floated=floated, prior_sig=prior_sig, n_nuis_grid=args.n_nuis_grid, n_sigma_nuis=args.n_sigma_nuis
    )
    sigma_r = sbc.marginal_sigma_r(
        t_r=t_r, t_l=t_L, t_res=t_res, cov_diag=cov_diag, floated=floated, prior_sig=prior_sig
    )
    r_grid = np.linspace(args.r_in - args.n_sigma_grid * sigma_r,
                         args.r_in + args.n_sigma_grid * sigma_r, args.n_grid)
    pred_flat = sbc.build_pred_grid(
        base, t_r, t_L, t_res, r_grid=r_grid, al_axis=nuis.al_axis, ares_axis=nuis.ares_axis
    )
    core = sbc.make_marginal_logpost(
        gauss0=gauss0, hl0=hl0, noise_floor=noise_floor, layout=layout, pred_flat=pred_flat,
        logprior_grid=nuis.logprior_grid, n_grid=args.n_grid, n_al=nuis.n_al, n_ar=nuis.n_ar,
    )

    n_test = data_self.shape[0]
    res_cross = sbc.run_coverage(core, data_cross, r_true=args.r_in, r_grid=r_grid, n_trials=n_test)
    res_self = sbc.run_coverage(core, data_self, r_true=args.r_in, r_grid=r_grid, n_trials=n_test)
    ks = bandpower_ks(mc_test.debiased_bandpowers, mc_train.mean_bandpower, mc_train.covariance)

    # --- report ---
    print("\nMC-ensemble HL coverage / SBC test  (real cut-sky masked-Wiener pipeline)")
    print(f"  fg={args.fg_model}  nside={args.nside}  lmax={args.lmax}  f_sky={fsky_real:.3f}  "
          f"r_in={args.r_in}  cleaner=NILC")
    print(f"  n_train={args.n_train}  n_test={n_test}  floated={sorted(floated) or 'none'}  "
          f"marginal sigma(r)={sigma_r:.3e}")
    for tag, res in (("self-debias (HEADLINE / calibration)", res_self),
                     ("cross-debias (diagnostic only -- expected to under-cover)", res_cross)):
        print(f"\n  --- {tag} ---")
        sbc.print_coverage_table(sbc.coverage_table(res, upper_level=args.upper_level))
        if res.edge_hits["gauss"] or res.edge_hits["hl"]:
            print(f"  note: grid-edge PIT hits gauss={res.edge_hits['gauss']} hl={res.edge_hits['hl']} "
                  f"(widen --n-sigma-grid if non-negligible)")
    print(f"\n  KS verdict: recommend={ks['recommend']!r}  "
          f"(gauss_rejected_bump={ks['gauss_rejected_bump']}, hl_better_bump={ks['hl_better_bump']})")
    print("\n  read-off (self-debias): both Gaussian & HL ~nominal => calibrated, HL adds nothing; "
          "Gaussian under-covers & HL ~nominal => HL helps; both under-cover => mc_calibrated.")

    if args.plot:
        marg = f"marginalized over {', '.join(sorted(floated))}" if floated else "nuisances fixed"
        title = (f"sigma(r) calibration -- cut-sky MC self-debias, fg={args.fg_model} ({marg})   "
                 f"nside={args.nside}, f_sky={fsky_real:.2f}, r_in={args.r_in}, "
                 f"n_train={args.n_train}, n_test={n_test}")
        sbc.make_coverage_plot(args.plot, res_self, title=title)
        print(f"\n  wrote {args.plot} (self-debias headline)")


if __name__ == "__main__":
    main()
