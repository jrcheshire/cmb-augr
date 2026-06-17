"""mapbased_grad_characterization.py -- is the straight-through MC gradient usable?

The Phase 1a driver ``optimize_mapbased.sigma_r_from_noise_design`` makes sigma(r)
differentiable through the cut-sky masked-Wiener Monte-Carlo component-separation
forward. Because the gradient flows through a *sample* covariance, it carries
Monte-Carlo noise. This script answers the plan's gating open question -- "is the
straight-through gradient usable for descent?" -- with two diagnostics:

  --mode demo (decisive, cheaper):
      Optimize the per-band detector allocation under a fixed total budget
      (n_det = N_total * softmax(logits)). Unlike an unconstrained noise descent
      (which runs to the n_det -> infinity boundary), a fixed-budget allocation has
      an interior optimum. The optimization runs at FIXED common random numbers, so
      the objective is deterministic; the real test is GENERALIZATION -- we then
      re-evaluate the optimized vs the uniform allocation on independent held-out
      CRN ensembles. If the optimized allocation is genuinely better on fresh sims
      (not just on the training sims), the gradient is usable.

  --mode stability:
      Compute grad sigma(r) w.r.t. the per-band NET vector on several independent
      CRN ensembles and measure how stable the descent DIRECTION is: the resultant
      length R of the unit gradients (1 = perfectly aligned, 0 = random) and the
      per-component coefficient of variation. The cut-sky MC ensemble is now a
      *traced* arg (``CutskyMCContext`` is an ``eqx.Module`` whose sim-batched
      ``harmonic_skies`` / ``noise_keys`` the forward ``lax.map``s over, PR #28), so
      ensembles of the SAME ``(n_sims, nside, lmax)`` reuse one compiled trace --
      only a new n_sims / nside recompiles. That is what makes the ladder below cheap.

  --mode ladder:
      Sweep ``--n-sims-ladder`` and, at each rung, run the demo descent +
      held-out generalization, recording train improvement, held-out gain
      (mean/min), and wall-time (first eval incl. compile, and steady-state
      per-eval). Writes a JSON + a plot of held-out-gain-vs-n_sims and
      per-eval-time-vs-n_sims. This is the Phase-2 read-off that gates B.3: the
      n_sims at which the held-out gap closes (min-gain stabilizes positive), and
      what one objective/gradient eval costs. Cheap on CPU at small n_sims; the
      real ladder runs on a GPU (``--backend jht``).

  --mode beam:
      The beam-lever sensitivity diagnostic (NOT an optimizer -- a free FWHM has no
      cost penalty and would run to 0). Reports the per-band partials
      d sigma(r)/d FWHM and d sigma(r)/d p (which bands' beams move sigma(r)) and the
      MC-stability (resultant R, per-component CoV) of the beam gradient direction
      across CRN ensembles, via optimize_mapbased.sigma_r_from_beam_design.

Tiny CMB-only config (nside=16, no PySM) so the diagnostic is cheap; the gradient
mechanism is foreground-independent. The scientifically interesting FG-driven
allocation needs fg_model="d1s1" at higher nside (heavier) -- deferred to a real
run. Pin BLAS/OMP to 1 thread for reproducible single-core timing.

Usage:
    pixi run python scripts/mapbased_grad_characterization.py --mode demo
    pixi run python scripts/mapbased_grad_characterization.py --mode stability --n-batches 4
    pixi run python scripts/mapbased_grad_characterization.py --mode beam --n-batches 3
    pixi run python scripts/mapbased_grad_characterization.py --mode both --backend jht
    pixi run python scripts/mapbased_grad_characterization.py --mode ladder \
        --n-sims-ladder 12 24 48 96 --backend jht

GPU run (TACC Vista, ``gh`` partition, account JPL-PUB):
    ``pip install jaxht`` (PyPI distribution name; the import is ``jht``) into the
    node Python, then run --mode ladder with --backend jht on one H200. The
    device-aware SHT backend puts the transforms on the GPU; a minimal sbatch
    (``-p gh -A JPL-PUB``) lives in the gitignored scratch launch dir.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from augr import masking as mk
from augr import sht
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.delensing import load_lensing_spectra
from augr.foregrounds import NullForegroundModel
from augr.optimize import make_optimization_context
from augr.optimize_mapbased import (
    sigma_r_from_beam_design,
    sigma_r_from_noise_design,
    w_inv_from_noise_design,
)
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context

# --- fixed fiducial design (3 bands) -----------------------------------------
FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)
N_DET = (200.0, 400.0, 200.0)
NET = (60.0, 50.0, 80.0)
ETA = (0.5, 0.5, 0.5)
MISSION_YEARS = 4.0
F_SKY = 0.6


def _priors(lmax):
    ls = load_lensing_spectra()
    cl_ee = jnp.clip(ls.cl_ee_len[: lmax + 1], 0.0, None)
    cl_bb = jnp.clip(ls.cl_bb_len[: lmax + 1], 0.0, None)
    return cl_ee, cl_bb


def _bin_matrix(ell_min, ell_max, delta_ell, ell_per_bin_below):
    sm = SignalModel(
        instrument=cleaned_map_instrument(f_sky=F_SKY),
        foreground_model=NullForegroundModel(),
        cmb_spectra=CMBSpectra(),
        ell_min=ell_min,
        ell_max=ell_max,
        delta_ell=delta_ell,
        ell_per_bin_below=ell_per_bin_below,
    )
    return jnp.asarray(sm.bin_matrix)


def build_contexts(base_seed, n_sims, *, nside, lmax, var_pix_ref=None):
    """Build (mc_ctx, opt_ctx, cleaner) for a CMB-only tiny config at one CRN seed."""
    ell_max, delta_ell, ell_per_bin_below = lmax, 8, 2
    cl_ee, cl_bb = _priors(lmax)
    bm = _bin_matrix(2, ell_max, delta_ell, ell_per_bin_below)
    true_b = mk.bin_spectrum(
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None),
        bm,
        2,
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
        base_seed=base_seed,
        fg_model=None,
        r_in=0.0,
        var_pix_ref=var_pix_ref,
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


# --- mode: stability ---------------------------------------------------------


def run_stability(args, var_pix_ref):
    """grad sigma(r) wrt NET on n_batches independent CRN ensembles -> direction stability."""
    net0 = jnp.asarray(NET)
    grads = []
    sigmas = []
    for b in range(args.n_batches):
        base_seed = 1000 * (b + 1)
        t0 = time.time()
        mc_ctx, opt_ctx, cleaner = build_contexts(
            base_seed,
            args.n_sims,
            nside=args.nside,
            lmax=args.lmax,
            var_pix_ref=var_pix_ref,
        )

        def loss(net, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner):
            return sigma_r_from_noise_design(
                jnp.asarray(N_DET),
                net,
                jnp.asarray(ETA),
                MISSION_YEARS,
                mc_ctx=mc_ctx,
                opt_ctx=opt_ctx,
                cleaner=cleaner,
            )

        s, g = jax.value_and_grad(loss)(net0)
        grads.append(np.asarray(g))
        sigmas.append(float(s))
        print(
            f"  batch {b} (seed {base_seed}): sigma(r)={float(s):.4e}  "
            f"grad={np.asarray(g)}  [{time.time() - t0:.0f}s]"
        )

    grads = np.stack(grads)  # (B, 3)
    sigmas = np.array(sigmas)
    units = grads / np.linalg.norm(grads, axis=1, keepdims=True)
    mean_unit = units.mean(axis=0)
    resultant = np.linalg.norm(mean_unit)  # R in [0, 1]
    mean_dir = mean_unit / resultant
    cos_to_mean = units @ mean_dir
    cov_per_comp = grads.std(axis=0) / np.abs(grads.mean(axis=0))

    print("\n=== gradient direction stability (wrt NET) ===")
    print(f"  n_sims={args.n_sims}  n_batches={args.n_batches}  nside={args.nside}")
    print(
        f"  sigma(r) across batches:  mean={sigmas.mean():.4e}  "
        f"std/mean={sigmas.std() / sigmas.mean():.3f}"
    )
    print(f"  resultant length R (1=aligned): {resultant:.4f}")
    print(f"  cosine to mean direction: min={cos_to_mean.min():.4f}  mean={cos_to_mean.mean():.4f}")
    print(f"  per-component CoV (std/|mean|): {cov_per_comp}")
    print("  interpretation: R near 1 and CoV small => the descent direction is")
    print("  MC-stable at this n_sims and the straight-through gradient is usable.")


# --- mode: beam --------------------------------------------------------------


def run_beam(args, var_pix_ref):
    """grad sigma(r) wrt the per-band beams (FWHM + shape p) on n_batches CRN ensembles.

    A *sensitivity* diagnostic, NOT an optimizer: a free FWHM has no cost penalty and an
    unconstrained descent would run to 0 (the f_sky->0 footgun analog), so we report the
    per-band partials d sigma(r)/d FWHM and d sigma(r)/d p and the MC-stability of the
    descent direction (over the concatenated [fwhm, p] vector) across independent CRN
    ensembles -- which bands' beams actually move sigma(r), and whether the gradient is
    MC-stable at this n_sims."""
    w_inv = w_inv_from_noise_design(
        jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, F_SKY
    )
    fwhm0 = jnp.asarray(BEAMS)
    p0 = jnp.ones(len(BEAMS))
    grads = []  # concatenated [d/dfwhm, d/dp] per batch
    g_fwhm_rows, g_p_rows, sigmas = [], [], []
    for b in range(args.n_batches):
        base_seed = 1000 * (b + 1)
        t0 = time.time()
        mc_ctx, opt_ctx, cleaner = build_contexts(
            base_seed, args.n_sims, nside=args.nside, lmax=args.lmax, var_pix_ref=var_pix_ref
        )

        def loss(bf, bp, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner):
            return sigma_r_from_beam_design(
                bf, bp, w_inv=w_inv, mc_ctx=mc_ctx, opt_ctx=opt_ctx, cleaner=cleaner
            )

        s, (g_fwhm, g_p) = jax.value_and_grad(loss, argnums=(0, 1))(fwhm0, p0)
        g_fwhm, g_p = np.asarray(g_fwhm), np.asarray(g_p)
        grads.append(np.concatenate([g_fwhm, g_p]))
        g_fwhm_rows.append(g_fwhm)
        g_p_rows.append(g_p)
        sigmas.append(float(s))
        print(
            f"  batch {b} (seed {base_seed}): sigma(r)={float(s):.4e}  "
            f"d/dFWHM={g_fwhm}  d/dp={g_p}  [{time.time() - t0:.0f}s]"
        )

    grads = np.stack(grads)  # (B, 2*n_band)
    sigmas = np.array(sigmas)
    units = grads / np.linalg.norm(grads, axis=1, keepdims=True)
    mean_unit = units.mean(axis=0)
    resultant = np.linalg.norm(mean_unit)
    cov_per_comp = grads.std(axis=0) / np.maximum(np.abs(grads.mean(axis=0)), 1e-300)

    print("\n=== beam sensitivity + gradient direction stability ===")
    print(f"  n_sims={args.n_sims}  n_batches={args.n_batches}  nside={args.nside}")
    print(f"  bands (GHz): {FREQS}   reference FWHM (arcmin): {BEAMS}")
    print(
        f"  sigma(r) across batches:  mean={sigmas.mean():.4e}  "
        f"std/mean={sigmas.std() / sigmas.mean():.3f}"
    )
    print(f"  mean d sigma(r)/d FWHM [per band, 1/arcmin]: {np.stack(g_fwhm_rows).mean(axis=0)}")
    print(f"  mean d sigma(r)/d p    [per band]:           {np.stack(g_p_rows).mean(axis=0)}")
    print(f"  resultant length R (1=aligned): {resultant:.4f}")
    print(f"  per-component CoV (std/|mean|): {cov_per_comp}")
    print("  interpretation: the per-band partials say which beams move sigma(r) (a")
    print("  negative d/dFWHM means a finer beam helps); R near 1 + small CoV => the")
    print("  beam gradient is MC-stable at this n_sims. (No cost model: this is a")
    print("  sensitivity readout, not a beam optimizer.)")


# --- mode: demo --------------------------------------------------------------


def run_demo(args, var_pix_ref, *, n_sims=None, return_metrics=False):
    """Fixed-budget allocation descent + held-out generalization check.

    ``n_sims`` overrides ``args.n_sims`` (the ladder passes one rung at a time);
    ``return_metrics`` returns the per-rung dict instead of only printing.
    """
    n_sims = args.n_sims if n_sims is None else n_sims
    n_total = float(sum(N_DET))
    net0 = jnp.asarray(NET)
    eta0 = jnp.asarray(ETA)

    def alloc(logits):
        return n_total * jax.nn.softmax(logits)

    def sigma_of(logits, mc_ctx, opt_ctx, cleaner):
        return sigma_r_from_noise_design(
            alloc(logits),
            net0,
            eta0,
            MISSION_YEARS,
            mc_ctx=mc_ctx,
            opt_ctx=opt_ctx,
            cleaner=cleaner,
        )

    # Training ensemble (fixed CRN). Eager value_and_grad here; the map-based sigma(r)
    # is now also jax.jit-able (the jnp + stop_gradient needlet-channel mask removed the
    # last np.asarray/float boundary), so this can run under jit on a GPU -- kept eager
    # for this small CPU diagnostic.
    print("Building training ensemble (eager value_and_grad) ...")
    t_build = time.time()
    mc_ctx, opt_ctx, cleaner = build_contexts(
        0, n_sims, nside=args.nside, lmax=args.lmax, var_pix_ref=var_pix_ref
    )
    build_s = time.time() - t_build
    vg = jax.value_and_grad(lambda lg: sigma_of(lg, mc_ctx, opt_ctx, cleaner))
    # Eager value_and_grad re-traces every call; on GPU the lax.map(NILC-clean) scan is
    # NOT cached between calls, so each eval pays the (multi-minute) XLA compile again.
    # jit compiles once and reuses the executable across all L-BFGS / held-out evals --
    # essential on GPU (where compile dominates), a small win on CPU.
    if getattr(args, "jit", False):
        vg = jax.jit(vg)

    def scipy_vg(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    logits0 = np.zeros(3)  # uniform allocation
    # First eval pays the XLA compile; a second identical eval is the steady-state
    # value+grad cost (the per-step price B.3's inner design loop pays).
    t_e0 = time.time()
    s_uniform_train = scipy_vg(logits0)[0]
    first_eval_s = time.time() - t_e0
    t_e1 = time.time()
    scipy_vg(logits0)
    per_eval_s = time.time() - t_e1
    print(
        f"  uniform allocation: sigma(r)={s_uniform_train:.6e}  "
        f"[build {build_s:.0f}s, first eval {first_eval_s:.0f}s, steady eval {per_eval_s:.1f}s]"
    )

    # Bound the eval count -- each eager eval is ~40s at n_sims=12, and L-BFGS line
    # searches can call the objective several times per iteration.
    res = minimize(
        scipy_vg,
        logits0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": args.maxiter, "maxfun": args.maxiter, "ftol": 1e-9, "gtol": 1e-7},
    )
    logits_opt = res.x
    alloc_opt = np.asarray(alloc(jnp.asarray(logits_opt)))
    print(
        f"  optimized allocation: sigma(r)={res.fun:.6e}  (n_iter={res.nit}, success={res.success})"
    )
    print(f"  n_det uniform  = {np.asarray(alloc(jnp.asarray(logits0)))}")
    print(f"  n_det optimized= {alloc_opt}")
    improvement = 100.0 * (s_uniform_train - res.fun) / s_uniform_train
    print(f"  TRAIN improvement: {improvement:.2f}%")

    # Generalization: re-evaluate on independent held-out CRN ensembles (eager
    # forward; each ensemble is a fresh sky set).
    print("\nGeneralization on held-out CRN ensembles:")
    gen_rows = []
    for seed in args.heldout_seeds:
        t1 = time.time()
        ho_mc, ho_opt, ho_cl = build_contexts(
            seed, n_sims, nside=args.nside, lmax=args.lmax, var_pix_ref=var_pix_ref
        )
        s_unif = float(sigma_of(jnp.asarray(logits0), ho_mc, ho_opt, ho_cl))
        s_opt = float(sigma_of(jnp.asarray(logits_opt), ho_mc, ho_opt, ho_cl))
        gain = 100.0 * (s_unif - s_opt) / s_unif
        gen_rows.append((seed, s_unif, s_opt, gain))
        print(
            f"  seed {seed}: uniform={s_unif:.6e}  optimized={s_opt:.6e}  "
            f"gain={gain:+.2f}%  [{time.time() - t1:.0f}s]"
        )

    gains = np.array([r[3] for r in gen_rows])
    print("\n=== demo descent summary ===")
    print(f"  train improvement: {improvement:+.2f}%")
    print(f"  held-out gain: mean={gains.mean():+.2f}%  min={gains.min():+.2f}%  (n={len(gains)})")
    if gains.min() > 0:
        print("  VERDICT: the optimized allocation is better on EVERY held-out")
        print("  ensemble => the straight-through gradient found a real optimum, not")
        print("  CRN overfit. The gradient is usable for design descent.")
    else:
        print("  VERDICT: at least one held-out ensemble did not improve => the MC")
        print("  noise may be drowning the descent at this n_sims; raise n_sims.")

    if return_metrics:
        return {
            "n_sims": int(n_sims),
            "train_improvement_pct": float(improvement),
            "heldout_gain_mean_pct": float(gains.mean()),
            "heldout_gain_min_pct": float(gains.min()),
            "n_heldout": len(gains),
            "build_s": float(build_s),
            "first_eval_s": float(first_eval_s),
            "per_eval_s": float(per_eval_s),
            "sigma_uniform_train": float(s_uniform_train),
            "sigma_opt_train": float(res.fun),
            "lbfgs_nit": int(res.nit),
            "lbfgs_success": bool(res.success),
        }
    return None


# --- mode: ladder ------------------------------------------------------------


def run_ladder(args):
    """Sweep n_sims; per rung run the demo descent + held-out check, then write a
    JSON table + a gap/time-vs-n_sims plot. The Phase-2 read-off gating B.3: the
    n_sims where the held-out gap closes, and the per-eval cost."""
    rows = []
    for n_sims in args.n_sims_ladder:
        print(f"\n########## ladder rung: n_sims = {n_sims} ##########")
        # Per-rung var_pix_ref (a filter knob -- self-consistent at each n_sims;
        # absorbed by the transfer/leakage debias, so it does not bias sigma(r)).
        cal_ctx, _, _ = build_contexts(0, n_sims, nside=args.nside, lmax=args.lmax)
        rows.append(run_demo(args, cal_ctx.var_pix_ref, n_sims=n_sims, return_metrics=True))

    print("\n=== n_sims ladder summary ===")
    print(
        f"  config: nside={args.nside} lmax={args.lmax} maxiter={args.maxiter} "
        f"backend={sht.get_sht_backend()}"
    )
    print(
        f"  {'n_sims':>7} {'train%':>8} {'ho_mean%':>9} {'ho_min%':>8} "
        f"{'eval_s':>8} {'build_s':>8}"
    )
    for m in rows:
        print(
            f"  {m['n_sims']:>7d} {m['train_improvement_pct']:>8.1f} "
            f"{m['heldout_gain_mean_pct']:>9.1f} {m['heldout_gain_min_pct']:>8.1f} "
            f"{m['per_eval_s']:>8.1f} {m['build_s']:>8.1f}"
        )
    closed = [m["n_sims"] for m in rows if m["heldout_gain_min_pct"] > 0]
    if closed:
        print(f"  held-out gap closes (min-gain > 0) at n_sims >= {min(closed)}.")
    else:
        print("  held-out gap NOT closed at any rung -- extend the ladder.")

    payload = {
        "config": {
            "nside": args.nside,
            "lmax": args.lmax,
            "maxiter": args.maxiter,
            "heldout_seeds": list(args.heldout_seeds),
            "backend": sht.get_sht_backend(),
        },
        "rungs": rows,
    }
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {args.out_prefix}.json")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable -- skipped plot)")
        return
    ns = [m["n_sims"] for m in rows]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    axL.axhline(0.0, color="0.6", lw=0.8, ls=":")
    axL.plot(ns, [m["train_improvement_pct"] for m in rows], "o-", color="0.5", label="train")
    axL.plot(ns, [m["heldout_gain_mean_pct"] for m in rows], "o-", color="C0", label="held-out mean")
    axL.plot(ns, [m["heldout_gain_min_pct"] for m in rows], "o-", color="C3", label="held-out min")
    axL.set_xlabel("n_sims")
    axL.set_ylabel("sigma(r) improvement [%]")
    axL.set_title("CRN overfit closes as held-out -> train")
    axL.legend(fontsize=8)
    axR.plot(ns, [m["per_eval_s"] for m in rows], "o-", color="C2")
    axR.set_xlabel("n_sims")
    axR.set_ylabel("steady-state value+grad eval [s]")
    axR.set_title(f"per-eval cost ({sht.get_sht_backend()})")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=140)
    print(f"  wrote {args.out_prefix}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["demo", "stability", "beam", "both", "ladder"], default="demo")
    p.add_argument("--n-sims", type=int, default=12)
    p.add_argument(
        "--n-sims-ladder",
        type=int,
        nargs="+",
        default=[12, 24, 48, 96],
        help="ladder: n_sims rungs to sweep",
    )
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--lmax", type=int, default=24)
    p.add_argument("--n-batches", type=int, default=4, help="stability: # CRN ensembles")
    p.add_argument("--maxiter", type=int, default=12, help="demo: L-BFGS-B max iters / fun-evals")
    p.add_argument("--heldout-seeds", type=int, nargs="+", default=[9001, 9002])
    p.add_argument("--backend", choices=["ducc", "jht"], default="ducc")
    p.add_argument(
        "--jit",
        action="store_true",
        help="jit the value_and_grad objective (compile once, reuse) -- needed on GPU, "
        "where the eager path recompiles the lax.map scan (~minutes) every eval.",
    )
    p.add_argument(
        "--out-prefix", default="grad_char_ladder", help="ladder: JSON/PNG output path prefix"
    )
    args = p.parse_args()

    sht.set_sht_backend(args.backend)
    print(f"SHT backend: {sht.get_sht_backend()}")

    if args.mode == "ladder":
        print("\n########## MODE: ladder ##########")
        run_ladder(args)
        return

    # Freeze var_pix_ref once so the only thing varying across ensembles is the CRN
    # (var_pix_ref is a filter knob; a common value isolates the MC noise we measure).
    print("Calibrating shared var_pix_ref ...")
    cal_ctx, _, _ = build_contexts(0, args.n_sims, nside=args.nside, lmax=args.lmax)
    var_pix_ref = cal_ctx.var_pix_ref
    print(f"  var_pix_ref = {var_pix_ref:.4e}")

    if args.mode in ("demo", "both"):
        print("\n########## MODE: demo ##########")
        run_demo(args, var_pix_ref)
    if args.mode in ("stability", "both"):
        print("\n########## MODE: stability ##########")
        run_stability(args, var_pix_ref)
    if args.mode == "beam":
        print("\n########## MODE: beam ##########")
        run_beam(args, var_pix_ref)


if __name__ == "__main__":
    main()
