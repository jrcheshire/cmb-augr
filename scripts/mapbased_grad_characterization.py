"""mapbased_grad_characterization.py -- is the straight-through MC gradient usable?

The Phase 1a driver ``optimize_mapbased.sigma_r_from_noise_design`` makes sigma(r)
differentiable through the cut-sky masked-Wiener Monte-Carlo component-separation
forward. Because the gradient flows through a *sample* covariance, it carries
Monte-Carlo noise. This script answers the plan's gating open question -- "is the
straight-through gradient usable for descent?" -- with two diagnostics:

  --mode demo (decisive, cheaper):
      Optimize the per-band detector allocation under a fixed total budget
      (n_det = N_total * softmax(logits)); a fixed-budget allocation has an interior
      optimum (an unconstrained noise descent runs to the n_det -> infinity
      boundary). Two optimizer protocols (--optimizer):
        adam  (default): RE-RANDOMIZED-CRN stochastic descent (optax Adam, a fresh
              sim ensemble every --resample-every steps). The optimizer never sees
              one realization twice, so it cannot fit a single one's empirical-ILC-
              bias noise -- the fix for the overfit the fixed-CRN path hit.
        lbfgs: FIXED-CRN L-BFGS-B baseline; run to convergence on one finite sample
              it overfits that sample. Kept to exhibit the optimism gap.
      Both track sigma(r) on a separate VALIDATION ensemble each iterate, keep the
      best-on-val design (early stopping), and report GENERALIZATION -- held-out
      gain vs the uniform allocation on independent DISJOINT test ensembles for both
      the final and the best-on-val design. The straight-through gradient is usable
      iff the best-on-val design improves every held-out ensemble.

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
      held-out generalization, recording held-out gain for the final AND the
      best-on-val design (mean/min), the optimism gap, and the steady-state
      per-eval cost. Writes a JSON + a plot of held-out-gain-vs-n_sims (final vs
      best-on-val) and per-eval-time-vs-n_sims. This is the Phase-2 read-off that
      gates B.3: whether the design GENERALIZES (best-on-val min-gain > 0) and
      holds as n_sims rises, and what one value+grad eval costs. Cheap on CPU at
      small n_sims; the real ladder runs on a GPU (``--backend jht``).

  --mode beam:
      The beam-lever sensitivity diagnostic (NOT an optimizer -- a free FWHM has no
      cost penalty and would run to 0). Reports the per-band partials
      d sigma(r)/d FWHM and d sigma(r)/d p (which bands' beams move sigma(r)) and the
      MC-stability (resultant R, per-component CoV) of the beam gradient direction
      across CRN ensembles, via optimize_mapbased.sigma_r_from_beam_design.

  --mode profile:
      Where a design gradient spends its time and its memory, as the input to
      deciding which parts of the forward are worth restructuring. The forward is
      one fused executable, so phases are separated by REGRESSION rather than by
      wall clock: the jitted graph is ``coupling build + n_sims * body +
      covariance/Fisher`` and only the middle term scales with n_sims, so two
      rungs give the per-sim body as the slope and everything else as the
      intercept; the MASTER coupling build is timed standalone to split that
      intercept. Reports compile vs steady state separately, the value+grad tax,
      and peak memory as increments of the (monotone) high-water mark. With
      ``--trace-dir`` it also captures a JAX profiler trace and prints the
      KERNEL-GAP HISTOGRAM -- device time is either inside a kernel or between
      kernels, and the between-fraction is what distinguishes a launch-latency-
      bound workload from an arithmetic-bound one. That question is only posed on
      a GPU backend: augr's CPU transforms are ducc ``pure_callback``s and never
      appear as device kernels.

Tiny CMB-only config (nside=16, no PySM) so the diagnostic is cheap; the gradient
mechanism is foreground-independent. The scientifically interesting FG-driven
allocation needs fg_model="d1s1" at higher nside (heavier) -- deferred to a real
run. Pin BLAS/OMP to 1 thread for reproducible single-core timing.

Usage:
    pixi run python scripts/mapbased_grad_characterization.py --mode demo
    pixi run python scripts/mapbased_grad_characterization.py --mode demo --optimizer lbfgs
    pixi run python scripts/mapbased_grad_characterization.py --mode stability --n-batches 4
    pixi run python scripts/mapbased_grad_characterization.py --mode beam --n-batches 3
    pixi run python scripts/mapbased_grad_characterization.py --mode ladder \
        --n-sims-ladder 12 24 48 96 --backend jht
    pixi run python scripts/mapbased_grad_characterization.py --mode profile
    pixi run -e gpu python scripts/mapbased_grad_characterization.py --mode profile \
        --backend jht --nside 128 --lmax 192 --profile-n-sims 8 16 --trace-dir prof_trace

The map-based sigma(r) objective is wrapped in ``eqx.filter_jit`` over
``(logits, mc_ctx)``, so it compiles ONCE and reuses the executable across all
descent / validation / held-out evals -- and, crucially, across re-drawn CRN
ensembles of the same ``(n_sims, nside, lmax)`` (the PR #28 traced ``mc_ctx``).
That is what makes re-randomizing the CRN every step nearly free.

GPU run (TACC Vista, ``gh`` partition, account JPL-PUB):
    ``pip install jaxht`` (PyPI distribution name; the import is ``jht``) into the
    node Python, then run --mode ladder with --backend jht on one H200. The
    device-aware SHT backend puts the transforms on the GPU; a minimal sbatch
    (``-p gh -A JPL-PUB``) lives in the gitignored scratch launch dir.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import resource
import sys
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
from augr.design_opt import build_design_objectives, held_out_gain, stochastic_design_descent
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


def _static_pieces(nside, lmax, delta_ell: int = 35, ell_per_bin_below: int = 30):
    """Design-INDEPENDENT pieces for the tiny CMB-only config, built once.

    Everything here is fixed across the design optimization: the binning, the
    E/B priors, the true BB transfer denominator, the cleaner, the fiducial
    ``w_inv`` (only used to seed ``var_pix_ref``), the analysis mask, and the
    OptimizationContext. The per-CRN ensemble is built separately by
    :func:`_mc_ctx` so a stochastic descent can re-draw it cheaply (same shapes
    reuse the compiled forward trace; only the sky/noise leaves change).

    The bin schedule defaults to ``SignalModel``'s own ``(ell_per_bin_below=30,
    delta_ell=35)``: per-ℓ bins across the reionization bump, coarse above it. It
    used to be ``(2, 8)``, which is backwards at both ends -- smearing the bump in
    Δℓ=8 chunks while resolving high ℓ far past what any real analysis does. Fixing
    it is not merely cheaper: measured on the analytic pico_like Fisher at
    lmax=1000, ``(30, 35)`` gives σ(r) = 6.74e-5 against ``(2, 8)``'s 7.55e-5, 11%
    TIGHTER on 56 bins rather than 125, and ``(30, 70)`` is indistinguishable from
    ``(30, 35)`` -- bin width above ℓ=30 buys essentially nothing. Bin count also
    sets the Hartlap floor on ``n_sims``, so this is the lever on how the whole
    thing scales with resolution."""
    ell_max = lmax
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
    return {
        "cl_ee": cl_ee,
        "cl_bb": cl_bb,
        "bm": bm,
        "true_b": true_b,
        "cleaner": cleaner,
        "w_inv_fid": w_inv_fid,
        "opt_ctx": opt_ctx,
        "mask": mk.galactic_mask(nside, F_SKY),
        "nside": nside,
        "lmax": lmax,
    }


def _mc_ctx(pieces, base_seed, n_sims, var_pix_ref=None):
    """Build the per-CRN cut-sky MC ensemble at ``base_seed`` from static ``pieces``.

    Pass a frozen ``var_pix_ref`` so the Wiener filter is identical across
    re-draws -- then the only thing that changes between ensembles of the same
    ``(n_sims, nside, lmax)`` is the CRN (sky + noise leaves), which the traced
    forward reuses one compiled executable for."""
    return make_cutsky_mc_context(
        cleaner=pieces["cleaner"],
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
        w_inv=pieces["w_inv_fid"],
        nside=pieces["nside"],
        lmax=pieces["lmax"],
        mask=pieces["mask"],
        cl_ee=pieces["cl_ee"],
        cl_bb_prior_unbeamed=pieces["cl_bb"],
        bin_matrix=pieces["bm"],
        ell_min=2,
        true_bb_binned=pieces["true_b"],
        n_sims=n_sims,
        base_seed=base_seed,
        fg_model=None,
        r_in=0.0,
        var_pix_ref=var_pix_ref,
    )


def build_contexts(
    base_seed, n_sims, *, nside, lmax, var_pix_ref=None, delta_ell=35, ell_per_bin_below=30
):
    """Build (mc_ctx, opt_ctx, cleaner) for a CMB-only tiny config at one CRN seed."""
    pieces = _static_pieces(nside, lmax, delta_ell, ell_per_bin_below)
    mc_ctx = _mc_ctx(pieces, base_seed, n_sims, var_pix_ref=var_pix_ref)
    return mc_ctx, pieces["opt_ctx"], pieces["cleaner"]


# --- disjoint CRN seed allocator + jitted objectives -------------------------
#
# The demo's train / validation / test ensembles must use DISJOINT sim seeds.
# Each ensemble at ``base`` occupies seeds ``[base, base + n_sims]`` (the
# n_sims sims + the var_pix_ref setup clean at ``base + n_sims``). Spacing the
# ensemble bases by ``SEED_STRIDE`` (>> any n_sims) keeps them disjoint. An
# earlier version spaced the held-out seeds by 1 (9001, 9002, 9003), so at
# n_sims=48 the "independent" ensembles shared 47/48 sims -- not independent.
SEED_STRIDE = 100_000
VAL_BASE = SEED_STRIDE  # the single fixed validation ensemble


def _train_base(resample_index, n_sims):
    """Train CRN base for the ``resample_index``-th re-draw: a fresh disjoint block.

    Block k uses seeds ``[k*(n_sims+1), k*(n_sims+1) + n_sims]``; all stay below
    ``VAL_BASE`` (asserted by the caller) so train never collides with val/test."""
    return resample_index * (n_sims + 1)


def _test_base(i):
    """Disjoint base for the i-th held-out TEST ensemble (spaced from train + val)."""
    return (i + 2) * SEED_STRIDE


def _make_objectives(pieces, n_total):
    """Return (value_fn, value_and_grad_fn) for the softmax-allocation objective.

    The objective ``sigma(r)(logits, mc_ctx)`` is jitted over ``(logits, mc_ctx)``
    by :func:`augr.design_opt.build_design_objectives`, so swapping in a fresh CRN
    ensemble of the same ``(n_sims, nside, lmax)`` reuses the one compiled
    executable. ``opt_ctx`` / ``cleaner`` are closed over (fixed across the
    optimization); the gradient is w.r.t. ``logits`` only (the allocation)."""
    opt_ctx = pieces["opt_ctx"]
    cleaner = pieces["cleaner"]
    net0 = jnp.asarray(NET)
    eta0 = jnp.asarray(ETA)

    def _loss(logits, mc_ctx):
        alloc = n_total * jax.nn.softmax(logits)
        return sigma_r_from_noise_design(
            alloc,
            net0,
            eta0,
            MISSION_YEARS,
            mc_ctx=mc_ctx,
            opt_ctx=opt_ctx,
            cleaner=cleaner,
        )

    return build_design_objectives(_loss)


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


def _descent_adam(args, pieces, n_sims, var_pix_ref, value_fn, vg_fn, val_ctx):
    """optax Adam over the softmax logits with RE-RANDOMIZED CRN (lever A).

    Thin wrapper over :func:`augr.design_opt.stochastic_design_descent`: a fresh
    sim ensemble (rolling disjoint base seed) every ``--resample-every`` steps, so
    the optimizer never fits a single realization's empirical-ILC-bias noise. The
    frozen ``var_pix_ref`` keeps the Wiener filter fixed, so re-draws reuse the one
    compiled forward trace. Returns (history, logits_final, logits_best_on_val,
    build_s, per_eval_s); ``history`` is (step, sigma_train, sigma_val).

    ``optax`` is imported here rather than at module scope for the reason
    ``design_opt.stochastic_design_descent`` gives: only the Adam descent needs
    it, and the slim aarch64 ``gpu`` env does not ship it. A module-level import
    made every mode of this script -- including the GPU-only ones -- unimportable
    there."""
    import optax

    n_band = len(N_DET)
    n_resamples = (args.steps - 1) // args.resample_every
    assert _train_base(n_resamples, n_sims) + n_sims < VAL_BASE, (
        f"train seed blocks reach {_train_base(n_resamples, n_sims) + n_sims} >= "
        f"VAL_BASE={VAL_BASE}; reduce --steps / --resample-every span or raise SEED_STRIDE."
    )
    result = stochastic_design_descent(
        value_fn,
        vg_fn,
        jnp.zeros(n_band),
        make_train_ctx=lambda i: _mc_ctx(pieces, _train_base(i, n_sims), n_sims, var_pix_ref),
        val_ctx=val_ctx,
        optimizer=optax.adam(args.lr),
        steps=args.steps,
        resample_every=args.resample_every,
    )
    history = list(
        zip(
            result.steps.tolist(),
            result.train_curve.tolist(),
            result.val_curve.tolist(),
            strict=True,
        )
    )
    # build time folds into the (cheap, untracked) per-step ensemble re-draws.
    return (
        history,
        np.asarray(result.params_final),
        np.asarray(result.params_best),
        0.0,
        result.per_eval_s,
    )


def _descent_lbfgs(args, pieces, n_sims, var_pix_ref, value_fn, vg_fn, val_ctx):
    """scipy L-BFGS-B at FIXED CRN (the overfit-prone baseline).

    Deterministic full-batch descent on one finite sample. Kept to exhibit the
    optimism gap the Adam path closes: validation is tracked per iterate (via the
    callback) so best-on-val early stopping still applies, but the converged
    ``res.x`` is the design that fits the training realization. Returns the same
    tuple as :func:`_descent_adam`."""
    n_band = len(N_DET)
    t_b = time.time()
    train_ctx = _mc_ctx(pieces, 0, n_sims, var_pix_ref)
    build_s = time.time() - t_b
    iterates = [np.zeros(n_band)]
    eval_times = []

    def scipy_vg(x):
        t_e = time.time()
        v, g = vg_fn(jnp.asarray(x), train_ctx)
        eval_times.append(time.time() - t_e)
        return float(v), np.asarray(g, dtype=np.float64)

    def cb(xk):
        iterates.append(np.array(xk))

    res = minimize(
        scipy_vg,
        iterates[0],
        jac=True,
        method="L-BFGS-B",
        callback=cb,
        options={"maxiter": args.maxiter, "maxfun": args.maxiter, "ftol": 1e-9, "gtol": 1e-7},
    )
    iterates.append(np.array(res.x))
    history = []
    for it, x in enumerate(iterates):
        s_tr = float(value_fn(jnp.asarray(x), train_ctx))
        s_va = float(value_fn(jnp.asarray(x), val_ctx))
        history.append((it, s_tr, s_va))
    val_curve = np.array([h[2] for h in history])
    logits_best = iterates[int(np.argmin(val_curve))]
    per_eval_s = float(np.median(eval_times[1:])) if len(eval_times) > 1 else eval_times[0]
    return history, np.asarray(res.x), np.asarray(logits_best), build_s, per_eval_s


def _plot_ucurve(args, n_sims, history, n_to_best):
    """Save the train-vs-val sigma(r) U-curve (the overfit made visible)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(steps, [h[1] for h in history], "o-", color="0.5", label="train sigma(r)")
    ax.plot(steps, [h[2] for h in history], "o-", color="C0", label="val sigma(r)")
    ax.axvline(history[n_to_best][0], color="C3", ls="--", lw=1.0, label="best-on-val")
    ax.set_xlabel("iterate")
    ax.set_ylabel("sigma(r)")
    ax.set_title(f"{args.optimizer} n_sims={n_sims}: train vs val")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{args.out_prefix}_ucurve_{args.optimizer}_n{n_sims}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def run_demo(args, var_pix_ref, *, n_sims=None, return_metrics=False):
    """Fixed-budget allocation descent with validation-gated early stopping.

    Optimizes the per-band detector allocation (``n_det = N_total *
    softmax(logits)``) under a fixed total budget. The optimizer is selected by
    ``--optimizer``:

      adam  (default, lever A): optax Adam with RE-RANDOMIZED CRN -- a fresh sim
            ensemble every ``--resample-every`` steps. The optimizer never sees
            the same realization twice, so it cannot fit a single one's
            empirical-ILC-bias noise; the descent only persists in directions
            that help on average (the stochastic-approximation protocol a real
            BOED loop should use).

      lbfgs (baseline): scipy L-BFGS-B at FIXED CRN -- deterministic, fast, and
            run to convergence on a finite sample it overfits that sample. Kept
            to exhibit the optimism gap the adam path closes.

    BOTH paths (lever B) track sigma(r) on a separate fixed VALIDATION ensemble
    each iterate, keep the best-on-val design (early stopping), and report
    held-out gain on independent disjoint TEST ensembles for BOTH the final and
    the best-on-val design. ``n_sims`` overrides ``args.n_sims`` (the ladder
    passes one rung); ``return_metrics`` returns the per-rung dict."""
    n_sims = args.n_sims if n_sims is None else n_sims
    n_total = float(sum(N_DET))
    n_band = len(N_DET)
    logits0 = np.zeros(n_band)

    pieces = _static_pieces(args.nside, args.lmax, args.delta_ell, args.ell_per_bin_below)
    value_fn, vg_fn = _make_objectives(pieces, n_total)

    # Disjoint validation + test ensembles, all sharing the frozen var_pix_ref filter.
    val_ctx = _mc_ctx(pieces, VAL_BASE, n_sims, var_pix_ref)
    test_ctxs = [
        _mc_ctx(pieces, _test_base(i), n_sims, var_pix_ref) for i in range(args.n_test_ensembles)
    ]

    descent = _descent_adam if args.optimizer == "adam" else _descent_lbfgs
    print(f"Running {args.optimizer} descent (n_sims={n_sims}) ...")
    history, logits_final, logits_best, build_s, per_eval_s = descent(
        args, pieces, n_sims, var_pix_ref, value_fn, vg_fn, val_ctx
    )

    def alloc(lg):
        return np.asarray(n_total * jax.nn.softmax(jnp.asarray(lg)))

    s_val_unif = float(value_fn(jnp.asarray(logits0), val_ctx))
    s_val_final = float(value_fn(jnp.asarray(logits_final), val_ctx))
    s_val_best = float(value_fn(jnp.asarray(logits_best), val_ctx))
    val_impr_final = 100.0 * (s_val_unif - s_val_final) / s_val_unif
    val_impr_best = 100.0 * (s_val_unif - s_val_best) / s_val_unif
    n_to_best = int(np.argmin(np.array([h[2] for h in history])))

    gains_final = held_out_gain(value_fn, test_ctxs, logits0, logits_final)
    gains_best = held_out_gain(value_fn, test_ctxs, logits0, logits_best)
    # Generalization gap: how much the validation-claimed improvement of the
    # CONVERGED design overstates its worst-case held-out improvement. Positive =>
    # the design looked better in-protocol than it actually generalizes (overfit).
    # Robust for both paths (min over disjoint test ensembles, not a noisy mean).
    gen_gap_final = val_impr_final - float(gains_final.min())

    print(f"  steps={len(history)}  [build {build_s:.0f}s, steady eval {per_eval_s:.1f}s]")
    print(f"  n_det uniform     = {alloc(logits0)}")
    print(f"  n_det final       = {alloc(logits_final)}")
    print(f"  n_det best-on-val = {alloc(logits_best)}  (iterate {n_to_best})")
    print(f"  val improvement:  final={val_impr_final:+.2f}%   best-on-val={val_impr_best:+.2f}%")
    print(f"  generalization gap (val - held-out min at final): {gen_gap_final:+.2f}%")
    print("  held-out TEST gain vs uniform:")
    print(
        f"    final       : mean={gains_final.mean():+.2f}%  min={gains_final.min():+.2f}%  "
        f"(n={len(gains_final)})"
    )
    print(
        f"    best-on-val : mean={gains_best.mean():+.2f}%  min={gains_best.min():+.2f}%  "
        f"(n={len(gains_best)})"
    )
    if gains_best.min() > 0:
        print("  VERDICT: best-on-val design improves EVERY held-out ensemble => the")
        print("  descent generalizes; the straight-through gradient is usable.")
    else:
        print("  VERDICT: best-on-val held-out gain not all-positive => either no real")
        print("  design leverage at this config, or the protocol still overfits.")

    _plot_ucurve(args, n_sims, history, n_to_best)

    if return_metrics:
        return {
            "n_sims": int(n_sims),
            "optimizer": args.optimizer,
            "val_impr_final_pct": float(val_impr_final),
            "val_impr_best_pct": float(val_impr_best),
            "gen_gap_final_pct": float(gen_gap_final),
            "heldout_final_mean_pct": float(gains_final.mean()),
            "heldout_final_min_pct": float(gains_final.min()),
            "heldout_best_mean_pct": float(gains_best.mean()),
            "heldout_best_min_pct": float(gains_best.min()),
            "n_to_best": n_to_best,
            "build_s": float(build_s),
            "per_eval_s": float(per_eval_s),
            "alloc_final": alloc(logits_final).tolist(),
            "alloc_best": alloc(logits_best).tolist(),
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
        cal_ctx, _, _ = build_contexts(
            0, n_sims, nside=args.nside, lmax=args.lmax,
            delta_ell=args.delta_ell, ell_per_bin_below=args.ell_per_bin_below,
        )
        rows.append(run_demo(args, cal_ctx.var_pix_ref, n_sims=n_sims, return_metrics=True))

    print("\n=== n_sims ladder summary ===")
    print(
        f"  optimizer={args.optimizer} nside={args.nside} lmax={args.lmax} "
        f"backend={sht.get_sht_backend()}"
    )
    print(
        f"  {'n_sims':>7} {'ho_fin_min':>10} {'ho_best_min':>11} {'ho_best_mean':>12} "
        f"{'gen_gap':>9} {'eval_s':>8}"
    )
    for m in rows:
        print(
            f"  {m['n_sims']:>7d} {m['heldout_final_min_pct']:>10.1f} "
            f"{m['heldout_best_min_pct']:>11.1f} {m['heldout_best_mean_pct']:>12.1f} "
            f"{m['gen_gap_final_pct']:>9.1f} {m['per_eval_s']:>8.1f}"
        )
    # Success = best-on-val held-out gain positive on EVERY test ensemble at a rung.
    closed = [m["n_sims"] for m in rows if m["heldout_best_min_pct"] > 0]
    if closed:
        print(f"  best-on-val generalizes (min-gain > 0) at n_sims in {sorted(closed)}.")
    else:
        print("  best-on-val held-out min-gain <= 0 at every rung -- no clean leverage.")
    print(
        "  (compare ho_fin_min vs ho_best_min: a large gap = the converged design "
        "overfits, early stopping recovers it.)"
    )

    payload = {
        "config": {
            "optimizer": args.optimizer,
            "nside": args.nside,
            "lmax": args.lmax,
            "maxiter": args.maxiter,
            "steps": args.steps,
            "lr": args.lr,
            "resample_every": args.resample_every,
            "n_test_ensembles": args.n_test_ensembles,
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
    axL.plot(ns, [m["heldout_final_min_pct"] for m in rows], "o-", color="0.5", label="final (min)")
    axL.plot(
        ns, [m["heldout_best_mean_pct"] for m in rows], "o-", color="C0", label="best-on-val (mean)"
    )
    axL.plot(
        ns, [m["heldout_best_min_pct"] for m in rows], "o-", color="C3", label="best-on-val (min)"
    )
    axL.set_xlabel("n_sims")
    axL.set_ylabel("held-out sigma(r) gain [%]")
    axL.set_title(f"{args.optimizer}: does the design generalize?")
    axL.legend(fontsize=8)
    axR.plot(ns, [m["per_eval_s"] for m in rows], "o-", color="C2")
    axR.set_xlabel("n_sims")
    axR.set_ylabel("steady-state value+grad eval [s]")
    axR.set_title(f"per-eval cost ({sht.get_sht_backend()})")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=140)
    print(f"  wrote {args.out_prefix}.png")


# --- mode: profile -----------------------------------------------------------
#
# Where does a design gradient spend its time and its memory? The forward is one
# fused executable, so the phases cannot be read off a wall clock directly. Two
# independent decompositions are reported so they can be checked against each
# other:
#
#   * n_sims regression. The jitted forward is `coupling build + n_sims * body +
#     covariance/Fisher`, and only the middle term scales with n_sims. Timing two
#     rung sizes gives the per-sim body as the slope and everything else as the
#     intercept; the MASTER coupling build is then timed standalone to split that
#     intercept, leaving covariance/Fisher as the remainder.
#   * kernel-gap histogram, from a JAX profiler trace. Device-stream time is
#     either inside a kernel or between kernels, and a large between-fraction is
#     the signature of a launch-latency-bound workload -- which is what the July
#     H200 verdict asserted from a single unprofiled wall-clock comparison.
#
# Peak memory is a monotone high-water mark on both backends, so what is reported
# per phase is the INCREMENT of that mark, never an independent per-phase peak.


def _peak_device_gb():
    """Peak device bytes if the backend reports them, else None (the CPU backend)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for k in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
        if k in stats:
            return stats[k] / 1e9
    return None


def _peak_rss_gb():
    """Peak RSS of this process. ``ru_maxrss`` is bytes on macOS, kilobytes on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def _timed(fn, *args, repeat: int):
    """``(first-call seconds including compile, median steady-state seconds)``.

    Handing in a FRESH function object is the caller's job: ``jax.jit`` caches on
    the function object, so reusing one across variants silently re-runs the
    first executable and every row comes back 1.0x (reference_jax_benchmark_traps).
    """
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    del out
    return first, float(np.median(ts))


def _coupling_build_fn(mc_ctx):
    """A fresh jitted MASTER coupling build for ``mc_ctx``'s mask, traced in the mask.

    This is the same call ``_mc_cutsky_cov_master`` makes once per evaluation, so
    timing it alone splits the n_sims-regression intercept. The mask is the traced
    argument (it is the sky-coverage design coordinate), which also stops XLA from
    constant-folding the whole build away.
    """
    from augr.instrument import beam_bl
    from augr.pseudo_cl_jax import MasterBBJax

    ells = jnp.arange(int(mc_ctx.lmax) + 1, dtype=float)
    bl = beam_bl(ells, float(min(BEAMS)))

    def build(mask):
        m = MasterBBJax.build(
            mask,
            bin_edges=mc_ctx.master_bin_edges,
            nside=mc_ctx.nside,
            lmax=mc_ctx.lmax,
            lmax_mask=mc_ctx.lmax_mask,
            beam_bl=bl,
        )
        return m.window

    return jax.jit(build)


def _kernel_gap_histogram(trace_dir):
    """Kernel durations and inter-kernel gaps from the newest trace under ``trace_dir``.

    Returns None when the trace carries no device stream. On CPU that absence is
    itself the answer: augr's CPU transforms are ducc ``pure_callback``s, so they
    never appear as device kernels and a gap histogram cannot be formed.
    """
    paths = sorted(glob.glob(os.path.join(trace_dir, "**", "*.trace.json.gz"), recursive=True))
    if not paths:
        return None
    with gzip.open(paths[-1], "rt") as fh:
        events = json.load(fh).get("traceEvents", [])
    pid_name, tid_name = {}, {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") == "process_name":
            pid_name[e.get("pid")] = e.get("args", {}).get("name", "")
        elif e.get("name") == "thread_name":
            tid_name[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "")
    # Kernels live on a device process, on the thread the profiler calls "XLA Ops".
    # That thread name is not contractual, so fall back to every thread on a device
    # process except the known non-kernel ones -- and REPORT which stream was used,
    # because an empty histogram from a naming change and a genuinely gap-free run
    # would otherwise look identical.
    on_device = [
        e for e in events if e.get("ph") == "X" and "/device:" in pid_name.get(e.get("pid"), "")
    ]
    if not on_device:
        return None
    not_kernels = {"XLA Modules", "Steps", "Launch Stats", "Source", "Framework Ops"}
    ops = [e for e in on_device if tid_name.get((e.get("pid"), e.get("tid"))) == "XLA Ops"]
    stream = "XLA Ops"
    if not ops:
        ops = [
            e
            for e in on_device
            if tid_name.get((e.get("pid"), e.get("tid")), "") not in not_kernels
        ]
        streams = sorted({tid_name.get((e.get("pid"), e.get("tid")), "?") for e in ops})
        stream = "fallback: " + ", ".join(streams)
    if not ops:
        return None
    ops.sort(key=lambda e: e["ts"])
    ts = np.array([float(e["ts"]) for e in ops])
    dur = np.array([float(e.get("dur", 0.0)) for e in ops])
    gaps = np.clip(ts[1:] - (ts[:-1] + dur[:-1]), 0.0, None)
    span = float((ts[-1] + dur[-1]) - ts[0])
    by_op = {}
    for e, d in zip(ops, dur, strict=True):
        by_op[e["name"]] = by_op.get(e["name"], 0.0) + float(d)
    return {
        "trace": paths[-1],
        "stream": stream,
        "n_kernels": len(ops),
        "kernel_us": float(dur.sum()),
        "gap_us": float(gaps.sum()),
        "span_us": span,
        "gap_fraction": float(gaps.sum() / span) if span > 0 else float("nan"),
        "gap_median_us": float(np.median(gaps)) if gaps.size else 0.0,
        "gap_p90_us": float(np.percentile(gaps, 90)) if gaps.size else 0.0,
        "kernel_median_us": float(np.median(dur)),
        "gaps": gaps,
        "top_ops": sorted(by_op.items(), key=lambda kv: -kv[1])[:10],
    }


_GAP_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def _print_gap_histogram(h):
    print(f"  trace           : {h['trace']}")
    print(f"  device stream   : {h['stream']}")
    print(f"  kernels         : {h['n_kernels']}  (median {h['kernel_median_us']:.1f} us)")
    print(f"  in kernel       : {h['kernel_us'] / 1e3:.1f} ms")
    print(f"  between kernels : {h['gap_us'] / 1e3:.1f} ms")
    print(
        f"  GAP FRACTION    : {h['gap_fraction'] * 100:.1f}% of the {h['span_us'] / 1e3:.1f} ms span"
        f"   (median gap {h['gap_median_us']:.1f} us, p90 {h['gap_p90_us']:.1f} us)"
    )
    gaps = h["gaps"]
    print("  gap histogram (us):")
    lo = 0.0
    for hi in (*_GAP_BUCKETS, float("inf")):
        n = int(np.sum((gaps >= lo) & (gaps < hi)))
        frac = n / max(gaps.size, 1)
        label = f"  {lo:>5.0f}-{hi:<5.0f}" if np.isfinite(hi) else f"  {lo:>5.0f}+     "
        print(f"  {label} {n:>7d}  {'#' * round(40 * frac)}")
        lo = hi
    print("  top device ops by total time:")
    for name, us in h["top_ops"]:
        print(f"    {us / 1e3:>9.1f} ms  {name[:78]}")


def _phase_split(rows, split, n_lo, n_hi, c_steady):
    """Per-sim slope and once-per-eval intercept, from two steady-state rungs.

    The jitted forward is ``coupling build + n_sims * body + covariance/Fisher``
    and only the middle term scales, so the slope is the per-sim body and the
    intercept is everything else; ``c_steady`` splits that intercept.
    """
    print("\n=== phase split (n_sims regression on the steady-state forward) ===")
    print(f"  {'':<11s} {'per-sim body':>14s} {'once-per-eval':>15s} {'of which coupling':>19s}")
    for what in ("value", "value+grad"):
        t_lo, t_hi = rows[(what, n_lo)][1], rows[(what, n_hi)][1]
        per_sim = (t_hi - t_lo) / (n_hi - n_lo)
        fixed = t_lo - per_sim * n_lo
        split[what] = {"per_sim_s": per_sim, "fixed_s": fixed, "coupling_s": c_steady}
        print(
            f"  {what:<11s} {per_sim:>13.4f}s {fixed:>14.3f}s {c_steady:>18.3f}s"
            f"   (residual = covariance/Fisher {fixed - c_steady:+.3f}s)"
        )
    print(
        "  A negative residual means the two-point regression has resolved the "
        "intercept no better than its own noise -- widen --profile-n-sims or "
        "--repeat before reading it."
    )


def _peak_fp64_flops(n: int = 4096, repeat: int = 3) -> float:
    """Realized fp64 matmul rate on this device [FLOP/s] -- measured, not a vendor figure."""
    a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.float64)
    mm = jax.jit(lambda x: x @ x)
    jax.block_until_ready(mm(a))
    t0 = time.perf_counter()
    for _ in range(repeat):
        jax.block_until_ready(mm(a))
    return 2 * n**3 / ((time.perf_counter() - t0) / repeat)


def _arithmetic_bound(fn, logits, ctx, n_sims, steady_s, rate):
    """Bracket the share of runtime that is fp64 arithmetic, without a profiler.

    ``cost_analysis`` counts a loop body ONCE regardless of trip count, so the
    per-sim scan contributes a single pass to the static figure. That turns into a
    bracket rather than a defect: true dynamic flops lie between the static count
    (body runs once) and ``n_sims`` times it (body runs every sim). When both ends
    of the bracket are negligible against the measured time, the workload is not
    arithmetic-bound and no profiler is needed to say so.

    Costs one extra compile: the cost analysis needs a ``Compiled``, and re-lowering
    does not hit the in-process cache (measured).
    """
    compiled = jax.jit(fn).lower(logits, ctx).compile()
    ca = compiled.cost_analysis()
    ca = ca[0] if isinstance(ca, list | tuple) else ca
    static = float(ca.get("flops", float("nan")))
    lo, hi = static / rate, static * n_sims / rate
    return {
        "static_flops": static,
        "peak_fp64_flops_per_s": rate,
        "arith_s_low": lo,
        "arith_s_high": hi,
        "arith_frac_low": lo / steady_s,
        "arith_frac_high": hi / steady_s,
    }


def run_profile(args):
    """Phase-resolved wall time and peak memory for the map-based design gradient.

    Reports, for value and for value+grad: context build, MASTER coupling build,
    per-sim body (n_sims slope), and covariance/Fisher (the residual intercept);
    plus a kernel-gap histogram when ``--trace-dir`` is given and the backend has
    a device stream."""
    n_lo, n_hi = args.profile_n_sims
    if n_hi <= n_lo:
        raise SystemExit(f"--profile-n-sims needs an increasing pair, got {n_lo} {n_hi}")
    backend = sht.get_sht_backend()
    print("=== configuration ===")
    print(f"  sht backend     : {backend}")
    print(f"  jax backend     : {jax.default_backend()}")
    for d in jax.devices():
        print(f"  device          : {d} kind={getattr(d, 'device_kind', '?')}")
    print(f"  nside / lmax    : {args.nside} / {args.lmax}")
    print(f"  n_sims rungs    : {n_lo}, {n_hi}   (repeat {args.repeat})")
    print(f"  rss at entry    : {_peak_rss_gb():.2f} GB", flush=True)

    rows = {}
    marks = []

    def mark(label):
        marks.append((label, _peak_device_gb(), _peak_rss_gb()))

    mark("entry")

    print("\n=== phase: context build (outside the jit) ===", flush=True)
    t0 = time.perf_counter()
    pieces = _static_pieces(args.nside, args.lmax, args.delta_ell, args.ell_per_bin_below)
    t_static = time.perf_counter() - t0
    # Both rungs must clear the Hartlap floor, or the low one dies inside the
    # forward AFTER paying its compile -- which on a GPU queue is the whole job.
    n_bins = int(np.asarray(pieces["bm"]).shape[0])
    if n_lo <= n_bins + 2:
        raise SystemExit(
            f"--profile-n-sims {n_lo} {n_hi}: the low rung is at or below the Hartlap "
            f"floor (n_sims > n_bins + 2 = {n_bins + 2} at lmax={args.lmax}, "
            f"{n_bins} bins). Raise it; the MC covariance is refused below that."
        )
    print(f"  bins / Hartlap floor  {n_bins} / n_sims > {n_bins + 2}")
    mark("static pieces")
    ctxs = {}
    t_ctx = {}
    for n in (n_lo, n_hi):
        t0 = time.perf_counter()
        ctxs[n] = jax.block_until_ready(_mc_ctx(pieces, 0, n))
        t_ctx[n] = time.perf_counter() - t0
        mark(f"mc_ctx n_sims={n}")
        print(f"  mc_ctx(n_sims={n:>3d})  {t_ctx[n]:8.2f} s", flush=True)
    print(f"  static pieces     {t_static:8.2f} s")

    print("\n=== phase: MASTER coupling build (once per evaluation, in-trace) ===", flush=True)
    build = _coupling_build_fn(ctxs[n_lo])
    mask = ctxs[n_lo].mask
    c_first, c_steady = _timed(build, mask, repeat=args.repeat)
    mark("coupling build")
    print(f"  compile + first   {c_first:8.2f} s")
    print(f"  steady state      {c_steady:8.3f} s")
    del build
    jax.clear_caches()

    print("\n=== phase: full forward, value and value+grad ===", flush=True)
    n_total = float(sum(N_DET))
    logits = jnp.zeros(len(FREQS))
    # A fresh function object per variant, because jax.jit caches on the object and
    # would otherwise re-run the first executable for every row. The exception is
    # the last variant: the trace leg below needs exactly that function at exactly
    # that n_sims, so it is kept alive rather than recompiled -- which at nside=128
    # is ~7 min of a job spent measuring nothing.
    traced_vg = None
    if args.trace_only:
        if not args.trace_dir:
            raise SystemExit("--trace-only needs --trace-dir; there is nothing else to do.")
        print("  --trace-only: skipping the timing rungs.", flush=True)
    plan = [("value+grad", n_hi)] if args.trace_only else [
        (w, n) for w in ("value", "value+grad") for n in (n_lo, n_hi)
    ]
    for what, n in plan:
        value_fn, vg_fn = _make_objectives(pieces, n_total)
        fn = value_fn if what == "value" else vg_fn
        first, steady = _timed(fn, logits, ctxs[n], repeat=args.repeat)
        rows[(what, n)] = (first, steady)
        mark(f"{what} n_sims={n}")
        print(
            f"  {what:<11s} n_sims={n:>3d}   compile+first {first:8.2f} s"
            f"   steady {steady:8.3f} s",
            flush=True,
        )
        keep = (args.trace_dir or not args.skip_arith) and what == "value+grad" and n == n_hi
        if keep:
            traced_vg = fn
        del value_fn, vg_fn, fn
        if not keep:
            jax.clear_caches()

    split, grad_tax = {}, float("nan")
    if args.trace_only:
        print("\n  (phase split skipped: --trace-only ran one rung.)")
    else:
        _phase_split(rows, split, n_lo, n_hi, c_steady)
        grad_tax = rows[("value+grad", n_hi)][1] / max(rows[("value", n_hi)][1], 1e-12)
        print(f"  value+grad / value at n_sims={n_hi}: {grad_tax:.2f}x")

    print("\n=== peak memory (monotone high-water mark; increments) ===")
    print(f"  {'phase':<24s} {'device GB':>10s} {'d(device)':>10s} {'rss GB':>8s} {'d(rss)':>8s}")
    prev_dev, prev_rss = None, None
    for label, dev, rss in marks:
        d_dev = "-" if (dev is None or prev_dev is None) else f"{dev - prev_dev:+.3f}"
        d_rss = "-" if prev_rss is None else f"{rss - prev_rss:+.3f}"
        dev_s = "-" if dev is None else f"{dev:.3f}"
        print(f"  {label:<24s} {dev_s:>10s} {d_dev:>10s} {rss:>8.2f} {d_rss:>8s}")
        prev_dev, prev_rss = dev, rss
    if marks[-1][1] is None:
        print("  (device counters absent: the CPU backend does not report them; rss is the gate.)")

    arith = None
    if not args.skip_arith and traced_vg is not None:
        print("\n=== arithmetic intensity (no profiler; one extra compile) ===", flush=True)
        rate = _peak_fp64_flops()
        print(f"  measured fp64   : {rate / 1e12:.2f} TFLOP/s", flush=True)
        steady_ref = rows[("value+grad", n_hi)][1]
        arith = _arithmetic_bound(traced_vg, logits, ctxs[n_hi], n_hi, steady_ref, rate)
        print(f"  static flops    : {arith['static_flops']:.4g}  (scan body counted once)")
        if backend == "ducc":
            print(
                "  NOTE: on the ducc backend the transforms are pure_callbacks into C++, "
                "so their flops never reach cost_analysis and this share is a LOWER "
                "bound on a number that is already small. Read it on jht."
            )
        print(
            f"  arithmetic      : {arith['arith_s_low'] * 1e3:.3f} ms to "
            f"{arith['arith_s_high'] * 1e3:.3f} ms of the {steady_ref:.2f} s measured"
        )
        print(
            f"  ARITHMETIC SHARE: {arith['arith_frac_low'] * 100:.5f}% to "
            f"{arith['arith_frac_high'] * 100:.5f}%"
            + (
                "   -> NOT arithmetic-bound: the cost is kernel launches and memory"
                if arith["arith_frac_high"] < 0.05
                else "   -> arithmetic is a real share; read the bracket"
            ),
            flush=True,
        )

    # The timings and the memory table are complete at this point, and the trace
    # leg below is the part that can fail: job 986936 lost 88 min of GB200 time to
    # a CUDA launch failure there, with every number above already printed to a log
    # and none of it written anywhere machine-readable. So the payload is written
    # HERE, and rewritten with the histogram if the trace leg survives.
    def _payload(hist):
        return {
            "config": {
                "sht_backend": backend,
                "jax_backend": jax.default_backend(),
                "devices": [str(d) for d in jax.devices()],
                "nside": args.nside,
                "lmax": args.lmax,
                "n_sims": [n_lo, n_hi],
                "n_bins": n_bins,
                "repeat": args.repeat,
            },
            "context_build_s": {"static": t_static, **{str(k): v for k, v in t_ctx.items()}},
            "coupling_build_s": {"compile_first": c_first, "steady": c_steady},
            "forward_s": {
                f"{w}/{n}": {"compile_first": v[0], "steady": v[1]} for (w, n), v in rows.items()
            },
            "phase_split_s": split,
            "grad_tax": grad_tax,
            "peak_gb": [{"phase": lb, "device": dv, "rss": rs} for lb, dv, rs in marks],
            "arithmetic": arith,
            "kernel_gap": None
            if hist is None
            else {k: v for k, v in hist.items() if k != "gaps"},
        }

    out = f"{args.out_prefix}_profile.json"

    def _write(hist):
        with open(out, "w") as fh:
            json.dump(_payload(hist), fh, indent=2)
        print(f"  wrote {out}", flush=True)

    print()
    _write(None)

    hist = None
    if args.trace_dir:
        print(
            f"\n=== kernel-gap histogram (value+grad, {args.trace_repeat} traced call(s)) ===",
            flush=True,
        )
        vg_fn = traced_vg
        # Timed, because "reused the executable" and "silently recompiled" differ
        # only in this number: it must land near the steady state above, not near
        # the compile.
        t0 = time.perf_counter()
        jax.block_until_ready(vg_fn(logits, ctxs[n_hi]))
        t_warm = time.perf_counter() - t0
        steady_ref = rows[("value+grad", n_hi)][1]
        print(
            f"  warm-up call    : {t_warm:.3f} s vs steady {steady_ref:.3f} s and "
            f"compile {rows[('value+grad', n_hi)][0]:.1f} s"
            + ("  (reused)" if t_warm < 3 * steady_ref else "  (RECOMPILED?)"),
            flush=True,
        )
        # Reproduced twice on TACC Vista `gb` (GB200): the call below fails with
        # CUDA_ERROR_LAUNCH_FAILED while the IDENTICAL call on the line above, same
        # executable and same inputs, succeeds -- 986936 and 987876, naming a
        # different fused kernel each time (loop_dynamic_slice_fusion_275,
        # input_reduce_fusion_234). The only difference is that the profiler is
        # active. In 987876 the executable was reused, not recompiled, and had
        # already run four times at 105 s, so it is not a compile or a warm-up
        # effect. JAX warns at startup on these nodes that cuBLAS < 13.2 frees TMEM
        # buffers multiple times when a kernel runs concurrently with another --
        # A jax-only reproducer -- matmul, lax.scan, and grad through that scan, five
        # calls each inside the profiler -- runs CLEAN on gb (job 988366,
        # scripts/profiler_smoke.py), so the profiler is not broken on these nodes
        # and this is not a site bug to file. What differs here is volume: the trace
        # leg was capturing three calls of ~105 s each. Hence --trace-repeat,
        # default 1; the histogram wants one call's kernel sequence, not an average.
        try:
            with jax.profiler.trace(args.trace_dir):
                for _ in range(args.trace_repeat):
                    jax.block_until_ready(vg_fn(logits, ctxs[n_hi]))
            hist = _kernel_gap_histogram(args.trace_dir)
        except Exception as exc:  # report what was kept, then re-raise
            print(f"  TRACE LEG FAILED: {type(exc).__name__}: {exc}", flush=True)
            print(
                f"  The timings and the memory table are already in {out} and are "
                "unaffected -- only the kernel-gap histogram is missing. Re-run with "
                "--trace-dir alone against a warm process, or drop it.",
                flush=True,
            )
            raise
        if hist is None:
            print("  no device stream in the trace.")
            print("  On CPU this is the answer, not a failure: augr's ducc transforms are")
            print("  pure_callbacks into C++, so they are invisible as device kernels and")
            print("  the launch-latency question is not even posed on this backend.")
        else:
            _print_gap_histogram(hist)
        del vg_fn
        jax.clear_caches()
        _write(hist)


def run_nside_ladder(args):
    """Measure how the per-sim body scales with resolution, instead of assuming it.

    Transform cost is often quoted as ``nside^3`` (an SHT at ``lmax proportional to
    nside``), and the sim count Hartlap forces adds another power, so the projected
    cost of a resolution is extremely sensitive to an exponent nobody has measured.
    This fits it: at each nside, time value+grad at two sim counts placed just above
    that resolution's own Hartlap floor, take the per-sim slope, and regress
    ``log(per_sim)`` on ``log(nside)``.

    Reports the fit and its residuals. Two points would give an exponent with no way
    to know it is wrong, so this wants three or more."""
    rows = []
    for nside in args.nside_ladder:
        lmax = round(args.lmax_factor * nside)
        pieces = _static_pieces(nside, lmax, args.delta_ell, args.ell_per_bin_below)
        n_bins = int(np.asarray(pieces["bm"]).shape[0])
        n_lo = n_bins + 3  # just clears the Hartlap floor at THIS resolution
        n_hi = 2 * n_lo
        print(
            f"\n########## nside={nside} lmax={lmax} bins={n_bins} "
            f"sims={n_lo},{n_hi} ##########",
            flush=True,
        )
        steady = {}
        for n in (n_lo, n_hi):
            ctx = jax.block_until_ready(_mc_ctx(pieces, 0, n))
            _, vg_fn = _make_objectives(pieces, float(sum(N_DET)))
            first, s = _timed(vg_fn, jnp.zeros(len(FREQS)), ctx, repeat=args.repeat)
            steady[n] = s
            print(
                f"  value+grad n_sims={n:>4d}  compile+first {first:8.1f} s  steady {s:8.3f} s",
                flush=True,
            )
            del ctx, vg_fn
            jax.clear_caches()
        per_sim = (steady[n_hi] - steady[n_lo]) / (n_hi - n_lo)
        print(f"  per-sim body: {per_sim:.4f} s", flush=True)
        rows.append({"nside": nside, "lmax": lmax, "n_bins": n_bins,
                     "n_sims": [n_lo, n_hi], "steady_s": [steady[n_lo], steady[n_hi]],
                     "per_sim_s": per_sim})

    print("\n=== measured resolution scaling ===")
    print(f"  {'nside':>6} {'bins':>5} {'per-sim':>10}")
    for r in rows:
        print(f"  {r['nside']:>6} {r['n_bins']:>5} {r['per_sim_s']:>9.4f}s")
    fit = None
    if len(rows) >= 2:
        x = np.log(np.array([r["nside"] for r in rows], dtype=float))
        y = np.log(np.array([r["per_sim_s"] for r in rows], dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        fit = {"exponent": float(slope), "log_residuals": resid.tolist()}
        print(f"  per_sim ~ nside^{slope:.3f}   (max |log resid| {np.abs(resid).max():.3f})")
        if len(rows) == 2:
            print("  TWO POINTS: this exponent has no residual to check it against.")
        print(
            f"  An EVALUATION costs this times n_sims, and n_sims is set by Hartlap "
            f"from n_bins -- so the evaluation exponent is nside^{slope + 1:.3f} ONLY "
            "while the bin schedule keeps n_bins proportional to lmax."
        )
        print(
            "  That proportionality is a hardcoded delta_ell, not a physical fact: "
            "_static_pieces fixes delta_ell=8 to lmax (SignalModel's own default is "
            "35, and it accepts explicit ell_bins). Coarsening high-ell scales the "
            "COEFFICIENT down; only a schedule whose bin count saturates with lmax "
            "-- log spacing above ell~300, or a cap -- removes the extra power. "
            "Gate any such change on the design gradient, not just sigma(r)."
        )
    payload = {"config": {"sht_backend": sht.get_sht_backend(),
                          "jax_backend": jax.default_backend(),
                          "lmax_factor": args.lmax_factor, "repeat": args.repeat},
               "rungs": rows, "fit": fit}
    out = f"{args.out_prefix}_nside_ladder.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["demo", "stability", "beam", "both", "ladder", "profile", "nside-ladder"],
        default="demo",
    )
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
    p.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs"],
        default="adam",
        help="demo: adam = re-randomized-CRN stochastic descent (the fix); "
        "lbfgs = fixed-CRN baseline (exhibits the optimism gap).",
    )
    p.add_argument("--steps", type=int, default=60, help="demo (adam): number of Adam steps")
    p.add_argument(
        "--lr", type=float, default=0.05, help="demo (adam): Adam learning rate on logits"
    )
    p.add_argument(
        "--resample-every",
        type=int,
        default=1,
        help="demo (adam): re-draw the train CRN ensemble every K steps (1 = fresh each step).",
    )
    p.add_argument("--maxiter", type=int, default=12, help="demo (lbfgs): L-BFGS-B max iters")
    p.add_argument(
        "--n-test-ensembles",
        type=int,
        default=3,
        help="demo: # disjoint held-out TEST ensembles for the generalization check.",
    )
    p.add_argument("--backend", choices=["ducc", "jht"], default="ducc")
    p.add_argument(
        "--out-prefix", default="grad_char_ladder", help="ladder/demo: JSON/PNG output path prefix"
    )
    p.add_argument(
        "--profile-n-sims",
        type=int,
        nargs=2,
        default=[6, 12],
        metavar=("LO", "HI"),
        help="profile: the two n_sims rungs the per-sim/fixed split is regressed on.",
    )
    p.add_argument(
        "--repeat", type=int, default=3, help="profile: steady-state repeats per timing (median)."
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        help="profile: capture a JAX profiler trace here and report the kernel-gap "
        "histogram. Needs a device stream, so it is a GPU-backend diagnostic.",
    )
    p.add_argument(
        "--delta-ell",
        type=int,
        default=35,
        help="bin width above --ell-per-bin-below (SignalModel's default; the "
        "measured sigma(r) is insensitive to it above ell=30).",
    )
    p.add_argument(
        "--ell-per-bin-below",
        type=int,
        default=30,
        help="per-ell bins below this multipole -- the reionization bump, which is "
        "where a space mission's constraint actually lives.",
    )
    p.add_argument(
        "--nside-ladder",
        type=int,
        nargs="+",
        default=[64, 128, 192],
        help="nside-ladder: resolutions to measure the per-sim scaling exponent over.",
    )
    p.add_argument(
        "--lmax-factor",
        type=float,
        default=1.5,
        help="nside-ladder: lmax = factor * nside at every rung, so the exponent is "
        "measured along the line the production configs actually sit on.",
    )
    p.add_argument(
        "--skip-arith",
        action="store_true",
        help="profile: skip the arithmetic-intensity bound. It costs one extra "
        "compile (the cost analysis needs a Compiled and re-lowering does not hit "
        "the cache), and it is the no-profiler answer to launch- vs arithmetic-bound.",
    )
    p.add_argument(
        "--trace-repeat",
        type=int,
        default=1,
        help="profile: calls to capture inside the profiler (default 1). The gap "
        "histogram needs one call's kernel sequence, not an average, and tracing a "
        "105 s call three times is what appears to overrun the device tracer.",
    )
    p.add_argument(
        "--trace-only",
        action="store_true",
        help="profile: skip the timing rungs and compile only value+grad at the high "
        "rung, for a re-run that just wants the kernel-gap histogram. At nside=128 "
        "that is one ~23 min compile instead of four.",
    )
    args = p.parse_args()

    sht.set_sht_backend(args.backend)
    print(f"SHT backend: {sht.get_sht_backend()}")

    if args.mode == "ladder":
        print("\n########## MODE: ladder ##########")
        run_ladder(args)
        return

    if args.mode == "nside-ladder":
        print("\n########## MODE: nside-ladder ##########")
        run_nside_ladder(args)
        return

    if args.mode == "profile":
        print("\n########## MODE: profile ##########")
        run_profile(args)
        return

    # Freeze var_pix_ref once so the only thing varying across ensembles is the CRN
    # (var_pix_ref is a filter knob; a common value isolates the MC noise we measure).
    # On the MASTER estimator -- the default these contexts are built with -- there
    # is no Wiener filter, so make_cutsky_mc_context returns None and there is
    # nothing to freeze; the ensembles already differ only by their CRN.
    print("Calibrating shared var_pix_ref ...")
    cal_ctx, _, _ = build_contexts(
        0, args.n_sims, nside=args.nside, lmax=args.lmax,
        delta_ell=args.delta_ell, ell_per_bin_below=args.ell_per_bin_below,
    )
    var_pix_ref = cal_ctx.var_pix_ref
    print(
        "  var_pix_ref = none (MASTER: no Wiener filter)"
        if var_pix_ref is None
        else f"  var_pix_ref = {var_pix_ref:.4e}"
    )

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
