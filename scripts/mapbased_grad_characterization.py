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
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
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


def _static_pieces(nside, lmax):
    """Design-INDEPENDENT pieces for the tiny CMB-only config, built once.

    Everything here is fixed across the design optimization: the binning, the
    E/B priors, the true BB transfer denominator, the cleaner, the fiducial
    ``w_inv`` (only used to seed ``var_pix_ref``), the analysis mask, and the
    OptimizationContext. The per-CRN ensemble is built separately by
    :func:`_mc_ctx` so a stochastic descent can re-draw it cheaply (same shapes
    reuse the compiled forward trace; only the sky/noise leaves change)."""
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


def build_contexts(base_seed, n_sims, *, nside, lmax, var_pix_ref=None):
    """Build (mc_ctx, opt_ctx, cleaner) for a CMB-only tiny config at one CRN seed."""
    pieces = _static_pieces(nside, lmax)
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


def _train_base(step, n_sims, resample_every):
    """Rolling train CRN base: a fresh disjoint block every ``resample_every`` steps.

    Block k uses seeds ``[k*(n_sims+1), k*(n_sims+1) + n_sims]``; all stay below
    ``VAL_BASE`` (asserted by the caller) so train never collides with val/test."""
    return (step // resample_every) * (n_sims + 1)


def _test_base(i):
    """Disjoint base for the i-th held-out TEST ensemble (spaced from train + val)."""
    return (i + 2) * SEED_STRIDE


def _make_objectives(pieces, n_total):
    """Return (value_fn, value_and_grad_fn), both ``eqx.filter_jit``-ed over
    ``(logits, mc_ctx)``.

    ``mc_ctx`` is a *traced argument* (not closed over), so swapping in a fresh
    CRN ensemble of the same ``(n_sims, nside, lmax)`` reuses the single compiled
    executable -- the whole point of re-randomizing CRN cheaply. ``opt_ctx`` /
    ``cleaner`` are closed over (fixed across the optimization). The gradient is
    w.r.t. ``logits`` only (the softmax allocation)."""
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

    value_fn = eqx.filter_jit(_loss)
    vg_fn = eqx.filter_jit(eqx.filter_value_and_grad(_loss))
    return value_fn, vg_fn


def _held_out_gains(value_fn, test_ctxs, logits0, logits):
    """Per-test-ensemble sigma(r) gain [%] of ``logits`` vs the uniform ``logits0``."""
    gains = []
    for ctx in test_ctxs:
        s_u = float(value_fn(jnp.asarray(logits0), ctx))
        s_o = float(value_fn(jnp.asarray(logits), ctx))
        gains.append(100.0 * (s_u - s_o) / s_u)
    return np.array(gains)


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

    A fresh sim ensemble (rolling disjoint ``base_seed``) every
    ``--resample-every`` steps, so the optimizer never fits a single
    realization's empirical-ILC-bias noise. The frozen ``var_pix_ref`` keeps the
    Wiener filter fixed, so re-draws reuse the one compiled forward trace.
    Returns (history, logits_final, logits_best_on_val, build_s, per_eval_s);
    ``history`` is (step, sigma_train, sigma_val) at the SAME logits."""
    n_band = len(N_DET)
    max_train_base = _train_base(args.steps - 1, n_sims, args.resample_every)
    assert max_train_base + n_sims < VAL_BASE, (
        f"train seed blocks reach {max_train_base + n_sims} >= VAL_BASE={VAL_BASE}; "
        "reduce --steps / --resample-every span or raise SEED_STRIDE."
    )
    logits = jnp.zeros(n_band)
    opt = optax.adam(args.lr)
    state = opt.init(logits)
    history = []
    best_val, best_logits = np.inf, logits
    train_ctx = None
    build_s = 0.0
    eval_times = []
    for step in range(args.steps):
        if step % args.resample_every == 0:
            t_b = time.time()
            train_ctx = _mc_ctx(
                pieces, _train_base(step, n_sims, args.resample_every), n_sims, var_pix_ref
            )
            build_s += time.time() - t_b
        t_e = time.time()
        s_tr, g = vg_fn(logits, train_ctx)
        s_tr = float(s_tr)
        eval_times.append(time.time() - t_e)
        s_va = float(value_fn(logits, val_ctx))
        history.append((step, s_tr, s_va))
        if s_va < best_val:
            best_val, best_logits = s_va, logits
        updates, state = opt.update(g, state, logits)
        logits = optax.apply_updates(logits, updates)
    per_eval_s = float(np.median(eval_times[1:])) if len(eval_times) > 1 else eval_times[0]
    return history, np.asarray(logits), np.asarray(best_logits), build_s, per_eval_s


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

    pieces = _static_pieces(args.nside, args.lmax)
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

    gains_final = _held_out_gains(value_fn, test_ctxs, logits0, logits_final)
    gains_best = _held_out_gains(value_fn, test_ctxs, logits0, logits_best)
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
        cal_ctx, _, _ = build_contexts(0, n_sims, nside=args.nside, lmax=args.lmax)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["demo", "stability", "beam", "both", "ladder"], default="demo"
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
