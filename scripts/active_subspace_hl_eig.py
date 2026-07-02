"""active_subspace_hl_eig.py -- the design active subspace, and HL-EIG along its axes.

The intertwining driver for the Bayesian-design capstone. It (1) builds the **design active
subspace** from the cheap, validated Gaussian-EIG / sigma(r) gradient through the cut-sky
masked-Wiener Monte-Carlo forward (Constantine ``C = E_xi[grad grad^T]``, eigendecomposed into
interpretable design directions), then (2) evaluates the expensive **non-Gaussian HL-EIG**
(value-only) *along* the 1--3 leading active directions -- where a dense scan is affordable.

The headline question: do the Gaussian-EIG active directions still capture where the
non-Gaussian HL-EIG varies? Cheap gradient builds the subspace; the costly estimator only has
to be *evaluated* on the reduced axes (sidestepping the HL-EIG-gradient-variance problem and
the curse of dimensionality).

**Physical horn-packing design.** The design vector is the dichroic-feedhorn focal-plane
tradespace (:mod:`augr.design_packing`): per-pixel-group focal-plane **area fractions** (a
softmax simplex over a fixed total cold focal plane, one group gauge-fixed), the **aperture**
(sets all beams + mirror cost), the **f-number** (bounded to a buildable range), and the
**mission years**. Detector counts (focal-plane packing), NETs (photon noise), and beams
(single aperture) are all *derived* (:func:`augr.optimize.design_to_channels`), so the activity
scores read out a real recommendation: given a fixed cold focal plane, which dichroic groups
deserve the area, and whether aperture or integration time dominates sigma(r).

Tiny config (nside=16, ``--fg-model None``) runs on a laptop CPU; pass ``--backend jht`` for a
GPU and ``--fg-model d1s1`` / ``d10`` for the realistic foreground legs. Apples-to-apples
controls: the subspace is *built* on disjoint CRN ensembles (a fresh one per design); the
profile scan is *evaluated* on one fixed held-out ensemble, never a construction ensemble.

Run:  pixi run python scripts/active_subspace_hl_eig.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from augr import masking as mk
from augr import sht
from augr.active_subspace import (
    GradientSample,
    active_subspace,
    activity_scores,
    bootstrap_eiguncertainty,
    collect_gradients,
    sample_designs,
)
from augr.cleaning import nilc_cleaner
from augr.config import cleaned_map_instrument
from augr.cost import CostModel
from augr.delensing import load_lensing_spectra
from augr.design_opt import build_design_objectives
from augr.design_packing import PackingDesignSpec
from augr.eig import (
    HLEIGContext,
    design_cost,
    hl_eig_from_external_cov,
    physical_design_objective,
)
from augr.foregrounds import NullForegroundModel
from augr.optimize import design_to_channels, make_optimization_context
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.parallel import parallel_map
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import (
    load_sky_cache,
    make_cutsky_mc_context,
    mc_cutsky_cov_traced,
)

# Dichroic-feedhorn band layout: 3 singles {20, 35, 615} + 3 dichroic pairs (horn set by the
# low band of each pair). The low-frequency bands are kept single (the 20/35 ratio is too wide
# for one horn+OMT).
FREQS_PER_GROUP = (
    (20.0,),
    (35.0,),
    (80.0, 115.0),
    (160.0, 225.0),
    (315.0, 440.0),
    (615.0,),
)

# Fiducial design. Equal focal-plane area per group is the neutral prior ("split the cold
# focal plane evenly"); the subspace then says how to reallocate. fp_diameter is the FIXED
# total area. f# is bounded; aperture / years are the cost-traded knobs.
N_GROUPS = len(FREQS_PER_GROUP)
FRAC_FID = np.full(N_GROUPS, 1.0 / N_GROUPS)
APERTURE_FID = 1.5  # m
F_NUMBER_FID = 1.8
F_BOUNDS = (1.4, 3.0)
YEARS_FID = 4.0
FP_DIAMETER_M = 0.3  # m (fixed cold focal-plane diameter)
ETA_TOTAL = 0.5
F_SKY = 0.6
SEED_STRIDE = 100_000  # disjoint CRN seed blocks (cf. mapbased_grad_characterization.py)


def _spec() -> PackingDesignSpec:
    return PackingDesignSpec(
        freqs_per_group=FREQS_PER_GROUP,
        frac_fid=FRAC_FID,
        aperture_fid=APERTURE_FID,
        f_number_fid=F_NUMBER_FID,
        years_fid=YEARS_FID,
        fp_diameter_m=FP_DIAMETER_M,
        f_bounds=F_BOUNDS,
        ref_group=0,
        eta_total=ETA_TOTAL,
    )


def _static_pieces(nside, lmax):
    """Design-independent pieces shared by every mc_ctx + the opt_ctx (built once)."""
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
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None),
        bm,
        2,
    )
    cleaner = nilc_cleaner(clean_e=True)
    mask = mk.galactic_mask(nside, F_SKY)
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
    return dict(
        cl_ee=cl_ee,
        cl_bb=cl_bb,
        bm=bm,
        true_b=true_b,
        cleaner=cleaner,
        mask=mask,
        opt_ctx=opt_ctx,
    )


def _fiducial_channels(spec):
    """Per-channel (n_det, net, beam) at the fiducial design (z=0)."""
    d0 = spec.design_pytree(jnp.zeros(spec.n_dim))
    return design_to_channels(
        d0["aperture_m"],
        d0["f_number"],
        spec.fp_diameter_m,
        d0["area_fractions"],
        spec.freqs_per_group,
    )


def _build_mc_ctx(static, spec, *, base_seed, n_sims, nside, lmax, fg_model, sky_cache=None):
    """A cut-sky MC ensemble at a CRN seed block (fiducial design's sky/noise reference).

    With ``sky_cache`` (a :class:`augr.spectrum_stages.SkyCache`), the foreground sky
    ensemble is taken from the cache and PySM is never invoked -- the pysm3-less GPU path.
    """
    n_det_fid, net_fid, beam_fid = _fiducial_channels(spec)
    w_inv_fid = np.asarray(
        w_inv_from_noise_design(n_det_fid, net_fid, spec.eta_total, spec.years_fid, F_SKY)
    )
    hs = nk = vpr = None
    if sky_cache is not None:
        hs, nk, vpr = (
            sky_cache.harmonic_skies,
            sky_cache.noise_keys,
            sky_cache.var_pix_ref,
        )
    return make_cutsky_mc_context(
        cleaner=static["cleaner"],
        freqs_ghz=spec.freqs_flat,
        beam_fwhm_arcmin=tuple(float(b) for b in np.asarray(beam_fid)),
        w_inv=w_inv_fid,
        nside=nside,
        lmax=lmax,
        mask=static["mask"],
        cl_ee=static["cl_ee"],
        cl_bb_prior_unbeamed=static["cl_bb"],
        bin_matrix=static["bm"],
        ell_min=2,
        true_bb_binned=static["true_b"],
        n_sims=n_sims,
        base_seed=base_seed,
        fg_model=fg_model,
        r_in=0.0,
        harmonic_skies=hs,
        noise_keys=nk,
        var_pix_ref=vpr,
    )


def _make_loss(spec, static, cost_model, budget):
    """z-space Gaussian-EIG loss(z, mc_ctx) -> physical_design_objective (jax.grad-able in z)."""

    def loss(z, mc_ctx):
        d = spec.design_pytree(z)
        return physical_design_objective(
            d,
            freqs_per_group=spec.freqs_per_group,
            fp_diameter_m=spec.fp_diameter_m,
            mc_ctx=mc_ctx,
            opt_ctx=static["opt_ctx"],
            cleaner=static["cleaner"],
            cost_model=cost_model,
            budget=budget,
            eta_total=spec.eta_total,
            objective="marginal_eig_r",
        )

    return loss


def _hl_template(lmax):
    """Residual-template ``(ells, C_ell)`` for the A_res HL parameter.

    Toy power-law placeholder. The science headline should pass the real post-cleaning FG
    residual -- the foregrounds projected through the *fiducial-design* ILC weights
    (``cl_residual_fg``; cf. :func:`augr.nilc_forecast.nilc_spectra` /
    ``spectrum_stages._cleaned_b_qu``). Deferred here: the cut-sky FG-only projection + the
    per-ell vs binned reconciliation in ``HLEIGContext`` deserve their own tested pass, and
    the template shape does not affect timing.
    """
    ells = np.arange(2, lmax + 1, dtype=float)
    return ells, (ells / 5.0) ** -2.4


def _build_hl_ctx(lmax, sigma_prior_r):
    """The design-independent HL-EIG context built from the residual template."""
    ells, cl = _hl_template(lmax)
    return HLEIGContext.build(
        template_ells=ells,
        template_cl=cl,
        f_sky=F_SKY,
        r_fid=0.0,
        floated=frozenset({"A_res"}),
        sigma_prior_r=sigma_prior_r,
        n_grid=400,
        n_nuis_grid=41,
        ell_max=lmax,
        delta_ell=8,
        ell_per_bin_below=2,
    )


def _scan_point(zv, spec, static, value_fn, eval_ctx, hl_ctx, n_outer):
    """(Gaussian-EIG, HL-EIG, cost) at one design point ``zv`` -- shared by serial + parallel."""
    gauss = -float(value_fn(zv, eval_ctx))  # EIG = -loss (budget slack)
    d = spec.design_pytree(zv)
    n_det, net, beam = design_to_channels(
        d["aperture_m"],
        d["f_number"],
        spec.fp_diameter_m,
        d["area_fractions"],
        spec.freqs_per_group,
    )
    w_inv = w_inv_from_noise_design(n_det, net, spec.eta_total, d["mission_years"], F_SKY)
    tr = mc_cutsky_cov_traced(w_inv, eval_ctx, static["cleaner"], beam_fwhm=beam)
    hl = float(
        hl_eig_from_external_cov(
            tr.covariance,
            tr.mean_bandpower,
            hl_ctx,
            key=jax.random.PRNGKey(0),
            n_outer=n_outer,
        )
    )
    cost = float(
        design_cost(
            n_det,
            beam,
            d["mission_years"],
            cost_model=CostModel(),
            freqs_ghz=spec.freqs_flat,
        )
    )
    return gauss, hl, cost


# --- parallel design fan-out (augr.parallel process pool; for SKX / many-core CPU) --------
# The gradient collect (over designs) and the HL scan (over scan points) are both
# embarrassingly parallel. With --workers > 1, augr.parallel pins BLAS/ducc to 1 thread per
# worker, so parallelism comes from the pool, not ducc threading -- the right tradeoff for
# many independent design evaluations on a many-core node, and it sidesteps the
# pin_blas-mutates-the-parent-env OMP conflict (the main process only does the cheap
# eigendecomp). Worker functions are module-level (picklable for spawn); ``_WORKER`` caches
# the per-process heavy pieces, built once per worker and reused across its tasks.
_WORKER: dict = {}


def _set_fft_mode(mode):
    """Apply the jht azimuth-FFT mode (process-global; set before the first compile).

    ``looped`` (jaxht>=0.2.0) routes the polar-cap FFTs through one common-length chirp-z
    ``lax.scan``, keeping the compiled graph O(1) in ring-kernel count at a ~1.1x runtime
    tax on the cap rings. Load-bearing regardless of ``--backend``: the masked-Wiener
    stage calls ``jht.wiener`` / ``jht.bandpower`` directly, and under jht's own
    ``unrolled`` default the SHT-heavy forward is uncompilable at nside>=64 (2.45 GB
    executable on H200; ~42 min / 12 GB on Grace CPU).
    """
    import jht

    jht.set_azimuth_fft_mode(mode)


def _worker_pieces(cfg):
    key = (
        cfg["nside"],
        cfg["lmax"],
        cfg["budget"],
        cfg["backend"],
        cfg["fg_model"],
        cfg["fft_mode"],
    )
    w = _WORKER.get(key)
    if w is None:
        sht.set_sht_backend(cfg["backend"])
        _set_fft_mode(cfg["fft_mode"])
        spec = _spec()
        static = _static_pieces(cfg["nside"], cfg["lmax"])
        value_fn, vg_fn = build_design_objectives(
            _make_loss(spec, static, CostModel(), cfg["budget"])
        )
        w = {"spec": spec, "static": static, "value_fn": value_fn, "vg_fn": vg_fn}
        _WORKER[key] = w
    return w


def _ctx_for(cfg, w, idx):
    sky_cache = None
    if cfg["sky_cache_dir"]:
        sky_cache = load_sky_cache(os.path.join(cfg["sky_cache_dir"], f"sky_{idx}.npz"))
    return _build_mc_ctx(
        w["static"],
        w["spec"],
        base_seed=SEED_STRIDE * (idx + 1),
        n_sims=cfg["n_sims"],
        nside=cfg["nside"],
        lmax=cfg["lmax"],
        fg_model=cfg["fg_model"],
        sky_cache=sky_cache,
    )


def _grad_worker(payload):
    """CRN-averaged Gaussian-EIG (i, value, grad, spread) for one design -- for parallel_map."""
    i, z_row, cfg = payload
    w = _worker_pieces(cfg)
    z_row = jnp.asarray(z_row)
    vs, gs = [], []
    for j in range(cfg["n_crn"]):
        idx = i * cfg["n_crn"] + j  # matches collect_gradients' crn_seed0=0 scheme
        t0 = time.time()
        v, g = w["vg_fn"](z_row, _ctx_for(cfg, w, idx))
        g = np.asarray(g, dtype=float)  # blocks until computed, so the timing is honest
        print(
            f"    [pid {os.getpid()}] grad design {i} crn {j}: {time.time() - t0:.0f}s"
            " (a worker's first call includes jit compile)",
            flush=True,
        )
        vs.append(float(v))
        gs.append(g)
    g_stack = np.stack(gs, axis=0)
    return i, float(np.mean(vs)), g_stack.mean(axis=0), g_stack.std(axis=0)


def _scan_worker(payload):
    """(k, Gaussian-EIG, HL-EIG, cost) at one scan point -- for parallel_map."""
    k, t, w1, cfg = payload
    w = _worker_pieces(cfg)
    if "eval" not in w:  # build the held-out ensemble + HL context once per worker
        w["eval"] = _ctx_for(cfg, w, cfg["eval_index"])
        w["hl_ctx"] = _build_hl_ctx(cfg["lmax"], cfg["sigma_prior_r"])
    zv = jnp.asarray(np.asarray(t) * np.asarray(w1))
    t0 = time.time()
    gauss, hl, cost = _scan_point(
        zv,
        w["spec"],
        w["static"],
        w["value_fn"],
        w["eval"],
        w["hl_ctx"],
        cfg["n_outer"],
    )
    print(
        f"    [pid {os.getpid()}] scan point {k}: {time.time() - t0:.0f}s",
        flush=True,
    )
    return k, gauss, hl, cost


def _fanout_cfg(args, fg_model, budget):
    """Picklable config dict carried to the pool workers."""
    return dict(
        nside=args.nside,
        lmax=args.lmax,
        n_sims=args.n_sims,
        fg_model=fg_model,
        budget=budget,
        n_crn=args.n_crn,
        sky_cache_dir=args.sky_cache_dir,
        backend=args.backend,
        fft_mode=args.fft_mode,
        eval_index=args.eval_index,
        sigma_prior_r=args.sigma_prior_r,
        n_outer=args.n_outer,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-sims", type=int, default=8, help="MC sims per ensemble (>= bins + 2)")
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--lmax", type=int, default=24)
    p.add_argument(
        "--fg-model",
        type=str,
        default="none",
        help="PySM foreground model (e.g. d1s1, d10) or 'none' for CMB+noise only",
    )
    p.add_argument(
        "--n-designs",
        type=int,
        default=24,
        help="M design samples for C (use M > D for full rank)",
    )
    p.add_argument("--sigma", type=float, default=0.12, help="design sampling radius (dex)")
    p.add_argument("--n-crn", type=int, default=1, help="CRN redraws averaged per design")
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--scan-points", type=int, default=9)
    p.add_argument(
        "--scan-half-width",
        type=float,
        default=0.2,
        help="scan +/- this in dex along dir-1",
    )
    p.add_argument("--n-outer", type=int, default=512, help="HL-EIG outer-MC draws")
    p.add_argument("--sigma-prior-r", type=float, default=0.05)
    p.add_argument(
        "--budget-factor",
        type=float,
        default=1e12,
        help="budget = factor x fiducial cost",
    )
    p.add_argument("--backend", choices=["ducc", "jht"], default="ducc")
    p.add_argument(
        "--fft-mode",
        choices=["looped", "unrolled"],
        default="looped",
        help="jht azimuth-FFT strategy (jaxht>=0.2.0). looped keeps compile O(1) in "
        "ring kernels -- required at nside>=64; unrolled (jht's own default) is "
        "marginally faster per transform but compile scales as nside x #SHTs.",
    )
    p.add_argument(
        "--sky-cache-dir",
        type=str,
        default=None,
        help="dir of precomputed FG sky caches (sky_<idx>.npz from scripts/build_sky_cache.py); "
        "skips PySM -- the pysm3-less GPU path. Must match --nside/--lmax/--n-sims/--fg-model.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process-pool workers for the design-gradient collect + HL scan (SKX many-core "
        "CPU). >1 pins ducc to 1 thread/worker (parallelism from the pool, not threading).",
    )
    p.add_argument(
        "--eval-index",
        type=int,
        default=9999,
        help="held-out scan ensemble index (cf. build_sky_cache)",
    )
    p.add_argument("--out", type=str, default="/tmp/active_subspace_hl_eig")
    args = p.parse_args()

    fg_model = None if args.fg_model.lower() == "none" else args.fg_model
    sht.set_sht_backend(args.backend)
    _set_fft_mode(args.fft_mode)
    # Print the JAX device up front: with --backend jht, a CPU device here means JAX fell
    # back off the GPU (jax[cuda12] failed to init), and the run will crawl -- jht on CPU is
    # ~100x slower than on the GPU (and far slower than ducc, which the gpu env omits).
    print(
        f"SHT backend: {sht.get_sht_backend()}   fft_mode: {args.fft_mode}   fg_model: {fg_model}"
    )
    print(f"JAX backend: {jax.default_backend()}   devices: {jax.devices()}")
    if args.backend == "jht" and jax.default_backend() == "cpu":
        print("  WARNING: --backend jht but JAX is on CPU -- the GPU was not initialized.")
    t0 = time.time()

    static = _static_pieces(args.nside, args.lmax)
    cost_model = CostModel()
    spec = _spec()
    n_det_fid, _, beam_fid = _fiducial_channels(spec)
    cost_fid = float(
        design_cost(
            n_det_fid,
            beam_fid,
            spec.years_fid,
            cost_model=cost_model,
            freqs_ghz=spec.freqs_flat,
        )
    )
    budget = args.budget_factor * cost_fid
    print(
        f"D={spec.n_dim}  fiducial cost=${cost_fid:.0f}M  total n_det={float(jnp.sum(n_det_fid)):.0f}"
    )

    loss = _make_loss(spec, static, cost_model, budget)
    value_fn, vg_fn = build_design_objectives(loss)

    def make_ctx(idx):
        cache = None
        if args.sky_cache_dir is not None:
            cache = load_sky_cache(os.path.join(args.sky_cache_dir, f"sky_{idx}.npz"))
            if cache.nside != args.nside or cache.lmax != args.lmax:
                raise ValueError(
                    f"sky cache sky_{idx} nside/lmax {cache.nside}/{cache.lmax} != "
                    f"args {args.nside}/{args.lmax}"
                )
        return _build_mc_ctx(
            static,
            spec,
            base_seed=SEED_STRIDE * (idx + 1),
            n_sims=args.n_sims,
            nside=args.nside,
            lmax=args.lmax,
            fg_model=fg_model,
            sky_cache=cache,
        )

    # --- 1. build the active subspace from the cheap Gaussian-EIG gradient ---
    print(
        f"\nsampling {args.n_designs} designs (D={spec.n_dim}) + collecting gradients "
        f"(n_crn={args.n_crn}) ..."
    )
    z = sample_designs(args.n_designs, spec.n_dim, sigma=args.sigma, method="lhs", seed=0)
    if args.workers > 1:
        cfg = _fanout_cfg(args, fg_model, budget)
        res = parallel_map(
            _grad_worker,
            [(i, np.asarray(z[i]), cfg) for i in range(args.n_designs)],
            workers=args.workers,
        )
        res.sort(key=lambda r: r[0])
        gs = GradientSample(
            z=z,
            values=np.array([r[1] for r in res]),
            grads=np.stack([r[2] for r in res], axis=0),
            crn_spread=np.stack([r[3] for r in res], axis=0),
        )
    else:
        gs = collect_gradients(vg_fn, z, make_ctx, n_crn=args.n_crn)
    sub = active_subspace(gs.grads)
    boot = bootstrap_eiguncertainty(gs.grads, n_boot=300, n_active=args.n_active)
    print(f"  [{time.time() - t0:.0f}s] energy spectrum: {np.round(sub.energy, 3)}")
    print(f"  cumulative energy: {np.round(sub.cumulative_energy, 3)}")
    print(
        f"  n_active(0.95) = {sub.n_active(0.95)}   subspace-distance p84 = "
        f"{boot['subspace_distance_p84']:.3f}"
    )
    scores = activity_scores(sub, n_active=args.n_active)
    order = np.argsort(scores)[::-1]
    print(f"  top knobs on the leading {args.n_active} directions:")
    for k in order:
        print(f"    {spec.knob_labels[k]:>16}: {scores[k]:.3f}")

    # --- 2. evaluate HL-EIG (value-only) along the leading active direction ---
    w1 = sub.eigenvectors[:, 0]
    ts = np.linspace(-args.scan_half_width, args.scan_half_width, args.scan_points)
    print(
        f"\nscanning {args.scan_points} points along active direction 1 "
        f"(Gaussian-EIG vs HL-EIG) ..."
    )
    if args.workers > 1:
        cfg = _fanout_cfg(args, fg_model, budget)
        res = parallel_map(
            _scan_worker,
            [(k, float(t), np.asarray(w1), cfg) for k, t in enumerate(ts)],
            workers=args.workers,
        )
        res.sort(key=lambda r: r[0])
        gauss_eig = [r[1] for r in res]
        hl_eig = [r[2] for r in res]
        cost_scan = [r[3] for r in res]
    else:
        hl_ctx = _build_hl_ctx(args.lmax, args.sigma_prior_r)
        eval_ctx = make_ctx(args.eval_index)  # fixed held-out ensemble (never a construction one)
        gauss_eig, hl_eig, cost_scan = [], [], []
        for t in ts:
            g, h, c = _scan_point(
                jnp.asarray(t * w1),
                spec,
                static,
                value_fn,
                eval_ctx,
                hl_ctx,
                args.n_outer,
            )
            gauss_eig.append(g)
            hl_eig.append(h)
            cost_scan.append(c)
    print(f"  [{time.time() - t0:.0f}s] done.")

    out = {
        "knob_labels": list(spec.knob_labels),
        "eigenvalues": sub.eigenvalues.tolist(),
        "energy": sub.energy.tolist(),
        "cumulative_energy": sub.cumulative_energy.tolist(),
        "n_active_0.95": sub.n_active(0.95),
        "activity_scores": scores.tolist(),
        "direction_1": w1.tolist(),
        "bootstrap": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in boot.items()},
        "scan_t": ts.tolist(),
        "scan_gauss_eig": gauss_eig,
        "scan_hl_eig": hl_eig,
        "scan_cost": cost_scan,
        "config": vars(args),
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {args.out}.json")
    _plot(args.out + ".png", ts, gauss_eig, hl_eig, cost_scan, budget)
    print(f"  wrote {args.out}.png")


def _plot(path, ts, gauss_eig, hl_eig, cost_scan, budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(gauss_eig) - gauss_eig[len(ts) // 2]
    h = np.asarray(hl_eig) - hl_eig[len(ts) // 2]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ts, g, "C3-o", lw=2, label="Gaussian EIG (Δ)")
    ax.plot(ts, h, "C0-s", lw=2, label="HL EIG (Δ)")
    ax.set(
        xlabel="displacement along active direction 1 (dex)",
        ylabel="ΔEIG from fiducial (nats)",
        title="HL-EIG vs Gaussian-EIG along the leading design direction",
    )
    ax.legend(loc="best", fontsize=9)
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
