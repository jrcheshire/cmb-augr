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
from augr.optimize import DelensCoupling, design_to_channels, make_optimization_context
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
# cmb-augr #50 (fixed): the ridge is now a fraction of each channel's own variance and
# the library default 1e-10 sits on the plateau even at a 10-1000 GHz band set. 1e-18
# is kept so production stays on the value the smoke and 945271 ran with.
RIDGE = 1e-18


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


def _static_pieces(nside, lmax, *, ridge=RIDGE, delens_cfg=None):
    """Design-independent pieces shared by every mc_ctx + the opt_ctx (built once).

    ``delens_cfg`` (``None`` = off) is ``{l_max_qe, n_iter}``: with it, a
    :class:`augr.optimize.DelensCoupling` is built at the *fiducial* design and the
    returned ``opt_ctx`` carries its reference residual as ``delensed_bb``. That is
    the frozen-Jacobian convention -- the model half of the coupling sits at the
    reference while the sims follow each design -- and it is what makes the aperture
    axis see any delensing benefit at all. Without it the forward runs ``A_lens = 1``
    and the trade is tilted against aperture by construction.
    """
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
    cleaner = nilc_cleaner(clean_e=True, ridge=ridge)
    mask = mk.galactic_mask(nside, F_SKY)

    delens = None
    delensed_kw = {}
    if delens_cfg is not None:
        spec = _spec()
        n_det_fid, net_fid, beam_fid = _fiducial_channels(spec)
        delens = DelensCoupling.build(
            lensing_spectra=ls,
            n_det=n_det_fid,
            net=net_fid,
            beam=beam_fid,
            eta=spec.eta_total,
            mission_years=spec.years_fid,
            f_sky=F_SKY,
            l_max_qe=delens_cfg["l_max_qe"],
            n_iter=delens_cfg["n_iter"],
        )
        delensed_kw = dict(delensed_bb=delens.cl_bb_res0, delensed_bb_ells=delens.ls)

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
        **delensed_kw,
    )
    return dict(
        cl_ee=cl_ee,
        cl_bb=cl_bb,
        bm=bm,
        true_b=true_b,
        cleaner=cleaner,
        mask=mask,
        opt_ctx=opt_ctx,
        delens=delens,
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
        # The sims' lensing B has to be separately addressable for a design-dependent
        # residual to rescale it; without the split the forward is stuck at A_lens = 1.
        split_lensing=static.get("delens") is not None,
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
            delens=static["delens"],
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
_N_GRAD_CALLS = [0]  # per-process (each spawn worker holds its own module state)


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


def _set_compile_cache(path):
    """Point jax's persistent compilation cache at ``path`` (before the first compile).

    Compile of the traced forward is ~hours at nside=128, so one warm job pays it for every
    later process/job with the same shapes -- including all pool workers of a production
    run. Historically useless here because the unrolled executable (2.45 GB) exceeded the
    2 GB serialization cap; under the looped FFT mode it should fit. A RESOURCE_EXHAUSTED
    cache-write warning in the log means the executable is STILL over the cap.
    """
    if path:
        os.makedirs(path, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", path)


# --- incremental part files (resume / multi-node) ------------------------------------------
# Every design gradient and scan point is written to its own .npz under --parts-dir the
# moment it finishes (atomic tmp+rename; one file per task, so concurrent writers never
# collide, incl. on Lustre). A task whose part already exists and matches the config is
# loaded instead of recomputed, so a killed or timed-out job resumes from what it has, and
# disjoint --design-range jobs on separate nodes fill one shared parts dir. Stage results
# are assembled FROM DISK, never from in-memory returns, so "computed here" and "computed
# elsewhere" are indistinguishable.


def _part_path(parts_dir, kind, idx):
    return os.path.join(parts_dir, f"{kind}_{idx:04d}.npz")


def _write_part(path, **arrays):
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_part(path):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception:  # truncated/corrupt write from a killed job -> recompute
        return None


def _grad_part_valid(part, z_row, cfg):
    # Finiteness is part of validity, not a downstream concern: a NaN gradient is a
    # dead run, and without this check 96 of them wrote cleanly and only surfaced
    # 7 h later as "Eigenvalues did not converge" out of np.linalg.eigh.
    return (
        part is not None
        and np.all(np.isfinite(part["grad"]))
        and np.isfinite(part["value"])
        and part["z"].shape == np.shape(z_row)
        and np.allclose(part["z"], z_row, rtol=1e-12, atol=0.0)
        and int(part["nside"]) == cfg["nside"]
        and int(part["lmax"]) == cfg["lmax"]
        and int(part["n_sims"]) == cfg["n_sims"]
        and int(part["n_crn"]) == cfg["n_crn"]
        and str(part["fg_model"]) == str(cfg["fg_model"])
        and np.isclose(float(part["budget"]), cfg["budget"])
    )


def _scan_part_valid(part, t, w1, cfg):
    return (
        part is not None
        and np.isclose(float(part["t"]), t)
        and part["w1"].shape == np.shape(w1)
        and np.allclose(part["w1"], w1, rtol=1e-12, atol=0.0)
        and int(part["nside"]) == cfg["nside"]
        and int(part["lmax"]) == cfg["lmax"]
        and int(part["n_sims"]) == cfg["n_sims"]
        and int(part["eval_index"]) == cfg["eval_index"]
        and int(part["n_outer"]) == cfg["n_outer"]
        and np.isclose(float(part["sigma_prior_r"]), cfg["sigma_prior_r"])
        and str(part["fg_model"]) == str(cfg["fg_model"])
        and np.isclose(float(part["budget"]), cfg["budget"])
    )


def _pieces_key(cfg):
    # The delens knobs and the ridge change what _static_pieces builds, so they have to
    # key the cache -- otherwise a worker silently reuses pieces built under other settings.
    dc = cfg.get("delens_cfg")
    return (
        cfg["nside"],
        cfg["lmax"],
        cfg["budget"],
        cfg["backend"],
        cfg["fg_model"],
        cfg["fft_mode"],
        cfg.get("ridge", RIDGE),
        None if dc is None else (dc["l_max_qe"], dc["n_iter"]),
    )


def _worker_pieces(cfg):
    key = _pieces_key(cfg)
    w = _WORKER.get(key)
    if w is None:
        sht.set_sht_backend(cfg["backend"])
        _set_fft_mode(cfg["fft_mode"])
        _set_compile_cache(cfg["compile_cache"])
        spec = _spec()
        static = _static_pieces(
            cfg["nside"],
            cfg["lmax"],
            ridge=cfg.get("ridge", RIDGE),
            delens_cfg=cfg.get("delens_cfg"),
        )
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
        if sky_cache.nside != cfg["nside"] or sky_cache.lmax != cfg["lmax"]:
            raise ValueError(
                f"sky cache sky_{idx} nside/lmax {sky_cache.nside}/{sky_cache.lmax} != "
                f"cfg {cfg['nside']}/{cfg['lmax']}"
            )
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
    """One design's CRN-averaged Gaussian-EIG gradient -> grad_<i>.npz (skipped if cached)."""
    i, z_row, cfg = payload
    path = _part_path(cfg["parts_dir"], "grad", i)
    if _grad_part_valid(_load_part(path), z_row, cfg):
        print(f"    [pid {os.getpid()}] grad design {i}: cached", flush=True)
        return i
    w = _worker_pieces(cfg)
    z_arr = jnp.asarray(z_row)
    vs, gs = [], []
    for j in range(cfg["n_crn"]):
        idx = i * cfg["n_crn"] + j  # matches collect_gradients' crn_seed0=0 scheme
        t0 = time.time()
        v, g = w["vg_fn"](z_arr, _ctx_for(cfg, w, idx))
        g = np.asarray(g, dtype=float)  # blocks until computed, so the timing is honest
        _N_GRAD_CALLS[0] += 1
        note = " (includes jit compile)" if _N_GRAD_CALLS[0] == 1 else ""
        print(
            f"    [pid {os.getpid()}] grad design {i} crn {j}: {time.time() - t0:.0f}s{note}",
            flush=True,
        )
        vs.append(float(v))
        gs.append(g)
    g_stack = np.stack(gs, axis=0)
    if not (np.all(np.isfinite(g_stack)) and np.all(np.isfinite(vs))):
        bad = sorted({int(k) for k in np.where(~np.isfinite(g_stack))[1]})
        raise FloatingPointError(
            f"design {i}: non-finite gradient, no part written. Knobs {bad} "
            f"(value={np.mean(vs):.6g}). A finite value with a NaN gradient is the "
            "signature of a 0 * inf in the backward pass -- run the design alone "
            "under JAX_DEBUG_NANS=1 rather than letting the sweep continue."
        )
    _write_part(
        path,
        z=np.asarray(z_row),
        value=np.mean(vs),
        grad=g_stack.mean(axis=0),
        crn_spread=g_stack.std(axis=0),
        nside=cfg["nside"],
        lmax=cfg["lmax"],
        n_sims=cfg["n_sims"],
        n_crn=cfg["n_crn"],
        fg_model=str(cfg["fg_model"]),
        budget=cfg["budget"],
    )
    return i


def _scan_worker(payload):
    """One HL-EIG scan point along direction 1 -> scan_<k>.npz (skipped if cached)."""
    k, t, w1, cfg = payload
    path = _part_path(cfg["parts_dir"], "scan", k)
    if _scan_part_valid(_load_part(path), t, w1, cfg):
        print(f"    [pid {os.getpid()}] scan point {k}: cached", flush=True)
        return k
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
    _write_part(
        path,
        t=t,
        w1=np.asarray(w1),
        gauss_eig=gauss,
        hl_eig=hl,
        cost=cost,
        nside=cfg["nside"],
        lmax=cfg["lmax"],
        n_sims=cfg["n_sims"],
        eval_index=cfg["eval_index"],
        n_outer=cfg["n_outer"],
        sigma_prior_r=cfg["sigma_prior_r"],
        fg_model=str(cfg["fg_model"]),
        budget=cfg["budget"],
    )
    return k


# --- iso-cost grid mode (poster/paper figure) ----------------------------------------------
# A value-only 2-D grid over (aperture, focal-plane diameter): the substitution question
# "more detectors or more aperture?" from eig_production_scoping.md, evaluated on the same
# map-based MC forward as the subspace run. Detector count enters as the total-focal-plane
# scale (n_det ~ fp_diameter^2 at fixed relative allocation and f-number); aperture enters
# through the existing z knob. Both are TRACED arguments of one jitted loss, so the whole
# grid shares a single compile. The budget penalty is disabled (GRID_BUDGET_MUSD) so the
# surface is pure marginal Gaussian EIG; the dollar cost of every grid point is stored
# alongside, and iso-cost contours are drawn by the plotting side, not baked into the value.
# Every point is evaluated on ONE fixed held-out CRN ensemble (--eval-index, the scan
# convention), so neighboring points differ by design, not by realization.

GRID_BUDGET_MUSD = 1.0e9


def _make_grid_loss(spec, static, cost_model):
    """params = (z_ap, fp_diameter_m) -> Gaussian-EIG loss (budget penalty inert)."""

    def loss(params, mc_ctx):
        z_ap, fp_d = params
        z = jnp.zeros(spec.n_dim).at[spec.n_groups - 1].set(z_ap)
        d = spec.design_pytree(z)
        return physical_design_objective(
            d,
            freqs_per_group=spec.freqs_per_group,
            fp_diameter_m=fp_d,
            mc_ctx=mc_ctx,
            opt_ctx=static["opt_ctx"],
            cleaner=static["cleaner"],
            cost_model=cost_model,
            budget=GRID_BUDGET_MUSD,
            eta_total=spec.eta_total,
            objective="marginal_eig_r",
            delens=static["delens"],
        )

    return loss


def _iso_part_valid(part, a_m, fp_d, cfg):
    dc = cfg.get("delens_cfg")
    return (
        part is not None
        and np.isfinite(part["eig"])
        and np.isclose(float(part["aperture_m"]), a_m)
        and np.isclose(float(part["fp_diameter_m"]), fp_d)
        and int(part["nside"]) == cfg["nside"]
        and int(part["lmax"]) == cfg["lmax"]
        and int(part["n_sims"]) == cfg["n_sims"]
        and int(part["eval_index"]) == cfg["eval_index"]
        and str(part["fg_model"]) == str(cfg["fg_model"])
        and int(part["delens_lmq"]) == (-1 if dc is None else dc["l_max_qe"])
    )


def _iso_worker(payload):
    """One (aperture, fp_diameter) grid point's Gaussian EIG + cost -> iso_<k>.npz."""
    k, a_m, fp_d, cfg = payload
    path = _part_path(cfg["parts_dir"], "iso", k)
    if _iso_part_valid(_load_part(path), a_m, fp_d, cfg):
        print(f"    [pid {os.getpid()}] iso point {k}: cached", flush=True)
        return k
    w = _worker_pieces(cfg)
    if "grid_fn" not in w:  # one jitted loss + one held-out ensemble per worker
        w["grid_fn"] = build_design_objectives(
            _make_grid_loss(w["spec"], w["static"], CostModel())
        )[0]
    if "eval" not in w:
        w["eval"] = _ctx_for(cfg, w, cfg["eval_index"])
    spec = w["spec"]
    z_ap = float(np.log(a_m / spec.aperture_fid))
    t0 = time.time()
    eig = -float(w["grid_fn"]((jnp.asarray(z_ap), jnp.asarray(fp_d)), w["eval"]))
    n_det, _, beam = design_to_channels(
        a_m,
        spec.f_number_fid,
        fp_d,
        jnp.asarray(spec.frac_fid),
        spec.freqs_per_group,
    )
    cost = float(
        design_cost(
            n_det,
            beam,
            spec.years_fid,
            cost_model=CostModel(),
            freqs_ghz=spec.freqs_flat,
        )
    )
    if not np.isfinite(eig):
        raise FloatingPointError(
            f"iso point {k} (aperture={a_m:.3f} m, fp_d={fp_d:.3f} m): non-finite EIG, "
            "no part written."
        )
    note = " (includes jit compile)" if time.time() - t0 > 300 else ""
    print(
        f"    [pid {os.getpid()}] iso point {k} (D={a_m:.2f} m, fp={fp_d:.3f} m): "
        f"{time.time() - t0:.0f}s{note}",
        flush=True,
    )
    dc = cfg.get("delens_cfg")
    _write_part(
        path,
        aperture_m=a_m,
        fp_diameter_m=fp_d,
        eig=eig,
        cost=cost,
        n_det_total=float(jnp.sum(n_det)),
        nside=cfg["nside"],
        lmax=cfg["lmax"],
        n_sims=cfg["n_sims"],
        eval_index=cfg["eval_index"],
        fg_model=str(cfg["fg_model"]),
        delens_lmq=(-1 if dc is None else dc["l_max_qe"]),
    )
    return k


def _run_iso_grid(args, fg_model, t0):
    """Grid-only mode: fan the (aperture x fp_diameter) points out, assemble from disk."""
    a_lo, a_hi, n_a = args.grid_aperture
    d_lo, d_hi, n_d = args.grid_fpd
    apertures = np.geomspace(a_lo, a_hi, int(n_a))
    fp_ds = np.geomspace(d_lo, d_hi, int(n_d))
    cfg = _fanout_cfg(args, fg_model, GRID_BUDGET_MUSD)
    print(
        f"\niso-cost grid: {int(n_a)} apertures [{a_lo:g}, {a_hi:g}] m x "
        f"{int(n_d)} fp diameters [{d_lo:g}, {d_hi:g}] m, value-only, "
        f"held-out ensemble {args.eval_index}"
    )
    payloads = [
        (i * int(n_d) + j, float(a), float(d), cfg)
        for i, a in enumerate(apertures)
        for j, d in enumerate(fp_ds)
    ]
    parallel_map(_iso_worker, payloads, workers=args.workers)
    parts = {
        k: p
        for k, a, d, _ in payloads
        if _iso_part_valid(p := _load_part(_part_path(args.parts_dir, "iso", k)), a, d, cfg)
    }
    if len(parts) < len(payloads):
        missing = sorted({k for k, *_ in payloads} - set(parts))
        print(
            f"  [{time.time() - t0:.0f}s] iso grid partial: {len(parts)}/{len(payloads)} "
            f"parts in {args.parts_dir} (missing e.g. {missing[:8]}); rerun to continue."
        )
        return
    shape = (int(n_a), int(n_d))
    grid = {
        key: np.array([float(parts[k][key]) for k, *_ in payloads]).reshape(shape)
        for key in ("eig", "cost", "n_det_total")
    }
    out = {
        "aperture_m": apertures.tolist(),
        "fp_diameter_m": fp_ds.tolist(),
        "eig": grid["eig"].tolist(),
        "cost_musd": grid["cost"].tolist(),
        "n_det_total": grid["n_det_total"].tolist(),
        "config": vars(args),
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  [{time.time() - t0:.0f}s] wrote {args.out}.json")
    _iso_plot(args.out + ".png", apertures, fp_ds, grid["eig"], grid["cost"])
    print(f"  wrote {args.out}.png")


def _iso_plot(path, apertures, fp_ds, eig, cost):
    """Quick diagnostic contour (the poster figure has its own plotting script)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    cf = ax.contourf(apertures, fp_ds, eig.T, levels=15, cmap="viridis")
    cs = ax.contour(apertures, fp_ds, cost.T, levels=8, colors="w", linewidths=1.0)
    ax.clabel(cs, fmt="$%.0fM")
    fig.colorbar(cf, ax=ax, label="Gaussian EIG (nats)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("aperture (m)")
    ax.set_ylabel("focal-plane diameter (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


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
        compile_cache=args.compile_cache,
        parts_dir=args.parts_dir,
        eval_index=args.eval_index,
        sigma_prior_r=args.sigma_prior_r,
        n_outer=args.n_outer,
        ridge=args.ridge,
        delens_cfg=(
            None
            if args.no_delens
            else {"l_max_qe": args.delens_l_max_qe, "n_iter": args.delens_n_iter}
        ),
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
        default=None,
        help="budget = factor x fiducial cost. Overrides --budget-musd when set. The "
        "old 1e12 default made the cost penalty identically zero, so nothing bounded "
        "the optimizer and 'bigger mirror, longer mission' was free.",
    )
    p.add_argument(
        "--budget-musd",
        type=float,
        default=2000.0,
        help="absolute budget in $M (default 2000 = $2B, the demo budget in "
        "paper_scope_decisions.md). Ignored if --budget-factor is given.",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=RIDGE,
        help=f"ILC ridge regularization (default {RIDGE:g}; cmb-augr #50 -- the library "
        "default 1e-10 is unconverged and 1e-14 is not converged either)",
    )
    p.add_argument(
        "--no-delens",
        action="store_true",
        help="run at A_lens = 1 with no design-dependent delensing. Credits aperture "
        "with none of its delensing benefit, so an aperture trade is tilted against "
        "aperture by construction -- for mechanics tests and A/B checks only.",
    )
    p.add_argument(
        "--delens-l-max-qe",
        type=int,
        default=4000,
        help="max QE multipole for the delensing solve (default 4000). N_0^MV saturation "
        "is beam-dependent (~1500 at 30', ~3000 at 10', ~4000 at 5'), so a lower cap "
        "under-credits delensing for small-beam/large-aperture designs -- an "
        "aperture-directional bias. 4000 is affordable with remat (134.6 MB tape; "
        "18.6 GB/worker, ~27 min/design on Vista gg at production shape). Lower it only "
        "for mechanics tests.",
    )
    p.add_argument("--delens-n-iter", type=int, default=5)
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
        "--compile-cache",
        type=str,
        default=None,
        help="jax persistent compilation-cache dir (shared filesystem, e.g. $SCRATCH). "
        "One job's ~hours compile at nside=128 pays for every later process/job with "
        "the same shapes. A RESOURCE_EXHAUSTED cache-write warning means the "
        "executable still exceeds the 2 GB serialization cap.",
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
    p.add_argument(
        "--parts-dir",
        type=str,
        default=None,
        help="dir for incremental per-task results (grad_<i>.npz / scan_<k>.npz; default "
        "<out>_parts). Finished tasks are skipped on restart, so a killed or timed-out "
        "job resumes where it died, and disjoint --design-range jobs fill one shared dir.",
    )
    p.add_argument(
        "--design-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("I0", "I1"),
        help="compute only designs I0 <= i < I1 this run (splits the gradient stage across "
        "jobs/nodes; indices refer to the full --n-designs LHS, which is seed-fixed). "
        "While any design is missing from --parts-dir the run exits 0 after its range, "
        "before the eigendecomp; the run that completes the set continues to the scan.",
    )
    p.add_argument(
        "--grid-aperture",
        type=float,
        nargs=3,
        default=None,
        metavar=("MIN", "MAX", "N"),
        help="with --grid-fpd, run the iso-cost grid mode instead of the subspace + scan: "
        "N geometrically spaced apertures [m] between MIN and MAX, value-only Gaussian "
        "EIG on the held-out ensemble, budget penalty off, dollar cost stored per point.",
    )
    p.add_argument(
        "--grid-fpd",
        type=float,
        nargs=3,
        default=None,
        metavar=("MIN", "MAX", "N"),
        help="focal-plane-diameter axis [m] of the iso-cost grid (detector-count scale: "
        "n_det ~ fp_diameter^2 at fixed relative allocation and f-number).",
    )
    p.add_argument("--out", type=str, default="/tmp/active_subspace_hl_eig")
    args = p.parse_args()
    if args.parts_dir is None:
        args.parts_dir = args.out + "_parts"
    os.makedirs(args.parts_dir, exist_ok=True)

    fg_model = None if args.fg_model.lower() == "none" else args.fg_model
    sht.set_sht_backend(args.backend)
    _set_fft_mode(args.fft_mode)
    _set_compile_cache(args.compile_cache)
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

    if (args.grid_aperture is None) != (args.grid_fpd is None):
        p.error("--grid-aperture and --grid-fpd must be given together")
    if args.grid_aperture is not None:
        _run_iso_grid(args, fg_model, t0)
        return

    delens_cfg = (
        None
        if args.no_delens
        else {"l_max_qe": args.delens_l_max_qe, "n_iter": args.delens_n_iter}
    )
    static = _static_pieces(
        args.nside, args.lmax, ridge=args.ridge, delens_cfg=delens_cfg
    )
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
    budget = (
        args.budget_factor * cost_fid
        if args.budget_factor is not None
        else args.budget_musd
    )
    print(
        f"D={spec.n_dim}  fiducial cost=${cost_fid:.0f}M  "
        f"budget=${budget:.0f}M  total n_det={float(jnp.sum(n_det_fid)):.0f}"
    )
    if budget > 100.0 * cost_fid:
        print(
            f"  WARNING: budget is {budget / cost_fid:.3g}x the fiducial cost -- the "
            "penalty is effectively off and the optimizer is unbounded upward."
        )
    print(
        "  delensing: OFF (A_lens = 1)"
        if delens_cfg is None
        else f"  delensing: l_max_qe={delens_cfg['l_max_qe']} n_iter={delens_cfg['n_iter']}"
    )
    print(f"  ILC ridge: {args.ridge:g}")

    loss = _make_loss(spec, static, cost_model, budget)
    value_fn, vg_fn = build_design_objectives(loss)

    cfg = _fanout_cfg(args, fg_model, budget)
    if args.workers <= 1:
        # Serial mode runs the same worker fns in-process (parallel_map(workers<=1) never
        # spawns); hand them the already-built pieces so nothing is rebuilt or recompiled.
        _WORKER[_pieces_key(cfg)] = {
            "spec": spec,
            "static": static,
            "value_fn": value_fn,
            "vg_fn": vg_fn,
        }

    # --- 1. build the active subspace from the cheap Gaussian-EIG gradient ---
    print(
        f"\nsampling {args.n_designs} designs (D={spec.n_dim}) + collecting gradients "
        f"(n_crn={args.n_crn}) ..."
    )
    z = sample_designs(args.n_designs, spec.n_dim, sigma=args.sigma, method="lhs", seed=0)
    i0, i1 = args.design_range or (0, args.n_designs)
    parallel_map(
        _grad_worker,
        [(i, np.asarray(z[i]), cfg) for i in range(i0, i1)],
        workers=args.workers,
    )
    parts = {
        i: p
        for i in range(args.n_designs)
        if _grad_part_valid(
            p := _load_part(_part_path(args.parts_dir, "grad", i)), np.asarray(z[i]), cfg
        )
    }
    if len(parts) < args.n_designs:
        missing = sorted(set(range(args.n_designs)) - set(parts))
        print(
            f"  [{time.time() - t0:.0f}s] gradient stage partial: {len(parts)}/"
            f"{args.n_designs} parts in {args.parts_dir} (missing e.g. {missing[:8]}). "
            "Run the remaining --design-range(s), then any rerun continues past the "
            "eigendecomp."
        )
        return
    gs = GradientSample(
        z=z,
        values=np.array([float(parts[i]["value"]) for i in range(args.n_designs)]),
        grads=np.stack([parts[i]["grad"] for i in range(args.n_designs)], axis=0),
        crn_spread=np.stack([parts[i]["crn_spread"] for i in range(args.n_designs)], axis=0),
    )
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
    parallel_map(
        _scan_worker,
        [(k, float(t), np.asarray(w1), cfg) for k, t in enumerate(ts)],
        workers=args.workers,
    )
    sparts = {
        k: p
        for k, t in enumerate(ts)
        if _scan_part_valid(
            p := _load_part(_part_path(args.parts_dir, "scan", k)), float(t), w1, cfg
        )
    }
    if len(sparts) < args.scan_points:
        print(
            f"  [{time.time() - t0:.0f}s] scan stage partial: {len(sparts)}/"
            f"{args.scan_points} parts in {args.parts_dir}; rerun to continue."
        )
        return
    gauss_eig = [float(sparts[k]["gauss_eig"]) for k in range(args.scan_points)]
    hl_eig = [float(sparts[k]["hl_eig"]) for k in range(args.scan_points)]
    cost_scan = [float(sparts[k]["cost"]) for k in range(args.scan_points)]
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
