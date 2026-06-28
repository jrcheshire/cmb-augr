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
import time

import jax
import jax.numpy as jnp
import numpy as np

from augr import masking as mk
from augr import sht
from augr.active_subspace import (
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
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context, mc_cutsky_cov_traced

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
SEED_STRIDE = (
    100_000  # disjoint CRN seed blocks (cf. mapbased_grad_characterization.py)
)


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


def _build_mc_ctx(static, spec, *, base_seed, n_sims, nside, lmax, fg_model):
    """A cut-sky MC ensemble at a CRN seed block (fiducial design's sky/noise reference)."""
    n_det_fid, net_fid, beam_fid = _fiducial_channels(spec)
    w_inv_fid = np.asarray(
        w_inv_from_noise_design(
            n_det_fid, net_fid, spec.eta_total, spec.years_fid, F_SKY
        )
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


def _hl_template(args):
    """Residual-template ``(ells, C_ell)`` for the A_res HL parameter.

    Toy power-law placeholder. The science headline should pass the real post-cleaning FG
    residual -- the foregrounds projected through the *fiducial-design* ILC weights
    (``cl_residual_fg``; cf. :func:`augr.nilc_forecast.nilc_spectra` /
    ``spectrum_stages._cleaned_b_qu``). Deferred here: the cut-sky FG-only projection + the
    per-ell vs binned reconciliation in ``HLEIGContext`` deserve their own tested pass, and
    the template shape does not affect Phase 0 timing.
    """
    ells = np.arange(2, args.lmax + 1, dtype=float)
    return ells, (ells / 5.0) ** -2.4


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--n-sims", type=int, default=8, help="MC sims per ensemble (>= bins + 2)"
    )
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
    p.add_argument(
        "--sigma", type=float, default=0.12, help="design sampling radius (dex)"
    )
    p.add_argument(
        "--n-crn", type=int, default=1, help="CRN redraws averaged per design"
    )
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
    p.add_argument("--out", type=str, default="/tmp/active_subspace_hl_eig")
    args = p.parse_args()

    fg_model = None if args.fg_model.lower() == "none" else args.fg_model
    sht.set_sht_backend(args.backend)
    print(f"SHT backend: {sht.get_sht_backend()}   fg_model: {fg_model}")
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
        return _build_mc_ctx(
            static,
            spec,
            base_seed=SEED_STRIDE * (idx + 1),
            n_sims=args.n_sims,
            nside=args.nside,
            lmax=args.lmax,
            fg_model=fg_model,
        )

    # --- 1. build the active subspace from the cheap Gaussian-EIG gradient ---
    print(
        f"\nsampling {args.n_designs} designs (D={spec.n_dim}) + collecting gradients "
        f"(n_crn={args.n_crn}) ..."
    )
    z = sample_designs(
        args.n_designs, spec.n_dim, sigma=args.sigma, method="lhs", seed=0
    )
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
    tmpl_ells, tmpl_cl = _hl_template(args)
    hl_ctx = HLEIGContext.build(
        template_ells=tmpl_ells,
        template_cl=tmpl_cl,
        f_sky=F_SKY,
        r_fid=0.0,
        floated=frozenset({"A_res"}),
        sigma_prior_r=args.sigma_prior_r,
        n_grid=400,
        n_nuis_grid=41,
        ell_max=args.lmax,
        delta_ell=8,
        ell_per_bin_below=2,
    )
    eval_ctx = make_ctx(9999)  # fixed held-out ensemble (never a construction ensemble)
    w1 = sub.eigenvectors[:, 0]
    ts = np.linspace(-args.scan_half_width, args.scan_half_width, args.scan_points)
    key = jax.random.PRNGKey(0)

    def hl_eig_at(zvec):
        d = spec.design_pytree(zvec)
        n_det, net, beam = design_to_channels(
            d["aperture_m"],
            d["f_number"],
            spec.fp_diameter_m,
            d["area_fractions"],
            spec.freqs_per_group,
        )
        w_inv = w_inv_from_noise_design(
            n_det, net, spec.eta_total, d["mission_years"], F_SKY
        )
        tr = mc_cutsky_cov_traced(w_inv, eval_ctx, static["cleaner"], beam_fwhm=beam)
        return float(
            hl_eig_from_external_cov(
                tr.covariance, tr.mean_bandpower, hl_ctx, key=key, n_outer=args.n_outer
            )
        )

    print(
        f"\nscanning {args.scan_points} points along active direction 1 "
        f"(Gaussian-EIG vs HL-EIG) ..."
    )
    gauss_eig, hl_eig, cost_scan = [], [], []
    for t in ts:
        zv = jnp.asarray(t * w1)
        gauss_eig.append(-float(value_fn(zv, eval_ctx)))  # EIG = -loss (budget slack)
        hl_eig.append(hl_eig_at(zv))
        d = spec.design_pytree(zv)
        n_det, _, beam = design_to_channels(
            d["aperture_m"],
            d["f_number"],
            spec.fp_diameter_m,
            d["area_fractions"],
            spec.freqs_per_group,
        )
        cost_scan.append(
            float(
                design_cost(
                    n_det,
                    beam,
                    d["mission_years"],
                    cost_model=cost_model,
                    freqs_ghz=spec.freqs_flat,
                )
            )
        )
    print(f"  [{time.time() - t0:.0f}s] done.")

    out = {
        "knob_labels": list(spec.knob_labels),
        "eigenvalues": sub.eigenvalues.tolist(),
        "energy": sub.energy.tolist(),
        "cumulative_energy": sub.cumulative_energy.tolist(),
        "n_active_0.95": sub.n_active(0.95),
        "activity_scores": scores.tolist(),
        "direction_1": w1.tolist(),
        "bootstrap": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in boot.items()
        },
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
