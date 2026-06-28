"""active_subspace_hl_eig.py -- the design active subspace, and HL-EIG along its axes.

The intertwining driver for the Bayesian-design capstone. It (1) builds the **design active
subspace** from the cheap, validated Gaussian-EIG / sigma(r) gradient through the cut-sky
masked-Wiener Monte-Carlo forward (Constantine ``C = E_xi[grad grad^T]``, eigendecomposed into
interpretable design directions), then (2) evaluates the expensive **non-Gaussian HL-EIG**
(value-only) *along* the 1--3 leading active directions -- where a dense scan is affordable.

The headline question: do the Gaussian-EIG active directions still capture where the
non-Gaussian HL-EIG varies? Cheap gradient builds the subspace; the costly estimator only has
to be *evaluated* on the reduced axes (sidestepping the HL-EIG-gradient-variance problem and
the curse of dimensionality at D ~ 13 knobs).

Tiny CMB-only config (nside=16, no PySM) so it runs on a laptop CPU; pass ``--backend jht``
for a GPU. Apples-to-apples controls: the subspace is *built* on disjoint CRN ensembles (a
fresh one per design); the profile scan is *evaluated* on one fixed held-out ensemble, never a
construction ensemble. Same opt_ctx / cost model / budget / fiducial / freqs throughout.

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
    DesignSpec,
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
from augr.eig import HLEIGContext, design_cost, design_objective, hl_eig_from_external_cov
from augr.foregrounds import NullForegroundModel
from augr.optimize import make_optimization_context
from augr.optimize_mapbased import w_inv_from_noise_design
from augr.signal import SignalModel
from augr.spectra import CMBSpectra
from augr.spectrum_stages import make_cutsky_mc_context, mc_cutsky_cov_traced

# Fiducial 3-band design (eta held fixed -- cost-free, so it runs away under any objective).
FREQS = (90.0, 150.0, 220.0)
BEAMS = (40.0, 30.0, 20.0)
N_DET = (200.0, 400.0, 200.0)
NET = (60.0, 50.0, 80.0)
ETA = (0.5, 0.5, 0.5)
MISSION_YEARS = 4.0
F_SKY = 0.6
SEED_STRIDE = 100_000  # disjoint CRN seed blocks (cf. mapbased_grad_characterization.py)

# The design vector: per-band {n_det, net, beam_fwhm, beam_p} + mission_years (13 knobs).
KNOB_LABELS = (
    *(f"n_det@{f:.0f}" for f in FREQS),
    *(f"NET@{f:.0f}" for f in FREQS),
    *(f"beam_fwhm@{f:.0f}" for f in FREQS),
    *(f"beam_p@{f:.0f}" for f in FREQS),
    "mission_years",
)


def _fiducial_design() -> dict:
    return {
        "n_det": jnp.asarray(N_DET),
        "net": jnp.asarray(NET),
        "beam_fwhm": jnp.asarray(BEAMS),
        "beam_p": jnp.ones(len(BEAMS)),
        "mission_years": jnp.asarray(float(MISSION_YEARS)),
    }


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
        jnp.clip(CMBSpectra().cl_bb(jnp.arange(lmax + 1, dtype=float), 0.0), 0.0, None), bm, 2
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
        cl_ee=cl_ee, cl_bb=cl_bb, bm=bm, true_b=true_b, cleaner=cleaner, mask=mask, opt_ctx=opt_ctx
    )


def _build_mc_ctx(static, *, base_seed, n_sims, nside, lmax):
    """A cut-sky MC ensemble at a given CRN seed block (the fiducial design's sky/noise)."""
    w_inv_fid = np.asarray(
        w_inv_from_noise_design(
            jnp.asarray(N_DET), jnp.asarray(NET), jnp.asarray(ETA), MISSION_YEARS, F_SKY
        )
    )
    return make_cutsky_mc_context(
        cleaner=static["cleaner"],
        freqs_ghz=FREQS,
        beam_fwhm_arcmin=BEAMS,
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
        fg_model=None,
        r_in=0.0,
    )


def _make_loss(spec, static, cost_model, budget):
    """z-space Gaussian-EIG loss(z, mc_ctx) -> the design_objective (jax.grad-able in z)."""

    def loss(z, mc_ctx):
        d = spec.design_pytree(z)
        return design_objective(
            d["n_det"],
            d["net"],
            jnp.asarray(ETA),
            d["mission_years"],
            d["beam_fwhm"],
            d["beam_p"],
            mc_ctx=mc_ctx,
            opt_ctx=static["opt_ctx"],
            cleaner=static["cleaner"],
            cost_model=cost_model,
            budget=budget,
            freqs_ghz=FREQS,
            objective="marginal_eig_r",
        )

    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-sims", type=int, default=8, help="MC sims per ensemble (>= bins + 2)")
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--lmax", type=int, default=24)
    p.add_argument(
        "--n-designs", type=int, default=24, help="M design samples for C (use M > D for full rank)"
    )
    p.add_argument("--sigma", type=float, default=0.12, help="design sampling radius (dex)")
    p.add_argument("--n-crn", type=int, default=1, help="CRN redraws averaged per design")
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--scan-points", type=int, default=9)
    p.add_argument(
        "--scan-half-width", type=float, default=0.2, help="scan +/- this in dex along dir-1"
    )
    p.add_argument("--n-outer", type=int, default=512, help="HL-EIG outer-MC draws")
    p.add_argument("--sigma-prior-r", type=float, default=0.05)
    p.add_argument(
        "--budget-factor", type=float, default=1e12, help="budget = factor x fiducial cost"
    )
    p.add_argument("--backend", choices=["ducc", "jht"], default="ducc")
    p.add_argument("--out", type=str, default="/tmp/active_subspace_hl_eig")
    args = p.parse_args()

    sht.set_sht_backend(args.backend)
    print(f"SHT backend: {sht.get_sht_backend()}")
    t0 = time.time()

    static = _static_pieces(args.nside, args.lmax)
    cost_model = CostModel()
    spec = DesignSpec.from_pytree(_fiducial_design(), KNOB_LABELS, mode="log")
    cost_fid = float(
        design_cost(
            jnp.asarray(N_DET),
            jnp.asarray(BEAMS),
            MISSION_YEARS,
            cost_model=cost_model,
            freqs_ghz=FREQS,
        )
    )
    budget = args.budget_factor * cost_fid

    loss = _make_loss(spec, static, cost_model, budget)
    value_fn, vg_fn = build_design_objectives(loss)

    def make_ctx(idx):
        return _build_mc_ctx(
            static,
            base_seed=SEED_STRIDE * (idx + 1),
            n_sims=args.n_sims,
            nside=args.nside,
            lmax=args.lmax,
        )

    # --- 1. build the active subspace from the cheap Gaussian-EIG gradient ---
    print(
        f"\nsampling {args.n_designs} designs (D={spec.n_dim}) + collecting gradients "
        f"(n_crn={args.n_crn}) ..."
    )
    z = sample_designs(args.n_designs, spec.n_dim, sigma=args.sigma, method="lhs", seed=0)
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
    for k in order[:6]:
        print(f"    {KNOB_LABELS[k]:>16}: {scores[k]:.3f}")

    # --- 2. evaluate HL-EIG (value-only) along the leading active direction ---
    hl_ctx = HLEIGContext.build(
        template_ells=np.arange(2, args.lmax + 1, dtype=float),
        template_cl=(np.arange(2, args.lmax + 1, dtype=float) / 5.0) ** -2.4,
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
        w_inv = w_inv_from_noise_design(
            d["n_det"], d["net"], jnp.asarray(ETA), d["mission_years"], F_SKY
        )
        tr = mc_cutsky_cov_traced(
            w_inv, eval_ctx, static["cleaner"], beam_fwhm=d["beam_fwhm"], beam_p=d["beam_p"]
        )
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
        zv = t * w1
        gauss_eig.append(-float(value_fn(jnp.asarray(zv), eval_ctx)))  # EIG = -loss (budget slack)
        hl_eig.append(hl_eig_at(jnp.asarray(zv)))
        d = spec.design_pytree(jnp.asarray(zv))
        cost_scan.append(
            float(
                design_cost(
                    d["n_det"],
                    d["beam_fwhm"],
                    d["mission_years"],
                    cost_model=cost_model,
                    freqs_ghz=FREQS,
                )
            )
        )
    print(f"  [{time.time() - t0:.0f}s] done.")

    out = {
        "knob_labels": list(KNOB_LABELS),
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
