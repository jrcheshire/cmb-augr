"""sample_sigma_r.py — Fisher vs NUTS σ(r): the non-Gaussian bandpower widening.

Builds an Asimov forecast and reports σ(r) three ways:

  * **Fisher** (``FisherForecast.sigma``) — the Gaussian/Knox curvature;
  * **Gaussian-likelihood NUTS** — sampling that same Gaussian (a parity check:
    it must recover the Fisher σ(r) up to Monte-Carlo error);
  * **Hamimeche-Lewis NUTS** — the non-Gaussian bandpower likelihood, which is
    wider at low mode count (the reionization bump).

The HL/Fisher ratio is the ~few-% optimism of the Knox/Gaussian Fisher that
only shows up when the likelihood is *sampled* (HL's Fisher equals Knox by
construction). Requires the ``[sampling]`` extra (blackjax).

Example::

    pixi run python scripts/sample_sigma_r.py --ell-min 2 --ell-max 200 --f-sky 0.6

Pass ``--init-from-mle`` to locate the mode by L-BFGS multistart and start the
chains in a small dither cloud around it (a more robust init than the ~1σ Fisher
draws), reporting the MLE point alongside the sampled σ(r).
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from augr.config import DEFAULT_FIXED, DEFAULT_PRIORS, FIDUCIAL_BK15
from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import Channel, Instrument, ScalarEfficiency
from augr.likelihood import (
    DEFAULT_POSITIVE_PARAMS,
    GaussianLikelihood,
    GaussianPrior,
    HLLikelihood,
    PositivityTransform,
    SignalSpectrumModel,
    constrain,
    converged,
    diagnostics_summary,
    draw_fisher_inits,
    make_dithered_starts,
    make_log_posterior,
    marginal_sigma,
    run_mle_search,
    run_nuts_chains,
)
from augr.signal import SignalModel, flatten_params
from augr.spectra import CMBSpectra


def build_forecast(ell_min: int, ell_max: int, delta_ell: int, f_sky: float, r_fid: float):
    """A small 3-channel (90/150/220) Asimov forecast for the demo."""
    eff = ScalarEfficiency(1.0, 1.0, 1.0, 1.0, 1.0)
    inst = Instrument(
        channels=(
            Channel(90.0, 500, 400.0, 30.0, efficiency=eff),
            Channel(150.0, 1000, 300.0, 20.0, efficiency=eff),
            Channel(220.0, 500, 500.0, 15.0, efficiency=eff),
        ),
        mission_duration_years=5.0,
        f_sky=f_sky,
    )
    fid = {**FIDUCIAL_BK15, "r": r_fid}
    sm = SignalModel(
        inst,
        GaussianForegroundModel(),
        CMBSpectra(),
        ell_min=ell_min,
        ell_max=ell_max,
        delta_ell=delta_ell,
    )
    return inst, sm, fid


def sigma_r_nuts(
    likelihood,
    model,
    signal_model,
    instrument,
    fid,
    fixed,
    key,
    *,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    target_accept: float,
    positive,
    init_from_mle: bool = False,
):
    """Marginal σ(r) + convergence diagnostics from multi-chain NUTS.

    Returns ``(sigma_r, diagnostics, mle_info)``. ``diagnostics`` is the dict from
    :func:`augr.likelihood.diagnostics_summary` (R-hat / ESS / E-BFMI /
    divergences) — the gate for whether the σ(r) is trustworthy enough to quote.
    ``mle_info`` is ``None`` unless ``init_from_mle`` is set, in which case the
    chains are started from a small dither cloud around the L-BFGS multistart MLE
    (a more robust init than the ~1σ Fisher draws, which occasionally freeze a
    chain in the foreground-index banana) and ``mle_info`` reports the located
    mode's r-value and convergence flags.
    """
    free_names = [n for n in signal_model.parameter_names if n not in fixed]
    prior = GaussianPrior.from_priors(free_names, fid, DEFAULT_PRIORS)
    transform = PositivityTransform.from_names(free_names, positive_params=positive)
    post = make_log_posterior(model, likelihood, prior, transform, fiducial=fid, fixed=fixed)

    ff = FisherForecast(signal_model, instrument, fid, priors=DEFAULT_PRIORS, fixed_params=fixed)
    fisher_cov = jnp.asarray(np.linalg.inv(np.asarray(ff.compute())))  # free-param order
    x_fid = post.fiducial_full[post.free_idx]

    init_key, mle_key, run_key = jax.random.split(key, 3)

    mle_info = None
    if init_from_mle:
        # Locate the mode by L-BFGS multistart from modest (0.3σ) Fisher-cov
        # draws — small enough that amplitudes rarely flip sign into the unbounded
        # foreground sqrt-NaN region, so the nan-safe best-of-n returns a clean
        # mode. Then disperse the chains in a small cloud (½σ) around it; dither
        # widths are valid in the unconstrained space under the identity transform.
        mle_inits = draw_fisher_inits(x_fid, fisher_cov, transform, init_key, num_chains, scale=0.3)
        mle = run_mle_search(post.log_prob, mle_inits)
        dither = 0.5 * jnp.sqrt(jnp.diag(fisher_cov))
        u0 = make_dithered_starts(mle.best.x, dither, num_chains, mle_key)
        r_idx = list(post.free_names).index("r")
        mle_info = {
            "r": float(constrain(mle.best.x[None, :], transform)[0, r_idx]),
            "n_iter": int(mle.best.n_iter),
            "converged": bool(mle.best.converged),
            "grad_norm": float(mle.best.grad_norm),
        }
    else:
        u0 = draw_fisher_inits(x_fid, fisher_cov, transform, init_key, num_chains)

    positions, info = run_nuts_chains(
        post.log_prob,
        u0,
        run_key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_acceptance_rate=target_accept,
    )
    constrained = constrain(positions, transform)
    sigma_r = marginal_sigma(constrained, post.free_names, "r")
    diag = diagnostics_summary(positions, info, post.free_names)
    return sigma_r, diag, mle_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ell-min", type=int, default=2)
    p.add_argument("--ell-max", type=int, default=200)
    p.add_argument("--delta-ell", type=int, default=10)
    p.add_argument("--f-sky", type=float, default=0.6)
    p.add_argument("--r-fid", type=float, default=0.0)
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--num-warmup", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=4000)
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument(
        "--bound-amplitudes",
        action="store_true",
        help="softplus-bound the non-negative amplitudes (changes their marginals; "
        "off by default so Gaussian-NUTS matches the unbounded Fisher).",
    )
    p.add_argument(
        "--init-from-mle",
        action="store_true",
        help="locate the mode by L-BFGS multistart and start the NUTS chains in a "
        "small dither cloud around it, instead of ~1σ Fisher draws at the fiducial.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    positive = DEFAULT_POSITIVE_PARAMS if args.bound_amplitudes else frozenset()

    inst, sm, fid = build_forecast(
        args.ell_min, args.ell_max, args.delta_ell, args.f_sky, args.r_fid
    )
    model = SignalSpectrumModel(sm)
    fixed = DEFAULT_FIXED
    fid_vec = flatten_params(fid, sm.parameter_names)

    ff = FisherForecast(sm, inst, fid, priors=DEFAULT_PRIORS, fixed_params=fixed)
    sigma_fisher = ff.sigma("r")

    gauss = GaussianLikelihood.from_forecast(sm, inst, fid_vec)
    hl = HLLikelihood.from_forecast(sm, inst, fid_vec)

    key_g, key_h = jax.random.split(jax.random.PRNGKey(args.seed))
    sigma_g, diag_g, mle_g = sigma_r_nuts(
        gauss,
        model,
        sm,
        inst,
        fid,
        fixed,
        key_g,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        target_accept=args.target_accept,
        positive=positive,
        init_from_mle=args.init_from_mle,
    )
    sigma_h, diag_h, mle_h = sigma_r_nuts(
        hl,
        model,
        sm,
        inst,
        fid,
        fixed,
        key_h,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        target_accept=args.target_accept,
        positive=positive,
        init_from_mle=args.init_from_mle,
    )

    def _diag_line(diag: dict) -> str:
        verdict = "✓ converged" if converged(diag) else "✗ NOT converged — do not quote σ(r)"
        return (
            f"R̂(r)={diag['r_hat']['r']:.3f}  R̂_max={diag['r_hat_max']:.3f}  "
            f"ESS(r)={diag['ess']['r']:.0f}  E-BFMI_min={diag['e_bfmi_min']:.2f}  "
            f"div={diag['n_divergent']}/{diag['n_total']}  →  {verdict}"
        )

    print(
        f"\nAsimov forecast: ell=[{args.ell_min},{args.ell_max}] Δℓ={args.delta_ell} "
        f"f_sky={args.f_sky} r_fid={args.r_fid}  ({sm.n_bins} bins, "
        f"{args.num_chains}×{args.num_samples} samples)"
    )
    print("-" * 64)
    print(f"  σ(r) Fisher (Knox)        : {sigma_fisher:.4e}")
    if args.init_from_mle:
        print("  MLE (L-BFGS multistart; chains then dithered around the mode):")
        for label, mi in (("Gaussian", mle_g), ("Hamimeche-Lewis", mle_h)):
            print(
                f"    {label:>15}: r_MLE={mi['r']:+.4e}  (fid={args.r_fid:+.4e}, "
                f"Δ={abs(mi['r'] - args.r_fid):.1e})  n_iter={mi['n_iter']}  "
                f"converged={mi['converged']}  |grad|={mi['grad_norm']:.1e}"
            )
        print(
            "    (HL |grad| may read NaN — the HL gradient is undefined at the exact "
            "Asimov optimum; see augr.likelihood.mle.)"
        )
    print(
        f"  σ(r) Gaussian-NUTS        : {sigma_g:.4e}   "
        f"({sigma_g / sigma_fisher - 1:+.1%} vs Fisher)"
    )
    print(f"      {_diag_line(diag_g)}")
    print(
        f"  σ(r) Hamimeche-Lewis-NUTS : {sigma_h:.4e}   "
        f"({sigma_h / sigma_fisher - 1:+.1%} vs Fisher)"
    )
    print(f"      {_diag_line(diag_h)}")
    print("-" * 64)
    print(f"  HL widening over Gaussian : {sigma_h / sigma_g - 1:+.1%}")
    print(f"  (Gaussian-NUTS parity     : {sigma_g / sigma_fisher - 1:+.1%})")
    print(
        "  Convergence gate: R̂(r) < 1.01, E-BFMI > 0.3, divergences ≪ 1% → σ(r) is trustworthy.\n"
    )


if __name__ == "__main__":
    main()
