"""plot_cutsky_hl_demo.py — illustrate the map-based HL-NUTS sigma(r) machinery.

A *methods demonstration* (synthetic data, not a science result): builds a single
cleaned-BB-map forecast at the reionization bump (ell 2-50) with a synthetic
Monte-Carlo ensemble (analytic Knox covariance + a chi^2 draw), then shows the two
things the cut-sky HL bridge (:mod:`augr.likelihood.from_cutsky`) delivers:

* **Left** — the marginal posterior ``P(r)``: the Gaussian/Knox-Fisher Gaussian, the
  Gaussian-likelihood NUTS (parity), and the Hamimeche-Lewis NUTS. HL is wider and
  right-skewed at low mode count — the non-Gaussian widening the Knox Fisher misses.
* **Right** — the KS calibration of the lowest bump bin: the cleaned-bandpower
  empirical CDF vs the Gaussian model and the scaled-chi^2 (HL-implied) model, with
  the :func:`augr.likelihood.bandpower_ks` verdict that *decides* which likelihood to
  headline.

Run:  pixi run python scripts/plot_cutsky_hl_demo.py --out cutsky_hl_demo.png
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from augr.covariance import _build_M, bandpower_covariance_full
from augr.fisher import FisherForecast
from augr.likelihood import (
    bandpower_ks,
    build_cutsky_signal_model,
    constrain,
    draw_fisher_inits,
    marginal_sigma,
    posterior_from_cutsky_mc,
    run_nuts_chains,
)
from augr.likelihood.mle import make_dithered_starts, run_mle_search
from augr.likelihood.ordering import SpectrumLayout, matrices_to_spectra
from augr.signal import flatten_params

R_FID = 0.01


class _SyntheticMC:
    def __init__(self, covariance, mean_bandpower, debiased_bandpowers):
        self.covariance = covariance
        self.mean_bandpower = mean_bandpower
        self.debiased_bandpowers = debiased_bandpowers


def _synthetic_case(n_sims, seed):
    """Cleaned-map SignalModel + synthetic chi^2 MC at the bump (analytic Knox cov)."""
    ells = jnp.arange(2, 260, dtype=float)
    template_cl = 5e-5 * (ells / 80.0) ** (-0.4)
    sm, inst = build_cutsky_signal_model(
        ells, template_cl, f_sky=0.6, ell_min=2, ell_max=50, delta_ell=8
    )
    fid = {"r": R_FID, "A_lens": 1.0, "A_res": 1.0}
    fid_vec = flatten_params(fid, sm.parameter_names)
    layout = SpectrumLayout.from_freq_pairs(sm.freq_pairs, sm.n_bins)
    total = np.asarray(matrices_to_spectra(_build_M(sm, inst, fid_vec), layout))
    cov = np.asarray(bandpower_covariance_full(sm, inst, fid_vec))
    var = np.diag(cov)
    nu = 2.0 * total**2 / var
    rng = np.random.default_rng(seed)
    ens = np.stack(
        [(total[b] / nu[b]) * rng.chisquare(nu[b], size=n_sims) for b in range(sm.n_bins)], axis=1
    )
    return sm, inst, fid, fid_vec, _SyntheticMC(cov, total, ens)


def _r_samples(mc, sm, inst, fid, kind, fisher_cov, key, *, n_chains, warmup, samples):
    """Constrained r posterior samples for one likelihood kind."""
    post, transform, free_names, _lik = posterior_from_cutsky_mc(mc, sm, fid, likelihood_kind=kind)
    x_fid = post.fiducial_full[post.free_idx]
    k_mle, k_init, k_run = jax.random.split(key, 3)
    inits_mle = draw_fisher_inits(x_fid, fisher_cov, transform, k_mle, 6, scale=0.5)
    mle = run_mle_search(post.log_prob, inits_mle)
    dither = jnp.sqrt(jnp.diag(fisher_cov))
    inits = make_dithered_starts(mle.best.x, dither, n_chains, k_init)
    positions, _info = run_nuts_chains(
        post.log_prob,
        inits,
        k_run,
        num_warmup=warmup,
        num_samples=samples,
        target_acceptance_rate=0.99,
    )
    constrained = constrain(positions, transform)
    flat = constrained.reshape(-1, constrained.shape[-1])
    r_idx = list(free_names).index("r")
    return np.asarray(flat[:, r_idx]), marginal_sigma(constrained, free_names, "r")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="cutsky_hl_demo.png")
    p.add_argument("--nsims", type=int, default=400)
    p.add_argument("--n-chains", type=int, default=3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    sm, inst, fid, _fid_vec, mc = _synthetic_case(args.nsims, args.seed)

    # Fisher baseline (sigma + the Gaussian curve) and the init covariance.
    ff = FisherForecast(
        sm, inst, fid, priors={}, fixed_params=[], external_covariance=jnp.asarray(mc.covariance)
    )
    sigma_fisher = float(ff.sigma("r"))
    fisher = np.asarray(ff.compute())
    w, vecs = np.linalg.eigh(0.5 * (fisher + fisher.T))
    fisher_cov = jnp.asarray((vecs / np.maximum(w, 1e-12 * w.max())) @ vecs.T)

    key = jax.random.PRNGKey(args.seed)
    kg, kh = jax.random.split(key)
    r_g, sig_g = _r_samples(
        mc,
        sm,
        inst,
        fid,
        "gaussian",
        fisher_cov,
        kg,
        n_chains=args.n_chains,
        warmup=args.warmup,
        samples=args.samples,
    )
    r_h, sig_h = _r_samples(
        mc,
        sm,
        inst,
        fid,
        "hl",
        fisher_cov,
        kh,
        n_chains=args.n_chains,
        warmup=args.warmup,
        samples=args.samples,
    )
    ks = bandpower_ks(mc.debiased_bandpowers, mc.mean_bandpower, mc.covariance)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # --- Left: P(r) ---
    rgrid = np.linspace(min(r_g.min(), r_h.min()), max(r_g.max(), r_h.max()), 400)
    bins = np.linspace(rgrid[0], rgrid[-1], 60)
    axL.hist(
        r_g,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.30,
        color="C0",
        label=f"Gaussian-NUTS   sigma(r)={sig_g:.2e}",
    )
    axL.hist(
        r_h,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.30,
        color="C3",
        label=f"HL-NUTS         sigma(r)={sig_h:.2e}",
    )
    gauss_pdf = stats.norm.pdf(rgrid, R_FID, sigma_fisher)
    axL.plot(
        rgrid, gauss_pdf, "k--", lw=1.6, label=f"Knox-Fisher Gaussian   sigma(r)={sigma_fisher:.2e}"
    )
    axL.axvline(R_FID, color="0.4", lw=0.8, ls=":")
    axL.set_xlabel("r")
    axL.set_ylabel("posterior density")
    axL.set_title(f"Marginal P(r)  —  HL is {sig_h / sigma_fisher:.2f}x wider than Knox-Fisher")
    axL.legend(fontsize=8, loc="upper right")

    # --- Right: KS calibration of the lowest bump bin ---
    ens0 = np.sort(np.asarray(mc.debiased_bandpowers)[:, 0])
    mean0 = float(np.asarray(mc.mean_bandpower)[0])
    var0 = float(np.asarray(mc.covariance)[0, 0])
    nu0 = float(ks["nu_eff"][0])
    ecdf = np.arange(1, ens0.size + 1) / ens0.size
    xg = np.linspace(ens0.min(), ens0.max(), 400)
    axR.step(
        ens0, ecdf, where="post", color="0.2", lw=1.8, label="cleaned-bandpower ensemble (bin 0)"
    )
    axR.plot(
        xg,
        stats.norm.cdf(xg, mean0, np.sqrt(var0)),
        color="C0",
        lw=1.6,
        label=f"Gaussian model   (KS p={ks['p_gauss'][0]:.2f})",
    )
    axR.plot(
        xg,
        stats.chi2.cdf(xg, nu0, 0.0, mean0 / nu0),
        color="C3",
        lw=1.6,
        label=f"scaled-chi^2 / HL  (nu={nu0:.0f}, KS p={ks['p_chi2'][0]:.2f})",
    )
    axR.set_xlabel(r"$C_b^{BB}$ (bin 0)")
    axR.set_ylabel("CDF")
    axR.set_title(f"KS calibration  —  verdict: recommend = {ks['recommend']!r}")
    axR.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Map-based HL-NUTS sigma(r) [SYNTHETIC methods demo: single cleaned BB map, ell 2-50, "
        f"chi^2 MC, n_sims={args.nsims}]",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    print(f"  Knox-Fisher sigma(r) = {sigma_fisher:.3e}")
    print(f"  Gaussian-NUTS        = {sig_g:.3e}  (x{sig_g / sigma_fisher:.3f})")
    print(f"  HL-NUTS              = {sig_h:.3e}  (x{sig_h / sigma_fisher:.3f})")
    print(f"  KS recommend         = {ks['recommend']!r}")


if __name__ == "__main__":
    main()
