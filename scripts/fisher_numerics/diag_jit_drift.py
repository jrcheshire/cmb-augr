"""Diagnose JIT vs eager drift in sigma_r_from_channels at PICO.

Instrument the pipeline to print per-stage statistics under both
modes; find which stage's output drifts.

Pipeline stages:
  1. noise_nl_continuous per channel       -> noise_nls
  2. bandpower_covariance_blocks_from_noise -> cov_blocks
  3. fisher._fisher_from_blocks (solve)    -> F
  4. + diag(prior_diag) and inv -> F_inv
  5. sqrt(F_inv[r,r])                      -> sigma_r

Run each stage in both jit and eager wrappers, print:
  - shape sanity
  - mean/max/min of each stage
  - relative diff jit-vs-eager at that stage
"""
import jax
import jax.numpy as jnp
import numpy as np

from augr.config import (
    DEFAULT_FIXED_MOMENT,
    DEFAULT_PRIORS_MOMENT,
    FIDUCIAL_MOMENT,
    pico_like,
)
from augr.covariance import bandpower_covariance_blocks_from_noise
from augr.fisher import _fisher_from_blocks
from augr.foregrounds import MomentExpansionModel
from augr.instrument import noise_nl_continuous
from augr.optimize import make_optimization_context, sigma_r_from_channels
from augr.spectra import CMBSpectra


def stats(label, x):
    """Compact summary of a JAX array."""
    a = np.asarray(x)
    if a.ndim == 0:
        return f"{label}: {float(a):.10e}"
    return (f"{label}: shape={a.shape}, mean={a.mean():.6e}, "
            f"max={np.abs(a).max():.6e}, min_abs={np.abs(a).min():.6e}")


def reldiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.maximum(np.abs(a), 1e-300)
    return float(np.max(np.abs(a - b) / denom))


def build_pipeline(opt_ctx, inst):
    """Returns six functions that compute each stage; same closure-vars."""
    ells = opt_ctx.ells
    n_chan = opt_ctx.n_det.shape[0]
    knee_arr = jnp.zeros(n_chan)
    alpha_arr = jnp.ones(n_chan)
    f_sky = inst.f_sky
    mission_years = inst.mission_duration_years

    def noise_stage(n_det, net, beam, eta):
        return jnp.stack([
            noise_nl_continuous(
                net[i], n_det[i], beam[i], eta[i],
                ells, mission_years, f_sky, knee_arr[i], alpha_arr[i],
            )
            for i in range(n_chan)
        ])

    def cov_stage(noise_nls):
        return bandpower_covariance_blocks_from_noise(
            opt_ctx.signal_model, noise_nls, f_sky, opt_ctx.fiducial_params)

    def fisher_stage(cov_blocks):
        return _fisher_from_blocks(opt_ctx.J_blocks, cov_blocks)

    def sigma_stage(F):
        F_total = F + jnp.diag(opt_ctx.prior_diag)
        F_inv = jnp.linalg.inv(F_total)
        return jnp.sqrt(F_inv[opt_ctx.r_idx, opt_ctx.r_idx])

    return noise_stage, cov_stage, fisher_stage, sigma_stage


def run_stagewise(opt_ctx, inst, mode):
    """Run each stage with input from the previous stage; print stats."""
    n_det, net, beam, eta = (
        opt_ctx.n_det, opt_ctx.net, opt_ctx.beam, opt_ctx.eta)
    noise_stage, cov_stage, fisher_stage, sigma_stage = build_pipeline(
        opt_ctx, inst)

    if mode == "jit":
        noise_stage = jax.jit(noise_stage)
        cov_stage = jax.jit(cov_stage)
        fisher_stage = jax.jit(fisher_stage)
        sigma_stage = jax.jit(sigma_stage)

    print(f"\n=== {mode.upper()} ===")
    noise = noise_stage(n_det, net, beam, eta)
    print(stats("  noise_nls", noise))
    cov = cov_stage(noise)
    print(stats("  cov_blocks", cov))
    F = fisher_stage(cov)
    print(stats("  F", F))
    sig = sigma_stage(F)
    print(stats("  sigma(r)", sig))
    return noise, cov, F, sig


def run_endtoend(opt_ctx, inst, jit_outer):
    """The "drift exists" reference: top-level jit'd vs not."""
    def f(n, net, beam, eta):
        return sigma_r_from_channels(
            n, net, beam, eta, opt_ctx,
            mission_years=inst.mission_duration_years, f_sky=inst.f_sky)
    if jit_outer:
        f = jax.jit(f)
    return float(f(opt_ctx.n_det, opt_ctx.net, opt_ctx.beam, opt_ctx.eta))


def main():
    inst = pico_like()
    ctx = make_optimization_context(
        inst, MomentExpansionModel(), CMBSpectra(),
        dict(FIDUCIAL_MOMENT), priors=DEFAULT_PRIORS_MOMENT,
        fixed_params=DEFAULT_FIXED_MOMENT,
        ell_min=2, ell_max=300, delta_ell=30)

    print("PICO 21-channel + MomentFG, ell=2-300, delta_ell=30")
    print(f"n_chan={ctx.n_det.shape[0]}, "
          f"n_pairs={len(ctx.signal_model.freq_pairs)}, "
          f"n_bins={ctx.signal_model.n_bins}, "
          f"n_free={len(ctx.signal_model.parameter_names) - 1}")  # T_dust fixed

    sig_eager = run_endtoend(ctx, inst, jit_outer=False)
    sig_jit = run_endtoend(ctx, inst, jit_outer=True)
    print(f"\n[end-to-end] eager sigma(r) = {sig_eager:.10e}")
    print(f"[end-to-end] jit   sigma(r) = {sig_jit:.10e}")
    print(f"[end-to-end] rel diff       = "
          f"{abs(sig_jit - sig_eager) / sig_eager:.2e}")

    eager = run_stagewise(ctx, inst, "eager")
    jit_  = run_stagewise(ctx, inst, "jit")

    print("\n=== STAGE-WISE rel diffs (jit vs eager) ===")
    print(f"  noise_nls:  {reldiff(jit_[0], eager[0]):.2e}")
    print(f"  cov_blocks: {reldiff(jit_[1], eager[1]):.2e}")
    print(f"  F:          {reldiff(jit_[2], eager[2]):.2e}")
    print(f"  sigma(r):   {reldiff(jit_[3], eager[3]):.2e}")


if __name__ == "__main__":
    main()
