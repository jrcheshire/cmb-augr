"""Diagnose L-BFGS gradient stability at PICO.

At z_pico (the 6-tier softmax that reproduces PICO's allocation):
  - jax.grad(sigma_r)(z) -> autodiff gradient
  - finite-difference gradient via central diff at h ~ 1e-4 -> reference
  - compare magnitudes, signs, alignment

Also test gradient reproducibility: jit vs eager.
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
from augr.foregrounds import MomentExpansionModel
from augr.optimize import make_optimization_context, sigma_r_from_channels
from augr.spectra import CMBSpectra

PICO_CHANNEL_TIERS = (
    (21.0, 25.0, 30.0),
    (36.0, 43.0, 52.0, 62.0),
    (75.0, 90.0, 108.0),
    (129.0, 155.0, 186.0),
    (223.0, 268.0, 321.0, 385.0, 462.0),
    (555.0, 666.0, 799.0),
)


def _tier_index_for_freq(nu_ghz):
    for t, g in enumerate(PICO_CHANNEL_TIERS):
        if any(abs(nu_ghz - x) < 1e-6 for x in g):
            return t
    raise ValueError(nu_ghz)


def main():
    inst = pico_like()
    ctx = make_optimization_context(
        inst, MomentExpansionModel(), CMBSpectra(),
        dict(FIDUCIAL_MOMENT), priors=DEFAULT_PRIORS_MOMENT,
        fixed_params=DEFAULT_FIXED_MOMENT,
        ell_min=2, ell_max=300, delta_ell=30)

    freqs = np.asarray(ctx.freqs)
    tier_idx = np.array([_tier_index_for_freq(float(f)) for f in freqs])
    tier_idx_j = jnp.asarray(tier_idx)
    n_det_pico = np.asarray(ctx.n_det)
    n_total = float(n_det_pico.sum())

    in_tier_ratio = np.zeros_like(n_det_pico)
    for t in range(len(PICO_CHANNEL_TIERS)):
        m = tier_idx == t
        in_tier_ratio[m] = n_det_pico[m] / n_det_pico[m].sum()
    in_tier_ratio_j = jnp.asarray(in_tier_ratio)

    pico_tier_total = np.array([
        n_det_pico[tier_idx == t].sum()
        for t in range(len(PICO_CHANNEL_TIERS))])
    z_pico_np = np.log(pico_tier_total / pico_tier_total.sum())
    z_pico = jnp.asarray(z_pico_np)

    def tier_to_ndet(z):
        tier_total = n_total * jax.nn.softmax(z)
        return tier_total[tier_idx_j] * in_tier_ratio_j

    def loss(z):
        return sigma_r_from_channels(
            tier_to_ndet(z), ctx.net, ctx.beam, ctx.eta, ctx,
            mission_years=inst.mission_duration_years, f_sky=inst.f_sky)

    sigma_pico = float(loss(z_pico))
    print(f"sigma(r) at z_pico = {sigma_pico:.6e}")

    # 1) jax.grad
    grad_eager = np.asarray(jax.grad(loss)(z_pico))
    grad_jit = np.asarray(jax.jit(jax.grad(loss))(z_pico))
    print(f"\njax.grad eager: {grad_eager}")
    print(f"jax.grad jit:   {grad_jit}")
    print(f"  rel diff (jit-vs-eager) per axis: "
          f"{np.abs(grad_jit - grad_eager) / np.maximum(np.abs(grad_eager), 1e-300)}")

    # 2) finite-difference at multiple h
    print("\nFinite-difference gradients (central diff):")
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        fd = np.zeros(6)
        for i in range(6):
            zp = z_pico_np.copy()
            zp[i] += h
            zm = z_pico_np.copy()
            zm[i] -= h
            fd[i] = (float(loss(jnp.asarray(zp))) -
                     float(loss(jnp.asarray(zm)))) / (2 * h)
        rel = np.linalg.norm(fd - grad_eager) / np.linalg.norm(grad_eager)
        cos = float(np.dot(fd, grad_eager) /
                    (np.linalg.norm(fd) * np.linalg.norm(grad_eager)))
        print(f"  h={h:.0e}: ||fd-grad||/||grad||={rel:.2e}  "
              f"cos(angle)={cos:.6f}  ||fd||={np.linalg.norm(fd):.3e}")
        if h == 1e-4:
            print(f"    fd: {fd}")

    # 3) Magnitudes
    print(f"\n|grad| / |sigma|: "
          f"{np.linalg.norm(grad_eager) / sigma_pico:.3e}")
    print(f"|grad|_inf:       {np.abs(grad_eager).max():.3e}")
    print(f"|grad|_min:       {np.abs(grad_eager).min():.3e}")

    # 4) descent direction from each: where does loss go after one step?
    print("\nLoss after one step of size step in -grad direction:")
    for step in (1e-3, 1e-2, 1e-1, 1.0):
        d = -grad_eager / np.linalg.norm(grad_eager)
        z_new = z_pico_np + step * d
        sig_new = float(loss(jnp.asarray(z_new)))
        print(f"  step={step:.0e}: sigma -> {sig_new:.6e} "
              f"(delta={sig_new - sigma_pico:+.3e})")


if __name__ == "__main__":
    main()
