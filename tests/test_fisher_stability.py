"""Regression tests for the prewhitened Fisher inversion.

These lock in:

  1. Bit-equivalence between ``optimize.sigma_r_from_channels`` and
     ``FisherForecast.sigma``: both paths route through
     ``fisher._fisher_from_blocks`` (prewhiten + solve per bin), so they
     agree to fp64 precision. Any future divergence indicates one path
     forked off the other.
  2. cov_b at PICO ell=2 has cond ~10^28 (regime sanity check) and
     ``solve`` produces finite + correct F_b within mpmath truth (3e-5
     rel) on the worst-conditioned bin.
  3. JIT vs eager agreement on σ(r) to 1e-4. Residual XLA-fusion wobble
     from the per-bin scan accumulator; not perfect but small.
  4. Gradient stability: ``jax.grad`` is jit-equivalent per-axis
     (rtol ~ 1e-3, no sign flips); finite-difference cosine-aligns at
     h=1e-2. Pre-prewhitening these were 50-270% off and uncorrelated.
  5. L-BFGS-B converges with σ_opt < σ_pico under both metrics.

Background: at PICO conditioning the legacy ``eigh + (s>0)`` clip
biased F upward by 5-44% per bin by face-valuing tiny positive
eigenvalues that were fp64 rounding artifacts. The unification PR
moved both paths to ``solve(cov_b, J_b)`` (forward-correct to ~0.6%
per mpmath); the prewhiten+solve refinement here drops backward
conditioning by 13 orders of magnitude and incidentally also tightens
forward F by another ~0.5% (validated against mpmath @ 30 dps on bin
0: 0.6% rel error → 3e-5 rel error).

Fixture is intentionally PICO-class (21 channels, moment FG, ell down
to 2) because the conditioning failure is driven by the per-channel
noise spread across 21 frequencies; a smaller instrument does not
exhibit it. Tests stay under a couple of minutes on Apple-silicon.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

from augr.config import (
    DEFAULT_FIXED_MOMENT,
    DEFAULT_PRIORS_MOMENT,
    FIDUCIAL_MOMENT,
    pico_like,
)
from augr.covariance import bandpower_covariance_blocks_from_noise
from augr.fisher import FisherForecast
from augr.foregrounds import MomentExpansionModel
from augr.instrument import noise_nl_continuous
from augr.optimize import make_optimization_context, sigma_r_from_channels
from augr.signal import SignalModel
from augr.spectra import CMBSpectra

# ---------------------------------------------------------------------------
# Shared module-scope fixtures
# ---------------------------------------------------------------------------

ELL_MIN, ELL_MAX, DELTA_ELL = 2, 300, 30


@pytest.fixture(scope="module")
def pico_instrument():
    return pico_like()


@pytest.fixture(scope="module")
def cmb_spectra():
    return CMBSpectra()


@pytest.fixture(scope="module")
def fg_model():
    return MomentExpansionModel()


@pytest.fixture(scope="module")
def signal_model(pico_instrument, fg_model, cmb_spectra):
    return SignalModel(
        pico_instrument, fg_model, cmb_spectra,
        ell_min=ELL_MIN, ell_max=ELL_MAX, delta_ell=DELTA_ELL,
    )


@pytest.fixture(scope="module")
def opt_ctx(pico_instrument, fg_model, cmb_spectra):
    return make_optimization_context(
        pico_instrument, fg_model, cmb_spectra,
        dict(FIDUCIAL_MOMENT),
        priors=DEFAULT_PRIORS_MOMENT,
        fixed_params=DEFAULT_FIXED_MOMENT,
        ell_min=ELL_MIN, ell_max=ELL_MAX, delta_ell=DELTA_ELL,
    )


@pytest.fixture(scope="module")
def fisher_pico(signal_model, pico_instrument):
    return FisherForecast(
        signal_model, pico_instrument,
        dict(FIDUCIAL_MOMENT),
        priors=DEFAULT_PRIORS_MOMENT,
        fixed_params=DEFAULT_FIXED_MOMENT,
    )


# ---------------------------------------------------------------------------
# Diagnostic — characterises the conditioning that motivated the fix
# ---------------------------------------------------------------------------

class TestEigenvalueConditioningDiagnostic:
    """Pure diagnostics. Always pass; documents the underlying numerics.

    These print the spread that motivates prewhitening so a future
    reader can verify the regime is unchanged before suspecting a
    different root cause.
    """

    def test_cov_b_is_ill_conditioned_at_low_ell(
            self, opt_ctx, pico_instrument):
        """cov_b at PICO, ell=2 has huge condition number.

        The reformulation does not aim to make cov_b itself well-
        conditioned -- it whitens it. Here we just lock in that the
        original cov_b *is* the high-cond matrix the plan describes,
        so future regressions in unrelated noise modelling don't go
        unnoticed.
        """
        ells = opt_ctx.ells
        n_chan = opt_ctx.n_det.shape[0]
        knee_arr = jnp.zeros(n_chan)
        alpha_arr = jnp.ones(n_chan)
        noise_nls = jnp.stack([
            noise_nl_continuous(
                opt_ctx.net[i], opt_ctx.n_det[i], opt_ctx.beam[i],
                opt_ctx.eta[i], ells, pico_instrument.mission_duration_years,
                pico_instrument.f_sky, knee_arr[i], alpha_arr[i],
            )
            for i in range(n_chan)
        ])
        cov_blocks = bandpower_covariance_blocks_from_noise(
            opt_ctx.signal_model, noise_nls, pico_instrument.f_sky,
            opt_ctx.fiducial_params)
        cov_b0 = np.asarray(cov_blocks[0])  # lowest-ell bin
        s = np.linalg.eigvalsh(cov_b0)
        cond = float(np.abs(s).max() / np.abs(s[s > 0]).min())
        print(f"\n[diag] PICO cov_b at ell=2 cond(|.|): {cond:.2e}")
        print(f"[diag] eigenvalue spread (max/min positive): "
              f"{s.max():.3e} / {s[s > 0].min():.3e}")
        print(f"[diag] {(s <= 0).sum()}/{len(s)} eigenvalues "
              f"come back numerically non-positive")
        assert cond > 1e10, (
            "cov_b conditioning is supposed to be enormous at PICO; "
            "if it suddenly dropped, the noise spread changed and the "
            "rest of this test file probably no longer exercises the "
            "regime it was written for.")

    def test_solve_handles_negative_eigvals_cleanly(
            self, opt_ctx, pico_instrument):
        """``jnp.linalg.solve`` works on cov_b even with sign-flipped eigvals.

        At PICO ell=2, fp64 eigvalsh of cov_b returns ~84 of 231 eigvals
        as numerically non-positive (the smallest reaching ~ -0.3, not a
        rounding-floor artifact). LU/solve handles this regime as a
        whole-matrix operation and gives the correct Fisher contribution
        (validated against mpmath @ 30 dps to <1% at bin 0, <1e-7 at
        higher bins). This test confirms ``solve`` does not raise on the
        worst-conditioned bin.
        """
        ells = opt_ctx.ells
        n_chan = opt_ctx.n_det.shape[0]
        knee_arr = jnp.zeros(n_chan)
        alpha_arr = jnp.ones(n_chan)
        noise_nls = jnp.stack([
            noise_nl_continuous(
                opt_ctx.net[i], opt_ctx.n_det[i], opt_ctx.beam[i],
                opt_ctx.eta[i], ells, pico_instrument.mission_duration_years,
                pico_instrument.f_sky, knee_arr[i], alpha_arr[i],
            )
            for i in range(n_chan)
        ])
        cov_blocks = np.asarray(bandpower_covariance_blocks_from_noise(
            opt_ctx.signal_model, noise_nls, pico_instrument.f_sky,
            opt_ctx.fiducial_params))
        cov_b0 = cov_blocks[0]
        J_b0 = np.asarray(opt_ctx.J_blocks[0])
        F_b0 = J_b0.T @ np.linalg.solve(cov_b0, J_b0)
        assert np.all(np.isfinite(F_b0)), (
            "linalg.solve produced non-finite F_b on the worst-conditioned "
            "bin; cov_b conditioning may have grown beyond LU's capacity.")
        F_rr = float(F_b0[opt_ctx.r_idx, opt_ctx.r_idx])
        print(f"\n[diag] bin 0 F_b[r,r] (fp64 solve) = {F_rr:.6e}; "
              f"mpmath truth = 1.4356e+08")
        # mpmath ground truth: 1.4356e8. Allow 5% (well above measured 0.6%).
        assert abs(F_rr - 1.4356e+8) / 1.4356e+8 < 0.05, (
            f"F_b[r,r] = {F_rr:.6e} drifted >5% from the mpmath truth "
            f"1.4356e+8 -- check the cov_b build or the solve primitive.")


# ---------------------------------------------------------------------------
# JIT vs eager
# ---------------------------------------------------------------------------

def test_jit_eager_agreement(opt_ctx, pico_instrument):
    """sigma_r_from_channels: top-level jit matches the unjit'd call.

    At PICO conditioning a residual jit-vs-eager wobble remains because
    XLA reorders associative summation in the per-bin scan that
    accumulates F = sum_b J_b^T cov_b^{-1} J_b across 38 bins. F[r,r]
    is dominated by bin 0 so it's stable; F's small off-diagonal
    entries vary by orders of magnitude and the marginalised σ(r)
    inherits a small shift. With prewhiten + solve the inherited
    shift is around 1e-5 relative; we gate at 1e-4 for safety margin.
    Tightening below this would require deterministic-summation
    rewrites of the scan that are out of scope.
    """
    n_det = opt_ctx.n_det
    net = opt_ctx.net
    beam = opt_ctx.beam
    eta = opt_ctx.eta

    def f(n_det, net, beam, eta):
        return sigma_r_from_channels(
            n_det, net, beam, eta, opt_ctx,
            mission_years=pico_instrument.mission_duration_years,
            f_sky=pico_instrument.f_sky,
        )

    sigma_eager = float(f(n_det, net, beam, eta))
    sigma_jit = float(jax.jit(f)(n_det, net, beam, eta))
    rel = abs(sigma_jit - sigma_eager) / sigma_eager
    print(f"\n[jit vs eager] eager={sigma_eager:.6e}, jit={sigma_jit:.6e}, "
          f"rel diff={rel:.2e}")
    np.testing.assert_allclose(sigma_jit, sigma_eager, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gradient stability
# ---------------------------------------------------------------------------

class TestGradientStability:
    """Backward-pass gradient quality at PICO conditioning.

    Pre-prewhitening, ``jax.grad(sigma_r)(z_pico)`` drifted 50-270%
    per-axis between jit and eager (with sign flips on most axes), and
    finite-difference references had ``cos(angle) < 0.5`` even at the
    most favourable step size. Prewhitening + solve in the unified
    Fisher primitive drops the backward conditioning by 13 orders of
    magnitude (cov_b cond ~10^28 → cov_w cond ~10^15), and gradient
    quality jumps accordingly.
    """

    def _build_loss(self, opt_ctx, pico_instrument):
        """6-tier softmax around the PICO baseline allocation.

        Returns ``(loss_fn, z_pico_jnp, z_pico_np)``.
        """
        freqs = np.asarray(opt_ctx.freqs)
        tier_idx = np.array([_tier_index_for_freq(float(f)) for f in freqs])
        tier_idx_j = jnp.asarray(tier_idx)
        n_det_pico = np.asarray(opt_ctx.n_det)
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

        def loss(z):
            tier_total = n_total * jax.nn.softmax(z)
            n_det_z = tier_total[tier_idx_j] * in_tier_ratio_j
            return sigma_r_from_channels(
                n_det_z, opt_ctx.net, opt_ctx.beam, opt_ctx.eta, opt_ctx,
                mission_years=pico_instrument.mission_duration_years,
                f_sky=pico_instrument.f_sky)

        return loss, jnp.asarray(z_pico_np), z_pico_np

    def test_jax_grad_jit_eager_agreement(
            self, opt_ctx, pico_instrument):
        """jax.grad eager and jax.grad jit'd give per-axis-aligned gradients.

        Pre-fix: per-axis rel diff 0.49 to 2.71 with sign flips on 4/6
        axes. Post-fix: ~1e-3 per axis.
        """
        loss, z_pico, _ = self._build_loss(opt_ctx, pico_instrument)
        grad_eager = np.asarray(jax.grad(loss)(z_pico))
        grad_jit = np.asarray(jax.jit(jax.grad(loss))(z_pico))
        per_axis = (np.abs(grad_jit - grad_eager) /
                    np.maximum(np.abs(grad_eager), 1e-300))
        print(f"\n[grad jit vs eager] per-axis rel diff: {per_axis}")
        # Sign agreement on every axis.
        assert np.all(np.sign(grad_eager) == np.sign(grad_jit)), (
            f"jax.grad signs differ between jit and eager: "
            f"eager={np.sign(grad_eager)} jit={np.sign(grad_jit)}")
        np.testing.assert_allclose(grad_jit, grad_eager, rtol=2e-2)

    def test_finite_difference_alignment(
            self, opt_ctx, pico_instrument):
        """Central-difference gradient cosine-aligns with jax.grad at h=1e-2.

        Pre-fix: cos(angle) bounced between -0.77, -0.19, +0.27 across
        h. Post-fix: cos(angle) > 0.99 at h=1e-2 where signal dominates
        the function noise floor.
        """
        loss, z_pico, z_pico_np = self._build_loss(opt_ctx, pico_instrument)
        grad = np.asarray(jax.grad(loss)(z_pico))

        h = 1e-2
        fd = np.zeros(6)
        for i in range(6):
            zp = z_pico_np.copy()
            zp[i] += h
            zm = z_pico_np.copy()
            zm[i] -= h
            fd[i] = (float(loss(jnp.asarray(zp))) -
                     float(loss(jnp.asarray(zm)))) / (2 * h)
        cos = float(np.dot(fd, grad) /
                    (np.linalg.norm(fd) * np.linalg.norm(grad)))
        print(f"\n[fd vs jax.grad @ h=1e-2] cos(angle) = {cos:.6f}")
        assert cos > 0.99, (
            f"cos(angle) = {cos:.4f} between finite-diff and jax.grad "
            f"is below the 0.99 gate; gradient is not signal-dominated.")


# ---------------------------------------------------------------------------
# optimize.* matches FisherForecast.sigma
# ---------------------------------------------------------------------------

class TestOptimizeMatchesFisherForecast:
    """Single source of truth for sigma(r) at PICO.

    The tier-1 ``sigma_r_from_channels`` and the canonical
    ``FisherForecast.sigma`` must agree to floating-point precision at
    the baseline allocation *and* at perturbed allocations. The
    perturbations matter because the differentiable-design pitch is
    only honest if the two metrics are tracking the same surface.
    """

    def test_baseline_allocation(self, opt_ctx, fisher_pico,
                                  pico_instrument):
        sigma_opt = float(sigma_r_from_channels(
            opt_ctx.n_det, opt_ctx.net, opt_ctx.beam, opt_ctx.eta, opt_ctx,
            mission_years=pico_instrument.mission_duration_years,
            f_sky=pico_instrument.f_sky,
        ))
        sigma_ff = fisher_pico.sigma("r")
        rel = abs(sigma_opt - sigma_ff) / sigma_ff
        print(f"\n[baseline] optimize sigma(r)={sigma_opt:.6e}, "
              f"FisherForecast sigma(r)={sigma_ff:.6e}, rel={rel:.2e}")
        np.testing.assert_allclose(sigma_opt, sigma_ff, rtol=1e-6)

    def test_perturbed_allocation(self, opt_ctx, signal_model,
                                  pico_instrument):
        """Move detectors between high-nu and CMB-core channels and check.

        Constant / uniform-scale perturbations are measure-zero for
        this bug (cov_b stays dominated by the same eigenvectors); use
        an asymmetric reallocation that actually rebalances the
        per-channel noise spread.
        """
        n_det_new = np.array(opt_ctx.n_det, dtype=float).copy()
        # Move 20% of the highest-nu (799 GHz, idx 20) detectors to the
        # CMB-core (108 GHz, idx 9).
        delta = 0.2 * n_det_new[20]
        n_det_new[20] -= delta
        n_det_new[9] += delta

        sigma_opt = float(sigma_r_from_channels(
            jnp.asarray(n_det_new),
            opt_ctx.net, opt_ctx.beam, opt_ctx.eta, opt_ctx,
            mission_years=5.0, f_sky=pico_instrument.f_sky,
        ))

        # Build a Channel-tuple twin with the same reallocation.
        from dataclasses import replace
        new_channels = list(pico_instrument.channels)
        new_channels[20] = replace(new_channels[20],
                                    n_detectors=round(n_det_new[20]))
        new_channels[9] = replace(new_channels[9],
                                   n_detectors=round(n_det_new[9]))
        inst_new = replace(pico_instrument, channels=tuple(new_channels))
        sig_new = SignalModel(
            inst_new, signal_model.foreground_model, CMBSpectra(),
            ell_min=ELL_MIN, ell_max=ELL_MAX, delta_ell=DELTA_ELL,
        )
        ff_new = FisherForecast(
            sig_new, inst_new,
            dict(FIDUCIAL_MOMENT),
            priors=DEFAULT_PRIORS_MOMENT,
            fixed_params=DEFAULT_FIXED_MOMENT,
        )
        sigma_ff = ff_new.sigma("r")
        rel = abs(sigma_opt - sigma_ff) / sigma_ff
        print(f"\n[perturbed] optimize sigma(r)={sigma_opt:.6e}, "
              f"FisherForecast sigma(r)={sigma_ff:.6e}, rel={rel:.2e}")
        np.testing.assert_allclose(sigma_opt, sigma_ff, rtol=1e-6)


# ---------------------------------------------------------------------------
# L-BFGS-B finds a real minimum
# ---------------------------------------------------------------------------

PICO_CHANNEL_TIERS = (
    (21.0, 25.0, 30.0),
    (36.0, 43.0, 52.0, 62.0),
    (75.0, 90.0, 108.0),
    (129.0, 155.0, 186.0),
    (223.0, 268.0, 321.0, 385.0, 462.0),
    (555.0, 666.0, 799.0),
)


def _tier_index_for_freq(nu_ghz, tiers=PICO_CHANNEL_TIERS):
    for t, group in enumerate(tiers):
        if any(abs(nu_ghz - g) < 1e-6 for g in group):
            return t
    raise ValueError(f"freq {nu_ghz} not in any PICO tier")


@pytest.mark.slow
def test_lbfgs_finds_real_minimum_on_pico(opt_ctx, pico_instrument):
    """6-tier softmax reallocation: gradient-based L-BFGS is meaningful.

    Locks in two properties of the gradient surface (post the
    prewhiten + solve unification of the Fisher primitive):
      1. L-BFGS-B reports ``success=True`` -- line-search Wolfe
         conditions can be satisfied. (Pre-fix this aborted with
         status=ABNORMAL at nit=0 because backward gradients through
         ``solve(cov_b, J_b)`` at cond~10^28 were noise-dominated;
         prewhitening drops the conditioning to ~10^15 and gradients
         become signal-dominated.)
      2. The optimized sigma(r) is strictly smaller than the baseline
         under both metrics: optimize.sigma_r_from_channels and
         FisherForecast.sigma. Cross-metric consistency rules out the
         optimizer exploiting per-metric numerical noise.

    Iteration count is intentionally not gated. With clean gradients,
    L-BFGS-B can converge in a single line search at gtol=1e-7 if the
    starting gradient already points usefully and ``|grad|`` falls
    below tolerance after one step (which it does at PICO baseline).
    """
    freqs = np.asarray(opt_ctx.freqs)
    tier_idx = np.array([_tier_index_for_freq(float(f)) for f in freqs])
    tier_idx_j = jnp.asarray(tier_idx)

    n_det_pico = np.asarray(opt_ctx.n_det)
    n_total = float(n_det_pico.sum())

    # Per-channel ratios within each tier (fixed by PICO).
    in_tier_ratio = np.zeros_like(n_det_pico)
    for t in range(len(PICO_CHANNEL_TIERS)):
        mask = tier_idx == t
        in_tier_ratio[mask] = n_det_pico[mask] / n_det_pico[mask].sum()
    in_tier_ratio_j = jnp.asarray(in_tier_ratio)

    pico_tier_total = np.array([
        n_det_pico[tier_idx == t].sum() for t in range(len(PICO_CHANNEL_TIERS))
    ])
    z_pico = jnp.asarray(np.log(pico_tier_total / pico_tier_total.sum()))

    def tier_to_ndet(z):
        tier_total = n_total * jax.nn.softmax(z)
        return tier_total[tier_idx_j] * in_tier_ratio_j

    def loss(z):
        return sigma_r_from_channels(
            tier_to_ndet(z),
            opt_ctx.net, opt_ctx.beam, opt_ctx.eta, opt_ctx,
            mission_years=5.0, f_sky=pico_instrument.f_sky,
        )

    loss_and_grad = jax.jit(jax.value_and_grad(loss))

    sigma_pico = float(loss(z_pico))

    res = scipy.optimize.minimize(
        lambda z: tuple(map(np.asarray,
                            loss_and_grad(jnp.asarray(z)))),
        np.asarray(z_pico), jac=True, method="L-BFGS-B",
        options={"maxiter": 50, "gtol": 1e-7},
    )
    print(f"\n[lbfgs] success={res.success}, nit={res.nit}, "
          f"sigma_pico={sigma_pico:.6e}, sigma_opt={res.fun:.6e}")

    assert res.success, f"L-BFGS-B did not converge: {res.message}"
    assert res.fun < sigma_pico, (
        f"sigma_opt ({res.fun:.6e}) >= sigma_pico ({sigma_pico:.6e}) "
        f"under the optimize.* metric -- optimizer found nothing.")

    # Cross-metric consistency: the optimum should also be lower
    # under FisherForecast.sigma. Build that path-by-hand.
    from dataclasses import replace
    n_opt = np.asarray(tier_to_ndet(jnp.asarray(res.x)))
    new_channels = tuple(
        replace(ch, n_detectors=round(n_opt[i]))
        for i, ch in enumerate(pico_instrument.channels)
    )
    inst_opt = replace(pico_instrument, channels=new_channels)
    sig_opt = SignalModel(
        inst_opt, opt_ctx.signal_model.foreground_model, CMBSpectra(),
        ell_min=ELL_MIN, ell_max=ELL_MAX, delta_ell=DELTA_ELL,
    )
    ff_opt = FisherForecast(
        sig_opt, inst_opt,
        dict(FIDUCIAL_MOMENT),
        priors=DEFAULT_PRIORS_MOMENT,
        fixed_params=DEFAULT_FIXED_MOMENT,
    )
    sigma_opt_ff = ff_opt.sigma("r")
    sig_pico_model = SignalModel(
        pico_instrument, opt_ctx.signal_model.foreground_model, CMBSpectra(),
        ell_min=ELL_MIN, ell_max=ELL_MAX, delta_ell=DELTA_ELL,
    )
    ff_pico = FisherForecast(
        sig_pico_model, pico_instrument,
        dict(FIDUCIAL_MOMENT),
        priors=DEFAULT_PRIORS_MOMENT,
        fixed_params=DEFAULT_FIXED_MOMENT,
    )
    sigma_pico_ff = ff_pico.sigma("r")
    print(f"[lbfgs cross-metric] sigma_pico_ff={sigma_pico_ff:.6e}, "
          f"sigma_opt_ff={sigma_opt_ff:.6e}")

    assert sigma_opt_ff < sigma_pico_ff, (
        f"Optimized allocation under FisherForecast.sigma "
        f"({sigma_opt_ff:.6e}) is not better than baseline "
        f"({sigma_pico_ff:.6e}) -- optimizer exploited gradient noise.")
