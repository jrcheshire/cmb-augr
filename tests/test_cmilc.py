"""Gate for augr.cmilc: constrained-moment ILC algebra + moment SEDs + the headline
spatial-β_dust residual reduction vs blind NILC.

The constrained solve, the ``Aᵀw = e`` deprojection, the moment SED columns (vs the
in-repo ``augr.units`` ground truth), and the active-channel degradation are exercised
with cheap linear algebra / small synthetic skies in the per-PR gate. The realistic d10
(spatially-varying β_dust) reduction and the grad-vs-FD differentiability check carry
``slow`` (PySM download / grad-through-SHT; out of the parallel gate). Map-based tests
need ducc0 (the [compsep] extra) — which the import chain also requires.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

from augr import units
from augr.cmilc import (
    CMILC06_MOMENTS,
    CMILC08_MOMENTS,
    _cilc_weights_from_cov,
    _global_cilc_weights,
    cmilc_clean,
    moment_sed_vectors,
)
from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.gnilc import alm2cl
from augr.nilc import _ilc_weights_from_cov, nilc_clean
from augr.nilc_forecast import nilc_spectra
from augr.spectra import CMBSpectra

# Wide space-mission-like 8-band config: cMILC08 (6 constraints) keeps >=2 DoF for the
# variance minimization at every band (need n_band > n_constraints to clean at all).
FREQS = (30.0, 45.0, 95.0, 150.0, 220.0, 280.0, 353.0, 402.0)
BEAMS = (72.0, 52.0, 28.0, 20.0, 14.0, 12.0, 10.0, 9.0)
W_INV = jnp.array([2.0e-4, 1.2e-4, 5.0e-5, 5.0e-5, 8.0e-5, 1.5e-4, 4.0e-4, 6.0e-4])


def _sim(nside: int, lmax: int, *, seed: int = 1, r_in: float = 0.0, fg_model=None):
    """Beamed sky + total/noise map sets, same recipe as test_gnilc."""
    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=r_in,
        nside=nside,
        lmax=lmax,
        fg_model=fg_model,
        cmb_seed=seed,
    )
    hit = jnp.ones(12 * nside * nside)
    total = assemble_band_maps(sky, W_INV, hit, noise_key=jax.random.PRNGKey(seed))
    noise = total - sky.cmb_qu - sky.fg_qu
    return sky, total, noise


def _band_mean(cl: np.ndarray, lo: int, hi: int) -> float:
    ell = np.arange(cl.shape[-1])
    return float(cl[(ell >= lo) & (ell < hi)].mean())


# --- constrained-ILC weight solve (pure linear algebra, no maps) -----------


def test_cilc_reduces_to_ilc() -> None:
    """A = ones, e = [1] recovers the blind NILC weights to machine precision."""
    rng = np.random.default_rng(0)
    n = 6
    a = rng.normal(size=(n, n))
    cov = jnp.asarray(a @ a.T + n * np.eye(n))
    w_c = _cilc_weights_from_cov(cov, jnp.ones((n, 1)), jnp.array([1.0]))
    w_i = _ilc_weights_from_cov(cov)
    np.testing.assert_allclose(np.asarray(w_c), np.asarray(w_i), rtol=1e-12, atol=1e-14)


def test_cilc_constraint_holds() -> None:
    """The constrained weights satisfy Aᵀw = e exactly for an arbitrary A."""
    rng = np.random.default_rng(1)
    n, k = 7, 4
    a = rng.normal(size=(n, n))
    cov = jnp.asarray(a @ a.T + n * np.eye(n))
    A = jnp.asarray(rng.normal(size=(n, k)))
    e = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    w = _cilc_weights_from_cov(cov, A, e)
    np.testing.assert_allclose(np.asarray(A.T @ w), np.asarray(e), rtol=0, atol=1e-12)


def test_cilc_constraint_holds_batched() -> None:
    """The per-pixel (batched-cov) path satisfies the constraint at every pixel."""
    rng = np.random.default_rng(2)
    n, k, npix = 6, 3, 5
    a = rng.normal(size=(n, n))
    cov = jnp.broadcast_to(jnp.asarray(a @ a.T + n * np.eye(n)), (npix, n, n))
    A = jnp.asarray(rng.normal(size=(n, k)))
    e = jnp.asarray([1.0, 0.0, 0.0])
    w = _cilc_weights_from_cov(cov, A, e)  # (npix, n)
    resid = jnp.einsum("nk,pn->pk", A, w) - e
    np.testing.assert_allclose(np.asarray(resid), 0.0, atol=1e-12)


# --- moment SED mixing matrix ----------------------------------------------


def test_moment_sed_vectors_match_units() -> None:
    """Each column equals the in-repo units.py SED (× log-derivative); CMB col is ones."""
    A = moment_sed_vectors(FREQS, moments=CMILC08_MOMENTS)
    nu = jnp.asarray(FREQS)
    bd, td, bs = 1.6, 19.6, -3.1  # FIDUCIAL_BK15 pivots
    f_d = units.dust_sed(nu, bd, td)
    f_s = units.sync_sed(nu, bs)
    assert A.shape == (len(FREQS), 1 + len(CMILC08_MOMENTS))
    np.testing.assert_allclose(np.asarray(A[:, 0]), 1.0, atol=1e-14)  # CMB = flat
    np.testing.assert_allclose(np.asarray(A[:, 1]), np.asarray(f_d), rtol=1e-12)  # f_dust
    np.testing.assert_allclose(np.asarray(A[:, 2]), np.asarray(f_s), rtol=1e-12)  # f_sync
    np.testing.assert_allclose(
        np.asarray(A[:, 3]), np.asarray(f_d * units.dust_sed_deriv_beta(nu)), rtol=1e-12
    )  # ∂_β f_dust
    np.testing.assert_allclose(
        np.asarray(A[:, 4]), np.asarray(f_s * units.sync_sed_deriv_beta(nu)), rtol=1e-12
    )  # ∂_β f_sync
    np.testing.assert_allclose(
        np.asarray(A[:, 5]), np.asarray(f_d * units.dust_sed_deriv_T(nu, td)), rtol=1e-12
    )  # ∂_T f_dust


def test_moment_sed_column_counts() -> None:
    assert moment_sed_vectors(FREQS, moments=CMILC08_MOMENTS).shape[1] == 6
    assert moment_sed_vectors(FREQS, moments=CMILC06_MOMENTS).shape[1] == 4


def test_moment_sed_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown moment key"):
        moment_sed_vectors(FREQS, moments=("f_dust", "nope"))


# --- active-channel degradation (synthetic beta + mask; no SHT) -------------


def test_active_channel_degradation() -> None:
    """A band with fewer active channels than constraints keeps the leading columns of A
    (CMB first), and the truncated constraint still holds on its active channels."""
    rng = np.random.default_rng(3)
    n_band, npix = 6, 48
    beta = jnp.asarray(rng.normal(size=(2, n_band, npix)))
    A = moment_sed_vectors(FREQS[:n_band], moments=CMILC08_MOMENTS)  # (6, 6) -> k_full = 6
    e = jnp.zeros(A.shape[1]).at[0].set(1.0)
    active = np.array(
        [[True] * 6, [True, True, True, False, False, False]]  # band 0: 6 active; band 1: 3 active
    )
    weights, cols = _global_cilc_weights(beta, A, e, 1e-10, active)
    assert cols == (6, 3)  # band 1 degrades to CMB + 2 moments

    # band 0: full constraint; band 1: leading-3-column constraint on its active channels.
    np.testing.assert_allclose(np.asarray(A.T @ weights[0]), np.asarray(e), atol=1e-10)
    idx1 = np.array([0, 1, 2])
    np.testing.assert_allclose(
        np.asarray(A[idx1][:, :3].T @ weights[1][idx1]), np.asarray(e[:3]), atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(weights[1][3:]), 0.0, atol=0)  # inactive -> exactly 0


# --- end-to-end on maps (network-free smoke; in the per-PR gate) ------------


def test_cmilc_cmb_transfer_unity() -> None:
    """cMILC preserves the CMB exactly (the Aᵀw=e CMB constraint): transfer -> 1.

    Network-free (fg_model=None) end-to-end smoke kept in the gate — exercises the full
    needlet / common-resolution / constrained-weight / recompose wiring and the
    NILCResult-consumer contract (nilc_spectra).
    """
    nside, lmax = 32, 64
    sky, total, _ = _sim(nside, lmax, fg_model=None)
    res, info = cmilc_clean(total, BEAMS, FREQS, lmax=lmax, nside=nside, return_diagnostics=True)
    assert info["n_constraints"] == 6
    assert all(k == 6 for k in info["retained_columns_per_band"])  # all bands resolve at low lmax

    zero = jnp.zeros_like(total)
    spec = nilc_spectra(res, total_qu=total, noise_qu=zero, fg_qu=zero, cmb_qu=sky.cmb_qu)
    transfer = float(np.mean(spec.transfer[(spec.ells >= 20) & (spec.ells <= 50)]))
    np.testing.assert_allclose(transfer, 1.0, rtol=2e-3)


# --- headline science: cMILC nulls the dust moment NILC leaves --------------
# Builds a realistic PySM d10 sky (spatially-varying β_dust, T_dust) over the network on a
# cold cache, so it carries `slow` (out of the parallel per-PR gate).


@pytest.mark.slow
def test_cmilc_reduces_fg_residual_vs_nilc() -> None:
    """On a spatially-varying-β_dust sky (d10), cMILC08's moment deprojection cuts the
    FG residual that blind NILC leaves by a large factor (≈20x at the recombination bump;
    measured 0.03-0.17 of NILC across ℓ). Loose bar: a >2x reduction in ℓ=30-100."""
    pytest.importorskip("pysm3")
    nside, lmax = 64, 128
    sky, total, _ = _sim(nside, lmax, fg_model="d10s5", seed=1)
    res_nilc = nilc_clean(total, BEAMS, lmax=lmax, nside=nside)
    res_cmilc = cmilc_clean(total, BEAMS, FREQS, lmax=lmax, nside=nside, moments=CMILC08_MOMENTS)

    fg_nilc = np.asarray(alm2cl(res_nilc.project(sky.fg_qu), lmax))
    fg_cmilc = np.asarray(alm2cl(res_cmilc.project(sky.fg_qu), lmax))
    assert np.all(np.isfinite(fg_cmilc))
    nilc_band = _band_mean(fg_nilc, 30, 100)
    cmilc_band = _band_mean(fg_cmilc, 30, 100)
    assert cmilc_band > 0
    assert cmilc_band < 0.5 * nilc_band, (cmilc_band, nilc_band, cmilc_band / nilc_band)


# --- differentiability (gates the per-band eigh/solve degeneracy risk) ------


@pytest.mark.slow
def test_cmilc_differentiable_in_noise() -> None:
    """The cleaned-map BB power is differentiable in the per-band noise scale (the weights
    depend on the data covariance); grad matches finite differences. Network-free."""
    nside, lmax = 32, 64
    sky = generate_band_sky(
        FREQS,
        BEAMS,
        spectra=CMBSpectra(),
        r_in=0.0,
        nside=nside,
        lmax=lmax,
        fg_model=None,
        cmb_seed=2,
    )
    hit = jnp.ones(12 * nside * nside)
    key = jax.random.PRNGKey(2)
    ell = jnp.arange(lmax + 1)
    band = (ell >= 4) & (ell <= 50)

    def power(scale):
        total = assemble_band_maps(sky, scale * W_INV, hit, noise_key=key)
        res = cmilc_clean(total, BEAMS, FREQS, lmax=lmax, nside=nside)
        return jnp.sum(jnp.where(band, alm2cl(res.cleaned_b_alm, lmax), 0.0))

    g = float(jax.grad(power)(1.0))
    assert np.isfinite(g)
    eps = 1e-3
    fd = float((power(1.0 + eps) - power(1.0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=2e-2, atol=1e-12)
