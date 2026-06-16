"""Stage-3 intrinsic gates for augr.nilc: needlet algebra, ILC constraint, CMB
transfer, the resolution->down-weighting coupling, and differentiability.

All gates here are PySM-free (random/synthetic skies). The foreground-leakage
validation on realistic PySM skies is Stage 5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

from augr.compsep_sims import assemble_band_maps, generate_band_sky
from augr.nilc import (
    _ilc_weights_from_cov,
    _ilc_weights_masked,
    _needlet_channel_mask,
    _ridge,
    combine_needlets,
    common_resolution_b_alm,
    cosine_needlet_bands,
    default_needlet_peaks,
    needlet_beta,
    nilc_clean,
)
from augr.sht import _m_of_alm, alm_size, map2alm
from augr.spectra import CMBSpectra


def _rand_b_alm(seed: int, lmax: int) -> jax.Array:
    """Random scalar (B-like) alm with imag(m=0)=0."""
    nlm = alm_size(lmax)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal(nlm) + 1j * rng.standard_normal(nlm)).astype(np.complex128)
    a[_m_of_alm(lmax) == 0] = a[_m_of_alm(lmax) == 0].real
    return jnp.asarray(a)


def _cmb_sky(beams, *, r_in, nside, lmax, seed=0):
    """CMB-only BandSky through the given per-band beams."""
    return generate_band_sky(
        tuple(100.0 + 10.0 * i for i in range(len(beams))),
        tuple(beams),
        spectra=CMBSpectra(),
        r_in=r_in,
        nside=nside,
        lmax=lmax,
        fg_model=None,
        cmb_seed=seed,
    )


# --- needlet algebra --------------------------------------------------------


def test_cosine_needlet_partition_of_unity() -> None:
    lmax = 96
    h = np.asarray(cosine_needlet_bands(lmax, default_needlet_peaks(lmax)))
    np.testing.assert_allclose(np.sum(h**2, axis=0), 1.0, atol=1e-12)


def test_needlet_decompose_recompose_roundtrip() -> None:
    """Σ_j h_j·map2alm(synthesis(h_j·a)) ≈ a (partition of unity + map2alm)."""
    nside, lmax = 32, 48
    bands = cosine_needlet_bands(lmax, [8, 24, lmax])
    a = _rand_b_alm(0, lmax)
    beta = needlet_beta(a[None, :], bands, lmax=lmax, nside=nside)  # (J, 1, npix)
    npix = beta.shape[-1]
    weights = jnp.ones((bands.shape[0], 1, npix))
    rec = np.asarray(combine_needlets(weights, beta, bands, lmax=lmax, nside=nside, n_iter=5))
    relerr = np.max(np.abs(rec - np.asarray(a))) / np.max(np.abs(np.asarray(a)))
    assert relerr < 1e-3


# --- ILC constraint and CMB transfer ---------------------------------------


@pytest.mark.parametrize("localization", [None, 600.0])
def test_ilc_weights_sum_to_one(localization) -> None:
    """aᵀw_j = 1 exactly, for both the global and localized covariance paths."""
    nside, lmax = 32, 48
    beams = [30.0, 12.0]
    sky = _cmb_sky(beams, r_in=0.05, nside=nside, lmax=lmax)
    maps = assemble_band_maps(
        sky, jnp.array([1.0e-4, 1.0e-4]), jnp.ones(sky.npix), noise_key=jax.random.PRNGKey(1)
    )
    res = nilc_clean(
        maps,
        beams,
        lmax=lmax,
        nside=nside,
        needlet_peaks=[8, 24, lmax],
        localization_fwhm_arcmin=localization,
    )
    wsum = np.asarray(jnp.sum(res.weights, axis=1))  # (J,) global / (J, npix) localized
    np.testing.assert_allclose(wsum, 1.0, atol=1e-8)


def test_cmb_transfer_is_unity() -> None:
    """Identical CMB across bands → cleaned B equals the common-resolution CMB B."""
    nside, lmax = 32, 48
    beams = [30.0, 12.0]
    sky = _cmb_sky(beams, r_in=0.05, nside=nside, lmax=lmax)
    maps = sky.cmb_qu  # noiseless, FG-free
    res = nilc_clean(maps, beams, lmax=lmax, nside=nside, needlet_peaks=[8, 24, lmax], n_iter=5)
    ref, _ = common_resolution_b_alm(maps, beams, lmax=lmax, nside=nside, n_iter=5)
    ref = ref[0]  # all bands identical at common resolution
    relerr = np.max(np.abs(np.asarray(res.cleaned_b_alm) - np.asarray(ref))) / np.max(
        np.abs(np.asarray(ref))
    )
    assert relerr < 2e-3


# --- the load-bearing resolution -> down-weighting coupling -----------------


def test_coarse_band_downweighted_at_fine_scales() -> None:
    """A coarse low-freq beam is deconvolved -> its noise inflates -> the ILC
    down-weights it in the finest needlet band."""
    nside, lmax = 64, 128
    beams = [120.0, 10.0]  # band 0 very coarse, band 1 fine (common res = 10')
    sky = generate_band_sky(
        (30.0, 150.0),
        tuple(beams),
        spectra=CMBSpectra(),
        r_in=0.0,
        nside=nside,
        lmax=lmax,
        fg_model=None,
        cmb_seed=0,
    )
    maps = assemble_band_maps(
        sky, jnp.array([1.0e-3, 1.0e-3]), jnp.ones(sky.npix), noise_key=jax.random.PRNGKey(0)
    )
    res = nilc_clean(maps, beams, lmax=lmax, nside=nside)
    w_last = np.asarray(res.weights[-1])  # finest needlet band per-channel weights (global → (n_band,))
    assert w_last[0] < w_last[1]  # coarse band carries less weight than the fine band
    assert w_last[0] < 0.25


# --- differentiability through the full cleaner -----------------------------


def test_nilc_cleaned_power_differentiable_in_noise() -> None:
    nside, lmax = 32, 48
    beams = [40.0, 12.0]
    sky = generate_band_sky(
        (30.0, 150.0),
        tuple(beams),
        spectra=CMBSpectra(),
        r_in=0.01,
        nside=nside,
        lmax=lmax,
        fg_model=None,
        cmb_seed=2,
    )
    hit = jnp.ones(sky.npix)
    key = jax.random.PRNGKey(3)
    w0 = jnp.array([5.0e-4, 5.0e-4])

    def cleaned_power(scale):
        maps = assemble_band_maps(sky, scale * w0, hit, noise_key=key)
        res = nilc_clean(maps, beams, lmax=lmax, nside=nside, needlet_peaks=[8, 24, lmax])
        return jnp.sum(jnp.abs(res.cleaned_b_alm) ** 2)

    g = float(jax.grad(cleaned_power)(1.0))
    assert np.isfinite(g)
    eps = 1e-3
    fd = float((cleaned_power(1.0 + eps) - cleaned_power(1.0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=2e-3, atol=1e-12)


# --- band-limiting (scale-dependent channel inclusion) ---------------------


def test_needlet_channel_mask_excludes_coarse_at_fine_bands() -> None:
    """A coarse beam joins low-ℓ needlet bands but is excluded from fine ones."""
    lmax = 128
    peaks = [8, 32, lmax]
    nb = cosine_needlet_bands(lmax, peaks)
    beams = [200.0, 5.0]  # extreme ratio: 200' cannot be deconvolved to 5' at ℓ=128
    mask = _needlet_channel_mask(nb, beams, min(beams), lmax, threshold=0.1)

    assert mask.shape == (len(peaks), 2)
    assert mask[:, 1].all()  # finest channel (= common beam) active in every band
    assert mask[0, 0]  # coarse channel resolves the low-ℓ band
    assert not mask[-1, 0]  # ... but is excluded from the finest band


def _rand_spd(seed, n, n_samp=200):
    """Random SPD covariance (n, n)."""
    x = np.random.default_rng(seed).standard_normal((n, n_samp))
    return jnp.asarray((x @ x.T) / n_samp)


def test_ilc_weights_masked_all_active_byte_identical():
    """All-active mask reproduces the plain ridge+ILC solve byte-for-byte."""
    cov = _rand_spd(0, 5)
    m = jnp.ones(5)
    w_masked = _ilc_weights_masked(cov, m, 1e-10)
    w_plain = _ilc_weights_from_cov(_ridge(cov, 1e-10))
    np.testing.assert_array_equal(np.asarray(w_masked), np.asarray(w_plain))


def test_ilc_weights_masked_matches_active_only_gather():
    """Forced-exclusion: the masked solve == the explicit active-channel gather to fp64,
    with inactive weights exactly 0 and the CMB constraint Σ_active w = 1 preserved."""
    cov = _rand_spd(1, 6)
    ridge = 1e-10
    m = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    idx = np.array([0, 2, 3, 5])

    w = _ilc_weights_masked(cov, m, ridge)

    # reference: solve over the active sub-block only, then scatter back
    cov_aa = cov[np.ix_(idx, idx)]
    w_active = _ilc_weights_from_cov(_ridge(cov_aa, ridge))
    w_ref = np.zeros(6)
    w_ref[idx] = np.asarray(w_active)

    np.testing.assert_allclose(np.asarray(w), w_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(w)[[1, 4]], 0.0)  # inactive → exactly 0
    assert abs(float(jnp.sum(m * w)) - 1.0) < 1e-12  # CMB preserved over active


def test_ilc_weights_masked_batched_pixel():
    """The masked solve is batch-safe over a leading pixel axis (localized path)."""
    rng = np.random.default_rng(2)
    npix, n_band = 5, 4
    x = rng.standard_normal((npix, n_band, 50))
    cov = jnp.asarray(np.einsum("pbi,pci->pbc", x, x) / 50.0)
    m = jnp.broadcast_to(jnp.array([1.0, 1.0, 0.0, 1.0]), (npix, n_band))
    w = _ilc_weights_masked(cov, m, 1e-10)
    assert w.shape == (npix, n_band)
    np.testing.assert_array_equal(np.asarray(w)[:, 2], 0.0)  # inactive band → 0 every pixel
    np.testing.assert_allclose(np.asarray(jnp.sum(m * w, axis=-1)), 1.0, rtol=1e-10)


# --- spin-2 Q/U cleaner (clean_e=True) -------------------------------------


def _eb_sky(beams, *, nside, lmax, seed=0):
    """CMB E+B BandSky + white noise through the given beams (non-degenerate E cov)."""
    from augr.delensing import load_lensing_spectra

    cl_ee = jnp.clip(load_lensing_spectra().cl_ee_len[: lmax + 1], 0.0, None)
    sky = generate_band_sky(
        tuple(100.0 + 10.0 * i for i in range(len(beams))),
        tuple(beams),
        spectra=CMBSpectra(),
        r_in=0.05,
        nside=nside,
        lmax=lmax,
        fg_model=None,
        cmb_seed=seed,
        cl_ee=cl_ee,
    )
    return assemble_band_maps(
        sky, jnp.array([1.0e-4] * len(beams)), jnp.ones(sky.npix), noise_key=jax.random.PRNGKey(seed)
    )


def test_clean_e_default_off_and_b_leg_byte_identical() -> None:
    """clean_e defaults off (no E products); turning it on does not move the B leg."""
    nside, lmax, peaks = 32, 48, [8, 24, 48]
    beams = [30.0, 12.0]
    maps = _eb_sky(beams, nside=nside, lmax=lmax)
    r0 = nilc_clean(maps, beams, lmax=lmax, nside=nside, needlet_peaks=peaks)
    assert r0.cleaned_e_alm is None and r0.weights_e is None
    r1 = nilc_clean(maps, beams, lmax=lmax, nside=nside, needlet_peaks=peaks, clean_e=True)
    assert r1.cleaned_e_alm is not None and r1.weights_e is not None
    np.testing.assert_array_equal(
        np.asarray(r1.cleaned_b_alm), np.asarray(r0.cleaned_b_alm)
    )  # B leg unchanged by clean_e
    np.testing.assert_array_equal(np.asarray(r1.weights), np.asarray(r0.weights))


def test_cleaned_qu_roundtrip() -> None:
    """map2alm(spin=2) of cleaned_qu recovers the stored cleaned E/B alm."""
    nside, lmax, peaks = 32, 48, [8, 24, 48]
    beams = [30.0, 12.0]
    res = nilc_clean(_eb_sky(beams, nside=nside, lmax=lmax), beams, lmax=lmax, nside=nside,
                     needlet_peaks=peaks, clean_e=True)
    qu = res.cleaned_qu()
    assert qu.shape == (2, 12 * nside * nside)
    eb = map2alm(qu, 2, lmax, nside, 5)
    for rec, ref in ((eb[0], res.cleaned_e_alm), (eb[1], res.cleaned_b_alm)):
        relerr = float(np.linalg.norm(np.asarray(rec - ref)) / np.linalg.norm(np.asarray(ref)))
        assert relerr < 1e-3


def test_e_ilc_weights_sum_to_one() -> None:
    """The independent E-mode ILC weights satisfy the CMB constraint aᵀw_E = 1."""
    nside, lmax, peaks = 32, 48, [8, 24, 48]
    beams = [30.0, 12.0]
    res = nilc_clean(_eb_sky(beams, nside=nside, lmax=lmax), beams, lmax=lmax, nside=nside,
                     needlet_peaks=peaks, clean_e=True)
    np.testing.assert_allclose(np.asarray(jnp.sum(res.weights_e, axis=1)), 1.0, atol=1e-8)


def test_cleaned_qu_and_project_e_raise_without_clean_e() -> None:
    nside, lmax = 32, 48
    beams = [30.0, 12.0]
    res = nilc_clean(_eb_sky(beams, nside=nside, lmax=lmax), beams, lmax=lmax, nside=nside)
    with pytest.raises(ValueError, match="clean_e"):
        res.cleaned_qu()
    with pytest.raises(ValueError, match="clean_e"):
        res.project_e(_eb_sky(beams, nside=nside, lmax=lmax))


def test_band_limit_keeps_weights_finite_at_extreme_beam_ratio() -> None:
    """Regression for the small-aperture / high-ℓ deconvolution blow-up.

    With a 200':5' beam pair the coarse channel would otherwise be deconvolved to
    astronomical noise at the finest needlet band, spiking cond(C) and the ILC
    weights. The band-limit mask excludes it there, so weights stay O(1) and the
    cleaned map stays finite.
    """
    nside, lmax = 64, 128
    beams = [200.0, 5.0]
    peaks = [8, 32, lmax]
    total = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (2, 2, 12 * nside * nside))
    res = nilc_clean(total, beams, lmax=lmax, nside=nside, needlet_peaks=peaks)

    W = np.asarray(res.weights)
    assert np.all(np.isfinite(W))
    assert np.max(np.abs(W)) < 5.0  # not 1/ridge ~ 1e10 from an ill-conditioned solve
    assert np.allclose(W[-1, 0], 0.0)  # coarse channel carries zero weight at finest band
    assert np.allclose(np.sum(W, axis=1), 1.0)  # ILC constraint aᵀw = 1 preserved
    assert np.all(np.isfinite(np.asarray(res.cleaned_b_alm)))
