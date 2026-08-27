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
    _whiten,
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


# --- cmb-augr #50: per-channel ridge + prewhitened solve ---------------------
#
# Channel-space model of the O'Brient 24-band set (10-1000 GHz) at l~88: CMB (lensing
# BB, r=0) + rank-1 dust + rank-1 synchrotron in CMB thermodynamic units + a
# beam-deconvolved white-noise diagonal (4 K optics, N=100 detector scaling, 1.5 m
# aperture; values from the JPL study's loading model). The diagonal spans 4e10, which
# is the regime where the old arithmetic-mean-scaled ridge swamped the CMB-carrying
# channels. Pure linear algebra: it exercises exactly the solvers the cleaners call.

_NU_GHZ_24 = np.array([10, 20, 24, 28, 34, 41, 49, 58, 69, 83, 99, 118, 141, 169, 202,
                       241, 288, 344, 411, 491, 586, 701, 837, 1000], float)
_NOISE_24 = np.array([  # uK^2 sr, beam-deconvolved at l=88
    1.698e-05, 3.514e-07, 1.735e-07, 1.103e-07, 6.771e-08, 4.844e-08, 3.876e-08, 3.328e-08,
    3.064e-08, 2.985e-08, 3.153e-08, 3.474e-08, 4.266e-08, 5.648e-08, 8.252e-08, 1.370e-07,
    2.487e-07, 4.767e-07, 1.155e-06, 3.322e-06, 1.185e-05, 8.298e-05, 1.294e-03, 9.925e-02,
])
_PREF_88 = 2 * np.pi / (88.0 * 89.0)
_C_CMB, _D_DUST, _D_SYNC = 0.05 * _PREF_88, 5.0 * _PREF_88, 0.02 * _PREF_88  # D_l uK^2


def _rj_to_cmb(nu_ghz):
    x = 6.62607015e-34 * nu_ghz * 1e9 / (1.380649e-23 * 2.7255)
    return np.expm1(x) ** 2 / (x**2 * np.exp(x))


def _dust_sed(nu, beta=1.54, t_dust=20.0, nu0=353.0):
    """Modified blackbody in CMB units, unit at nu0."""
    h_k = 6.62607015e-34 * 1e9 / 1.380649e-23

    def bb(n):
        return n**3 / np.expm1(h_k * n / t_dust)

    rj = (nu / nu0) ** beta * bb(nu) / bb(nu0) * (nu0 / nu) ** 2
    return rj * _rj_to_cmb(nu) / _rj_to_cmb(nu0)


def _sync_sed(nu, beta=-3.0, nu0=23.0):
    return (nu / nu0) ** beta * _rj_to_cmb(nu) / _rj_to_cmb(nu0)


def _wide_band_cov(sel):
    """(C, C_fg + N) for the selected channels; residual variance is w^T (C_fg + N) w."""
    f, q, n = _dust_sed(_NU_GHZ_24)[sel], _sync_sed(_NU_GHZ_24)[sel], _NOISE_24[sel]
    fg_plus_n = _D_DUST * np.outer(f, f) + _D_SYNC * np.outer(q, q) + np.diag(n)
    return jnp.asarray(_C_CMB + fg_plus_n), fg_plus_n


def _legacy_ridge(cov, ridge):
    """The pre-#50 form: ``ridge * tr(C)/n * I`` -- the arithmetic-mean scaling."""
    n = cov.shape[-1]
    return cov + ridge * jnp.trace(cov) / n * jnp.eye(n)


def _resid_nk(w, fg_plus_n):
    w = np.asarray(w)
    return 1e3 * np.sqrt(w @ fg_plus_n @ w / _PREF_88)


def test_ridge_is_a_fraction_of_each_channels_own_variance():
    cov = _rand_spd(3, 5)
    r = 1e-3
    out = np.asarray(_ridge(cov, r))
    d = np.diag(np.asarray(cov))
    np.testing.assert_allclose(np.diag(out), d * (1 + r), rtol=1e-15)
    off = ~np.eye(5, dtype=bool)
    np.testing.assert_array_equal(out[off], np.asarray(cov)[off])


def test_whiten_unit_diagonal_and_roundtrip():
    cov, _ = _wide_band_cov(np.ones(24, bool))
    ct, d = _whiten(cov)
    np.testing.assert_allclose(np.diag(np.asarray(ct)), 1.0, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(ct * d[:, None] * d[None, :]), np.asarray(cov),
                               rtol=1e-15)
    # a zero-variance (dataless) channel maps to d=1 and stays finite
    z = cov.at[0, :].set(0.0).at[:, 0].set(0.0)
    ctz, dz = _whiten(z)
    assert float(dz[0]) == 1.0 and bool(jnp.all(jnp.isfinite(ctz)))


def test_prewhitened_solves_match_direct_at_moderate_dynamic_range():
    """At a diagonal spread fp64 handles unwhitened, whitening is a pure no-op: the plain,
    masked and (see test_cmilc) constrained solves agree with the direct solve to 1e-12."""
    cov = _rand_spd(4, 6)
    scale = jnp.asarray(10.0 ** np.linspace(-2, 2, 6))
    cov = cov * scale[:, None] * scale[None, :]  # spread 1e8 on the diagonal
    a = np.ones(6)
    direct = np.linalg.solve(np.asarray(cov), a)
    direct /= direct.sum()
    np.testing.assert_allclose(np.asarray(_ilc_weights_from_cov(cov)), direct, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(_ilc_weights_masked(cov, jnp.ones(6), 0.0)), direct,
                               rtol=1e-12)
    m = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    idx = np.array([0, 2, 3, 5])
    direct_m = np.linalg.solve(np.asarray(cov)[np.ix_(idx, idx)], np.ones(4))
    w_ref = np.zeros(6)
    w_ref[idx] = direct_m / direct_m.sum()
    np.testing.assert_allclose(np.asarray(_ilc_weights_masked(cov, m, 0.0)), w_ref,
                               rtol=1e-12, atol=1e-15)


def test_wide_band_ridge_default_is_on_the_plateau():
    """cmb-augr #50: at the 24-band set the library-default ridge (1e-10) must be
    indistinguishable from ridge=0. Measured 2026-08-27: identical to 4 decimals at
    1e-10 and 1e-6 (2.1751 nK); the old form cost 2.5x at 1e-10 and 54x at 1e-6."""
    cov, fg_n = _wide_band_cov(np.ones(24, bool))
    m = jnp.ones(24)
    r0 = _resid_nk(_ilc_weights_masked(cov, m, 0.0), fg_n)
    for ridge in (1e-18, 1e-10, 1e-6):
        r = _resid_nk(_ilc_weights_masked(cov, m, ridge), fg_n)
        assert r <= r0 * (1 + 1e-3), (ridge, r, r0)


def test_wide_band_adding_channels_never_hurts_and_the_legacy_ridge_did():
    """The violated invariant that exposed #50: a constrained MV estimator cannot get
    worse when channels are added. New form: 24 ch beats 18 ch (<= 400 GHz) at the
    default ridge; the retained legacy form is the negative control -- at 1e-10 it
    made 24 ch 2.5x WORSE than 18 ch (measured 5.36 vs 2.38 nK), so this test cannot
    silently stop discriminating."""
    sel24 = np.ones(24, bool)
    sel18 = _NU_GHZ_24 <= 400
    cov24, fg24 = _wide_band_cov(sel24)
    cov18, fg18 = _wide_band_cov(sel18)
    ridge = 1e-10
    new24 = _resid_nk(_ilc_weights_masked(cov24, jnp.ones(24), ridge), fg24)
    new18 = _resid_nk(_ilc_weights_masked(cov18, jnp.ones(int(sel18.sum())), ridge), fg18)
    assert new24 <= new18, (new24, new18)
    legacy24 = _resid_nk(_ilc_weights_from_cov(_legacy_ridge(cov24, ridge)), fg24)
    assert legacy24 > 2.0 * new24, (legacy24, new24)  # measured 2.46x
    assert legacy24 > new18  # the unphysical ordering the issue reported


def test_masked_solve_ridge_applies_to_active_diagonal_only():
    """Inactive channels decouple regardless of ridge; active weights match the
    active-only gather with the per-channel ridge."""
    cov = _rand_spd(5, 6)
    m = jnp.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    idx = np.array([0, 1, 3, 5])
    ridge = 1e-3  # large enough to matter
    w = _ilc_weights_masked(cov, m, ridge)
    w_act = _ilc_weights_from_cov(_ridge(cov[np.ix_(idx, idx)], ridge))
    w_ref = np.zeros(6)
    w_ref[idx] = np.asarray(w_act)
    np.testing.assert_allclose(np.asarray(w), w_ref, rtol=1e-12, atol=1e-15)
