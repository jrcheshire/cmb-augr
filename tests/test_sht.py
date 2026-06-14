"""Stage-0 gate for augr.sht: ducc0 HEALPix SHTs vs healpy + AD correctness.

Validates (a) spin-0 and spin-2 synthesis against healpy alm2map, (b) that
``adjoint_synthesis`` is the math adjoint of ``synthesis`` under the triangular
alm inner product, and (c) that the ``custom_vjp`` reverse-mode gradient matches
finite differences (exercising the m>0 ``2·conj`` convention).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ducc0")

import healpy as hp

from augr.sht import (
    _default_backend,
    _ell_of_alm,
    _m_of_alm,
    _resolve_nthreads,
    adjoint_synthesis,
    alm_size,
    band_limit,
    check_band_limit,
    get_sht_backend,
    map2alm,
    set_sht_backend,
    sht_backend,
    synthesis,
    synthesis_pol,
)

NSIDE = 16
LMAX = 2 * NSIDE
NLM = alm_size(LMAX)
NPIX = 12 * NSIDE * NSIDE


def _random_alm(seed: int, ncomp: int = 1) -> np.ndarray:
    """Random complex alm with imag(m=0)=0 (physical real-sky alms)."""
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((ncomp, NLM)) + 1j * rng.standard_normal((ncomp, NLM))).astype(
        np.complex128
    )
    m0 = _m_of_alm(LMAX) == 0
    a[:, m0] = a[:, m0].real
    return a


# --- forward synthesis vs healpy -------------------------------------------


def test_synthesis_spin0_matches_healpy() -> None:
    alm = _random_alm(seed=0, ncomp=1)
    out = np.asarray(synthesis(alm, 0, LMAX, NSIDE))[0]
    ref = hp.alm2map(alm[0].copy(), NSIDE, lmax=LMAX)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-10)


def test_synthesis_pol_matches_healpy() -> None:
    aT = _random_alm(seed=1)[0]
    aE = _random_alm(seed=2)[0]
    aB = _random_alm(seed=3)[0]
    T, Q, U = (np.asarray(x) for x in synthesis_pol(aT, aE, aB, lmax=LMAX, nside=NSIDE))
    T_ref, Q_ref, U_ref = hp.alm2map([aT.copy(), aE.copy(), aB.copy()], NSIDE, lmax=LMAX, pol=True)
    np.testing.assert_allclose(T, T_ref, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(Q, Q_ref, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(U, U_ref, rtol=1e-8, atol=1e-9)


# --- adjoint identity (raw kernels, no AD) ---------------------------------


def test_adjoint_synthesis_is_math_adjoint() -> None:
    """<Y a, m>_pix == <a, Yᵀ m>_alm with triangular inner product (m>0 doubled)."""
    alm = _random_alm(seed=4, ncomp=1)
    rng = np.random.default_rng(5)
    m = rng.standard_normal((1, NPIX)).astype(np.float64)

    lhs = float(np.sum(np.asarray(synthesis(alm, 0, LMAX, NSIDE)) * m))

    adj = np.asarray(adjoint_synthesis(m, 0, LMAX, NSIDE))
    m_is0 = _m_of_alm(LMAX) == 0
    w = np.where(m_is0, 1.0, 2.0)
    rhs = float(np.sum(w * np.real(np.conj(alm[0]) * adj[0])))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-10)


# --- custom_vjp gradient vs finite difference ------------------------------


def test_synthesis_vjp_matches_fd() -> None:
    """jax.grad through synthesis (linear) must equal the exact finite difference.

    Discriminating check on the m>0 2·conj VJP convention: loss(t) is quadratic
    in real t, so the central difference is exact and grad must match it.
    """
    alm0 = _random_alm(seed=6, ncomp=1)
    da = _random_alm(seed=7, ncomp=1)

    def loss_jax(t):
        m = synthesis(alm0 + t * da, 0, LMAX, NSIDE)
        return (m**2).sum()

    g = float(jax.grad(loss_jax)(0.0))
    eps = 1e-3
    fd = float((loss_jax(eps) - loss_jax(-eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_synthesis_jit_clean() -> None:
    """synthesis is jittable (pure_callback lowers cleanly)."""
    alm = _random_alm(seed=8, ncomp=1)
    out_eager = np.asarray(synthesis(alm, 0, LMAX, NSIDE))
    out_jit = np.asarray(jax.jit(lambda a: synthesis(a, 0, LMAX, NSIDE))(alm))
    np.testing.assert_allclose(out_jit, out_eager, rtol=1e-12, atol=1e-13)


# --- map2alm (Jacobi-iteration analysis) -----------------------------------


def _physical_alm(seed: int, ncomp: int, min_ell: int) -> np.ndarray:
    """Random alm with imag(m=0)=0 and modes below ``min_ell`` zeroed.

    Spin-s transforms ignore alm with ℓ < s, so those slots are unrecoverable;
    a band-limited *physical* field must have them zero for a round-trip test.
    """
    a = _random_alm(seed, ncomp=ncomp)
    a[:, _ell_of_alm(LMAX) < min_ell] = 0.0
    return a


def test_map2alm_spin0_matches_healpy() -> None:
    """Our Jacobi iteration reproduces healpy.map2alm(iter=3, use_weights=False)."""
    alm = _physical_alm(seed=10, ncomp=1, min_ell=0)
    m = synthesis(alm, 0, LMAX, NSIDE)
    rec = np.asarray(map2alm(m, 0, LMAX, NSIDE, n_iter=3))[0]
    ref = hp.map2alm(np.asarray(m)[0].copy(), lmax=LMAX, iter=3, use_weights=False)
    np.testing.assert_allclose(rec, ref, rtol=1e-10, atol=1e-12)


def test_map2alm_recovers_bandlimited_spin0() -> None:
    """map2alm inverts synthesis on a band-limited map (HEALPix quadrature)."""
    alm = _physical_alm(seed=11, ncomp=1, min_ell=0)
    m = synthesis(alm, 0, LMAX, NSIDE)
    rec = np.asarray(map2alm(m, 0, LMAX, NSIDE, n_iter=5))
    # iter=5 at lmax=2*nside: observed max-relerr ~8e-7.
    relerr = np.max(np.abs(rec - alm)) / np.max(np.abs(alm))
    assert relerr < 5e-6


def test_map2alm_recovers_bandlimited_spin2() -> None:
    """Spin-2 round-trip recovers E/B for a physical (ℓ>=2) band-limited field."""
    alm = _physical_alm(seed=12, ncomp=2, min_ell=2)
    qu = synthesis(alm, 2, LMAX, NSIDE)
    rec = np.asarray(map2alm(qu, 2, LMAX, NSIDE, n_iter=5))
    # iter=5 at lmax=2*nside: observed max-relerr ~4e-7.
    relerr = np.max(np.abs(rec - alm)) / np.max(np.abs(alm))
    assert relerr < 5e-6


def test_map2alm_vjp_matches_fd() -> None:
    """jax.grad through the iterative map2alm matches finite differences."""
    m0 = np.asarray(synthesis(_physical_alm(seed=13, ncomp=1, min_ell=0), 0, LMAX, NSIDE))
    dm = np.asarray(synthesis(_physical_alm(seed=14, ncomp=1, min_ell=0), 0, LMAX, NSIDE))

    def loss(t):
        alm = map2alm(jnp.asarray(m0) + t * jnp.asarray(dm), 0, LMAX, NSIDE, n_iter=3)
        return (jnp.abs(alm) ** 2).sum()

    g = float(jax.grad(loss)(0.0))
    eps = 1e-3
    fd = float((loss(eps) - loss(-eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


# --- band-limit ↔ nside coupling -------------------------------------------


def test_band_limit_default_is_conservative() -> None:
    """band_limit ties lmax to nside; default is the conservative 2·nside."""
    assert band_limit(128) == 256
    assert band_limit(256) == 512
    assert band_limit(128, factor=3.0) == 384  # opt into the looser grid limit


def test_check_band_limit_warns_past_grid_limit() -> None:
    """The soft guard fires only above the HEALPix grid limit 3·nside − 1."""
    import warnings as _w

    nside = 128
    # Conservative and grid-edge band limits are silent.
    with _w.catch_warnings():
        _w.simplefilter("error")
        check_band_limit(band_limit(nside), nside)  # 2·nside
        check_band_limit(3 * nside - 1, nside)  # exactly the grid limit
    # One mode past the grid limit warns.
    with pytest.warns(UserWarning, match="grid limit"):
        check_band_limit(3 * nside, nside)


# --- ducc thread-count resolution (process-pool oversubscription guard) ----


def test_resolve_nthreads_env_and_explicit(monkeypatch) -> None:
    """Explicit nthreads>0 wins; nthreads=0 falls back to AUGR_SHT_NTHREADS (default 0)."""
    monkeypatch.delenv("AUGR_SHT_NTHREADS", raising=False)
    assert _resolve_nthreads(0) == 0  # unset env → ducc all-threads
    assert _resolve_nthreads(4) == 4  # explicit wins even with env unset
    monkeypatch.setenv("AUGR_SHT_NTHREADS", "8")
    assert _resolve_nthreads(0) == 8  # env caps when caller left the default
    assert _resolve_nthreads(2) == 2  # explicit still wins over env


def test_synthesis_result_invariant_to_thread_count(monkeypatch) -> None:
    """AUGR_SHT_NTHREADS changes only the ducc thread count, not the result."""
    alm = _random_alm(seed=20, ncomp=1)
    monkeypatch.delenv("AUGR_SHT_NTHREADS", raising=False)
    out0 = np.asarray(synthesis(alm, 0, LMAX, NSIDE))
    monkeypatch.setenv("AUGR_SHT_NTHREADS", "1")
    out1 = np.asarray(synthesis(alm, 0, LMAX, NSIDE))
    np.testing.assert_allclose(out0, out1, rtol=1e-12, atol=1e-13)


# --- backend switch: jht vs ducc parity ------------------------------------

pytest.importorskip("jht")

# jht's band-limit ceiling is lmax <= 1.5*nside (it raises above), narrower than
# ducc's grid limit, so the parity config sits at the ceiling.
_NS_P = 64
_LMAX_P = 96  # = 1.5 * 64
_NLM_P = alm_size(_LMAX_P)


def _random_alm_p(seed: int, ncomp: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((ncomp, _NLM_P)) + 1j * rng.standard_normal((ncomp, _NLM_P))).astype(
        np.complex128
    )
    m0 = _m_of_alm(_LMAX_P) == 0
    a[:, m0] = a[:, m0].real
    return a


class TestBackendSelector:
    def test_default_matches_device(self) -> None:
        # Device-aware default: jht on GPU/TPU, ducc on CPU. The env var overrides.
        assert get_sht_backend() == _default_backend()
        if jax.default_backend() == "cpu":
            assert _default_backend() == "ducc"

    def test_context_manager_restores(self) -> None:
        start = get_sht_backend()
        other = "jht" if start == "ducc" else "ducc"
        with sht_backend(other):
            assert get_sht_backend() == other
        assert get_sht_backend() == start

    def test_invalid_backend_raises_and_leaves_state(self) -> None:
        start = get_sht_backend()
        with pytest.raises(ValueError, match="unknown SHT backend"):
            set_sht_backend("s2fft")
        assert get_sht_backend() == start  # failed set must not change the backend


class TestBackendParity:
    """The jht backend matches ducc to fp64 — the swap is numerically exact.

    Validated to ~1e-14 in a standalone probe; the gate here is rtol=1e-10.
    Convention parity (alm packing, spin-2 (E,B)->(Q,U) sign, normalization) is
    what these assert.
    """

    @staticmethod
    def _both(fn):
        with sht_backend("ducc"):
            d = np.asarray(fn())
        with sht_backend("jht"):
            j = np.asarray(fn())
        return d, j

    def test_synthesis_spin0(self) -> None:
        alm = jnp.asarray(_random_alm_p(seed=0, ncomp=1))
        d, j = self._both(lambda: synthesis(alm, 0, _LMAX_P, _NS_P))
        np.testing.assert_allclose(j, d, rtol=1e-10, atol=1e-12)

    def test_synthesis_spin2(self) -> None:
        alm = jnp.asarray(_random_alm_p(seed=1, ncomp=2))
        d, j = self._both(lambda: synthesis(alm, 2, _LMAX_P, _NS_P))
        # ducc0 (C++) vs jht (JAX recursion) agree to ~4e-12 absolute on maps of
        # magnitude ~50; per-element rtol blows up only at near-zero-crossing
        # pixels, so the absolute floor carries the gate. Still catches O(1)
        # convention bugs (wrong (E,B)->(Q,U) sign / normalization).
        np.testing.assert_allclose(j, d, rtol=1e-7, atol=1e-9)

    def test_adjoint_spin0(self) -> None:
        m = jnp.asarray(np.random.default_rng(5).standard_normal((1, 12 * _NS_P**2)))
        d, j = self._both(lambda: adjoint_synthesis(m, 0, _LMAX_P, _NS_P))
        np.testing.assert_allclose(j, d, rtol=1e-10, atol=1e-12)

    def test_adjoint_spin2(self) -> None:
        m = jnp.asarray(np.random.default_rng(6).standard_normal((2, 12 * _NS_P**2)))
        d, j = self._both(lambda: adjoint_synthesis(m, 2, _LMAX_P, _NS_P))
        np.testing.assert_allclose(j, d, rtol=1e-10, atol=1e-12)

    def test_map2alm_inherits_backend(self) -> None:
        """map2alm composes synthesis+adjoint, so both backends are bit-identical."""
        alm = jnp.asarray(_random_alm_p(seed=2, ncomp=2))

        def _recover(backend):
            with sht_backend(backend):
                mp = synthesis(alm, 2, _LMAX_P, _NS_P)
                return np.asarray(map2alm(mp, 2, _LMAX_P, _NS_P, n_iter=3))

        np.testing.assert_allclose(_recover("jht"), _recover("ducc"), rtol=1e-10, atol=1e-12)

    def test_grad_parity_through_real_roundtrip(self) -> None:
        """The pipeline-relevant gradient (through map->alm->map on a REAL map)
        matches between backends.

        A direct gradient w.r.t. a free complex alm differs between backends on
        ONE non-physical degree of freedom: the imaginary part of the m=0
        coefficient (zero by the real-sky reality constraint). ducc's custom_vjp
        projects it to zero; jht's native AD returns the ambient complex cotangent.
        The physical DOFs (m>0 full complex, m=0 real part) agree to ~1e-9. That
        m=0-imaginary DOF never appears in the map-based pipeline, where every alm
        is produced by map2alm from a real map (real m=0), so the real map ->
        alm -> map gradient the cleaners actually backprop through matches cleanly
        and sidesteps the convention entirely."""
        m = jnp.asarray(np.random.default_rng(8).standard_normal((2, 12 * _NS_P**2)))

        def scalar(x, backend):
            with sht_backend(backend):
                alm = map2alm(x, 2, _LMAX_P, _NS_P, n_iter=3)
                return jnp.sum(synthesis(alm, 2, _LMAX_P, _NS_P) ** 2)

        g_d = jax.grad(lambda x: scalar(x, "ducc"))(m)
        g_j = jax.grad(lambda x: scalar(x, "jht"))(m)
        np.testing.assert_allclose(np.asarray(g_j), np.asarray(g_d), rtol=1e-7, atol=1e-9)
