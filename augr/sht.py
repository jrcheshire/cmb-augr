"""sht.py — differentiable HEALPix spherical-harmonic transforms (ducc0 backend).

Wraps ``ducc0.sht.experimental.synthesis`` / ``adjoint_synthesis`` via
``jax.pure_callback`` + ``jax.custom_vjp``, giving JAX-traceable, reverse-mode
differentiable SHTs on the HEALPix RING grid for spin-0 (T) and spin-2 (Q/U ↔ E/B).

Why ducc0 and not a JAX-native SHT
----------------------------------
``s2fft`` (the leading JAX-native candidate) has a structural defect in its
HEALPix spin-2 *inverse* transform as of v1.4.0 (the current PyPI release):
multi-mode polarization error of several percent and per-mode failures of
O(10–28%) for ``m < ℓ``. A B-mode component-separation pipeline lives or dies on
spin-2 accuracy, so we route through ducc0 (correct, fast) and accept the
``pure_callback`` boundary. ``jax-healpy`` delegates SHTs to ``s2fft`` and
inherits the same defect, so it is not an alternative.

Differentiation convention
---------------------------
``synthesis`` (alm → map) and ``adjoint_synthesis`` (map → alm) are a linear
transpose pair, so each one's reverse-mode VJP is a call to its partner — but
with two corrections for healpy *triangular* alm packing of a real sky
(see ``_jax_convention_vjp_*`` and the port source
``~/bicepkeck/bk-jax/src/bk_jax/sht/_primitives.py``):

1. ducc's ``adjoint_synthesis`` returns the Hermitian adjoint ``Yᴴ`` (not the
   strict transpose ``Yᵀ``), so the JAX-convention cotangent needs ``conj(·)``.
2. ducc synthesis at ``m > 0`` computes ``2·Re(alm·Y_lm)`` — the factor of 2
   implicitly accounts for the ``m < 0`` modes via the real-sky constraint — so
   ``synthesis`` is ℝ-linear in alm and the VJP carries a factor 2 on ``m > 0``.

Combined: VJP of synthesis = ``adjoint_synthesis(cot)`` with ``m=0`` unchanged
and ``m>0`` mapped to ``2·conj(·)``; VJP of adjoint_synthesis = ``synthesis(·)``
of a cotangent with ``m=0`` → real part and ``m>0`` → ``conj(·)/2``. These factors
are what make ``jax.grad`` agree with finite differences; they are validated by
the Stage-0 gate.

``adjoint_synthesis`` is the *mathematical adjoint* ``Yᵀ m``, NOT ``map2alm``
(which carries quadrature weights). On HEALPix isolatitudes the two are related
by ``adjoint = (Npix/4π)·map2alm(iter=0)``. The quadrature-accurate analysis
(map → alm) is provided by :func:`map2alm` below — a differentiable Jacobi
iteration on ``adjoint_synthesis`` (the same scheme healpy uses for its ``iter``
argument), reusable by the NILC and by external consumers.

Polarization output is HEALPix-internal Q, U (not IAU); flip U at the boundary
if an IAU consumer needs it.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# ducc0 lazy import (optional [compsep] dependency)
# ---------------------------------------------------------------------------


def _require_ducc():
    """Import ducc0 submodules or raise a helpful error pointing at the extra."""
    try:
        import ducc0
        import ducc0.healpix
        import ducc0.sht.experimental

        return ducc0
    except ImportError as exc:  # pragma: no cover - exercised only without ducc0
        raise ImportError(
            "augr.sht requires the 'ducc0' package, which ships with the "
            "component-separation extra. Install it with:\n"
            "    pip install 'cmb-augr[compsep]'\n"
            "or, in the development env:\n"
            "    pixi add ducc0"
        ) from exc


# ---------------------------------------------------------------------------
# alm storage helpers (healpy triangular packing, m-major)
# ---------------------------------------------------------------------------


def alm_size(lmax: int, mmax: int | None = None) -> int:
    """Number of complex alm coefficients in triangular storage."""
    if mmax is None:
        mmax = lmax
    return mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1


@lru_cache(maxsize=64)
def _ell_of_alm(lmax: int) -> np.ndarray:
    """alm-index → ℓ for healpy triangular packing (m outer, ℓ inner)."""
    ell = np.empty(alm_size(lmax), dtype=np.int64)
    idx = 0
    for m in range(lmax + 1):
        for ell_value in range(m, lmax + 1):
            ell[idx] = ell_value
            idx += 1
    return ell


@lru_cache(maxsize=64)
def _m_of_alm(lmax: int) -> np.ndarray:
    """alm-index → m for healpy triangular packing."""
    m_arr = np.empty(alm_size(lmax), dtype=np.int64)
    idx = 0
    for m in range(lmax + 1):
        for _ in range(m, lmax + 1):
            m_arr[idx] = m
            idx += 1
    return m_arr


@lru_cache(maxsize=64)
def _m_zero_mask(lmax: int) -> np.ndarray:
    """Boolean mask of shape (nlm,): True iff the alm index has m == 0."""
    return _m_of_alm(lmax) == 0


def almxfl(alm: jax.Array, fl: jax.Array, lmax: int) -> jax.Array:
    """Multiply ``alm[(ℓ,m)]`` by ``fl[ℓ]`` (healpy.almxfl), JAX-differentiable.

    Used to apply beam / needlet-band window functions ``B(ℓ)`` / ``h_j(ℓ)``
    on alms. Differentiable in both ``alm`` and ``fl``.
    """
    ell = jnp.asarray(_ell_of_alm(lmax))
    return alm * fl[ell]


# ---------------------------------------------------------------------------
# band-limit ↔ nside coupling (HEALPix grid policy)
# ---------------------------------------------------------------------------

# HEALPix has no exact quadrature; the synthesis/analysis pair degrades as the
# band limit approaches the formal grid limit ``lmax ≈ 3·nside − 1``. The map2alm
# Jacobi iteration here inherits that, so a map should be transformed only up to a
# band limit comfortably below the grid edge. ``2·nside`` is a deliberately
# conservative default (it keeps every mode well inside the reliable range, the
# convention favoured for B-mode work where transform error is unacceptable); the
# grid limit ``3·nside − 1`` is the hard ceiling beyond which transforms are simply
# wrong.
_CONSERVATIVE_LMAX_FACTOR = 2.0
_HARD_LMAX_FACTOR = 3.0


def band_limit(nside: int, factor: float = _CONSERVATIVE_LMAX_FACTOR) -> int:
    """Recommended SHT band limit ``lmax`` for a HEALPix ``nside``.

    Returns ``int(factor · nside)``. The default ``factor=2.0`` is the conservative
    coupling (modes well inside the reliable range); pass a larger ``factor`` to push
    toward the ``3·nside − 1`` grid limit at the cost of transform accuracy. This is
    the single knob that ties ``lmax`` to ``nside`` — derive ``lmax`` from ``nside``
    through here rather than choosing them independently.
    """
    return int(factor * int(nside))


def check_band_limit(lmax: int, nside: int, *, stacklevel: int = 2) -> None:
    """Warn if ``lmax`` exceeds the HEALPix grid limit ``3·nside − 1``.

    A soft guard for the top-level entry points: transforms past the grid limit are
    unreliable regardless of iteration count. Stays silent for the conservative
    ``lmax ≤ 2·nside`` regime :func:`band_limit` produces. Does not raise — callers
    that genuinely want the edge can ignore the warning.
    """
    hard = int(_HARD_LMAX_FACTOR * int(nside)) - 1
    if int(lmax) > hard:
        warnings.warn(
            f"lmax={int(lmax)} exceeds the HEALPix grid limit 3·nside−1={hard} "
            f"at nside={int(nside)}; spherical-harmonic transforms are unreliable "
            f"there. Use augr.sht.band_limit(nside) for a conservative lmax "
            f"(default 2·nside).",
            stacklevel=stacklevel + 1,
        )


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


def _real_dtype_for(complex_dtype) -> jnp.dtype:
    if complex_dtype == jnp.complex128:
        return jnp.dtype(jnp.float64)
    if complex_dtype == jnp.complex64:
        return jnp.dtype(jnp.float32)
    raise TypeError(f"alm dtype {complex_dtype} must be complex64 or complex128.")


def _complex_dtype_for(real_dtype) -> jnp.dtype:
    if real_dtype == jnp.float64:
        return jnp.dtype(jnp.complex128)
    if real_dtype == jnp.float32:
        return jnp.dtype(jnp.complex64)
    raise TypeError(f"map dtype {real_dtype} must be float32 or float64.")


# ---------------------------------------------------------------------------
# numpy kernels (run inside jax.pure_callback)
# ---------------------------------------------------------------------------


def _resolve_nthreads(nthreads: int) -> int:
    """ducc thread count for an SHT call.

    An explicit ``nthreads > 0`` always wins. ``nthreads == 0`` (the default
    everywhere in the NILC) falls back to the ``AUGR_SHT_NTHREADS`` env var, itself
    defaulting to 0 = ducc's all-hardware-threads. This lets a ``process_pool``
    worker cap ducc to ``cpu_count // n_workers`` threads via the inherited env so
    ``n_workers`` concurrent cells do not each grab all cores and oversubscribe.
    """
    if int(nthreads) > 0:
        return int(nthreads)
    return int(os.environ.get("AUGR_SHT_NTHREADS", "0"))


@lru_cache(maxsize=8)
def _sht_info(nside: int) -> dict:
    """HEALPix RING-scheme ring geometry for ducc0, cached per nside."""
    ducc0 = _require_ducc()
    return ducc0.healpix.Healpix_Base(int(nside), "RING").sht_info()


def _synthesis_np(
    alm: np.ndarray, *, spin: int, lmax: int, nside: int, nthreads: int
) -> np.ndarray:
    import ducc0.sht.experimental as ducc_sht

    return ducc_sht.synthesis(
        alm=np.asarray(alm),
        spin=int(spin),
        lmax=int(lmax),
        nthreads=_resolve_nthreads(nthreads),
        **_sht_info(nside),
    )


def _adjoint_synthesis_np(
    m: np.ndarray, *, spin: int, lmax: int, nside: int, nthreads: int
) -> np.ndarray:
    import ducc0.sht.experimental as ducc_sht

    # ducc0 may write into the input map; copy to keep the JAX side clean.
    return ducc_sht.adjoint_synthesis(
        map=np.array(m, copy=True),
        spin=int(spin),
        lmax=int(lmax),
        nthreads=_resolve_nthreads(nthreads),
        **_sht_info(nside),
    )


# ---------------------------------------------------------------------------
# raw pure_callback wrappers (no AD; the differentiable API wraps these)
# ---------------------------------------------------------------------------


def _synthesis_raw(alm: jax.Array, spin: int, lmax: int, nside: int, nthreads: int) -> jax.Array:
    npix = 12 * int(nside) ** 2
    nmaps = 1 if int(spin) == 0 else 2
    out = jax.ShapeDtypeStruct((nmaps, npix), _real_dtype_for(alm.dtype))
    return jax.pure_callback(
        partial(_synthesis_np, spin=spin, lmax=lmax, nside=nside, nthreads=nthreads),
        out,
        alm,
        vmap_method="sequential",
    )


def _adjoint_synthesis_raw(
    m: jax.Array, spin: int, lmax: int, nside: int, nthreads: int
) -> jax.Array:
    nlm = alm_size(int(lmax))
    ncomp = 1 if int(spin) == 0 else 2
    out = jax.ShapeDtypeStruct((ncomp, nlm), _complex_dtype_for(m.dtype))
    return jax.pure_callback(
        partial(_adjoint_synthesis_np, spin=spin, lmax=lmax, nside=nside, nthreads=nthreads),
        out,
        m,
        vmap_method="sequential",
    )


# ---------------------------------------------------------------------------
# JAX-convention VJP corrections (triangular real-sky packing)
# ---------------------------------------------------------------------------


def _jax_convention_vjp_synth(raw: jax.Array, lmax: int) -> jax.Array:
    """Convert ducc adjoint_synthesis(map_cot) into the JAX VJP of synthesis.

    m=0 unchanged (real); m>0 → ``2·conj(·)`` (ducc m>0 doubling + Hermitian→transpose).
    """
    is_m0 = jnp.asarray(_m_zero_mask(int(lmax)))
    return jnp.where(is_m0[None, :], raw, 2.0 * jnp.conj(raw))


def _jax_convention_vjp_adjsynth(alm_cot: jax.Array, lmax: int) -> jax.Array:
    """Build the synthesis input so the result is the JAX VJP of adjoint_synthesis.

    m=0 → real part (m=0 alms are real in triangular packing); m>0 → ``conj(·)/2``.
    """
    is_m0 = jnp.asarray(_m_zero_mask(int(lmax)))
    real_at_m0 = alm_cot.real.astype(alm_cot.dtype)
    return jnp.where(is_m0[None, :], real_at_m0, jnp.conj(alm_cot) / 2.0)


# ---------------------------------------------------------------------------
# public differentiable API
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def synthesis(alm: jax.Array, spin: int, lmax: int, nside: int, nthreads: int = 0) -> jax.Array:
    """HEALPix synthesis ``alm → map``, JAX-traceable and reverse-mode differentiable.

    Parameters
    ----------
    alm
        Complex alm, shape ``(ncomp, Nlm)`` with ``Nlm = alm_size(lmax)``.
        ``ncomp=1`` for ``spin=0``; ``ncomp=2`` (E-like, B-like) for ``spin>0``.
    spin, lmax, nside, nthreads
        Spin of the transform; max multipole; HEALPix RING nside; ducc thread
        count (0 = all hardware threads).

    Returns
    -------
    Real map, shape ``(nmaps, Npix)`` with ``nmaps=1`` (spin 0) or ``2`` (spin>0).
    """
    return _synthesis_raw(alm, spin, lmax, nside, nthreads)


def _synthesis_fwd(alm, spin, lmax, nside, nthreads):
    return _synthesis_raw(alm, spin, lmax, nside, nthreads), None


def _synthesis_bwd(spin, lmax, nside, nthreads, _res, map_cot):
    raw = _adjoint_synthesis_raw(map_cot, spin, lmax, nside, nthreads)
    return (_jax_convention_vjp_synth(raw, lmax),)


synthesis.defvjp(_synthesis_fwd, _synthesis_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def adjoint_synthesis(
    m: jax.Array, spin: int, lmax: int, nside: int, nthreads: int = 0
) -> jax.Array:
    """Adjoint of HEALPix synthesis ``Yᵀ m`` (NOT ``map2alm``), differentiable.

    Parameters
    ----------
    m
        Real map, shape ``(nmaps, Npix)`` (``nmaps=1`` spin 0, ``2`` spin>0).
    spin, lmax, nside, nthreads
        As in :func:`synthesis`.

    Returns
    -------
    Complex alm, shape ``(ncomp, Nlm)``.
    """
    return _adjoint_synthesis_raw(m, spin, lmax, nside, nthreads)


def _adjoint_synthesis_fwd(m, spin, lmax, nside, nthreads):
    return _adjoint_synthesis_raw(m, spin, lmax, nside, nthreads), None


def _adjoint_synthesis_bwd(spin, lmax, nside, nthreads, _res, alm_cot):
    modified = _jax_convention_vjp_adjsynth(alm_cot, lmax)
    return (_synthesis_raw(modified, spin, lmax, nside, nthreads),)


adjoint_synthesis.defvjp(_adjoint_synthesis_fwd, _adjoint_synthesis_bwd)


# ---------------------------------------------------------------------------
# quadrature analysis (map -> alm) via Jacobi iteration
# ---------------------------------------------------------------------------


def map2alm(
    m: jax.Array,
    spin: int,
    lmax: int,
    nside: int,
    n_iter: int = 3,
    nthreads: int = 0,
) -> jax.Array:
    """Quadrature-accurate HEALPix analysis ``map → alm``, differentiable.

    HEALPix has no exact quadrature, so this recovers the band-limited alm by
    Jacobi iteration on the adjoint transform — the same scheme healpy applies
    via its ``iter`` argument (with uniform pixel weights ``4π/N_pix``):

        ``a⁰ = (4π/N_pix)·Yᴴ m``  then  ``aᵏ⁺¹ = aᵏ + (4π/N_pix)·Yᴴ(m − Y aᵏ)``.

    For a band-limited map ``m = Y a`` with ``lmax ≲ 2·nside`` this converges to
    ``a`` in a few iterations. Because ``synthesis`` (``Y``) and
    ``adjoint_synthesis`` (``Yᴴ``) are both ``custom_vjp`` primitives and the
    iteration is a finite composition of them with linear ops, ``map2alm`` is
    fully reverse-mode differentiable.

    Parameters
    ----------
    m
        Real map, shape ``(nmaps, Npix)`` (``nmaps=1`` spin 0, ``2`` spin>0).
    spin, lmax, nside, nthreads
        As in :func:`synthesis`.
    n_iter
        Number of Jacobi refinement iterations (``n_iter=0`` returns the 0th-order
        ``(4π/N_pix)·adjoint_synthesis`` estimate). Default 3, matching healpy.

    Returns
    -------
    Complex alm, shape ``(ncomp, Nlm)`` (``ncomp=1`` spin 0, ``2`` spin>0).
    """
    npix = 12 * int(nside) ** 2
    scale = 4.0 * jnp.pi / npix
    alm = scale * adjoint_synthesis(m, spin, lmax, nside, nthreads)
    for _ in range(int(n_iter)):
        residual = m - synthesis(alm, spin, lmax, nside, nthreads)
        alm = alm + scale * adjoint_synthesis(residual, spin, lmax, nside, nthreads)
    return alm


# ---------------------------------------------------------------------------
# polarization convenience wrappers
# ---------------------------------------------------------------------------


def synthesis_pol(
    alm_T: jax.Array,
    alm_E: jax.Array,
    alm_B: jax.Array,
    *,
    lmax: int,
    nside: int,
    nthreads: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """T/E/B alms → T/Q/U HEALPix maps (HEALPix-internal Q/U convention).

    ``alm_T/E/B`` are 1-D complex arrays of shape ``(Nlm,)``; returns three 1-D
    real maps ``(T, Q, U)`` of shape ``(Npix,)``. Negate U afterwards for IAU.
    """
    T_map = synthesis(alm_T[None, :], 0, lmax, nside, nthreads)[0]
    QU = synthesis(jnp.stack([alm_E, alm_B], axis=0), 2, lmax, nside, nthreads)
    return T_map, QU[0], QU[1]
