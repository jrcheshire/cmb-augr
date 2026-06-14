"""sht.py — differentiable HEALPix spherical-harmonic transforms (ducc0 + jht backends).

JAX-traceable, reverse-mode differentiable SHTs on the HEALPix RING grid for
spin-0 (T) and spin-2 (Q/U ↔ E/B), behind a public API (:func:`synthesis`,
:func:`adjoint_synthesis`, :func:`map2alm`, :func:`synthesis_pol`) that dispatches
to one of two backends.

Backends (ducc0 default, jht optional)
--------------------------------------
Selected by :func:`set_sht_backend` / the :func:`sht_backend` context manager /
the ``AUGR_SHT_BACKEND`` env var (default ``"ducc"``):

* **ducc0** (default) — ``ducc0.sht.experimental`` via ``jax.pure_callback`` +
  hand-written ``jax.custom_vjp``. Correct and fast on CPU up to the HEALPix grid
  limit (``lmax ≈ 3·nside − 1``); but the ``pure_callback`` is a host hop, so it
  does NOT run on a GPU — under ``jax.jit`` it round-trips device→host→device.
* **jht** (``pip install jaxht``) — native-JAX spin-0/2 SHTs (pure JAX, no C++),
  so every transform runs on CUDA with no code change and is differentiated by
  JAX directly (no ``custom_vjp``). Validated bit-for-bit against the ducc backend
  to fp64 (~1e-14) on synthesis / adjoint, spin-0 and spin-2. Use it for the GPU /
  end-to-end-differentiable map path. Its validated regime is ``lmax ≲ 1.5·nside``
  (it raises above that) — narrower than ducc, so ducc stays the default for
  high-band-limit forward production runs.

``s2fft`` (the other JAX-native candidate) has a structural spin-2 HEALPix
*inverse* defect as of v1.4.0, so it is not used; jht is the JAX-native backend.

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
from contextlib import contextmanager
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# backend imports (ducc0: [compsep] extra; jht: [masking] extra / main dep)
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


def _require_jht():
    """Import jht or raise a helpful error pointing at the jaxht dependency."""
    try:
        import jht

        return jht
    except ImportError as exc:  # pragma: no cover - exercised only without jht
        raise ImportError(
            "augr.sht jht backend requires 'jht' (PyPI distribution 'jaxht'). "
            "Install it with:\n"
            "    pip install 'cmb-augr[masking]'\n"
            "or, in the development env:\n"
            "    pixi add --pypi 'jaxht>=0.1.2'"
        ) from exc


# ---------------------------------------------------------------------------
# backend selection (ducc default, jht for the GPU / differentiable map path)
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ("ducc", "jht")
_BACKEND = os.environ.get("AUGR_SHT_BACKEND", "ducc").lower()
if _BACKEND not in _VALID_BACKENDS:  # pragma: no cover - guards a typo'd env var
    raise ValueError(
        f"AUGR_SHT_BACKEND={_BACKEND!r} is not a valid SHT backend; "
        f"expected one of {_VALID_BACKENDS}."
    )


def get_sht_backend() -> str:
    """Return the active SHT backend (``"ducc"`` or ``"jht"``)."""
    return _BACKEND


def set_sht_backend(name: str) -> None:
    """Set the active SHT backend process-wide (``"ducc"`` or ``"jht"``).

    The choice is static config, read by :func:`synthesis` / :func:`adjoint_synthesis`
    at trace time — it is not a traced value, so it is safe under ``jax.jit`` /
    ``jax.grad`` (the chosen branch is baked into the compiled graph). For scoped
    switching prefer the :func:`sht_backend` context manager.
    """
    global _BACKEND
    name = name.lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown SHT backend {name!r}; expected one of {_VALID_BACKENDS}."
        )
    _BACKEND = name


@contextmanager
def sht_backend(name: str):
    """Context manager: temporarily switch the SHT backend, then restore it."""
    prev = get_sht_backend()
    set_sht_backend(name)
    try:
        yield
    finally:
        set_sht_backend(prev)


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
def _synthesis_ducc(alm: jax.Array, spin: int, lmax: int, nside: int, nthreads: int = 0) -> jax.Array:
    """ducc0-backend HEALPix synthesis ``alm → map`` (pure_callback + custom_vjp).

    The differentiable ducc primitive behind the public :func:`synthesis`
    dispatcher; see it for the array conventions.
    """
    return _synthesis_raw(alm, spin, lmax, nside, nthreads)


def _synthesis_fwd(alm, spin, lmax, nside, nthreads):
    return _synthesis_raw(alm, spin, lmax, nside, nthreads), None


def _synthesis_bwd(spin, lmax, nside, nthreads, _res, map_cot):
    raw = _adjoint_synthesis_raw(map_cot, spin, lmax, nside, nthreads)
    return (_jax_convention_vjp_synth(raw, lmax),)


_synthesis_ducc.defvjp(_synthesis_fwd, _synthesis_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _adjoint_synthesis_ducc(
    m: jax.Array, spin: int, lmax: int, nside: int, nthreads: int = 0
) -> jax.Array:
    """ducc0-backend adjoint synthesis ``Yᵀ m`` (NOT ``map2alm``), differentiable.

    The differentiable ducc primitive behind the public :func:`adjoint_synthesis`
    dispatcher; see it for the array conventions.
    """
    return _adjoint_synthesis_raw(m, spin, lmax, nside, nthreads)


def _adjoint_synthesis_fwd(m, spin, lmax, nside, nthreads):
    return _adjoint_synthesis_raw(m, spin, lmax, nside, nthreads), None


def _adjoint_synthesis_bwd(spin, lmax, nside, nthreads, _res, alm_cot):
    modified = _jax_convention_vjp_adjsynth(alm_cot, lmax)
    return (_synthesis_raw(modified, spin, lmax, nside, nthreads),)


_adjoint_synthesis_ducc.defvjp(_adjoint_synthesis_fwd, _adjoint_synthesis_bwd)


# ---------------------------------------------------------------------------
# jht backend (native-JAX; differentiated by JAX directly, no custom_vjp)
# ---------------------------------------------------------------------------


def _synthesis_jht(alm: jax.Array, spin: int, lmax: int, nside: int) -> jax.Array:
    """jht-backend synthesis ``alm → map``; matches the ducc primitive's shapes.

    jht uses bare ``(Nlm,)`` / ``(Npix,)`` arrays for spin 0; this wrapper adds /
    strips the leading size-1 axis so the public ``(ncomp, ...)`` contract holds.
    Spin 2 ``(2, Nlm) → (2, Npix)`` passes straight through. Convention parity
    with the ducc primitive is validated to fp64 in ``tests/test_sht.py``.

    Gradient convention note: jht is differentiated by JAX natively (no
    ``custom_vjp``), and its VJP returns the *ambient* complex cotangent — it does
    NOT project the m=0 coefficient onto the real axis the way the ducc primitive's
    hand-written VJP does. So ``jax.grad`` w.r.t. a *free* complex alm differs
    between backends on exactly one non-physical DOF: ``Im(alm[m=0])`` (zero by the
    real-sky reality constraint). All physical DOFs (m>0, and ``Re(alm[m=0])``)
    agree to fp64. This never bites the map-based pipeline, where every alm is
    produced by ``map2alm`` from a real map (real m=0); parameterize a free sky by
    real DOFs (jht's ``real_to_alm`` / ``alm_to_real``) if you need m=0-imaginary
    gradients to agree.
    """
    jht = _require_jht()
    if int(spin) == 0:
        return jht.synthesis(alm[0], nside=int(nside), lmax=int(lmax), spin=0)[None, :]
    return jht.synthesis(alm, nside=int(nside), lmax=int(lmax), spin=int(spin))


def _adjoint_synthesis_jht(m: jax.Array, spin: int, lmax: int, nside: int) -> jax.Array:
    """jht-backend adjoint synthesis ``Yᵀ m``; matches the ducc primitive's shapes."""
    jht = _require_jht()
    if int(spin) == 0:
        return jht.adjoint_synthesis(m[0], nside=int(nside), lmax=int(lmax), spin=0)[None, :]
    return jht.adjoint_synthesis(m, nside=int(nside), lmax=int(lmax), spin=int(spin))


# ---------------------------------------------------------------------------
# public dispatchers (route to the active backend)
# ---------------------------------------------------------------------------


def synthesis(alm: jax.Array, spin: int, lmax: int, nside: int, nthreads: int = 0) -> jax.Array:
    """HEALPix synthesis ``alm → map``, JAX-traceable and reverse-mode differentiable.

    Dispatches to the active backend (:func:`get_sht_backend`): ducc0
    (``pure_callback`` + ``custom_vjp``, CPU) or jht (native JAX, GPU-capable).

    Parameters
    ----------
    alm
        Complex alm, shape ``(ncomp, Nlm)`` with ``Nlm = alm_size(lmax)``.
        ``ncomp=1`` for ``spin=0``; ``ncomp=2`` (E-like, B-like) for ``spin>0``.
    spin, lmax, nside
        Spin of the transform; max multipole; HEALPix RING nside.
    nthreads
        ducc thread count (0 = all hardware threads); ignored by the jht backend
        (XLA manages threading).

    Returns
    -------
    Real map, shape ``(nmaps, Npix)`` with ``nmaps=1`` (spin 0) or ``2`` (spin>0).
    """
    if _BACKEND == "jht":
        return _synthesis_jht(alm, spin, lmax, nside)
    return _synthesis_ducc(alm, spin, lmax, nside, nthreads)


def adjoint_synthesis(
    m: jax.Array, spin: int, lmax: int, nside: int, nthreads: int = 0
) -> jax.Array:
    """Adjoint of HEALPix synthesis ``Yᵀ m`` (NOT ``map2alm``), differentiable.

    Dispatches to the active backend (:func:`get_sht_backend`).

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
    if _BACKEND == "jht":
        return _adjoint_synthesis_jht(m, spin, lmax, nside)
    return _adjoint_synthesis_ducc(m, spin, lmax, nside, nthreads)


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
