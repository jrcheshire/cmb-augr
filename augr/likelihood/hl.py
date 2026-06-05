"""Hamimeche-Lewis bandpower log-likelihood.

The non-Gaussian (offset-lognormal-ish) bandpower likelihood of Hamimeche &
Lewis 2008 (arXiv:0801.0554). At low mode count the bandpower estimator is
χ²-skewed; the HL transform ``g(x) = sign(x-1)·sqrt(2(x - log x - 1))`` maps the
per-bin matrices so a single Gaussian quadratic form in the transformed residual
``X_g`` captures that skew. Sampling this likelihood (Phase B) recovers the
~few-% wider σ(r) that the Knox/Gaussian Fisher misses at the reionization bump.

The core (:func:`_eigh_sqrtm`, :func:`_eigh_inv_sqrtm`, :func:`_safe_g`,
:func:`_per_bin_xg`, :func:`hamimeche_lewis_likelihood`) is lifted from the
MATLAB-validated bk-jax implementation (``bk_jax.likelihood.hl``), de-coupled
from BK structures: it takes plain per-bin matrices + a dense covariance inverse
+ a :class:`~augr.likelihood.ordering.SpectrumLayout`, and flattens ``X_g`` in
augr's ``spec``-slowest order via :func:`~augr.likelihood.ordering.matrices_to_spectra`
(bk-jax used lag-major ``vecp``). Autodiff-safety patterns are preserved: the
``g(1)=0`` differentiate-through-``where`` pattern (finite cotangents at the
analytical zero) and a per-bin eigenvalue floor (real, finite likelihood on
sampler boundaries).

Architectural pin (from bk-jax): the **full** ``M_f^{-1}`` is used, not a
per-bin block-diagonal approximation — off-block-diagonal bin coupling is real
once BPWFs overlap. ``HLLikelihood`` sources it from
``covariance.bandpower_covariance_full``.

**Known edge — Asimov degeneracy (relevant to a Phase-B sampler).** At the
*exact* Asimov fiducial the data equals the model, so every per-bin
``M = R·Ĉ·R = I`` is the identity with fully-degenerate eigenvalues. ``eigh``'s
eigenvector gradient diverges as ``1/(λ_i − λ_j)`` there, so ``jax.grad`` of
``log_prob`` returns NaN exactly at the fiducial (and degrades for very small
displacements, where ``M`` is still near I). The likelihood *value* is correct
(peak = 0). A NUTS sampler started off the fiducial — by ~1σ in each free
parameter, or by jittering the Asimov data — sits comfortably in the regime
where the gradient is finite and accurate. This is the only known sharp edge in
the autodiff path.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from augr.likelihood.ordering import SpectrumLayout, matrices_to_spectra
from augr.likelihood.protocols import BinnedSpectra

# Eigenvalue floor relative to the largest eigenvalue per bin: below this an
# eigenvalue is treated as machine zero. 1e-12 is below fp64 condition for any
# well-posed C_l covariance but well above the eigh cancellation noise.
_EIG_FLOOR_REL = 1e-12


def _eigh_sqrtm(m: jax.Array) -> jax.Array:
    """``sqrtm(M)`` for symmetric PSD ``M`` via eigendecomposition (batched, differentiable)."""
    d, u = jnp.linalg.eigh(m)
    d_max = jnp.max(d, axis=-1, keepdims=True)
    floor = _EIG_FLOOR_REL * jnp.maximum(d_max, 1.0)
    d_clipped = jnp.maximum(d, floor)
    return (u * jnp.sqrt(d_clipped)[..., None, :]) @ jnp.swapaxes(u, -1, -2)


def _eigh_inv_sqrtm(m: jax.Array) -> jax.Array:
    """``inv(sqrtm(M))`` for symmetric PSD ``M``; the eigenvalue floor guards 1/0 and NaNs."""
    d, u = jnp.linalg.eigh(m)
    d_max = jnp.max(d, axis=-1, keepdims=True)
    floor = _EIG_FLOOR_REL * jnp.maximum(d_max, 1.0)
    d_clipped = jnp.maximum(d, floor)
    inv_sqrt_d = 1.0 / jnp.sqrt(d_clipped)
    return (u * inv_sqrt_d[..., None, :]) @ jnp.swapaxes(u, -1, -2)


def _safe_g(d: jax.Array) -> jax.Array:
    """The H-L ``g(d) = sign(d-1)·sqrt(2(d - log d - 1))`` transform.

    ``g(1) = 0`` analytically. Two autodiff snags are handled: ``log(d)`` for
    ``d <= 0`` (clip to a tiny positive floor) and ``sqrt(0)`` at ``d == 1``
    (gradient ``+inf``, ``0·inf = NaN``) — the argument is replaced by ``1.0``
    inside ``where`` *before* the sqrt, then the output masked back to ``0``, so
    ``jax.grad`` returns finite cotangents at the zero.
    """
    d_pos = jnp.maximum(d, _EIG_FLOOR_REL)
    arg = 2.0 * (d_pos - jnp.log(d_pos) - 1.0)
    is_zero = arg <= 0
    arg_safe = jnp.where(is_zero, 1.0, arg)
    sqrt_term = jnp.where(is_zero, 0.0, jnp.sqrt(arg_safe))
    return jnp.sign(d_pos - 1.0) * sqrt_term


def _per_bin_xg(c_l_b: jax.Array, c_l_hat_b: jax.Array, c_fl_12_b: jax.Array) -> jax.Array:
    """Per-bin ``X_b`` from the H-L recipe (one ``(M, M)`` bin).

    ``R = inv(sqrtm(C_l))``; ``[U, Λ] = eigh(R C_l_hat R)``;
    ``X = C_fl_12 · U · diag(g(Λ)) · U.T · C_fl_12``.
    """
    r = _eigh_inv_sqrtm(c_l_b)
    m = r @ c_l_hat_b @ r
    m_sym = 0.5 * (m + jnp.swapaxes(m, -1, -2))
    d, u = jnp.linalg.eigh(m_sym)
    g_d = _safe_g(d)
    udu = (u * g_d[..., None, :]) @ jnp.swapaxes(u, -1, -2)
    return c_fl_12_b @ udu @ c_fl_12_b


def _hl_xg_vector(
    c_fl_12: jax.Array,
    layout: SpectrumLayout,
    c_l_hat: jax.Array,
    c_l: jax.Array,
) -> jax.Array:
    """The flat H-L residual ``X_g`` ``(n_data,)`` in augr ``spec``-slowest order.

    ``c_l_hat`` / ``c_l`` / ``c_fl_12`` are ``(n_field, n_field, n_bins)``.
    """
    c_l_t = jnp.moveaxis(c_l, 2, 0)
    c_l_hat_t = jnp.moveaxis(c_l_hat, 2, 0)
    c_fl_12_t = jnp.moveaxis(c_fl_12, 2, 0)
    x_per_bin = jax.vmap(_per_bin_xg)(c_l_t, c_l_hat_t, c_fl_12_t)  # (n_bins, M, M)
    x_mats = jnp.moveaxis(x_per_bin, 0, -1)  # (M, M, n_bins)
    return matrices_to_spectra(x_mats, layout)


def hamimeche_lewis_likelihood(
    m_f_inv: jax.Array,
    c_fl_12: jax.Array,
    layout: SpectrumLayout,
    c_l_hat: jax.Array,
    c_l: jax.Array,
) -> jax.Array:
    """Scalar H-L log-likelihood ``-½ X_gᵀ M_f^{-1} X_g``.

    ``m_f_inv`` is the dense ``(n_data, n_data)`` fiducial bandpower-covariance
    inverse (ordering matching ``layout`` / ``SignalModel.data_vector``);
    ``c_fl_12[...,b]`` the per-bin ``sqrtm`` of the fiducial signal+noise
    covariance; ``c_l_hat`` / ``c_l`` the per-bin data / model signal+noise
    matrices ``(n_field, n_field, n_bins)``.
    """
    x_g = _hl_xg_vector(c_fl_12, layout, c_l_hat, c_l)
    return -0.5 * x_g @ m_f_inv @ x_g


def _dense_cov_inv(cov: jax.Array) -> jax.Array:
    """Symmetrised dense inverse, computed on host (numpy) then moved to device.

    Cross-vendor BLAS ``inv`` leaks asymmetry at ~1e-7; symmetrise so the
    downstream quadratic forms are platform-stable.
    """
    inv_np = np.linalg.inv(np.asarray(cov))
    return jnp.asarray(0.5 * (inv_np + inv_np.T))


class HLLikelihood(eqx.Module):
    """Hamimeche-Lewis :class:`~augr.likelihood.protocols.Likelihood` over augr bandpowers.

    Built once at the fiducial via :meth:`from_forecast`, which sources the
    dense ``M_f`` from ``covariance.bandpower_covariance_full`` and the per-bin
    fiducial signal+noise matrices from ``covariance._build_M`` /
    ``_build_M_signal``. The noise is held at its fiducial value and added to the
    parameter-dependent model signal each call (mirroring the MATLAB
    ``like_hl(model, data, noise_bias, prep)`` convention).

    For a *forecast* the "data" is Asimov — the model at the fiducial — so
    ``log_prob`` peaks (=0) at the fiducial parameters.
    """

    m_f_inv: jax.Array  # (n_data, n_data)
    c_fl_12: jax.Array  # (n_field, n_field, n_bins)
    data_matrices: jax.Array  # (n_field, n_field, n_bins) Ĉ = fiducial S+N (Asimov)
    noise_matrices: jax.Array  # (n_field, n_field, n_bins) fiducial N (added to model)
    layout: SpectrumLayout = eqx.field(static=True)

    def _model_matrices(self, prediction: BinnedSpectra) -> jax.Array:
        return prediction.as_bin_matrices() + self.noise_matrices

    def residual_vector(self, prediction: BinnedSpectra) -> jax.Array:
        """The flat H-L residual ``X_g`` for this prediction (=0 at the fiducial)."""
        return _hl_xg_vector(
            self.c_fl_12, self.layout, self.data_matrices, self._model_matrices(prediction)
        )

    def log_prob(self, prediction: BinnedSpectra) -> jax.Array:
        x_g = self.residual_vector(prediction)
        return -0.5 * x_g @ self.m_f_inv @ x_g

    @classmethod
    def from_forecast(
        cls,
        signal_model,
        instrument,
        fiducial_params: jax.Array,
    ) -> HLLikelihood:
        """Build the Asimov HL likelihood at ``fiducial_params`` (flat, in ``parameter_names`` order)."""
        # Private covariance builders are the plan-sanctioned reuse points for
        # the per-bin S+N matrices and the dense M_f (see plan "Reuse points").
        from augr.covariance import (
            _build_M,
            _build_M_signal,
            bandpower_covariance_full,
        )

        fid = jnp.asarray(fiducial_params)
        m_signal = _build_M_signal(signal_model, fid)  # (M, M, n_bins) signal-only S
        m_full = _build_M(signal_model, instrument, fid)  # (M, M, n_bins) S + N
        noise = m_full - m_signal
        c_fl_12 = jnp.moveaxis(_eigh_sqrtm(jnp.moveaxis(m_full, 2, 0)), 0, 2)
        m_f_inv = _dense_cov_inv(bandpower_covariance_full(signal_model, instrument, fid))
        layout = SpectrumLayout.from_freq_pairs(signal_model.freq_pairs, signal_model.n_bins)
        return cls(
            m_f_inv=m_f_inv,
            c_fl_12=c_fl_12,
            data_matrices=m_full,
            noise_matrices=noise,
            layout=layout,
        )
