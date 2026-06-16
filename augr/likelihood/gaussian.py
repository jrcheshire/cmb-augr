"""Gaussian (Knox) bandpower log-likelihood.

The baseline ``-½ (d - μ(θ))ᵀ C⁻¹ (d - μ(θ))`` with ``C`` the fixed fiducial
Knox covariance — exactly the likelihood whose curvature is augr's Fisher. It
is the parity reference for the sampler: sampling this reproduces the Fisher
σ(r), while sampling the Hamimeche-Lewis likelihood (:mod:`augr.likelihood.hl`)
recovers the wider non-Gaussian σ(r). For a forecast the "data" is Asimov (the
fiducial model), so ``log_prob`` peaks (=0) at the fiducial parameters.
"""

from __future__ import annotations

import equinox as eqx
import jax

from augr.likelihood.hl import _dense_cov_inv
from augr.likelihood.protocols import BinnedSpectra


class GaussianLikelihood(eqx.Module):
    """Knox/Gaussian :class:`~augr.likelihood.protocols.Likelihood` over augr bandpowers."""

    cov_inv: jax.Array  # (n_data, n_data) fiducial Knox covariance inverse
    data: jax.Array  # (n_data,) Asimov data vector (fiducial model)

    def residual_vector(self, prediction: BinnedSpectra) -> jax.Array:
        return prediction.as_vector() - self.data

    def log_prob(self, prediction: BinnedSpectra) -> jax.Array:
        r = self.residual_vector(prediction)
        return -0.5 * r @ self.cov_inv @ r

    @classmethod
    def from_forecast(
        cls,
        signal_model,
        instrument,
        fiducial_params: jax.Array,
    ) -> GaussianLikelihood:
        """Build the Asimov Gaussian likelihood at ``fiducial_params`` (flat, ``parameter_names`` order)."""
        import jax.numpy as jnp

        from augr.covariance import bandpower_covariance_full

        fid = jnp.asarray(fiducial_params)
        cov_inv = _dense_cov_inv(bandpower_covariance_full(signal_model, instrument, fid))
        data = signal_model.data_vector(fid)
        return cls(cov_inv=cov_inv, data=data)

    @classmethod
    def from_external(
        cls,
        signal_model,
        fiducial_params: jax.Array,
        covariance: jax.Array,
    ) -> GaussianLikelihood:
        """Build the Asimov Gaussian likelihood from a precomputed external covariance.

        Mirrors :meth:`from_forecast` but takes a precomputed ``(n_data, n_data)``
        bandpower covariance (e.g. the cut-sky masked-Wiener Monte-Carlo
        covariance ``augr.spectrum_stages.CutskyMC.covariance``) instead of
        building the analytic Knox covariance. The Gaussian mean is the Asimov
        signal ``signal_model.data_vector(fid)``; the post-separation noise lives
        in the covariance — matching the Gaussian-Fisher
        ``forecast_from_spectra(external_covariance=...)`` path, whose curvature
        this reproduces exactly (the parity check for the cut-sky bridge).
        """
        import jax.numpy as jnp

        fid = jnp.asarray(fiducial_params)
        cov_inv = _dense_cov_inv(jnp.asarray(covariance))
        data = signal_model.data_vector(fid)
        return cls(cov_inv=cov_inv, data=data)
