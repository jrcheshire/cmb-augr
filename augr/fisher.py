"""
fisher.py — Fisher matrix computation and parameter constraints.

    F_{αβ} = J^T Σ⁻¹ J  +  P

where J = ∂μ/∂θ (Jacobian from JAX autodiff), Σ is the bandpower covariance,
and P is the diagonal Gaussian prior matrix: P_{αα} = 1/σ_prior_α².

Fixed parameters are removed from the Fisher matrix entirely (their rows
and columns are dropped before inversion), so they do not contribute to
marginalization penalties.

Marginalized constraint:  σ_α = √(F⁻¹)_{αα}
Conditional constraint:   σ_α^{cond} = 1/√(F_{αα})
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from augr.signal import SignalModel, flatten_params
from augr.covariance import bandpower_covariance
from augr.instrument import Instrument


@jax.jit
def _fisher_cholesky(J: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Compute J^T Σ⁻¹ J via Cholesky solve.

    Fast and accurate when the covariance is well-conditioned.
    Returns NaN on the diagonal if Cholesky fails.
    """
    L = jnp.linalg.cholesky(cov)
    X = jax.scipy.linalg.solve_triangular(L, J, lower=True)
    return X.T @ X


@jax.jit
def _fisher_eigh(J: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Compute J^T Σ⁻¹ J via eigendecomposition (fallback).

    When the covariance is ill-conditioned or not quite positive-definite
    (e.g. deep high-frequency channels producing nearly degenerate cross-
    spectra), eigenvalues that are zero or negative are discarded and only
    the positive-definite subspace is used.  This is equivalent to saying
    "ignore data-space directions that carry no independent information."

    The Fisher matrix is:
        F = (U^T J)^T  diag(1/s_+)  (U^T J)
    where Σ = U diag(s) U^T and s_+ keeps only the positive eigenvalues.
    """
    s, U = jnp.linalg.eigh(cov)
    s_inv = jnp.where(s > 0.0, 1.0 / s, 0.0)
    UtJ = U.T @ J
    W = jnp.sqrt(s_inv)[:, None] * UtJ
    return W.T @ W


def _fisher_from_jacobian_and_cov(J: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Compute J^T Σ⁻¹ J, with automatic fallback for ill-conditioned Σ.

    Tries Cholesky first (faster, more accurate for well-conditioned
    problems).  If the result contains NaN — indicating the covariance
    was not positive-definite — falls back to an eigendecomposition that
    discards non-positive eigenvalues.
    """
    F = _fisher_cholesky(J, cov)
    if jnp.any(jnp.isnan(F)):
        F = _fisher_eigh(J, cov)
    return F


class FisherForecast:
    """Fisher information matrix and parameter constraints.

    Args:
        signal_model:   SignalModel defining the data vector and Jacobian.
        instrument:     Instrument specification (for noise in covariance).
        fiducial_params: Dict of all parameter fiducial values.
        priors:         Dict mapping parameter name → prior width σ.
                        Adds 1/σ² to the diagonal of F.
        fixed_params:   List of parameter names to hold fixed (not varied).
                        These are excluded from the Fisher matrix.
    """

    def __init__(self,
                 signal_model: SignalModel,
                 instrument: Instrument,
                 fiducial_params: dict[str, float],
                 priors: dict[str, float] | None = None,
                 fixed_params: list[str] | None = None):
        self._signal = signal_model
        self._instrument = instrument
        self._fiducial = dict(fiducial_params)
        self._priors = priors or {}
        self._fixed = set(fixed_params or [])

        # All parameter names (full set including fixed)
        self._all_names = signal_model.parameter_names

        # Free (varied) parameter names and their indices in the full array
        self._free_names = [n for n in self._all_names if n not in self._fixed]
        self._free_idx = jnp.array([self._all_names.index(n) for n in self._free_names])

        # Cached results (computed lazily)
        self._fisher_matrix: jnp.ndarray | None = None
        self._fisher_inverse: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def free_parameter_names(self) -> list[str]:
        """Names of free (varied) parameters in Fisher matrix order."""
        return list(self._free_names)

    @property
    def n_free(self) -> int:
        return len(self._free_names)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> jnp.ndarray:
        """Compute the Fisher information matrix for free parameters.

        Steps:
            1. Flatten fiducial params to JAX array.
            2. Compute bandpower covariance Σ at fiducial.
            3. Compute Jacobian J = ∂μ/∂θ via jacfwd (all params).
            4. Select columns of J for free parameters only.
            5. Solve Σ⁻¹ J via Cholesky for numerical stability.
            6. F = J_free^T Σ⁻¹ J_free + prior matrix.

        Returns:
            Fisher matrix of shape (n_free, n_free).
        """
        params = flatten_params(self._fiducial, self._all_names)

        # Bandpower covariance at fiducial
        cov = bandpower_covariance(self._signal, self._instrument, params)

        # Full Jacobian: (n_data, n_all_params)
        J_full = self._signal.jacobian(params)

        # Select free parameter columns
        J = J_full[:, self._free_idx]   # (n_data, n_free)

        # F = J^T Σ⁻¹ J via Cholesky (JIT-compiled)
        F = _fisher_from_jacobian_and_cov(J, cov)

        # Add Gaussian priors
        for name, sigma_prior in self._priors.items():
            if name in self._fixed:
                continue
            if name in self._free_names:
                idx = self._free_names.index(name)
                F = F.at[idx, idx].add(1.0 / sigma_prior**2)

        self._fisher_matrix = F
        self._fisher_inverse = None   # invalidate cached inverse
        return F

    @property
    def fisher_matrix(self) -> jnp.ndarray:
        """The Fisher matrix. Computed on first access."""
        if self._fisher_matrix is None:
            self.compute()
        return self._fisher_matrix

    @property
    def inverse(self) -> jnp.ndarray:
        """Parameter covariance matrix F⁻¹, shape (n_free, n_free)."""
        if self._fisher_inverse is None:
            self._fisher_inverse = jnp.linalg.inv(self.fisher_matrix)
        return self._fisher_inverse

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def sigma(self, param: str = "r") -> float:
        """Marginalized 1σ constraint: √(F⁻¹)_{αα}.

        This is the constraint on `param` after marginalizing over all
        other free parameters.
        """
        idx = self._free_names.index(param)
        return float(jnp.sqrt(self.inverse[idx, idx]))

    def sigma_conditional(self, param: str = "r") -> float:
        """Conditional 1σ constraint: 1/√(F_{αα}).

        This is the constraint on `param` with all other parameters
        held fixed at their fiducial values.
        """
        idx = self._free_names.index(param)
        return float(1.0 / jnp.sqrt(self.fisher_matrix[idx, idx]))

    def marginalized_2d(self, param_i: str, param_j: str) -> dict:
        """2D marginalized sub-covariance and error ellipse parameters.

        Returns dict with:
            'cov_2d':    2×2 marginalized covariance sub-matrix
            'sigma_i':   marginalized σ for param_i
            'sigma_j':   marginalized σ for param_j
            'rho':       correlation coefficient
            'angle_deg': orientation angle of the 1σ ellipse [degrees]
        """
        idx_i = self._free_names.index(param_i)
        idx_j = self._free_names.index(param_j)

        F_inv = self.inverse
        cov_2d = jnp.array([
            [F_inv[idx_i, idx_i], F_inv[idx_i, idx_j]],
            [F_inv[idx_j, idx_i], F_inv[idx_j, idx_j]],
        ])

        sigma_i = float(jnp.sqrt(cov_2d[0, 0]))
        sigma_j = float(jnp.sqrt(cov_2d[1, 1]))
        rho = float(cov_2d[0, 1] / (sigma_i * sigma_j))

        # Ellipse orientation: angle of semi-major axis
        diff = float(cov_2d[0, 0] - cov_2d[1, 1])
        angle = 0.5 * float(jnp.arctan2(2.0 * cov_2d[0, 1], diff))

        return {
            "cov_2d": cov_2d,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "rho": rho,
            "angle_deg": float(jnp.degrees(angle)),
        }
