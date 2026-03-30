"""
fisher.py — Fisher matrix computation and parameter constraints.

    F_{αβ} = Σ_b  J_b^T  Σ_b⁻¹  J_b   +   P

where J_b is the Jacobian slice for bin b, Σ_b is the (n_spec, n_spec)
Knox covariance block for that bin, and P is the diagonal Gaussian prior
matrix: P_{αα} = 1/σ_prior_α².

The covariance is block-diagonal across ℓ-bins (Knox approximation), so
rather than inverting the full (n_data, n_data) matrix, we Cholesky-solve
each small (n_spec, n_spec) block independently.  This is both faster
(O(n_bins × n_spec³) vs O(n_data³)) and perfectly conditioned even when
the full matrix would have condition numbers > 10²⁰.

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
from augr.covariance import bandpower_covariance_blocks
from augr.instrument import Instrument


@jax.jit
def _fisher_from_blocks(J_blocks: jnp.ndarray,
                        cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Compute F = sum_b J_b^T Sigma_b^{-1} J_b from per-bin blocks.

    Args:
        J_blocks:   (n_bins, n_spec, n_free) — Jacobian reshaped per bin.
        cov_blocks: (n_bins, n_spec, n_spec) — per-bin Knox covariance.

    Returns:
        Fisher matrix of shape (n_free, n_free).

    Each bin's covariance block is inverted via eigendecomposition,
    discarding non-positive eigenvalues. For well-conditioned blocks
    (most instruments) this is equivalent to Cholesky. For deep
    instruments with many channels, some blocks have numerically
    degenerate cross-spectra; the eigh approach handles these by
    projecting out the degenerate directions. Because each block is
    small (n_spec x n_spec, typically < 100), the eigendecomposition
    is both fast and accurate.
    """
    def _one_bin(carry, inputs):
        J_b, cov_b = inputs
        s, U = jnp.linalg.eigh(cov_b)
        s_inv = jnp.where(s > 0.0, 1.0 / s, 0.0)
        UtJ = U.T @ J_b
        W = jnp.sqrt(s_inv)[:, None] * UtJ
        return carry + W.T @ W, None

    n_free = J_blocks.shape[2]
    F, _ = jax.lax.scan(_one_bin, jnp.zeros((n_free, n_free)),
                         (J_blocks, cov_blocks))
    return F


class FisherForecast:
    """Fisher information matrix and parameter constraints.

    Args:
        signal_model:   SignalModel defining the data vector and Jacobian.
        instrument:     Instrument specification (for noise in covariance).
        fiducial_params: Dict of all parameter fiducial values.
        priors:         Dict mapping parameter name -> prior width sigma.
                        Adds 1/sigma^2 to the diagonal of F.
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

        self._all_names = signal_model.parameter_names
        self._free_names = [n for n in self._all_names
                            if n not in self._fixed]
        self._free_idx = jnp.array(
            [self._all_names.index(n) for n in self._free_names])

        self._fisher_matrix: jnp.ndarray | None = None
        self._fisher_inverse: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def free_parameter_names(self) -> list[str]:
        return list(self._free_names)

    @property
    def n_free(self) -> int:
        return len(self._free_names)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> jnp.ndarray:
        """Compute the Fisher information matrix for free parameters.

        Uses per-bin Cholesky solves on the block-diagonal covariance,
        avoiding the need to invert the full (n_data, n_data) matrix.

        Returns:
            Fisher matrix of shape (n_free, n_free).
        """
        params = flatten_params(self._fiducial, self._all_names)

        # Per-bin covariance blocks: (n_bins, n_spec, n_spec)
        cov_blocks = bandpower_covariance_blocks(
            self._signal, self._instrument, params)

        # Full Jacobian: (n_data, n_all_params) where n_data = n_spec * n_bins
        J_full = self._signal.jacobian(params)

        # Select free parameter columns
        J = J_full[:, self._free_idx]   # (n_spec * n_bins, n_free)

        # Reshape J into per-bin blocks: (n_bins, n_spec, n_free)
        n_spec = len(self._signal.freq_pairs)
        n_bins = self._signal.n_bins
        # Data ordering is (spec, bin): data[s * n_bins + b]
        # Reshape to (n_spec, n_bins, n_free) then transpose to
        # (n_bins, n_spec, n_free)
        J_blocks = J.reshape(n_spec, n_bins, -1).transpose(1, 0, 2)

        # F = sum_b J_b^T Sigma_b^{-1} J_b via per-bin Cholesky
        F = _fisher_from_blocks(J_blocks, cov_blocks)

        # Add Gaussian priors
        for name, sigma_prior in self._priors.items():
            if name in self._fixed:
                continue
            if name in self._free_names:
                idx = self._free_names.index(name)
                F = F.at[idx, idx].add(1.0 / sigma_prior**2)

        self._fisher_matrix = F
        self._fisher_inverse = None
        return F

    @property
    def fisher_matrix(self) -> jnp.ndarray:
        if self._fisher_matrix is None:
            self.compute()
        return self._fisher_matrix

    @property
    def inverse(self) -> jnp.ndarray:
        if self._fisher_inverse is None:
            self._fisher_inverse = jnp.linalg.inv(self.fisher_matrix)
        return self._fisher_inverse

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def sigma(self, param: str = "r") -> float:
        """Marginalized 1-sigma constraint: sqrt((F^-1)_{aa})."""
        idx = self._free_names.index(param)
        return float(jnp.sqrt(self.inverse[idx, idx]))

    def sigma_conditional(self, param: str = "r") -> float:
        """Conditional 1-sigma constraint: 1/sqrt(F_{aa})."""
        idx = self._free_names.index(param)
        return float(1.0 / jnp.sqrt(self.fisher_matrix[idx, idx]))

    def marginalized_2d(self, param_i: str, param_j: str) -> dict:
        """2D marginalized sub-covariance and error ellipse parameters."""
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

        diff = float(cov_2d[0, 0] - cov_2d[1, 1])
        angle = 0.5 * float(jnp.arctan2(2.0 * cov_2d[0, 1], diff))

        return {
            "cov_2d": cov_2d,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "rho": rho,
            "angle_deg": float(jnp.degrees(angle)),
        }
