"""
signal.py — Total signal model and JAX-autodiff Jacobian.

Combines CMB BB spectra and foreground spectra into a data vector μ(θ)
of binned bandpowers across all unique frequency cross-spectra, and
computes ∂μ/∂θ exactly via jax.jacfwd.

Data vector ordering:
    [(ν₁×ν₁, b₁), ..., (ν₁×ν₁, bN), (ν₁×ν₂, b₁), ..., (νM×νM, bN)]

where (νᵢ×νⱼ) runs over unique pairs with i ≤ j, and bₖ runs over
bandpower bins.

Parameter vector ordering:
    [r, A_lens, <foreground parameters in model order>]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from augr.instrument import Instrument
from augr.foregrounds import ForegroundModel
from augr.spectra import CMBSpectra


# -----------------------------------------------------------------------
# Binning helpers (executed once at init, not during JAX tracing)
# -----------------------------------------------------------------------

def _make_bin_edges(ell_min: int,
                    ell_max: int,
                    ell_per_bin_below: int,
                    delta_ell: int,
                    ell_bins: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Generate (lo, hi) bin edge pairs.

    If ell_bins is provided, it is an array of bin edges and we use those.
    Otherwise: per-ℓ bins for [ell_min, ell_per_bin_below), then uniform
    Δℓ bins from ell_per_bin_below to ell_max.
    """
    if ell_bins is not None:
        edges = np.asarray(ell_bins, dtype=int)
        return [(int(edges[i]), int(edges[i + 1] - 1))
                for i in range(len(edges) - 1)]

    bins = []
    # Per-ℓ bins below threshold
    for ell in range(ell_min, min(ell_per_bin_below, ell_max + 1)):
        bins.append((ell, ell))
    # Uniform Δℓ bins above
    lo = max(ell_per_bin_below, ell_min)
    while lo <= ell_max:
        hi = min(lo + delta_ell - 1, ell_max)
        bins.append((lo, hi))
        lo = hi + 1
    return bins


def _build_bin_matrix(ells: np.ndarray,
                      bin_edges: list[tuple[int, int]],
                      window: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the binning weight matrix W and bin centers.

    W has shape (n_bins, n_ells). Each row is normalized to sum to 1.
    Multiplying W @ C_ℓ gives the binned bandpowers.

    Args:
        ells:      1-D integer array of all multipoles in the grid.
        bin_edges: List of (lo, hi) pairs from _make_bin_edges.
        window:    'tophat' or 'gaussian'.

    Returns:
        (W, bin_centers) as JAX arrays.
    """
    n_ells = len(ells)
    n_bins = len(bin_edges)
    W = np.zeros((n_bins, n_ells))
    centers = np.zeros(n_bins)

    for b, (lo, hi) in enumerate(bin_edges):
        center = (lo + hi) / 2.0
        centers[b] = center
        mask = (ells >= lo) & (ells <= hi)

        if window == "tophat":
            W[b, mask] = 1.0
        elif window == "gaussian":
            sigma = max((hi - lo) / 2.0, 0.5)
            W[b, mask] = np.exp(-(ells[mask] - center) ** 2
                                / (2.0 * sigma ** 2))
        else:
            raise ValueError(f"Unknown window type: {window!r}. "
                             f"Use 'tophat' or 'gaussian'.")

        row_sum = W[b].sum()
        if row_sum > 0:
            W[b] /= row_sum

    return jnp.array(W), jnp.array(centers)


# -----------------------------------------------------------------------
# Parameter flatten / unflatten
# -----------------------------------------------------------------------

def flatten_params(params_dict: dict[str, float],
                   names: list[str]) -> jnp.ndarray:
    """Convert a parameter dict to a flat JAX array in the given order.

    Missing keys raise KeyError so mismatches are caught early.
    """
    return jnp.array([params_dict[n] for n in names])


def unflatten_params(params_array: jnp.ndarray,
                     names: list[str]) -> dict[str, float]:
    """Convert a flat JAX array back to a parameter dict."""
    return {n: float(params_array[i]) for i, n in enumerate(names)}


# -----------------------------------------------------------------------
# Signal model
# -----------------------------------------------------------------------

class SignalModel:
    """Total signal model: CMB + foregrounds → binned bandpower data vector.

    The data vector μ(θ) stacks all unique (i ≤ j) cross-frequency spectra
    across all bandpower bins. It is differentiable in all parameters via
    jax.jacfwd.

    Args:
        instrument:        Instrument specification (defines channels, f_sky).
        foreground_model:  Any object satisfying the ForegroundModel Protocol.
        cmb_spectra:       CMBSpectra instance with loaded templates.
        ell_bins:          Optional array of bin edges (overrides auto-binning).
        ell_min:           Minimum multipole (default 2).
        ell_max:           Maximum multipole (default 300).
        delta_ell:         Bin width for uniform bins above ell_per_bin_below.
        ell_per_bin_below: Per-ℓ bins for ℓ < this value (default 30).
        window:            Bin window function: 'tophat' or 'gaussian'.
    """

    def __init__(self,
                 instrument: Instrument,
                 foreground_model: ForegroundModel,
                 cmb_spectra: CMBSpectra,
                 ell_bins: np.ndarray | None = None,
                 ell_min: int = 2,
                 ell_max: int = 300,
                 delta_ell: int = 35,
                 ell_per_bin_below: int = 30,
                 window: str = "tophat",
                 use_jit: bool = True):
        self._instrument = instrument
        self._fg_model = foreground_model
        self._cmb = cmb_spectra

        # Parameter names: [r, A_lens, <foreground params>]
        self._param_names = ["r", "A_lens"] + foreground_model.parameter_names

        # Frequency pairs: unique (i ≤ j), stored as (channel_index, channel_index)
        n_chan = len(instrument.channels)
        self._freq_pairs: list[tuple[int, int]] = [
            (i, j)
            for i in range(n_chan)
            for j in range(i, n_chan)
        ]
        # Store frequencies as plain Python floats (not traced by JAX)
        self._freqs: tuple[float, ...] = tuple(
            ch.nu_ghz for ch in instrument.channels
        )

        # Ell grid: all integer multipoles from ell_min to ell_max
        ells_np = np.arange(ell_min, ell_max + 1, dtype=float)
        self._ells = jnp.array(ells_np)

        # Binning
        bin_edges = _make_bin_edges(ell_min, ell_max,
                                    ell_per_bin_below, delta_ell, ell_bins)
        self._bin_matrix, self._bin_centers = _build_bin_matrix(
            ells_np, bin_edges, window)
        self._bin_edges = bin_edges

        # JIT-compiled Jacobian: traced once, cached for same-shape inputs.
        # data_vector itself stays un-JIT'd so jacfwd can trace through it.
        _jacfwd = jax.jacfwd(self.data_vector)
        self._jacobian_fn = jax.jit(_jacfwd) if use_jit else _jacfwd

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        """All model parameters in data-vector order."""
        return list(self._param_names)

    @property
    def n_params(self) -> int:
        return len(self._param_names)

    @property
    def n_bins(self) -> int:
        return len(self._bin_centers)

    @property
    def n_spectra(self) -> int:
        """Number of unique cross-frequency spectra (i ≤ j)."""
        return len(self._freq_pairs)

    @property
    def n_data(self) -> int:
        """Total length of the data vector: n_spectra × n_bins."""
        return self.n_spectra * self.n_bins

    @property
    def freq_pairs(self) -> list[tuple[int, int]]:
        """List of (i, j) channel index pairs with i ≤ j."""
        return list(self._freq_pairs)

    @property
    def bin_centers(self) -> jnp.ndarray:
        """1-D array of bin center multipoles."""
        return self._bin_centers

    @property
    def ells(self) -> jnp.ndarray:
        """Full ℓ grid used for spectrum evaluation."""
        return self._ells

    # ------------------------------------------------------------------
    # Data vector and Jacobian
    # ------------------------------------------------------------------

    def data_vector(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute the model bandpower data vector μ(θ).

        Args:
            params: Flat JAX array of length n_params.
                    Order: [r, A_lens, <foreground params>].

        Returns:
            Flat JAX array of length n_data = n_spectra × n_bins.
            Ordering: spectra vary slowest, bins vary fastest.
        """
        r = params[0]
        A_lens = params[1]
        fg_params = params[2:]

        # CMB BB spectrum on the full ℓ grid (same for all freq pairs)
        cl_cmb = self._cmb.cl_bb(self._ells, r, A_lens)

        # Build bandpowers for each frequency pair
        bandpowers = []
        for i_ch, j_ch in self._freq_pairs:
            nu_i = self._freqs[i_ch]
            nu_j = self._freqs[j_ch]
            cl_fg = self._fg_model.cl_bb(nu_i, nu_j, self._ells, fg_params)
            cl_total = cl_cmb + cl_fg
            bp = self._bin_matrix @ cl_total    # (n_bins,)
            bandpowers.append(bp)

        return jnp.concatenate(bandpowers)

    def jacobian(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute ∂μ/∂θ via JAX forward-mode automatic differentiation.

        Forward mode (jacfwd) is optimal here because n_params (~10)
        is much smaller than n_data (~hundreds).

        Args:
            params: Flat JAX array of length n_params.

        Returns:
            Jacobian array of shape (n_data, n_params).
        """
        return self._jacobian_fn(params)

    # ------------------------------------------------------------------
    # Convenience: indexing into the data vector
    # ------------------------------------------------------------------

    def spectrum_slice(self, i_ch: int, j_ch: int) -> slice:
        """Return the slice of the data vector for cross-spectrum (i, j).

        Useful for extracting a single cross-spectrum's bandpowers from
        the full data vector.
        """
        try:
            idx = self._freq_pairs.index((i_ch, j_ch))
        except ValueError:
            # Try swapped order
            idx = self._freq_pairs.index((j_ch, i_ch))
        return slice(idx * self.n_bins, (idx + 1) * self.n_bins)
