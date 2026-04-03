"""
covariance.py — Bandpower covariance matrix via the Knox formula.

For cross-spectra (i,j) and (k,l) in bandpower bin b:

    Cov(Ĉ_b^{ij}, Ĉ_b^{kl}) = [M_b^{ik} M_b^{jl} + M_b^{il} M_b^{jk}] / ν_b

where M = S + N (signal + noise), and ν_b = f_sky × Σ_{ℓ in b} (2ℓ+1).

The covariance is block-diagonal across ℓ-bins — different bins are
uncorrelated under the Gaussian/Knox approximation.  We exploit this by
returning per-bin blocks (n_spec × n_spec) rather than the full
(n_data × n_data) matrix.  This is both faster and numerically stabler:
each small block has condition number ~ 10⁴-10⁶, while the assembled
matrix can exceed 10²⁰ for instruments with many frequency channels.

The covariance is evaluated once at the fiducial model and held fixed
during Fisher computation (we do NOT include the second-order Fisher
term Tr(Σ⁻¹ dΣ/dθ ...) — see CLAUDE.md for rationale).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from augr.instrument import Instrument, noise_nl
from augr.signal import SignalModel


def _nu_b(bin_edges: list[tuple[int, int]], f_sky: float) -> jnp.ndarray:
    """Effective number of modes per bin: ν_b = f_sky × Σ_{ℓ in b} (2ℓ+1).

    For a bin [lo, hi]: Σ(2ℓ+1) = (hi-lo+1)(lo+hi+1)  [arithmetic series].
    """
    nu = np.array([
        f_sky * (hi - lo + 1) * (lo + hi + 1)
        for lo, hi in bin_edges
    ])
    return jnp.array(nu)


def _build_M(signal_model: SignalModel,
             instrument: Instrument,
             fiducial_params: jnp.ndarray) -> jnp.ndarray:
    """Build total M = S + N matrix at fiducial, shape (n_chan, n_chan, n_bins).

    M is symmetric: M[i,j,b] = M[j,i,b].
    Diagonal (i==i) entries include noise; off-diagonal noise is zero.
    """
    n_chan = len(instrument.channels)
    n_bins = signal_model.n_bins
    ells = signal_model.ells
    W = signal_model._bin_matrix          # (n_bins, n_ells)

    cl_cmb = signal_model.cmb_bb_unbinned(fiducial_params)
    fg_params = signal_model.fg_params_from(fiducial_params)

    # Signal: M is symmetric, so compute only upper triangle (i ≤ j)
    M = jnp.zeros((n_chan, n_chan, n_bins))
    for i in range(n_chan):
        for j in range(i, n_chan):
            nu_i = float(instrument.channels[i].nu_ghz)
            nu_j = float(instrument.channels[j].nu_ghz)
            cl_fg = signal_model._fg_model.cl_bb(nu_i, nu_j, ells, fg_params)
            bp = W @ (cl_cmb + cl_fg)
            M = M.at[i, j, :].set(bp)
            if i != j:
                M = M.at[j, i, :].set(bp)

    # Add noise on diagonal
    for i in range(n_chan):
        nl_i = noise_nl(instrument.channels[i], ells,
                        instrument.mission_duration_years, instrument.f_sky)
        M = M.at[i, i, :].add(W @ nl_i)

    return M


def bandpower_covariance(signal_model: SignalModel,
                         instrument: Instrument,
                         fiducial_params: jnp.ndarray) -> jnp.ndarray:
    """Compute the Knox-formula bandpower covariance matrix.

    Args:
        signal_model:    Defines binning, freq pairs, and signal model.
        instrument:      Provides noise parameters.
        fiducial_params: Flat parameter array (same order as
                         signal_model.parameter_names) at which to
                         evaluate M = S + N.

    Returns:
        Symmetric positive-definite covariance matrix of shape
        (n_data, n_data), where n_data = n_spectra × n_bins.
        Block-diagonal over bins: Cov[s1*nb+b, s2*nb+b'] = 0 for b ≠ b'.
    """
    n_bins  = signal_model.n_bins
    pairs   = signal_model.freq_pairs       # list of (i, j) with i ≤ j
    n_spec  = len(pairs)

    # Total M = S + N at fiducial: (n_chan, n_chan, n_bins)
    M = _build_M(signal_model, instrument, fiducial_params)

    # Effective modes per bin
    nu = _nu_b(signal_model._bin_edges, instrument.f_sky)   # (n_bins,)

    # Vectorised Knox formula
    # For spectra s1=(i,j) and s2=(k,l):
    #   Cov[s1, s2, b] = (M[i,k,b]*M[j,l,b] + M[i,l,b]*M[j,k,b]) / nu[b]
    i_arr = jnp.array([p[0] for p in pairs])  # (n_spec,)
    j_arr = jnp.array([p[1] for p in pairs])

    # Each of the four M slices: shape (n_spec, n_spec, n_bins)
    M_ik = M[i_arr[:, None], i_arr[None, :], :]
    M_jl = M[j_arr[:, None], j_arr[None, :], :]
    M_il = M[i_arr[:, None], j_arr[None, :], :]
    M_jk = M[j_arr[:, None], i_arr[None, :], :]

    # cov_blocks[s1, s2, b] — shape (n_spec, n_spec, n_bins)
    cov_blocks = (M_ik * M_jl + M_il * M_jk) / nu[None, None, :]

    # Assemble block-diagonal full covariance (n_data, n_data)
    # cov_4d[s1, b1, s2, b2] = cov_blocks[s1, s2, b1] * delta(b1, b2)
    eye_b = jnp.eye(n_bins)                                 # (n_bins, n_bins)
    cov_4d = cov_blocks[:, :, :, None] * eye_b[None, None, :, :]
    # cov_4d has shape (n_spec, n_spec, n_bins, n_bins)
    # Reorder to (n_spec, n_bins, n_spec, n_bins) then flatten to (n_data, n_data)
    cov_4d = cov_4d.transpose(0, 2, 1, 3)
    return cov_4d.reshape(n_spec * n_bins, n_spec * n_bins)


def bandpower_covariance_blocks(signal_model: SignalModel,
                                instrument: Instrument,
                                fiducial_params: jnp.ndarray,
                                ) -> jnp.ndarray:
    """Compute per-bin Knox covariance blocks.

    Returns shape (n_bins, n_spec, n_spec) — one small covariance matrix
    per ell-bin.  Each block is well-conditioned (typ. condition number
    ~10^4-10^6) even when the full assembled matrix would have condition
    numbers > 10^20.

    This is the preferred interface for the Fisher computation, which
    can Cholesky-solve each block independently.
    """
    n_bins = signal_model.n_bins
    pairs  = signal_model.freq_pairs
    n_spec = len(pairs)

    M  = _build_M(signal_model, instrument, fiducial_params)
    nu = _nu_b(signal_model._bin_edges, instrument.f_sky)

    i_arr = jnp.array([p[0] for p in pairs])
    j_arr = jnp.array([p[1] for p in pairs])

    M_ik = M[i_arr[:, None], i_arr[None, :], :]
    M_jl = M[j_arr[:, None], j_arr[None, :], :]
    M_il = M[i_arr[:, None], j_arr[None, :], :]
    M_jk = M[j_arr[:, None], i_arr[None, :], :]

    # (n_spec, n_spec, n_bins)
    cov_blocks = (M_ik * M_jl + M_il * M_jk) / nu[None, None, :]

    # Transpose to (n_bins, n_spec, n_spec)
    return cov_blocks.transpose(2, 0, 1)
