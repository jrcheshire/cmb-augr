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

import jax.numpy as jnp
import numpy as np

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


def _build_M_signal_unbinned(signal_model: SignalModel,
                              fiducial_params: jnp.ndarray) -> jnp.ndarray:
    """Unbinned signal-only M_signal(ℓ), shape (n_chan, n_chan, n_ells).

    M_signal[i, j, ℓ] = C_ℓ^CMB + C_ℓ^{FG, ij} + δ_{ij} C_ℓ^{res}.
    The residual template A_res·T_res(ℓ) enters only on i==j auto-blocks,
    matching the post-CompSep convention that the residual lives in the
    single cleaned map.
    """
    n_chan = len(signal_model.frequencies)
    ells = signal_model.ells
    freqs = signal_model.frequencies
    fg_model = signal_model.foreground_model

    cl_cmb = signal_model.cmb_bb_unbinned(fiducial_params)
    cl_res = signal_model.residual_bb_unbinned(fiducial_params)
    fg_params = signal_model.fg_params_from(fiducial_params)

    M = jnp.zeros((n_chan, n_chan, ells.shape[0]))
    for i in range(n_chan):
        for j in range(i, n_chan):
            cl_fg = fg_model.cl_bb(freqs[i], freqs[j], ells, fg_params)
            cl_total = cl_cmb + cl_fg
            if i == j:
                cl_total = cl_total + cl_res
            M = M.at[i, j, :].set(cl_total)
            if i != j:
                M = M.at[j, i, :].set(cl_total)
    return M


def _build_M_signal(signal_model: SignalModel,
                    fiducial_params: jnp.ndarray) -> jnp.ndarray:
    """Binned signal-only block M_signal, shape (n_chan, n_chan, n_bins).

    Convenience wrapper around ``_build_M_signal_unbinned`` that applies
    the bin matrix W along the last axis. Used by the per-bin
    block-diagonal fast path; the BPWF-aware full-covariance path uses
    ``_build_M_signal_unbinned`` directly to keep the per-ℓ spectrum.
    """
    M_ell = _build_M_signal_unbinned(signal_model, fiducial_params)
    W = signal_model.bin_matrix           # (n_bins, n_ells)
    # Contract on the last axis: M[i, j, b] = Σ_ℓ W[b, ℓ] M_ell[i, j, ℓ].
    return jnp.einsum('be,ije->ijb', W, M_ell)


def _knox_blocks(M: jnp.ndarray,
                 signal_model: SignalModel,
                 f_sky: float) -> jnp.ndarray:
    """Apply the Knox 4-point formula to M = S + N.

    Returns per-bin covariance blocks of shape (n_bins, n_spec, n_spec):
        Cov_b[s1=(i,j), s2=(k,l)] = (M_ik M_jl + M_il M_jk) / ν_b.
    """
    nu = _nu_b(signal_model.bin_edges, f_sky)
    pairs = signal_model.freq_pairs
    i_arr = jnp.array([p[0] for p in pairs])
    j_arr = jnp.array([p[1] for p in pairs])
    M_ik = M[i_arr[:, None], i_arr[None, :], :]
    M_jl = M[j_arr[:, None], j_arr[None, :], :]
    M_il = M[i_arr[:, None], j_arr[None, :], :]
    M_jk = M[j_arr[:, None], i_arr[None, :], :]
    cov_blocks = (M_ik * M_jl + M_il * M_jk) / nu[None, None, :]
    return cov_blocks.transpose(2, 0, 1)


def _build_M(signal_model: SignalModel,
             instrument: Instrument,
             fiducial_params: jnp.ndarray) -> jnp.ndarray:
    """Build total M = S + N at fiducial, shape (n_chan, n_chan, n_bins).

    Noise N_ℓ per channel comes from instrument.noise_nl; it is binned
    via W and added only on the i==i diagonal.
    """
    M = _build_M_signal(signal_model, fiducial_params)
    W = signal_model.bin_matrix
    ells = signal_model.ells
    for i, ch in enumerate(instrument.channels):
        nl_i = noise_nl(ch, ells,
                        instrument.mission_duration_years, instrument.f_sky)
        M = M.at[i, i, :].add(W @ nl_i)
    return M


def _build_M_unbinned(signal_model: SignalModel,
                      instrument: Instrument,
                      fiducial_params: jnp.ndarray) -> jnp.ndarray:
    """Build total M(ℓ) = S(ℓ) + N(ℓ), shape (n_chan, n_chan, n_ells).

    Like ``_build_M`` but keeps the per-ℓ spectrum without applying W.
    Used by the BPWF-aware full-covariance path, which contracts with
    W_b(ℓ) W_{b'}(ℓ) inside the Knox sum rather than averaging M(ℓ)
    across a bin first.
    """
    M = _build_M_signal_unbinned(signal_model, fiducial_params)
    ells = signal_model.ells
    for i, ch in enumerate(instrument.channels):
        nl_i = noise_nl(ch, ells,
                        instrument.mission_duration_years, instrument.f_sky)
        M = M.at[i, i, :].add(nl_i)
    return M


def _knox_full(M_ell: jnp.ndarray,
               signal_model: SignalModel,
               f_sky: float) -> jnp.ndarray:
    """Apply the BPWF-aware Knox formula to per-ℓ M = S + N.

    Returns the full bandpower covariance,

        Cov[(ij,b), (kl,b')] = Σ_ℓ W_b(ℓ) W_{b'}(ℓ)
                             × [M_ik(ℓ) M_jl(ℓ) + M_il(ℓ) M_jk(ℓ)]
                             / [(2ℓ+1) f_sky]

    of shape ``(n_data, n_data)`` with ``n_data = n_spec * n_bins`` and
    ordering matching ``SignalModel.data_vector`` (spec varies slowest,
    bin fastest).

    This generalises the per-bin block-diagonal Knox formula to
    arbitrary (overlapping) bandpower window functions: bins couple
    through Σ_ℓ W_b(ℓ) W_{b'}(ℓ) (2ℓ+1), so the result is *not* block
    diagonal in bins. For non-overlapping single-ℓ delta windows it
    reduces exactly to the block-diagonal form returned by
    ``_knox_blocks``; for finite-width tophats the two paths agree to
    leading order in the variation of M(ℓ) across a bin.
    """
    W = signal_model.bin_matrix
    ells = signal_model.ells
    pairs = signal_model.freq_pairs
    n_spec = len(pairs)
    n_bins = signal_model.n_bins
    i_arr = jnp.array([p[0] for p in pairs])
    j_arr = jnp.array([p[1] for p in pairs])

    M_ik = M_ell[i_arr[:, None], i_arr[None, :], :]
    M_jl = M_ell[j_arr[:, None], j_arr[None, :], :]
    M_il = M_ell[i_arr[:, None], j_arr[None, :], :]
    M_jk = M_ell[j_arr[:, None], i_arr[None, :], :]
    knox_per_ell = (M_ik * M_jl + M_il * M_jk) \
                   / ((2.0 * ells + 1.0) * f_sky)
    # shape: (n_spec, n_spec, n_ells)

    # cov[s, b, S, c] = Σ_ℓ W[b, ℓ] W[c, ℓ] knox_per_ell[s, S, ℓ]
    cov_4d = jnp.einsum('be,ce,sSe->sbSc', W, W, knox_per_ell)
    return cov_4d.reshape(n_spec * n_bins, n_spec * n_bins)


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

        For the synthetic top-hat / Gaussian binning (the default), the
        result is block-diagonal in bins (different ℓ-bins are
        uncorrelated under the Knox approximation). When the
        SignalModel was constructed with a measured BPWF, bins couple
        through Σ_ℓ W_b(ℓ) W_{b'}(ℓ) (2ℓ+1) and the matrix is *not*
        block-diagonal -- the full covariance is computed via
        ``bandpower_covariance_full``.
    """
    if signal_model.has_measured_bpwf:
        return bandpower_covariance_full(signal_model, instrument,
                                         fiducial_params)

    n_bins = signal_model.n_bins
    n_spec = len(signal_model.freq_pairs)

    # Per-bin blocks, then inflate to a block-diagonal (n_data, n_data).
    blocks = bandpower_covariance_blocks(signal_model, instrument,
                                         fiducial_params)
    # blocks has shape (n_bins, n_spec, n_spec); reorder to
    # (n_spec, n_bins, n_spec, n_bins) with a delta on the bin axes.
    eye_b = jnp.eye(n_bins)
    cov_4d = blocks.transpose(1, 0, 2)[:, :, :, None] \
             * eye_b[None, :, None, :]
    # cov_4d shape: (n_spec, n_bins, n_spec, n_bins)
    return cov_4d.reshape(n_spec * n_bins, n_spec * n_bins)


def bandpower_covariance_full(signal_model: SignalModel,
                               instrument: Instrument,
                               fiducial_params: jnp.ndarray
                               ) -> jnp.ndarray:
    """Full BPWF-aware bandpower covariance (analytic noise).

    Returns the (n_data, n_data) covariance via the per-ℓ Knox sum
    contracted with the bin matrix, capturing bin-bin coupling for
    overlapping or measured bandpower window functions. Reduces exactly
    to the per-bin block-diagonal fast path for non-overlapping
    single-ℓ delta windows.

    Use this directly only when the synthetic-binning fast path is
    inappropriate (overlapping bins, measured BPWFs); for the default
    top-hat case the per-bin path is faster and equivalent.
    """
    M_ell = _build_M_unbinned(signal_model, instrument, fiducial_params)
    return _knox_full(M_ell, signal_model, instrument.f_sky)


def bandpower_covariance_full_from_noise(
    signal_model: SignalModel,
    noise_nls: jnp.ndarray,
    f_sky: float,
    fiducial_params: jnp.ndarray,
) -> jnp.ndarray:
    """Full BPWF-aware covariance from pre-computed noise arrays.

    Like ``bandpower_covariance_full`` but takes a ``(n_chan, n_ells)``
    noise array instead of an ``Instrument`` (matching
    ``bandpower_covariance_blocks_from_noise``). The same beam-
    deconvolved-noise contract applies -- see that function's docstring.
    """
    M_ell = _build_M_signal_unbinned(signal_model, fiducial_params)
    n_chan = noise_nls.shape[0]
    for i in range(n_chan):
        M_ell = M_ell.at[i, i, :].add(noise_nls[i])
    return _knox_full(M_ell, signal_model, f_sky)


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

    Raises ``NotImplementedError`` when ``signal_model.has_measured_bpwf``
    is True: bins couple through the BPWF overlap, so the per-bin
    block-diagonal structure does not exist. Use
    ``bandpower_covariance_full`` (or ``bandpower_covariance``, which
    dispatches automatically) instead.
    """
    if signal_model.has_measured_bpwf:
        raise NotImplementedError(
            "bandpower_covariance_blocks does not support measured BPWFs: "
            "overlapping bins couple through Σ_ℓ W_b(ℓ) W_{b'}(ℓ) (2ℓ+1) "
            "and the per-bin block-diagonal Knox approximation breaks. "
            "Use bandpower_covariance_full or bandpower_covariance "
            "(which dispatches automatically).")
    M = _build_M(signal_model, instrument, fiducial_params)
    return _knox_blocks(M, signal_model, instrument.f_sky)


def bandpower_covariance_blocks_from_noise(
    signal_model: SignalModel,
    noise_nls: jnp.ndarray,
    f_sky: float,
    fiducial_params: jnp.ndarray,
) -> jnp.ndarray:
    """Per-bin Knox covariance from pre-computed noise arrays.

    Like bandpower_covariance_blocks(), but takes a (n_chan, n_ells) noise
    array instead of an Instrument object. This allows JAX to trace through
    the noise computation for gradient-based instrument optimization.

    Args:
        signal_model:    Defines binning, freq pairs, and signal model.
        noise_nls:       Pre-computed noise N_ℓ per channel, shape
                         (n_chan, n_ells) where n_ells = len(signal_model.ells).
                         **MUST be beam-deconvolved**: the signal side uses
                         raw C_ℓ with no B_ℓ² factor, so beam-convolved noise
                         will make the Fisher over-optimistic at every ℓ
                         where B_ℓ² < 1.  NILC / compsep outputs are
                         naturally beam-deconvolved; raw anafast auto-
                         spectra are not -- use deconvolve_noise_bb()
                         below if you have the latter.
        f_sky:           Sky fraction (for effective mode count).
        fiducial_params: Flat parameter array at which to evaluate the
                         signal part of M = S + N.

    Returns:
        Per-bin covariance blocks, shape (n_bins, n_spec, n_spec).

    Raises:
        NotImplementedError: when ``signal_model.has_measured_bpwf`` is
            True. Use ``bandpower_covariance_full_from_noise`` instead.
    """
    if signal_model.has_measured_bpwf:
        raise NotImplementedError(
            "bandpower_covariance_blocks_from_noise does not support "
            "measured BPWFs: bins couple through "
            "Σ_ℓ W_b(ℓ) W_{b'}(ℓ) (2ℓ+1) and the per-bin block-diagonal "
            "Knox approximation breaks. Use "
            "bandpower_covariance_full_from_noise instead.")
    M = _build_M_signal(signal_model, fiducial_params)
    W = signal_model.bin_matrix
    n_chan = noise_nls.shape[0]
    for i in range(n_chan):
        M = M.at[i, i, :].add(W @ noise_nls[i])
    return _knox_blocks(M, signal_model, f_sky)
