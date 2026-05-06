"""
signal.py — Total signal model and JAX-autodiff Jacobian.

Combines CMB BB spectra and foreground spectra into a data vector μ(θ)
of binned bandpowers across all unique frequency cross-spectra, and
computes ∂μ/∂θ exactly via jax.jacfwd (forward-mode autodiff, optimal
since n_params ~ 10 << n_data ~ hundreds).

Data vector ordering:
    [(ν₁×ν₁, b₁), ..., (ν₁×ν₁, bN), (ν₁×ν₂, b₁), ..., (νM×νM, bN)]

where (νᵢ×νⱼ) runs over unique pairs with i ≤ j, and bₖ runs over
bandpower bins.

Parameter vector ordering (standard mode):
    [r, A_lens, <foreground parameters in model order>]

In delensed mode, A_lens is replaced by a precomputed residual lensing
BB spectrum from the iterative QE delensing procedure (see delensing.py),
and the parameter vector becomes:
    [r, <foreground parameters in model order>]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from augr.foregrounds import ForegroundModel
from augr.instrument import Instrument
from augr.spectra import CMBSpectra

# -----------------------------------------------------------------------
# Binning helpers (executed once at init, not during JAX tracing)
# -----------------------------------------------------------------------
#
# Two binning paths are supported:
#
# 1. Synthetic top-hat / Gaussian over contiguous integer (lo, hi) bin
#    ranges -- ``_make_bin_edges`` + ``_build_bin_matrix`` below. This
#    is the analytic-forecast default and supports the per-bin Knox
#    block-diagonal covariance fast path in ``covariance.py``.
#
# 2. Measured bandpower window functions (BICEP/Keck releases, NaMaster
#    output, bk-jax outputs) -- ``_bandpower_window_to_bin_matrix``
#    below. The user supplies a (n_bins, n_ells) matrix W with mask-mode
#    coupling, beam smoothing, apodization, and transfer-function
#    corrections already baked in. BPWFs typically overlap between
#    adjacent bins, so the bandpower covariance is no longer
#    block-diagonal: see ``covariance.bandpower_covariance_full`` for the
#    overlap-aware Knox sum. ``FisherForecast`` dispatches to the full
#    path automatically when ``signal_model.has_measured_bpwf`` is True.
#
# Sibling contract: BPWFs released by analysis pipelines have the beam
# baked in, so the noise spectrum that goes alongside them must be
# beam-deconvolved. ``FisherForecast`` enforces this by requiring
# ``external_noise_bb`` whenever ``has_measured_bpwf`` is True.
#
# Phase 2 (deferred): per-spectrum BPWFs. BICEP/Keck-style releases give
# one BPWF per cross-spectrum (i, j); the current API applies one
# ``bin_matrix`` to every spectrum. Generalizing to a (n_bins, n_ells,
# n_pairs) tensor will require updates to ``data_vector`` and
# ``_build_M_*``.

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


def _bandpower_window_to_bin_matrix(
        bpwf: np.ndarray,
        bpwf_ells: np.ndarray,
        target_ells: np.ndarray,
        ell_min: int,
        ell_max: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate a measured BPWF onto target ell grid; compute bin centers.

    The user-supplied BPWF is *not* re-normalised: BICEP/Keck-style
    pipelines release windows already-normalised so that
    <C_b> = Σ W_bℓ C_ℓ matches the bandpower estimator, and forcing
    row-sum=1 here would corrupt that calibration.

    Args:
        bpwf:        (n_bins, n_ells_in) array of W_b(ell) values.
        bpwf_ells:   (n_ells_in,) array of multipoles for ``bpwf``.
                     Must be strictly increasing and cover
                     [ell_min, ell_max].
        target_ells: (n_ells_target,) ell grid of the SignalModel.
        ell_min, ell_max: SignalModel ell range; used only for the
                     coverage check.

    Returns:
        (W, centers): JAX arrays of shape (n_bins, n_ells_target) and
        (n_bins,). ``centers[b] = Σ_ℓ ℓ W_b(ℓ) / Σ_ℓ W_b(ℓ)``; for rows
        whose weight sums to ~zero, falls back to the |W_b| argmax to
        keep the array finite.
    """
    bpwf = np.asarray(bpwf, dtype=float)
    bpwf_ells = np.asarray(bpwf_ells, dtype=float)
    target_ells = np.asarray(target_ells, dtype=float)
    if bpwf.ndim != 2:
        raise ValueError(
            f"bandpower_window must be 2-D (n_bins, n_ells); "
            f"got shape {bpwf.shape}.")
    if bpwf.shape[1] != bpwf_ells.shape[0]:
        raise ValueError(
            f"bandpower_window has {bpwf.shape[1]} ell columns but "
            f"bandpower_window_ells has length {bpwf_ells.shape[0]}.")
    if bpwf_ells.shape[0] < 2:
        raise ValueError(
            "bandpower_window_ells must have length >= 2 for "
            f"interpolation; got length {bpwf_ells.shape[0]}.")
    if not np.all(np.isfinite(bpwf)):
        raise ValueError("bandpower_window contains non-finite values.")
    if not np.all(np.isfinite(bpwf_ells)):
        raise ValueError("bandpower_window_ells contains non-finite values.")
    if not np.all(np.diff(bpwf_ells) > 0):
        raise ValueError(
            "bandpower_window_ells must be strictly increasing.")
    lo, hi = float(bpwf_ells[0]), float(bpwf_ells[-1])
    # Mirror the delensed_bb invariant: zero-extrapolation at the
    # reionization bump (or high-ell tail) would silently null the
    # BPWF response at multipoles where sigma(r) is most sensitive
    # for a space mission. Force the user to supply a window that
    # spans the SignalModel range.
    if lo > ell_min or hi < ell_max:
        raise ValueError(
            f"bandpower_window_ells range [{lo:g}, {hi:g}] must cover "
            f"the SignalModel ell range [{ell_min}, {ell_max}].")

    n_bins = bpwf.shape[0]
    n_ells_target = target_ells.shape[0]
    W_out = np.zeros((n_bins, n_ells_target))
    for b in range(n_bins):
        # Linear interp on the supplied grid; outside [bpwf_ells[0],
        # bpwf_ells[-1]] zero-extrapolate (we already enforced the
        # supplied range covers [ell_min, ell_max], so this is only
        # exercised when target_ells extends outside that range, which
        # shouldn't happen for a SignalModel-derived grid).
        W_out[b] = np.interp(target_ells, bpwf_ells, bpwf[b],
                             left=0.0, right=0.0)

    centers = np.zeros(n_bins)
    for b in range(n_bins):
        wsum = W_out[b].sum()
        if abs(wsum) < 1e-15:
            # Pathological row (e.g. a BPWF whose support is entirely
            # outside [ell_min, ell_max] after interpolation). Fall back
            # to the |W| peak position so bin_centers stays finite; the
            # row will contribute negligibly to the data vector anyway.
            centers[b] = float(target_ells[int(np.argmax(np.abs(W_out[b])))])
        else:
            centers[b] = float((target_ells * W_out[b]).sum() / wsum)

    return jnp.array(W_out), jnp.array(centers)


def _pack_per_spectrum_bpwfs(
        bpwf_by_pair: dict[tuple[int, int], np.ndarray],
        bpwf_ells: np.ndarray,
        target_ells: np.ndarray,
        ell_min: int,
        ell_max: int,
        freq_pairs: list[tuple[int, int]]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pack a dict of measured BPWFs into the (n_pairs, n_bins, n_ells) tensor.

    Per-spectrum BPWFs: BICEP/Keck-style multi-frequency analyses produce
    one bandpower window per (i, j) cross-spectrum because per-channel
    masks, transfer functions, and beam scales differ. The dict keys are
    `(i_ch, j_ch)` channel-index tuples; either ordering `(i, j)` or
    `(j, i)` is accepted and canonicalised to `(min, max)`.

    Args:
        bpwf_by_pair: ``{(i_ch, j_ch): W}`` with each W shape
                      ``(n_bins, n_ells_in)``. Every pair listed in
                      ``freq_pairs`` must be present; no extras allowed.
        bpwf_ells:    Shared ℓ grid for every BPWF in the dict.
        target_ells:  SignalModel ℓ grid.
        ell_min, ell_max: SignalModel ell range.
        freq_pairs:   Canonical (i ≤ j) pair list from SignalModel,
                      defining the order along the leading axis of the
                      output tensor.

    Returns:
        ``(W_3d, centers_first_pair)`` -- W_3d shape
        ``(n_pairs, n_bins, n_ells_target)``; centers_first_pair shape
        ``(n_bins,)`` derived from ``freq_pairs[0]`` (mode-agnostic
        first-pair representative consumed by ``bin_centers``).
    """
    if not isinstance(bpwf_by_pair, dict):
        raise TypeError(
            f"bandpower_window must be a dict for per-spectrum mode; "
            f"got {type(bpwf_by_pair).__name__}.")
    if len(bpwf_by_pair) == 0:
        raise ValueError("bandpower_window dict is empty.")

    # Canonicalise keys to (min, max) and detect collisions.
    canonical: dict[tuple[int, int], np.ndarray] = {}
    for key, W in bpwf_by_pair.items():
        if not (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, (int, np.integer)) for k in key)):
            raise ValueError(
                f"bandpower_window keys must be 2-tuples of channel "
                f"indices; got {key!r}.")
        ck = (int(min(key)), int(max(key)))
        if ck in canonical:
            raise ValueError(
                f"bandpower_window has duplicate entry for pair {ck} "
                f"(after canonicalising {key} to (min, max)).")
        canonical[ck] = np.asarray(W)

    expected = set(tuple(sorted(p)) for p in freq_pairs)
    missing = expected - set(canonical.keys())
    extra = set(canonical.keys()) - expected
    if missing:
        raise ValueError(
            f"bandpower_window is missing entries for cross-spectra "
            f"{sorted(missing)}. Every pair in SignalModel.freq_pairs "
            f"must have a BPWF in per-spectrum mode.")
    if extra:
        raise ValueError(
            f"bandpower_window has entries for unknown cross-spectra "
            f"{sorted(extra)}. Channel indices must be in "
            f"[0, n_channels).")

    # Build the per-pair (n_bins, n_ells_target) blocks via the existing
    # 2-D helper (validates shape, finiteness, range, monotonicity), then
    # stack in freq_pairs order.
    rows: list[jnp.ndarray] = []
    centers_first: jnp.ndarray | None = None
    n_bins_ref: int | None = None
    for pair in freq_pairs:
        ck = tuple(sorted(pair))
        W_block, centers_block = _bandpower_window_to_bin_matrix(
            canonical[ck], bpwf_ells, target_ells, ell_min, ell_max)
        if n_bins_ref is None:
            n_bins_ref = int(W_block.shape[0])
            centers_first = centers_block
        elif int(W_block.shape[0]) != n_bins_ref:
            raise ValueError(
                f"bandpower_window entries have inconsistent n_bins: "
                f"pair {sorted(freq_pairs[0])} has {n_bins_ref} bins, "
                f"pair {ck} has {W_block.shape[0]}. Per-spectrum BPWFs "
                f"must share a common bandpower binning.")
        rows.append(W_block)

    W_3d = jnp.stack(rows, axis=0)
    return W_3d, centers_first


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
        bandpower_window:  Optional measured BPWF. Two modes:

                           * Shared (Phase 1): a 2-D array of shape
                             ``(n_bins, n_ells)`` applied identically
                             to every cross-frequency spectrum.
                           * Per-spectrum (Phase 2): a dict
                             ``{(i_ch, j_ch): W}`` with one
                             ``(n_bins, n_ells)`` BPWF per cross-
                             spectrum -- captures per-channel mask /
                             transfer-function / beam differences in
                             multi-frequency releases (BICEP/Keck,
                             bk-jax). Either key ordering ``(i, j)``
                             or ``(j, i)`` is accepted; entries are
                             canonicalised to ``(min, max)``. Every
                             pair in ``freq_pairs`` must be supplied.

                           When supplied (either mode), ``ell_bins`` /
                           ``delta_ell`` / ``ell_per_bin_below`` /
                           ``window`` are ignored and ``has_measured_bpwf``
                           becomes True so that downstream consumers
                           (FisherForecast, the covariance routines)
                           take the overlap-aware full-covariance path.
        bandpower_window_ells: 1-D array of multipoles for
                           ``bandpower_window``; must span
                           [ell_min, ell_max]. Required when
                           ``bandpower_window`` is supplied. In
                           per-spectrum mode this is a single shared
                           grid for every BPWF in the dict; pre-
                           interpolate if your sources used different
                           grids.
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
                 use_jit: bool = True,
                 delensed_bb: jnp.ndarray | None = None,
                 delensed_bb_ells: jnp.ndarray | None = None,
                 residual_template_cl: jnp.ndarray | None = None,
                 residual_template_ells: jnp.ndarray | None = None,
                 bandpower_window: (
                     np.ndarray | jnp.ndarray
                     | dict[tuple[int, int], np.ndarray] | None) = None,
                 bandpower_window_ells: (
                     np.ndarray | jnp.ndarray | None) = None):
        self._instrument = instrument
        self._fg_model = foreground_model
        self._cmb = cmb_spectra

        # Delensed mode: precomputed residual lensing BB replaces A_lens
        self._delensed = delensed_bb is not None

        # Slice bounds for the foreground-parameter block within the full
        # flat parameter vector. Using explicit (start, end) avoids relying
        # on "everything after r/A_lens" and lets extra trailing params
        # (e.g. A_res from a residual template) coexist cleanly.
        n_fg = len(foreground_model.parameter_names)
        self._fg_start = 1 if self._delensed else 2
        self._fg_end = self._fg_start + n_fg

        if self._delensed:
            base_names = ["r", *list(foreground_model.parameter_names)]
        else:
            base_names = ["r", "A_lens", *list(foreground_model.parameter_names)]

        # Residual-template mode: optional additive post-CompSep residual
        # with amplitude A_res appended to the parameter vector.
        self._residual_template = residual_template_cl is not None
        if self._residual_template:
            self._a_res_idx = len(base_names)
            self._param_names = [*base_names, "A_res"]
        else:
            self._a_res_idx = None
            self._param_names = base_names

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

        # Pre-interpolate delensed BB onto our ell grid (not JAX-traced).
        # The supplied array must cover [ell_min, ell_max] in full;
        # zero-extrapolation outside that range would silently null the
        # residual lensing BB (critical at the reionization bump where
        # sigma(r) is most sensitive for a space mission, and equally
        # bad at high ell where the residual rises toward the lensing
        # peak).
        if self._delensed:
            if delensed_bb_ells is None:
                raise ValueError(
                    "delensed_bb requires delensed_bb_ells.")
            ells_in = jnp.asarray(delensed_bb_ells, dtype=float)
            cl_in = jnp.asarray(delensed_bb, dtype=float)
            if cl_in.shape != ells_in.shape:
                raise ValueError(
                    f"delensed_bb shape {cl_in.shape} must match "
                    f"delensed_bb_ells shape {ells_in.shape}.")
            if cl_in.shape[0] < 2:
                raise ValueError(
                    "delensed_bb and delensed_bb_ells must have length >= 2 "
                    f"for interpolation; got length {cl_in.shape[0]}.")
            lo, hi = float(ells_in[0]), float(ells_in[-1])
            if lo > ell_min or hi < ell_max:
                raise ValueError(
                    f"delensed_bb_ells range [{lo:g}, {hi:g}] must cover the "
                    f"SignalModel ell range [{ell_min}, {ell_max}]; "
                    "supply a wider delensed_bb grid or narrow ell_min/"
                    "ell_max.")
            self._delensed_bb = jnp.interp(self._ells, ells_in, cl_in)
        else:
            self._delensed_bb = None

        # Pre-interpolate residual template onto our ell grid (not JAX-traced).
        # The template represents the MC-averaged post-CompSep foreground
        # residual C_ell^BB; A_res scales it as a nuisance amplitude.
        # Outside the provided ell range we use nearest-neighbour
        # (jnp.interp default: fp[0] / fp[-1]) rather than zero. The
        # reionization bump (ell <~ 10) typically sits below the first
        # BROOM bandpower center, and zero-extrapolation there silently
        # nulls the A_res constraint exactly where sigma(r) is most
        # sensitive for a space mission.
        if self._residual_template:
            if residual_template_ells is None:
                raise ValueError(
                    "residual_template_cl requires residual_template_ells.")
            cl_in = jnp.asarray(residual_template_cl, dtype=float)
            ells_in = jnp.asarray(residual_template_ells, dtype=float)
            if cl_in.shape[0] < 2 or ells_in.shape[0] < 2:
                raise ValueError(
                    "residual_template_cl and residual_template_ells must "
                    "have length >= 2 for interpolation; got "
                    f"lengths {ells_in.shape[0]} and {cl_in.shape[0]}.")
            if cl_in.shape != ells_in.shape:
                raise ValueError(
                    f"residual_template_cl shape {cl_in.shape} must match "
                    f"residual_template_ells shape {ells_in.shape}.")
            self._residual_template_cl = jnp.interp(
                self._ells, ells_in, cl_in)
        else:
            self._residual_template_cl = None

        # Binning. Three paths converge into a single internal 3-D
        # tensor ``_bin_matrix_3d`` of shape (n_pairs, n_bins, n_ells)
        # that the covariance / data-vector code consumes uniformly.
        # Shared / synthetic modes also keep a 2-D ``_bin_matrix`` for
        # backward-compat reads via the public ``bin_matrix`` property.
        if (bandpower_window is None) != (bandpower_window_ells is None):
            raise ValueError(
                "bandpower_window and bandpower_window_ells must be "
                "supplied together.")
        self._has_measured_bpwf = bandpower_window is not None
        self._is_per_spectrum_bpwf = isinstance(bandpower_window, dict)

        n_pairs = len(self._freq_pairs)
        if self._is_per_spectrum_bpwf:
            self._bin_matrix_3d, self._bin_centers = (
                _pack_per_spectrum_bpwfs(
                    bandpower_window,
                    np.asarray(bandpower_window_ells),
                    ells_np, ell_min, ell_max,
                    self._freq_pairs))
            # The 2-D shared bin_matrix is undefined in per-spectrum
            # mode; access raises with a pointer to the 3-D / accessor
            # API.
            self._bin_matrix = None
            self._bin_edges = None
        elif self._has_measured_bpwf:
            self._bin_matrix, self._bin_centers = (
                _bandpower_window_to_bin_matrix(
                    np.asarray(bandpower_window),
                    np.asarray(bandpower_window_ells),
                    ells_np, ell_min, ell_max))
            # broadcast_to is a view (no extra memory); JAX handles
            # ``W_3d[s]`` indexing without materialising.
            self._bin_matrix_3d = jnp.broadcast_to(
                self._bin_matrix,
                (n_pairs, *self._bin_matrix.shape))
            # bin_edges has no meaningful definition for an arbitrary
            # BPWF (a Gaussian wing extends to infinity at any
            # tolerance). Downstream consumers gate on
            # has_measured_bpwf before reading it.
            self._bin_edges = None
        else:
            bin_edges = _make_bin_edges(ell_min, ell_max,
                                        ell_per_bin_below, delta_ell,
                                        ell_bins)
            self._bin_matrix, self._bin_centers = _build_bin_matrix(
                ells_np, bin_edges, window)
            self._bin_matrix_3d = jnp.broadcast_to(
                self._bin_matrix,
                (n_pairs, *self._bin_matrix.shape))
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

    @property
    def bin_matrix(self) -> jnp.ndarray:
        """Shared bandpower-binning weight matrix W, shape (n_bins, n_ells).

        Applied to a C_ℓ vector on the ``ells`` grid, W @ C yields the
        bandpowers on the ``bin_centers`` grid. Available in synthetic
        binning and shared-BPWF modes; raises in per-spectrum BPWF mode
        (where W differs per cross-spectrum) -- use
        ``bin_matrix_per_spectrum`` or ``bandpower_window_for(i, j)``
        instead.
        """
        if self._is_per_spectrum_bpwf:
            raise ValueError(
                "bin_matrix is not defined in per-spectrum BPWF mode "
                "(W differs per cross-spectrum). Use "
                "signal_model.bin_matrix_per_spectrum for the full "
                "(n_pairs, n_bins, n_ells) tensor, or "
                "signal_model.bandpower_window_for(i_ch, j_ch) for a "
                "single pair.")
        return self._bin_matrix

    @property
    def bin_matrix_per_spectrum(self) -> jnp.ndarray:
        """Per-spectrum BPWF tensor of shape (n_pairs, n_bins, n_ells).

        Always defined: in synthetic and shared-BPWF modes this is a
        broadcast view of the 2-D ``bin_matrix``; in per-spectrum BPWF
        mode it carries one window per cross-spectrum, ordered to match
        ``freq_pairs``.
        """
        return self._bin_matrix_3d

    def bandpower_window_for(self, i_ch: int,
                             j_ch: int) -> jnp.ndarray:
        """Return the BPWF for cross-spectrum (i_ch, j_ch), shape (n_bins, n_ells).

        Mode-agnostic accessor: works in synthetic, shared-BPWF, and
        per-spectrum BPWF modes. Channel-index ordering is canonicalised
        to ``(min, max)``.
        """
        a, b = sorted((int(i_ch), int(j_ch)))
        try:
            s = self._freq_pairs.index((a, b))
        except ValueError as exc:
            raise ValueError(
                f"({i_ch}, {j_ch}) is not a valid cross-spectrum for "
                f"this instrument; valid pairs are {self._freq_pairs}."
            ) from exc
        return self._bin_matrix_3d[s]

    @property
    def bin_edges(self) -> list[tuple[int, int]] | None:
        """Bandpower bin edges as a list of (lo, hi) multipole pairs.

        Returns ``None`` when the bin matrix was supplied as a measured
        BPWF, where (lo, hi) intervals are not a meaningful description
        of the bin support. Use ``bin_centers`` for the per-bin
        characteristic multipole and ``bin_matrix`` for the full window.
        """
        return self._bin_edges

    @property
    def has_measured_bpwf(self) -> bool:
        """True if ``bin_matrix`` was supplied as measured BPWFs.

        Downstream consumers (``FisherForecast``, the covariance
        routines) gate on this to take the overlap-aware full-covariance
        path: the per-bin block-diagonal Knox approximation breaks for
        BPWFs that overlap between adjacent bins.
        """
        return self._has_measured_bpwf

    @property
    def is_per_spectrum_bpwf(self) -> bool:
        """True iff a per-spectrum (dict-of-pairs) BPWF was supplied.

        Implies ``has_measured_bpwf`` is also True. Steers the
        covariance Knox sum onto the 3-D einsum that picks up
        per-pair BPWF differences.
        """
        return self._is_per_spectrum_bpwf

    @property
    def frequencies(self) -> tuple[float, ...]:
        """Channel frequencies in GHz, in the same order as freq_pairs."""
        return self._freqs

    @property
    def foreground_model(self) -> ForegroundModel:
        """The ForegroundModel used to build the data vector."""
        return self._fg_model

    # ------------------------------------------------------------------
    # Data vector and Jacobian
    # ------------------------------------------------------------------

    def data_vector(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute the model bandpower data vector μ(θ).

        Args:
            params: Flat JAX array of length n_params.
                    Order: [r, (A_lens), <foreground params>, (A_res)].
                    A_lens is present only in non-delensed mode; A_res is
                    present only when a residual template is attached.

        Returns:
            Flat JAX array of length n_data = n_spectra × n_bins.
            Ordering: spectra vary slowest, bins vary fastest.
        """
        r = params[0]
        if self._delensed:
            # CMB BB = r × tensor + precomputed residual lensing
            cl_cmb = (r * self._cmb.cl_tensor_r1(self._ells)
                      + self._delensed_bb)
        else:
            A_lens = params[1]
            cl_cmb = self._cmb.cl_bb(self._ells, r, A_lens)

        fg_params = params[self._fg_start:self._fg_end]

        # Precompute the residual-template contribution (auto-spectra only).
        # Post-component-separation the residual lives in the single
        # cleaned map, so it only enters the i==j auto-BB blocks.
        if self._residual_template:
            cl_residual = params[self._a_res_idx] * self._residual_template_cl
        else:
            cl_residual = None

        # Build bandpowers for each frequency pair. In per-spectrum BPWF
        # mode the bin matrix differs per pair; in shared / synthetic
        # modes ``_bin_matrix_3d`` is a broadcast view of the 2-D matrix
        # so the same code path works without extra memory.
        bandpowers = []
        for s, (i_ch, j_ch) in enumerate(self._freq_pairs):
            nu_i = self._freqs[i_ch]
            nu_j = self._freqs[j_ch]
            cl_fg = self._fg_model.cl_bb(nu_i, nu_j, self._ells, fg_params)
            cl_total = cl_cmb + cl_fg
            if cl_residual is not None and i_ch == j_ch:
                cl_total = cl_total + cl_residual
            if self._is_per_spectrum_bpwf:
                bp = self._bin_matrix_3d[s] @ cl_total    # (n_bins,)
            else:
                bp = self._bin_matrix @ cl_total
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
    # Unbinned spectrum (used by covariance.py)
    # ------------------------------------------------------------------

    def cmb_bb_unbinned(self, params: jnp.ndarray) -> jnp.ndarray:
        """CMB BB spectrum on the ell grid, from parameters.

        Handles both standard (r, A_lens) and delensed (r only) modes.
        """
        r = params[0]
        if self._delensed:
            return r * self._cmb.cl_tensor_r1(self._ells) + self._delensed_bb
        else:
            A_lens = params[1]
            return self._cmb.cl_bb(self._ells, r, A_lens)

    def residual_bb_unbinned(self, params: jnp.ndarray) -> jnp.ndarray:
        """A_res-scaled residual-template BB on the ell grid.

        Post-component-separation, the residual lives in the single
        cleaned map, so this contribution only enters the auto-spectrum
        (i == j) blocks of M = S + N -- callers must apply it to the
        diagonal only. Returns zeros when no residual template is
        attached, so it is always safe to add.
        """
        if self._residual_template:
            return params[self._a_res_idx] * self._residual_template_cl
        return jnp.zeros_like(self._ells)

    def fg_params_from(self, params: jnp.ndarray) -> jnp.ndarray:
        """Extract foreground parameters from the full parameter vector.

        Slices exactly the foreground-parameter block, so trailing
        parameters (e.g. A_res) are not accidentally included.
        """
        return params[self._fg_start:self._fg_end]

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
