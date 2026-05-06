"""bandpower_windows.py — Load measured bandpower window functions.

A measured bandpower window function (BPWF) is a (n_bins, n_ells) matrix
W such that the bandpower estimator from an analysis pipeline is

    <C_b> = Σ_ℓ W_b(ℓ) C_ℓ,

where C_ℓ is the underlying (beam-deconvolved) sky spectrum. The kernel
typically encodes mask mode coupling, transfer-function corrections, and
beam smoothing -- everything between the measurement and the underlying
sky. Measured BPWFs are a drop-in replacement for augr's synthetic
top-hat / Gaussian binning when the forecast is meant to mirror a
specific real-data pipeline (BICEP/Keck releases, NaMaster output,
bk-jax outputs, ...). See ``SignalModel(bandpower_window=..., ...)`` and
``covariance.bandpower_covariance_full`` for the consumer side.

Two loader entry points:

* ``load_bandpower_window(path)`` -- a single 2-D BPWF (one file, one
  pair / one shared kernel). Returns ``(ells, W)``.

* ``load_bandpower_window_set(spec)`` -- a per-spectrum dict
  ``{(i_ch, j_ch): W}``, suitable for
  ``SignalModel(bandpower_window=...)`` in per-spectrum mode. ``spec``
  is either a directory / glob pointing at one
  ``bpwf_{i}_{j}.{ext}`` file per cross-spectrum, or a single ``.npz``
  carrying a 3-D ``window`` array plus a ``freq_pairs`` index map.
  Returns ``(ells, dict)``.

The convention adopted across all text and .npy formats is a 2-D table

    (n_ells, 1 + n_bins),

with column 0 = ℓ and columns 1..n_bins = W_b(ℓ). The .npz format is
dict-like and uses named arrays ``ells`` (shape ``(n_ells,)``) and
``window`` (shape ``(n_bins, n_ells)`` for single, or
``(n_pairs, n_bins, n_ells)`` for the per-spectrum tensor).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_bandpower_window(path: str | Path
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Load a measured BPWF from disk.

    Supported formats (sniffed from file extension):

    - ``.npy``: 2-D numpy array, shape ``(n_ells, 1 + n_bins)``.
      Column 0 is ℓ; columns 1..n_bins are W_b(ℓ).
    - ``.npz``: numpy archive containing arrays ``ells`` (shape
      ``(n_ells,)``) and ``window`` (shape ``(n_bins, n_ells)``).
    - ``.csv``, ``.dat``, ``.txt``: whitespace- or comma-delimited
      table (``np.loadtxt`` compatible), shape ``(n_ells, 1 + n_bins)``
      with the same column convention as ``.npy``. Lines starting with
      ``#`` are treated as comments. The delimiter is auto-detected:
      ``.csv`` defaults to comma, others to whitespace.

    Args:
        path: Path to the BPWF file.

    Returns:
        ``(ells, W)`` where ``ells`` has shape ``(n_ells,)`` and ``W``
        has shape ``(n_bins, n_ells)``. Suitable for direct use as
        ``SignalModel(bandpower_window=W, bandpower_window_ells=ells,
        ...)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: For unsupported extensions or malformed contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BPWF file not found: {path}")
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        ells, W = _split_table(arr, path)
    elif suffix == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            if "ells" not in keys or "window" not in keys:
                raise ValueError(
                    f"{path}: .npz BPWF must contain 'ells' and 'window' "
                    f"arrays; got keys {keys}.")
            ells = np.asarray(data["ells"], dtype=float)
            W = np.asarray(data["window"], dtype=float)
        if W.ndim != 2:
            raise ValueError(
                f"{path}: 'window' must be 2-D (n_bins, n_ells); "
                f"got shape {W.shape}.")
        if ells.ndim != 1:
            raise ValueError(
                f"{path}: 'ells' must be 1-D (n_ells,); "
                f"got shape {ells.shape}.")
        if W.shape[1] != ells.shape[0]:
            raise ValueError(
                f"{path}: 'window' has {W.shape[1]} ell columns but "
                f"'ells' has length {ells.shape[0]}.")
    elif suffix in (".csv", ".dat", ".txt"):
        delimiter = "," if suffix == ".csv" else None
        arr = np.loadtxt(path, comments="#", delimiter=delimiter)
        ells, W = _split_table(arr, path)
    else:
        raise ValueError(
            f"{path}: unsupported BPWF extension {suffix!r}. "
            f"Supported: .npy, .npz, .csv, .dat, .txt.")

    return ells, W


def _split_table(arr: np.ndarray,
                 path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a (n_ells, 1 + n_bins) table into ``(ells, W)``."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"{path}: BPWF table must be 2-D with shape "
            f"(n_ells, 1 + n_bins); got shape {arr.shape}.")
    ells = arr[:, 0]
    # Columns 1..n_bins are W_b(ℓ); transpose to canonical
    # (n_bins, n_ells).
    W = arr[:, 1:].T
    return ells, W


# -----------------------------------------------------------------------
# Per-spectrum (Phase 2) loader
# -----------------------------------------------------------------------

# Filename convention: ``bpwf_{i}_{j}.{ext}`` with i and j integer
# channel indices. Filename ordering is canonicalised to (min, max) by
# the SignalModel constructor, so either ``bpwf_0_1`` or ``bpwf_1_0`` is
# accepted; duplicate canonicals raise.
_BPWF_FILENAME = re.compile(r"^bpwf_(\d+)_(\d+)$")
_BPWF_EXTENSIONS = (".npy", ".npz", ".csv", ".dat", ".txt")


def load_bandpower_window_set(spec: str | Path) -> tuple[
        np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Load a per-spectrum BPWF dict for ``SignalModel`` Phase-2 mode.

    Two input shapes:

    1. **Directory of single-file BPWFs.** ``spec`` is a directory
       containing files named ``bpwf_{i}_{j}.{ext}`` (one per
       cross-spectrum). Each file is parsed with the existing single-
       file ``load_bandpower_window`` machinery, so any of the
       per-format conventions there (.npy / .npz / .csv / .dat / .txt)
       work. Channel indices are parsed from the filename and
       canonicalised to ``(min, max)``.

    2. **Single tensor .npz.** ``spec`` is a path to an ``.npz``
       carrying:

       - ``ells`` (shape ``(n_ells,)``)
       - ``window`` (shape ``(n_pairs, n_bins, n_ells)``)
       - ``freq_pairs`` (shape ``(n_pairs, 2)``, integer channel
         indices)

    Args:
        spec: Directory path / glob, or path to a single ``.npz``.

    Returns:
        ``(ells, by_pair)`` where ``ells`` is shape ``(n_ells,)`` and
        ``by_pair`` is ``{(i_ch, j_ch): W}`` with each W shape
        ``(n_bins, n_ells)``. Pass directly as
        ``SignalModel(bandpower_window=by_pair,
        bandpower_window_ells=ells, ...)``.

    Raises:
        FileNotFoundError: If ``spec`` does not exist or yields no
            matching files.
        ValueError: For malformed npz contents, inconsistent ell grids
            across files, or duplicate canonicalised pair indices.
    """
    spec = Path(spec)

    # Tensor .npz path.
    if spec.is_file() and spec.suffix.lower() == ".npz":
        return _load_set_from_npz(spec)

    # Directory / glob path.
    if spec.is_dir():
        candidates = [
            p for p in sorted(spec.iterdir())
            if p.is_file() and p.suffix.lower() in _BPWF_EXTENSIONS
            and _BPWF_FILENAME.match(p.stem)
        ]
    else:
        # Treat ``spec`` as a glob pattern (e.g. ``runs/.../bpwf_*.npy``).
        parent = spec.parent if spec.parent != Path("") else Path(".")
        if not parent.exists():
            raise FileNotFoundError(
                f"BPWF directory / glob parent not found: {parent}")
        candidates = [
            p for p in sorted(parent.glob(spec.name))
            if p.is_file() and p.suffix.lower() in _BPWF_EXTENSIONS
            and _BPWF_FILENAME.match(p.stem)
        ]

    if not candidates:
        raise FileNotFoundError(
            f"{spec}: no BPWF files matching 'bpwf_<i>_<j>.<ext>' found "
            f"(supported extensions: {', '.join(_BPWF_EXTENSIONS)}).")

    by_pair: dict[tuple[int, int], np.ndarray] = {}
    ells_ref: np.ndarray | None = None
    ref_path: Path | None = None
    for path in candidates:
        m = _BPWF_FILENAME.match(path.stem)
        assert m is not None  # filtered above
        i, j = int(m.group(1)), int(m.group(2))
        canonical = (min(i, j), max(i, j))
        if canonical in by_pair:
            raise ValueError(
                f"Duplicate BPWF for pair {canonical} in {spec}: "
                f"{path.name} collides with an earlier file (likely a "
                f"`bpwf_i_j` and `bpwf_j_i` pair pointing at the same "
                f"cross-spectrum).")
        ells_p, W_p = load_bandpower_window(path)
        if ells_ref is None:
            ells_ref = np.asarray(ells_p, dtype=float)
            ref_path = path
        else:
            if ells_p.shape != ells_ref.shape or not np.allclose(
                    ells_p, ells_ref):
                raise ValueError(
                    f"BPWF ell grids disagree across files: {path.name} "
                    f"vs {ref_path.name if ref_path else '?'}. "
                    f"Per-spectrum BPWFs must share a common ell grid; "
                    f"pre-interpolate before loading.")
        by_pair[canonical] = W_p

    return ells_ref, by_pair  # type: ignore[return-value]


def _load_set_from_npz(path: Path) -> tuple[
        np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Unpack a 3-D BPWF tensor + freq_pairs index from a single .npz."""
    with np.load(path) as data:
        keys = list(data.keys())
        for required in ("ells", "window", "freq_pairs"):
            if required not in keys:
                raise ValueError(
                    f"{path}: per-spectrum tensor .npz must contain "
                    f"'ells', 'window', and 'freq_pairs' arrays; got "
                    f"keys {keys}.")
        ells = np.asarray(data["ells"], dtype=float)
        window = np.asarray(data["window"], dtype=float)
        freq_pairs = np.asarray(data["freq_pairs"], dtype=int)

    if ells.ndim != 1:
        raise ValueError(
            f"{path}: 'ells' must be 1-D (n_ells,); "
            f"got shape {ells.shape}.")
    if window.ndim != 3:
        raise ValueError(
            f"{path}: 'window' must be 3-D "
            f"(n_pairs, n_bins, n_ells); got shape {window.shape}.")
    if window.shape[2] != ells.shape[0]:
        raise ValueError(
            f"{path}: 'window' has {window.shape[2]} ell columns but "
            f"'ells' has length {ells.shape[0]}.")
    if freq_pairs.ndim != 2 or freq_pairs.shape[1] != 2:
        raise ValueError(
            f"{path}: 'freq_pairs' must have shape (n_pairs, 2); "
            f"got {freq_pairs.shape}.")
    if freq_pairs.shape[0] != window.shape[0]:
        raise ValueError(
            f"{path}: 'freq_pairs' length {freq_pairs.shape[0]} does "
            f"not match 'window' leading axis {window.shape[0]}.")

    by_pair: dict[tuple[int, int], np.ndarray] = {}
    for s, (i, j) in enumerate(freq_pairs.tolist()):
        canonical = (min(int(i), int(j)), max(int(i), int(j)))
        if canonical in by_pair:
            raise ValueError(
                f"{path}: 'freq_pairs' has duplicate entry for "
                f"canonicalised pair {canonical} at index {s}.")
        by_pair[canonical] = window[s]
    return ells, by_pair
