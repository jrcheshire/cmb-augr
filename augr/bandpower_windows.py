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

This module provides a single loader that sniffs the file extension and
parses the BPWF in a unified format. The convention adopted across all
text and .npy formats is a 2-D table

    (n_ells, 1 + n_bins),

with column 0 = ℓ and columns 1..n_bins = W_b(ℓ). The .npz format is
dict-like and uses named arrays ``ells`` (shape ``(n_ells,)``) and
``window`` (shape ``(n_bins, n_ells)``). The loader returns
``(ells, W)`` with W reshaped to the augr-canonical (n_bins, n_ells).
"""

from __future__ import annotations

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
