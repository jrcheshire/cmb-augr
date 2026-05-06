"""Tests for bandpower_windows.py."""

import numpy as np
import pytest

from augr.bandpower_windows import load_bandpower_window


# -----------------------------------------------------------------------
# Synthetic BPWF builder (overlapping Gaussians)
# -----------------------------------------------------------------------

def _make_bpwf(n_bins: int = 4, ell_min: int = 20, ell_max: int = 200
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (ells, W) with W shape (n_bins, n_ells)."""
    ells = np.arange(ell_min, ell_max + 1, dtype=float)
    centers = np.linspace(ell_min + 10, ell_max - 10, n_bins)
    W = np.array([
        np.exp(-(ells - c) ** 2 / (2.0 * 12.0 ** 2)) for c in centers
    ])
    W /= W.sum(axis=1, keepdims=True)
    return ells, W


# -----------------------------------------------------------------------
# Round-trips per format
# -----------------------------------------------------------------------

def test_load_npy_round_trip(tmp_path):
    ells, W = _make_bpwf()
    # Canonical layout: (n_ells, 1 + n_bins) with ell as column 0.
    table = np.column_stack([ells, W.T])
    path = tmp_path / "bpwf.npy"
    np.save(path, table)

    ells_back, W_back = load_bandpower_window(path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    np.testing.assert_allclose(W_back, W, rtol=1e-12)


def test_load_npz_round_trip(tmp_path):
    ells, W = _make_bpwf()
    path = tmp_path / "bpwf.npz"
    np.savez(path, ells=ells, window=W)

    ells_back, W_back = load_bandpower_window(path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    np.testing.assert_allclose(W_back, W, rtol=1e-12)


def test_load_csv_round_trip(tmp_path):
    ells, W = _make_bpwf()
    table = np.column_stack([ells, W.T])
    path = tmp_path / "bpwf.csv"
    np.savetxt(path, table, delimiter=",", header="ell,W1,W2,W3,W4")

    ells_back, W_back = load_bandpower_window(path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    np.testing.assert_allclose(W_back, W, rtol=1e-12)


def test_load_dat_round_trip(tmp_path):
    ells, W = _make_bpwf()
    table = np.column_stack([ells, W.T])
    path = tmp_path / "bpwf.dat"
    np.savetxt(path, table, header="ell W1 W2 W3 W4")

    ells_back, W_back = load_bandpower_window(path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    np.testing.assert_allclose(W_back, W, rtol=1e-12)


def test_load_txt_with_extra_comments(tmp_path):
    """Comment lines starting with '#' are stripped."""
    ells, W = _make_bpwf(n_bins=2, ell_min=30, ell_max=80)
    table = np.column_stack([ells, W.T])
    path = tmp_path / "bpwf.txt"
    with path.open("w") as f:
        f.write("# augr test BPWF\n")
        f.write("# generated 2026-05-05\n")
        for row in table:
            f.write("  ".join(f"{v:.10e}" for v in row) + "\n")

    ells_back, W_back = load_bandpower_window(path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    # Hand-formatted text round-trip loses precision past the format
    # spec; the format-specific round-trips above are what test full
    # numerical fidelity.
    np.testing.assert_allclose(W_back, W, rtol=1e-9)


# -----------------------------------------------------------------------
# Failure modes
# -----------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bandpower_window(tmp_path / "no_such.npy")


def test_load_unsupported_extension_raises(tmp_path):
    path = tmp_path / "bpwf.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="unsupported BPWF extension"):
        load_bandpower_window(path)


def test_load_npy_wrong_shape_raises(tmp_path):
    """A 1-D .npy array is not a valid BPWF table."""
    path = tmp_path / "bad.npy"
    np.save(path, np.arange(10.0))
    with pytest.raises(ValueError, match="2-D"):
        load_bandpower_window(path)


def test_load_npy_one_column_raises(tmp_path):
    """A 2-D table with no W columns (only ell) is not a valid BPWF."""
    path = tmp_path / "ell_only.npy"
    np.save(path, np.arange(10.0).reshape(-1, 1))
    with pytest.raises(ValueError, match="1 \\+ n_bins"):
        load_bandpower_window(path)


def test_load_npz_missing_keys_raises(tmp_path):
    """An .npz lacking 'ells' / 'window' arrays should raise a clear error."""
    path = tmp_path / "bad.npz"
    np.savez(path, foo=np.arange(5))
    with pytest.raises(ValueError, match="'ells' and 'window'"):
        load_bandpower_window(path)


def test_load_npz_mismatched_shapes_raises(tmp_path):
    """ells length must match W's second axis."""
    path = tmp_path / "mismatched.npz"
    ells = np.arange(20, 50, dtype=float)
    W = np.ones((2, 50))  # wrong: 50 columns but 30 ells
    np.savez(path, ells=ells, window=W)
    with pytest.raises(ValueError, match="ell columns"):
        load_bandpower_window(path)
