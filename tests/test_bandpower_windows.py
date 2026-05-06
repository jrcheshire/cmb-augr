"""Tests for bandpower_windows.py."""

import numpy as np
import pytest

from augr.bandpower_windows import (
    load_bandpower_window,
    load_bandpower_window_set,
)


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


# -----------------------------------------------------------------------
# Per-spectrum loader (Phase 2)
# -----------------------------------------------------------------------

def _build_pair_dict(pairs, ells, n_bins=2):
    """Distinct-per-pair Gaussian BPWF dict for round-trip tests."""
    out = {}
    for s, (i, j) in enumerate(pairs):
        centers = np.linspace(ells[5], ells[-5], n_bins) + 2.0 * s
        rows = np.array([
            np.exp(-(ells - c) ** 2 / (2.0 * 12.0 ** 2)) for c in centers
        ])
        rows /= rows.sum(axis=1, keepdims=True)
        out[(i, j)] = rows
    return out


def test_load_set_multi_file_round_trip(tmp_path):
    """Directory of bpwf_i_j files round-trips through the loader."""
    ells = np.arange(20, 201, dtype=float)
    pairs = [(0, 0), (0, 1), (1, 1)]
    expected = _build_pair_dict(pairs, ells)
    for (i, j), W in expected.items():
        # Mix formats (.npy and .csv) to exercise both per-file paths.
        ext = ".npy" if i == j else ".csv"
        path = tmp_path / f"bpwf_{i}_{j}{ext}"
        table = np.column_stack([ells, W.T])
        if ext == ".npy":
            np.save(path, table)
        else:
            np.savetxt(path, table, delimiter=",")

    ells_back, by_pair = load_bandpower_window_set(tmp_path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    assert set(by_pair.keys()) == {(0, 0), (0, 1), (1, 1)}
    for k, W in expected.items():
        np.testing.assert_allclose(by_pair[k], W, rtol=1e-9)


def test_load_set_canonicalisation(tmp_path):
    """`bpwf_1_0` is canonicalised to (0, 1); `bpwf_0_1` collides."""
    ells = np.arange(20, 100, dtype=float)
    pair = _build_pair_dict([(0, 1)], ells)[(0, 1)]
    table = np.column_stack([ells, pair.T])
    np.save(tmp_path / "bpwf_1_0.npy", table)
    np.save(tmp_path / "bpwf_0_1.npy", table)
    with pytest.raises(ValueError, match="Duplicate BPWF for pair"):
        load_bandpower_window_set(tmp_path)


def test_load_set_inconsistent_ells_raises(tmp_path):
    """ell grids must agree across files."""
    ells_a = np.arange(20, 100, dtype=float)
    ells_b = np.arange(20, 80, dtype=float)
    W_a = _build_pair_dict([(0, 0)], ells_a)[(0, 0)]
    W_b = _build_pair_dict([(0, 1)], ells_b)[(0, 1)]
    np.save(tmp_path / "bpwf_0_0.npy", np.column_stack([ells_a, W_a.T]))
    np.save(tmp_path / "bpwf_0_1.npy", np.column_stack([ells_b, W_b.T]))
    with pytest.raises(ValueError, match="ell grids disagree"):
        load_bandpower_window_set(tmp_path)


def test_load_set_no_matching_files_raises(tmp_path):
    """An empty directory (or one without bpwf_i_j files) raises."""
    (tmp_path / "unrelated.txt").write_text("nothing here\n")
    with pytest.raises(FileNotFoundError, match="no BPWF files"):
        load_bandpower_window_set(tmp_path)


def test_load_set_npz_tensor_round_trip(tmp_path):
    """Single .npz with 3-D window + freq_pairs unpacks into a dict."""
    ells = np.arange(20, 201, dtype=float)
    pairs = [(0, 0), (0, 1), (1, 1)]
    by_pair = _build_pair_dict(pairs, ells)
    n_pairs = len(pairs)
    n_bins = list(by_pair.values())[0].shape[0]
    n_ells = ells.size
    window_3d = np.zeros((n_pairs, n_bins, n_ells))
    for s, p in enumerate(pairs):
        window_3d[s] = by_pair[p]
    freq_pairs = np.array(pairs, dtype=int)
    path = tmp_path / "bpwf_set.npz"
    np.savez(path, ells=ells, window=window_3d, freq_pairs=freq_pairs)

    ells_back, by_pair_back = load_bandpower_window_set(path)
    np.testing.assert_allclose(ells_back, ells, rtol=1e-12)
    for p, W in by_pair.items():
        np.testing.assert_allclose(by_pair_back[p], W, rtol=1e-12)


def test_load_set_npz_missing_keys_raises(tmp_path):
    """The 3-D npz must carry ells / window / freq_pairs together."""
    path = tmp_path / "bad.npz"
    np.savez(path, ells=np.arange(10), window=np.ones((1, 1, 10)))
    with pytest.raises(ValueError,
                        match="'ells', 'window', and 'freq_pairs'"):
        load_bandpower_window_set(path)


def test_load_set_npz_freq_pairs_shape_validation(tmp_path):
    """freq_pairs leading axis must match window leading axis."""
    path = tmp_path / "mismatch.npz"
    ells = np.arange(10, 20, dtype=float)
    window = np.ones((3, 2, ells.size))
    freq_pairs = np.array([[0, 0], [0, 1]])  # 2 entries, not 3
    np.savez(path, ells=ells, window=window, freq_pairs=freq_pairs)
    with pytest.raises(ValueError,
                        match="does not match 'window' leading axis"):
        load_bandpower_window_set(path)
