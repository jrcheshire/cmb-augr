"""Tests for augr.likelihood.ordering + the BinnedSpectra carrier (pure synthetic)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.likelihood.ordering import (
    SpectrumLayout,
    matrices_to_spectra,
    spectra_to_matrices,
)
from augr.likelihood.protocols import BinnedSpectra


def _upper_tri_pairs(m: int) -> list[tuple[int, int]]:
    """Row-major upper-triangle pairs (i <= j) — augr's freq_pairs convention."""
    return [(i, j) for i in range(m) for j in range(i, m)]


class TestSpectrumLayout:
    def test_from_freq_pairs_infers_dims(self):
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(3), n_bins=4)
        assert layout.n_field == 3
        assert layout.n_spec == 6  # 3*4/2
        assert layout.n_bins == 4
        assert layout.n_data == 24

    def test_hashable_for_static_field(self):
        # Must be hashable to ride as eqx.field(static=True).
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(2), n_bins=2)
        assert isinstance(hash(layout), int)

    def test_rejects_inconsistent_n_field(self):
        with pytest.raises(ValueError, match="inconsistent"):
            SpectrumLayout(pair_idx=((0, 0), (0, 1)), n_field=3, n_bins=1)

    def test_rejects_noncanonical_pair(self):
        with pytest.raises(ValueError, match="canonical"):
            SpectrumLayout(pair_idx=((0, 0), (1, 0), (1, 1)), n_field=2, n_bins=1)


class TestConversions:
    def test_hand_case_2x2(self):
        # pair order (0,0),(0,1),(1,1); vec is spec-slowest / bin-fastest.
        layout = SpectrumLayout(pair_idx=((0, 0), (0, 1), (1, 1)), n_field=2, n_bins=2)
        a0, a1, b0, b1, c0, c1 = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        vec = jnp.array([a0, a1, b0, b1, c0, c1])
        mats = spectra_to_matrices(vec, layout)
        assert mats.shape == (2, 2, 2)
        np.testing.assert_allclose(mats[:, :, 0], [[a0, b0], [b0, c0]])
        np.testing.assert_allclose(mats[:, :, 1], [[a1, b1], [b1, c1]])

    def test_roundtrip(self):
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(4), n_bins=5)
        rng = np.random.default_rng(0)
        vec = jnp.asarray(rng.standard_normal(layout.n_data))
        back = matrices_to_spectra(spectra_to_matrices(vec, layout), layout)
        np.testing.assert_allclose(np.asarray(back), np.asarray(vec), rtol=0, atol=0)

    def test_matrices_are_symmetric(self):
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(3), n_bins=2)
        rng = np.random.default_rng(1)
        vec = jnp.asarray(rng.standard_normal(layout.n_data))
        mats = spectra_to_matrices(vec, layout)
        np.testing.assert_allclose(np.asarray(mats), np.asarray(jnp.swapaxes(mats, 0, 1)))


class TestBinnedSpectra:
    def test_views_agree(self):
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(3), n_bins=4)
        rng = np.random.default_rng(2)
        cl = jnp.asarray(rng.standard_normal(layout.n_data))
        bs = BinnedSpectra(cl=cl, layout=layout)
        np.testing.assert_allclose(np.asarray(bs.as_vector()), np.asarray(cl))
        np.testing.assert_allclose(
            np.asarray(bs.as_bin_matrices()),
            np.asarray(spectra_to_matrices(cl, layout)),
        )

    def test_pytree_leaf_is_cl_only(self):
        # equinox: cl is the single traced leaf; the layout is static metadata.
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(2), n_bins=2)
        bs = BinnedSpectra(cl=jnp.ones(layout.n_data), layout=layout)
        leaves = jax.tree_util.tree_leaves(bs)
        assert len(leaves) == 1
        np.testing.assert_allclose(np.asarray(leaves[0]), np.ones(layout.n_data))

    def test_differentiable_through_carrier(self):
        # grad flows through cl into the per-bin matrix view.
        layout = SpectrumLayout.from_freq_pairs(_upper_tri_pairs(2), n_bins=1)

        def f(cl):
            bs = BinnedSpectra(cl=cl, layout=layout)
            return jnp.sum(bs.as_bin_matrices() ** 2)

        cl = jnp.array([1.0, 2.0, 3.0])  # 3 spectra * 1 bin
        g = jax.grad(f)(cl)
        # autos (0,0),(1,1) appear once in the matrix; cross (0,1) appears twice
        # → d/dcl sum(C^2) = 2*cl for autos, 4*cl for the cross.
        np.testing.assert_allclose(np.asarray(g), [2 * 1.0, 4 * 2.0, 2 * 3.0])
