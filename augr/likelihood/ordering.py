"""Spectrum layout + flat-vector ↔ per-bin-matrix conversions for the HL likelihood.

augr's binned cross-spectrum data vector is ordered ``spec`` slowest, ``bin``
fastest (``n_data = n_spec * n_bins``), where ``spec`` runs over the row-major
upper-triangle ``freq_pairs`` (i ≤ j) from :class:`augr.signal.SignalModel` and
matches the ordering produced by ``covariance._knox_full`` /
``SignalModel.data_vector``. The Hamimeche-Lewis likelihood works on per-bin
symmetric ``(n_field, n_field)`` matrices, so it must map between the flat
vector and the per-bin matrix stack.

:func:`spectra_to_matrices` / :func:`matrices_to_spectra` are the
descriptor-driven generalisation of bk-jax's lag-major ``vecp`` / ``ivecp``
(``bk_jax.likelihood.ordering``): rather than assuming the lag-major diagonal
order, they scatter / gather via the explicit ``pair_idx`` list, so augr's
row-major ``freq_pairs`` order works with no reordering. A different ordering
convention (e.g. bk-jax's lag-major, when bk-jax adopts this layer) is supported
by supplying a :class:`SpectrumLayout` carrying that ``pair_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SpectrumLayout:
    """Static descriptor for a binned cross-spectrum data vector.

    Frozen + hashable so it can serve as an ``eqx.field(static=True)`` on
    :class:`augr.likelihood.protocols.BinnedSpectra`.

    Attributes
    ----------
    pair_idx
        Channel/field index pair ``(i, j)`` with ``i <= j`` for each spectrum,
        in data-vector order. Length ``n_spec``.
    n_field
        Number of fields/channels ``M``; the per-bin matrices are ``(M, M)``.
    n_bins
        Number of bandpower bins.
    """

    pair_idx: tuple[tuple[int, int], ...]
    n_field: int
    n_bins: int

    def __post_init__(self) -> None:
        m = self.n_field
        expected = m * (m + 1) // 2
        if self.n_spec != expected:
            raise ValueError(
                f"n_spec = {self.n_spec} inconsistent with n_field = {m}: "
                f"expected M*(M+1)/2 = {expected} unique cross-spectra."
            )
        for i, j in self.pair_idx:
            if not (0 <= i <= j < m):
                raise ValueError(
                    f"pair {(i, j)} out of range or not canonical (need 0 <= i <= j < {m})."
                )

    @property
    def n_spec(self) -> int:
        return len(self.pair_idx)

    @property
    def n_data(self) -> int:
        return self.n_spec * self.n_bins

    @classmethod
    def from_freq_pairs(cls, freq_pairs: list[tuple[int, int]], n_bins: int) -> SpectrumLayout:
        """Build from a ``SignalModel.freq_pairs`` list and a bin count.

        ``n_field`` is inferred as ``max channel index + 1`` (augr cross-spectra
        cover the full upper triangle, so this is the channel count).
        """
        pairs = tuple((int(i), int(j)) for i, j in freq_pairs)
        if not pairs:
            raise ValueError("freq_pairs is empty.")
        n_field = max(max(i, j) for i, j in pairs) + 1
        return cls(pair_idx=pairs, n_field=n_field, n_bins=int(n_bins))


def spectra_to_matrices(vec: jax.Array, layout: SpectrumLayout) -> jax.Array:
    """Flat data vector ``(n_data,)`` → per-bin symmetric matrices ``(M, M, n_bins)``.

    Inverse of :func:`matrices_to_spectra`. Descriptor-driven scatter: each
    spectrum ``s`` with pair ``(i, j)`` fills ``C[i, j, :]`` (and ``C[j, i, :]``
    for off-diagonal pairs). Assumes the augr ``spec``-slowest / ``bin``-fastest
    flattening, so ``vec.reshape(n_spec, n_bins)`` recovers the per-spectrum rows.
    """
    cl2d = vec.reshape(layout.n_spec, layout.n_bins)
    m, n_bins = layout.n_field, layout.n_bins
    c = jnp.zeros((m, m, n_bins), dtype=vec.dtype)
    for s, (i, j) in enumerate(layout.pair_idx):
        c = c.at[i, j].set(cl2d[s])
        if i != j:
            c = c.at[j, i].set(cl2d[s])
    return c


def matrices_to_spectra(mats: jax.Array, layout: SpectrumLayout) -> jax.Array:
    """Per-bin symmetric matrices ``(M, M, n_bins)`` → flat data vector ``(n_data,)``.

    Inverse of :func:`spectra_to_matrices`. Descriptor-driven gather: reads
    ``mats[i, j, :]`` for each pair ``(i, j)`` in ``pair_idx`` order, then
    flattens ``spec``-slowest / ``bin``-fastest.
    """
    rows = [mats[i, j] for (i, j) in layout.pair_idx]
    return jnp.stack(rows, axis=0).reshape(-1)
