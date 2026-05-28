"""Inference-layer Protocols + the :class:`BinnedSpectra` carrier.

equinox enters augr **here, and (for now) only here**: :class:`BinnedSpectra`
is an ``eqx.Module`` so its ``cl`` array is a differentiable pytree leaf while
the :class:`~augr.likelihood.ordering.SpectrumLayout` descriptor rides along as
static metadata. The existing frozen-dataclass config types (``Instrument``,
``Channel``, ``TelescopeDesign``, ``CMBSpectra``) are unchanged — they remain
hashable static config; ``eqx.Module`` is reserved for structured *traced* state.

The three Protocols are structural (duck-typed), mirroring
``augr.foregrounds``'s pluggable-model pattern: anything with the right methods
qualifies. They never name augr- or bk-jax-internal classes, so both augr's
Knox/Fisher path and (later) bk-jax's BPCM path are first-class consumers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax

from augr.likelihood.ordering import SpectrumLayout, spectra_to_matrices


class BinnedSpectra(eqx.Module):
    """Binned cross-spectra plus the layout needed to interpret them.

    One carrier, two views: :meth:`as_vector` (flat ``(n_data,)``, what the
    Gaussian/Knox likelihood wants) and :meth:`as_bin_matrices` (per-bin
    symmetric ``(n_field, n_field, n_bins)``, what Hamimeche-Lewis wants), both
    derived from the same static ``layout``.

    ``cl`` is the traced leaf (flows through ``jax.grad`` / ``jit``); ``layout``
    is static metadata.
    """

    cl: jax.Array
    layout: SpectrumLayout = eqx.field(static=True)

    def as_vector(self) -> jax.Array:
        """Flat data vector ``(n_data,)``, ``spec``-slowest / ``bin``-fastest."""
        return self.cl

    def as_bin_matrices(self) -> jax.Array:
        """Per-bin symmetric matrices ``(n_field, n_field, n_bins)``."""
        return spectra_to_matrices(self.cl, self.layout)


@runtime_checkable
class SpectrumModel(Protocol):
    """params → predicted :class:`BinnedSpectra`. The forward model."""

    @property
    def parameter_names(self) -> list[str]: ...

    def predict(self, params: jax.Array) -> BinnedSpectra: ...


@runtime_checkable
class Likelihood(Protocol):
    """predicted :class:`BinnedSpectra` → scalar log-likelihood.

    Implementations own the data + covariance/prep (built once at the
    fiducial); only the prediction varies per call.
    """

    def log_prob(self, prediction: BinnedSpectra) -> jax.Array: ...


class SignalSpectrumModel:
    """Adapter wrapping an augr ``SignalModel`` as a :class:`SpectrumModel`.

    Kept in the likelihood layer (not as a ``SignalModel`` method) so the
    dependency direction stays one-way: the likelihood layer depends on
    ``SignalModel``, never the reverse. Duck-typed — any object exposing
    ``freq_pairs``, ``n_bins``, ``parameter_names`` and ``data_vector(params)``
    works, so the same wrapper serves a future bk-jax model.
    """

    def __init__(self, signal_model) -> None:
        self._signal = signal_model
        self._layout = SpectrumLayout.from_freq_pairs(signal_model.freq_pairs, signal_model.n_bins)

    @property
    def parameter_names(self) -> list[str]:
        return self._signal.parameter_names

    @property
    def layout(self) -> SpectrumLayout:
        return self._layout

    def predict(self, params: jax.Array) -> BinnedSpectra:
        return BinnedSpectra(cl=self._signal.data_vector(params), layout=self._layout)
