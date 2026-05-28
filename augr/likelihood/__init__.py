"""augr inference layer: non-Gaussian likelihoods over binned cross-spectra.

A generic, experiment-agnostic likelihood/inference layer built around three
structural Protocols (``SpectrumModel`` / ``Likelihood`` / ``Prior``) and one
carrier (:class:`BinnedSpectra`). augr's Knox/Fisher path is the first consumer;
the same Protocols let bk-jax's BPCM path plug in later.

Phase A (current): the Hamimeche-Lewis likelihood + a Gaussian/Knox baseline,
both evaluable on a ``SignalModel`` prediction. Phase B adds the prior, the
log-posterior assembler, and a BlackJAX NUTS sampler (``[sampling]`` extra).
"""

from __future__ import annotations

from augr.likelihood.gaussian import GaussianLikelihood
from augr.likelihood.hl import HLLikelihood, hamimeche_lewis_likelihood
from augr.likelihood.ordering import (
    SpectrumLayout,
    matrices_to_spectra,
    spectra_to_matrices,
)
from augr.likelihood.protocols import (
    BinnedSpectra,
    Likelihood,
    SignalSpectrumModel,
    SpectrumModel,
)

__all__ = [
    "BinnedSpectra",
    "GaussianLikelihood",
    "HLLikelihood",
    "Likelihood",
    "SignalSpectrumModel",
    "SpectrumLayout",
    "SpectrumModel",
    "hamimeche_lewis_likelihood",
    "matrices_to_spectra",
    "spectra_to_matrices",
]
