"""augr inference layer: non-Gaussian likelihoods over binned cross-spectra.

A generic, experiment-agnostic likelihood/inference layer built around three
structural Protocols (``SpectrumModel`` / ``Likelihood`` / ``Prior``) and one
carrier (:class:`BinnedSpectra`). augr's Knox/Fisher path is the first consumer;
the same Protocols let bk-jax's BPCM path plug in later.

The Hamimeche-Lewis likelihood + a Gaussian/Knox baseline (both evaluable on a
``SignalModel`` prediction), the Gaussian prior + positivity bijector, the
log-posterior assembler, and a BlackJAX NUTS sampler (the sampler is behind the
``[sampling]`` extra; ``nuts`` imports blackjax lazily so the rest of the layer
needs no extra).
"""

from __future__ import annotations

from augr.likelihood.gaussian import GaussianLikelihood
from augr.likelihood.hl import HLLikelihood, hamimeche_lewis_likelihood
from augr.likelihood.mle import (
    MLEResult,
    MLESearchResult,
    make_dithered_starts,
    run_mle,
    run_mle_search,
)
from augr.likelihood.nuts import (
    chain_e_bfmi,
    chain_ess,
    chain_rhat,
    constrain,
    converged,
    diagnostics_summary,
    draw_fisher_init,
    draw_fisher_inits,
    marginal_sigma,
    run_nuts,
    run_nuts_chains,
)
from augr.likelihood.ordering import (
    SpectrumLayout,
    matrices_to_spectra,
    spectra_to_matrices,
)
from augr.likelihood.posterior import Posterior, make_log_posterior
from augr.likelihood.prior import (
    DEFAULT_POSITIVE_PARAMS,
    GaussianPrior,
    PositivityTransform,
)
from augr.likelihood.protocols import (
    BinnedSpectra,
    Likelihood,
    Prior,
    SignalSpectrumModel,
    SpectrumModel,
)

__all__ = [
    "DEFAULT_POSITIVE_PARAMS",
    "BinnedSpectra",
    "GaussianLikelihood",
    "GaussianPrior",
    "HLLikelihood",
    "Likelihood",
    "MLEResult",
    "MLESearchResult",
    "PositivityTransform",
    "Posterior",
    "Prior",
    "SignalSpectrumModel",
    "SpectrumLayout",
    "SpectrumModel",
    "chain_e_bfmi",
    "chain_ess",
    "chain_rhat",
    "constrain",
    "converged",
    "diagnostics_summary",
    "draw_fisher_init",
    "draw_fisher_inits",
    "hamimeche_lewis_likelihood",
    "make_dithered_starts",
    "make_log_posterior",
    "marginal_sigma",
    "matrices_to_spectra",
    "run_mle",
    "run_mle_search",
    "run_nuts",
    "run_nuts_chains",
    "spectra_to_matrices",
]
