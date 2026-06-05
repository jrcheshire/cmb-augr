"""prior.py — independent Gaussian priors + a positivity bijector for NUTS.

The forecast posterior (assembled in :mod:`augr.likelihood.posterior`) is
sampled over the *free* parameters, with fixed parameters held at the fiducial.
NUTS explores an unconstrained space, so each non-negative parameter is mapped
to ``(0, ∞)`` by a softplus bijector (identity for the rest), with the bijector
supplying the change-of-variables log-det-Jacobian.

:class:`GaussianPrior` reproduces ``FisherForecast``'s prior convention — a 1σ
Gaussian on the diagonal — as a sampled density: a parameter named in
``priors`` gets ``Normal(fiducial, σ)``; parameters without an entry are flat
(improper). ``r`` is intentionally left unconstrained by default, so the
sampled σ(r) is an apples-to-apples comparison with the (symmetric) Fisher
σ(r) — the Hamimeche-Lewis skew then shows up as a wider σ rather than being
hidden by a hard boundary; the physical non-negative amplitudes are bounded for
sampler stability.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

# Non-negative parameters mapped through softplus during sampling. ``r`` is
# deliberately absent (see module docstring): the σ(r)-vs-Fisher comparison is
# cleanest with r unconstrained.
DEFAULT_POSITIVE_PARAMS: frozenset[str] = frozenset({"A_lens", "A_dust", "A_sync", "A_res"})


def _inv_softplus(x: jax.Array) -> jax.Array:
    """Inverse of :func:`jax.nn.softplus`; ``log(expm1(x))`` written stably for ``x > 0``."""
    return x + jnp.log(-jnp.expm1(-x))


class PositivityTransform(eqx.Module):
    """Per-parameter softplus (where ``positive``) / identity bijector.

    ``forward`` maps the unconstrained sampling vector ``u`` to constrained
    parameters; ``forward_log_det_jacobian`` is the summed ``log|dx/du|``
    (softplus: ``log σ(u) = -softplus(-u)``; identity: 0). The ``positive`` mask
    is a constant boolean array carried as a (dynamic) leaf — it never changes
    within a run, so XLA folds it as a captured constant.
    """

    positive: jax.Array  # (n_free,) bool

    def forward(self, u: jax.Array) -> jax.Array:
        return jnp.where(self.positive, jax.nn.softplus(u), u)

    def inverse(self, x: jax.Array) -> jax.Array:
        # _inv_softplus needs x > 0; only used on positive slots (x there > 0).
        x_safe = jnp.where(self.positive, jnp.maximum(x, jnp.finfo(x.dtype).tiny), 1.0)
        return jnp.where(self.positive, _inv_softplus(x_safe), x)

    def forward_log_det_jacobian(self, u: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(self.positive, -jax.nn.softplus(-u), 0.0))

    @classmethod
    def from_names(
        cls, free_names: list[str], positive_params: frozenset[str] = DEFAULT_POSITIVE_PARAMS
    ) -> PositivityTransform:
        return cls(positive=jnp.array([n in positive_params for n in free_names]))


class GaussianPrior(eqx.Module):
    """Independent Gaussian priors over the free parameters (flat where absent).

    ``log_prob`` is ``-½ Σ_i prec_i (x_i - mean_i)²`` over the constrained free
    vector ``x``; a parameter with no prior carries ``prec_i = 0`` (flat).
    """

    means: jax.Array  # (n_free,) prior means (fiducial values)
    precisions: jax.Array  # (n_free,) 1/σ²; 0 = flat

    def log_prob(self, free_params: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(self.precisions * (free_params - self.means) ** 2)

    @classmethod
    def from_priors(
        cls, free_names: list[str], fiducial: dict[str, float], priors: dict[str, float]
    ) -> GaussianPrior:
        """Build from a fiducial dict (means) and a ``{name: σ}`` priors dict."""
        means = jnp.array([float(fiducial[n]) for n in free_names])
        precisions = jnp.array(
            [1.0 / float(priors[n]) ** 2 if n in priors else 0.0 for n in free_names]
        )
        return cls(means=means, precisions=precisions)
