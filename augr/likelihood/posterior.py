"""posterior.py — assemble (model, likelihood, prior, transform) into a log-posterior.

:func:`make_log_posterior` returns a :class:`Posterior` whose ``log_prob(u)``
maps an *unconstrained* free-parameter vector ``u`` to the total log-posterior

    log L(model.predict(full)) + log π(x) + log|dx/du|,

where ``x = transform.forward(u)`` are the constrained free parameters, ``full``
scatters them into the fiducial vector (fixed parameters held at the fiducial),
and the final term is the bijector's change-of-variables Jacobian. Parameter
packing reuses ``FisherForecast``'s free/fixed-index convention: free names in
``model.parameter_names`` order, fixed parameters held at their fiducial values.

The forecast "data" is Asimov (the fiducial model), so the posterior peaks at
the fiducial. A NUTS sampler must be initialised *off* the fiducial — see
:func:`Posterior.fiducial_unconstrained` and the
:mod:`augr.likelihood.nuts` init helper — because ``jax.grad`` of the
Hamimeche-Lewis ``log_prob`` is NaN exactly at the Asimov point (degenerate
per-bin ``eigh``; documented in :mod:`augr.likelihood.hl`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Posterior:
    """A log-posterior over the unconstrained free-parameter vector.

    Attributes
    ----------
    log_prob
        ``(u: (n_free,)) -> scalar``; the total log-posterior in unconstrained
        space (ready for a NUTS sampler).
    free_names
        Free-parameter names, in ``model.parameter_names`` order.
    free_idx
        Positions of the free parameters within the full parameter vector.
    fiducial_full
        The full fiducial parameter vector ``(n_all,)``.
    """

    log_prob: Callable[[jax.Array], jax.Array]
    free_names: tuple[str, ...]
    free_idx: jax.Array
    fiducial_full: jax.Array

    def fiducial_unconstrained(self, transform) -> jax.Array:
        """The fiducial free-vector mapped to unconstrained space (sampler reference)."""
        return transform.inverse(self.fiducial_full[self.free_idx])


def make_log_posterior(
    model,
    likelihood,
    prior,
    transform,
    *,
    fiducial: dict[str, float],
    fixed: tuple[str, ...] | list[str] = (),
) -> Posterior:
    """Assemble a :class:`Posterior` over the free parameters.

    Parameters
    ----------
    model
        A :class:`~augr.likelihood.protocols.SpectrumModel` (``parameter_names``,
        ``predict(full_params) -> BinnedSpectra``).
    likelihood
        A :class:`~augr.likelihood.protocols.Likelihood` built at the fiducial
        (e.g. ``HLLikelihood.from_forecast`` / ``GaussianLikelihood.from_forecast``).
    prior
        A :class:`~augr.likelihood.protocols.Prior` over the constrained free vector.
    transform
        A bijector (e.g. :class:`~augr.likelihood.prior.PositivityTransform`)
        mapping unconstrained ``u`` to constrained free parameters.
    fiducial
        Fiducial parameter dict; supplies the held-fixed values and the means.
    fixed
        Parameter names to hold fixed (not sampled), matching ``FisherForecast``.
    """
    names = list(model.parameter_names)
    fixed_set = set(fixed)
    free_names = tuple(n for n in names if n not in fixed_set)
    free_idx = jnp.array([names.index(n) for n in free_names])
    fid_full = jnp.array([float(fiducial[n]) for n in names])

    def log_prob(u: jax.Array) -> jax.Array:
        x = transform.forward(u)
        full = fid_full.at[free_idx].set(x)
        prediction = model.predict(full)
        return (
            likelihood.log_prob(prediction)
            + prior.log_prob(x)
            + transform.forward_log_det_jacobian(u)
        )

    return Posterior(
        log_prob=log_prob,
        free_names=free_names,
        free_idx=free_idx,
        fiducial_full=fid_full,
    )
