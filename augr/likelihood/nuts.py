"""nuts.py — BlackJAX window-adaptation + NUTS over an augr log-posterior.

Behind the optional ``[sampling]`` extra (blackjax). Samples the *unconstrained*
free-parameter posterior built by :func:`augr.likelihood.posterior.make_log_posterior`,
returning the unconstrained draws; :func:`constrain` maps them back to parameter
space and :func:`marginal_sigma` reads off a parameter's posterior width.

The sampler MUST start off the Asimov fiducial — the Hamimeche-Lewis gradient is
NaN there (degenerate per-bin ``eigh``; see :mod:`augr.likelihood.hl`).
:func:`draw_fisher_init` draws a well-scaled start from the Fisher Gaussian
approximation, which sits inside the posterior's typical set (a plain
"fiducial + 1σ in every parameter" lands deep in the tail because the marginal
σ's overshoot the well-constrained directions when applied jointly).

blackjax is imported lazily inside :func:`run_nuts` so importing this module
(and the package) does not require the ``[sampling]`` extra.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def draw_fisher_init(
    fiducial_free: jax.Array,
    fisher_cov: jax.Array,
    transform,
    key: jax.Array,
    scale: float = 1.0,
) -> jax.Array:
    """One unconstrained init drawn from ``N(fiducial_free, scale² · fisher_cov)``.

    ``fiducial_free`` and ``fisher_cov`` are in free-parameter order
    (``Posterior.free_names``). The constrained draw is mapped to unconstrained
    space by ``transform.inverse`` (which floors non-negative parameters, so a
    draw that pushes an amplitude to ≤0 still yields a finite start).
    """
    chol = jnp.linalg.cholesky(fisher_cov)
    z = jax.random.normal(key, (fiducial_free.shape[0],))
    x_init = fiducial_free + scale * (chol @ z)
    return transform.inverse(x_init)


def run_nuts(
    log_prob,
    init_position: jax.Array,
    key: jax.Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    target_acceptance_rate: float = 0.8,
):
    """Window-adapted NUTS over ``log_prob``.

    Returns ``(positions, info)`` where ``positions`` is ``(num_samples, n_free)``
    unconstrained draws and ``info`` is the per-sample blackjax ``NUTSInfo``
    (acceptance rate, divergences, ...).
    """
    import blackjax

    key, warmup_key, sample_key = jax.random.split(key, 3)
    warmup = blackjax.window_adaptation(
        blackjax.nuts, log_prob, target_acceptance_rate=target_acceptance_rate
    )
    (state, parameters), _ = warmup.run(warmup_key, init_position, num_steps=num_warmup)
    step = blackjax.nuts(log_prob, **parameters).step

    @jax.jit
    def one_step(carry, rng):
        carry, info = step(rng, carry)
        return carry, (carry.position, info)

    keys = jax.random.split(sample_key, num_samples)
    _, (positions, info) = jax.lax.scan(one_step, state, keys)
    return positions, info


def constrain(positions: jax.Array, transform) -> jax.Array:
    """Map unconstrained draws ``(n, n_free)`` to constrained parameters."""
    return jax.vmap(transform.forward)(positions)


def marginal_sigma(constrained: jax.Array, free_names, name: str) -> float:
    """Posterior std of a named free parameter from constrained draws."""
    return float(jnp.std(constrained[:, list(free_names).index(name)]))
