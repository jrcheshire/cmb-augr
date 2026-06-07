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

Multi-chain sampling (:func:`run_nuts_chains`, started from the over-dispersed
:func:`draw_fisher_inits`) plus convergence diagnostics (:func:`chain_rhat`,
:func:`chain_ess`, :func:`chain_e_bfmi`, aggregated by
:func:`diagnostics_summary`) turn a single *indicative* run into a *checkable*
one: R-hat ≈ 1 across chains, healthy ESS, and E-BFMI ≳ 0.3 are the gates for
quoting a sampled σ(r). The diagnostic function names mirror
``bk_jax.likelihood.nuts`` so the eventual shared core (bk-jax importing this
layer, Phase 3) reconciles without renames.

blackjax is imported lazily inside :func:`run_nuts` / the diagnostics so
importing this module (and the package) does not require the ``[sampling]`` extra.
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


def draw_fisher_inits(
    fiducial_free: jax.Array,
    fisher_cov: jax.Array,
    transform,
    key: jax.Array,
    n_chains: int,
    *,
    scale: float = 1.0,
) -> jax.Array:
    """``n_chains`` unconstrained starts from ``N(fiducial_free, scale² · fisher_cov)``.

    Returns ``(n_chains, n_free)``. Defaults to ``scale=1`` (as the single-chain
    :func:`draw_fisher_init`): the ``n_chains`` *independent* multivariate draws
    already start in different ~1σ directions across the posterior bulk, which is
    enough between-chain dispersion for :func:`chain_rhat` to detect
    failure-to-mix (measured: 4 chains agree to R-hat(r) ≈ 1.00 from distinct
    starts).

    Do **not** over-disperse aggressively here. At ``scale=2`` the
    foreground-index directions (β_dust / β_sync exponents) put the start deep in
    the banana tail where NUTS freezes: every transition diverges, the chain never
    moves, ``Var(E)=0`` and E-BFMI is NaN (measured: 2 of 4 chains frozen at
    ``scale=2`` on the ℓ≤50 bump config; R-hat blows up to ~3). ``scale`` is left
    tunable for posteriors well-conditioned in every direction. Each draw maps
    through ``transform.inverse`` (flooring the non-negative slots) so an amplitude
    pushed ≤0 still yields a finite start, off the Asimov fiducial where the
    Hamimeche-Lewis gradient is defined.
    """
    chol = jnp.linalg.cholesky(fisher_cov)
    z = jax.random.normal(key, (n_chains, fiducial_free.shape[0]))
    x_init = fiducial_free[None, :] + scale * (z @ chol.T)  # row i = fiducial + scale·chol·z_i
    return jax.vmap(transform.inverse)(x_init)


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


def run_nuts_chains(
    log_prob,
    init_positions: jax.Array,
    key: jax.Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    target_acceptance_rate: float = 0.8,
):
    """Run ``n_chains`` independent NUTS chains, each with its own warmup.

    ``init_positions`` is ``(n_chains, n_free)`` (see :func:`draw_fisher_inits`).
    Returns ``(positions, info)`` where ``positions`` is
    ``(n_chains, num_samples, n_free)`` and ``info`` is the per-sample blackjax
    ``NUTSInfo`` stacked on a leading chain axis — i.e. the single-chain return of
    :func:`run_nuts` with one extra axis, so ``info.energy`` /
    ``info.is_divergent`` are ``(n_chains, num_samples)`` and feed the diagnostics
    directly.

    The chains run in a **Python loop**, not under ``jax.vmap``: NUTS expands each
    trajectory with ``jax.lax.while_loop``, so vmapped chains would advance in
    lockstep to the slowest-converging trajectory at every step (the well-known
    NUTS-vmap synchronization). At augr-scale posteriors a chain is seconds, so a
    serial loop over a handful of chains is cheaper than that synchronization
    overhead. (Mirrors ``bk_jax.likelihood.nuts.run_nuts_chains``.)

    Always gate the result through :func:`converged` (or inspect
    :func:`diagnostics_summary`): at hard configs ~1 in 10 fresh runs has a single
    chain freeze in the foreground-index banana (every transition diverges → NaN
    E-BFMI, inflated R-hat), which the gate catches; the fix is a different seed or
    proper priors on the weakly-constrained nuisance directions, not a silent drop.
    """
    if init_positions.ndim != 2:
        raise ValueError(
            f"init_positions must be 2-D (n_chains, n_free); got shape {init_positions.shape}"
        )
    n_chains = init_positions.shape[0]
    keys = jax.random.split(key, n_chains)

    positions_list = []
    info_list = []
    for c in range(n_chains):
        positions_c, info_c = run_nuts(
            log_prob,
            init_positions[c],
            keys[c],
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_acceptance_rate=target_acceptance_rate,
        )
        positions_list.append(positions_c)
        info_list.append(info_c)

    positions = jnp.stack(positions_list, axis=0)
    info = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *info_list)
    return positions, info


def constrain(positions: jax.Array, transform) -> jax.Array:
    """Map unconstrained draws to constrained parameters.

    Accepts any leading batch shape ``(..., n_free)`` (single-chain ``(n, n_free)``
    or multi-chain ``(n_chains, num_samples, n_free)``); ``transform.forward`` is
    applied over the flattened batch and the original shape is restored.
    """
    flat = positions.reshape(-1, positions.shape[-1])
    return jax.vmap(transform.forward)(flat).reshape(positions.shape)


def marginal_sigma(constrained: jax.Array, free_names, name: str) -> float:
    """Posterior std of a named free parameter, pooling any leading batch axes.

    Works for single-chain ``(n, n_free)`` and multi-chain
    ``(n_chains, num_samples, n_free)`` constrained draws alike.
    """
    flat = constrained.reshape(-1, constrained.shape[-1])
    return float(jnp.std(flat[:, list(free_names).index(name)]))


# ---------------------------------------------------------------------------
# Convergence diagnostics. Operate on the multi-chain outputs of
# :func:`run_nuts_chains`: ``positions`` (n_chains, num_samples, n_free) for the
# between/within-chain statistics, ``info.energy`` (n_chains, num_samples) for
# E-BFMI. Names mirror ``bk_jax.likelihood.nuts``.
# ---------------------------------------------------------------------------


def chain_rhat(positions: jax.Array) -> jax.Array:
    """Gelman-Rubin R-hat per parameter from ``(n_chains, num_samples, n_free)``.

    Wraps :func:`blackjax.diagnostics.potential_scale_reduction`. Returns
    ``(n_free,)``. R-hat ≈ 1 means the chains agree; the usual non-convergence
    flags are > 1.01 (Vehtari+ 2021) or the older > 1.05 (Stan). Needs ≥ 2
    over-dispersed chains to be meaningful (see :func:`draw_fisher_inits`).
    """
    import blackjax.diagnostics as bjd

    return bjd.potential_scale_reduction(positions, chain_axis=0, sample_axis=1)


def chain_ess(positions: jax.Array) -> jax.Array:
    """Cross-chain effective sample size per parameter, ``(n_free,)``.

    Wraps :func:`blackjax.diagnostics.effective_sample_size` over
    ``(n_chains, num_samples, n_free)`` draws. ESS as a fraction of the total
    ``n_chains · num_samples`` is the sampling efficiency; low ESS flags high
    autocorrelation (a marginal σ read off too few independent draws is noisy).
    """
    import blackjax.diagnostics as bjd

    return bjd.effective_sample_size(positions, chain_axis=0, sample_axis=1)


def chain_e_bfmi(energy: jax.Array) -> jax.Array:
    """Energy-Bayesian fraction of missing information per chain, ``(n_chains,)``.

    Betancourt 2016 (arXiv:1604.00695) diagnostic, from the HMC ``energy`` trace
    ``info.energy`` of shape ``(n_chains, num_samples)``:
    ``Var(ΔE) / Var(E)`` along the sample axis. E-BFMI ≲ 0.3 means the momentum
    resampling is not exploring the energy distribution efficiently — heavy tails
    relative to the mass matrix — so the chain mixes slowly even without
    divergences.
    """
    e = jnp.asarray(energy)
    return jnp.var(jnp.diff(e, axis=-1), axis=-1, ddof=1) / jnp.var(e, axis=-1, ddof=1)


def diagnostics_summary(positions: jax.Array, info, free_names) -> dict:
    """Aggregate the multi-chain diagnostics into a plain, printable dict.

    ``positions`` / ``info`` are the :func:`run_nuts_chains` outputs. Returns
    per-parameter ``r_hat`` and ``ess`` dicts, per-chain ``e_bfmi`` (list),
    ``n_divergent`` (summed over chains), and the chain/sample counts. Convenience
    summaries — ``r_hat_max``, ``e_bfmi_min`` — make a pass/fail gate a one-liner.
    """
    rhat = chain_rhat(positions)
    ess = chain_ess(positions)
    ebfmi = chain_e_bfmi(info.energy)
    names = list(free_names)
    n_chains, n_samples = positions.shape[0], positions.shape[1]
    return {
        "r_hat": {n: float(rhat[i]) for i, n in enumerate(names)},
        "ess": {n: float(ess[i]) for i, n in enumerate(names)},
        "e_bfmi": [float(x) for x in ebfmi],
        "n_divergent": int(jnp.sum(info.is_divergent)),
        "n_chains": int(n_chains),
        "n_samples": int(n_samples),
        "n_total": int(n_chains * n_samples),
        "r_hat_max": float(jnp.max(rhat)),
        "e_bfmi_min": float(jnp.min(ebfmi)),
    }


def converged(
    diag: dict,
    *,
    param: str | None = "r",
    rhat_threshold: float = 1.01,
    e_bfmi_threshold: float = 0.3,
    max_divergent_fraction: float = 0.01,
) -> bool:
    """Pass/fail the standard convergence gate from a :func:`diagnostics_summary` dict.

    Returns ``True`` only if all of the following hold:

    * **R-hat** below ``rhat_threshold`` (1.01, Vehtari+ 2021) for ``param`` — the
      science parameter whose σ is being quoted (default ``"r"``). Pass
      ``param=None`` to gate on the worst parameter (``r_hat_max``) instead, which
      additionally demands the weakly-constrained foreground nuisances have mixed
      (often ~1.05 even when ``r`` is fully converged, so reserve that for runs
      with proper priors on every direction).
    * **E-BFMI** above ``e_bfmi_threshold`` (0.3, Betancourt 2016) for *every*
      chain — a frozen chain has ``e_bfmi_min`` NaN and ``nan > 0.3`` is ``False``,
      so it fails here automatically.
    * **divergences** below ``max_divergent_fraction`` of the total transitions.

    This is the indicative-vs-publishable line: a σ read off a run that fails this
    gate should not be quoted.
    """
    rhat = diag["r_hat_max"] if param is None else diag["r_hat"][param]
    return bool(
        rhat < rhat_threshold
        and diag["e_bfmi_min"] > e_bfmi_threshold
        and diag["n_divergent"] < max_divergent_fraction * diag["n_total"]
    )
