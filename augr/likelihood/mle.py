"""mle.py — L-BFGS maximum-likelihood point estimate + dithered multistart.

Port of ``bk_jax.likelihood.mle`` (itself a port of MATLAB ``MLsearch.m``),
adapted to augr's :class:`~augr.likelihood.posterior.Posterior`:

* augr's ``log_prob(u)`` already works in the **unconstrained** free-parameter
  space (the bijector is folded into :func:`make_log_posterior`) and is a
  quantity to **maximize**, so :func:`run_mle` minimizes ``-log_prob(u)``
  internally and stores the maximized ``log_prob`` at the optimum. (bk-jax's
  ``_from_posterior`` convenience wrappers are therefore unnecessary —
  ``Posterior.log_prob`` is already the closed-over callable, so
  ``run_mle(post.log_prob, ...)`` is the equivalent.)
* :func:`run_mle_search` takes pre-built ``init_positions`` ``(n_starts, n_free)``
  — mirroring :func:`augr.likelihood.nuts.run_nuts_chains` — so it consumes
  either :func:`make_dithered_starts` or the existing
  :func:`~augr.likelihood.nuts.draw_fisher_inits`.

The MLE serves three roles: a robust NUTS init (dither around the located mode
instead of ~1σ draws at the fiducial, which occasionally freezes a chain in the
foreground-index banana — see :func:`~augr.likelihood.nuts.draw_fisher_inits`),
the foundation for profile-likelihood σ, and bk-jax parity ahead of the shared
inference core.

``optax`` is imported lazily inside :func:`run_mle` (behind the ``[sampling]``
extra), mirroring the lazy ``blackjax`` import in :mod:`augr.likelihood.nuts`, so
importing this module does not require the extra.

Note on Asimov data: the posterior peaks at the fiducial, so the MLE trivially
recovers it. For the Hamimeche-Lewis likelihood the gradient is NaN *exactly* at
the fiducial (degenerate per-bin ``eigh``; see :mod:`augr.likelihood.hl`), so an
HL :func:`run_mle` converges via the loss-change criterion with ``grad_norm``
reading NaN — expected, not a failure. The Gaussian likelihood is smooth there.
The MLE's value is for off-fiducial / real data, robust sampler inits (which sit
off the exact mode), and profile-σ.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class MLEResult(eqx.Module):
    """Result of a single L-BFGS optimization run.

    Attributes
    ----------
    x : ``(n_free,)``
        MLE point, in the unconstrained free-parameter ordering of
        ``Posterior.free_names`` (map back with
        :func:`augr.likelihood.nuts.constrain`).
    log_prob : scalar
        Maximized log-posterior at ``x`` (``= -final_loss``).
    grad_norm : scalar
        ``||∇log_prob(x)||_∞`` at ``x`` (sign-independent, so identical to the
        loss gradient norm). The primary convergence criterion; values
        ``<= g_tol`` indicate a clean local optimum. May be NaN at the exact
        Hamimeche-Lewis Asimov optimum (see module docstring).
    n_iter : scalar int
        Number of L-BFGS update steps taken (``<= max_iter``).
    converged : scalar bool
        True if ``grad_norm <= g_tol`` or
        ``|log_prob_change| <= f_tol * (1 + |log_prob|)`` triggered exit; False
        if ``max_iter`` was hit.
    """

    x: jax.Array
    log_prob: jax.Array
    grad_norm: jax.Array
    n_iter: jax.Array
    converged: jax.Array


def run_mle(
    log_prob: Callable[[jax.Array], jax.Array],
    init_position: jax.Array,
    *,
    max_iter: int = 200,
    g_tol: float = 1e-6,
    f_tol: float = 1e-12,
    memory_size: int = 10,
) -> MLEResult:
    """Maximize a scalar JAX-traceable ``log_prob`` via :func:`optax.lbfgs`.

    Minimizes ``loss = -log_prob`` over the unconstrained free vector. Uses
    ``optax.value_and_grad_from_state`` so the per-iteration value and gradient
    are reused from the line search rather than recomputed; the default
    ``optax.lbfgs`` includes a zoom (Hager–Zhang style) line search robust on
    non-quadratic objectives.

    Termination is tested at the START of each iteration on the value/gradient at
    the *current* iterate. The loop exits when either

    - ``||∇loss(x)||_∞ <= g_tol`` (gradient near zero), or
    - ``|loss(x) - loss(x_prev)| <= f_tol * (1 + |loss(x)|)`` (small relative
      change), or
    - ``n_iter >= max_iter``.

    Parameters
    ----------
    log_prob
        Scalar log-posterior of a flat unconstrained free vector, e.g.
        ``Posterior.log_prob``. Must be ``jax.value_and_grad``-able.
    init_position
        Initial unconstrained free vector ``(n_free,)``.
    max_iter
        Hard upper bound on L-BFGS update steps.
    g_tol
        Gradient infinity-norm tolerance.
    f_tol
        Relative loss-change tolerance (relative to ``1 + |loss|`` so it also
        bounds absolute change near zero).
    memory_size
        L-BFGS history length (number of past (s, y) pairs retained).

    Returns
    -------
    :class:`MLEResult`. A JAX pytree (equinox Module); safe to ``vmap`` over and
    to return from a ``jax.jit``-ed function.
    """
    import optax

    def loss_fn(u: jax.Array) -> jax.Array:
        return -log_prob(u)

    opt = optax.lbfgs(memory_size=memory_size)
    val_grad = optax.value_and_grad_from_state(loss_fn)

    init_state = opt.init(init_position)
    init_loss, init_grad = val_grad(init_position, state=init_state)
    init_gnorm = jnp.linalg.norm(init_grad, ord=jnp.inf)
    init_converged = init_gnorm <= g_tol

    # Carry: (x, opt_state, loss, grad_norm, prev_loss_for_ftol, n_iter, converged).
    # The carry stays in loss space to mirror the MATLAB-validated bk-jax
    # reference exactly; we negate only at the boundaries.
    init_carry = (
        init_position,
        init_state,
        init_loss,
        init_gnorm,
        jnp.asarray(jnp.inf, dtype=init_loss.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        init_converged,
    )

    def cond(carry):
        _x, _state, _loss, _gnorm, _prev_loss, n_iter, converged = carry
        return jnp.logical_and(n_iter < max_iter, jnp.logical_not(converged))

    def body(carry):
        x, state, loss, _gnorm, _prev_loss, n_iter, _converged = carry
        _val, grad = val_grad(x, state=state)
        update, new_state = opt.update(
            grad,
            state,
            x,
            value=loss,
            grad=grad,
            value_fn=loss_fn,
        )
        x_new = optax.apply_updates(x, update)
        new_loss, new_grad = val_grad(x_new, state=new_state)
        new_gnorm = jnp.linalg.norm(new_grad, ord=jnp.inf)
        df = jnp.abs(new_loss - loss)
        rel_floor = 1.0 + jnp.abs(new_loss)
        converged_g = new_gnorm <= g_tol
        converged_f = df <= f_tol * rel_floor
        converged = jnp.logical_or(converged_g, converged_f)
        return (x_new, new_state, new_loss, new_gnorm, loss, n_iter + 1, converged)

    final = jax.lax.while_loop(cond, body, init_carry)
    x_f, _state_f, loss_f, gnorm_f, _prev_f, n_iter_f, converged_f = final
    return MLEResult(
        x=x_f,
        log_prob=-loss_f,
        grad_norm=gnorm_f,
        n_iter=n_iter_f,
        converged=converged_f,
    )


def make_dithered_starts(
    center: jax.Array,
    dither: jax.Array,
    n_starts: int,
    key: jax.Array,
) -> jax.Array:
    """Generate ``n_starts`` dithered starting points around ``center``.

    ``x0[j] = center + N(0, dither**2)`` (the ``MLsearch.m:464`` recipe), one
    independent draw per start. Operates in the **unconstrained** free space:
    ``center`` is unconstrained (e.g. ``mle.best.x`` or
    ``Posterior.fiducial_unconstrained(transform)``) and ``dither`` is per-param
    unconstrained widths (e.g. ``sqrt(diag(fisher_cov))`` under the default
    identity transform). Reproducible from the JAX ``key``.

    Returns ``(n_starts, n_free)``.
    """
    if center.shape != dither.shape:
        raise ValueError(f"center.shape {center.shape} != dither.shape {dither.shape}")
    if center.ndim != 1:
        raise ValueError(f"center must be 1-D, got shape {center.shape}")
    n_free = center.shape[0]
    eps = jax.random.normal(key, shape=(n_starts, n_free), dtype=center.dtype)
    return center[None, :] + eps * dither[None, :]


class MLESearchResult(eqx.Module):
    """Multi-start dithered MLE result, plus a best-of-n selection.

    Attributes
    ----------
    best : :class:`MLEResult`
        The start whose optimization reached the highest ``log_prob``. Use
        ``best.x`` for the MLE estimate.
    all_x : ``(n_starts, n_free)``
        End-point per start.
    all_log_prob : ``(n_starts,)``
        Maximized log-posterior per start.
    all_grad_norm : ``(n_starts,)``
        Final ``||grad||_∞`` per start.
    all_n_iter : ``(n_starts,)``
        Iteration count per start.
    all_converged : ``(n_starts,)``
        Convergence flag per start.
    init_positions : ``(n_starts, n_free)``
        The starting points fed in.
    """

    best: MLEResult
    all_x: jax.Array
    all_log_prob: jax.Array
    all_grad_norm: jax.Array
    all_n_iter: jax.Array
    all_converged: jax.Array
    init_positions: jax.Array


def run_mle_search(
    log_prob: Callable[[jax.Array], jax.Array],
    init_positions: jax.Array,
    *,
    max_iter: int = 200,
    g_tol: float = 1e-6,
    f_tol: float = 1e-12,
    memory_size: int = 10,
) -> MLESearchResult:
    """Multi-start dithered MLE: maximize ``log_prob`` from each start, keep the best.

    ``init_positions`` is ``(n_starts, n_free)`` (see :func:`make_dithered_starts`
    or :func:`augr.likelihood.nuts.draw_fisher_inits`). The best-of-n is selected
    by ``nanargmax(all_log_prob)`` (NaN-safe; see Notes) regardless of per-start
    ``converged`` flags (non-convergence is a downstream diagnostic, not a filter
    — matches MATLAB MLsearch).

    The starts run in a **Python loop**, not under ``jax.vmap``: each
    :func:`run_mle` is a ``jax.lax.while_loop`` whose vmapped iterates would sync
    to the slowest-converging start (the same synchronization that makes
    :func:`augr.likelihood.nuts.run_nuts_chains` serial). For batched production
    runs, ``vmap`` over the outer (e.g. sim) axis instead.

    Returns a :class:`MLESearchResult`.

    Notes
    -----
    Under the default identity transform augr's foreground amplitudes are
    unbounded, and the BK15 dust-sync cross-spectrum takes ``sqrt`` of the
    amplitudes — so a start (or L-BFGS line search) that wanders to a negative
    amplitude yields a NaN ``log_prob`` that poisons *that* run. The best-of-n
    therefore uses ``nanargmax`` to keep the best *finite* mode, which makes the
    multistart robust as long as at least one start stays in the valid region
    (use modest dither, e.g. ``draw_fisher_inits(..., scale=0.3)``, so most do).
    bk-jax's equivalent uses plain ``argmin`` because its priors have bounded
    support and never reach an invalid region. To remove the NaN regions entirely
    (rather than dodge them), bound the amplitudes via
    :class:`~augr.likelihood.prior.PositivityTransform`, at some cost in
    optimizer conditioning from the softplus geometry.
    """
    if init_positions.ndim != 2:
        raise ValueError(
            f"init_positions must be 2-D (n_starts, n_free); got shape {init_positions.shape}"
        )
    n_starts = init_positions.shape[0]

    results: list[MLEResult] = []
    for j in range(n_starts):
        results.append(
            run_mle(
                log_prob,
                init_positions[j],
                max_iter=max_iter,
                g_tol=g_tol,
                f_tol=f_tol,
                memory_size=memory_size,
            )
        )

    all_x = jnp.stack([r.x for r in results], axis=0)
    all_log_prob = jnp.stack([r.log_prob for r in results], axis=0)
    all_grad_norm = jnp.stack([r.grad_norm for r in results], axis=0)
    all_n_iter = jnp.stack([r.n_iter for r in results], axis=0)
    all_converged = jnp.stack([r.converged for r in results], axis=0)

    # NaN-safe: a start that wandered into an invalid region (negative foreground
    # amplitude -> sqrt -> NaN log_prob) is skipped; keep the best finite mode.
    best_idx = jnp.nanargmax(all_log_prob)
    best = MLEResult(
        x=all_x[best_idx],
        log_prob=all_log_prob[best_idx],
        grad_norm=all_grad_norm[best_idx],
        n_iter=all_n_iter[best_idx],
        converged=all_converged[best_idx],
    )

    return MLESearchResult(
        best=best,
        all_x=all_x,
        all_log_prob=all_log_prob,
        all_grad_norm=all_grad_norm,
        all_n_iter=all_n_iter,
        all_converged=all_converged,
        init_positions=init_positions,
    )
