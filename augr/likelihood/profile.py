"""profile.py — post-MLE parameter errors: Hessian-at-MLE + profile-likelihood σ.

Port of bk-jax's `likelihood/fisher.py`, adapted to augr. Named ``profile`` (not
``fisher``) to avoid clashing with the top-level :mod:`augr.fisher`
(:class:`~augr.fisher.FisherForecast`).

Two σ estimates at a located maximum:

* :func:`compute_fisher_at_mle` — the **curvature** σ: Hessian of ``-log_prob`` at
  the MLE, ``σ = sqrt(diag(inv(H)))``. The Gaussian/Laplace approximation. For the
  Gaussian likelihood at the Asimov fiducial this equals the Knox
  :class:`~augr.fisher.FisherForecast`. **Caveat:** for the Hamimeche-Lewis
  likelihood the Hessian is NaN *exactly* at the Asimov fiducial (degenerate per-bin
  ``eigh``; see :mod:`augr.likelihood.hl`); the HL MLE lands arbitrarily close, so
  the Hessian-at-MLE is unreliable there (NaN or near-degenerate, occasionally a
  finite value ≈ Knox) — use ``FisherForecast`` as the HL curvature baseline instead.

* :func:`compute_profile_sigma` — the **non-Gaussian-robust** σ: scan the target
  parameter on a grid, profile out (re-minimize) the others at each grid point, and
  fit a parabola to the ``Δ(-log P)`` curve. Uses log_prob *values* only (finite even
  at the HL Asimov point), so it works for HL where the Hessian path cannot. The
  headline use: profile σ(r) vs ``FisherForecast`` σ(r) measures the same
  reionization-bump non-Gaussian widening as HL-NUTS, without sampling.

Both are generic over a scalar ``log_prob(u)`` of the *unconstrained* free vector
(the same surface :class:`~augr.likelihood.posterior.Posterior` exposes), so they are
unit-testable against synthetic quadratics. The profile reuses
:func:`augr.likelihood.mle.run_mle_search` (nan-safe best-of-n) for the inner refits.

Spaces: the curvature Hessian and the inner refits live in the posterior's
*unconstrained* space; under the default identity transform that equals physical
space, so the reported σ is physical (in particular for ``r``, which is always
unconstrained). Under a softplus-bounded transform a Jacobian factor would be needed
for the bounded params — not applied (the default config is unbounded).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from augr.likelihood.mle import make_dithered_starts, run_mle_search


class FisherAtMLE(eqx.Module):
    """Curvature (Laplace) Fisher information = Hessian of ``-log_prob`` at the MLE.

    Attributes
    ----------
    hessian : ``(n_free, n_free)``
        Hessian of ``-log_prob`` at ``x`` = Fisher information (unconstrained space).
    cov : ``(n_free, n_free)``
        ``inv(hessian)``.
    sigmas : ``(n_free,)``
        ``sqrt(diag(cov))`` — per-parameter curvature σ (physical under the default
        identity transform).
    gradient : ``(n_free,)``
        ``∇(-log_prob)`` at ``x``; ≈ 0 at a converged MLE (a diagnostic — large
        entries flag a stale point).
    eigvals : ``(n_free,)``
        Ascending eigenvalues of ``hessian`` (PSD check + condition number).
    condition_number : scalar
        ``max(eigvals) / max(min(eigvals), 1e-30)``.
    x : ``(n_free,)``
        The MLE point (unconstrained), as supplied.
    free_names : ``tuple[str, ...]`` (static)
        Parameter names in free-vector order.
    """

    hessian: jax.Array
    cov: jax.Array
    sigmas: jax.Array
    gradient: jax.Array
    eigvals: jax.Array
    condition_number: jax.Array
    x: jax.Array
    free_names: tuple[str, ...] = eqx.field(static=True)


def compute_fisher_at_mle(
    log_prob: Callable[[jax.Array], jax.Array],
    x_mle: jax.Array,
    free_names: Sequence[str],
) -> FisherAtMLE:
    """Curvature Fisher = Hessian of ``-log_prob`` at ``x_mle``.

    Generic over any scalar, twice-differentiable ``log_prob`` of the unconstrained
    free vector (e.g. ``Posterior.log_prob``). For the Gaussian likelihood at the
    Asimov fiducial the result matches the Knox :class:`~augr.fisher.FisherForecast`;
    for Hamimeche-Lewis the Hessian is unreliable near the Asimov point (NaN exactly
    at the fiducial; use ``FisherForecast`` there instead — see module docstring).

    Returns a :class:`FisherAtMLE`.
    """
    n_free = int(x_mle.shape[0])
    if len(free_names) != n_free:
        raise ValueError(f"len(free_names) ({len(free_names)}) != x_mle.shape[0] ({n_free})")

    def neg(u: jax.Array) -> jax.Array:
        return -log_prob(u)

    gradient = jax.grad(neg)(x_mle)
    hessian = jax.hessian(neg)(x_mle)
    cov = jnp.linalg.inv(hessian)
    sigmas = jnp.sqrt(jnp.diag(cov))
    eigvals = jnp.linalg.eigvalsh(hessian)
    condition_number = jnp.max(eigvals) / jnp.maximum(
        jnp.min(eigvals), jnp.asarray(1e-30, dtype=eigvals.dtype)
    )
    return FisherAtMLE(
        hessian=hessian,
        cov=cov,
        sigmas=sigmas,
        gradient=gradient,
        eigvals=eigvals,
        condition_number=condition_number,
        x=x_mle,
        free_names=tuple(free_names),
    )


def compute_profile_sigma(
    log_prob: Callable[[jax.Array], jax.Array],
    free_names: Sequence[str],
    target: str,
    x_mle: jax.Array,
    fisher_sigma: jax.Array,
    transform,
    *,
    n_grid: int = 15,
    n_search: int = 3,
    n_sigma: float = 3.0,
    inner_scale: float = 0.3,
    key: jax.Array | None = None,
    max_iter: int = 200,
    g_tol: float = 1e-6,
    return_curve: bool = False,
) -> float | tuple[float, dict[str, np.ndarray | float]]:
    """Profile-likelihood σ for one free parameter.

    Scans ``target`` on a grid spanning ``±n_sigma · fisher_sigma[target]`` (in
    *physical* units), profiles out the other free parameters at each grid point via
    :func:`augr.likelihood.mle.run_mle_search`, and fits a parabola to the
    ``Δ(-log P)`` curve to read off σ. Non-Gaussian-robust (uses log_prob values, not
    the Hessian), so it works for the Hamimeche-Lewis likelihood where the
    Hessian-at-MLE is NaN.

    Parameters
    ----------
    log_prob
        Scalar log-posterior of the unconstrained free vector (``Posterior.log_prob``).
    free_names
        Parameter names in free-vector order.
    target
        Name of the parameter to profile (must be in ``free_names``).
    x_mle
        ``(n_free,)`` unconstrained MLE (e.g. ``mle.best.x``); grid centre + the
        starting point for the inner refits.
    fisher_sigma
        ``(n_free,)`` marginal σ per free parameter (e.g. ``sqrt(diag(fisher_cov))``);
        sets the grid half-width (``n_sigma·σ_target``) and the inner dither
        (``inner_scale·σ`` on the other params).
    transform
        The (elementwise) bijector mapping unconstrained ↔ physical (e.g.
        :class:`~augr.likelihood.prior.PositivityTransform`); identity by default.
    n_grid, n_search, n_sigma, inner_scale
        Grid resolution, inner dithers per grid point, grid half-width in σ, and the
        inner-dither fraction of σ.
    key
        PRNGKey for the per-grid-point dithers (default ``PRNGKey(0)``); each grid
        point uses ``jax.random.fold_in(key, i)``.
    max_iter, g_tol
        Forwarded to the inner :func:`run_mle`.
    return_curve
        If True, also return a dict ``{grid, logp_profile, fit_window_mask, x0,
        half_width}`` for plotting/diagnostics.

    Returns
    -------
    σ : float
        Profile-likelihood width estimate (physical units of ``target``).
    curve : dict (only if ``return_curve``)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    names = list(free_names)
    n_free = int(x_mle.shape[0])
    if len(names) != n_free:
        raise ValueError(f"len(free_names) ({len(names)}) != x_mle.shape[0] ({n_free})")
    if target not in names:
        raise ValueError(f"target {target!r} not in free_names {names}")
    t = names.index(target)
    other_idx = [i for i in range(n_free) if i != t]
    other_idx_arr = jnp.asarray(other_idx)

    x_mle_phys = transform.forward(x_mle)
    v0 = float(x_mle_phys[t])
    half_width = float(n_sigma * float(fisher_sigma[t]))
    grid = np.linspace(v0 - half_width, v0 + half_width, n_grid)

    u_red0 = x_mle[other_idx_arr]
    dither_red = inner_scale * fisher_sigma[other_idx_arr]

    logp_profile = np.empty(n_grid, dtype=np.float64)
    for i, v in enumerate(grid):
        # Fixed unconstrained component for this grid value (elementwise transform).
        u_t = transform.inverse(x_mle_phys.at[t].set(v))[t]

        def reduced(u_red: jax.Array, _u_t: jax.Array = u_t) -> jax.Array:
            u_full = (
                jnp.zeros(n_free, dtype=x_mle.dtype).at[t].set(_u_t).at[other_idx_arr].set(u_red)
            )
            return log_prob(u_full)

        inits = make_dithered_starts(u_red0, dither_red, n_search, jax.random.fold_in(key, i))
        res = run_mle_search(reduced, inits, max_iter=max_iter, g_tol=g_tol)
        logp_profile[i] = float(res.best.log_prob)

    peak = float(np.nanmax(logp_profile))
    delta = peak - logp_profile  # Δ(-log P) ≥ 0
    dx = grid - v0

    window = np.isfinite(delta) & (delta < 2.0)
    if int(window.sum()) < 3:
        window = np.isfinite(delta)

    dxw = dx[window]
    d = delta[window]
    a_num = float(np.sum(d * dxw**2))
    a_den = float(np.sum(dxw**4))
    if a_den < 1e-30 or a_num < 1e-30:
        # Degenerate parabola → half-width at Δ = 0.5 (average both sides).
        right = np.where((grid > v0) & (delta > 0.5))[0]
        left = np.where((grid < v0) & (delta > 0.5))[0]
        hw_r = (grid[right[0]] - v0) if len(right) else half_width
        hw_l = (v0 - grid[left[-1]]) if len(left) else half_width
        sigma = float(0.5 * (hw_r + hw_l))
    else:
        a = a_num / a_den  # Δ = a·dx²  ⇒  a = 1/(2σ²)
        sigma = float(1.0 / np.sqrt(2.0 * a))

    if return_curve:
        curve: dict[str, np.ndarray | float] = {
            "grid": grid,
            "logp_profile": logp_profile,
            "fit_window_mask": window,
            "x0": v0,
            "half_width": half_width,
        }
        return sigma, curve
    return sigma
