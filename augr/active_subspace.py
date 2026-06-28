"""active_subspace.py -- Constantine active subspaces over instrument design knobs.

Given a scalar design objective ``f(xi)`` (e.g. ``sigma(r)`` or the Gaussian EIG through the
differentiable map-based forecast) and its gradient w.r.t. the design vector ``xi``, the
**active subspace** (Constantine 2015) is the dominant eigenspace of

    ``C = E_xi[ grad f(xi) grad f(xi)^T ]``

evaluated over a sample of designs. Its eigenvalues are an *energy spectrum* -- a sharp gap
after the first few says the design problem is effectively low-dimensional -- and the leading
eigenvectors are interpretable **design directions** (which combinations of knobs move the QoI
most). This module is the cheap, low-dimensional half of the HL-EIG capstone: the subspace is
built from the cheap, validated Gaussian-EIG / ``sigma(r)`` gradient (see
:func:`augr.optimize_mapbased.sigma_r_from_noise_design`, :func:`augr.eig.design_objective`),
and the expensive non-Gaussian HL-EIG (:func:`augr.eig.hl_eig_from_external_cov`) is then
*evaluated* only along the 1--3 surviving directions (the driver,
``scripts/active_subspace_hl_eig.py``).

**Standardization.** Design knobs are heterogeneous in scale (n_det ~ 100s, NET ~ 10s,
beam_fwhm ~ arcmin, mission_years ~ few), so ``C`` must be formed in a dimensionless space or
it is dominated by units. :class:`DesignSpec` provides a log-scale (multiplicative) standard
coordinate ``z = log(xi / xi_fid)`` -- dimensionless, O(1) across knobs, keeps every sampled
design physically positive, and (via the chain rule through ``unstandardize``) makes
``jax.grad`` of ``loss(design_pytree(z), ctx)`` return the gradient already in ``z``-space.
Build the objective on ``z`` with :func:`augr.design_opt.build_design_objectives`, then feed
its value-and-grad to :func:`collect_gradients`.

**MC noise.** When the objective routes through the cut-sky MC forward the gradient carries
sample-covariance noise; un-averaged, that noise enters ``C`` quadratically and inflates the
diagonal (fake high dimensionality). :func:`collect_gradients` averages the gradient over
``n_crn`` disjoint CRN ensembles per design before forming ``C``; pair with the
resultant-length / CoV pre-flight in ``scripts/mapbased_grad_characterization.py``.

At design dimensionality (D ~ 13--37) ``C`` is a small dense matrix and ``np.linalg.eigh`` is
the right eigensolve (full, exact). The HVP/Krylov machinery for D ~ millions (the
likelihood-informed-subspace arc) is a separate, future concern.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DesignSpec:
    """Fiducial design + standardization between a flat ``z`` vector and the loss's pytree.

    ``unravel`` maps a flat ``(D,)`` raw-unit design back to the pytree the objective consumes
    (build it with ``jax.flatten_util.ravel_pytree`` on the fiducial design pytree -- see
    :meth:`from_pytree`). ``mode`` is ``"log"`` (multiplicative; the default, keeps designs
    positive) or ``"affine"`` (``z = (xi - xi_fid) / scale``).
    """

    xi_fid_flat: np.ndarray  # (D,) fiducial design, raw units
    knob_labels: tuple[str, ...]
    unravel: Callable[[np.ndarray], Any]  # flat (D,) -> design pytree
    mode: str = "log"
    scale: np.ndarray | None = None  # (D,) per-knob step for affine mode

    @property
    def n_dim(self) -> int:
        return int(self.xi_fid_flat.size)

    def standardize(self, xi_flat) -> np.ndarray:
        """Raw design ``(D,)`` -> standardized ``z``."""
        xi = np.asarray(xi_flat)
        if self.mode == "log":
            return np.log(xi / self.xi_fid_flat)
        return (xi - self.xi_fid_flat) / self.scale

    def unstandardize(self, z):
        """Standardized ``z`` -> raw design ``(D,)`` (jnp.exp -> works on arrays or tracers)."""
        if self.mode == "log":
            return self.xi_fid_flat * jnp.exp(z)
        return self.xi_fid_flat + self.scale * z

    def design_pytree(self, z) -> Any:
        """Standardized ``z`` -> the design pytree the objective consumes."""
        return self.unravel(self.unstandardize(z))

    @classmethod
    def from_pytree(cls, fiducial_pytree, knob_labels, *, mode="log", scale=None) -> DesignSpec:
        """Build from a fiducial design pytree via ``jax.flatten_util.ravel_pytree``.

        ``knob_labels`` must list the flattened knobs in ``ravel_pytree`` traversal order
        (so eigenvector components map back to named knobs). ``scale`` (affine mode only) is a
        per-knob step in the same flat order.
        """
        from jax.flatten_util import ravel_pytree

        flat, unravel = ravel_pytree(fiducial_pytree)
        xi_fid = np.asarray(flat, dtype=float)
        if len(knob_labels) != xi_fid.size:
            raise ValueError(f"{len(knob_labels)} labels for {xi_fid.size} flattened knobs.")
        sc = None if scale is None else np.asarray(scale, dtype=float)
        return cls(
            xi_fid_flat=xi_fid,
            knob_labels=tuple(knob_labels),
            unravel=unravel,
            mode=mode,
            scale=sc,
        )


def sample_designs(
    n_designs: int,
    n_dim: int,
    *,
    sigma=0.15,
    method: str = "lhs",
    seed: int = 0,
    n_sigma_lhs: float = 2.0,
) -> np.ndarray:
    """``M`` standardized design points ``(M, D)`` around the fiducial ``z = 0``.

    ``method="gaussian"`` draws ``z ~ N(0, sigma^2 I)`` (Constantine's canonical Gaussian
    input measure -- so ``C = E[grad grad^T]`` is w.r.t. that measure). ``method="lhs"``
    Latin-hypercube fills ``[-n_sigma_lhs * sigma, +n_sigma_lhs * sigma]^D`` (better
    space-filling at small ``M``; the default). ``sigma`` is in standardized units (dex, for
    log mode). ``sigma`` may be a scalar or a ``(D,)`` per-knob vector.
    """
    rng = np.random.default_rng(seed)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (n_dim,))
    if method == "gaussian":
        return rng.standard_normal((n_designs, n_dim)) * sig[None, :]
    if method == "lhs":
        from scipy.stats import qmc

        u = qmc.LatinHypercube(d=n_dim, seed=rng).random(n_designs)  # (M, D) in [0,1]
        return (2.0 * u - 1.0) * (n_sigma_lhs * sig)[None, :]
    raise ValueError(f"unknown method {method!r}; expected 'gaussian' or 'lhs'.")


@dataclass(frozen=True)
class GradientSample:
    """Objective values + standardized-space gradients at the sampled designs."""

    z: np.ndarray  # (M, D) standardized design points
    values: np.ndarray  # (M,) objective value (CRN-averaged)
    grads: np.ndarray  # (M, D) gradient in z-space (CRN-averaged)
    crn_spread: np.ndarray  # (M, D) per-component std across CRN redraws (0 if n_crn == 1)


def collect_gradients(
    vg_fn: Callable[[Any, Any], tuple[Any, Any]],
    z_samples: np.ndarray,
    make_ctx: Callable[[int], Any],
    *,
    n_crn: int = 3,
    crn_seed0: int = 0,
    on_sample: Callable[[int, float, np.ndarray], None] | None = None,
) -> GradientSample:
    """Collect CRN-averaged objective gradients at each standardized design.

    ``vg_fn(z, ctx) -> (value, grad)`` is the jitted value-and-grad of the z-space objective
    (from :func:`augr.design_opt.build_design_objectives` on a loss that maps ``z`` through a
    :class:`DesignSpec`); ``grad`` is a flat ``(D,)`` array in z-space. Each ``(design, crn)``
    pair gets a unique index ``crn_seed0 + i * n_crn + j`` passed to ``make_ctx``, so every
    design sees a *fresh, disjoint* CRN ensemble (the subspace cannot fit one ensemble's
    sample-covariance noise) and the per-design gradient is averaged over ``n_crn`` redraws --
    damping that noise *before* it enters ``C = grad grad^T`` (where it would otherwise bias
    the trailing eigenvalues upward). ``make_ctx`` maps an index to an ensemble; for a
    noise-free objective use ``n_crn=1`` and ``make_ctx`` may ignore the index.
    """
    z_samples = np.asarray(z_samples)
    m, d = z_samples.shape
    values = np.empty(m)
    grads = np.empty((m, d))
    spread = np.zeros((m, d))
    for i in range(m):
        gs, vs = [], []
        for j in range(n_crn):
            ctx = make_ctx(crn_seed0 + i * n_crn + j)
            v, g = vg_fn(z_samples[i], ctx)
            vs.append(float(v))
            gs.append(np.asarray(g, dtype=float))
        g_stack = np.stack(gs, axis=0)  # (n_crn, D)
        grads[i] = g_stack.mean(axis=0)
        values[i] = float(np.mean(vs))
        if n_crn > 1:
            spread[i] = g_stack.std(axis=0)
        if on_sample is not None:
            on_sample(i, values[i], grads[i])
    return GradientSample(z=z_samples, values=values, grads=grads, crn_spread=spread)


@dataclass(frozen=True)
class ActiveSubspace:
    """The gradient-covariance ``C``, its eigendecomposition (descending), and the energy spectrum."""

    C: np.ndarray  # (D, D) symmetric PSD gradient covariance
    eigenvalues: np.ndarray  # (D,) descending
    eigenvectors: np.ndarray  # (D, D) columns = active directions (W[:, 0] = direction 1)
    energy: np.ndarray  # (D,) eigenvalues / sum(eigenvalues)

    @property
    def cumulative_energy(self) -> np.ndarray:
        return np.cumsum(self.energy)

    def n_active(self, threshold: float = 0.95) -> int:
        """Smallest dimension whose cumulative energy reaches ``threshold``."""
        return int(np.searchsorted(self.cumulative_energy, threshold) + 1)


def active_subspace(grads: np.ndarray, *, weights: np.ndarray | None = None) -> ActiveSubspace:
    """``C = (1/M) sum_i w_i g_i g_i^T``, then descending eigendecomposition.

    ``grads`` is ``(M, D)`` (the ``GradientSample.grads``); ``weights`` default uniform. ``C``
    is symmetric PSD by construction, so ``np.linalg.eigh`` is exact -- the eigenvalues are the
    energy spectrum (a gap reveals the active dimension), the eigenvectors the design directions.
    """
    g = np.asarray(grads)
    m = g.shape[0]
    w = np.ones(m) / m if weights is None else np.asarray(weights) / np.sum(weights)
    c = (g * w[:, None]).T @ g  # (D, D) = sum_i w_i g_i g_i^T
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0.0, None)  # tiny negatives are eigh round-off on a PSD matrix
    total = np.sum(vals)
    energy = vals / total if total > 0 else np.zeros_like(vals)
    return ActiveSubspace(C=c, eigenvalues=vals, eigenvectors=vecs, energy=energy)


def activity_scores(subspace: ActiveSubspace, n_active: int = 1) -> np.ndarray:
    """Per-knob activity ``a_k = sum_{j < n_active} lambda_j W[k, j]^2`` (Constantine & Diaz 2017).

    Returns the normalized ``(D,)`` scores (sum to 1): how much each raw knob loads on the
    leading ``n_active`` active directions. Pair with ``DesignSpec.knob_labels`` to read off
    which knobs drive the QoI.
    """
    lam = subspace.eigenvalues[:n_active]
    w = subspace.eigenvectors[:, :n_active]
    a = (w**2) @ lam  # (D,)
    s = np.sum(a)
    return a / s if s > 0 else a


def bootstrap_eiguncertainty(
    grads: np.ndarray, *, n_boot: int = 500, n_active: int = 3, seed: int = 0
) -> dict:
    """Bootstrap eigenvalue bars + active-subspace stability over the ``M`` gradient samples.

    Resamples designs with replacement, rebuilds ``C``, re-eigendecomposes, and reports
    eigenvalue percentiles (16/50/84) and the Constantine subspace distance
    ``||W_k W_k^T - W_k^boot W_k^boot^T||_2`` (``k = n_active``) vs the full-sample subspace.
    A leading-eigenvalue interval that clears the bulk -- and a small subspace distance --
    means the active dimension is robust to ``M`` and the MC gradient noise.
    """
    g = np.asarray(grads)
    m, d = g.shape
    rng = np.random.default_rng(seed)
    full = active_subspace(g)
    w_full = full.eigenvectors[:, :n_active]
    p_full = w_full @ w_full.T

    boot_vals = np.empty((n_boot, d))
    sub_dist = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        sub = active_subspace(g[idx])
        boot_vals[b] = sub.eigenvalues
        w_b = sub.eigenvectors[:, :n_active]
        sub_dist[b] = np.linalg.norm(p_full - w_b @ w_b.T, ord=2)
    pct = np.percentile(boot_vals, [16, 50, 84], axis=0)
    return {
        "eigenvalues": full.eigenvalues,
        "eig_p16": pct[0],
        "eig_p50": pct[1],
        "eig_p84": pct[2],
        "subspace_distance_p50": float(np.percentile(sub_dist, 50)),
        "subspace_distance_p84": float(np.percentile(sub_dist, 84)),
        "n_active": n_active,
    }


def subspace_alignment(u: np.ndarray, v: np.ndarray) -> float:
    """``|cos|`` between two unit vectors (sign-agnostic) -- e.g. the sigma(r) vs EIG direction-1."""
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
