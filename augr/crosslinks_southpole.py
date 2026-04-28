"""crosslinks_southpole.py - h_k for South Pole / BICEP-Keck scan strategy.

At lat = -90 deg (treated as exact at this stage), the year-averaged
crosslink coefficient

    h_k = <exp(-i k alpha)>_hits

reduces to a discrete weighted sum over the telescope deck distribution.
There is no orbital phase-space integral. Derivation:
``scripts/southpole_derivation/01_geometry.md``.

This is the South Pole companion to ``augr.crosslinks`` (L2 satellite
scans). The two modules share the Wallis 2017 spin formalism downstream
(differential-systematic propagation via ``|h_k|^2``) but their
geometries are qualitatively different: L2 needs a 1-D adaptive
quadrature over orbital phase space with Chebyshev singularity
absorption; the South Pole reduces to a finite weighted sum over deck
angles.

Convention: deck angle increases clockwise as seen on the projected
boresight, matching the BK pipeline's ``chi2alpha.m``.

Public API:
    h_k_boresight - h_k at the boresight (focal-plane radius r = 0).

Off-axis detectors and per-pixel maps over the BICEP CMB field are not
yet implemented; only the boresight closed form is exposed.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["h_k_boresight"]


def h_k_boresight(
    deck_deg: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    chi_deg: float = 0.0,
    k: int = 2,
) -> jnp.ndarray:
    """Year-averaged h_k at the boresight for a discrete deck distribution.

    At the boresight (r = 0), polarization angle on sky reduces to
    ``alpha(deck) = -90 + chi + deck`` (the r=0 branch of
    ``augr._chi2alpha.chi2alpha``), so

        h_k = sum_d w_d exp(-i k alpha_d).

    The function is JAX-differentiable in ``weights`` and ``chi_deg``;
    ``deck_deg`` is JAX-differentiable too if you want gradients with
    respect to the schedule angles themselves (deck-schedule design).

    Args:
        deck_deg: 1-D array of deck angles in degrees.
        weights: integration-time weights, broadcast to ``deck_deg``;
            normalized to sum to 1 internally. Default uniform.
        chi_deg: per-detector polarization fiducial angle in degrees.
            Acts as a global phase on h_k; ``|h_k|^2`` is invariant under
            ``chi_deg``.
        k: spin order (positive integer; typically 1, 2, or 4 — the
            orders that show up in Wallis 2017 Eqs. 20-22).

    Returns:
        ``jnp.complex128`` scalar h_k.
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    deck = jnp.asarray(deck_deg, dtype=jnp.float64)
    if weights is None:
        w = jnp.ones_like(deck) / deck.size
    else:
        w = jnp.asarray(weights, dtype=jnp.float64)
        w = w / w.sum()

    alpha_rad = jnp.deg2rad(-90.0 + chi_deg + deck)
    return jnp.sum(w * jnp.exp(-1j * k * alpha_rad)).astype(jnp.complex128)
