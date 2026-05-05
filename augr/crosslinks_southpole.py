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

Polarization angle on sky is RA-invariant at lat = -90 deg, so h_k for
a single detector depends on declination, focal-plane offset, deck
distribution, and k -- not RA. Consequence: a 2-D h_k map is constant
along RA and varies along Dec.

Convention: deck angle increases clockwise as seen on the projected
boresight, matching the BK pipeline's ``chi2alpha.m``.

Public API:
    h_k_boresight        - h_k at the boresight (focal-plane radius r=0).
    h_k_offaxis          - h_k for a single off-axis detector vs. dec.
    h_k_map_southpole    - 2-D flat-sky h_k map for one detector.
    southpole_field_mask - boolean mask of pixels inside the BK field.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from augr._chi2alpha import chi2alpha

__all__ = [
    "BA_DECK_ANGLES_8",
    "h_k_boresight",
    "h_k_map_southpole",
    "h_k_offaxis",
    "southpole_field_mask",
]


# BICEP Array CMB schedule cycles through 8 deck angles at 45 deg intervals,
# one deck per 2-day schedule, governed by ``SN mod 8`` (see e.g. the header
# of ``~/bicepkeck/gcp/config/sch/CMB/9_baCMB_03_000.sch``). Over a 16-day
# cycle the deck distribution is uniform 1/8 over these eight angles.
#
# For 8 evenly-spaced decks at 45 deg, the discrete sum
# ``sum_n exp(-i k * 45 deg * n)`` for ``n in {0, ..., 7}`` is zero for any
# ``k`` not a multiple of 8. So at the boresight, the BA 8-deck schedule
# null-suppresses ``h_k`` for ``k in {1, 2, 3, 4, 5, 6, 7}`` -- every spin
# moment in Wallis 2017's contamination list.
BA_DECK_ANGLES_8 = (23.0, 68.0, 113.0, 158.0, 203.0, 248.0, 293.0, 338.0)


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


def _validate_k(k: int) -> None:
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def h_k_offaxis(
    dec_deg: jnp.ndarray,
    deck_deg: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    r_deg: float = 0.0,
    theta_fp_deg: float = 0.0,
    chi_deg: float = 0.0,
    k: int = 2,
) -> jnp.ndarray:
    """Year-averaged h_k for an off-axis detector at one or more declinations.

    Computes alpha(dec, deck) via ``augr._chi2alpha.chi2alpha`` for each
    (dec, deck) combination, then sums

        h_k(dec) = sum_d w_d exp(-i k alpha(dec, deck=d, r, theta_fp, chi))

    over the deck distribution. JAX-differentiable in all numeric arguments,
    including ``r_deg`` and ``theta_fp_deg`` (use cases: focal-plane design
    sensitivities, schedule optimization).

    Args:
        dec_deg: declination(s) in degrees, scalar or array. Output shape
            matches.
        deck_deg: 1-D array of deck angles in degrees.
        weights: integration-time weights for ``deck_deg`` (sum normalized
            internally). Default uniform.
        r_deg: focal-plane angular radius of the detector (deg). r=0
            recovers ``h_k_boresight``.
        theta_fp_deg: focal-plane angular position of the detector (deg).
        chi_deg: detector polarization fiducial angle (deg). Pure phase on
            h_k.
        k: spin order (positive integer).

    Returns:
        ``jnp.complex128`` array of shape ``dec_deg.shape``.
    """
    _validate_k(k)

    dec = jnp.asarray(dec_deg, dtype=jnp.float64)
    deck = jnp.asarray(deck_deg, dtype=jnp.float64)
    if weights is None:
        w = jnp.ones_like(deck) / deck.size
    else:
        w = jnp.asarray(weights, dtype=jnp.float64)
        w = w / w.sum()

    # Broadcast: dec[..., None] against deck[None, ...] -> alpha shape (..., n_deck).
    dec_b = dec[..., None]
    deck_b = jnp.broadcast_to(deck, dec_b.shape[:-1] + deck.shape)
    alpha_deg = chi2alpha(
        ra=jnp.zeros_like(dec_b),  # ra-invariant; choose any value.
        dec=dec_b,
        r=r_deg,
        theta=theta_fp_deg,
        chi=chi_deg,
        thetaref=deck_b,
    )
    alpha_rad = jnp.deg2rad(alpha_deg)
    integrand = jnp.exp(-1j * k * alpha_rad)
    return jnp.sum(w * integrand, axis=-1).astype(jnp.complex128)


def h_k_map_southpole(
    ra_grid_deg: jnp.ndarray,
    dec_grid_deg: jnp.ndarray,
    deck_deg: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    r_deg: float = 0.0,
    theta_fp_deg: float = 0.0,
    chi_deg: float = 0.0,
    k: int = 2,
) -> jnp.ndarray:
    """2-D flat-sky h_k map for a single detector.

    The map is RA-invariant by construction (the lat=-90 deg geometry
    guarantees alpha(pixel, deck) does not depend on RA), so the function
    just computes ``h_k_offaxis`` at each Dec and broadcasts across RA.

    Args:
        ra_grid_deg: 1-D array of RA pixel centers (deg).
        dec_grid_deg: 1-D array of Dec pixel centers (deg).
        deck_deg, weights, r_deg, theta_fp_deg, chi_deg, k: see
            ``h_k_offaxis``.

    Returns:
        ``jnp.complex128`` array of shape ``(n_ra, n_dec)``.
    """
    ra = jnp.asarray(ra_grid_deg, dtype=jnp.float64)
    h_k_dec = h_k_offaxis(
        dec_grid_deg, deck_deg, weights=weights,
        r_deg=r_deg, theta_fp_deg=theta_fp_deg, chi_deg=chi_deg, k=k,
    )  # shape (n_dec,)
    return jnp.broadcast_to(h_k_dec[None, :], (ra.shape[0], *h_k_dec.shape)).astype(jnp.complex128)


def southpole_field_mask(
    ra_grid_deg: jnp.ndarray,
    dec_grid_deg: jnp.ndarray,
    ra_min: float = -60.0,
    ra_max: float = 60.0,
    dec_min: float = -73.0,
    dec_max: float = -38.0,
) -> jnp.ndarray:
    """Boolean field-bounds mask on a flat-sky grid.

    Defaults match the BICEP CMB field. Returns shape ``(n_ra, n_dec)``,
    ``True`` for in-field pixels.

    This is the simplest sensible "hit map": a binary indicator of which
    pixels are observed at all. A more refined hit map weighted by
    integration time per (Az, El) is a future extension; not needed for
    h_k since the closed form already gives the per-pixel value, but
    useful when forming sky averages like ``<|h_k|^2>``.
    """
    ra = jnp.asarray(ra_grid_deg, dtype=jnp.float64)
    dec = jnp.asarray(dec_grid_deg, dtype=jnp.float64)
    ra_in = (ra >= ra_min) & (ra <= ra_max)
    dec_in = (dec >= dec_min) & (dec <= dec_max)
    return ra_in[:, None] & dec_in[None, :]
