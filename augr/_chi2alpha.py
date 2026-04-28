"""JAX port of bk_analysis/pipeline/util/chi2alpha.m.

Polarization angle on sky for a BICEP-style detector, given:

* ``ra``, ``dec`` -- boresight pointing (deg).
* ``r``, ``theta`` -- detector focal-plane offset: ``r`` is angular distance
  from the boresight, ``theta`` is the bearing within the focal plane (deg).
* ``chi`` -- polarization-sensitive direction relative to a fiducial focal-
  plane direction (deg).
* ``thetaref`` -- fiducial reference direction for ``chi``. The telescope
  deck angle enters here: a deck rotation of Delta increments ``thetaref``
  by Delta uniformly across all detectors; ``theta`` and ``chi`` are
  intrinsic and don't change. Sign convention: clockwise rotation of the
  projected boresight on sky = positive deck.

At lat = -90 deg, ``alpha`` is a function of ``(dec, r, theta, chi,
thetaref)`` only -- not ``ra`` -- because ``reckon`` and ``azimuth`` are
both equivariant under a common ra shift on the sphere.

For r = 0 (boresight detector) the formula collapses to the closed form

    alpha(boresight) = -90 + chi + thetaref

i.e. ``const + deck`` as derived in ``scripts/southpole_derivation/01_geometry.md``.

Output is wrapped to ``[-180, 180)``.
"""
from __future__ import annotations

import jax.numpy as jnp

__all__ = ["chi2alpha"]


def _reckon(lat1: jnp.ndarray, lon1: jnp.ndarray, dist: jnp.ndarray,
            bearing: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Great-circle destination from a starting point. All inputs in radians."""
    sin_lat1, cos_lat1 = jnp.sin(lat1), jnp.cos(lat1)
    sin_d, cos_d = jnp.sin(dist), jnp.cos(dist)
    sin_b, cos_b = jnp.sin(bearing), jnp.cos(bearing)
    sin_lat2 = sin_lat1 * cos_d + cos_lat1 * sin_d * cos_b
    lat2 = jnp.arcsin(jnp.clip(sin_lat2, -1.0, 1.0))
    lon2 = lon1 + jnp.arctan2(sin_b * sin_d * cos_lat1,
                              cos_d - sin_lat1 * sin_lat2)
    return lat2, lon2


def _azimuth(lat1: jnp.ndarray, lon1: jnp.ndarray,
             lat2: jnp.ndarray, lon2: jnp.ndarray) -> jnp.ndarray:
    """Great-circle initial bearing from point 1 to point 2 (radians)."""
    dlon = lon2 - lon1
    cos_lat2 = jnp.cos(lat2)
    return jnp.arctan2(
        jnp.sin(dlon) * cos_lat2,
        jnp.cos(lat1) * jnp.sin(lat2)
        - jnp.sin(lat1) * cos_lat2 * jnp.cos(dlon),
    )


def chi2alpha(
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    r: jnp.ndarray,
    theta: jnp.ndarray,
    chi: jnp.ndarray,
    thetaref: jnp.ndarray,
) -> jnp.ndarray:
    """Polarization angle on sky, in degrees, wrapped to ``[-180, 180)``.

    JAX-port of ``bk_analysis/pipeline/util/chi2alpha.m``. All angle inputs
    in degrees. Inputs broadcast to a common shape.

    The ``r == 0`` branch uses the closed-form Taylor limit of the off-axis
    formula (``alpha = -90 + theta + chi`` before the fiducial correction)
    so that the calculation stays well-defined at the boresight.
    """
    ra_r = jnp.deg2rad(ra)
    dec_r = jnp.deg2rad(dec)
    r_r = jnp.deg2rad(r)
    bearing_r = jnp.deg2rad(theta - 90.0)

    dec_bol_r, ra_bol_r = _reckon(dec_r, ra_r, r_r, bearing_r)
    az_deg = jnp.rad2deg(_azimuth(dec_bol_r, ra_bol_r, dec_r, ra_r))

    alpha_off = az_deg - 180.0 + chi
    alpha_zero = -90.0 + theta + chi
    alpha = jnp.where(r == 0.0, alpha_zero, alpha_off)

    alpha = alpha + thetaref - theta

    return (alpha + 180.0) % 360.0 - 180.0
