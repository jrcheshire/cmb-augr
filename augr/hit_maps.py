"""hit_maps.py - HEALPix hit-map generator for L2 scan strategies.

Produces full-sky relative-exposure maps N_hit(n_hat) for an L2-orbiting
satellite with boresight spinning at angle alpha from the spin axis,
the spin axis precessing at angle beta from the anti-sun direction.
The underlying year-averaged ergodic scan-depth density comes from
`augr.crosslinks.yearavg_depth_1d` (the k=0 case of the spin-coefficient
machinery); this module lifts it to a 2-D HEALPix grid in any of the
standard coordinate frames (galactic, ecliptic, celestial).

Intended consumer: map-based CMB component-separation simulators
(e.g. BROOM) that accept a per-channel hit map and scale pixel noise
by 1 / sqrt(N_hit).

See `mean_pixel_rescale_factor` for the sky-average-normalization
helper used when the instrument depth spec should describe the
average surveyed pixel rather than the best pixel (BROOM's default).
"""

from __future__ import annotations

import healpy as hp
import jax.numpy as jnp
import numpy as np

from augr.crosslinks import yearavg_depth_1d

__all__ = ["l2_hit_map", "mean_pixel_rescale_factor"]


_ALLOWED_COORDS = ("G", "E", "C")


def l2_hit_map(
    nside: int,
    spin_angle_deg: float = 50.0,
    precession_angle_deg: float = 45.0,
    coord: str = "G",
) -> np.ndarray:
    """HEALPix hit map for an L2-spinning satellite.

    Year-averaged ergodic scan-depth density at each ecliptic colatitude,
    computed via :func:`augr.crosslinks.yearavg_depth_1d` (the k=0 case
    of the spin-coefficient closed form). Built in the ecliptic frame,
    optionally rotated to galactic or celestial.

    The depth is a 1-D function of ecliptic colatitude only (azimuthally
    symmetric around the ecliptic pole by construction of the L2 scan).

    Relation to Maris et al. 2006, Sec. 2 ("Polar Holes and Deep
    Fields"):

    * The closed form captures the precession-band caustics in the
      bulk of the support but NOT the sharp Deep Field ring at
      theta_ecl ~ |beta - alpha| where overlapping circles
      concentrate over short timescales. The real DF ring has
      angular size ~300 deg^2; the year-averaged ergodic limit
      smooths it into a broader peak.
    * Per-detector feedhorn offsets shift each channel's DF ring
      location by ~1 degree (Maris Sec. 2). This function returns
      one common map for all channels; see the ``feedhorn_offsets``
      hook reserved in ``make_hit_maps.py`` for future per-channel
      differentiation.

    Example (alpha, beta) pairs:
        (50, 45)  - LiteBIRD-like (augr default).
        (65, 30)  - wider precession cone; fuller mid-latitude band.
        (85, 7.5) - Planck-like precessed SS (Maris Sec. 2: boresight
                    at 85 deg from spin axis, small precession up to
                    10 deg).

    Args:
        nside: HEALPix nside (power of 2).
        spin_angle_deg: alpha, boresight-to-spin-axis angle (deg).
        precession_angle_deg: beta, spin-axis-to-antisun angle (deg).
        coord: Output coordinate frame. "G" (galactic), "E" (ecliptic),
               or "C" (celestial / equatorial J2000).

    Returns:
        Float64 array of length npix = 12 * nside**2 (RING ordering).
        Unnormalized relative integration time. Units are arbitrary;
        the consumer is expected to rescale (see
        :func:`mean_pixel_rescale_factor`).
    """
    if coord not in _ALLOWED_COORDS:
        raise ValueError(
            f"coord={coord!r} not in {_ALLOWED_COORDS}"
        )

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    if coord == "E":
        theta_ecl = theta
    else:
        rot = hp.Rotator(coord=[coord, "E"])
        theta_ecl, _ = rot(theta, phi)

    return np.array(yearavg_depth_1d(
        jnp.asarray(theta_ecl),
        spin_angle_deg=spin_angle_deg,
        precession_angle_deg=precession_angle_deg,
    ))


def mean_pixel_rescale_factor(hits: np.ndarray) -> float:
    """Depth rescale factor for sky-average-matched noise normalization.

    BROOM normalizes an input hit map internally to max=1 and applies
    pixel noise variance V(p) = sigma^2 / h_norm(p). Without a
    rescale, spec `depth_P` describes the best pixel (ecliptic pole);
    dividing `depth_P` by the factor returned here makes `depth_P`
    describe the *sky-average* surveyed pixel, with the polar DFs
    correspondingly deeper and the ecliptic equator shallower.

    Which convention matches an instrument spec is a calibration
    choice -- LiteBIRD PTEP Table 3 quotes sky-effective sensitivity,
    so the sky-average convention is the faithful one here; other
    specs may quote peak-pixel or something else.  Changing
    convention is ~k^2 on per-pixel noise variance (k ~ 3 for the
    L2 defaults), so it's worth being explicit in whatever the
    consuming spec documents.

        V_avg = sigma^2 * max(H) * mean_surveyed(1/H)
              = sigma^2 * k
        sigma_use = sigma_spec / sqrt(k)

    Args:
        hits: HEALPix hit map (any normalization). Unsurveyed pixels
              should be exactly zero; they are excluded from the
              surveyed-sky average.

    Returns:
        sqrt(max(H) * mean_surveyed(1/H)). Dimensionless, >= 1. The
        precise value depends on the (alpha, beta) configuration --
        for the L2 default (alpha=50, beta=45) the rigorous
        scan-depth density peaks just inside the support boundaries
        and the rescale factor is O(few).

    Raises:
        ValueError: if all pixels are unsurveyed (hits == 0).
    """
    h = np.asarray(hits, dtype=float)
    surveyed = h > 0
    if not np.any(surveyed):
        raise ValueError("hits has no surveyed pixels (all zero)")
    h_max = float(h[surveyed].max())
    mean_inv = float(np.mean(h_max / h[surveyed]))
    return float(np.sqrt(mean_inv))
