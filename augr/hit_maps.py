"""hit_maps.py - HEALPix hit-map generator for L2 scan strategies.

Produces full-sky relative-exposure maps N_hit(n_hat) for an L2-orbiting
satellite with boresight spinning at angle alpha from the spin axis,
the spin axis precessing at angle beta from the anti-sun direction.
The underlying 1/sin(theta) envelope comes from
`sky_patches.l2_scan_depth`; this module lifts it to a 2-D HEALPix
grid in any of the standard coordinate frames (galactic, ecliptic,
celestial).

Intended consumer: map-based CMB component-separation simulators
(e.g. BROOM) that accept a per-channel hit map and scale pixel noise
by 1 / sqrt(N_hit).

See `mean_pixel_rescale_factor` for the sky-average-normalization
helper used when the instrument depth spec should describe the
average surveyed pixel rather than the best pixel (BROOM's default).
"""

from __future__ import annotations

import healpy as hp
import numpy as np

from augr.sky_patches import l2_scan_depth

__all__ = ["l2_hit_map", "mean_pixel_rescale_factor"]


_ALLOWED_COORDS = ("G", "E", "C")


def l2_hit_map(
    nside: int,
    spin_angle_deg: float = 50.0,
    precession_angle_deg: float = 45.0,
    coord: str = "G",
) -> np.ndarray:
    """HEALPix hit map for an L2-spinning satellite.

    Envelope-level analytic model: integration time per pixel scales
    as 1/sin(theta_ecl) within the observable colatitude band
    [|beta - alpha|, beta + alpha] and is zero elsewhere. Built in
    the ecliptic frame via `sky_patches.l2_scan_depth`, optionally
    rotated to galactic or celestial.

    Relation to Maris et al. 2006, Sec. 2 ("Polar Holes and Deep
    Fields"):

    * The 1/sin(theta_ecl) envelope captures the scan-circle pile-up
      near the ecliptic poles but NOT the sharp Deep Field ring at
      theta_ecl ~ |beta - alpha| where overlapping circles
      concentrate. The real DF ring has angular size ~300 deg^2;
      this model smooths it into the envelope.
    * The zero-fill region at theta_ecl < |beta - alpha| is a
      *single-precession-cycle* envelope artifact: over a full
      year the anti-sun direction sweeps the ecliptic and real
      annual coverage extends further toward the pole than the
      envelope suggests. For the defaults (alpha=50, beta=45)
      the envelope cap is |ecl_lat| > 85 (~0.8% of the sky) but
      the annual coverage actually reaches the pole at very low
      exposure. For alpha much larger than beta (Planck nominal,
      alpha=85 beta=0) the single-cycle envelope collapses to a
      single ring while the annual coverage is |ecl_lat| <= 85;
      the envelope model is inappropriate for that regime and
      a proper simulator would fold in the anti-sun sweep.
      Maris Sec. 2 ("Polar Holes and Deep Fields") discusses the
      same effect from the real-mission side: the nominal SS
      leaves an unobserved cap inside the polar DF ring, which
      the precessed SS mitigates by raising beta.
    * Per-detector feedhorn offsets shift each channel's DF ring
      location by ~1 degree (Maris Sec. 2). This function returns
      one common map for all channels; see the `feedhorn_offsets`
      hook reserved in `make_hit_maps.py` for future per-channel
      differentiation.

    Example (alpha, beta) pairs:
        (50, 45)  - LiteBIRD-like (augr default); ~3x deeper at poles.
        (65, 30)  - wider precession cone; fuller mid-latitude band.
        (85, 7.5) - Planck-like precessed SS (Maris Sec. 2: boresight
                    at 85 deg from spin axis, small precession up to
                    10 deg).  Note: Planck's *nominal* SS with
                    beta=0 is NOT captured by this model -- see caveat
                    on annual anti-sun motion below.

    Args:
        nside: HEALPix nside (power of 2).
        spin_angle_deg: alpha, boresight-to-spin-axis angle (deg).
        precession_angle_deg: beta, spin-axis-to-antisun angle (deg).
        coord: Output coordinate frame. "G" (galactic), "E" (ecliptic),
               or "C" (celestial / equatorial J2000).

    Returns:
        Float64 array of length npix = 12 * nside**2 (RING ordering).
        Unnormalized relative integration time: surveyed pixels > 0,
        unsurveyed pixels = 0. Units are arbitrary; the consumer is
        expected to rescale (see `mean_pixel_rescale_factor`).
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

    ecl_lat_deg = np.degrees(0.5 * np.pi - theta_ecl)
    return l2_scan_depth(
        ecl_lat_deg,
        spin_angle_deg=spin_angle_deg,
        precession_angle_deg=precession_angle_deg,
    )


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
        sqrt(max(H) * mean_surveyed(1/H)). Dimensionless, >= 1. For
        the L2 defaults (alpha=50, beta=45) typical value is ~3.0
        -- the polar DF max is ~11.5 (from 1/sin(5 deg)) while the
        area-weighted mean of 1/H over the surveyed sky is ~0.81.

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
