"""
sky_patches.py — Sky patch definitions and scan strategy model.

Divides the sky into patches defined by galactic latitude cuts, each with
its own foreground amplitude scaling and noise depth from the scan strategy.

Foreground amplitude scalings are calibrated relative to the BK15 fiducial
(BICEP field at |b| ~ 60°, f_sky ~ 0.01).  The "clean" patch with
|b| > 50° has A_dust_scale ~ 1.0 by construction; dirtier patches scale
up from there following Planck 353 GHz BB power spectrum measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkyPatch:
    """One region of sky with uniform properties for Fisher forecasting.

    Attributes:
        name:           Human-readable label (e.g., "clean", "moderate").
        f_sky:          Sky fraction of this patch.
        A_dust_scale:   Multiplier on fiducial A_dust (BK15 reference).
        A_sync_scale:   Multiplier on fiducial A_sync.
        noise_weight:   Relative integration time density vs uniform scanning.
                        >1 means deeper (ecliptic poles); <1 means shallower.
                        Normalization: Σ(f_sky_p × noise_weight_p) = f_sky_total.
    """
    name: str
    f_sky: float
    A_dust_scale: float
    A_sync_scale: float
    noise_weight: float = 1.0


@dataclass(frozen=True)
class SkyModel:
    """Collection of sky patches covering the observed sky."""
    patches: tuple[SkyPatch, ...]
    description: str = ""

    @property
    def total_f_sky(self) -> float:
        return sum(p.f_sky for p in self.patches)

    def validate(self) -> None:
        """Check consistency of the sky model."""
        total = self.total_f_sky
        if total <= 0.0 or total > 1.0:
            raise ValueError(f"total f_sky = {total}, must be in (0, 1]")
        for p in self.patches:
            if p.f_sky <= 0:
                raise ValueError(f"patch '{p.name}' has f_sky <= 0")
            if p.A_dust_scale < 0 or p.A_sync_scale < 0:
                raise ValueError(f"patch '{p.name}' has negative amplitude scale")
            if p.noise_weight <= 0:
                raise ValueError(f"patch '{p.name}' has noise_weight <= 0")
        # Check noise weight normalization
        weighted_sum = sum(p.f_sky * p.noise_weight for p in self.patches)
        if abs(weighted_sum - total) > 0.01 * total:
            raise ValueError(
                f"Σ(f_sky × noise_weight) = {weighted_sum:.4f}, "
                f"expected {total:.4f} (tolerance 1%)")


# ---------------------------------------------------------------------------
# L2 scan strategy model
# ---------------------------------------------------------------------------

# Galactic-ecliptic tilt: ecliptic pole is at galactic latitude ~30°,
# or equivalently the ecliptic plane is tilted ~60° from the galactic plane.
_ECLIPTIC_GALACTIC_TILT_DEG = 60.19


def l2_scan_depth(ecliptic_lat_deg: np.ndarray,
                  spin_angle_deg: float = 50.0,
                  precession_angle_deg: float = 45.0,
                  ) -> np.ndarray:
    """Relative integration time per pixel for an L2 spinning satellite.

    The scan strategy has the spin axis precessing around the Sun-Earth
    line (anti-sun direction) at angle beta from that axis.  The telescope
    boresight is at angle alpha from the spin axis.

    The boresight traces a cone of half-angle alpha around the spin axis.
    As the spin axis precesses, the boresight sweeps out bands of the sky.
    Pixels near the ecliptic poles (where multiple scan circles overlap)
    get more integration time than the ecliptic equator.

    Defaults: alpha=50°, beta=45° gives full-sky coverage (theta_min=5°,
    theta_max=95°) with ~3× deeper coverage at the ecliptic poles vs
    the equator.  These are typical of LiteBIRD-like scan strategies.

    Analytic model:
        Time per unit ecliptic latitude band ~ 1/sin(theta) where theta
        is the colatitude from the ecliptic pole, within the observable
        range [|beta - alpha|, beta + alpha].

    Args:
        ecliptic_lat_deg:   Ecliptic latitude(s) in degrees, [-90, 90].
        spin_angle_deg:     Boresight angle from spin axis (alpha), degrees.
        precession_angle_deg: Spin axis angle from anti-sun direction (beta).

    Returns:
        Relative integration time per pixel (not normalized).
        Higher values = deeper coverage.
    """
    beta = np.radians(precession_angle_deg)
    alpha = np.radians(spin_angle_deg)
    lat = np.radians(np.asarray(ecliptic_lat_deg, dtype=float))
    theta = np.pi / 2.0 - np.abs(lat)  # colatitude from ecliptic pole

    # Observable range of colatitudes from the precession axis
    theta_min = max(abs(beta - alpha), 1e-6)
    theta_max = min(beta + alpha, np.pi - 1e-6)

    # Time density ~ 1/sin(theta) within observable range
    # The 1/sin comes from the scan circle spending more time per solid
    # angle at small theta (near the poles).
    eps = 1e-10  # avoid float boundary issues
    depth = np.where(
        (theta >= theta_min - eps) & (theta <= theta_max + eps),
        1.0 / np.maximum(np.sin(theta), 1e-6),
        0.0,
    )
    return depth


def _galactic_to_ecliptic_lat(gal_lat_deg: float) -> np.ndarray:
    """Map a galactic latitude to a range of ecliptic latitudes.

    Due to the ~60° tilt, a ring at fixed galactic latitude |b|
    spans a range of ecliptic latitudes.  Returns a 1-d array of
    ecliptic latitudes sampled around the ring, for averaging.
    """
    b = np.radians(gal_lat_deg)
    tilt = np.radians(_ECLIPTIC_GALACTIC_TILT_DEG)
    # Sample galactic longitudes around the ring
    gal_lon = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # Convert (b, l) -> ecliptic latitude via rotation
    # sin(beta_ecl) = sin(b)*cos(tilt) - cos(b)*sin(tilt)*sin(l - l_offset)
    # Simplified: the ecliptic latitude of a point at galactic (b, l) is
    # approximately given by rotating the coordinate frame by the tilt angle.
    sin_ecl_lat = (np.sin(b) * np.cos(tilt)
                   - np.cos(b) * np.sin(tilt) * np.sin(gal_lon))
    ecl_lat_deg = np.degrees(np.arcsin(np.clip(sin_ecl_lat, -1, 1)))
    return ecl_lat_deg


def patch_noise_weights(patches: tuple[SkyPatch, ...],
                        spin_angle_deg: float = 50.0,
                        precession_angle_deg: float = 45.0,
                        n_lat_samples: int = 100,
                        ) -> tuple[float, ...]:
    """Compute per-patch noise weights from the L2 scan strategy.

    For each patch (defined by galactic latitude range), average the
    scan depth over the patch area, accounting for the ecliptic-galactic
    tilt.

    Returns normalized weights satisfying Σ(f_sky_p × w_p) = Σ(f_sky_p).
    """
    # Infer galactic latitude ranges from patch ordering
    # Patches are ordered cleanest-first, which means highest |b| first
    lat_boundaries = _infer_lat_boundaries(patches)

    raw_weights = []
    for (b_lo, b_hi) in lat_boundaries:
        # Sample galactic latitudes in this band (both hemispheres)
        b_samples = np.linspace(b_lo, b_hi, n_lat_samples)
        depths = []
        for b in b_samples:
            # For each galactic latitude, get ecliptic latitudes around ring
            ecl_lats = _galactic_to_ecliptic_lat(b)
            d = l2_scan_depth(ecl_lats, spin_angle_deg, precession_angle_deg)
            # Also southern hemisphere
            ecl_lats_s = _galactic_to_ecliptic_lat(-b)
            d_s = l2_scan_depth(ecl_lats_s, spin_angle_deg,
                                precession_angle_deg)
            depths.append(0.5 * (np.mean(d) + np.mean(d_s)))
        raw_weights.append(float(np.mean(depths)))

    # Normalize: Σ(f_sky_p × w_p) = f_sky_total
    f_sky_total = sum(p.f_sky for p in patches)
    weighted = sum(p.f_sky * w for p, w in zip(patches, raw_weights))
    if weighted <= 0:
        return tuple(1.0 for _ in patches)
    scale = f_sky_total / weighted
    return tuple(w * scale for w in raw_weights)


def _infer_lat_boundaries(patches: tuple[SkyPatch, ...],
                          ) -> list[tuple[float, float]]:
    """Infer galactic latitude boundaries from patch f_sky values.

    Assumes patches are ordered from cleanest (highest |b|) to dustiest
    (lowest |b|), and that each patch covers a symmetric band in both
    hemispheres.  The sky fraction of a band from |b_lo| to |b_hi| is
    f_sky = sin(b_hi) - sin(b_lo).
    """
    boundaries = []
    b_hi = 90.0  # start from the pole
    for p in patches:
        sin_hi = math.sin(math.radians(b_hi))
        sin_lo = sin_hi - p.f_sky  # f_sky = sin(b_hi) - sin(b_lo)
        sin_lo = max(sin_lo, 0.0)
        b_lo = math.degrees(math.asin(sin_lo))
        boundaries.append((b_lo, b_hi))
        b_hi = b_lo
    return boundaries


# ---------------------------------------------------------------------------
# Default sky models
# ---------------------------------------------------------------------------

def single_patch_model(f_sky: float = 0.7) -> SkyModel:
    """Single-patch model for backward compatibility."""
    return SkyModel(
        patches=(SkyPatch("full", f_sky, 1.0, 1.0, 1.0),),
        description=f"Single patch, f_sky={f_sky}",
    )


def default_3patch_model(include_scan: bool = True) -> SkyModel:
    """Three-patch model: clean / moderate / dusty.

    Based on galactic latitude cuts and Planck 353 GHz BB measurements.
    A_dust_scale = 1.0 corresponds to the BK15 fiducial (|b| ~ 60°).
    """
    patches_no_scan = (
        SkyPatch("clean",    f_sky=0.12, A_dust_scale=1.0,  A_sync_scale=1.0),
        SkyPatch("moderate", f_sky=0.20, A_dust_scale=5.0,  A_sync_scale=2.0),
        SkyPatch("dusty",    f_sky=0.20, A_dust_scale=25.0, A_sync_scale=3.0),
    )
    if include_scan:
        weights = patch_noise_weights(patches_no_scan)
        patches = tuple(
            SkyPatch(p.name, p.f_sky, p.A_dust_scale, p.A_sync_scale, w)
            for p, w in zip(patches_no_scan, weights)
        )
    else:
        patches = patches_no_scan
    model = SkyModel(patches, "3-patch galactic latitude model")
    model.validate()
    return model


def default_4patch_model(include_scan: bool = True) -> SkyModel:
    """Four-patch model with finer slicing of the clean sky.

    Dust amplitude scalings from Planck 2018 XXX Table 1 (approximate).
    """
    patches_no_scan = (
        SkyPatch("cleanest", f_sky=0.07, A_dust_scale=0.5,  A_sync_scale=0.8),
        SkyPatch("clean",    f_sky=0.08, A_dust_scale=1.5,  A_sync_scale=1.0),
        SkyPatch("moderate", f_sky=0.17, A_dust_scale=5.0,  A_sync_scale=2.0),
        SkyPatch("dusty",    f_sky=0.20, A_dust_scale=20.0, A_sync_scale=3.0),
    )
    if include_scan:
        weights = patch_noise_weights(patches_no_scan)
        patches = tuple(
            SkyPatch(p.name, p.f_sky, p.A_dust_scale, p.A_sync_scale, w)
            for p, w in zip(patches_no_scan, weights)
        )
    else:
        patches = patches_no_scan
    model = SkyModel(patches, "4-patch galactic latitude model")
    model.validate()
    return model
