"""Tests for augr.hit_maps - L2 HEALPix hit-map generator."""

from __future__ import annotations

import unittest

import healpy as hp
import numpy as np

from augr.hit_maps import l2_hit_map, mean_pixel_rescale_factor
from augr.sky_patches import (
    _infer_lat_boundaries,
    default_3patch_model,
    l2_scan_depth,
    patch_noise_weights,
)


class TestL2HitMap(unittest.TestCase):

    def test_ecliptic_frame_identity(self):
        """coord='E' map equals l2_scan_depth at each pixel's ecliptic lat.

        No rotation is applied in the ecliptic-frame path; the map is
        literally the analytic model evaluated on the HEALPix grid.
        """
        nside = 32
        m = l2_hit_map(nside=nside, coord="E")

        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        ecl_lat_deg = 90.0 - np.degrees(theta)
        expected = l2_scan_depth(ecl_lat_deg)

        np.testing.assert_allclose(m, expected, rtol=0, atol=1e-12)

    def test_envelope_matches_analytic_mean(self):
        """Area-weighted band means match the 1/sin(theta) envelope.

        Ring-average of 1/sin(theta) over a colatitude band [th1, th2]
        has closed form (th2 - th1) / (cos(th1) - cos(th2)).  Compare
        HEALPix area-weighted band means against this analytic reference
        for (a) a deep band near the ecliptic pole, (b) the ecliptic
        equator.  Both bands are fully inside the observable region
        [theta_min = 5 deg, theta_max = 95 deg] for the alpha=50,
        beta=45 default so zero-fill does not contaminate the average.
        """
        nside = 128
        m = l2_hit_map(nside=nside, coord="E")

        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        ecl_lat_deg = 90.0 - np.degrees(theta)

        def analytic_ring_mean(th_lo_deg: float, th_hi_deg: float) -> float:
            th1, th2 = np.radians(th_lo_deg), np.radians(th_hi_deg)
            return (th2 - th1) / (np.cos(th1) - np.cos(th2))

        # Deep band: ecl_lat in [80, 82] -> theta in [8, 10]
        pole_band = (ecl_lat_deg >= 80.0) & (ecl_lat_deg <= 82.0)
        self.assertGreater(pole_band.sum(), 100)
        np.testing.assert_allclose(
            m[pole_band].mean(), analytic_ring_mean(8.0, 10.0), rtol=0.03
        )

        # Equator: ecl_lat in [-1, 1] -> theta in [89, 91]
        eq_band = np.abs(ecl_lat_deg) <= 1.0
        self.assertGreater(eq_band.sum(), 100)
        np.testing.assert_allclose(
            m[eq_band].mean(), analytic_ring_mean(89.0, 91.0), rtol=0.01
        )

    def test_galactic_bands_match_patch_weights(self):
        """G-frame band averages agree with 1-D patch_noise_weights.

        Independent coord-handling paths: this test uses hp.Rotator (2-D),
        patch_noise_weights uses a simplified analytic rotation (1-D ring
        sampling).  Agreement to ~5% confirms both transforms give the
        same answer for galactic-band-averaged quantities.

        The residual is dominated by the weighting-scheme difference
        (HEALPix area-weighted vs linspace-in-b longitude sampling),
        not by coord-transform error.
        """
        nside = 64
        m = l2_hit_map(nside=nside, coord="G")

        patches = default_3patch_model(include_scan=False).patches
        boundaries = _infer_lat_boundaries(patches)

        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        abs_b = np.abs(90.0 - np.degrees(theta))

        raw_weights = []
        for (b_lo, b_hi) in boundaries:
            mask = (abs_b >= b_lo) & (abs_b < b_hi)
            self.assertGreater(mask.sum(), 100)
            raw_weights.append(float(m[mask].mean()))

        # Normalize identically to patch_noise_weights:
        #   sum(f_sky_p * w_p) = sum(f_sky_p)
        f_sky_total = sum(p.f_sky for p in patches)
        weighted = sum(p.f_sky * w for p, w in zip(patches, raw_weights))
        scale = f_sky_total / weighted
        map_weights = tuple(w * scale for w in raw_weights)

        ref_weights = patch_noise_weights(patches)

        for mw, rw in zip(map_weights, ref_weights):
            np.testing.assert_allclose(mw, rw, rtol=0.05)

    def test_coord_invalid_raises(self):
        with self.assertRaises(ValueError):
            l2_hit_map(nside=16, coord="X")

    def test_unsurveyed_pixels_zero(self):
        """Pixels outside [|beta-alpha|, beta+alpha] have depth 0.

        With spin=30, precession=20 the observable band is theta in
        [10 deg, 50 deg] -- the ecliptic equator (theta=90) is NOT
        observed and should be exactly zero.
        """
        m = l2_hit_map(nside=32, spin_angle_deg=30.0,
                       precession_angle_deg=20.0, coord="E")
        npix = hp.nside2npix(32)
        theta, _ = hp.pix2ang(32, np.arange(npix))
        ecl_lat_deg = 90.0 - np.degrees(theta)
        eq_mask = np.abs(ecl_lat_deg) < 5.0
        self.assertGreater(eq_mask.sum(), 10)
        self.assertTrue(np.all(m[eq_mask] == 0.0))


class TestMeanPixelRescaleFactor(unittest.TestCase):

    def test_uniform_returns_one(self):
        """A constant hit map has no spatial variation -- factor = 1."""
        self.assertAlmostEqual(
            mean_pixel_rescale_factor(np.full(1000, 7.3)), 1.0
        )

    def test_l2_factor_greater_than_one(self):
        """For any non-uniform surveyed sky, mean(max/h) > 1."""
        m = l2_hit_map(nside=32, coord="E")
        k = mean_pixel_rescale_factor(m)
        self.assertGreater(k, 1.0)
        # Sanity bound: for the 1/sin envelope between 5 and 90 deg,
        # factor should be O(1-5), not huge.
        self.assertLess(k, 5.0)

    def test_ignores_unsurveyed_pixels(self):
        """Zero pixels are excluded from the surveyed-sky average."""
        hits = np.concatenate([np.ones(500), np.zeros(500)])
        self.assertAlmostEqual(mean_pixel_rescale_factor(hits), 1.0)

    def test_all_zero_raises(self):
        with self.assertRaises(ValueError):
            mean_pixel_rescale_factor(np.zeros(100))


if __name__ == "__main__":
    unittest.main()
