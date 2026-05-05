"""Tests for augr.hit_maps - L2 HEALPix hit-map generator."""

from __future__ import annotations

import unittest

import healpy as hp
import jax.numpy as jnp
import numpy as np

from augr.crosslinks import yearavg_depth_1d
from augr.hit_maps import l2_hit_map, mean_pixel_rescale_factor
from augr.sky_patches import (
    _infer_lat_boundaries,
    default_3patch_model,
    patch_noise_weights,
)


class TestL2HitMap(unittest.TestCase):

    def test_ecliptic_frame_identity(self):
        """coord='E' map is yearavg_depth_1d evaluated at each pixel's
        ecliptic colatitude. No rotation in the ecliptic-frame path."""
        nside = 32
        m = l2_hit_map(nside=nside, coord="E")

        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        expected = np.array(yearavg_depth_1d(jnp.asarray(theta)))

        np.testing.assert_allclose(m, expected, rtol=0, atol=1e-12)

    def test_pole_deeper_than_bulk(self):
        """The rigorous depth peaks just inside the polar support edge
        (theta_ecl ~ |prec - spin|), and is finite-and-smaller in the
        bulk. Verify with HEALPix-area-weighted band means."""
        nside = 128
        m = l2_hit_map(nside=nside, coord="E")

        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        ecl_lat_deg = 90.0 - np.degrees(theta)

        # Pole-side band (just inside the |prec-spin|=5 edge)
        pole_band = (ecl_lat_deg >= 80.0) & (ecl_lat_deg <= 85.0)
        # Mid-latitude bulk band, well inside the support
        mid_band = (ecl_lat_deg >= 30.0) & (ecl_lat_deg <= 45.0)
        self.assertGreater(pole_band.sum(), 100)
        self.assertGreater(mid_band.sum(), 100)
        self.assertGreater(m[pole_band].mean(), m[mid_band].mean())

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
        weighted = sum(p.f_sky * w for p, w in zip(patches, raw_weights, strict=False))
        scale = f_sky_total / weighted
        map_weights = tuple(w * scale for w in raw_weights)

        ref_weights = patch_noise_weights(patches)

        for mw, rw in zip(map_weights, ref_weights, strict=False):
            np.testing.assert_allclose(mw, rw, rtol=0.05)

    def test_coord_invalid_raises(self):
        with self.assertRaises(ValueError):
            l2_hit_map(nside=16, coord="X")

    def test_unsurveyed_polar_caps(self):
        """The rigorous year-averaged form has zero density on polar
        caps that the precession-band-x-spherical-triangle support
        cannot reach. With spin=30, prec=20: spin axis colatitude in
        [70, 110]; boresight cone of radius 30 reaches ecl colatitudes
        in [40, 140], so |ecl_lat| > 50 (theta_ecl < 40 or > 140) is
        unsurveyed and must be exactly 0.
        """
        m = l2_hit_map(nside=32, spin_angle_deg=30.0,
                       precession_angle_deg=20.0, coord="E")
        npix = hp.nside2npix(32)
        theta, _ = hp.pix2ang(32, np.arange(npix))
        ecl_lat_deg = 90.0 - np.degrees(theta)
        polar_mask = np.abs(ecl_lat_deg) > 60.0   # well above |b| = 50
        self.assertGreater(polar_mask.sum(), 10)
        self.assertTrue(np.all(m[polar_mask] == 0.0))


class TestMeanPixelRescaleFactor(unittest.TestCase):

    def test_uniform_returns_one(self):
        """A constant hit map has no spatial variation -- factor = 1."""
        self.assertAlmostEqual(
            mean_pixel_rescale_factor(np.full(1000, 7.3)), 1.0
        )

    def test_l2_factor_greater_than_one(self):
        """For any non-uniform surveyed sky, mean(max/h) > 1.

        The rigorous year-averaged density has caustic peaks just
        inside the support edges, so the factor is meaningfully
        above unity even with smoothed peaks.
        """
        m = l2_hit_map(nside=32, coord="E")
        k = mean_pixel_rescale_factor(m)
        self.assertGreater(k, 1.0)

    def test_ignores_unsurveyed_pixels(self):
        """Zero pixels are excluded from the surveyed-sky average."""
        hits = np.concatenate([np.ones(500), np.zeros(500)])
        self.assertAlmostEqual(mean_pixel_rescale_factor(hits), 1.0)

    def test_all_zero_raises(self):
        with self.assertRaises(ValueError):
            mean_pixel_rescale_factor(np.zeros(100))


if __name__ == "__main__":
    unittest.main()
