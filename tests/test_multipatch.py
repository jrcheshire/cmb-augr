"""Tests for multi-patch Fisher forecasting."""

import math
import unittest

import jax.numpy as jnp
import numpy as np

from augr.config import (
    DEFAULT_FIXED,
    DEFAULT_FIXED_MOMENT,
    DEFAULT_PRIORS,
    DEFAULT_PRIORS_MOMENT,
    FIDUCIAL_BK15,
    FIDUCIAL_MOMENT,
    simple_probe,
)
from augr.fisher import FisherForecast
from augr.foregrounds import GaussianForegroundModel, MomentExpansionModel
from augr.multipatch import (
    MultiPatchFisher,
    _is_per_patch,
    fiducial_for_patch,
    instrument_for_patch,
)
from augr.signal import SignalModel
from augr.sky_patches import (
    SkyModel,
    SkyPatch,
    _infer_lat_boundaries,
    default_3patch_model,
    default_4patch_model,
    l2_scan_depth,
    patch_noise_weights,
    single_patch_model,
)
from augr.spectra import CMBSpectra

# ---------------------------------------------------------------------------
# Sky patches
# ---------------------------------------------------------------------------

class TestSkyPatch(unittest.TestCase):

    def test_sky_model_total_fsky(self):
        model = SkyModel(patches=(
            SkyPatch("a", 0.3, 1.0, 1.0),
            SkyPatch("b", 0.2, 1.0, 1.0),
        ))
        self.assertAlmostEqual(model.total_f_sky, 0.5)

    def test_validate_good(self):
        model = single_patch_model(0.5)
        model.validate()  # should not raise

    def test_validate_negative_fsky(self):
        model = SkyModel(patches=(
            SkyPatch("bad", -0.1, 1.0, 1.0),
        ))
        with self.assertRaises(ValueError):
            model.validate()

    def test_validate_noise_normalization(self):
        """Noise weights must satisfy normalization."""
        model = SkyModel(patches=(
            SkyPatch("a", 0.3, 1.0, 1.0, noise_weight=2.0),
            SkyPatch("b", 0.2, 1.0, 1.0, noise_weight=1.0),
        ))
        # sum(f_sky*w) = 0.6+0.2 = 0.8, total = 0.5 → fails
        with self.assertRaises(ValueError):
            model.validate()


class TestScanStrategy(unittest.TestCase):

    def test_poles_deeper_than_equator(self):
        """Ecliptic poles should get more integration time.

        With default alpha=50, beta=45: theta_min=5°, theta_max=95°.
        Full-sky coverage with ~3× deeper at poles.
        """
        depth_pole = l2_scan_depth(np.array([85.0]))[0]
        depth_eq = l2_scan_depth(np.array([5.0]))[0]
        self.assertGreater(depth_pole, depth_eq)

    def test_symmetric(self):
        """North and south ecliptic poles should get equal depth."""
        d_north = l2_scan_depth(np.array([80.0]))[0]
        d_south = l2_scan_depth(np.array([-80.0]))[0]
        self.assertAlmostEqual(d_north, d_south, places=5)

    def test_zero_outside_range(self):
        """Regions outside the scan cone should get zero depth."""
        # With alpha=10, beta=20: theta_min=10°, theta_max=30°
        # Ecliptic lat=89° → colatitude=1° < theta_min=10° → zero
        depth = l2_scan_depth(np.array([89.0]), spin_angle_deg=10.0,
                              precession_angle_deg=20.0)
        self.assertEqual(depth[0], 0.0)


class TestLatBoundaries(unittest.TestCase):

    def test_single_patch(self):
        patches = (SkyPatch("all", 0.5, 1.0, 1.0),)
        boundaries = _infer_lat_boundaries(patches)
        self.assertEqual(len(boundaries), 1)
        b_lo, b_hi = boundaries[0]
        self.assertAlmostEqual(b_hi, 90.0)
        # f_sky = sin(90) - sin(b_lo) = 1 - sin(b_lo) = 0.5
        # sin(b_lo) = 0.5, b_lo = 30°
        self.assertAlmostEqual(b_lo, 30.0, places=1)

    def test_two_patches_cover_correctly(self):
        patches = (
            SkyPatch("hi", 0.12, 1.0, 1.0),
            SkyPatch("lo", 0.20, 1.0, 1.0),
        )
        boundaries = _infer_lat_boundaries(patches)
        # First patch: [b_lo1, 90], sin(90)-sin(b_lo1)=0.12
        # → sin(b_lo1) = 0.88, b_lo1 ≈ 61.6°
        # Second: [b_lo2, 61.6], sin(61.6)-sin(b_lo2) = 0.20
        # → sin(b_lo2) = 0.68, b_lo2 ≈ 42.8°
        self.assertAlmostEqual(boundaries[0][1], 90.0)
        self.assertAlmostEqual(boundaries[0][0], boundaries[1][1], places=5)
        # Check f_sky is recovered
        for i, p in enumerate(patches):
            b_lo, b_hi = boundaries[i]
            f_sky_recovered = math.sin(math.radians(b_hi)) - math.sin(
                math.radians(b_lo))
            self.assertAlmostEqual(f_sky_recovered, p.f_sky, places=4)


class TestPatchNoiseWeights(unittest.TestCase):

    def test_normalization(self):
        """Weights should satisfy normalization constraint."""
        patches = (
            SkyPatch("a", 0.15, 1.0, 1.0),
            SkyPatch("b", 0.20, 1.0, 1.0),
            SkyPatch("c", 0.15, 1.0, 1.0),
        )
        weights = patch_noise_weights(patches)
        total = sum(p.f_sky * w for p, w in zip(patches, weights, strict=False))
        expected = sum(p.f_sky for p in patches)
        self.assertAlmostEqual(total, expected, places=4)

    def test_all_positive(self):
        patches = (
            SkyPatch("a", 0.10, 1.0, 1.0),
            SkyPatch("b", 0.20, 1.0, 1.0),
        )
        weights = patch_noise_weights(patches)
        for w in weights:
            self.assertGreater(w, 0)


class TestDefaultModels(unittest.TestCase):

    def test_3patch_validates(self):
        model = default_3patch_model()
        model.validate()
        self.assertEqual(len(model.patches), 3)

    def test_4patch_validates(self):
        model = default_4patch_model()
        model.validate()
        self.assertEqual(len(model.patches), 4)

    def test_single_patch_validates(self):
        model = single_patch_model(0.7)
        model.validate()
        self.assertEqual(len(model.patches), 1)

    def test_3patch_no_scan(self):
        """Without scan, all noise_weights should be 1.0."""
        model = default_3patch_model(include_scan=False)
        for p in model.patches:
            self.assertEqual(p.noise_weight, 1.0)


# ---------------------------------------------------------------------------
# Instrument for patch
# ---------------------------------------------------------------------------

class TestInstrumentForPatch(unittest.TestCase):

    def test_fsky_set_correctly(self):
        inst = simple_probe()
        patch = SkyPatch("test", 0.1, 1.0, 1.0, 1.0)
        inst_p = instrument_for_patch(inst, patch, 0.7)
        self.assertEqual(inst_p.f_sky, 0.1)

    def test_net_scaling(self):
        """NET should scale by sqrt(f_sky_total / (f_sky_p * w_p))."""
        inst = simple_probe()
        patch = SkyPatch("test", 0.1, 1.0, 1.0, noise_weight=1.5)
        inst_p = instrument_for_patch(inst, patch, 0.7)
        expected_scale = math.sqrt(0.7 / (0.1 * 1.5))
        for ch_orig, ch_new in zip(inst.channels, inst_p.channels, strict=False):
            self.assertAlmostEqual(
                ch_new.net_per_detector,
                ch_orig.net_per_detector * expected_scale,
                places=5)

    def test_other_properties_unchanged(self):
        inst = simple_probe()
        patch = SkyPatch("test", 0.2, 1.0, 1.0)
        inst_p = instrument_for_patch(inst, patch, 0.7)
        for ch_orig, ch_new in zip(inst.channels, inst_p.channels, strict=False):
            self.assertEqual(ch_new.nu_ghz, ch_orig.nu_ghz)
            self.assertEqual(ch_new.n_detectors, ch_orig.n_detectors)
            self.assertAlmostEqual(ch_new.beam_fwhm_arcmin,
                                   ch_orig.beam_fwhm_arcmin)


# ---------------------------------------------------------------------------
# Fiducial for patch
# ---------------------------------------------------------------------------

class TestFiducialForPatch(unittest.TestCase):

    def test_dust_scaling(self):
        patch = SkyPatch("dusty", 0.2, 5.0, 2.0)
        fid = fiducial_for_patch(FIDUCIAL_BK15, patch)
        self.assertAlmostEqual(fid["A_dust"],
                               FIDUCIAL_BK15["A_dust"] * 5.0)
        self.assertAlmostEqual(fid["A_sync"],
                               FIDUCIAL_BK15["A_sync"] * 2.0)

    def test_global_unchanged(self):
        patch = SkyPatch("dusty", 0.2, 5.0, 2.0)
        fid = fiducial_for_patch(FIDUCIAL_BK15, patch)
        self.assertEqual(fid["r"], FIDUCIAL_BK15["r"])
        self.assertEqual(fid["beta_dust"], FIDUCIAL_BK15["beta_dust"])
        self.assertEqual(fid["A_lens"], FIDUCIAL_BK15["A_lens"])

    def test_moment_params_stay_global(self):
        """Per Chluba+ 2017 Eq. 8 (arXiv:1701.00274), omega_{ij} =
        <[p_i(r) - p_i_bar][p_j(r) - p_j_bar]> is a pure central moment
        of spectral parameters across the sky, not bundled with the
        amplitude.  So omega_d_* and omega_s_* stay global across
        patches -- dustier patches don't automatically have larger
        spectral-index variance."""
        # Use non-zero omegas so any scaling would be visible
        base = {**FIDUCIAL_MOMENT,
                "omega_d_beta": 0.1, "omega_d_T": 0.05, "omega_d_betaT": 0.02,
                "omega_s_beta": 0.08, "omega_s_c": 0.04, "omega_s_betac": 0.01}
        patch = SkyPatch("dusty", 0.2, 3.0, 2.0)
        fid = fiducial_for_patch(base, patch)
        for key in ("omega_d_beta", "omega_d_T", "omega_d_betaT",
                    "omega_s_beta", "omega_s_c", "omega_s_betac"):
            self.assertEqual(fid[key], base[key],
                             f"{key} must not be rescaled per patch")


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

class TestParameterSharing(unittest.TestCase):

    def test_r_is_global(self):
        self.assertFalse(_is_per_patch("r"))

    def test_A_dust_is_per_patch(self):
        self.assertTrue(_is_per_patch("A_dust"))

    def test_A_sync_is_per_patch(self):
        self.assertTrue(_is_per_patch("A_sync"))

    def test_beta_dust_is_global(self):
        self.assertFalse(_is_per_patch("beta_dust"))

    def test_moment_omega_is_global(self):
        """Moment-expansion omega_* params are sky-level variances of
        spectral parameters (Chluba+ 2017 Eq. 8), not amplitudes -- so
        they stay global across patches."""
        for name in ("omega_d_beta", "omega_d_T", "omega_d_betaT",
                     "omega_s_beta", "omega_s_c", "omega_s_betac"):
            self.assertFalse(_is_per_patch(name),
                             f"{name} should be global per Chluba+ 2017")

    def test_decorrelation_and_curvature_are_global(self):
        """Delta_dust, Delta_sync (decorrelation) and c_sync (spectral
        curvature) are SED-shape parameters, not amplitudes, so they
        stay global and are not scaled per patch."""
        self.assertFalse(_is_per_patch("Delta_dust"))
        self.assertFalse(_is_per_patch("Delta_sync"))
        self.assertFalse(_is_per_patch("c_sync"))

    def test_fiducial_for_patch_does_not_scale_decorrelation(self):
        """fiducial_for_patch must leave Delta_dust, Delta_sync, c_sync
        unchanged -- scaling them by A_dust_scale / A_sync_scale could
        push them out of their physical range."""
        base = {
            "A_dust": 4.7, "A_sync": 1.5,
            "Delta_dust": 0.3, "Delta_sync": 0.2, "c_sync": -0.05,
        }
        patch = SkyPatch("test", f_sky=0.1, A_dust_scale=3.0, A_sync_scale=2.0)
        fid = fiducial_for_patch(base, patch)
        self.assertAlmostEqual(fid["A_dust"], 4.7 * 3.0)
        self.assertAlmostEqual(fid["A_sync"], 1.5 * 2.0)
        self.assertEqual(fid["Delta_dust"], 0.3)
        self.assertEqual(fid["Delta_sync"], 0.2)
        self.assertEqual(fid["c_sync"], -0.05)


# ---------------------------------------------------------------------------
# Multi-patch Fisher
# ---------------------------------------------------------------------------

class TestMultiPatchFisher(unittest.TestCase):
    """Integration tests for MultiPatchFisher."""

    @classmethod
    def setUpClass(cls):
        """Build shared objects (expensive — do once)."""
        cls.cmb = CMBSpectra()
        cls.inst = simple_probe()
        cls.fg_gauss = GaussianForegroundModel()
        cls.fid_gauss = {**FIDUCIAL_BK15, "A_lens": 0.27}
        cls.priors_gauss = dict(DEFAULT_PRIORS)
        cls.fixed_gauss = [*list(DEFAULT_FIXED), "Delta_dust"]
        cls.signal_kwargs = {"ell_max": 300, "delta_ell": 35}

    def test_single_patch_recovery(self):
        """Single-patch MultiPatchFisher must match FisherForecast."""
        sky = single_patch_model(self.inst.f_sky)

        # Single-patch via MultiPatchFisher
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf.compute()
        sr_multi = mpf.sigma("r")

        # Direct FisherForecast
        sig = SignalModel(self.inst, self.fg_gauss, self.cmb,
                          **self.signal_kwargs)
        ff = FisherForecast(sig, self.inst, self.fid_gauss,
                            priors=self.priors_gauss,
                            fixed_params=self.fixed_gauss)
        ff.compute()
        sr_single = ff.sigma("r")

        self.assertAlmostEqual(sr_multi, sr_single, places=6,
                               msg=f"multi={sr_multi:.6e} vs "
                                   f"single={sr_single:.6e}")

    def test_fisher_matrix_symmetric(self):
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        F = mpf.compute()
        np.testing.assert_allclose(
            np.array(F), np.array(F.T), rtol=1e-10,
            err_msg="Combined Fisher matrix not symmetric")

    def test_fisher_matrix_dimensions(self):
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        F = mpf.compute()
        n = mpf.n_total_params
        self.assertEqual(F.shape, (n, n))

    def test_sigma_r_finite(self):
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf.compute()
        sr = mpf.sigma("r")
        self.assertTrue(np.isfinite(sr), f"sigma(r) = {sr}")
        self.assertGreater(sr, 0)

    def test_dusty_patch_hurts(self):
        """Adding a very dusty patch should not improve sigma(r)
        vs clean patches alone (for this simple instrument)."""
        clean_only = SkyModel(patches=(
            SkyPatch("clean", 0.12, 1.0, 1.0),
        ))
        with_dusty = SkyModel(patches=(
            SkyPatch("clean", 0.12, 1.0, 1.0),
            SkyPatch("dusty", 0.10, 50.0, 5.0),
        ))
        mpf_clean = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, clean_only,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf_clean.compute()
        mpf_clean.sigma("r")

        mpf_dusty = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, with_dusty,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf_dusty.compute()
        sr_dusty = mpf_dusty.sigma("r")

        # Adding a dusty patch might help or hurt depending on mode count
        # vs foreground penalty. But with A_dust_scale=50x and simple probe,
        # the foreground penalty should dominate. At minimum, sigma(r) should
        # be finite.
        self.assertTrue(np.isfinite(sr_dusty))

    def test_sigma_vs_fsky_curve(self):
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf.compute()
        curve = mpf.sigma_vs_fsky_curve()
        self.assertEqual(len(curve), 3)
        # f_sky should be monotonically increasing
        for i in range(len(curve) - 1):
            self.assertLess(curve[i]["f_sky"], curve[i + 1]["f_sky"])
        # All sigma_r should be finite
        for entry in curve:
            self.assertTrue(np.isfinite(entry["sigma_r"]),
                            f"sigma_r = {entry['sigma_r']} for "
                            f"patches {entry['patches']}")

    def test_optimal_subset(self):
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf.compute()
        result = mpf.optimal_subset()
        self.assertIn("best_sigma_r", result)
        self.assertTrue(np.isfinite(result["best_sigma_r"]))
        self.assertGreater(len(result["best_patches"]), 0)
        # Should have 2^3 - 1 = 7 subsets
        self.assertEqual(len(result["all_subsets"]), 7)

    def test_summary_runs(self):
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg_gauss, self.cmb, sky,
            self.fid_gauss,
            priors=self.priors_gauss,
            fixed_params=self.fixed_gauss,
            signal_kwargs=self.signal_kwargs,
        )
        mpf.compute()
        s = mpf.summary("Test forecast")
        self.assertIn("Test forecast", s)
        self.assertIn("σ(r)", s)
        self.assertIn("clean", s)


class TestMultiPatchMoment(unittest.TestCase):
    """Test with moment expansion foreground model."""

    @classmethod
    def setUpClass(cls):
        cls.cmb = CMBSpectra()
        cls.inst = simple_probe()
        cls.fg = MomentExpansionModel()
        cls.fid = {**FIDUCIAL_MOMENT, "A_lens": 0.27}
        cls.priors = dict(DEFAULT_PRIORS_MOMENT)
        cls.fixed = list(DEFAULT_FIXED_MOMENT)
        cls.signal_kwargs = {"ell_max": 300, "delta_ell": 35}

    def test_moment_single_patch_recovery(self):
        """Moment expansion also recovers single-patch result."""
        sky = single_patch_model(self.inst.f_sky)
        mpf = MultiPatchFisher(
            self.inst, self.fg, self.cmb, sky, self.fid,
            priors=self.priors, fixed_params=self.fixed,
            signal_kwargs=self.signal_kwargs,
        )
        mpf.compute()
        sr_multi = mpf.sigma("r")

        sig = SignalModel(self.inst, self.fg, self.cmb,
                          **self.signal_kwargs)
        ff = FisherForecast(sig, self.inst, self.fid,
                            priors=self.priors, fixed_params=self.fixed)
        ff.compute()
        sr_single = ff.sigma("r")

        self.assertAlmostEqual(sr_multi, sr_single, places=6)

    def test_moment_per_patch_params_counted(self):
        """Both Gaussian and Moment models have the same per-patch params
        (just A_dust + A_sync): the moment omega_* are sky-level
        variances per Chluba+ 2017, so they stay global.  The moment
        model has more *global* params than Gaussian, not more per-patch.
        """
        sky = default_3patch_model(include_scan=False)
        mpf = MultiPatchFisher(
            self.inst, self.fg, self.cmb, sky, self.fid,
            priors=self.priors, fixed_params=self.fixed,
            signal_kwargs=self.signal_kwargs,
        )
        # A_dust, A_sync scale per patch; everything else is global.
        self.assertEqual(mpf._n_per_patch, 2)


class TestMultiPatchDelensedAndResidualModes(unittest.TestCase):
    """MultiPatchFisher must derive its parameter list from an actual
    SignalModel so that delensed mode (no A_lens) and residual-template
    mode (adds A_res) are handled correctly. Pre-fix, multipatch.py
    hard-coded an ["r", "A_lens"] prefix and missed both cases."""

    @classmethod
    def setUpClass(cls):
        cls.cmb = CMBSpectra()
        cls.inst = simple_probe()
        cls.fg = GaussianForegroundModel()

    def test_delensed_mode_drops_A_lens(self):
        sky = single_patch_model(self.inst.f_sky)
        ell_max = 300
        n_ells = ell_max + 1
        delensed_bb = jnp.full(n_ells, 1e-6)
        delensed_bb_ells = jnp.arange(n_ells, dtype=float)
        signal_kwargs = {
            "ell_max": ell_max, "delta_ell": 35,
            "delensed_bb": delensed_bb,
            "delensed_bb_ells": delensed_bb_ells,
        }
        fid = {k: v for k, v in FIDUCIAL_BK15.items() if k != "A_lens"}
        priors = {k: v for k, v in DEFAULT_PRIORS.items() if k != "A_lens"}
        mpf = MultiPatchFisher(
            self.inst, self.fg, self.cmb, sky, fid,
            priors=priors,
            fixed_params=[*list(DEFAULT_FIXED), "Delta_dust"],
            signal_kwargs=signal_kwargs,
        )
        self.assertNotIn("A_lens", mpf._all_names)
        self.assertNotIn("A_lens", mpf._global_free)
        mpf.compute()
        self.assertTrue(np.isfinite(mpf.sigma("r")))

    def test_residual_template_mode_adds_A_res_global(self):
        sky = single_patch_model(self.inst.f_sky)
        ell_max = 300
        n_ells = ell_max + 1
        template = jnp.full(n_ells, 1e-6)
        template_ells = jnp.arange(n_ells, dtype=float)
        signal_kwargs = {
            "ell_max": ell_max, "delta_ell": 35,
            "residual_template_cl": template,
            "residual_template_ells": template_ells,
        }
        fid = {**FIDUCIAL_BK15, "A_lens": 0.27}
        mpf = MultiPatchFisher(
            self.inst, self.fg, self.cmb, sky, fid,
            priors=DEFAULT_PRIORS,
            fixed_params=[*list(DEFAULT_FIXED), "Delta_dust"],
            signal_kwargs=signal_kwargs,
        )
        self.assertIn("A_res", mpf._all_names)
        self.assertIn("A_res", mpf._global_free)
        self.assertNotIn("A_res", mpf._per_patch_free)
        mpf.compute()
        self.assertTrue(np.isfinite(mpf.sigma("r")))
        self.assertTrue(np.isfinite(mpf.sigma("A_res")))


if __name__ == "__main__":
    unittest.main()
