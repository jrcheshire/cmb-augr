"""Tests for augr.crosslinks - L2 year-averaged spin coefficients h_k."""

from __future__ import annotations

import unittest

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from augr.crosslinks import h_k_map, pack_cos_sin, yearavg_h_k_1d


class TestYearAvgHK1D(unittest.TestCase):

    def test_returns_complex128(self):
        out = yearavg_h_k_1d(jnp.array([np.radians(33.0)]), k=2)
        self.assertEqual(out.dtype, jnp.complex128)

    def test_scalar_input_works(self):
        # JAX 0-d arrays should work; check shape is preserved.
        out = yearavg_h_k_1d(jnp.array(np.radians(60.0)), k=2)
        self.assertEqual(out.shape, ())

    def test_array_input_shape(self):
        thetas = jnp.linspace(np.radians(20.0), np.radians(160.0), 17)
        out = yearavg_h_k_1d(thetas, k=2)
        self.assertEqual(out.shape, (17,))

    def test_amplitude_bound(self):
        """|h_k| <= 1 on the observed sky (it's an average of unit complex).

        Sample the bulk of the LiteBIRD-config support and check
        magnitudes are in [0, 1] up to a small numerical slack from
        the Chebyshev quadrature.
        """
        thetas = jnp.linspace(np.radians(15.0), np.radians(165.0), 25)
        for k in (1, 2, 4):
            h = yearavg_h_k_1d(thetas, k=k)
            np.testing.assert_array_less(jnp.abs(h), 1.0 + 1e-3)

    def test_even_k_imag_near_zero(self):
        """Even k: phase prefactor is real, so h_k is real."""
        thetas = jnp.linspace(np.radians(15.0), np.radians(165.0), 25)
        for k in (2, 4):
            h = yearavg_h_k_1d(thetas, k=k)
            # Imag should be exactly 0 mod floating-point noise.
            np.testing.assert_allclose(jnp.imag(h), 0.0, atol=1e-12)

    def test_odd_k_real_near_zero(self):
        """Odd k: phase prefactor is imaginary, so h_k is imaginary."""
        thetas = jnp.linspace(np.radians(15.0), np.radians(165.0), 25)
        for k in (1, 3):
            h = yearavg_h_k_1d(thetas, k=k)
            np.testing.assert_allclose(jnp.real(h), 0.0, atol=1e-12)

    def test_litebird_reference_values(self):
        """Spot-check against the Falcons-validated LiteBIRD numbers.

        Values come from the validation pipeline at
        scripts/falcons_validation/, which agreed with Falcons.jl to
        ~0.001 absolute. Tolerance here is 0.005 to allow for
        Chebyshev convergence with n_quad=200.
        """
        theta = jnp.array([np.radians(33.0), np.radians(60.0), np.radians(90.0)])
        # spin=50, prec=45 are the augr defaults (LiteBIRD config).
        h_1 = yearavg_h_k_1d(theta, k=1)
        h_2 = yearavg_h_k_1d(theta, k=2)
        h_4 = yearavg_h_k_1d(theta, k=4)
        # Reference values from compare_yearavg.py at n_quad=4001 (scipy.quad).
        np.testing.assert_allclose(jnp.imag(h_1), [-0.2410, -0.1898, 0.0], atol=0.005)
        np.testing.assert_allclose(jnp.real(h_2), [0.3841, 0.2649, -0.0466], atol=0.005)
        np.testing.assert_allclose(jnp.real(h_4), [0.1814, -0.0263, -0.2896], atol=0.005)

    def test_north_south_symmetry(self):
        """h_k under theta_ecl -> pi - theta_ecl: even k same, odd k flips sign.

        Year-averaged map is symmetric across the ecliptic equator
        (both hemispheres see the scan equally). For even k the real
        h_k is symmetric; for odd k the imaginary h_k is antisymmetric.
        """
        thetas = jnp.array([np.radians(t) for t in (20, 30, 50, 70)])
        thetas_mirror = jnp.pi - thetas
        for k in (1, 2, 4):
            h = yearavg_h_k_1d(thetas, k=k)
            h_mirror = yearavg_h_k_1d(thetas_mirror, k=k)
            if k % 2 == 0:
                np.testing.assert_allclose(jnp.real(h), jnp.real(h_mirror), atol=1e-3)
            else:
                np.testing.assert_allclose(jnp.imag(h), -jnp.imag(h_mirror), atol=1e-3)

    def test_unsupported_input_returns_nan(self):
        """theta_ecl outside the precession+spin support gives NaN.

        With spin=20, prec=10, the support [|prec - spin|, prec + spin]
        is theta_ecl in [10 deg, 30 deg] ish; theta_ecl = 60 deg is well
        outside.
        """
        out = yearavg_h_k_1d(
            jnp.array([np.radians(60.0)]),
            spin_angle_deg=20.0,
            precession_angle_deg=10.0,
            k=2,
        )
        self.assertTrue(jnp.all(jnp.isnan(jnp.real(out))))

    def test_invalid_k_raises(self):
        for bad_k in (0, -1, 1.5, "two"):
            with self.assertRaises((ValueError, TypeError)):
                yearavg_h_k_1d(jnp.array([np.radians(60.0)]), k=bad_k)

    def test_jax_grad_runs(self):
        """jax.grad with respect to spin_angle_deg returns a finite number.

        Differentiability is the design goal of the Chebyshev
        substitution; this just sanity-checks that nothing in the
        chain breaks under autodiff.
        """
        def g(spin_deg):
            return jnp.real(
                yearavg_h_k_1d(
                    jnp.array(np.radians(60.0)),
                    spin_angle_deg=spin_deg,
                    k=2,
                )
            )
        d = float(jax.grad(g)(50.0))
        self.assertTrue(np.isfinite(d))


class TestHKMap(unittest.TestCase):

    def test_smoke(self):
        m = h_k_map(nside=8, k=2)
        self.assertEqual(m.shape, (hp.nside2npix(8),))
        self.assertEqual(m.dtype, jnp.complex128)

    def test_ecliptic_frame_consistency(self):
        """coord='E' map equals yearavg_h_k_1d at each pixel's colatitude.

        No rotation in the ecliptic-frame path; map is literally the
        analytic 1-D function evaluated on the HEALPix grid.
        """
        nside = 16
        m = h_k_map(nside=nside, coord="E", k=2)
        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        expected = yearavg_h_k_1d(jnp.asarray(theta), k=2)
        # Comparing complex arrays; allow tiny numerical noise.
        np.testing.assert_allclose(np.asarray(m), np.asarray(expected), atol=1e-12)

    def test_galactic_rotation_changes_map(self):
        """coord='G' is not the same as coord='E' for non-trivial colatitudes."""
        m_e = np.asarray(h_k_map(nside=16, coord="E", k=2))
        m_g = np.asarray(h_k_map(nside=16, coord="G", k=2))
        # They must differ in at least some pixels (they describe the
        # same h_k field viewed in different frames).
        self.assertFalse(np.allclose(m_e, m_g, equal_nan=True))

    def test_coord_invalid_raises(self):
        with self.assertRaises(ValueError):
            h_k_map(nside=8, coord="X")

    def test_extreme_planck_config_runs(self):
        """spin=85, prec=7.5 (Planck-extreme) runs without numerical failure.

        Reaches close to the singular limit where prec is small and
        the Chebyshev support is tight; we only verify it runs and
        the bulk of the sky has finite values, not the absolute
        accuracy (the validation pipeline shows pole regions deviate
        from Falcons under this config because of non-ergodicity).
        """
        m = h_k_map(nside=16, spin_angle_deg=85.0, precession_angle_deg=7.5,
                    k=2, coord="E")
        # The observed band is theta_ecl in [|prec-spin|, prec+spin] = [77.5, 92.5];
        # equator colatitude theta=90 should yield finite h_k.
        npix = hp.nside2npix(16)
        theta, _ = hp.pix2ang(16, np.arange(npix))
        eq_band = (theta > np.radians(85)) & (theta < np.radians(95))
        self.assertGreater(eq_band.sum(), 5)
        finite = np.isfinite(np.real(np.asarray(m[eq_band])))
        self.assertTrue(np.all(finite))


class TestPackCosSin(unittest.TestCase):

    def test_basic_combination(self):
        """pack_cos_sin returns cos - 1j sin (Falcons / Wallis convention)."""
        c = np.array([0.5, 0.3, -0.2])
        s = np.array([0.1, -0.4, 0.7])
        out = pack_cos_sin(c, s)
        np.testing.assert_allclose(np.real(out), c)
        np.testing.assert_allclose(np.imag(out), -s)

    def test_returns_complex128(self):
        out = pack_cos_sin(np.array([0.5], dtype=np.float32),
                           np.array([0.1], dtype=np.float32))
        self.assertEqual(out.dtype, jnp.complex128)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pack_cos_sin(np.array([1.0, 2.0]), np.array([1.0]))

    def test_round_trip_with_h_k_map(self):
        """Splitting an h_k_map output and packing back gives the original."""
        m = h_k_map(nside=8, coord="E", k=2)
        m_np = np.asarray(m)
        # Drop NaN pixels for the round-trip check.
        finite = np.isfinite(np.real(m_np))
        cos_map = np.real(m_np)[finite]
        sin_map = -np.imag(m_np)[finite]
        repacked = pack_cos_sin(cos_map, sin_map)
        np.testing.assert_allclose(np.asarray(repacked), m_np[finite], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
