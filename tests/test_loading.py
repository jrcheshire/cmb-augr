"""Tests for augr.loading — Galactic dust + synchrotron optical loading.

These exercise the functional form only (representative fiducials); the
calibrated region-mean normalizations in config.GALACTIC_LOADING are checked
separately by the calibration script's cross-check.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.loading import (
    dust_loading_occupation,
    make_galactic_extra_loading,
    sync_loading_occupation,
)
from augr.telescope import photon_noise_net, photon_noise_net_jax
from augr.units import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    occupation_from_intensity,
)

# Representative (not calibrated) Galactic-loading parameters for magnitude
# tests. τ_353 ~ high-latitude GNILC scale; sync anchored Haslam-style.
FID = dict(
    tau_353=1.2e-5,
    beta_d=1.53,
    T_d=19.6,
    T_sync_ref_K=20.0,
    beta_s=-3.1,
    sync_nu_ref_ghz=0.408,
    dust_nu_ref_ghz=353.0,
)


def _planck_bnu(nu_hz, T):
    """Total (both-pol) blackbody specific intensity B_ν(T) [SI]."""
    x = H_PLANCK * nu_hz / (K_BOLTZMANN * T)
    return 2.0 * H_PLANCK * nu_hz**3 / C_LIGHT**2 / (np.exp(x) - 1.0)


class TestOccupationFromIntensity:
    """The intensity→occupation primitive is the exact inverse of B_ν."""

    def test_roundtrip_recovers_bose_einstein(self):
        nu = np.linspace(20e9, 700e9, 50)
        for T in (2.7255, 4.0, 19.6):
            n_direct = 1.0 / (np.exp(H_PLANCK * nu / (K_BOLTZMANN * T)) - 1.0)
            n_via = occupation_from_intensity(nu, _planck_bnu(nu, T))
            np.testing.assert_allclose(np.asarray(n_via), n_direct, rtol=1e-12)


class TestDustOccupation:
    def test_formula(self):
        nu = np.linspace(30e9, 700e9, 40)
        got = np.asarray(dust_loading_occupation(nu, FID["tau_353"], FID["beta_d"], FID["T_d"]))
        nu_ref = 353e9
        want = (
            FID["tau_353"]
            * (nu / nu_ref) ** FID["beta_d"]
            / (np.exp(H_PLANCK * nu / (K_BOLTZMANN * FID["T_d"])) - 1.0)
        )
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_equals_intensity_path(self):
        """Closed form == occupation_from_intensity of the GNILC MBB intensity."""
        nu = np.linspace(30e9, 700e9, 40)
        nu_ref = 353e9
        I_nu = FID["tau_353"] * (nu / nu_ref) ** FID["beta_d"] * _planck_bnu(nu, FID["T_d"])
        via_intensity = np.asarray(occupation_from_intensity(nu, I_nu))
        closed = np.asarray(dust_loading_occupation(nu, FID["tau_353"], FID["beta_d"], FID["T_d"]))
        np.testing.assert_allclose(closed, via_intensity, rtol=1e-12)


class TestSyncOccupation:
    def test_formula(self):
        nu = np.linspace(20e9, 400e9, 30)
        got = np.asarray(
            sync_loading_occupation(nu, FID["T_sync_ref_K"], FID["beta_s"], FID["sync_nu_ref_ghz"])
        )
        nu_ref = FID["sync_nu_ref_ghz"] * 1e9
        T_rj = FID["T_sync_ref_K"] * (nu / nu_ref) ** FID["beta_s"]
        want = K_BOLTZMANN * T_rj / (H_PLANCK * nu)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_negligible_vs_dust_in_band(self):
        """Synchrotron loading is orders of magnitude below dust at 150 GHz."""
        nu = np.array([150e9])
        n_sync = float(
            sync_loading_occupation(nu, FID["T_sync_ref_K"], FID["beta_s"], FID["sync_nu_ref_ghz"])[
                0
            ]
        )
        n_dust = float(dust_loading_occupation(nu, FID["tau_353"], FID["beta_d"], FID["T_d"])[0])
        assert n_sync < 1e-2 * n_dust


class TestNetInflation:
    """Loading is negligible at 150 GHz and sizeable in the submm."""

    def _ratio(self, nu_ghz):
        extra = make_galactic_extra_loading(**FID)
        net0 = photon_noise_net(nu_ghz)
        net1 = photon_noise_net(nu_ghz, extra_loading=extra)
        return net1 / net0

    def test_negligible_at_150(self):
        assert self._ratio(150.0) - 1.0 < 5e-3

    def test_monotonic_and_sizeable_in_submm(self):
        r150 = self._ratio(150.0)
        r340 = self._ratio(340.0)
        r600 = self._ratio(600.0)
        assert r150 < r340 < r600
        # 600 GHz dust loading is a real O(few–tens %) NET penalty, not a bug.
        assert 1.01 < r600 < 2.0


class TestDifferentiability:
    """The loading callable is jnp-traceable through the NET forward."""

    def test_grad_through_photon_noise_net_jax(self):
        extra = make_galactic_extra_loading(**FID)

        def f(eta):
            return photon_noise_net_jax(jnp.asarray(600.0), eta_optical=eta, extra_loading=extra)

        val = f(0.35)
        g = jax.grad(f)(0.35)
        assert np.isfinite(float(val))
        assert np.isfinite(float(g))

    def test_jax_matches_numpy_path(self):
        extra = make_galactic_extra_loading(**FID)
        for nu in (40.0, 150.0, 340.0, 600.0):
            n_np = photon_noise_net(nu, extra_loading=extra)
            n_jax = float(photon_noise_net_jax(jnp.asarray(nu), extra_loading=extra))
            assert n_np == pytest.approx(n_jax, rel=1e-6)
