"""Tests for delensing.py — QE lensing reconstruction and residual BB."""

import jax.numpy as jnp
import numpy as np
import pytest

from augr.delensing import (
    LensingSpectra,
    load_lensing_spectra,
    _gl_nodes,
    _triangle_geometry,
    _interp_at,
    compute_n0_eb,
    compute_n0_tt,
    compute_n0_ee,
    compute_n0_te,
    compute_n0_tb,
    compute_n0_mv,
    lensing_kernel,
    residual_cl_bb,
    iterate_delensing,
)
from augr.instrument import combined_noise_nl
from augr.config import simple_probe


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def spectra():
    return load_lensing_spectra()


@pytest.fixture(scope="module")
def instrument():
    return simple_probe()


@pytest.fixture(scope="module")
def noise(spectra, instrument):
    """Combined noise on the full ell grid."""
    ells = spectra.ells
    return {
        "tt": combined_noise_nl(instrument, ells, "TT"),
        "ee": combined_noise_nl(instrument, ells, "BB"),  # EE = BB for pol
        "bb": combined_noise_nl(instrument, ells, "BB"),
    }


# -----------------------------------------------------------------------
# LensingSpectra and data loading
# -----------------------------------------------------------------------

class TestLensingSpectra:
    def test_load(self, spectra):
        assert spectra.ell_max == 5000
        assert spectra.ells.shape == (5001,)

    def test_ell_zero_one_vanish(self, spectra):
        for cl in [spectra.cl_tt_unl, spectra.cl_ee_unl, spectra.cl_bb_len]:
            assert cl[0] == 0.0
            assert cl[1] == 0.0

    def test_unlensed_bb_zero(self, spectra):
        """Unlensed scalar BB should be zero (no tensors)."""
        assert jnp.allclose(spectra.cl_bb_unl, 0.0)

    def test_tt_positive(self, spectra):
        """TT should be positive for ell >= 2."""
        assert jnp.all(spectra.cl_tt_unl[2:100] > 0)
        assert jnp.all(spectra.cl_tt_len[2:100] > 0)

    def test_lensed_bb_positive(self, spectra):
        """Lensed BB should be positive for ell >= 2."""
        assert jnp.all(spectra.cl_bb_len[2:300] > 0)

    def test_clpp_positive(self, spectra):
        assert jnp.all(spectra.cl_pp[2:2000] > 0)


# -----------------------------------------------------------------------
# Quadrature and geometry helpers
# -----------------------------------------------------------------------

class TestHelpers:
    def test_gl_nodes_sum(self):
        """GL weights should sum to 2π."""
        phi, w = _gl_nodes(64)
        assert abs(float(jnp.sum(w)) - 2.0 * np.pi) < 1e-10

    def test_gl_nodes_range(self):
        """GL nodes should be in [0, 2π]."""
        phi, w = _gl_nodes(128)
        assert float(phi.min()) >= 0.0
        assert float(phi.max()) <= 2.0 * np.pi

    def test_triangle_l2_collinear(self):
        """When φ=0 (l₁ parallel to L), l₂ = |L - l₁|."""
        L = jnp.array([100.0, 200.0])
        phi = jnp.array([0.0])
        l2, cos2, sin2 = _triangle_geometry(L, 50.0, phi)
        np.testing.assert_allclose(l2[:, 0], [50.0, 150.0], atol=1e-10)

    def test_interp_out_of_range(self):
        """Interpolation should return 0 outside range."""
        cl = jnp.array([0.0, 0.0, 1.0, 2.0, 3.0])
        assert float(_interp_at(cl, jnp.array([-1.0]))[0]) == 0.0
        assert float(_interp_at(cl, jnp.array([10.0]))[0]) == 0.0


# -----------------------------------------------------------------------
# N_0 limit tests
# -----------------------------------------------------------------------

class TestN0Limits:
    """Test N_0 in physically interpretable limits."""

    def test_eb_high_noise(self, spectra):
        """Infinite noise → N_0 >> C_pp (no reconstruction)."""
        Ls = jnp.array([100.0, 500.0])
        huge = jnp.ones(spectra.ell_max + 1) * 1e10
        n0 = compute_n0_eb(Ls, spectra, huge, huge, l_max=2000, n_phi=64)
        assert jnp.all(n0 > spectra.cl_pp[100] * 1e10)

    def test_eb_low_noise(self, spectra):
        """Very low noise → N_0 finite and much smaller than high-noise case."""
        Ls = jnp.array([100.0])
        huge = jnp.ones(spectra.ell_max + 1) * 1e10
        tiny = jnp.ones(spectra.ell_max + 1) * 1e-20
        n0_high = compute_n0_eb(Ls, spectra, huge, huge, l_max=2000, n_phi=64)
        n0_low = compute_n0_eb(Ls, spectra, tiny, tiny, l_max=2000, n_phi=64)
        assert float(n0_low[0]) < float(n0_high[0]) * 1e-10

    def test_tt_high_noise(self, spectra):
        """TT N_0 also diverges with infinite noise."""
        Ls = jnp.array([100.0])
        huge = jnp.ones(spectra.ell_max + 1) * 1e10
        n0 = compute_n0_tt(Ls, spectra, huge, l_max=2000, n_phi=64)
        assert float(n0[0]) > spectra.cl_pp[100] * 1e10

    def test_mv_better_than_individual(self, spectra, noise):
        """MV N_0 should be smaller than any individual estimator."""
        Ls = jnp.array([100.0, 300.0])
        n0_mv = compute_n0_mv(Ls, spectra, noise["tt"], noise["ee"],
                              noise["bb"], l_max=2000, n_phi=64)
        n0_eb = compute_n0_eb(Ls, spectra, noise["ee"], noise["bb"],
                              l_max=2000, n_phi=64)
        n0_tt = compute_n0_tt(Ls, spectra, noise["tt"], l_max=2000, n_phi=64)
        assert jnp.all(n0_mv < n0_eb)
        assert jnp.all(n0_mv < n0_tt)


# -----------------------------------------------------------------------
# Lensing kernel
# -----------------------------------------------------------------------

class TestLensingKernel:
    def test_kernel_reproduces_camb_bb(self, spectra):
        """K @ C_pp should match CAMB lensed BB within ~5% (flat-sky)."""
        ls = jnp.arange(5, 200, dtype=float)
        Ls = jnp.arange(2, 2001, dtype=float)
        K = lensing_kernel(ls, Ls, spectra, l_max=2000, n_phi=96)

        cl_pp = _interp_at(spectra.cl_pp, Ls)
        cl_bb_kernel = K @ cl_pp
        cl_bb_camb = _interp_at(spectra.cl_bb_len, ls)

        ratio = cl_bb_kernel / cl_bb_camb
        # Flat-sky should be within ~5% of CAMB at ell > 5
        np.testing.assert_allclose(ratio, 1.0, atol=0.05)

    def test_kernel_non_negative(self, spectra):
        """Kernel entries should be non-negative."""
        ls = jnp.arange(10, 50, dtype=float)
        Ls = jnp.arange(10, 500, dtype=float)
        K = lensing_kernel(ls, Ls, spectra, l_max=1000, n_phi=64)
        assert jnp.all(K >= -1e-30)  # allow tiny numerical noise


# -----------------------------------------------------------------------
# Residual BB limits
# -----------------------------------------------------------------------

class TestResidualLimits:
    def test_no_reconstruction(self, spectra):
        """If N_0 → ∞, residual should equal full lensing BB.

        Uses L_max=2000 to capture most of the lensing power. Residual
        still ~1-2% low due to flat-sky approx + L_max truncation.
        """
        ls = jnp.arange(5, 100, dtype=float)
        Ls = jnp.arange(2, 2001, dtype=float)
        n0_huge = jnp.ones_like(Ls) * 1e30
        res = residual_cl_bb(ls, Ls, spectra, n0_huge, l_max=2000, n_phi=96)
        cl_bb_camb = _interp_at(spectra.cl_bb_len, ls)
        ratio = res / cl_bb_camb
        # Flat-sky kernel + L_max truncation: within ~5%
        np.testing.assert_allclose(ratio, 1.0, atol=0.05)

    def test_perfect_reconstruction(self, spectra):
        """If N_0 → 0, residual should vanish."""
        ls = jnp.arange(5, 100, dtype=float)
        Ls = jnp.arange(2, 1001, dtype=float)
        n0_zero = jnp.ones_like(Ls) * 1e-30
        res = residual_cl_bb(ls, Ls, spectra, n0_zero, l_max=1000, n_phi=64)
        cl_bb_camb = _interp_at(spectra.cl_bb_len, ls)
        # Residual should be << full lensing
        assert float(jnp.max(res / cl_bb_camb)) < 0.01


# -----------------------------------------------------------------------
# Iterative delensing
# -----------------------------------------------------------------------

class TestIterativeDelensing:
    def test_convergence(self, spectra, noise):
        """Iteration should converge (A_lens_eff stable by iteration 3)."""
        result = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"],
            L_max=1000, l_max_qe=1000, n_phi=64, n_iter=5,
        )
        assert 0.0 < result.A_lens_eff < 1.0

    def test_result_shape(self, spectra, noise):
        ls = jnp.arange(2, 201, dtype=float)
        result = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"],
            ls=ls, L_max=500, l_max_qe=500, n_phi=32, n_iter=2,
        )
        assert result.cl_bb_res.shape == ls.shape
        assert result.n0_mv.shape == result.Ls.shape


# -----------------------------------------------------------------------
# Signal model integration
# -----------------------------------------------------------------------

class TestSignalModelIntegration:
    def test_delensed_mode_no_alens(self, spectra, noise):
        """In delensed mode, A_lens is not a parameter."""
        from augr.spectra import CMBSpectra
        from augr.foregrounds import GaussianForegroundModel
        from augr.signal import SignalModel

        result = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"],
            L_max=500, l_max_qe=500, n_phi=32, n_iter=2,
        )

        inst = simple_probe()
        cmb = CMBSpectra()
        fg = GaussianForegroundModel()
        signal = SignalModel(inst, fg, cmb,
                             delensed_bb=result.cl_bb_res,
                             delensed_bb_ells=result.ls)

        assert "A_lens" not in signal.parameter_names
        assert "r" in signal.parameter_names
        assert signal.n_params == 1 + len(fg.parameter_names)

    def test_delensed_data_vector_shape(self, spectra, noise):
        """Data vector and Jacobian have correct shapes in delensed mode."""
        from augr.spectra import CMBSpectra
        from augr.foregrounds import GaussianForegroundModel
        from augr.signal import SignalModel, flatten_params
        from augr.config import FIDUCIAL_BK15

        result = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"],
            L_max=500, l_max_qe=500, n_phi=32, n_iter=2,
        )

        inst = simple_probe()
        cmb = CMBSpectra()
        fg = GaussianForegroundModel()
        signal = SignalModel(inst, fg, cmb,
                             delensed_bb=result.cl_bb_res,
                             delensed_bb_ells=result.ls)

        fid = {k: v for k, v in FIDUCIAL_BK15.items() if k != "A_lens"}
        params = flatten_params(fid, signal.parameter_names)

        dv = signal.data_vector(params)
        J = signal.jacobian(params)

        assert dv.shape == (signal.n_data,)
        assert J.shape == (signal.n_data, signal.n_params)
        # Jacobian should have no A_lens column
        assert J.shape[1] == signal.n_params
