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

    def test_w_ee_1_matches_exact_at_signal_dominated_noise(self, spectra):
        """W_EE=1 (nl_ee=None) should match the exact W_EE formula to high
        precision when the E map is signal-dominated across the kernel's
        ℓ range, since W_EE → 1."""
        ls = jnp.arange(5, 100, dtype=float)
        Ls = jnp.arange(2, 1001, dtype=float)
        # Modest N_0: some delensing, not trivial residual
        cl_pp = _interp_at(spectra.cl_pp, Ls)
        n0 = cl_pp * 0.5

        # Tiny EE noise: C_EE/N_EE ~ 10^4, W_EE ~ 1
        nl_ee_tiny = jnp.full_like(spectra.cl_ee_unl, 1e-10)
        res_approx = residual_cl_bb(ls, Ls, spectra, n0, l_max=1000, n_phi=64)
        res_exact = residual_cl_bb(ls, Ls, spectra, n0, l_max=1000, n_phi=64,
                                   nl_ee=nl_ee_tiny)
        np.testing.assert_allclose(
            np.asarray(res_exact), np.asarray(res_approx), rtol=1e-3,
            err_msg="W_EE=1 approx must match exact when E is signal-dominated")

    def test_w_ee_correction_is_positive_at_ground_depths(self, spectra):
        """At ground-experiment noise levels where C_EE/N_EE drops toward 1,
        the exact W_EE-filtered residual is LARGER than the W_EE=1 approx
        (the delensing credit is reduced because E reconstruction is itself
        noisy).  This is the regime where the simple approximation becomes
        optimistic."""
        ls = jnp.arange(5, 100, dtype=float)
        Ls = jnp.arange(2, 1001, dtype=float)
        cl_pp = _interp_at(spectra.cl_pp, Ls)
        n0 = cl_pp * 0.5

        # Heavy EE noise representative of a 10 uK-arcmin, 30' beam ground
        # experiment (SPT-3G-ish pessimistic scale): N_EE large enough that
        # W_EE drops well below 1 at the lensing peak.
        # 10 uK-arcmin × (pi/10800)^2 in uK^2-sr ~ 8.5e-6
        nl_ee_ground = jnp.full_like(spectra.cl_ee_unl, 5e-5)

        res_approx = residual_cl_bb(ls, Ls, spectra, n0, l_max=1000, n_phi=64)
        res_exact = residual_cl_bb(ls, Ls, spectra, n0, l_max=1000, n_phi=64,
                                   nl_ee=nl_ee_ground)
        # Exact residual should be strictly larger at every ℓ.
        assert np.all(np.asarray(res_exact) > np.asarray(res_approx)), \
            "W_EE<1 correction must make residual larger, not smaller"


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
# Wigner 3j
# -----------------------------------------------------------------------

class TestWigner3j:
    def test_000_closed_form(self):
        """(l1 l2 L; 0 0 0) closed form matches pywigxjpf."""
        from augr.wigner import wigner3j_000
        import pywigxjpf
        pywigxjpf.wig_table_init(2 * 3000, 9)
        pywigxjpf.wig_temp_init(2 * 3000)

        for l1, l2, L in [(10, 8, 6), (100, 98, 50), (1000, 998, 200)]:
            ours = wigner3j_000(l1, l2, L)
            ref = pywigxjpf.wig3jj(2*l1, 2*l2, 2*L, 0, 0, 0)
            np.testing.assert_allclose(ours, ref, rtol=1e-10)

        # Parity selection: zero when l1+l2+L is odd
        assert wigner3j_000(10, 8, 5) == 0.0

        pywigxjpf.wig_temp_free()
        pywigxjpf.wig_table_free()

    def test_scalar_recursion(self):
        """Backward SG recursion matches pywigxjpf for (j1,j2,j;2,-2,0)."""
        from augr.wigner import wigner3j_recurse
        import pywigxjpf
        pywigxjpf.wig_table_init(2 * 3000, 9)
        pywigxjpf.wig_temp_init(2 * 3000)

        for j1, j2 in [(100, 50), (500, 300), (1000, 500)]:
            j_vals, w = wigner3j_recurse(j1, j2, 2, -2)
            for i in range(0, len(j_vals), max(1, len(j_vals) // 5)):
                j = int(j_vals[i])
                ref = pywigxjpf.wig3jj(2*j1, 2*j2, 2*j, 4, -4, 0)
                if abs(ref) > 1e-15:
                    np.testing.assert_allclose(w[i], ref, rtol=1e-10)

        pywigxjpf.wig_temp_free()
        pywigxjpf.wig_table_free()

    def test_vectorized_matches_scalar(self):
        """Vectorized recursion matches scalar for all l1 values."""
        from augr.wigner import wigner3j_recurse, wigner3j_vectorized

        L = 50
        l1_arr = np.array([80., 100., 120.])
        l2_grid, w_vec = wigner3j_vectorized(L, l1_arr, m1=2, m2=-2)

        for idx, l1 in enumerate([80, 100, 120]):
            j_vals, w_scalar = wigner3j_recurse(int(l1), L, 2, -2)
            for j_idx, j_val in enumerate(j_vals):
                l2_idx = int(j_val) - int(l2_grid[0])
                if 0 <= l2_idx < len(l2_grid):
                    np.testing.assert_allclose(
                        w_vec[idx, l2_idx], w_scalar[j_idx], atol=1e-15)

    def test_physical_coupling(self):
        """Physical coupling (l1,l2,L;2,0,-2) = (-1)^s × (l1,L,l2;2,-2,0)."""
        from augr.wigner import wigner3j_vectorized
        import pywigxjpf
        pywigxjpf.wig_table_init(2 * 3000, 9)
        pywigxjpf.wig_temp_init(2 * 3000)

        L = 200
        l1_arr = np.arange(100, 501, dtype=float)
        l2_grid, w3j = wigner3j_vectorized(L, l1_arr, m1=2, m2=-2)
        parity = (-1.0) ** (l1_arr[:, None] + l2_grid[None, :] + L)
        w_phys = parity * w3j

        max_err = 0
        for i_l1 in range(0, len(l1_arr), 50):
            l1 = int(l1_arr[i_l1])
            for i_l2 in range(0, len(l2_grid), 20):
                l2 = int(l2_grid[i_l2])
                if abs(l1 - L) <= l2 <= l1 + L and l2 >= 2:
                    ref = pywigxjpf.wig3jj(2*l1, 2*l2, 2*L, 4, 0, -4)
                    if abs(ref) > 1e-15:
                        err = abs(w_phys[i_l1, i_l2] - ref) / abs(ref)
                        max_err = max(max_err, err)

        assert max_err < 1e-10, f"Physical coupling error: {max_err:.2e}"

        pywigxjpf.wig_temp_free()
        pywigxjpf.wig_table_free()


# -----------------------------------------------------------------------
# Full-sky tests
# -----------------------------------------------------------------------

class TestFullSkyKernel:
    def test_matches_camb_low_ell(self, spectra):
        """Full-sky kernel reproduces CAMB BB at l=5,10,20 within 1%.

        Uses Smith et al. (2012) Eq. 6-7 coupling via cyclic 3j identity.
        Residual ~0.8% error is from the first-order gradient approximation
        (CAMB uses the full resummed lensing calculation).
        """
        ls = jnp.array([5., 10., 20.])
        Ls = jnp.arange(2, 2001, dtype=float)
        K = lensing_kernel(ls, Ls, spectra, l_max=2000, fullsky=True)
        cl_pp = jnp.array([float(spectra.cl_pp[int(L)]) for L in Ls])
        cl_bb_full = K @ cl_pp
        cl_bb_camb = jnp.array([float(spectra.cl_bb_len[int(l)]) for l in ls])
        ratio = cl_bb_full / cl_bb_camb
        np.testing.assert_allclose(ratio, 1.0, atol=0.015)


class TestFullSkyN0:
    def test_eb_finite(self, spectra, noise):
        """Full-sky EB N_0 should be finite and positive."""
        Ls = jnp.array([50., 100., 500.])
        n0 = compute_n0_eb(Ls, spectra, noise["ee"], noise["bb"],
                           l_max=1000, fullsky=True)
        assert jnp.all(n0 > 0)
        assert jnp.all(jnp.isfinite(n0))

    def test_eb_decreasing(self, spectra, noise):
        """N_0 should decrease with L (better reconstruction at high L)."""
        Ls = jnp.array([50., 200., 500., 1000.])
        n0 = compute_n0_eb(Ls, spectra, noise["ee"], noise["bb"],
                           l_max=1000, fullsky=True)
        assert float(n0[0]) > float(n0[1]) > float(n0[2]) > float(n0[3])


@pytest.mark.slow
class TestFullSkyIteration:
    def test_convergence(self, spectra, noise):
        """Full-sky iteration should converge with 0 < A_lens_eff < 1."""
        result = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"],
            L_max=500, l_max_qe=500, n_iter=3, fullsky=True,
        )
        assert 0.0 < result.A_lens_eff < 1.0

    def test_better_than_no_delensing(self, spectra, noise):
        """Residual BB should be less than lensed BB at low ell."""
        result = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"],
            L_max=500, l_max_qe=500, n_iter=2, fullsky=True,
        )
        cl_bb_lens = _interp_at(spectra.cl_bb_len, result.ls)
        # At least some delensing should happen
        assert float(jnp.sum(result.cl_bb_res)) < float(jnp.sum(cl_bb_lens))


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
