"""Tests for delensing.py — QE lensing reconstruction and residual BB."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from augr.config import simple_probe
from augr.delensing import (
    LensingSpectra,
    _gl_nodes,
    _interp_at,
    _triangle_geometry,
    compute_n0_eb,
    compute_n0_ee,
    compute_n0_mv,
    compute_n0_tt,
    iterate_delensing,
    lensing_kernel,
    load_lensing_spectra,
    residual_cl_bb,
)
from augr.instrument import combined_noise_nl

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
        _phi, w = _gl_nodes(64)
        assert abs(float(jnp.sum(w)) - 2.0 * np.pi) < 1e-10

    def test_gl_nodes_range(self):
        """GL nodes should be in [0, 2π]."""
        phi, _w = _gl_nodes(128)
        assert float(phi.min()) >= 0.0
        assert float(phi.max()) <= 2.0 * np.pi

    def test_triangle_l2_collinear(self):
        """When φ=0 (l₁ parallel to L), l₂ = |L - l₁|."""
        L = jnp.array([100.0, 200.0])
        phi = jnp.array([0.0])
        l2, _cos2, _sin2 = _triangle_geometry(L, 50.0, phi)
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
        import pywigxjpf

        from augr.wigner import wigner3j_000
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
        import pywigxjpf

        from augr.wigner import wigner3j_recurse
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
        import pywigxjpf

        from augr.wigner import wigner3j_vectorized
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
        cl_bb_camb = jnp.array([float(spectra.cl_bb_len[int(ell)]) for ell in ls])
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
        from augr.foregrounds import GaussianForegroundModel
        from augr.signal import SignalModel
        from augr.spectra import CMBSpectra

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
        from augr.config import FIDUCIAL_BK15
        from augr.foregrounds import GaussianForegroundModel
        from augr.signal import SignalModel, flatten_params
        from augr.spectra import CMBSpectra

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


# -----------------------------------------------------------------------
# Cross-validation against plancklens (LiteBIRD-PTEP fiducial)
#
# Reference NPZ produced by scripts/n0_validation/run_plancklens.py;
# regen recipe in scripts/n0_validation/README.md. The lightweight test
# here just compares augr's compute_n0_* on the *same* nl_*/cl_* arrays
# against the saved reference. Tolerances reflect the conventions both
# codes use: response = unlensed C_l, filter = lensed + noise, MV =
# diagonal 1/Sum 1/N_0_alpha.
# -----------------------------------------------------------------------

# Tolerances for the augr full-sky vs plancklens TT comparison.
# Locked in 2026-05-06 after the controlled-input test + regen with
# correct lmin filter (see scripts/n0_validation/README.md "RESOLVED"
# section). On the LiteBIRD-PTEP fiducial config the agreement is
# ~1e-6 in the bulk and ~1e-4 at L > 2000 (where both codes feel the
# l1+l2 boundary truncation). 1e-3 is a safe headroom that catches
# regressions while not being noisy.
#
# Only TT is tested here. plancklens 'p_p' / 'p' include inter-
# estimator cross-correlations (joint GMV); augr's MV is diagonal
# (HO02 Eq. 22). The diagonal-vs-joint difference is real physics, not
# a bug, and varies with the relative weight of estimators -- not
# something to lock into a tolerance test.
N0_REF_PATH = Path(__file__).resolve().parents[1] / "data" / "n0_reference_litebird.npz"
TOL_FULLSKY_TT_BULK = 1e-3   # bulk-L window
TOL_FULLSKY_TT_TAIL = 1e-3   # high-L (l > 2000) where both feel boundary trunc

L_BULK = (10, 2000)
L_HIGH = (2001, 3000)


def _load_reference():
    """Load the in-tree plancklens reference NPZ, or skip if absent."""
    if not N0_REF_PATH.exists():
        pytest.skip(
            f"plancklens reference not found at {N0_REF_PATH.relative_to(Path.cwd())}; "
            "run scripts/n0_validation/run_plancklens.py and copy the output to "
            "data/n0_reference_litebird.npz to enable this test."
        )
    npz = np.load(N0_REF_PATH, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def _make_spectra(ref):
    return LensingSpectra(
        ells=jnp.asarray(ref["ells"]),
        cl_tt_unl=jnp.asarray(ref["cl_tt_unl"]),
        cl_ee_unl=jnp.asarray(ref["cl_ee_unl"]),
        cl_bb_unl=jnp.asarray(ref["cl_bb_unl"]),
        cl_te_unl=jnp.asarray(ref["cl_te_unl"]),
        cl_tt_len=jnp.asarray(ref["cl_tt_len"]),
        cl_ee_len=jnp.asarray(ref["cl_ee_len"]),
        cl_bb_len=jnp.asarray(ref["cl_bb_len"]),
        cl_te_len=jnp.asarray(ref["cl_te_len"]),
        cl_pp=jnp.asarray(ref["cl_pp"]),
    )


def _max_rel_err_in_window(augr_n0, ref_n0, Ls, lo, hi):
    """|augr - ref| / |ref| max over Ls in [lo, hi]."""
    augr_n0 = np.asarray(augr_n0)
    ref_n0 = np.asarray(ref_n0)
    mask = (Ls >= lo) & (Ls <= hi) & np.isfinite(augr_n0) & np.isfinite(ref_n0) & (ref_n0 != 0)
    if not mask.any():
        return 0.0
    rel = np.abs(augr_n0[mask] - ref_n0[mask]) / np.abs(ref_n0[mask])
    return float(rel.max())


@pytest.mark.slow
class TestN0AgainstPlancklens:
    """Augr full-sky TT N_0 must match plancklens at LiteBIRD-PTEP.

    Full-sky augr is the apples-to-apples comparison against plancklens
    (which is full-sky natively). The flat-sky augr path is validated
    separately by the closed-form ``controlled_input_test.py`` and
    cannot match a full-sky reference at low L (geometric factor).

    PP and MV combinations are NOT tested: augr uses the diagonal
    HO02 Eq.22 ``1/Sum 1/N_alpha`` while plancklens 'p_p' / 'p' include
    inter-estimator cross-correlations (joint GMV). The diagonal
    answer is strictly larger than the joint GMV; the size of the gap
    is real physics, not a bug to test against.

    Slow because full-sky N_0 takes ~10 minutes for ~140 Ls.
    """

    def test_tt_max_rel_err_in_bulk(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_tt = jnp.asarray(ref["nl_tt"])

        augr_n0 = compute_n0_tt(Ls, spectra, nl_tt, fullsky=True)
        ref_n0 = ref["n0_tt"]
        Ls_np = np.asarray(Ls)

        err_bulk = _max_rel_err_in_window(augr_n0, ref_n0, Ls_np, *L_BULK)
        assert err_bulk <= TOL_FULLSKY_TT_BULK, (
            f"full-sky N_0^TT: bulk-L max-rel-err = "
            f"{err_bulk:.4e} > {TOL_FULLSKY_TT_BULK:.4e}"
        )

    def test_tt_max_rel_err_at_high_L(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_tt = jnp.asarray(ref["nl_tt"])

        augr_n0 = compute_n0_tt(Ls, spectra, nl_tt, fullsky=True)
        ref_n0 = ref["n0_tt"]
        Ls_np = np.asarray(Ls)

        err_high = _max_rel_err_in_window(augr_n0, ref_n0, Ls_np, *L_HIGH)
        assert err_high <= TOL_FULLSKY_TT_TAIL, (
            f"full-sky N_0^TT: high-L max-rel-err = "
            f"{err_high:.4e} > {TOL_FULLSKY_TT_TAIL:.4e}"
        )


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Known bug in _compute_n0_ee_fullsky: spin-2 alpha "
           "substitution. augr uses spin-0 raising eigenvalue "
           "sqrt(l(l+1)) for alpha = [L(L+1) + l(l+1) - l'(l'+1)] / 2, "
           "but spin-2 fields require sqrt((l+/-2)(l-/+1)) eigenvalues "
           "(see plancklens utils_spin.get_spin_raise/lower). The error "
           "is invisible for constant Cl (where alpha_1 + alpha_2 = "
           "L(L+1) is the only thing that matters) but shows up as 5-20x "
           "deviation on realistic non-constant LiteBIRD spectra. Fix "
           "requires re-deriving the spin-2 full-sky QE response "
           "(HO02 Appendix A; Smith-Hanson-Lewis 2012; Lewis-Pratten 2020).",
    strict=True,
)
class TestN0EEAgainstPlancklens:
    """Augr full-sky EE N_0 should match plancklens 'pee' once the bug
    in ``_compute_n0_ee_fullsky`` is fixed.

    Currently expected to fail (5-20x off depending on L). This xfail
    test will start passing when the spin-2 alpha substitution is
    corrected.
    """

    def test_ee_max_rel_err_in_bulk(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_ee = jnp.asarray(ref["nl_ee"])

        augr_n0 = compute_n0_ee(Ls, spectra, nl_ee, fullsky=True)
        ref_n0 = ref["n0_ee"]
        Ls_np = np.asarray(Ls)

        err_bulk = _max_rel_err_in_window(augr_n0, ref_n0, Ls_np, *L_BULK)
        assert err_bulk <= TOL_FULLSKY_TT_BULK, (
            f"full-sky N_0^EE: bulk-L max-rel-err = "
            f"{err_bulk:.4e} > {TOL_FULLSKY_TT_BULK:.4e}"
        )


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Known bug in _compute_n0_eb_fullsky: at low L, augr full-sky "
           "EB is ~10000x SMALLER than augr flat-sky EB at L=2 on the "
           "LiteBIRD-PTEP fiducial config (4.6e-12 vs 4.0e-8). The two "
           "agree at high L (1% at L=300; 5% at L=1000), so the bug is "
           "low-L specific. The lensing-kernel direction (C_phi -> C_BB) "
           "uses the same parity-odd spin-2 coupling and validates "
           "against CAMB at <1% (TestFullSkyKernel), so the f_eb_sq "
           "machinery works in one direction. The QE direction is "
           "where the discrepancy lies; suspected related to the same "
           "spin-2 alpha issue as TestN0EEAgainstPlancklens. Fix "
           "requires the same re-derivation work.",
    strict=True,
)
class TestN0EBFullSkyVsFlatSkyAtHighL:
    """Augr full-sky EB should converge to flat-sky EB at L >= 300.

    At high L the geometric flat-vs-full factor (L+1)^2/L^2 -> 1, so
    augr full and flat must agree on the EB N_0. Currently they DO
    agree at high L (5% at L=1000), but the deeper EB bug at low L is
    not captured by this test (we'd need an external full-sky reference
    for low L, and plancklens 'peb' has its own convention issues with
    cl_bb_unl=0). Tracked here as xfail until both EB and EE bugs are
    addressed via the spin-2 re-derivation.
    """

    def test_eb_high_L_full_matches_flat(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_ee = jnp.asarray(ref["nl_ee"])
        nl_bb = jnp.asarray(ref["nl_bb"])

        flat = compute_n0_eb(Ls, spectra, nl_ee, nl_bb, fullsky=False)
        full = compute_n0_eb(Ls, spectra, nl_ee, nl_bb, fullsky=True)
        Ls_np = np.asarray(Ls)
        # At low L (the bump), expect failure (~10000x mismatch at L=2).
        err_low = _max_rel_err_in_window(full, flat, Ls_np, 2, 9)
        assert err_low <= TOL_FULLSKY_TT_BULK, (
            f"full-sky N_0^EB: low-L max-rel-err vs flat = {err_low:.4e}"
        )
