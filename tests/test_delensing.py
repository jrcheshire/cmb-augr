"""Tests for delensing.py — QE lensing reconstruction and residual BB."""

from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.config import simple_probe
from augr.delensing import (
    LensingSpectra,
    _gl_nodes,
    _interp_at,
    _qe_domain,
    _triangle_geometry,
    compute_n0_eb,
    compute_n0_ee,
    compute_n0_mv,
    compute_n0_tb,
    compute_n0_te,
    compute_n0_tt,
    delens_residual_bb,
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
    @pytest.mark.slow
    def test_convergence(self, spectra, noise):
        """Iteration should converge (A_lens_eff stable by iteration 3).

        Slow: flat-sky iterate_delensing at l_max_qe=1000, n_iter=5 is ~25s locally
        and >90s on the 2-core CI runner (its fullsky sibling is already marked slow).
        The lighter test_result_shape below stays in the gate as the iterate smoke.
        """
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


# Light delensing config (mirrors test_result_shape) so the differentiable
# tests below stay in the fast gate.
_DIFF_KW = dict(ls=jnp.arange(2, 201, dtype=float),
                L_max=500, l_max_qe=500, n_phi=32, n_iter=2)


class TestDifferentiableDelensing:
    """delens_residual_bb (flat-sky) is jit- and grad-clean (issue #45)."""

    def test_matches_iterate(self, spectra, noise):
        """delens_residual_bb equals iterate_delensing.cl_bb_res bit-for-bit."""
        cl = delens_residual_bb(
            spectra, noise["tt"], noise["ee"], noise["bb"], **_DIFF_KW)
        res = iterate_delensing(
            spectra, noise["tt"], noise["ee"], noise["bb"], **_DIFF_KW)
        np.testing.assert_array_equal(np.asarray(cl),
                                      np.asarray(res.cl_bb_res))

    def test_jit_matches_eager(self, spectra, noise):
        """jit (spectra + knobs closed over) reproduces the eager result."""
        eager = delens_residual_bb(
            spectra, noise["tt"], noise["ee"], noise["bb"], **_DIFF_KW)
        jit_fn = jax.jit(
            lambda a, b, c: delens_residual_bb(spectra, a, b, c, **_DIFF_KW))
        jitted = jit_fn(noise["tt"], noise["ee"], noise["bb"])
        # The whole flat-sky trace is lax.scan + jnp, so jit reorders only
        # fusion, not math; 1e-12 is far below any physical tolerance.
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager),
                                   rtol=1e-12, atol=0.0)

    def test_grad_finite_and_fd_consistent(self, spectra, noise):
        """jax.grad wrt each noise spectrum is finite and matches central FD.

        FD is taken along the scale-consistent direction v = nl_bb
        (multiplicative perturbation), which stays in the smooth
        positive-noise interior instead of pushing low-ell noise negative.
        """
        def scalar_of(a, b, c):
            return jnp.sum(
                delens_residual_bb(spectra, a, b, c, **_DIFF_KW))

        nl_tt, nl_ee, nl_bb = noise["tt"], noise["ee"], noise["bb"]
        g_tt, g_ee, g_bb = jax.grad(scalar_of, argnums=(0, 1, 2))(
            nl_tt, nl_ee, nl_bb)
        for g in (g_tt, g_ee, g_bb):
            assert jnp.all(jnp.isfinite(g))

        h = 1e-4
        f_plus = scalar_of(nl_tt, nl_ee, nl_bb * (1 + h))
        f_minus = scalar_of(nl_tt, nl_ee, nl_bb * (1 - h))
        fd = float((f_plus - f_minus) / (2 * h))
        ad = float(jnp.dot(g_bb, nl_bb))
        np.testing.assert_allclose(fd, ad, rtol=1e-4)


# -----------------------------------------------------------------------
# QE domain cut (l_min <= l2 <= l_max)
# -----------------------------------------------------------------------

class TestQEDomainCut:
    """The QE reconstruction cut is geometric, not a property of the noise.

    ``l2 = L - l1`` runs over ``[|L - l1|, L + l1]``, so it leaves
    ``[l_min, l_max]``. Below ``l_min`` the CAMB templates are exactly
    zero while the *total* retains the noise floor, so the historical
    ``denom > 0`` test let those cells through with a denominator ~1e10
    too small. Measured on TT that made flat-sky N_0 400-2000x too
    small versus the plancklens-validated full-sky path.
    """

    _KW: ClassVar = {"ls": jnp.arange(2, 101, dtype=float), "L_max": 200,
                     "l_min_qe": 2, "l_max_qe": 200, "n_iter": 1}

    def test_qe_domain_excludes_out_of_range_l2(self):
        """Cells with l2 outside [l_min, l_max] are masked out even when
        the denominator is comfortably positive."""
        l2 = jnp.array([0.5, 1.0, 2.0, 100.0, 200.0, 200.5, 300.0])
        denom = jnp.ones_like(l2)          # positive everywhere
        ok = np.asarray(_qe_domain(l2, denom, 2, 200))
        assert list(ok) == [False, False, True, True, True, False, False]

    def test_positive_denominator_alone_is_not_enough(self):
        """Regression on the specific failure: at l=0,1 the templates are
        zero but the total is the (positive) noise floor, so `denom > 0`
        passes and the geometric cut is what rejects the cell."""
        l2 = jnp.array([1.0])
        denom = jnp.array([1.7e-7])        # noise-floor-sized, but > 0
        assert bool(denom[0] > 0)
        assert not bool(_qe_domain(l2, denom, 2, 200)[0])

    def test_residual_converges_under_phi_refinement(self, spectra, noise):
        """C_res must settle as n_phi grows; it used to wobble at ~1e-3.

        Measured in THIS config (simple_probe, l_max_qe=200, n_iter=1):
        max relative change over 128 -> 256 is 3.0e-6. The pre-fix code
        gives ~1e-3 and non-monotone, so the 1e-4 gate separates them by
        ~30x on both sides.
        """
        a = np.asarray(delens_residual_bb(spectra, noise["tt"], noise["bb"],
                                          noise["bb"], n_phi=128, **self._KW))
        b = np.asarray(delens_residual_bb(spectra, noise["tt"], noise["bb"],
                                          noise["bb"], n_phi=256, **self._KW))
        assert np.max(np.abs(b / a - 1.0)) < 1e-4

    @pytest.mark.parametrize("fwhm_arcmin", [30.0, 12.0])
    def test_grad_matches_fd_along_ell_reweighting_direction(
            self, spectra, noise, fwhm_arcmin):
        """AD vs FD along a direction that reweights ell, on two designs.

        The other gradient gates in this file and in test_optimize.py all
        step along a *uniform rescale* of nl_bb (or of n_det, which
        `_combined_white_nl_bb` turns into one). That direction averages
        the per-multipole quadrature error away and cannot move the
        domain boundary, so it passed even against the pre-fix gradient.
        A beam change is the honest design direction:
        d(nl)/d(fwhm) ~ nl * l(l+1) * fwhm / (4 ln 2).

        Measured here: rel = 1.4e-9 (30') and 3.8e-9 (12'); the pre-fix
        code gives 3e-4 at the same step and its AD is wrong in
        magnitude by ~31x, not merely noisy.
        """
        arcmin = np.pi / 180.0 / 60.0
        ells = np.asarray(spectra.ells)
        f = fwhm_arcmin * arcmin
        nl = noise["bb"]
        v = nl * jnp.array(ells * (ells + 1) * (f * arcmin) / (4 * np.log(2)))

        def scalar(n):
            return jnp.sum(delens_residual_bb(spectra, noise["tt"], n, n,
                                              n_phi=128, **self._KW))

        ad = float(jnp.dot(jax.grad(scalar)(nl), v))
        h = 1e-4
        fd = float((scalar(nl + h * v) - scalar(nl - h * v)) / (2 * h))
        assert abs(fd / ad - 1.0) < 1e-6

    @pytest.mark.slow
    def test_flat_sky_agrees_with_fullsky_after_geometric_factor(
            self, spectra, noise):
        """Flat-sky N_0^TT must track the plancklens-validated full-sky
        path up to the known (L+1)^2/L^2 flat-vs-full geometric factor
        (quantified in scripts/n0_validation/controlled_input_test.py).

        Measured: 0.993 / 1.003 / 1.005 / 1.005 at L = 10/30/80/150.
        Pre-fix the same ratio was 5e-4 to 7e-3 -- i.e. flat-sky N_0 was
        400-2000x too small. Gate at 5%.
        """
        Ls = jnp.array([10.0, 30.0, 80.0, 150.0])
        flat = np.asarray(compute_n0_tt(Ls, spectra, noise["tt"], 2, 300,
                                        n_phi=512))
        full = np.asarray(compute_n0_tt(Ls, spectra, noise["tt"], 2, 300,
                                        fullsky=True))
        Lv = np.asarray(Ls)
        ratio = (flat / full) / ((Lv + 1) ** 2 / Lv ** 2)
        assert np.all(np.abs(ratio - 1.0) < 0.05)


# -----------------------------------------------------------------------
# TE filter (HO02 Eq. 13)
# -----------------------------------------------------------------------

class TestTEFilter:
    """The TE filter is HO02 Eq. 13, not a diagonal approximation.

    Unlike the other four estimators TE has C^{xx'} != 0, so neither
    Eq. 14 (x = x') nor Eq. 15 (C~^{xx'} = 0) applies. The historical
    'ho02_diag_approx' denominator C_TT(l1)C_EE(l2) + C_TE(l1)C_TE(l2)
    is not sign-definite (the arguments differ -- it is not a square),
    which puts a negative contribution into an inverse variance.
    """

    _LS: ClassVar = jnp.array([2.0, 50.0, 200.0, 350.0])
    _KW: ClassVar = {"l_min": 2, "l_max": 500}

    def test_cauchy_schwarz_holds_on_total_spectra(self, spectra, noise):
        """Eq. 13's denominator is non-negative because C_TE^2 <= C_TT*C_EE.

        Checked on the actual *total* (lensed + noise) spectra the filter
        uses, not just in principle -- the guarantee is what makes the
        exact filter's integrand sign-definite.
        """
        tt = np.asarray(spectra.cl_tt_len + noise["tt"])
        ee = np.asarray(spectra.cl_ee_len + noise["ee"])
        te = np.asarray(spectra.cl_te_len)
        assert np.all(te**2 <= tt * ee)

    def test_exact_filter_is_finite_where_diag_approx_diverges(
            self, spectra, noise):
        """The historical filter returns inf / flips sign; Eq. 13 does not.

        n_phi=256 at L=200 is the measured failure point:  goes
        negative there under 'ho02_diag_approx' and trips the total>0
        guard. Eq. 13 is finite at every n_phi.
        """
        for n_phi in (128, 256, 512):
            exact = compute_n0_te(self._LS, spectra, noise["tt"], noise["ee"],
                                  n_phi=n_phi, te_filter="ho02_exact",
                                  **self._KW)
            assert np.all(np.isfinite(np.asarray(exact)))
            assert np.all(np.asarray(exact) > 0)

    def test_exact_filter_converges_under_phi_refinement(
            self, spectra, noise):
        """Refining n_phi must shrink the change; the old filter wobbles.

        Tolerance measured in this configuration (simple_probe noise,
        l_max=500): the 256->512 step moves N_0^TE by <1e-3 at every L
        tested, against O(1) swings (and an inf) for 'ho02_diag_approx'.
        """
        a = np.asarray(compute_n0_te(self._LS, spectra, noise["tt"],
                                     noise["ee"], n_phi=256,
                                     te_filter="ho02_exact", **self._KW))
        b = np.asarray(compute_n0_te(self._LS, spectra, noise["tt"],
                                     noise["ee"], n_phi=512,
                                     te_filter="ho02_exact", **self._KW))
        assert np.all(np.abs(b / a - 1.0) < 1e-3)

    def test_te_filter_is_honoured_on_the_flat_sky_path(
            self, spectra, noise):
        """te_filter used to be silently ignored unless fullsky=True."""
        vals = [
            np.asarray(compute_n0_te(self._LS, spectra, noise["tt"],
                                     noise["ee"], n_phi=128, te_filter=f,
                                     **self._KW))
            for f in ("ho02_exact", "ho02_diag_approx", "strict_diagonal")
        ]
        # Compare RELATIVE differences: every value here is below
        # np.allclose's default atol=1e-8, so allclose would call them
        # equal no matter how far apart they are.
        assert np.max(np.abs(vals[1] / vals[0] - 1.0)) > 1e-3
        assert np.max(np.abs(vals[2] / vals[0] - 1.0)) > 1e-3

    def test_unknown_te_filter_raises(self, spectra, noise):
        with pytest.raises(ValueError, match="te_filter must be one of"):
            compute_n0_te(self._LS, spectra, noise["tt"], noise["ee"],
                          te_filter="not_a_filter")


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
    @pytest.mark.slow
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

# TE has a structural ~5% bulk-L residual not shared with the other 4
# estimators; see TestN0TEAgainstPlancklens for diagnosis. Bulk-L band
# stops at L=1800 to avoid the C_TE zero-crossings near l~1850 where
# the response amplitude vanishes and any structural residual blows
# up relative to plancklens.
TOL_FULLSKY_TE_BULK = 6e-2
L_BULK_TE = (10, 1800)

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
class TestN0EEAgainstPlancklens:
    """Augr full-sky EE N_0 must match plancklens 'pee' to <1e-3 in bulk-L.

    Locks in the resolution of the previously documented "5-20x"
    EE / EB residual: it was a sign error in the m_3 term of
    augr.wigner._sg_b (Schulten-Gordon recursion coefficient), not
    a missing spin-lowering branch. The earlier `derivation.md`
    diagnosis ("two-branch fix needed") is superseded; once
    ``_sg_b`` honors SG 1975 Eq. 5 sign convention, the existing
    Hu-Okamoto Eq. 14 single-bracket implementation matches
    plancklens to <1e-7 in the realistic bulk.
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
class TestN0EBAgainstPlancklens:
    """Augr full-sky EB N_0 must match plancklens 'p_eb' (symmetrized).

    augr's ``compute_n0_eb`` weights both the EE and BB legs
    symmetrically (Smith+ 2012 Eq. 6-7 with parity-odd coupling),
    so the apples-to-apples plancklens reference is the symmetrized
    'p_eb' = (peb + pbe)/2 variant exposed by ``run_plancklens.py``.
    """

    def test_eb_max_rel_err_in_bulk(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_ee = jnp.asarray(ref["nl_ee"])
        nl_bb = jnp.asarray(ref["nl_bb"])

        augr_n0 = compute_n0_eb(Ls, spectra, nl_ee, nl_bb, fullsky=True)
        ref_n0 = ref["n0_eb"]
        Ls_np = np.asarray(Ls)

        err_bulk = _max_rel_err_in_window(augr_n0, ref_n0, Ls_np, *L_BULK)
        assert err_bulk <= TOL_FULLSKY_TT_BULK, (
            f"full-sky N_0^EB: bulk-L max-rel-err = "
            f"{err_bulk:.4e} > {TOL_FULLSKY_TT_BULK:.4e}"
        )


@pytest.mark.slow
class TestN0TEAgainstPlancklens:
    """Augr full-sky TE N_0 vs plancklens 'p_te' (symmetrized).

    Locked at ``TOL_FULLSKY_TE_BULK`` (6e-2) in ``L_BULK_TE`` (10..1800),
    a deliberately looser gate than the <1e-3 bulk-L lock-in for TT / EE
    / EB / TB. The looseness is structural, not a tolerance kludge:

    * **Production filter mismatch**: augr's ``compute_n0_te`` defaults
      to HO02 Eq. 13's diagonal-approximation filter
      ``1/(C_TT*C_EE + C_TE^2)``. Plancklens forces ``fal['te']=0``,
      giving the strict-diagonal filter ``1/(C_TT*C_EE)``. This test
      calls with ``te_filter='strict_diagonal'`` to align the filters
      exactly; that part is apples-to-apples.
    * **Symmetrization residual** (the structural ~5%): plancklens
      ``p_te`` is the symmetric estimator ``g_pte + g_pet``, whose
      variance is ``Var(pte) + Var(pet) + 2 Cov(pte, pet)``. Augr's
      ``_compute_n0_te_fullsky`` implements OkaHu 2003 Table I's
      single-projection response (E-leg spin-2, T-leg spin-0), which
      reproduces ``Var(pte)`` only -- it does NOT capture the
      ``Cov(pte, pet)`` cross-Wick contraction. With ``fal['te']=0``
      the cross term is non-zero because ``cls_ivfs[te] = cl_te /
      (C_TT_total * C_EE_total)`` is non-zero, and contributes a few
      percent at all L. Closing it requires porting plancklens's
      ``nhl._get_nhl`` cross-Wick logic to harmonic space (the
      already-validated ``augr/_qe.py`` is the leg-construction
      reference) -- deferred; out of scope for this test.
    * **C_TE zero-crossings at L~1850**: the response amplitude
      vanishes there, so any residual structural percent-level error
      blows up to 10-20% in relative terms. The bulk-L band stops at
      L=1800 to keep the test informative about the structural floor
      rather than dominated by these localized blow-ups.

    Per ``compute_n0_te``'s own docstring, TE contributes ~1-2% to
    ``N_0^MV`` at space-experiment noise levels, so the 5% TE residual
    propagates as <0.1% on N_0^MV and <1% on A_L for realistic delensing
    efficiencies -- well below decision-relevance for sigma(r) forecasts.
    Full-sky is production-grade for space-mission applications (where
    the reionization bump dominates the sigma(r) constraint and the
    (L+1)^2 / L^2 flat-vs-full geometric correction matters at low L);
    flat-sky remains the ``iterate_delensing`` default for runtime
    (~5x faster) but is no longer the math/physics preference.
    """

    def test_te_max_rel_err_in_bulk(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_tt = jnp.asarray(ref["nl_tt"])
        nl_ee = jnp.asarray(ref["nl_ee"])

        augr_n0 = compute_n0_te(
            Ls, spectra, nl_tt, nl_ee,
            fullsky=True, te_filter='strict_diagonal',
        )
        ref_n0 = ref["n0_te"]
        Ls_np = np.asarray(Ls)

        err_bulk = _max_rel_err_in_window(augr_n0, ref_n0, Ls_np, *L_BULK_TE)
        assert err_bulk <= TOL_FULLSKY_TE_BULK, (
            f"full-sky N_0^TE: bulk-L max-rel-err = "
            f"{err_bulk:.4e} > {TOL_FULLSKY_TE_BULK:.4e}"
        )


@pytest.mark.slow
class TestN0TBAgainstPlancklens:
    """Augr full-sky TB N_0 must match plancklens 'p_tb' (symmetrized)."""

    def test_tb_max_rel_err_in_bulk(self):
        ref = _load_reference()
        spectra = _make_spectra(ref)
        Ls = jnp.asarray(ref["Ls"], dtype=float)
        nl_tt = jnp.asarray(ref["nl_tt"])
        nl_bb = jnp.asarray(ref["nl_bb"])

        augr_n0 = compute_n0_tb(Ls, spectra, nl_tt, nl_bb, fullsky=True)
        ref_n0 = ref["n0_tb"]
        Ls_np = np.asarray(Ls)

        err_bulk = _max_rel_err_in_window(augr_n0, ref_n0, Ls_np, *L_BULK)
        assert err_bulk <= TOL_FULLSKY_TT_BULK, (
            f"full-sky N_0^TB: bulk-L max-rel-err = "
            f"{err_bulk:.4e} > {TOL_FULLSKY_TT_BULK:.4e}"
        )


# -----------------------------------------------------------------------
# Differentiable full-sky backend (issue #45 Stage 3)
# -----------------------------------------------------------------------

class TestFullSkyJaxBackend:
    """Pure-jnp full-sky drivers reproduce the numpy full-sky path and are
    jax.grad-traceable. Slow: the numpy reference runs the Wigner drivers."""

    @pytest.mark.slow
    def test_drivers_match_numpy(self, spectra, noise):
        from augr.delensing import (
            _compute_n0_eb_fullsky,
            _compute_n0_ee_fullsky,
            _compute_n0_tb_fullsky,
            _compute_n0_te_fullsky,
            _compute_n0_tt_fullsky,
            _lensing_kernel_fullsky,
        )
        from augr.delensing_fullsky_jax import (
            compute_n0_eb_fullsky_jax,
            compute_n0_ee_fullsky_jax,
            compute_n0_tb_fullsky_jax,
            compute_n0_te_fullsky_jax,
            compute_n0_tt_fullsky_jax,
            lensing_kernel_fullsky_jax,
        )
        nt, ne, nb = noise["tt"], noise["ee"], noise["bb"]
        Ls = jnp.arange(2, 151, dtype=float)
        lmin, lmax = 2, 250

        def rel(a, b):
            a, b = np.asarray(a), np.asarray(b)
            fin = np.isfinite(a) & np.isfinite(b)
            return np.max(np.abs(a[fin] - b[fin]) / (np.abs(b[fin]) + 1e-300))

        assert rel(compute_n0_eb_fullsky_jax(Ls, spectra, ne, nb, lmin, lmax),
                   _compute_n0_eb_fullsky(Ls, spectra, ne, nb, lmin, lmax)) < 1e-6
        assert rel(compute_n0_tb_fullsky_jax(Ls, spectra, nt, nb, lmin, lmax),
                   _compute_n0_tb_fullsky(Ls, spectra, nt, nb, lmin, lmax)) < 1e-6
        assert rel(compute_n0_tt_fullsky_jax(Ls, spectra, nt, lmin, lmax),
                   _compute_n0_tt_fullsky(Ls, spectra, nt, lmin, lmax)) < 1e-6
        assert rel(compute_n0_ee_fullsky_jax(Ls, spectra, ne, lmin, lmax),
                   _compute_n0_ee_fullsky(Ls, spectra, ne, lmin, lmax)) < 1e-6
        assert rel(compute_n0_te_fullsky_jax(Ls, spectra, nt, ne, lmin, lmax),
                   _compute_n0_te_fullsky(Ls, spectra, nt, ne, lmin, lmax)) < 1e-5
        ls = jnp.arange(2, 101, dtype=float)
        assert rel(lensing_kernel_fullsky_jax(ls, Ls, spectra, lmin, lmax),
                   _lensing_kernel_fullsky(ls, Ls, spectra, lmin, lmax)) < 1e-6

    @pytest.mark.slow
    def test_iterate_fullsky_jax_matches_numpy(self, spectra, noise):
        kw = dict(ls=jnp.arange(2, 151, dtype=float), L_max=200,
                  l_max_qe=200, n_iter=2, fullsky=True)
        rn = iterate_delensing(spectra, noise["tt"], noise["ee"], noise["bb"],
                               backend="numpy", **kw)
        rj = iterate_delensing(spectra, noise["tt"], noise["ee"], noise["bb"],
                               backend="jax", **kw)
        np.testing.assert_allclose(np.asarray(rj.cl_bb_res),
                                   np.asarray(rn.cl_bb_res), rtol=1e-6, atol=1e-30)
        np.testing.assert_allclose(rj.A_lens_eff, rn.A_lens_eff, rtol=1e-6)

    @pytest.mark.slow
    def test_fullsky_jax_grad_finite(self, spectra, noise):
        from augr.delensing import _delens_core
        nt, ne = noise["tt"], noise["ee"]
        ls = jnp.arange(2, 151, dtype=float)
        Ls = jnp.arange(2, 201, dtype=float)

        def f(nl_bb):
            cl, _n0, _a, _h = _delens_core(
                spectra, nt, ne, nl_bb, ls, Ls, n_iter=2, l_min_qe=2,
                l_max_qe=200, n_phi=128, fullsky=True, backend="jax")
            return jnp.sum(cl)

        g = jax.grad(f)(noise["bb"])
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.linalg.norm(g)) > 0
