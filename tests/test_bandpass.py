"""Tests for bandpass.py — the per-band Bandpass representation."""

import jax
import jax.numpy as jnp

from augr.bandpass import N_QUAD_SMOOTH, N_QUAD_TOPHAT, Bandpass


def test_tophat_grid_edges_and_weights():
    """tophat spans [nu_c(1-f/2), nu_c(1+f/2)] with uniform weights."""
    bp = Bandpass.tophat(150.0, 0.25)
    assert bp.nu_ghz.shape == (N_QUAD_TOPHAT,)
    assert abs(float(bp.nu_ghz[0]) - 150.0 * (1.0 - 0.125)) < 1e-9
    assert abs(float(bp.nu_ghz[-1]) - 150.0 * (1.0 + 0.125)) < 1e-9
    assert jnp.allclose(bp.weights, 1.0)
    assert not bp.is_monochromatic


def test_tophat_custom_nquad():
    bp = Bandpass.tophat(95.0, 0.3, n_quad=33)
    assert bp.nu_ghz.shape == (33,)


def test_tophat_zero_width_is_monochromatic():
    """fractional_bandwidth <= 0 collapses to the delta-function limit."""
    bp = Bandpass.tophat(150.0, 0.0)
    assert bp.is_monochromatic
    assert bp.nu_ghz.shape == (1,)
    assert abs(float(bp.nu_ghz[0]) - 150.0) < 1e-12


def test_monochromatic_factory():
    bp = Bandpass.monochromatic(220.0)
    assert bp.is_monochromatic
    assert abs(float(bp.nu_ghz[0]) - 220.0) < 1e-12
    assert abs(float(bp.weights[0]) - 1.0) < 1e-12


def test_smooth_tophat_shape_and_range():
    """smooth_tophat weights lie in [0,1] and peak near unity inside the band."""
    bp = Bandpass.smooth_tophat(150.0, 0.25)
    assert bp.nu_ghz.shape == (N_QUAD_SMOOTH,)
    assert float(bp.weights.min()) >= -1e-12
    assert float(bp.weights.max()) <= 1.0 + 1e-12
    # Interior weight ~1, far-out-of-band weight ~0 (grid xi in [0.5, 1.5]).
    assert float(bp.weights.max()) > 0.99
    assert float(bp.weights[0]) < 1e-3  # xi = 0.5 -> nu = 75 GHz, well outside
    assert float(bp.weights[-1]) < 1e-3  # xi = 1.5 -> nu = 225 GHz, well outside


def test_smooth_tophat_symmetric_mean():
    """Response-weighted mean frequency equals the band center for a symmetric band."""
    bp = Bandpass.smooth_tophat(150.0, 0.25)
    mean_nu = jnp.sum(bp.nu_ghz * bp.weights) / jnp.sum(bp.weights)
    assert abs(float(mean_nu) - 150.0) < 1e-3


def test_smooth_tophat_grad_wrt_nu_center():
    """d(sum nu_grid)/d(nu_center) = sum(xi) since nu = nu_center * xi (fixed xi)."""

    def total_nu(nu_c):
        return jnp.sum(Bandpass.smooth_tophat(nu_c, 0.25).nu_ghz)

    g = jax.grad(total_nu)(150.0)
    xi = jnp.linspace(1.0 - 0.5, 1.0 + 0.5, N_QUAD_SMOOTH)
    assert jnp.isfinite(g)
    assert abs(float(g) - float(jnp.sum(xi))) < 1e-6


def test_smooth_tophat_grad_wrt_frac_bw_finite():
    """Gradient flows through the smooth weights w.r.t. fractional bandwidth."""

    def total_w(f):
        return jnp.sum(Bandpass.smooth_tophat(150.0, f).weights)

    gf = jax.grad(total_w)(0.25)
    assert jnp.isfinite(gf)
    assert float(gf) > 0.0  # widening the band adds weight


def test_from_profile_weighted_mean_center():
    nu = jnp.array([95.0, 100.0, 105.0])
    w = jnp.array([1.0, 2.0, 1.0])
    bp = Bandpass.from_profile(nu, w)
    assert abs(float(bp.nu_center_ghz) - 100.0) < 1e-9
    assert not bp.is_monochromatic
