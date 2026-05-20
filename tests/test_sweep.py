"""Tests for augr.sweep -- jax.vmap wrappers over the differentiable forward."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.config import FIDUCIAL_BK15, simple_probe
from augr.foregrounds import GaussianForegroundModel
from augr.optimize import (
    make_optimization_context,
    sigma_r_from_channels,
    sigma_r_from_design,
)
from augr.spectra import CMBSpectra
from augr.sweep import (
    sigma_r_over_beam,
    sigma_r_over_eta,
    sigma_r_over_n_det,
    sigma_r_over_net,
    vmap_channels,
    vmap_design,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def instrument():
    return simple_probe()


@pytest.fixture(scope="module")
def ctx(instrument):
    return make_optimization_context(
        instrument,
        GaussianForegroundModel(),
        CMBSpectra(),
        dict(FIDUCIAL_BK15),
        priors={"beta_dust": 0.11, "beta_sync": 0.3},
        fixed_params=["T_dust", "Delta_dust"],
        ell_min=2, ell_max=300, delta_ell=30, ell_per_bin_below=10,
    )


@pytest.fixture(scope="module")
def design_args(ctx):
    """Common args for sigma_r_from_design calls."""
    return dict(
        f_number=jnp.asarray(2.0),
        fp_diameter_m=jnp.asarray(0.4),
        area_fractions=jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        freqs_per_group=((30., 40.), (85., 150.), (220., 340.)),
        net_override=ctx.net,
        eta_total=ctx.eta,
    )


# -----------------------------------------------------------------------
# Channel-level sweeps (sigma_r_from_channels)
# -----------------------------------------------------------------------

class TestChannelSweeps:
    def test_n_det_vmap_matches_serial_loop(self, ctx):
        """sigma_r_over_n_det == [sigma_r_from_channels(scale * n_det, ...) for scale]."""
        scales = jnp.array([0.5, 1.0, 1.5, 2.0])
        n_det_grid = scales[:, None] * ctx.n_det  # shape (4, n_chan)

        vmap_result = sigma_r_over_n_det(
            n_det_grid, ctx.net, ctx.beam, ctx.eta, ctx,
        )
        serial = jnp.array([
            sigma_r_from_channels(n_det_grid[i], ctx.net, ctx.beam, ctx.eta, ctx)
            for i in range(scales.shape[0])
        ])
        np.testing.assert_allclose(
            np.array(vmap_result), np.array(serial), rtol=1e-12, atol=0,
        )

    def test_net_vmap_matches_serial_loop(self, ctx):
        scales = jnp.array([0.5, 1.0, 2.0])
        net_grid = scales[:, None] * ctx.net

        vmap_result = sigma_r_over_net(
            ctx.n_det, net_grid, ctx.beam, ctx.eta, ctx,
        )
        serial = jnp.array([
            sigma_r_from_channels(ctx.n_det, net_grid[i], ctx.beam, ctx.eta, ctx)
            for i in range(scales.shape[0])
        ])
        np.testing.assert_allclose(
            np.array(vmap_result), np.array(serial), rtol=1e-12, atol=0,
        )

    def test_beam_vmap_matches_serial_loop(self, ctx):
        scales = jnp.array([0.8, 1.0, 1.25])
        beam_grid = scales[:, None] * ctx.beam

        vmap_result = sigma_r_over_beam(
            ctx.n_det, ctx.net, beam_grid, ctx.eta, ctx,
        )
        serial = jnp.array([
            sigma_r_from_channels(ctx.n_det, ctx.net, beam_grid[i], ctx.eta, ctx)
            for i in range(scales.shape[0])
        ])
        np.testing.assert_allclose(
            np.array(vmap_result), np.array(serial), rtol=1e-12, atol=0,
        )

    def test_eta_vmap_matches_serial_loop(self, ctx):
        scales = jnp.array([0.9, 1.0, 1.1])
        eta_grid = scales[:, None] * ctx.eta

        vmap_result = sigma_r_over_eta(
            ctx.n_det, ctx.net, ctx.beam, eta_grid, ctx,
        )
        serial = jnp.array([
            sigma_r_from_channels(ctx.n_det, ctx.net, ctx.beam, eta_grid[i], ctx)
            for i in range(scales.shape[0])
        ])
        np.testing.assert_allclose(
            np.array(vmap_result), np.array(serial), rtol=1e-12, atol=0,
        )


# -----------------------------------------------------------------------
# Design-level sweeps (sigma_r_from_design)
# -----------------------------------------------------------------------

class TestDesignSweeps:
    def test_aperture_vmap_matches_serial_loop(self, ctx, design_args):
        apertures = jnp.array([1.0, 1.5, 2.0, 3.0, 5.0])

        # Bind the array kwargs (net_override, eta_total) before vmap so they
        # aren't themselves vmapped along their leading channel axis.
        sweep = vmap_design(
            "aperture_m",
            net_override=design_args["net_override"],
            eta_total=design_args["eta_total"],
        )
        vmap_result = sweep(
            apertures,
            design_args["f_number"],
            design_args["fp_diameter_m"],
            design_args["area_fractions"],
            ctx,
            design_args["freqs_per_group"],
        )
        serial = jnp.array([
            sigma_r_from_design(
                aperture_m=float(apertures[i]),
                f_number=design_args["f_number"],
                fp_diameter_m=design_args["fp_diameter_m"],
                area_fractions=design_args["area_fractions"],
                ctx=ctx,
                freqs_per_group=design_args["freqs_per_group"],
                net_override=design_args["net_override"],
                eta_total=design_args["eta_total"],
            )
            for i in range(apertures.shape[0])
        ])
        np.testing.assert_allclose(
            np.array(vmap_result), np.array(serial), rtol=1e-10, atol=0,
        )

    def test_f_number_vmap_matches_serial_loop(self, ctx, design_args):
        f_grid = jnp.array([1.5, 2.0, 2.5])

        sweep = vmap_design(
            "f_number",
            net_override=design_args["net_override"],
            eta_total=design_args["eta_total"],
        )
        vmap_result = sweep(
            jnp.asarray(1.5),  # aperture scalar
            f_grid,
            design_args["fp_diameter_m"],
            design_args["area_fractions"],
            ctx,
            design_args["freqs_per_group"],
        )
        serial = jnp.array([
            sigma_r_from_design(
                aperture_m=jnp.asarray(1.5),
                f_number=float(f_grid[i]),
                fp_diameter_m=design_args["fp_diameter_m"],
                area_fractions=design_args["area_fractions"],
                ctx=ctx,
                freqs_per_group=design_args["freqs_per_group"],
                net_override=design_args["net_override"],
                eta_total=design_args["eta_total"],
            )
            for i in range(f_grid.shape[0])
        ])
        np.testing.assert_allclose(
            np.array(vmap_result), np.array(serial), rtol=1e-10, atol=0,
        )


# -----------------------------------------------------------------------
# grad composition
# -----------------------------------------------------------------------

class TestGradComposition:
    def test_jax_grad_through_aperture_sweep(self, ctx, design_args):
        """jax.grad of a reduction of vmap'd sigma_r_over_aperture is finite."""
        apertures = jnp.array([1.0, 2.0, 3.0])
        sweep = vmap_design(
            "aperture_m",
            net_override=design_args["net_override"],
            eta_total=design_args["eta_total"],
        )

        def loss(aps):
            sigmas = sweep(
                aps,
                design_args["f_number"],
                design_args["fp_diameter_m"],
                design_args["area_fractions"],
                ctx,
                design_args["freqs_per_group"],
            )
            return jnp.sum(sigmas)

        g = jax.grad(loss)(apertures)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0.0)


# -----------------------------------------------------------------------
# Factory error paths
# -----------------------------------------------------------------------

class TestFactoryErrors:
    def test_vmap_channels_rejects_unknown_axis(self):
        with pytest.raises(ValueError, match="axis must be one of"):
            vmap_channels("foo")

    def test_vmap_design_rejects_unknown_axis(self):
        with pytest.raises(ValueError, match="axis must be one of"):
            vmap_design("aperture")  # close, but the right name is aperture_m
