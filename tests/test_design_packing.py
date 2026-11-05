"""Tests for the horn-packing design reparam + the design->channels physics.

Fast (pure JAX, no SHT): the focal-plane packing in
:func:`augr.optimize.design_to_channels` and the z-space reparam
:class:`augr.design_packing.PackingDesignSpec`. The end-to-end
:func:`augr.eig.physical_design_objective` through the cut-sky MC forward is exercised by
the slow gates in ``tests/test_eig.py`` and the driver smoke run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.config import galactic_extra_loading
from augr.design_packing import PackingDesignSpec
from augr.optimize import design_to_channels
from augr.telescope import (
    BandSpec,
    FocalPlaneSpec,
    PixelGroup,
    TelescopeDesign,
    ThermalSpec,
    beam_fwhm_arcmin,
    count_pixels_continuous,
    hex_cell_area,
    horn_diameter,
    photon_noise_net_jax,
    to_instrument,
)

# Measured 2026-11-05 on the fixtures below: jax-vs-numpy NET <= 5.6e-16 (with and
# without Galactic loading); grad-vs-central-FD 5.6e-11 at h = 1e-5.
NET_NP_JAX_RTOL = 1e-13
NET_GRAD_FD_RTOL = 1e-9

FREQS_PER_GROUP = (
    (20.0,),
    (35.0,),
    (80.0, 115.0),
    (160.0, 225.0),
    (315.0, 440.0),
    (615.0,),
)
FP_DIAMETER = 0.3
APERTURE = 1.5
F_NUMBER = 1.8


def _spec(**kw):
    base = dict(
        freqs_per_group=FREQS_PER_GROUP,
        frac_fid=np.full(len(FREQS_PER_GROUP), 1.0 / len(FREQS_PER_GROUP)),
        aperture_fid=APERTURE,
        f_number_fid=F_NUMBER,
        years_fid=4.0,
        fp_diameter_m=FP_DIAMETER,
    )
    base.update(kw)
    return PackingDesignSpec(**base)


# ---------------------------------------------------------------------------
# design_to_channels: focal-plane packing physics
# ---------------------------------------------------------------------------


def test_design_to_channels_matches_packing_primitives():
    """n_det / beam / NET reproduce the telescope.py packing primitives exactly."""
    fracs = jnp.asarray(np.full(len(FREQS_PER_GROUP), 1.0 / len(FREQS_PER_GROUP)))
    n_det, net, beam = design_to_channels(
        APERTURE, F_NUMBER, FP_DIAMETER, fracs, FREQS_PER_GROUP
    )

    a_fp = np.pi * (FP_DIAMETER / 2.0) ** 2
    chan = 0
    for g, freqs in enumerate(FREQS_PER_GROUP):
        nu_low = min(freqs)
        a_cell = hex_cell_area(horn_diameter(nu_low, F_NUMBER))
        n_pix = count_pixels_continuous(fracs[g] * a_fp, a_cell, 0.80)
        expect_ndet = 2.0 * float(n_pix)
        for nu in freqs:
            assert float(n_det[chan]) == pytest.approx(expect_ndet, rel=1e-12)
            assert float(beam[chan]) == pytest.approx(
                beam_fwhm_arcmin(nu, APERTURE), rel=1e-12
            )
            assert float(net[chan]) == pytest.approx(
                float(photon_noise_net_jax(nu)), rel=1e-12
            )
            chan += 1
    assert n_det.shape[0] == 9  # 3 singles + 3 dichroic pairs = 9 channels


def test_dichroic_groups_share_horn_and_count():
    """Both bands of a dichroic pair get the same n_det; horn set by the low band."""
    fracs = jnp.asarray(np.full(len(FREQS_PER_GROUP), 1.0 / len(FREQS_PER_GROUP)))
    n_det, _, _ = design_to_channels(
        APERTURE, F_NUMBER, FP_DIAMETER, fracs, FREQS_PER_GROUP
    )
    # flat channel order: 20 | 35 | 80,115 | 160,225 | 315,440 | 615
    assert float(n_det[2]) == float(n_det[3])  # (80, 115)
    assert float(n_det[4]) == float(n_det[5])  # (160, 225)
    assert float(n_det[6]) == float(n_det[7])  # (315, 440)
    # higher-frequency horns are smaller -> more pixels -> more detectors
    assert float(n_det[8]) > float(n_det[6]) > float(n_det[2]) > float(n_det[0])


def test_design_to_channels_differentiable():
    """Gradients flow through the packing: detector count in f# / fractions, beam in aperture."""
    fracs = jnp.asarray(np.full(len(FREQS_PER_GROUP), 1.0 / len(FREQS_PER_GROUP)))

    # n_det ~ 1/f#^2 -> negative, nonzero gradient w.r.t. f#
    g_f = jax.grad(
        lambda f: jnp.sum(
            design_to_channels(APERTURE, f, FP_DIAMETER, fracs, FREQS_PER_GROUP)[0]
        )
    )(F_NUMBER)
    assert np.isfinite(float(g_f)) and float(g_f) < 0.0

    # beam ~ 1/aperture -> total beam decreases with aperture
    g_ap = jax.grad(
        lambda a: jnp.sum(
            design_to_channels(a, F_NUMBER, FP_DIAMETER, fracs, FREQS_PER_GROUP)[2]
        )
    )(APERTURE)
    assert np.isfinite(float(g_ap)) and float(g_ap) < 0.0

    # detector counts respond to the area allocation
    g_frac = jax.grad(
        lambda fr: jnp.sum(
            design_to_channels(APERTURE, F_NUMBER, FP_DIAMETER, fr, FREQS_PER_GROUP)[0]
        )
    )(fracs)
    assert np.all(np.isfinite(np.asarray(g_frac)))
    assert np.any(np.asarray(g_frac) > 0.0)


def test_default_thermal_is_thermalspec_default():
    """``thermal=None`` is bit-identical to an explicit ``ThermalSpec()``."""
    fracs = jnp.asarray(np.full(len(FREQS_PER_GROUP), 1.0 / len(FREQS_PER_GROUP)))
    _, net_none, _ = design_to_channels(APERTURE, F_NUMBER, FP_DIAMETER, fracs, FREQS_PER_GROUP)
    _, net_spec, _ = design_to_channels(APERTURE, F_NUMBER, FP_DIAMETER, fracs, FREQS_PER_GROUP,
                                        thermal=ThermalSpec())
    assert np.array_equal(np.asarray(net_none), np.asarray(net_spec))


@pytest.mark.parametrize("galactic", [False, True])
def test_design_to_channels_net_matches_to_instrument(galactic):
    """Non-default thermal stack + bandwidth + Galactic loading: same NET as to_instrument.

    PICO-like optics with a two-temperature chain (17 K primary, 4.5 K secondary/stop),
    eta 0.40 and 20% bands. ``to_instrument`` is the numpy reference path.
    """
    thermal = ThermalSpec(T_telescope_K=17.0, emissivity=0.015, eta_optical=0.40,
                          extra_stages=((4.5, 0.17),))
    extra = galactic_extra_loading() if galactic else None
    freqs = ((15.0,), (90.0,), (350.0,), (900.0,))
    fracs = jnp.asarray([0.4, 0.3, 0.2, 0.1])
    _, net, beam = design_to_channels(1.4, 1.42, 0.4, fracs, freqs, thermal=thermal,
                                      fractional_bandwidth=0.20, extra_loading=extra)
    design = TelescopeDesign(
        focal_plane=FocalPlaneSpec(aperture_m=1.4, f_number=1.42, fp_diameter_m=0.4),
        thermal=thermal,
        pixel_groups=tuple(
            PixelGroup(bands=(BandSpec(f[0], fractional_bandwidth=0.20, extra_loading=extra),),
                       area_fraction=float(a))
            for f, a in zip(freqs, fracs, strict=True)
        ),
    )
    ref = to_instrument(design).channels  # sorted by frequency == freqs order here
    for i, ch in enumerate(ref):
        assert float(net[i]) == pytest.approx(ch.net_per_detector, rel=NET_NP_JAX_RTOL)
        assert float(beam[i]) == pytest.approx(ch.beam_fwhm_arcmin, rel=1e-12)


def test_net_gradient_in_cold_stage_emissivity():
    """The NET is differentiable in an extra stage's emissivity and matches FD."""
    def f(eps):
        th = ThermalSpec(T_telescope_K=17.0, emissivity=0.015, eta_optical=0.40,
                         extra_stages=((4.5, eps),))
        return design_to_channels(1.4, 1.42, 0.4, jnp.asarray([1.0]), ((150.0,),),
                                  thermal=th)[1][0]

    g = float(jax.grad(f)(0.17))
    h = 1e-5
    fd = (float(f(0.17 + h)) - float(f(0.17 - h))) / (2 * h)
    assert g > 0.0
    assert g == pytest.approx(fd, rel=NET_GRAD_FD_RTOL)


def test_net_override_bypasses_photon_noise():
    fracs = jnp.asarray(np.full(len(FREQS_PER_GROUP), 1.0 / len(FREQS_PER_GROUP)))
    override = jnp.arange(1.0, 10.0)  # 9 channels
    _, net, _ = design_to_channels(
        APERTURE, F_NUMBER, FP_DIAMETER, fracs, FREQS_PER_GROUP, net_override=override
    )
    assert np.allclose(np.asarray(net), np.asarray(override))


# ---------------------------------------------------------------------------
# PackingDesignSpec: the z-space reparam
# ---------------------------------------------------------------------------


def test_z0_recovers_fiducial():
    spec = _spec()
    d = spec.design_pytree(jnp.zeros(spec.n_dim))
    assert float(d["aperture_m"]) == pytest.approx(APERTURE)
    assert float(d["f_number"]) == pytest.approx(F_NUMBER)
    assert float(d["mission_years"]) == pytest.approx(4.0)
    assert np.allclose(np.asarray(d["area_fractions"]), spec.frac_fid)


def test_n_dim_and_labels():
    spec = _spec()
    assert spec.n_dim == (len(FREQS_PER_GROUP) - 1) + 3  # 8
    assert len(spec.knob_labels) == spec.n_dim
    assert spec.knob_labels[-3:] == ("aperture", "f_number", "mission_years")
    # the reference group (index 0 = "20") is gauge-fixed, not a free knob
    assert not any(lbl.startswith("alloc@20") for lbl in spec.knob_labels)


def test_allocation_is_a_simplex():
    spec = _spec()
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.standard_normal(spec.n_dim) * 0.5
        fracs = np.asarray(spec.design_pytree(jnp.asarray(z))["area_fractions"])
        assert np.all(fracs > 0.0)
        assert float(fracs.sum()) == pytest.approx(1.0)


def test_f_number_stays_in_bounds():
    spec = _spec(f_bounds=(1.4, 3.0))
    n_alloc = len(FREQS_PER_GROUP) - 1
    # f# stays in the CLOSED range everywhere (the sigmoid saturates to the bound only in
    # the extreme-z rail); for moderate displacements it is strictly interior.
    for zf in (-50.0, -5.0, 0.0, 5.0, 50.0):
        z = np.zeros(spec.n_dim)
        z[n_alloc + 1] = zf
        f = float(spec.design_pytree(jnp.asarray(z))["f_number"])
        assert 1.4 <= f <= 3.0
        if abs(zf) <= 5.0:
            assert 1.4 < f < 3.0


def test_zspace_gradient_flows():
    """jax.grad of a scalar of the design pytree w.r.t. z is finite and nonzero."""
    spec = _spec()

    def scalar(z):
        d = spec.design_pytree(z)
        # touches every knob family: aperture, f#, years, and the allocation simplex
        return (
            d["aperture_m"]
            + d["f_number"]
            + d["mission_years"]
            + jnp.sum(d["area_fractions"] ** 2)
        )

    g = jax.grad(scalar)(jnp.zeros(spec.n_dim))
    g = np.asarray(g)
    assert np.all(np.isfinite(g))
    assert np.any(np.abs(g) > 0.0)


def test_validation_errors():
    with pytest.raises(ValueError, match="sum to 1"):
        _spec(frac_fid=np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.05]))
    with pytest.raises(ValueError, match="strictly inside"):
        _spec(f_number_fid=3.5)
    with pytest.raises(ValueError, match="one entry per pixel group"):
        _spec(frac_fid=np.array([0.5, 0.5]))
