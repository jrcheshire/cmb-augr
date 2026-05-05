"""Tests for augr.telescope — physical telescope model."""

import math

import numpy as np
import pytest

from augr.telescope import (
    BandSpec,
    FocalPlaneSpec,
    PixelGroup,
    TelescopeDesign,
    ThermalSpec,
    beam_fwhm_arcmin,
    count_pixels,
    flagship_design,
    hex_cell_area,
    horn_diameter,
    photon_noise_net,
    probe_design,
    to_instrument,
)
from augr.units import H_PLANCK, K_BOLTZMANN

# ---------------------------------------------------------------------------
# Geometry functions
# ---------------------------------------------------------------------------

class TestBeamFwhm:
    """Diffraction-limited beam size."""

    def test_150ghz_1p5m(self):
        """1.5 m aperture at 150 GHz: FWHM ≈ 5.6 arcmin (Airy)."""
        fwhm = beam_fwhm_arcmin(150.0, 1.5)
        # λ = 2.0 mm, FWHM = 1.22 × 2mm / 1.5m = 1.627e-3 rad = 5.59'
        assert 5.4 < fwhm < 5.8

    def test_150ghz_3m(self):
        """3 m aperture at 150 GHz: half the beam of 1.5 m."""
        fwhm_small = beam_fwhm_arcmin(150.0, 1.5)
        fwhm_large = beam_fwhm_arcmin(150.0, 3.0)
        assert abs(fwhm_large - fwhm_small / 2.0) < 0.01

    def test_frequency_scaling(self):
        """Beam scales as 1/ν at fixed aperture."""
        fwhm_90 = beam_fwhm_arcmin(90.0, 1.5)
        fwhm_150 = beam_fwhm_arcmin(150.0, 1.5)
        assert abs(fwhm_90 / fwhm_150 - 150.0 / 90.0) < 0.01

    def test_illumination_factor(self):
        """Custom illumination factor scales linearly."""
        fwhm_airy = beam_fwhm_arcmin(150.0, 1.5, illumination_factor=1.22)
        fwhm_tapered = beam_fwhm_arcmin(150.0, 1.5, illumination_factor=1.40)
        assert abs(fwhm_tapered / fwhm_airy - 1.40 / 1.22) < 0.01

    def test_pico_comparison(self):
        """Compare with PICO: 6.2' at 155 GHz with 1.4 m aperture.

        PICO's actual illumination is slightly larger than Airy 1.22;
        our Airy prediction should be somewhat smaller.
        """
        fwhm = beam_fwhm_arcmin(155.0, 1.4)
        # PICO reports 6.2'; diffraction limit is ~5.5', so ours is smaller
        assert 5.0 < fwhm < 6.2


class TestHornDiameter:
    """Feedhorn aperture sizing."""

    def test_150ghz_f2(self):
        """f/2 at 150 GHz: d = 2 × 2 × 2mm = 8 mm."""
        d = horn_diameter(150.0, 2.0)
        assert abs(d - 0.008) < 0.001

    def test_30ghz_f2(self):
        """f/2 at 30 GHz: d = 2 × 2 × 10mm = 40 mm."""
        d = horn_diameter(30.0, 2.0)
        assert abs(d - 0.040) < 0.001

    def test_frequency_scaling(self):
        """Horn diameter scales as 1/ν."""
        d_30 = horn_diameter(30.0, 2.0)
        d_150 = horn_diameter(150.0, 2.0)
        assert abs(d_30 / d_150 - 5.0) < 0.01

    def test_fnumber_scaling(self):
        """Horn diameter scales linearly with f-number."""
        d_f2 = horn_diameter(150.0, 2.0)
        d_f3 = horn_diameter(150.0, 3.0)
        assert abs(d_f3 / d_f2 - 1.5) < 0.01


class TestHexCellArea:
    """Hexagonal close-packing cell area."""

    def test_unit_diameter(self):
        """d = 1 m: cell area = √3/2 ≈ 0.8660 m²."""
        area = hex_cell_area(1.0)
        assert abs(area - math.sqrt(3) / 2.0) < 1e-10

    def test_scaling(self):
        """Area scales as d²."""
        a1 = hex_cell_area(0.01)
        a2 = hex_cell_area(0.02)
        assert abs(a2 / a1 - 4.0) < 1e-10


class TestCountPixels:
    """Focal plane pixel counting."""

    def test_basic(self):
        """Simple count: 100 cm² / 1 cm² × 0.8 = 80 pixels."""
        n = count_pixels(fp_area=100e-4, cell_area=1e-4,
                         packing_efficiency=0.80)
        assert n == 80

    def test_floor_rounding(self):
        """floor() ensures conservative count."""
        n = count_pixels(fp_area=10.5e-4, cell_area=1e-4,
                         packing_efficiency=1.0)
        assert n == 10  # floor(10.5)

    def test_zero_for_large_horn(self):
        """Returns 0 if horn is larger than allocated area."""
        n = count_pixels(fp_area=1e-4, cell_area=1e-2,
                         packing_efficiency=0.8)
        assert n == 0

    def test_area_scaling(self):
        """4× area = 4× pixels."""
        n1 = count_pixels(fp_area=1e-2, cell_area=1e-4,
                          packing_efficiency=0.8)
        n4 = count_pixels(fp_area=4e-2, cell_area=1e-4,
                          packing_efficiency=0.8)
        assert n4 == 4 * n1


# ---------------------------------------------------------------------------
# Photon noise NET
# ---------------------------------------------------------------------------

class TestPhotonNoiseNet:
    """Photon-noise-limited NET calculation."""

    def test_150ghz_reasonable_range(self):
        """At 150 GHz with PICO-like parameters, NET ~ 30–80 μK√s."""
        net = photon_noise_net(150.0, T_telescope=4.0, emissivity=0.01,
                               eta_optical=0.35)
        assert 20.0 < net < 100.0, f"NET = {net:.1f} μK√s out of range"

    def test_pico_155ghz_comparison(self):
        """Compare with PICO CBE NET of 27.5 μK√s at 155 GHz.

        Our simplified model won't match exactly (different optical chain
        details), but should be within a factor of ~2.
        """
        net = photon_noise_net(155.0, T_telescope=4.0, emissivity=0.01,
                               eta_optical=0.35)
        pico_net = 27.5
        ratio = net / pico_net
        assert 0.5 < ratio < 2.5, f"NET = {net:.1f}, ratio to PICO = {ratio:.2f}"

    def test_low_freq_higher_net(self):
        """NET at 30 GHz should be higher than at 150 GHz (RJ tail)."""
        net_30 = photon_noise_net(30.0)
        net_150 = photon_noise_net(150.0)
        assert net_30 > net_150

    def test_high_freq_higher_net(self):
        """NET at 500 GHz should be higher than at 150 GHz (tel emission)."""
        net_500 = photon_noise_net(500.0)
        net_150 = photon_noise_net(150.0)
        assert net_500 > net_150

    def test_zero_emissivity_floor(self):
        """With no telescope emission, NET is the CMB photon noise floor."""
        net_notel = photon_noise_net(150.0, emissivity=0.0)
        net_tel = photon_noise_net(150.0, emissivity=0.05)
        # Adding telescope emission should increase NET
        assert net_tel > net_notel

    def test_emissivity_monotonic(self):
        """Higher emissivity → higher NET (more loading)."""
        nets = [photon_noise_net(150.0, emissivity=e)
                for e in [0.0, 0.01, 0.05, 0.10]]
        for i in range(len(nets) - 1):
            assert nets[i + 1] > nets[i]

    def test_eta_optical_effect(self):
        """Higher optical efficiency → lower NET (more signal per noise)."""
        net_low = photon_noise_net(150.0, eta_optical=0.20)
        net_high = photon_noise_net(150.0, eta_optical=0.50)
        assert net_high < net_low

    def test_bandwidth_effect(self):
        """Wider bandwidth → lower NET (more photons collected)."""
        net_narrow = photon_noise_net(150.0, fractional_bandwidth=0.10)
        net_wide = photon_noise_net(150.0, fractional_bandwidth=0.30)
        assert net_wide < net_narrow

    def test_positive(self):
        """NET must be positive for any reasonable parameters."""
        for nu in [30.0, 90.0, 150.0, 220.0, 340.0, 500.0, 800.0]:
            net = photon_noise_net(nu)
            assert net > 0, f"NET at {nu} GHz = {net}"

    def test_extra_loading_none_is_baseline(self):
        """``extra_loading=None`` reproduces the prior no-extra-loading result.

        Regression-proof against accidentally short-circuiting the default
        path through the new code branch.
        """
        net_default = photon_noise_net(150.0)
        net_none = photon_noise_net(150.0, extra_loading=None)
        assert net_default == net_none

    def test_extra_loading_zero_callable_is_baseline(self):
        """A callable that returns zero occupation matches the default."""
        net_default = photon_noise_net(150.0)
        net_zero = photon_noise_net(
            150.0,
            extra_loading=lambda nu: np.zeros_like(nu),
        )
        # Both go through the same arithmetic; result must be identical.
        assert abs(net_zero - net_default) < 1e-12

    def test_extra_loading_monotonic(self):
        """Increasing the extra-loading occupation must increase NET."""
        nets = [
            photon_noise_net(
                150.0,
                extra_loading=lambda nu, A=A: A * np.ones_like(nu),
            )
            for A in [0.0, 0.01, 0.05, 0.10]
        ]
        for i in range(len(nets) - 1):
            assert nets[i + 1] > nets[i], (
                f"NET non-monotonic in extra_loading: {nets}"
            )

    def test_extra_loading_per_band_independence(self):
        """A loading function applied at one band is independent of others.

        Important for the per-band atmospheric-loading use case: feeding
        a different ``extra_loading`` callable per BandSpec must not
        couple bands through any shared state.
        """
        # Two different per-band loading functions
        def atm_at_90(nu):
            # Effective T ≈ 25 K graybody, evaluated as occupation number
            return 1.0 / (np.exp(H_PLANCK * nu / (K_BOLTZMANN * 25.0)) - 1.0)

        def atm_at_150(nu):
            # Hotter band → larger occupation
            return 1.0 / (np.exp(H_PLANCK * nu / (K_BOLTZMANN * 40.0)) - 1.0)

        net_90_with = photon_noise_net(90.0, extra_loading=atm_at_90)
        net_90_without = photon_noise_net(90.0)
        net_150_with = photon_noise_net(150.0, extra_loading=atm_at_150)
        net_150_without = photon_noise_net(150.0)

        # Each band increases its NET vs the no-loading baseline.
        assert net_90_with > net_90_without
        assert net_150_with > net_150_without
        # Reusing the same band call without extra_loading still gives the
        # baseline (no leakage between calls via mutable state).
        assert photon_noise_net(90.0) == net_90_without


# ---------------------------------------------------------------------------
# Data structure validation
# ---------------------------------------------------------------------------

class TestDataStructures:
    """Dataclass construction and validation."""

    def test_bandspec_defaults(self):
        b = BandSpec(150.0)
        assert b.nu_ghz == 150.0
        assert b.fractional_bandwidth == 0.25
        assert b.extra_loading is None

    def test_bandspec_with_extra_loading(self):
        """BandSpec accepts a callable extra_loading; to_instrument
        threads it through to photon_noise_net so the resulting Channel
        has a higher NET than a sibling band without loading."""
        # Two-band probe with an extra-loading function on the 90 GHz band only
        def atm(nu):
            return 0.05 * np.ones_like(nu)
        b_loaded = BandSpec(90.0, extra_loading=atm)
        b_clean = BandSpec(90.0)
        assert b_loaded.extra_loading is atm
        assert b_clean.extra_loading is None

        # Build matched single-band probes -- only difference is the loading
        design_loaded = TelescopeDesign(
            focal_plane=FocalPlaneSpec(aperture_m=1.5, f_number=2.0,
                                       fp_diameter_m=0.4),
            thermal=ThermalSpec(),
            pixel_groups=(PixelGroup(bands=(b_loaded,), area_fraction=1.0),),
        )
        design_clean = TelescopeDesign(
            focal_plane=FocalPlaneSpec(aperture_m=1.5, f_number=2.0,
                                       fp_diameter_m=0.4),
            thermal=ThermalSpec(),
            pixel_groups=(PixelGroup(bands=(b_clean,), area_fraction=1.0),),
        )
        net_loaded = to_instrument(design_loaded).channels[0].net_per_detector
        net_clean = to_instrument(design_clean).channels[0].net_per_detector
        assert net_loaded > net_clean

    def test_pixelgroup_single_band(self):
        pg = PixelGroup(bands=(BandSpec(150.0),), area_fraction=0.5)
        assert len(pg.bands) == 1

    def test_pixelgroup_dichroic(self):
        pg = PixelGroup(
            bands=(BandSpec(90.0), BandSpec(150.0)),
            area_fraction=0.5,
        )
        assert len(pg.bands) == 2

    def test_pixelgroup_wrong_order_raises(self):
        with pytest.raises(ValueError, match=r"bands\[0\].nu_ghz < bands\[1\]"):
            PixelGroup(
                bands=(BandSpec(150.0), BandSpec(90.0)),
                area_fraction=0.5,
            )

    def test_pixelgroup_three_bands_raises(self):
        with pytest.raises(ValueError, match="1 or 2 bands"):
            PixelGroup(
                bands=(BandSpec(90.0), BandSpec(150.0), BandSpec(220.0)),
                area_fraction=0.5,
            )

    def test_frozen(self):
        """All dataclasses are immutable."""
        b = BandSpec(150.0)
        with pytest.raises(AttributeError):
            b.nu_ghz = 220.0

        fp = FocalPlaneSpec(1.5, 2.0, 0.4)
        with pytest.raises(AttributeError):
            fp.aperture_m = 3.0


# ---------------------------------------------------------------------------
# to_instrument() integration tests
# ---------------------------------------------------------------------------

class TestToInstrument:
    """End-to-end instrument builder."""

    def test_probe_design_runs(self):
        """probe_design() -> to_instrument() produces a valid Instrument."""
        inst = to_instrument(probe_design())
        assert len(inst.channels) == 6
        assert inst.mission_duration_years == 5.0
        assert inst.f_sky == 0.7

    def test_flagship_design_runs(self):
        """flagship_design() -> to_instrument() produces a valid Instrument."""
        inst = to_instrument(flagship_design())
        assert len(inst.channels) == 8
        assert inst.mission_duration_years == 5.0

    def test_channels_sorted_by_frequency(self):
        inst = to_instrument(probe_design())
        freqs = [ch.nu_ghz for ch in inst.channels]
        assert freqs == sorted(freqs)

    def test_all_channels_positive(self):
        """All channels have positive NET, n_detectors, and beam."""
        inst = to_instrument(probe_design())
        for ch in inst.channels:
            assert ch.n_detectors > 0, f"{ch.nu_ghz} GHz: n_det = {ch.n_detectors}"
            assert ch.net_per_detector > 0, f"{ch.nu_ghz} GHz: NET = {ch.net_per_detector}"
            assert ch.beam_fwhm_arcmin > 0, f"{ch.nu_ghz} GHz: FWHM = {ch.beam_fwhm_arcmin}"

    def test_dichroic_pair_same_n_detectors(self):
        """Both bands in a dichroic pair get the same detector count."""
        design = TelescopeDesign(
            focal_plane=FocalPlaneSpec(1.5, 2.0, 0.4),
            thermal=ThermalSpec(),
            pixel_groups=(
                PixelGroup(
                    bands=(BandSpec(90.0), BandSpec(150.0)),
                    area_fraction=1.0,
                ),
            ),
        )
        inst = to_instrument(design)
        assert len(inst.channels) == 2
        assert inst.channels[0].n_detectors == inst.channels[1].n_detectors

    def test_low_freq_fewer_detectors(self):
        """Low-frequency horns are bigger, so fewer fit in the same area."""
        design = TelescopeDesign(
            focal_plane=FocalPlaneSpec(1.5, 2.0, 0.4),
            thermal=ThermalSpec(),
            pixel_groups=(
                PixelGroup(bands=(BandSpec(30.0),), area_fraction=0.5),
                PixelGroup(bands=(BandSpec(150.0),), area_fraction=0.5),
            ),
        )
        inst = to_instrument(design)
        ch_30 = next(ch for ch in inst.channels if ch.nu_ghz == 30.0)
        ch_150 = next(ch for ch in inst.channels if ch.nu_ghz == 150.0)
        assert ch_30.n_detectors < ch_150.n_detectors
        # Should scale roughly as (150/30)² = 25
        ratio = ch_150.n_detectors / ch_30.n_detectors
        assert 20.0 < ratio < 30.0

    def test_area_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            to_instrument(TelescopeDesign(
                focal_plane=FocalPlaneSpec(1.5, 2.0, 0.4),
                thermal=ThermalSpec(),
                pixel_groups=(
                    PixelGroup(bands=(BandSpec(150.0),), area_fraction=0.3),
                    PixelGroup(bands=(BandSpec(220.0),), area_fraction=0.3),
                ),
            ))

    def test_focal_plane_conservation(self):
        """Total packed area should not exceed focal plane area."""
        design = probe_design()
        fp = design.focal_plane
        a_fp = math.pi * (fp.fp_diameter_m / 2.0) ** 2

        total_packed = 0.0
        for pg in design.pixel_groups:
            nu_low = min(b.nu_ghz for b in pg.bands)
            d_horn = horn_diameter(nu_low, fp.f_number)
            a_cell = hex_cell_area(d_horn)
            n_pix = count_pixels(
                pg.area_fraction * a_fp, a_cell, fp.packing_efficiency
            )
            total_packed += n_pix * a_cell

        assert total_packed <= a_fp

    def test_flagship_more_detectors_than_probe(self):
        """Flagship (3m, 0.6m FP) should have more detectors at 150 GHz."""
        probe_inst = to_instrument(probe_design())
        flagship_inst = to_instrument(flagship_design())

        probe_150 = next(ch for ch in probe_inst.channels if ch.nu_ghz == 150.0)
        flagship_150 = next(ch for ch in flagship_inst.channels if ch.nu_ghz == 150.0)
        assert flagship_150.n_detectors > probe_150.n_detectors

    def test_flagship_smaller_beams_than_probe(self):
        """Flagship (3m) has smaller beams than probe (1.5m)."""
        probe_inst = to_instrument(probe_design())
        flagship_inst = to_instrument(flagship_design())

        probe_150 = next(ch for ch in probe_inst.channels if ch.nu_ghz == 150.0)
        flagship_150 = next(ch for ch in flagship_inst.channels if ch.nu_ghz == 150.0)
        assert flagship_150.beam_fwhm_arcmin < probe_150.beam_fwhm_arcmin
        # Should be exactly half for same illumination factor
        ratio = probe_150.beam_fwhm_arcmin / flagship_150.beam_fwhm_arcmin
        assert abs(ratio - 2.0) < 0.01
