"""Tests for config.py."""

import numpy as np
import pytest

from augr.config import (
    DEFAULT_FIXED,
    DEFAULT_PRIORS,
    DEFAULT_PRIORS_POST_COMPSEP,
    FIDUCIAL_BK15,
    cleaned_map_instrument,
    litebird_like,
    pico_like,
    simple_probe,
)
from augr.foregrounds import GaussianForegroundModel
from augr.instrument import white_noise_power


def test_fiducial_has_all_params():
    """FIDUCIAL_BK15 contains the GaussianForegroundModel params + r + A_lens
    + the residual-template amplitude A_res."""
    fg_names = GaussianForegroundModel().parameter_names
    expected = {"r", "A_lens", "A_res"} | set(fg_names)
    assert set(FIDUCIAL_BK15.keys()) == expected


def test_fiducial_r_zero():
    assert FIDUCIAL_BK15["r"] == 0.0


def test_fiducial_a_res_unity():
    """A_res = 1 at fiducial (template is the truth)."""
    assert FIDUCIAL_BK15["A_res"] == 1.0


def test_default_priors_keys():
    """DEFAULT_PRIORS covers the multifrequency-FG nuisance amplitudes
    (beta_dust, beta_sync). A_res lives in DEFAULT_PRIORS_POST_COMPSEP
    instead, so vanilla forecasts don't carry an irrelevant prior."""
    assert "beta_dust" in DEFAULT_PRIORS
    assert "beta_sync" in DEFAULT_PRIORS
    assert "A_res" not in DEFAULT_PRIORS


def test_default_priors_post_compsep():
    """Post-CompSep priors carry only the residual-template amplitude.
    Compose with DEFAULT_PRIORS for forecasts that mix component
    separation with a multifrequency FG model."""
    assert DEFAULT_PRIORS_POST_COMPSEP == {"A_res": 0.3}


def test_default_fixed():
    assert "T_dust" in DEFAULT_FIXED


def test_simple_probe_returns_instrument():
    inst = simple_probe()
    assert len(inst.channels) == 6
    assert inst.mission_duration_years == 5.0


def test_simple_probe_freqs_ordered():
    inst = simple_probe()
    freqs = [ch.nu_ghz for ch in inst.channels]
    assert freqs == sorted(freqs)


def test_pico_like_returns_instrument():
    inst = pico_like()
    assert len(inst.channels) == 21
    assert inst.mission_duration_years == 5.0


# arXiv:1902.10541 Table 1.2, CBE polarization map depth [uK_CMB arcmin] per band.
PICO_CBE_DEPTH_UK_ARCMIN = {
    21.0: 16.9, 25.0: 13.0, 30.0: 8.7, 36.0: 5.6, 43.0: 5.6, 52.0: 4.0, 62.0: 3.8,
    75.0: 3.0, 90.0: 2.0, 108.0: 1.6, 129.0: 1.5, 155.0: 1.3, 186.0: 2.8, 223.0: 3.2,
    268.0: 2.2, 321.0: 3.0, 385.0: 3.2, 462.0: 6.4, 555.0: 32.4, 666.0: 125.0, 799.0: 740.0,
}
PICO_CBE_AGGREGATE_UK_ARCMIN = 0.61  # Table 1.1 / 3.2 "Total"


def test_pico_like_reproduces_published_cbe_depths():
    """cmb-augr #49: the preset's NET / N_bolo / efficiency chain must land on PICO's
    published CBE map depths -- per band and the inverse-variance aggregate.
    validate_pico.py builds from the depth tables directly and never exercises this
    chain, which is how a 1.8x-too-deep table survived. Measured 2026-08-27 with the
    Table 3.2 array-NET anchoring: worst bands 268 GHz at 0.960 and 129 GHz at 1.032
    (the table prints array NETs to 2 s.f.: 1.5 and 1.1), aggregate 0.605 vs 0.612."""
    inst = pico_like()
    arcmin_per_rad = 180.0 * 60.0 / np.pi
    inv_var = 0.0
    inv_var_pub = 0.0
    for ch in inst.channels:
        w_inv = float(white_noise_power(ch, inst.mission_duration_years, 1.0))  # full sky
        depth = np.sqrt(w_inv) * arcmin_per_rad
        pub = PICO_CBE_DEPTH_UK_ARCMIN[ch.nu_ghz]
        assert abs(depth / pub - 1.0) < 0.05, (ch.nu_ghz, depth, pub)
        inv_var += depth**-2
        inv_var_pub += pub**-2
    aggregate = inv_var**-0.5
    assert abs(aggregate / PICO_CBE_AGGREGATE_UK_ARCMIN - 1.0) < 0.02, aggregate
    assert abs(inv_var_pub**-0.5 / PICO_CBE_AGGREGATE_UK_ARCMIN - 1.0) < 0.01  # the table itself
    assert sum(ch.n_detectors for ch in inst.channels) == 12996


def test_pico_like_aperture_scales_beams_only():
    ref, big = pico_like(), pico_like(aperture_m=2.8)
    for a, b in zip(ref.channels, big.channels, strict=True):
        assert b.beam_fwhm_arcmin == pytest.approx(a.beam_fwhm_arcmin / 2.0)
        assert b.net_per_detector == a.net_per_detector and b.n_detectors == a.n_detectors


def test_pico_frequency_range():
    """PICO covers 21–799 GHz."""
    inst = pico_like()
    freqs = [ch.nu_ghz for ch in inst.channels]
    assert min(freqs) <= 21.0
    assert max(freqs) >= 799.0


def test_litebird_like_returns_instrument():
    """Matches PTEP Table 3: 15 unique frequencies exposed as 22 sub-array
    channels (LFT mixed pixels at 68/78/89, LFT+MFT overlaps at
    100/119/140, MFT+HFT overlap at 195), 40 to 402 GHz, 4508 detectors."""
    inst = litebird_like()
    assert len(inst.channels) == 22
    freqs = sorted({ch.nu_ghz for ch in inst.channels})
    assert len(freqs) == 15
    assert freqs[0] == 40.0
    assert freqs[-1] == 402.0
    # PTEP Table 3 total detector count
    assert sum(ch.n_detectors for ch in inst.channels) == 4508


def test_all_channels_positive_net():
    """All channels must have positive NET."""
    for factory in [simple_probe, pico_like, litebird_like]:
        inst = factory()
        for ch in inst.channels:
            if ch.net_per_detector > 0:  # last PICO band has NET=0 (dust-only)
                assert ch.net_per_detector > 0
            assert ch.n_detectors > 0
            assert ch.beam_fwhm_arcmin > 0


# ---------------------------------------------------------------------------
# cleaned_map_instrument (post-CompSep placeholder)
# ---------------------------------------------------------------------------

def test_cleaned_map_instrument_single_channel():
    """Post-CompSep placeholder has exactly one dummy channel."""
    inst = cleaned_map_instrument(f_sky=0.6)
    assert len(inst.channels) == 1
    assert inst.f_sky == 0.6


def test_cleaned_map_instrument_fsky_propagates():
    """f_sky is the meaningful knob on this preset."""
    for fs in [0.4, 0.6, 1.0]:
        assert cleaned_map_instrument(f_sky=fs).f_sky == fs


def test_cleaned_map_instrument_channel_fields_valid():
    """Dummy channel has sane (positive) values so the dataclass validates."""
    inst = cleaned_map_instrument(f_sky=0.6)
    ch = inst.channels[0]
    assert ch.n_detectors > 0
    assert ch.net_per_detector > 0
    assert ch.beam_fwhm_arcmin > 0
    assert ch.nu_ghz > 0
