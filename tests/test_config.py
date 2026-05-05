"""Tests for config.py."""

import pytest
from augr.config import (
    FIDUCIAL_BK15,
    DEFAULT_PRIORS,
    DEFAULT_PRIORS_POST_COMPSEP,
    DEFAULT_FIXED,
    simple_probe,
    pico_like,
    litebird_like,
    cleaned_map_instrument,
)
from augr.foregrounds import GaussianForegroundModel


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
