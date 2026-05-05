"""
config.py — Fiducial parameters, priors, and instrument presets.

Fiducial foreground parameters match the BK15 maximum-likelihood values
(Buza 2019 thesis, Sec. 6.3.2; BICEP2/Keck Array 2018, arXiv:1810.05216).

Instrument presets are approximate representations of published experiments
for comparison and scaling studies. Channel-level numbers are drawn from
published sensitivity tables; efficiencies are physically motivated defaults.

References:
    PICO: Hanany et al. 2019 (arXiv:1902.10541), Table 4-1
    LiteBIRD: LiteBIRD Collaboration 2023 (arXiv:2202.02773), Table 1
"""

from __future__ import annotations

from augr.instrument import Channel, Instrument, ScalarEfficiency


# ---------------------------------------------------------------------------
# Foreground fiducial parameters
# ---------------------------------------------------------------------------

# flatten_params only reads keys that appear in SignalModel.parameter_names,
# so A_res here is a harmless convenience for the post-CompSep workflow --
# it is consumed only when a residual template is attached, and silently
# ignored otherwise. Delensed mode similarly ignores A_lens.
FIDUCIAL_BK15: dict[str, float] = {
    "r":           0.0,    # tensor-to-scalar ratio (target: r = 0)
    "A_lens":      1.0,    # lensing amplitude (1 = no delensing)
    "A_dust":      4.7,    # dust power at 353 GHz, ℓ=80 [μK²]
    "beta_dust":   1.6,    # dust spectral index
    "alpha_dust": -0.58,   # dust ℓ-dependence power law
    "T_dust":     19.6,    # dust temperature [K] (usually fixed)
    "A_sync":      1.5,    # sync power at 23 GHz, ℓ=80 [μK²]
    "beta_sync":  -3.1,    # synchrotron spectral index (RJ)
    "alpha_sync": -0.6,    # sync ℓ-dependence power law
    "epsilon":     0.0,    # dust–sync correlation ([-1, 1])
    "Delta_dust":  0.0,    # dust frequency decorrelation strength
    "A_res":       1.0,    # residual-template amplitude (post-CompSep mode only)
}

# ---------------------------------------------------------------------------
# Default priors
# ---------------------------------------------------------------------------

DEFAULT_PRIORS: dict[str, float] = {
    "beta_dust":  0.11,   # Planck 2015 polarization (arXiv:1502.01588)
    "beta_sync":  0.3,    # WMAP/Planck synchrotron index uncertainty
    "A_res":      0.3,    # residual-template amplitude, placeholder
                          # (Carones 2025 uses a flat prior; 0.3 is a
                          # conservative Gaussian default for Fisher use.
                          # Remove from the priors dict to reproduce Carones.)
}

# Parameters commonly held fixed (not varied in Fisher matrix)
DEFAULT_FIXED: list[str] = ["T_dust"]


# ---------------------------------------------------------------------------
# Moment expansion model fiducials (extends BK15 with new parameters)
# ---------------------------------------------------------------------------

FIDUCIAL_MOMENT: dict[str, float] = {
    **FIDUCIAL_BK15,
    "c_sync":         0.0,    # synchrotron spectral curvature (ARCADE: ~-0.052)
    "Delta_sync":     0.0,    # synchrotron frequency decorrelation
    "omega_d_beta":   0.0,    # dust β_d moment ∝ Var(β_d)
    "omega_d_T":      0.0,    # dust T_d moment ∝ Var(T_d)
    "omega_d_betaT":  0.0,    # dust β_d–T_d cross moment
    "omega_s_beta":   0.0,    # sync β_s moment ∝ Var(β_s)
    "omega_s_c":      0.0,    # sync c_s moment ∝ Var(c_s)
    "omega_s_betac":  0.0,    # sync β_s–c_s cross moment
}

DEFAULT_PRIORS_MOMENT: dict[str, float] = {
    "beta_dust":  0.11,
    "beta_sync":  0.3,
    "A_res":      0.3,   # see note in DEFAULT_PRIORS
}

DEFAULT_FIXED_MOMENT: list[str] = ["T_dust"]


# ---------------------------------------------------------------------------
# Space-mission efficiency defaults
# ---------------------------------------------------------------------------
#
# TODO(cleanup, L2_EFFICIENCY redundancy):
# This constant is also defined identically in ``telescope.py`` (with a
# stale comment claiming the duplication avoids a circular dependency
# that doesn't actually exist today), and ``instrument.ScalarEfficiency``
# carries dataclass defaults that disagree with both:
#
#   instrument.ScalarEfficiency()      → eta_total = 0.654  (0.85, 0.85, 0.85, 0.95, 0.95)
#   _L2_EFFICIENCY (here + telescope)  → eta_total = 0.711  (0.85, 0.85, 0.90, 0.97, 0.95)
#
# Anyone constructing ``Channel(...)`` without an explicit ``efficiency=``
# silently gets the more pessimistic dataclass-default values, ~8% off.
# Right fix: a single canonical ``L2_EFFICIENCY`` (no underscore --
# promote to public preset) in ``instrument.py`` next to ``ScalarEfficiency``,
# imported from there in both ``config.py`` and ``telescope.py``; drop
# dataclass defaults to force explicit construction; update tests
# accordingly. Small mechanical refactor (~30 lines across three
# files), out of scope for this cleanup-pass commit but flagged here.
# ---------------------------------------------------------------------------

# L2 space mission: high efficiency, modest cosmic-ray deadtime
_L2_EFFICIENCY = ScalarEfficiency(
    detector_yield=0.85,
    observing_efficiency=0.85,
    data_cut_fraction=0.90,
    cosmic_ray_deadtime=0.97,   # ~3% deadtime from GCR glitches at L2
    polarization_efficiency=0.95,
)


# ---------------------------------------------------------------------------
# Instrument presets
# ---------------------------------------------------------------------------
#
# TODO(cleanup, preset overlap with telescope.py):
# Two families of "probe-class space mission" presets exist in parallel
# and don't talk to each other:
#
#   - This module's ``simple_probe()``: hand-tuned 6-band Instrument
#     with explicit (NET, beam, n_det) per channel. Loosely PICO-like
#     at reduced channel count.
#   - ``telescope.probe_design()``: physics-based TelescopeDesign
#     (1.5 m, f/2, 0.4 m focal plane, 6 bands in 3 dichroic pairs)
#     that goes through ``to_instrument()`` to derive NETs from
#     photon_noise_net + detector counts from feedhorn packing. Has
#     ``probe_idealized()`` / ``flagship_design()`` / ``flagship_idealized()``
#     siblings for optics-comparison forecasts.
#
# They cover different band layouts and use different methodologies,
# so they aren't strictly redundant -- but a user seeing both is
# justifiably confused about which to use. The right resolution is
# probably:
#   (a) Keep the published-numbers presets here (``pico_like``,
#       ``litebird_like``, ``so_like``, ``cmbs4_like``,
#       ``cleaned_map_instrument``) -- these reproduce specific public
#       forecasts and don't have a physics-derived counterpart.
#   (b) Move ``simple_probe`` either out (let users get a probe via
#       ``to_instrument(probe_design())``) or down-rank it to a clearly-
#       labeled "smoke-test fixture" (small, hand-tuned for fast tests,
#       not meant as a representative forecast).
#   (c) Document the divide in this module's header: "physics-derived
#       designs live in telescope.py; published-numbers presets live
#       here".
#
# Also see the related _L2_EFFICIENCY redundancy TODO -- both stem from
# the same accreted-over-time presets-of-presets layout.
# ---------------------------------------------------------------------------

def simple_probe() -> Instrument:
    """Minimal 6-band space probe for quick tests and debugging.

    Covers the foreground minimum (150 GHz) plus two dust and two sync
    channels and one cross-check band. Sensitivity levels are loosely
    PICO-like at reduced channel count.
    """
    eff = _L2_EFFICIENCY
    channels = (
        Channel(nu_ghz=30.0,  n_detectors=12,  net_per_detector=114.0, beam_fwhm_arcmin=38.4, efficiency=eff),
        Channel(nu_ghz=90.0,  n_detectors=48,  net_per_detector=52.0,  beam_fwhm_arcmin=12.8, efficiency=eff),
        Channel(nu_ghz=150.0, n_detectors=96,  net_per_detector=43.0,  beam_fwhm_arcmin=7.7,  efficiency=eff),
        Channel(nu_ghz=220.0, n_detectors=96,  net_per_detector=65.0,  beam_fwhm_arcmin=5.3,  efficiency=eff),
        Channel(nu_ghz=340.0, n_detectors=48,  net_per_detector=199.0, beam_fwhm_arcmin=3.4,  efficiency=eff),
        Channel(nu_ghz=500.0, n_detectors=12,  net_per_detector=930.0, beam_fwhm_arcmin=2.3,  efficiency=eff),
    )
    return Instrument(channels=channels, mission_duration_years=5.0, f_sky=0.7)


def pico_like() -> Instrument:
    """PICO-like probe-class instrument (arXiv:1902.10541, Table 3.2).

    21 frequency bands from 21 to 799 GHz, 12,996 TES bolometers at 0.1 K,
    5-year L2 mission. Targets σ(r) ≈ 5×10⁻⁴ at 5σ.

    NET values are CBE per-bolometer temperature NETs from Table 3.2.
    Detector counts include both polarizations per pixel (factor of 2
    already in N_bolo). Beam FWHM = 6.2' × (155 GHz / ν_c).
    PICO assumed ~95% survey efficiency and 90% detector yield from L2.
    """
    eff = _L2_EFFICIENCY
    # (nu_ghz, n_bolo, CBE bolo NET [μK_CMB √s], FWHM [arcmin])
    # Source: PICO report Table 3.2 (arXiv:1902.10541, p.36)
    # NET is per-bolometer temperature NET; code applies √2 for polarization.
    # N_bolo = (tiles) × (pixels/tile) × 2 polarizations.
    _bands = [
        ( 21.0,  120,  175.0, 38.4),
        ( 25.0,  200,  108.0, 32.0),
        ( 30.0,  120,   75.8, 28.3),
        ( 36.0,  200,   52.6, 23.6),
        ( 43.0,  120,   41.5, 22.2),
        ( 52.0,  200,   30.5, 18.4),
        ( 62.0,  732,   22.8, 12.8),
        ( 75.0, 1020,   17.3, 10.7),
        ( 90.0,  732,   15.9,  9.5),
        (108.0, 1020,   14.0,  7.9),
        (129.0,  732,   15.4,  7.4),
        (155.0, 1020,   27.5,  6.2),
        (186.0,  960,   27.0,  4.3),
        (223.0,  900,   37.0,  3.6),
        (268.0,  960,   62.0,  3.2),
        (321.0,  900,  144.0,  2.6),
        (385.0,  960,  384.0,  2.5),
        (462.0,  900, 1240.0,  2.1),
        (555.0,  440, 4650.0,  1.5),
        (666.0,  400, 19400.0, 1.3),
        (799.0,  360, 50000.0, 1.1),  # highest band — placeholder NET
    ]
    channels = tuple(
        Channel(nu_ghz=nu, n_detectors=nd, net_per_detector=net,
                beam_fwhm_arcmin=fwhm, efficiency=eff)
        for nu, nd, net, fwhm in _bands
    )
    return Instrument(channels=channels, mission_duration_years=5.0, f_sky=0.7)


def litebird_like() -> Instrument:
    """LiteBIRD PTEP baseline (Hazumi+ 2023, arXiv:2202.02773, Table 3).

    15 frequency bands from 40 to 402 GHz across three telescopes
    (LFT 40-140 GHz, MFT 100-195 GHz, HFT 195-402 GHz); 4508 detectors
    total; 3-year L2 mission; f_sky = 0.7.

    PTEP Table 3 lists each frequency as one or two sub-arrays (LFT+MFT
    or MFT+HFT at shared frequencies; LFT mixed pixel sizes at 68/78/89).
    We expose each sub-array as its own augr Channel (22 total) so that
    augr's Fisher machinery handles the MV combination per ℓ via the
    cross-frequency covariance, preserving each sub-array's distinct
    FWHM rather than collapsing to a single effective beam.  Per-sub-
    array NET_det and FWHM come verbatim from PTEP Table 3; detector
    counts sum to 4508 across all 22 entries.

    Efficiency factors track PTEP Eq. 32: detector_yield=0.80,
    observing_efficiency=0.85 (duty cycle), data_cut_fraction=0.95
    (margin), cosmic_ray_deadtime=0.95, polarization_efficiency=0.90
    (augr models finite HWP efficiency; PTEP Eq. 32 assumes 1.0 and
    folds HWP losses into NET_det).
    """
    eff = ScalarEfficiency(
        detector_yield=0.80,
        observing_efficiency=0.85,
        data_cut_fraction=0.95,
        cosmic_ray_deadtime=0.95,
        polarization_efficiency=0.90,
    )
    # (nu_ghz, n_det, NET_det [μK√s], FWHM [arcmin])
    # Values verbatim from Hazumi+ 2023 Table 3; bands 4-9 and 11 are
    # split into their contributing sub-arrays.
    _bands = [
        # --- LFT only, single-pixel bands ---
        ( 40.0,   48, 114.63, 70.5),   # LFT
        ( 50.0,   24,  72.48, 58.5),   # LFT
        ( 60.0,   48,  65.28, 51.1),   # LFT
        # --- LFT mixed-pixel bands (16mm + 32mm sub-arrays) ---
        ( 68.0,  144, 105.64, 41.6),   # LFT 16mm
        ( 68.0,   24,  68.81, 47.1),   # LFT 32mm
        ( 78.0,  144,  82.51, 36.9),   # LFT 16mm
        ( 78.0,   48,  58.61, 43.8),   # LFT 32mm
        ( 89.0,  144,  65.18, 33.0),   # LFT 16mm
        ( 89.0,   24,  62.33, 41.5),   # LFT 32mm
        # --- LFT + MFT overlaps ---
        (100.0,  144,  54.88, 30.2),   # LFT
        (100.0,  366,  71.70, 37.8),   # MFT
        (119.0,  144,  40.78, 26.3),   # LFT
        (119.0,  488,  55.65, 33.6),   # MFT
        (140.0,  144,  38.44, 23.7),   # LFT
        (140.0,  366,  54.00, 30.8),   # MFT
        # --- MFT only ---
        (166.0,  488,  54.37, 28.9),
        # --- MFT + HFT overlap ---
        (195.0,  366,  59.61, 28.0),   # MFT
        (195.0,  254,  73.96, 28.6),   # HFT
        # --- HFT only ---
        (235.0,  254,  76.06, 24.7),
        (280.0,  254,  97.26, 22.5),
        (337.0,  254, 154.64, 20.9),
        (402.0,  338, 385.69, 17.9),
    ]
    channels = tuple(
        Channel(nu_ghz=nu, n_detectors=nd, net_per_detector=net,
                beam_fwhm_arcmin=fwhm, efficiency=eff)
        for nu, nd, net, fwhm in _bands
    )
    return Instrument(channels=channels, mission_duration_years=3.0, f_sky=0.7)


def cleaned_map_instrument(f_sky: float,
                           mission_years: float = 3.0,
                           nu_ghz: float = 150.0) -> Instrument:
    """Single-channel placeholder Instrument for post-CompSep forecasts.

    Represents the cleaned CMB map produced by an external component-
    separation pipeline (e.g. NILC). The Channel's frequency, NET, detector
    count, and beam are all dummy values: in post-CompSep mode the actual
    noise is expected to be passed via FisherForecast(external_noise_bb=...),
    and the CMB / residual-template contributions to the Fisher are
    evaluated on the SignalModel's internal ell grid without needing per-
    channel SEDs (NullForegroundModel takes care of that).

    The only field that meaningfully enters the Fisher is f_sky, which sets
    the Knox mode count nu_b = (2 ell + 1) * delta_ell * f_sky. For a GAL60
    validation run pass f_sky=0.6; for full-sky pass 1.0.

    Args:
        f_sky:         Effective sky fraction (Knox mode weighting).
        mission_years: Nominal mission duration; dummy but must be > 0.
        nu_ghz:        Dummy band centre; any positive value is fine.

    Returns:
        Instrument with a single Channel carrying placeholder noise
        parameters. Use with FisherForecast(external_noise_bb=...).
    """
    dummy_channel = Channel(
        nu_ghz=nu_ghz,
        n_detectors=1,
        net_per_detector=1.0,   # placeholder; overridden by external noise
        beam_fwhm_arcmin=1.0,   # placeholder; spectra are beam-deconvolved
        efficiency=_L2_EFFICIENCY,
    )
    return Instrument(
        channels=(dummy_channel,),
        mission_duration_years=mission_years,
        f_sky=f_sky,
        requires_external_noise=True,
    )


# ---------------------------------------------------------------------------
# Ground-based efficiency defaults
# ---------------------------------------------------------------------------

# Atmospheric experiments: low observing efficiency due to weather and
# scan-strategy turnarounds; no cosmic-ray deadtime concern.
_GROUND_EFFICIENCY = ScalarEfficiency(
    detector_yield=0.80,
    observing_efficiency=0.25,   # weather, turnarounds, calibration
    data_cut_fraction=0.80,      # weather/quality cuts
    cosmic_ray_deadtime=1.00,    # not applicable on the ground
    polarization_efficiency=0.90,
)


def so_like() -> Instrument:
    """SO LAT-like ground-based wide survey (arXiv:1808.07445, Table 2 baseline).

    6 frequency bands from 27–280 GHz, ~21k TES bolometers, 5-year survey
    over f_sky=0.4. NET values are tuned to reproduce the SO baseline map
    depths: 71/36/8/10/22/54 μK-arcmin at 27/39/93/145/225/280 GHz.

    Notes:
        - 1/f atmospheric noise is NOT included (set knee_ell > 0 for detailed
          modeling).  Forecasts are therefore optimistic at ℓ ≲ 30.
        - Effective ground ℓ_min is ~30 after common-mode filtering.
        - No frequency coverage above 280 GHz (atmosphere opaque).
    """
    eff = _GROUND_EFFICIENCY
    # (nu_ghz, n_det, NET [μK√s], FWHM [arcmin])
    # FWHM scaled as 1.4' × (145/ν) for a ~6 m aperture (SO LAT).
    _bands = [
        ( 27.0,   36,  190.0, 7.4),
        ( 39.0,   36,   95.0, 5.1),
        ( 93.0, 7600,  305.0, 2.2),
        (145.0, 9800,  433.0, 1.4),
        (225.0, 2500,  481.0, 0.9),
        (280.0, 1000,  747.0, 0.7),
    ]
    channels = tuple(
        Channel(nu_ghz=nu, n_detectors=nd, net_per_detector=net,
                beam_fwhm_arcmin=fwhm, efficiency=eff)
        for nu, nd, net, fwhm in _bands
    )
    return Instrument(channels=channels, mission_duration_years=5.0, f_sky=0.4)


def cmbs4_like() -> Instrument:
    """CMB-S4-like ground-based wide survey (arXiv:2203.08024, CDT report).

    7 frequency bands from 30–270 GHz, ~343k TES bolometers, 7-year survey
    over f_sky=0.4. NET values reproduce approximate CMB-S4 wide-survey target
    depths: ~50/30/3.5/3.0/1.7/5.0/10 μK-arcmin at 30/40/85/95/150/220/270 GHz.

    Notes:
        - 1/f atmospheric noise NOT included; effective ℓ_min ≈ 30 from ground.
        - No frequency coverage above 270 GHz.
        - Deep delensing patch (f_sky~0.03, ~0.5 μK-arcmin at 150 GHz) not
          modelled here; this is the wide-survey configuration only.
    """
    eff = _GROUND_EFFICIENCY
    # (nu_ghz, n_det, NET [μK√s], FWHM [arcmin])
    # FWHM scaled as 1.5' × (150/ν) for a ~6 m aperture.
    _bands = [
        ( 30.0,    200,  370.0, 7.4),
        ( 40.0,    280,  260.0, 5.6),
        ( 85.0,  50000,  405.0, 2.6),
        ( 95.0,  68000,  405.0, 2.3),
        (150.0, 150000,  340.0, 1.5),
        (220.0,  50000,  580.0, 1.0),
        (270.0,  25000,  820.0, 0.8),
    ]
    channels = tuple(
        Channel(nu_ghz=nu, n_detectors=nd, net_per_detector=net,
                beam_fwhm_arcmin=fwhm, efficiency=eff)
        for nu, nd, net, fwhm in _bands
    )
    return Instrument(channels=channels, mission_duration_years=7.0, f_sky=0.4)
