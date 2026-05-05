"""
instrument.py — Instrument specification and noise power spectra.

Defines the instrument as frozen dataclasses (Channel, Instrument) with
physically-named efficiency factors. Computes N_ℓ^BB from first principles:

    N_ℓ = w⁻¹ / B_ℓ² × [1 + (ℓ_knee/ℓ)^α_knee]

where the white noise level w⁻¹ is built up from NET per detector,
detector count, mission duration, sky fraction, and scalar efficiency
factors. The 1/f term is per-channel (different detectors/readout can
have different knee frequencies).

All noise functions are pure functions of their arguments (no hidden
state), suitable for JIT compilation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0   # 31,557,600 s
ARCMIN_TO_RAD = jnp.pi / (180.0 * 60.0)


# -----------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarEfficiency:
    """Multiplicative efficiency factors that reduce effective integration time.

    Each factor is in [0, 1]. Their product η_total enters as an effective
    reduction in the number of detector-seconds of integration.

    Field defaults match :data:`L2_EFFICIENCY` (η_total = 0.711); see the
    module-level platform presets ``L2_EFFICIENCY`` / ``GROUND_EFFICIENCY``
    / ``IDEALIZED_EFFICIENCY`` for the canonical platform-specific values.

    Attributes:
        detector_yield:         Fraction of optically-coupled detectors that work.
        observing_efficiency:   Science observing time / total mission time
                                (accounts for calibration, slewing, SAA crossings, etc.)
        data_cut_fraction:      Fraction of timestream data passing quality cuts
                                (glitch flagging, weather cuts for ground, etc.)
        cosmic_ray_deadtime:    1 - fractional deadtime from cosmic ray glitch
                                recovery. Primarily relevant at L2 (~2-5% loss).
        polarization_efficiency: HWP or polarization modulator efficiency.
                                < 1 if modulator doesn't perfectly separate Q/U.
    """
    detector_yield: float = 0.85
    observing_efficiency: float = 0.85
    data_cut_fraction: float = 0.90
    cosmic_ray_deadtime: float = 0.97
    polarization_efficiency: float = 0.95

    @property
    def total(self) -> float:
        """Product of all efficiency factors."""
        return (self.detector_yield
                * self.observing_efficiency
                * self.data_cut_fraction
                * self.cosmic_ray_deadtime
                * self.polarization_efficiency)


# -----------------------------------------------------------------------
# Platform-specific efficiency presets
# -----------------------------------------------------------------------
# Single source of truth for the standard platform efficiencies; imported
# by config.py and telescope.py rather than redefined there.

# L2 space mission baseline: high yield/uptime, modest cosmic-ray deadtime.
# η_total = 0.711.
L2_EFFICIENCY = ScalarEfficiency(
    detector_yield=0.85,
    observing_efficiency=0.85,
    data_cut_fraction=0.90,
    cosmic_ray_deadtime=0.97,   # ~3% deadtime from GCR glitches at L2
    polarization_efficiency=0.95,
)

# Ground-based atmospheric experiments: low observing efficiency due to
# weather and scan-strategy turnarounds; cosmic rays not a concern.
# η_total = 0.137.
GROUND_EFFICIENCY = ScalarEfficiency(
    detector_yield=0.80,
    observing_efficiency=0.25,   # weather, turnarounds, calibration
    data_cut_fraction=0.80,      # weather/quality cuts
    cosmic_ray_deadtime=1.00,    # not applicable on the ground
    polarization_efficiency=0.90,
)

# Idealized space mission ("idealized" telescope-design variants): PICO-like
# observing efficiency assumption; everything else matches L2.
# η_total = 0.762.
IDEALIZED_EFFICIENCY = ScalarEfficiency(
    detector_yield=0.85,
    observing_efficiency=0.95,   # PICO assumption
    data_cut_fraction=0.90,
    cosmic_ray_deadtime=0.97,
    polarization_efficiency=0.95,
)


@dataclass(frozen=True)
class Channel:
    """A single frequency channel of the instrument.

    Attributes:
        nu_ghz:            Band center frequency [GHz].
        n_detectors:       Number of polarization-sensitive detectors.
                           For a pair-differencing experiment this is
                           the number of detector pairs × 2.
        net_per_detector:  Noise equivalent temperature per detector [μK√s],
                           in CMB thermodynamic units. This is the
                           *single-detector* NET for *temperature* (not
                           polarization). The √2 factor for polarization
                           sensitivity is applied in the noise calculation.
        beam_fwhm_arcmin:  Beam full width at half maximum [arcmin].
        knee_ell:          1/f noise knee multipole for this channel.
                           Set to 0 for white noise only.
        alpha_knee:        1/f noise spectral index in multipole space.
                           N_ℓ ∝ (ℓ_knee/ℓ)^alpha_knee at ℓ < ℓ_knee.
        efficiency:        Scalar efficiency factors for this channel.
    """
    nu_ghz: float
    n_detectors: int
    net_per_detector: float
    beam_fwhm_arcmin: float
    knee_ell: float = 0.0
    alpha_knee: float = 1.0
    efficiency: ScalarEfficiency = L2_EFFICIENCY


@dataclass(frozen=True)
class Instrument:
    """Full instrument specification.

    Attributes:
        channels:               Tuple of Channel objects (immutable).
        mission_duration_years: Total mission lifetime [years].
        f_sky:                  Observed sky fraction (0 < f_sky ≤ 1).
        requires_external_noise: If True, the Channel noise parameters are
                                placeholders and FisherForecast MUST be
                                supplied with external_noise_bb. Set by
                                presets such as cleaned_map_instrument
                                that represent a post-component-separation
                                cleaned map whose noise comes from a
                                sim-based pipeline rather than NET × √t_obs.
    """
    channels: tuple[Channel, ...]
    mission_duration_years: float = 5.0
    f_sky: float = 0.7
    requires_external_noise: bool = False


# -----------------------------------------------------------------------
# Noise computation — pure functions
# -----------------------------------------------------------------------

def white_noise_power(channel: Channel,
                      mission_years: float,
                      f_sky: float) -> float:
    """White noise power w⁻¹ for a single channel in polarization (BB).

    w⁻¹ = (NET_det × √2)² × 4π f_sky / (N_det × η_total × t_obs)

    The √2 converts single-detector temperature NET to polarization NET
    (each detector measures one linear polarization; you need both Q and U,
    and each has √2 higher noise than temperature from the same detector).

    Returns:
        w_inv: White noise power [μK² sr]. This is the noise per steradian
               in the map, before beam deconvolution.
    """
    t_obs = mission_years * SECONDS_PER_YEAR
    eta = channel.efficiency.total
    net_pol_sq = (channel.net_per_detector * jnp.sqrt(2.0)) ** 2
    omega_survey = 4.0 * jnp.pi * f_sky
    return net_pol_sq * omega_survey / (channel.n_detectors * eta * t_obs)


def beam_bl(ells: jnp.ndarray, fwhm_arcmin: float) -> jnp.ndarray:
    """Gaussian beam transfer function B_ℓ.

    B_ℓ = exp(-ℓ(ℓ+1) σ_beam² / 2)
    σ_beam = FWHM / √(8 ln 2)

    Args:
        ells:          1-D array of multipole values.
        fwhm_arcmin:   Beam FWHM [arcmin].

    Returns:
        B_ℓ array of same shape as ells.
    """
    sigma_beam = fwhm_arcmin * ARCMIN_TO_RAD / jnp.sqrt(8.0 * jnp.log(2.0))
    return jnp.exp(-ells * (ells + 1.0) * sigma_beam**2 / 2.0)


def deconvolve_noise_bb(noise_convolved: jnp.ndarray,
                        ells: jnp.ndarray,
                        beam_fwhm_arcmin: float) -> jnp.ndarray:
    """Deconvolve a Gaussian beam from a noise auto-spectrum.

    Use this when you have a noise power spectrum N_ℓ measured from a
    beam-smoothed map (e.g. an anafast auto-spectrum of a simulated noise
    realization before beam deconvolution), and need the beam-free N_ℓ
    for augr's FisherForecast(external_noise_bb=...) or
    bandpower_covariance_blocks_from_noise(noise_nls=...), which both
    assume beam-deconvolved input.

        N_ℓ^{deconv} = N_ℓ^{convolved} / B_ℓ²

    NILC / GNILC / compsep pipelines typically return already-deconvolved
    spectra; use this helper only when you are sure the input is still
    beam-convolved.

    **Numerical range caveat**: B_ℓ² falls off like exp(-ℓ²σ_beam²), so
    it drops below float64 precision (~1e-300) at ℓ ~ 20/σ_beam ~
    (20·√(8 ln 2) · 10800/π) / FWHM_arcmin ~ 180 / (FWHM/5'). For a 30'
    beam the safe range is roughly ℓ ≲ 900; for a 70' LFT beam it is
    roughly ℓ ≲ 400.  Division above those ranges silently produces
    astronomical or infinite values.  Restrict the input to a sane ℓ
    range before calling, or clip the output.

    Args:
        noise_convolved: Beam-convolved noise spectrum on the given ell
                         grid, shape (n_ells,) or (n_chan, n_ells).
        ells:            Multipoles, shape (n_ells,).
        beam_fwhm_arcmin: Beam FWHM [arcmin].

    Returns:
        Beam-deconvolved noise, same shape as noise_convolved.
    """
    bl2 = beam_bl(ells, beam_fwhm_arcmin) ** 2
    return noise_convolved / bl2


def noise_nl(channel: Channel,
             ells: jnp.ndarray,
             mission_years: float,
             f_sky: float) -> jnp.ndarray:
    """Noise power spectrum N_ℓ^BB for a single channel.

    N_ℓ = w⁻¹ / B_ℓ² × [1 + (ℓ_knee/ℓ)^α_knee]

    The 1/f term raises noise at low ℓ. Set channel.knee_ell = 0
    to get pure white noise.

    Args:
        channel:       Channel specification.
        ells:          1-D array of multipoles (should be > 0).
        mission_years: Mission duration [years].
        f_sky:         Observed sky fraction.

    Returns:
        N_ℓ array in [μK²] (dimensionless C_ℓ units).
    """
    w_inv = white_noise_power(channel, mission_years, f_sky)
    bl = beam_bl(ells, channel.beam_fwhm_arcmin)
    # 1/f noise: use jnp.where to avoid 0/0 when ℓ_knee=0 or ℓ=0
    one_over_f = jnp.where(
        (channel.knee_ell > 0) & (ells > 0),
        (channel.knee_ell / jnp.maximum(ells, 1.0)) ** channel.alpha_knee,
        0.0,
    )
    return w_inv / bl**2 * (1.0 + one_over_f)


def noise_nl_temperature(channel: Channel,
                         ells: jnp.ndarray,
                         mission_years: float,
                         f_sky: float) -> jnp.ndarray:
    """Noise power spectrum N_ℓ^TT for a single channel (temperature).

    Same as noise_nl() but without the √2 polarization factor:
    w⁻¹_T = w⁻¹_pol / 2.

    Returns:
        N_ℓ^TT array in [μK²].
    """
    return noise_nl(channel, ells, mission_years, f_sky) / 2.0


# -----------------------------------------------------------------------
# Functional noise interface (for differentiable optimization)
# -----------------------------------------------------------------------

def white_noise_power_continuous(
    net_per_detector: jnp.ndarray,
    n_detectors: jnp.ndarray,
    eta_total: jnp.ndarray,
    mission_years: float,
    f_sky: float,
) -> jnp.ndarray:
    """White noise power from raw JAX scalars (no Channel object).

    Same formula as white_noise_power(), but accepts raw values so that
    JAX can trace through the computation for gradient-based optimization.
    """
    t_obs = mission_years * SECONDS_PER_YEAR
    net_pol_sq = (net_per_detector * jnp.sqrt(2.0)) ** 2
    omega_survey = 4.0 * jnp.pi * f_sky
    return net_pol_sq * omega_survey / (n_detectors * eta_total * t_obs)


def noise_nl_continuous(
    net_per_detector: jnp.ndarray,
    n_detectors: jnp.ndarray,
    beam_fwhm_arcmin: jnp.ndarray,
    eta_total: jnp.ndarray,
    ells: jnp.ndarray,
    mission_years: float,
    f_sky: float,
    knee_ell: jnp.ndarray = 0.0,
    alpha_knee: jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Noise N_ℓ^BB from raw JAX scalars (no Channel object).

    Same formula as noise_nl(), but accepts raw values for differentiable
    optimization. All scalar arguments can be JAX-traced.
    """
    w_inv = white_noise_power_continuous(
        net_per_detector, n_detectors, eta_total, mission_years, f_sky)
    bl = beam_bl(ells, beam_fwhm_arcmin)
    one_over_f = jnp.where(
        (knee_ell > 0) & (ells > 0),
        (knee_ell / jnp.maximum(ells, 1.0)) ** alpha_knee,
        0.0,
    )
    return w_inv / bl**2 * (1.0 + one_over_f)


def combined_noise_nl(instrument: Instrument,
                      ells: jnp.ndarray,
                      spectrum: str = "BB") -> jnp.ndarray:
    """Inverse-variance combined noise across all channels.

    1/N_ℓ^{combined} = Σ_i 1/N_ℓ^i

    Args:
        instrument: Instrument specification.
        ells:       1-D array of multipoles.
        spectrum:   "BB" (or "EE") for polarization, "TT" for temperature.

    Returns:
        Combined N_ℓ array of same shape as ells [μK²].
    """
    noise_fn = noise_nl_temperature if spectrum == "TT" else noise_nl
    inv_nl_sum = jnp.zeros_like(ells, dtype=float)
    for ch in instrument.channels:
        nl_ch = noise_fn(ch, ells, instrument.mission_duration_years,
                         instrument.f_sky)
        inv_nl_sum = inv_nl_sum + 1.0 / nl_ch
    return 1.0 / inv_nl_sum


def noise_nl_matrix(instrument: Instrument,
                    ells: jnp.ndarray) -> jnp.ndarray:
    """Full noise matrix N_ℓ^{ij} across all channels.

    Cross-channel noise is zero (diagonal): independent detectors
    at different frequencies have uncorrelated noise.

    Args:
        instrument: Full instrument specification.
        ells:       1-D array of multipoles.

    Returns:
        Array of shape (n_channels, n_channels, n_ells).
        Only diagonal entries (i == j) are nonzero.
    """
    n_chan = len(instrument.channels)
    n_ell = len(ells)
    nl = jnp.zeros((n_chan, n_chan, n_ell))
    for i, ch in enumerate(instrument.channels):
        nl_i = noise_nl(ch, ells, instrument.mission_duration_years,
                        instrument.f_sky)
        nl = nl.at[i, i, :].set(nl_i)
    return nl
