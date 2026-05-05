"""
telescope.py — Physical telescope model for deriving instrument parameters.

Derives detector counts, beam sizes, and photon-noise-limited NETs from
physical telescope specifications (aperture, focal plane geometry, thermal
loading). Produces Instrument objects compatible with the Fisher forecast
pipeline.

Supports single-frequency and dichroic (two-band) feedhorn-coupled pixel
designs. Horn diameter is set by the lowest frequency in each pixel group;
focal plane area is allocated among groups by user-specified fractions.

References:
    - Feedhorn coupling: Griffin et al. 2002 (single-moded horns, d ~ 2Fλ)
    - Photon NEP: Zmuidzinas 2003 (JLTP 12, 4), Richards 1994
    - Hex packing: Conway & Sloane, "Sphere Packings" (π/2√3 ≈ 0.9069)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from augr.instrument import (
    Channel,
    IDEALIZED_EFFICIENCY,
    Instrument,
    L2_EFFICIENCY,
    ScalarEfficiency,
)
from augr.units import H_PLANCK, K_BOLTZMANN, C_LIGHT, T_CMB


# Conversion factor
_RAD_TO_ARCMIN = 180.0 * 60.0 / math.pi


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSpec:
    """Specification for a single frequency band.

    Attributes:
        nu_ghz: Band center frequency [GHz].
        fractional_bandwidth: Δν/ν (default 0.25, typical for feedhorns).
    """
    nu_ghz: float
    fractional_bandwidth: float = 0.25


@dataclass(frozen=True)
class PixelGroup:
    """One or two bands sharing a feedhorn.

    For single-frequency pixels, ``bands`` has length 1.
    For dichroic pixels, ``bands`` has length 2 (low freq first).
    The horn diameter is set by the lowest frequency in the group.
    Each horn yields 2 detectors per band (two orthogonal polarizations).

    Attributes:
        bands: Tuple of 1 or 2 BandSpec objects.
        area_fraction: Fraction of total focal plane area allocated to this
            group.  All fractions across groups must sum to 1.
    """
    bands: tuple[BandSpec, ...]
    area_fraction: float

    def __post_init__(self) -> None:
        if not 1 <= len(self.bands) <= 2:
            raise ValueError(
                f"PixelGroup supports 1 or 2 bands, got {len(self.bands)}"
            )
        if len(self.bands) == 2 and self.bands[0].nu_ghz >= self.bands[1].nu_ghz:
            raise ValueError(
                "Dichroic pair must have bands[0].nu_ghz < bands[1].nu_ghz"
            )


@dataclass(frozen=True)
class FocalPlaneSpec:
    """Telescope optics and focal plane geometry.

    Attributes:
        aperture_m: Primary mirror diameter [m].
        f_number: Focal ratio f/D (typically 1.5–2.5).
        fp_diameter_m: Usable focal plane diameter [m].
        illumination_factor: FWHM = factor × λ/D. Default 1.22 (Airy).
        packing_efficiency: Fraction of ideal hex-packed area that is
            actually usable after accounting for mechanical margins,
            readout wiring, and edge effects. Default 0.80.
    """
    aperture_m: float
    f_number: float
    fp_diameter_m: float
    illumination_factor: float = 1.22
    packing_efficiency: float = 0.80


@dataclass(frozen=True)
class ThermalSpec:
    """Thermal loading model for photon noise calculation.

    Single effective emissivity model: the telescope contributes
    ε × B(T_tel) to the optical loading on each detector.

    Attributes:
        T_telescope_K: Telescope physical temperature [K].
        emissivity: Effective emissivity of warm optics (dimensionless).
        eta_optical: End-to-end optical efficiency from sky to detector.
    """
    T_telescope_K: float = 4.0
    emissivity: float = 0.01
    eta_optical: float = 0.35


@dataclass(frozen=True)
class TelescopeDesign:
    """Complete telescope design specification.

    Holds everything needed to derive an Instrument via ``to_instrument()``.

    Attributes:
        focal_plane: Optics and focal plane geometry.
        thermal: Thermal loading parameters.
        pixel_groups: Tuple of PixelGroup objects defining the band layout
            and focal plane area allocation.
        mission_duration_years: Total mission duration [years].
        f_sky: Sky fraction observed.
        efficiency: ScalarEfficiency applied to all channels.
        knee_ell: 1/f knee multipole (0 = white noise only).
        alpha_knee: 1/f spectral index in ell-space.
    """
    focal_plane: FocalPlaneSpec
    thermal: ThermalSpec
    pixel_groups: tuple[PixelGroup, ...]
    mission_duration_years: float = 5.0
    f_sky: float = 0.7
    efficiency: ScalarEfficiency = L2_EFFICIENCY
    knee_ell: float = 0.0
    alpha_knee: float = 1.0


# ---------------------------------------------------------------------------
# Geometry functions
# ---------------------------------------------------------------------------

def beam_fwhm_arcmin(nu_ghz: float, aperture_m: float,
                     illumination_factor: float = 1.22) -> float:
    """Diffraction-limited beam FWHM [arcmin].

    FWHM = illumination_factor × λ/D, converted to arcmin.

    Args:
        nu_ghz: Frequency [GHz].
        aperture_m: Aperture diameter [m].
        illumination_factor: Multiplier on λ/D (1.22 for Airy disk).

    Returns:
        Beam FWHM in arcmin.
    """
    wavelength_m = C_LIGHT / (nu_ghz * 1e9)
    fwhm_rad = illumination_factor * wavelength_m / aperture_m
    return fwhm_rad * _RAD_TO_ARCMIN


def horn_diameter(nu_ghz: float, f_number: float) -> float:
    """Feedhorn aperture diameter for single-mode coupling [m].

    d_horn = 2 × F × λ, where F is the focal ratio and λ = c/ν.

    Args:
        nu_ghz: Frequency [GHz].
        f_number: Telescope focal ratio f/D.

    Returns:
        Horn diameter in meters.
    """
    wavelength_m = C_LIGHT / (nu_ghz * 1e9)
    return 2.0 * f_number * wavelength_m


def hex_cell_area(diameter: float) -> float:
    """Area of one hexagonal cell in a close-packed array [m²].

    For circles of the given diameter packed in a hexagonal lattice,
    each cell occupies (√3/2) × d².

    Args:
        diameter: Circle (horn) diameter [m].

    Returns:
        Cell area in m².
    """
    return (math.sqrt(3) / 2.0) * diameter ** 2


def count_pixels(fp_area: float, cell_area: float,
                 packing_efficiency: float) -> int:
    """Number of pixels that fit in a focal plane area.

    n = floor(packing_efficiency × fp_area / cell_area)

    Uses floor() to conservatively avoid overcounting.

    Args:
        fp_area: Allocated focal plane area [m²].
        cell_area: Hex cell area per pixel [m²].
        packing_efficiency: Fraction of ideal packing achieved.

    Returns:
        Integer number of pixels (≥ 0).
    """
    if cell_area <= 0:
        raise ValueError(f"cell_area must be positive, got {cell_area}")
    n = packing_efficiency * fp_area / cell_area
    return max(0, int(math.floor(n)))


def count_pixels_continuous(fp_area: jnp.ndarray, cell_area: jnp.ndarray,
                            packing_efficiency: float) -> jnp.ndarray:
    """Continuous relaxation of count_pixels for gradient-based optimization.

    Returns the real-valued pixel count without floor(), so that JAX can
    differentiate through the focal plane packing. Round to integer after
    optimization to get the physical instrument.
    """
    return jnp.maximum(0.0, packing_efficiency * fp_area / cell_area)


def photon_noise_net_jax(
    nu_ghz: jnp.ndarray,
    fractional_bandwidth: float = 0.25,
    T_cmb: float = T_CMB,
    T_telescope: float = 4.0,
    emissivity: float = 0.01,
    eta_optical: float = 0.35,
    n_quad: int = 512,
) -> jnp.ndarray:
    """JAX-traceable photon-noise NET [μK√s].

    Same physics as photon_noise_net() but uses jnp instead of np, enabling
    differentiation w.r.t. telescope thermal/optical parameters. See
    photon_noise_net() for the full docstring, including the list of
    loading sources omitted from n_total (galactic foregrounds,
    atmosphere, zodi).
    """
    nu_center_hz = nu_ghz * 1e9
    delta_nu_hz = fractional_bandwidth * nu_center_hz
    nu_lo = nu_center_hz - delta_nu_hz / 2.0
    nu_hi = nu_center_hz + delta_nu_hz / 2.0
    nu = jnp.linspace(nu_lo, nu_hi, n_quad)

    h = H_PLANCK
    k = K_BOLTZMANN

    x_cmb = h * nu / (k * T_cmb)
    n_cmb = 1.0 / (jnp.exp(x_cmb) - 1.0)

    x_tel = h * nu / (k * T_telescope)
    n_tel = 1.0 / (jnp.exp(x_tel) - 1.0)

    # TODO(extend): see photon_noise_net() — needs an extra_loading term
    # for galactic-FG / atmospheric repurposings.
    n_total = eta_optical * n_cmb + emissivity * n_tel

    integrand_nep2 = 2.0 * h**2 * nu**2 * n_total * (1.0 + n_total)
    nep_squared = 2.0 * jnp.trapezoid(integrand_nep2, nu)

    ex_cmb = jnp.exp(x_cmb)
    dndt_cmb = (h * nu / (k * T_cmb**2)) * ex_cmb / (ex_cmb - 1.0)**2
    integrand_dpdt = 2.0 * h * nu * dndt_cmb * eta_optical
    dpdt = jnp.trapezoid(integrand_dpdt, nu)

    net_K = jnp.sqrt(nep_squared) / jnp.abs(dpdt)
    return net_K * 1e6


# ---------------------------------------------------------------------------
# Photon noise calculation
# ---------------------------------------------------------------------------

def photon_noise_net(
    nu_ghz: float,
    fractional_bandwidth: float = 0.25,
    T_cmb: float = T_CMB,
    T_telescope: float = 4.0,
    emissivity: float = 0.01,
    eta_optical: float = 0.35,
    n_quad: int = 512,
) -> float:
    """Photon-noise-limited single-detector temperature NET [μK√s].

    Computes the fundamental photon noise limit for a single-moded detector
    with throughput AΩ = λ² observing through a cold telescope at L2.

    Includes both shot noise and wave bunching (full Bose-Einstein statistics):
        NEP² = 2 ∫ 2h²ν² n(1+n) dν

    where n is the total occupation number from CMB + telescope emission,
    and the ν² (rather than ν⁴) comes from the single-mode AΩ = (c/ν)².

    The returned NET is the single-detector, single-polarization, temperature
    NET in CMB thermodynamic units, matching the Channel.net_per_detector
    convention. The √2 polarization factor is applied downstream in
    white_noise_power().

    Loading sources NOT included in n_total
    ---------------------------------------
    - **Galactic foregrounds** (diffuse dust + synchrotron). At ν ≲ 300 GHz
      and high galactic latitude these contribute brightness temperatures
      orders of magnitude below CMB+telescope, so the omission costs ≪1 %
      on NET. Breaks down at the high end of the probe-class submillimetre
      band (≳ 600 GHz, where dust's modified blackbody overtakes the CMB
      Wien tail) and at low galactic latitude.
    - **Atmospheric loading.** Zero by assumption — this routine bakes in
      an L2 orbit. Re-using it for a ground-based or balloon concept
      requires adding water-vapour and O2 loading.
    - **Zodiacal light, Earth/Moon limb sidelobe pickup.** Sub-percent for
      typical L2 sun-shielded geometries; ignored.

    See the TODO at the n_total assignment in the body for how to extend
    n_total = η_opt n_cmb + ε n_tel with an explicit extra-loading term.

    Args:
        nu_ghz: Band center frequency [GHz].
        fractional_bandwidth: Δν/ν (default 0.25).
        T_cmb: CMB temperature [K].
        T_telescope: Telescope temperature [K].
        emissivity: Effective telescope emissivity.
        eta_optical: End-to-end optical efficiency (sky to detector).
        n_quad: Number of quadrature points for integration.

    Returns:
        NET in μK√s.
    """
    nu_center_hz = nu_ghz * 1e9
    delta_nu_hz = fractional_bandwidth * nu_center_hz
    nu_lo = nu_center_hz - delta_nu_hz / 2.0
    nu_hi = nu_center_hz + delta_nu_hz / 2.0
    nu = np.linspace(nu_lo, nu_hi, n_quad)

    h = H_PLANCK
    k = K_BOLTZMANN

    # Bose-Einstein occupation numbers
    x_cmb = h * nu / (k * T_cmb)
    n_cmb = 1.0 / (np.exp(x_cmb) - 1.0)

    x_tel = h * nu / (k * T_telescope)
    n_tel = 1.0 / (np.exp(x_tel) - 1.0)

    # Total occupation number seen by the detector
    # CMB couples through full optical efficiency; telescope emits with
    # emissivity epsilon (seen by detector through remaining optics).
    # TODO(extend): add an `extra_loading` term (Bose-Einstein occupation
    # number, or a per-frequency lookup) so callers can fold in galactic
    # foreground loading at high ν / low galactic latitude, atmospheric
    # loading for ground/balloon repurposings, etc. See the "Loading
    # sources NOT included" block in the docstring above.
    n_total = eta_optical * n_cmb + emissivity * n_tel

    # Photon NEP² (single spatial mode: AΩ = λ² = (c/ν)²)
    # NEP² = 2 × ∫ (AΩ/c²) × 2h²ν⁴ × n(1+n) dν
    #       = 2 × ∫ 2h²ν² × n(1+n) dν     [since AΩ/c² = 1/ν²]
    integrand_nep2 = 2.0 * h**2 * nu**2 * n_total * (1.0 + n_total)
    nep_squared = 2.0 * np.trapezoid(integrand_nep2, nu)

    # CMB power responsivity: dP_CMB/dT_CMB
    # dP/dT = ∫ (AΩ/c²) × 2hν³ × (dn_BE/dT) × η_opt dν
    #       = ∫ 2hν × (dn_BE/dT) × η_opt dν
    # where dn_BE/dT = (hν/kT²) × exp(x) / (exp(x) - 1)²
    ex_cmb = np.exp(x_cmb)
    dndt_cmb = (h * nu / (k * T_cmb**2)) * ex_cmb / (ex_cmb - 1.0)**2
    integrand_dpdt = 2.0 * h * nu * dndt_cmb * eta_optical
    dpdt = np.trapezoid(integrand_dpdt, nu)

    # NET_CMB = NEP / |dP/dT| in K√s
    net_K = np.sqrt(nep_squared) / abs(dpdt)

    # Convert K -> μK
    return float(net_K * 1e6)


# ---------------------------------------------------------------------------
# Instrument builder
# ---------------------------------------------------------------------------

def to_instrument(design: TelescopeDesign) -> Instrument:
    """Convert a TelescopeDesign into an Instrument for Fisher forecasting.

    Derives beam sizes from diffraction, detector counts from focal plane
    packing, and NETs from photon noise. Returns an Instrument compatible
    with FisherForecast.

    Args:
        design: Complete telescope design specification.

    Returns:
        An Instrument with one Channel per band.

    Raises:
        ValueError: If area fractions do not sum to ~1.
    """
    fp = design.focal_plane
    th = design.thermal

    # Validate area fractions
    total_frac = sum(pg.area_fraction for pg in design.pixel_groups)
    if abs(total_frac - 1.0) > 0.01:
        raise ValueError(
            f"Area fractions must sum to 1.0, got {total_frac:.4f}"
        )

    # Total focal plane area
    a_fp = math.pi * (fp.fp_diameter_m / 2.0) ** 2

    # Build channels from each pixel group
    channels = []
    for pg in design.pixel_groups:
        # Horn size set by lowest frequency
        nu_low = min(b.nu_ghz for b in pg.bands)
        d_horn = horn_diameter(nu_low, fp.f_number)
        a_cell = hex_cell_area(d_horn)
        a_alloc = pg.area_fraction * a_fp
        n_pixels = count_pixels(a_alloc, a_cell, fp.packing_efficiency)
        # Each horn has 2 orthogonal polarization detectors per band.
        # Dichroic pixels (multiple bands per horn) share the same horn
        # but have independent detector pairs, so n_det is per-band.
        n_det = 2 * n_pixels

        for band in pg.bands:
            fwhm = beam_fwhm_arcmin(
                band.nu_ghz, fp.aperture_m, fp.illumination_factor
            )
            net = photon_noise_net(
                nu_ghz=band.nu_ghz,
                fractional_bandwidth=band.fractional_bandwidth,
                T_cmb=T_CMB,
                T_telescope=th.T_telescope_K,
                emissivity=th.emissivity,
                eta_optical=th.eta_optical,
            )
            channels.append(Channel(
                nu_ghz=band.nu_ghz,
                n_detectors=n_det,
                net_per_detector=net,
                beam_fwhm_arcmin=fwhm,
                efficiency=design.efficiency,
                knee_ell=design.knee_ell,
                alpha_knee=design.alpha_knee,
            ))

    # Sort by frequency (convention matching config.py presets)
    channels.sort(key=lambda ch: ch.nu_ghz)

    return Instrument(
        channels=tuple(channels),
        mission_duration_years=design.mission_duration_years,
        f_sky=design.f_sky,
    )


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def probe_design() -> TelescopeDesign:
    """Probe-class (~$1B) space mission design.

    1.5 m aperture, f/2 optics, 0.4 m focal plane.
    6 bands in 3 dichroic pairs: (30, 40), (85, 150), (220, 340) GHz.
    Equal area allocation, 5-year L2 mission, f_sky = 0.7.
    """
    pixel_groups = (
        PixelGroup(
            bands=(BandSpec(30.0), BandSpec(40.0)),
            area_fraction=1.0 / 3.0,
        ),
        PixelGroup(
            bands=(BandSpec(85.0), BandSpec(150.0)),
            area_fraction=1.0 / 3.0,
        ),
        PixelGroup(
            bands=(BandSpec(220.0), BandSpec(340.0)),
            area_fraction=1.0 / 3.0,
        ),
    )
    return TelescopeDesign(
        focal_plane=FocalPlaneSpec(
            aperture_m=1.5, f_number=2.0, fp_diameter_m=0.4,
        ),
        thermal=ThermalSpec(),
        pixel_groups=pixel_groups,
    )


def flagship_design() -> TelescopeDesign:
    """Flagship-class (~$2B) space mission design.

    3.0 m aperture, f/2 optics, 0.6 m focal plane.
    8 bands in 4 dichroic pairs: (30, 40), (85, 150), (220, 280),
    (340, 500) GHz. Equal area allocation, 5-year L2 mission, f_sky = 0.7.
    """
    pixel_groups = (
        PixelGroup(
            bands=(BandSpec(30.0), BandSpec(40.0)),
            area_fraction=0.25,
        ),
        PixelGroup(
            bands=(BandSpec(85.0), BandSpec(150.0)),
            area_fraction=0.25,
        ),
        PixelGroup(
            bands=(BandSpec(220.0), BandSpec(280.0)),
            area_fraction=0.25,
        ),
        PixelGroup(
            bands=(BandSpec(340.0), BandSpec(500.0)),
            area_fraction=0.25,
        ),
    )
    return TelescopeDesign(
        focal_plane=FocalPlaneSpec(
            aperture_m=3.0, f_number=2.0, fp_diameter_m=0.6,
        ),
        thermal=ThermalSpec(),
        pixel_groups=pixel_groups,
    )


# ---------------------------------------------------------------------------
# "PICO-comparable" idealized designs
#
# These use PICO-level optimism for everything that isn't constrained by
# the choice of feedhorn-coupled (rather than sinuous) detectors:
#   - f/1.42 focal ratio (PICO's open-Dragone, achievable with feedhorns)
#   - eta_optical = 0.50 (optimistic but plausible for feedhorns; PICO
#     claimed >70% for sinuous, but their CBE NETs imply ~50-55% end-to-end)
#   - 95% observing efficiency (PICO Sec 2.10)
#   - Same focal plane area as PICO (~2900 cm^2 from Fig 3.3)
#   - Same 25% fractional bandwidth, 5-year mission, f_sky = 0.7
# The constraint from feedhorns: 2Fλ pixel pitch (vs PICO's ~Fλ sinuous).
# This means ~4x fewer pixels per unit FP area at the same f/#.
# ---------------------------------------------------------------------------

_IDEALIZED_THERMAL = ThermalSpec(
    T_telescope_K=4.5,     # PICO secondary reflector temperature
    emissivity=0.01,
    eta_optical=0.50,      # Optimistic feedhorn; PICO claimed >0.70 for sinuous
)

# PICO focal plane: 78 cm × 47 cm ellipse ≈ 2880 cm².
# Equivalent circular diameter = 2 * sqrt(0.78*0.47 / (4*pi)) * 2... no,
# area of ellipse = pi * a * b = pi * 0.39 * 0.235 = 0.2878 m²
# equivalent circle diameter = 2 * sqrt(0.2878/pi) = 0.605 m
_PICO_FP_EQUIV_DIAMETER = 0.605  # m


def probe_idealized() -> TelescopeDesign:
    """Probe-class with PICO-level optimism, feedhorn-coupled.

    1.5 m aperture, f/1.42 (PICO's focal ratio), PICO-sized focal plane.
    eta_optical = 0.50 (optimistic feedhorn), 95% observing efficiency.
    12 bands in 6 dichroic pairs spanning 30-400 GHz.
    Weighted area allocation: 15% low-freq, 50% CMB, 35% high-freq.
    """
    pixel_groups = (
        PixelGroup(
            bands=(BandSpec(30.0), BandSpec(40.0)),
            area_fraction=0.075,
        ),
        PixelGroup(
            bands=(BandSpec(55.0), BandSpec(75.0)),
            area_fraction=0.075,
        ),
        PixelGroup(
            bands=(BandSpec(95.0), BandSpec(130.0)),
            area_fraction=0.25,
        ),
        PixelGroup(
            bands=(BandSpec(155.0), BandSpec(195.0)),
            area_fraction=0.25,
        ),
        PixelGroup(
            bands=(BandSpec(235.0), BandSpec(290.0)),
            area_fraction=0.175,
        ),
        PixelGroup(
            bands=(BandSpec(340.0), BandSpec(400.0)),
            area_fraction=0.175,
        ),
    )
    return TelescopeDesign(
        focal_plane=FocalPlaneSpec(
            aperture_m=1.5, f_number=1.42, fp_diameter_m=_PICO_FP_EQUIV_DIAMETER,
        ),
        thermal=_IDEALIZED_THERMAL,
        pixel_groups=pixel_groups,
        efficiency=IDEALIZED_EFFICIENCY,
    )


def flagship_idealized() -> TelescopeDesign:
    """Flagship-class with PICO-level optimism, feedhorn-coupled.

    3.0 m aperture, f/1.42, 0.8 m focal plane (larger than PICO's,
    reflecting flagship budget).
    eta_optical = 0.50, 95% observing efficiency.
    16 bands in 8 dichroic pairs spanning 25-550 GHz.
    Weighted area allocation: 10% low-freq, 50% CMB, 40% high-freq.
    """
    pixel_groups = (
        PixelGroup(
            bands=(BandSpec(25.0), BandSpec(35.0)),
            area_fraction=0.05,
        ),
        PixelGroup(
            bands=(BandSpec(45.0), BandSpec(60.0)),
            area_fraction=0.05,
        ),
        PixelGroup(
            bands=(BandSpec(75.0), BandSpec(100.0)),
            area_fraction=0.50 / 3,
        ),
        PixelGroup(
            bands=(BandSpec(120.0), BandSpec(150.0)),
            area_fraction=0.50 / 3,
        ),
        PixelGroup(
            bands=(BandSpec(175.0), BandSpec(210.0)),
            area_fraction=0.50 / 3,
        ),
        PixelGroup(
            bands=(BandSpec(250.0), BandSpec(300.0)),
            area_fraction=0.40 / 3,
        ),
        PixelGroup(
            bands=(BandSpec(340.0), BandSpec(400.0)),
            area_fraction=0.40 / 3,
        ),
        PixelGroup(
            bands=(BandSpec(450.0), BandSpec(550.0)),
            area_fraction=0.40 / 3,
        ),
    )
    return TelescopeDesign(
        focal_plane=FocalPlaneSpec(
            aperture_m=3.0, f_number=1.42, fp_diameter_m=0.8,
        ),
        thermal=_IDEALIZED_THERMAL,
        pixel_groups=pixel_groups,
        efficiency=IDEALIZED_EFFICIENCY,
    )
