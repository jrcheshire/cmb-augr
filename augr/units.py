"""
units.py — Physical constants and unit conversions for CMB forecasting.

All functions are JAX-traceable (jnp operations only, no Python control flow
on traced values). Frequencies in GHz, temperatures in K, spectra in μK².

Unit convention: CMB thermodynamic temperature units throughout.
"""

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

T_CMB = 2.7255          # K, Fixsen 2009
H_PLANCK = 6.62607015e-34  # J·s
K_BOLTZMANN = 1.380649e-23  # J/K
C_LIGHT = 2.99792458e8     # m/s

# Reference frequencies for foreground SEDs
NU_DUST_REF_GHZ = 353.0   # GHz
NU_SYNC_REF_GHZ = 23.0    # GHz

# ---------------------------------------------------------------------------
# Core unit conversion
# ---------------------------------------------------------------------------

def x_factor(nu_ghz: float) -> float:
    """Dimensionless frequency: x = hν / kT_CMB."""
    nu_hz = nu_ghz * 1e9
    return H_PLANCK * nu_hz / (K_BOLTZMANN * T_CMB)


def rj_to_cmb(nu_ghz: float) -> float:
    """Unit conversion factor: ΔT_RJ / ΔT_CMB = x² eˣ / (eˣ - 1)²

    This is < 1 at typical CMB frequencies (e.g. ~0.576 at 150 GHz): a given
    brightness fluctuation corresponds to a *larger* thermodynamic temperature
    than Rayleigh-Jeans would imply, because the CMB spectrum curves away from
    RJ at these frequencies.

    Primary use: foreground SED unit conversions via the ratio
        rj_to_cmb(ν_ref) / rj_to_cmb(ν)
    which converts a SED from RJ-normalised brightness to CMB thermodynamic
    units (normalised to 1 at ν_ref).

    To convert a temperature map from RJ to CMB units, use cmb_to_rj instead.
    """
    x = x_factor(nu_ghz)
    ex = jnp.exp(x)
    return x**2 * ex / (ex - 1.0)**2


def cmb_to_rj(nu_ghz: float) -> float:
    """Inverse of rj_to_cmb: CMB thermodynamic → Rayleigh-Jeans units."""
    return 1.0 / rj_to_cmb(nu_ghz)


# ---------------------------------------------------------------------------
# Foreground spectral energy distributions (SEDs)
# Normalized to 1 at their respective reference frequencies.
# All returned values are dimensionless frequency scalings in CMB thermo units.
# ---------------------------------------------------------------------------

def dust_sed(nu_ghz: float, beta_d: float, T_d: float,
             nu_ref_ghz: float = NU_DUST_REF_GHZ) -> float:
    """Modified blackbody dust SED in CMB thermodynamic units.

    f_d(ν) = (ν/ν_ref)^(β_d+1) × [exp(hν_ref/kT_d) - 1] / [exp(hν/kT_d) - 1]
             × rj_to_cmb(ν_ref) / rj_to_cmb(ν)

    Normalized so f_d(ν_ref) = 1. Default ν_ref = 353 GHz.

    Args:
        nu_ghz:     Frequency [GHz]
        beta_d:     Dust spectral index (fiducial ~1.6)
        T_d:        Dust temperature [K] (fiducial ~19.6 K)
        nu_ref_ghz: Reference frequency [GHz]

    Returns:
        Dimensionless SED scaling in CMB thermodynamic units.
    """
    nu_hz = nu_ghz * 1e9
    nu_ref_hz = nu_ref_ghz * 1e9

    # Modified blackbody ratio (Planck function ratio in RJ regime generalization)
    mbb_ratio = (nu_ghz / nu_ref_ghz) ** (beta_d + 1.0)
    planck_ratio = (jnp.exp(H_PLANCK * nu_ref_hz / (K_BOLTZMANN * T_d)) - 1.0) / \
                   (jnp.exp(H_PLANCK * nu_hz   / (K_BOLTZMANN * T_d)) - 1.0)

    # Unit conversion: SED was computed in RJ-like units, convert to CMB thermo
    unit_ratio = rj_to_cmb(nu_ref_ghz) / rj_to_cmb(nu_ghz)

    return mbb_ratio * planck_ratio * unit_ratio


def sync_sed(nu_ghz: float, beta_s: float,
             nu_ref_ghz: float = NU_SYNC_REF_GHZ) -> float:
    """Power-law synchrotron SED in CMB thermodynamic units.

    f_s(ν) = (ν/ν_ref)^β_s × rj_to_cmb(ν_ref) / rj_to_cmb(ν)

    Normalized so f_s(ν_ref) = 1. Default ν_ref = 23 GHz.

    Args:
        nu_ghz:     Frequency [GHz]
        beta_s:     Synchrotron spectral index in RJ units (fiducial ~-3.1)
        nu_ref_ghz: Reference frequency [GHz]

    Returns:
        Dimensionless SED scaling in CMB thermodynamic units.
    """
    power_law = (nu_ghz / nu_ref_ghz) ** beta_s
    unit_ratio = rj_to_cmb(nu_ref_ghz) / rj_to_cmb(nu_ghz)
    return power_law * unit_ratio


def sync_sed_curved(nu_ghz: float, beta_s: float, c_s: float,
                    nu_ref_ghz: float = NU_SYNC_REF_GHZ) -> float:
    """Curved power-law synchrotron SED in CMB thermodynamic units.

    f_s(ν) = (ν/ν_ref)^{β_s + c_s ln(ν/ν_ref)} × rj_to_cmb(ν_ref) / rj_to_cmb(ν)

    Following PanEx Eq. (2) with ν_c = ν_ref. Reduces to sync_sed() when c_s = 0.

    Args:
        nu_ghz:     Frequency [GHz]
        beta_s:     Synchrotron spectral index in RJ units (fiducial ~-3.1)
        c_s:        Spectral curvature (ARCADE: ~-0.052)
        nu_ref_ghz: Reference frequency [GHz]
    """
    ln_ratio = jnp.log(nu_ghz / nu_ref_ghz)
    power_law = jnp.exp((beta_s + c_s * ln_ratio) * ln_ratio)
    unit_ratio = rj_to_cmb(nu_ref_ghz) / rj_to_cmb(nu_ghz)
    return power_law * unit_ratio


# ---------------------------------------------------------------------------
# SED log-derivatives for moment expansion
# (Chluba et al. 2017, arXiv:1701.00274)
# ---------------------------------------------------------------------------

def dust_sed_deriv_beta(nu_ghz: float,
                        nu_ref_ghz: float = NU_DUST_REF_GHZ) -> float:
    """∂ ln f_dust / ∂ β_d = ln(ν / ν_ref).

    Exact: β_d enters the MBB only through (ν/ν_ref)^(β_d+1).
    Vanishes at ν = ν_ref.
    """
    return jnp.log(nu_ghz / nu_ref_ghz)


def dust_sed_deriv_T(nu_ghz: float, T_d: float,
                     nu_ref_ghz: float = NU_DUST_REF_GHZ) -> float:
    """∂ ln f_dust / ∂ T_d.

    = [x eˣ/(eˣ-1) − x_ref e^{x_ref}/(e^{x_ref}-1)] / T_d

    where x = hν/(kT_d). Vanishes at ν = ν_ref.
    """
    nu_hz = nu_ghz * 1e9
    nu_ref_hz = nu_ref_ghz * 1e9
    x = H_PLANCK * nu_hz / (K_BOLTZMANN * T_d)
    x_ref = H_PLANCK * nu_ref_hz / (K_BOLTZMANN * T_d)
    term_nu = x * jnp.exp(x) / (jnp.exp(x) - 1.0)
    term_ref = x_ref * jnp.exp(x_ref) / (jnp.exp(x_ref) - 1.0)
    return (term_nu - term_ref) / T_d


def sync_sed_deriv_beta(nu_ghz: float,
                        nu_ref_ghz: float = NU_SYNC_REF_GHZ) -> float:
    """∂ ln f_sync / ∂ β_s = ln(ν / ν_ref).

    Same for both curved and uncurved SEDs. Vanishes at ν = ν_ref.
    """
    return jnp.log(nu_ghz / nu_ref_ghz)


def sync_sed_deriv_c(nu_ghz: float,
                     nu_ref_ghz: float = NU_SYNC_REF_GHZ) -> float:
    """∂ ln f_sync / ∂ c_s = [ln(ν / ν_ref)]².

    Curvature derivative of the curved power-law SED. Vanishes at ν = ν_ref.
    """
    ln_ratio = jnp.log(nu_ghz / nu_ref_ghz)
    return ln_ratio ** 2
