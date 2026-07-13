"""
loading.py — Galactic-foreground optical loading for the photon-noise NET.

The ``telescope.photon_noise_net`` / ``photon_noise_net_jax`` photon-noise
model forms the sky-side occupation as CMB + telescope graybody only. Diffuse
Galactic emission (thermal dust + synchrotron) is an additional sky-side
loading term that this module supplies as an ``extra_loading`` callable.

Physics
-------
The ``extra_loading`` hook must return a photon **occupation number** in the
same convention as ``n_cmb = 1/(exp(hν/kT_CMB) − 1)``. From the blackbody
identity ``B_ν(T) = 2hν³/c² · n_BE`` (see ``units.occupation_from_intensity``),
a sky component of total specific intensity ``I_ν`` contributes occupation
``n(ν) = I_ν · c²/(2hν³)``. Substituting the standard component intensities
collapses the conversion to closed forms that reuse only the SED parameters
plus one absolute normalization each:

- **Dust** — Planck GNILC modified blackbody
  ``I_ν = τ_353 · (ν/ν_ref)^β_d · B_ν(T_d)`` gives
  ``n_dust(ν) = τ_353 · (ν/ν_ref)^β_d / (exp(hν/kT_d) − 1)``.
- **Synchrotron** — power-law Rayleigh-Jeans brightness
  ``T_s(ν) = T_ref · (ν/ν_ref)^β_s`` gives
  ``n_sync(ν) = (k·T_ref/(h·ν)) · (ν/ν_ref)^β_s``.

Both are pure ``jnp`` functions of the band-integration frequency grid with
constant sky parameters, so a single callable is jnp-traceable through
``photon_noise_net_jax`` for the autodiff design forward.

Normalization
-------------
The absolute brightnesses are **total-intensity (Stokes-I)** quantities,
distinct from (and ~10–20× larger than) the *polarized* ``A_dust`` / ``A_sync``
amplitudes that set the C_ℓ foreground model. Their calibrated region-mean
values live in ``config.GALACTIC_LOADING`` (sourced from the Planck GNILC dust
model and Planck Commander synchrotron products over the science mask; see
``scripts/calibrate_galactic_loading.py``). This module carries only the
functional form — no data, no map I/O.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from augr.units import H_PLANCK, K_BOLTZMANN, NU_DUST_REF_GHZ


def dust_loading_occupation(
    nu_hz: jnp.ndarray,
    tau_353: float,
    beta_d: float,
    T_d: float,
    nu_ref_ghz: float = NU_DUST_REF_GHZ,
) -> jnp.ndarray:
    """Thermal-dust photon occupation number (modified blackbody).

    n_dust(ν) = τ_353 · (ν/ν_ref)^β_d / (exp(hν/kT_d) − 1)

    Args:
        nu_hz:      Frequency grid [Hz].
        tau_353:    Dust optical depth at ``nu_ref_ghz`` (dimensionless;
                    the Planck GNILC "Opacity" quantity for ν_ref = 353 GHz).
        beta_d:     Dust emissivity spectral index (total-intensity value).
        T_d:        Dust temperature [K].
        nu_ref_ghz: Reference frequency for τ [GHz]. Default 353 GHz.

    Returns:
        Dimensionless occupation number on ``nu_hz``.
    """
    nu_ref_hz = nu_ref_ghz * 1e9
    mbb = tau_353 * (nu_hz / nu_ref_hz) ** beta_d
    occ = 1.0 / (jnp.exp(H_PLANCK * nu_hz / (K_BOLTZMANN * T_d)) - 1.0)
    return mbb * occ


def sync_loading_occupation(
    nu_hz: jnp.ndarray,
    T_ref_K: float,
    beta_s: float,
    nu_ref_ghz: float,
) -> jnp.ndarray:
    """Synchrotron photon occupation number (power-law RJ brightness).

    n_sync(ν) = (k·T_ref / (h·ν)) · (ν/ν_ref)^β_s

    ``T_ref_K`` is the Rayleigh-Jeans brightness temperature at
    ``nu_ref_ghz``; ``β_s`` is the RJ power-law index (≈ −3.1). In-band this
    term is negligible relative to dust and the CMB, but it is included for a
    complete, symmetric Galactic-loading model.

    Args:
        nu_hz:      Frequency grid [Hz].
        T_ref_K:    Synchrotron RJ brightness temperature at ``nu_ref_ghz`` [K].
        beta_s:     Synchrotron RJ spectral index.
        nu_ref_ghz: Reference frequency [GHz].

    Returns:
        Dimensionless occupation number on ``nu_hz``.
    """
    nu_ref_hz = nu_ref_ghz * 1e9
    T_rj = T_ref_K * (nu_hz / nu_ref_hz) ** beta_s
    return K_BOLTZMANN * T_rj / (H_PLANCK * nu_hz)


def make_galactic_extra_loading(
    *,
    tau_353: float,
    beta_d: float,
    T_d: float,
    T_sync_ref_K: float,
    beta_s: float,
    sync_nu_ref_ghz: float,
    dust_nu_ref_ghz: float = NU_DUST_REF_GHZ,
    include_dust: bool = True,
    include_sync: bool = True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an ``extra_loading`` callable for Galactic dust + synchrotron.

    Returns ``n_extra(nu_hz) -> n_dust + n_sync`` suitable for
    ``BandSpec.extra_loading``. The callable is a pure ``jnp`` function of the
    band grid (constant sky parameters), so it is jnp-traceable through
    ``photon_noise_net_jax`` and safe to share across every band of a design.

    Args:
        tau_353:         Dust optical depth at ``dust_nu_ref_ghz``.
        beta_d:          Dust spectral index (total intensity).
        T_d:             Dust temperature [K].
        T_sync_ref_K:    Synchrotron RJ brightness temperature at
                         ``sync_nu_ref_ghz`` [K].
        beta_s:          Synchrotron RJ spectral index.
        sync_nu_ref_ghz: Synchrotron reference frequency [GHz].
        dust_nu_ref_ghz: Dust reference frequency [GHz]. Default 353 GHz.
        include_dust:    Include the dust term (default True).
        include_sync:    Include the synchrotron term (default True).

    Returns:
        Callable mapping a frequency grid [Hz] to the added occupation number.
    """

    def n_extra(nu_hz: jnp.ndarray) -> jnp.ndarray:
        nu_hz = jnp.asarray(nu_hz)
        total = jnp.zeros_like(nu_hz)
        if include_dust:
            total = total + dust_loading_occupation(nu_hz, tau_353, beta_d, T_d, dust_nu_ref_ghz)
        if include_sync:
            total = total + sync_loading_occupation(nu_hz, T_sync_ref_K, beta_s, sync_nu_ref_ghz)
        return total

    return n_extra
