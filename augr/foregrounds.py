"""
foregrounds.py — Foreground models for CMB BB power spectrum forecasting.

Defines a Protocol (structural type) for foreground models and provides
a Gaussian parametric implementation matching the BICEP/Keck BK18 model.

All implementations must be JAX-traceable: no Python control flow on
parameter values, all array ops via jnp.

Parameter interface: flat jnp.ndarray + list[str] for names. JAX cannot
trace through dict lookups, so we use positional indexing internally and
provide the name list for external bookkeeping.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from augr.units import (
    dust_sed, sync_sed, sync_sed_curved,
    dust_sed_deriv_beta, dust_sed_deriv_T,
    sync_sed_deriv_beta, sync_sed_deriv_c,
)


# -----------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------

@runtime_checkable
class ForegroundModel(Protocol):
    """Structural type for foreground models.

    Any class with matching method signatures satisfies this Protocol —
    no inheritance required.
    """

    @property
    def parameter_names(self) -> list[str]:
        """Ordered foreground parameter names matching the flat array."""
        ...

    def cl_bb(self,
              nu_i: float,
              nu_j: float,
              ells: jnp.ndarray,
              params: jnp.ndarray) -> jnp.ndarray:
        """Foreground C_ℓ^BB for frequency pair (ν_i, ν_j).

        Args:
            nu_i:   First frequency [GHz].
            nu_j:   Second frequency [GHz].
            ells:   1-D array of multipoles.
            params: Flat JAX array of parameter values (same order as
                    parameter_names).

        Returns:
            C_ℓ array of shape (n_ells,) in μK² (CMB thermodynamic).
        """
        ...


# -----------------------------------------------------------------------
# Null model (post-component-separation use)
# -----------------------------------------------------------------------

class NullForegroundModel:
    """Zero-contribution foreground model.

    For post-component-separation forecasts where foregrounds have been
    removed by an external pipeline (e.g. NILC) and the leftover residual
    is modelled via a separate additive template on the signal side. This
    class satisfies the ForegroundModel Protocol with an empty parameter
    list and a constant zero BB spectrum, so the multifrequency FG
    machinery cleanly sits out while the rest of SignalModel / Fisher runs
    unchanged.
    """

    _PARAM_NAMES: list[str] = []

    @property
    def parameter_names(self) -> list[str]:
        return []

    def cl_bb(self,
              nu_i: float,
              nu_j: float,
              ells: jnp.ndarray,
              params: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(ells, dtype=jnp.float64)


# -----------------------------------------------------------------------
# Gaussian parametric model (BK18-style)
# -----------------------------------------------------------------------

# Reference multipole for amplitude parameters
ELL_REF = 80.0


class GaussianForegroundModel:
    """Parametric dust + synchrotron + dust-sync correlation model.

    This matches the BICEP/Keck BK15 foreground model (arXiv:1810.05216,
    PRL 121, 221301; data through the 2015 season) as implemented in
    Buza (2019) thesis Sec. 3.3 and 7.1.3.

    Power spectra (all in CMB thermodynamic μK²):

        C_ℓ^dust(ν_i,ν_j) = A_d × f_d(ν_i) × f_d(ν_j) × (ℓ/80)^α_d
                             × Δ_d(ν_i,ν_j)

        C_ℓ^sync(ν_i,ν_j) = A_s × f_s(ν_i) × f_s(ν_j) × (ℓ/80)^α_s

        C_ℓ^{d×s}(ν_i,ν_j) = ε × [√(D_ii × S_jj) + √(D_jj × S_ii)] / 2

    where D_ii = C_ℓ^dust(ν_i,ν_i) and S_jj = C_ℓ^sync(ν_j,ν_j) are the
    single-frequency auto-spectra. The symmetrized form ensures the cross
    term is symmetric under i↔j and reduces to ε√(DS) when ν_i = ν_j.

    Frequency decorrelation:
        Δ_d(ν_i,ν_j) = exp(-Δ_dust × |ln(ν_i/ν_j)|)
    Applied only to the dust term. Δ_dust = 0 means perfect correlation.

    Parameters (9 total):
        A_dust      Dust amplitude at 353 GHz, ℓ=80 [μK²]
        beta_dust   Dust spectral index (~1.6)
        alpha_dust  Dust ℓ-dependence power law (~-0.58)
        T_dust      Dust temperature [K] (~19.6, often fixed)
        A_sync      Synchrotron amplitude at 23 GHz, ℓ=80 [μK²]
        beta_sync   Synchrotron spectral index in RJ (~-3.1)
        alpha_sync  Synchrotron ℓ-dependence power law (~-0.6)
        epsilon     Dust-synchrotron correlation coefficient [-1, 1]
        Delta_dust  Dust frequency decorrelation strength (≥0)
    """

    _PARAM_NAMES: list[str] = [
        "A_dust", "beta_dust", "alpha_dust", "T_dust",
        "A_sync", "beta_sync", "alpha_sync",
        "epsilon", "Delta_dust",
    ]

    @property
    def parameter_names(self) -> list[str]:
        return list(self._PARAM_NAMES)

    def cl_bb(self,
              nu_i: float,
              nu_j: float,
              ells: jnp.ndarray,
              params: jnp.ndarray) -> jnp.ndarray:
        """Total foreground C_ℓ^BB = dust + sync + dust×sync cross.

        Args:
            nu_i:   First frequency [GHz].
            nu_j:   Second frequency [GHz].
            ells:   1-D array of multipoles (> 0).
            params: Flat array [A_d, β_d, α_d, T_d, A_s, β_s, α_s, ε, Δ_d].

        Returns:
            C_ℓ^BB_fg array, shape (n_ells,), in μK².
        """
        A_d = params[0]
        beta_d = params[1]
        alpha_d = params[2]
        T_d = params[3]
        A_s = params[4]
        beta_s = params[5]
        alpha_s = params[6]
        epsilon = params[7]
        Delta_d = params[8]

        ell_ratio = ells / ELL_REF  # shape (n_ells,)

        # Convert from D_ell to C_ell convention.
        # A_dust/A_sync are bandpower amplitudes: D_ell = ell(ell+1)C_ell/(2pi).
        # CMB spectra use C_ell, so we divide by ell(ell+1)/(2pi).
        dl_to_cl = 2.0 * jnp.pi / (ells * (ells + 1.0))

        # --- Dust ---
        fd_i = dust_sed(nu_i, beta_d, T_d)
        fd_j = dust_sed(nu_j, beta_d, T_d)
        dust_ij = A_d * fd_i * fd_j * ell_ratio ** alpha_d * dl_to_cl
        # Frequency decorrelation (only for cross-frequency)
        decor = jnp.exp(-Delta_d * jnp.abs(jnp.log(nu_i / nu_j)))
        dust_ij = dust_ij * decor

        # --- Synchrotron ---
        fs_i = sync_sed(nu_i, beta_s)
        fs_j = sync_sed(nu_j, beta_s)
        sync_ij = A_s * fs_i * fs_j * ell_ratio ** alpha_s * dl_to_cl

        # --- Dust-sync cross-correlation (symmetrized) ---
        # Auto-spectra needed: dust at each freq with itself, sync at each freq
        # with itself (no decorrelation for auto = Δ(ν,ν) = 1).
        dust_ii = A_d * fd_i * fd_i * ell_ratio ** alpha_d * dl_to_cl
        dust_jj = A_d * fd_j * fd_j * ell_ratio ** alpha_d * dl_to_cl
        sync_ii = A_s * fs_i * fs_i * ell_ratio ** alpha_s * dl_to_cl
        sync_jj = A_s * fs_j * fs_j * ell_ratio ** alpha_s * dl_to_cl

        # Symmetrized: average of "dust at i × sync at j" and "dust at j × sync at i"
        cross = epsilon * (jnp.sqrt(dust_ii * sync_jj)
                           + jnp.sqrt(dust_jj * sync_ii)) / 2.0

        return dust_ij + sync_ij + cross


# -----------------------------------------------------------------------
# Moment expansion helpers (module-level for testability)
# -----------------------------------------------------------------------

def _dust_moment_factor(nu_i: float, nu_j: float, T_d: float,
                        om_beta: float, om_T: float,
                        om_betaT: float) -> float:
    """Moment correction factor for dust cross-frequency spectrum.

    Returns 1 + Σ g_a(ν_i) g_b(ν_j) ω_ab, where g are SED log-derivatives
    and ω are moment amplitudes proportional to the variance/covariance of
    spectral parameters across the sky.
    """
    gb_i = dust_sed_deriv_beta(nu_i)
    gb_j = dust_sed_deriv_beta(nu_j)
    gT_i = dust_sed_deriv_T(nu_i, T_d)
    gT_j = dust_sed_deriv_T(nu_j, T_d)
    return (1.0
            + gb_i * gb_j * om_beta
            + gT_i * gT_j * om_T
            + (gb_i * gT_j + gT_i * gb_j) * om_betaT)


def _sync_moment_factor(nu_i: float, nu_j: float,
                        om_beta: float, om_c: float,
                        om_betac: float) -> float:
    """Moment correction factor for synchrotron cross-frequency spectrum."""
    gb_i = sync_sed_deriv_beta(nu_i)
    gb_j = sync_sed_deriv_beta(nu_j)
    gc_i = sync_sed_deriv_c(nu_i)
    gc_j = sync_sed_deriv_c(nu_j)
    return (1.0
            + gb_i * gb_j * om_beta
            + gc_i * gc_j * om_c
            + (gb_i * gc_j + gc_i * gb_j) * om_betac)


# -----------------------------------------------------------------------
# Moment expansion model
# -----------------------------------------------------------------------

class MomentExpansionModel:
    """Foreground model with moment expansion for spatially varying SEDs.

    Extends the BK18-style Gaussian model with:

    1. **Moment expansion** (Chluba et al. 2017, Azzoni et al. 2021):
       parameterizes the bandpower-level effect of spatially varying
       spectral parameters (β_d, T_d, β_s) through second-order moment
       amplitudes ω. These produce frequency decorrelation and capture
       the leading non-Gaussian effects for large-f_sky surveys.

    2. **Synchrotron spectral curvature** c_s (PanEx s7 model):
       S_ν ∝ (ν/ν_ref)^{β_s + c_s ln(ν/ν_ref)}

    3. **Synchrotron frequency decorrelation** Δ_sync, analogous to Δ_dust.

    The cross-frequency dust BB spectrum is:

        C_ℓ^dust(ν_i,ν_j) = A_d f_d(ν_i) f_d(ν_j) (ℓ/80)^α_d [2π/ℓ(ℓ+1)]
            × exp(-Δ_d |ln(ν_i/ν_j)|)
            × {1 + g_β(ν_i) g_β(ν_j) ω_{d,β}
                 + g_T(ν_i) g_T(ν_j) ω_{d,T}
                 + [g_β(ν_i) g_T(ν_j) + g_T(ν_i) g_β(ν_j)] ω_{d,βT}}

    where g_β(ν) = ∂ ln f_d/∂β_d = ln(ν/ν_ref) and g_T(ν) = ∂ ln f_d/∂T_d.
    Analogous moment terms apply to synchrotron.

    When all new parameters (indices 9–16) are zero, reduces exactly to
    GaussianForegroundModel.

    Parameters (17 total):
        [0–8]   Same as GaussianForegroundModel
        c_sync          Synchrotron spectral curvature
        Delta_sync      Synchrotron frequency decorrelation
        omega_d_beta    Dust β_d moment (∝ Var(β_d))
        omega_d_T       Dust T_d moment (∝ Var(T_d))
        omega_d_betaT   Dust β_d–T_d cross moment
        omega_s_beta    Sync β_s moment (∝ Var(β_s))
        omega_s_c       Sync c_s moment (∝ Var(c_s))
        omega_s_betac   Sync β_s–c_s cross moment
    """

    _PARAM_NAMES: list[str] = [
        "A_dust", "beta_dust", "alpha_dust", "T_dust",
        "A_sync", "beta_sync", "alpha_sync",
        "epsilon", "Delta_dust",
        # --- new parameters ---
        "c_sync", "Delta_sync",
        "omega_d_beta", "omega_d_T", "omega_d_betaT",
        "omega_s_beta", "omega_s_c", "omega_s_betac",
    ]

    @property
    def parameter_names(self) -> list[str]:
        return list(self._PARAM_NAMES)

    def cl_bb(self,
              nu_i: float,
              nu_j: float,
              ells: jnp.ndarray,
              params: jnp.ndarray) -> jnp.ndarray:
        """Total foreground C_ℓ^BB with moment expansion corrections.

        Args:
            nu_i:   First frequency [GHz].
            nu_j:   Second frequency [GHz].
            ells:   1-D array of multipoles (> 0).
            params: Flat array of 17 parameter values.

        Returns:
            C_ℓ^BB_fg array, shape (n_ells,), in μK².
        """
        A_d       = params[0]
        beta_d    = params[1]
        alpha_d   = params[2]
        T_d       = params[3]
        A_s       = params[4]
        beta_s    = params[5]
        alpha_s   = params[6]
        epsilon   = params[7]
        Delta_d   = params[8]
        c_s       = params[9]
        Delta_s   = params[10]
        om_d_b    = params[11]
        om_d_T    = params[12]
        om_d_bT   = params[13]
        om_s_b    = params[14]
        om_s_c    = params[15]
        om_s_bc   = params[16]

        ell_ratio = ells / ELL_REF
        dl_to_cl = 2.0 * jnp.pi / (ells * (ells + 1.0))

        # --- Dust (moment-corrected) ---
        fd_i = dust_sed(nu_i, beta_d, T_d)
        fd_j = dust_sed(nu_j, beta_d, T_d)
        decor_d = jnp.exp(-Delta_d * jnp.abs(jnp.log(nu_i / nu_j)))
        moment_d = _dust_moment_factor(nu_i, nu_j, T_d, om_d_b, om_d_T, om_d_bT)
        dust_ij = (A_d * fd_i * fd_j * ell_ratio ** alpha_d
                   * dl_to_cl * decor_d * moment_d)

        # --- Synchrotron (curved SED + moment-corrected) ---
        fs_i = sync_sed_curved(nu_i, beta_s, c_s)
        fs_j = sync_sed_curved(nu_j, beta_s, c_s)
        decor_s = jnp.exp(-Delta_s * jnp.abs(jnp.log(nu_i / nu_j)))
        moment_s = _sync_moment_factor(nu_i, nu_j, om_s_b, om_s_c, om_s_bc)
        sync_ij = (A_s * fs_i * fs_j * ell_ratio ** alpha_s
                   * dl_to_cl * decor_s * moment_s)

        # --- Dust-sync cross-correlation (symmetrized) ---
        # The cross term ε√(D_ii × S_jj) uses SINGLE-frequency auto-spectra:
        # no cross-frequency decorrelation (exp(-Δ|ln ν_i/ν_j|) = 1 when i=j)
        # but moment corrections still apply at each frequency.
        moment_d_ii = _dust_moment_factor(nu_i, nu_i, T_d, om_d_b, om_d_T, om_d_bT)
        moment_d_jj = _dust_moment_factor(nu_j, nu_j, T_d, om_d_b, om_d_T, om_d_bT)
        moment_s_ii = _sync_moment_factor(nu_i, nu_i, om_s_b, om_s_c, om_s_bc)
        moment_s_jj = _sync_moment_factor(nu_j, nu_j, om_s_b, om_s_c, om_s_bc)

        dust_ii = A_d * fd_i ** 2 * ell_ratio ** alpha_d * dl_to_cl * moment_d_ii
        dust_jj = A_d * fd_j ** 2 * ell_ratio ** alpha_d * dl_to_cl * moment_d_jj
        sync_ii = A_s * fs_i ** 2 * ell_ratio ** alpha_s * dl_to_cl * moment_s_ii
        sync_jj = A_s * fs_j ** 2 * ell_ratio ** alpha_s * dl_to_cl * moment_s_jj

        cross = epsilon * (jnp.sqrt(dust_ii * sync_jj)
                           + jnp.sqrt(dust_jj * sync_ii)) / 2.0

        return dust_ij + sync_ij + cross
