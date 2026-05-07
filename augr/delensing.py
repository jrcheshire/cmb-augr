"""
delensing.py — Self-consistent QE delensing for Fisher forecasting.

Computes the quadratic-estimator (QE) lensing reconstruction noise N_0
for all five estimators (TT, TE, TB, EE, EB) plus a minimum-variance
combination, the residual lensing B-mode power spectrum after delensing,
and an iterative procedure that self-consistently updates both.

Two implementations:
  - **Flat-sky** (default): 2-D Fourier integrals via Gauss-Legendre
    quadrature over the azimuthal angle φ. Follows Hu & Okamoto (2002)
    Table 1 weight functions directly. Accurate to ~1% vs CAMB for ℓ > 5.
  - **Full-sky**: Wigner 3j coupling via Schulten-Gordon recursion
    (augr/wigner.py). Uses Smith et al. (2012) Eq. 6-7 coupling with
    the cyclic 3j identity for efficient vectorized computation. Correct
    at low ℓ where flat-sky breaks down; the first-order gradient
    approximation limits accuracy to ~2% at ℓ ~ 50.

The residual BB formula C_l^{BB,res} = K @ C_φφ^{res} approximates
W_EE ≈ 1 (perfect E-mode Wiener filter), valid for space missions
where C_EE >> N_EE at the relevant multipoles. See residual_cl_bb()
for details.

Key references:
  - Hu & Okamoto (2002), astro-ph/0111606 — flat-sky QE formalism
  - Okamoto & Hu (2003), astro-ph/0301031 — full-sky QE formalism
  - Smith et al. (2012), 1010.0048 — residual BB, iterative delensing
  - Trendafilova, Hotinli & Meyers (2024), JCAP 06, 017; arXiv:2312.02954
    — CLASS_delens iterative procedure

All spectra in C_ell convention [μK²] for CMB, dimensionless for φφ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax

# Data file locations
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_UNLENSED_FILE = os.path.join(_DATA_DIR, "camb_unlensed_cls.dat")
DEFAULT_LENSED_FILE = os.path.join(_DATA_DIR, "camb_lensed_cls.dat")
DEFAULT_CLPP_FILE = os.path.join(_DATA_DIR, "camb_clpp.dat")


# -----------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class LensingSpectra:
    """CMB and lensing potential spectra needed for QE delensing.

    All arrays are indexed by multipole: array[ell] gives the value at
    that ell, from ell=0 to ell=ell_max. ell=0,1 entries are zero.

    CMB spectra in μK² (C_ell convention). φφ is dimensionless.

    Attributes:
        ells:       1-D array of multipole values [0, ..., ell_max].
        cl_tt_unl:  Unlensed TT C_ell [μK²].
        cl_ee_unl:  Unlensed EE C_ell [μK²].
        cl_bb_unl:  Unlensed BB C_ell [μK²] (zero for scalar-only).
        cl_te_unl:  Unlensed TE C_ell [μK²].
        cl_tt_len:  Lensed TT C_ell [μK²].
        cl_ee_len:  Lensed EE C_ell [μK²].
        cl_bb_len:  Lensed BB C_ell [μK²].
        cl_te_len:  Lensed TE C_ell [μK²].
        cl_pp:      Lensing potential C_L^{φφ} [dimensionless].
    """
    ells: jnp.ndarray
    cl_tt_unl: jnp.ndarray
    cl_ee_unl: jnp.ndarray
    cl_bb_unl: jnp.ndarray
    cl_te_unl: jnp.ndarray
    cl_tt_len: jnp.ndarray
    cl_ee_len: jnp.ndarray
    cl_bb_len: jnp.ndarray
    cl_te_len: jnp.ndarray
    cl_pp: jnp.ndarray

    @property
    def ell_max(self) -> int:
        return int(self.ells[-1])


def load_lensing_spectra(
    unlensed_file: str = DEFAULT_UNLENSED_FILE,
    lensed_file: str = DEFAULT_LENSED_FILE,
    clpp_file: str = DEFAULT_CLPP_FILE,
) -> LensingSpectra:
    """Load CMB and lensing potential spectra from CAMB output files.

    Files must have columns: ell, TT, EE, BB, TE for CMB; L, C_L^{φφ} for φφ.
    All in C_ell convention (not D_ell).
    """
    unl = np.loadtxt(unlensed_file, comments="#")
    lns = np.loadtxt(lensed_file, comments="#")
    cpp = np.loadtxt(clpp_file, comments="#")

    # Ensure all files cover the same ell range
    ell_max = min(int(unl[-1, 0]), int(lns[-1, 0]), int(cpp[-1, 0]))

    # Slice to common range (files start at ell=0)
    unl = unl[:ell_max + 1]
    lns = lns[:ell_max + 1]
    cpp = cpp[:ell_max + 1]

    ells = jnp.arange(ell_max + 1, dtype=float)

    return LensingSpectra(
        ells=ells,
        cl_tt_unl=jnp.array(unl[:, 1]),
        cl_ee_unl=jnp.array(unl[:, 2]),
        cl_bb_unl=jnp.array(unl[:, 3]),
        cl_te_unl=jnp.array(unl[:, 4]),
        cl_tt_len=jnp.array(lns[:, 1]),
        cl_ee_len=jnp.array(lns[:, 2]),
        cl_bb_len=jnp.array(lns[:, 3]),
        cl_te_len=jnp.array(lns[:, 4]),
        cl_pp=jnp.array(cpp[:, 1]),
    )


# -----------------------------------------------------------------------
# Quadrature and geometry helpers
# -----------------------------------------------------------------------

def _gl_nodes(n_phi: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Legendre nodes and weights for the azimuthal φ integral.

    The QE N_0 integral is over d²l₁ = l₁ dl₁ dφ / (2π)².
    We handle the φ integral via GL quadrature on [0, 2π].

    Returns:
        (phi_nodes, phi_weights): arrays of shape (n_phi,).
        Weights sum to 2π.
    """
    # GL nodes on [-1, 1], then map to [0, 2π]
    nodes, weights = np.polynomial.legendre.leggauss(n_phi)
    phi = jnp.array((nodes + 1.0) * jnp.pi)      # [0, 2π]
    w = jnp.array(weights * jnp.pi)                # weights sum to 2π
    return phi, w


def _triangle_geometry(L: jnp.ndarray, l1: float,
                       phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute |l₂| and cos(2φ_{l₁,L}) for the triangle l₂ = L - l₁.

    In the flat-sky convention, we place L along the x-axis. Then:
        l₁ = l₁ (cos φ, sin φ)
        l₂ = L - l₁ = (L - l₁ cos φ, -l₁ sin φ)

    The angle φ_{l₁,L} between l₁ and L is just φ (our integration variable).

    Args:
        L:   1-D array of reconstruction multipoles, shape (n_L,).
        l1:  Scalar, the magnitude |l₁|.
        phi: 1-D array of azimuthal angles, shape (n_phi,).

    Returns:
        l2:        |l₂|, shape (n_L, n_phi).
        cos2phi12: cos(2(φ_{l₁} - φ_{l₂})), shape (n_L, n_phi).
                   This is the geometric factor for spin-2 coupling.
    """
    # Shape: (n_L, 1) and (1, n_phi) for broadcasting
    L_ = L[:, None]
    phi_ = phi[None, :]

    l2x = L_ - l1 * jnp.cos(phi_)
    l2y = -l1 * jnp.sin(phi_)
    l2 = jnp.sqrt(l2x**2 + l2y**2)

    # Angle of l₁ w.r.t. x-axis is φ; angle of l₂ is atan2(l2y, l2x)
    # φ_{l₁} - φ_{l₂} is the relative angle
    phi_l1 = phi_                           # angle of l₁
    phi_l2 = jnp.arctan2(l2y, l2x)         # angle of l₂
    dphi = phi_l1 - phi_l2
    cos2phi12 = jnp.cos(2.0 * dphi)
    sin2phi12 = jnp.sin(2.0 * dphi)

    return l2, cos2phi12, sin2phi12


def _interp_at(cl: jnp.ndarray, l_vals: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a spectrum cl (indexed by integer ell) at arbitrary l_vals.

    Uses linear interpolation. Values outside [0, len(cl)-1] are zeroed.
    """
    ells = jnp.arange(len(cl), dtype=float)
    return jnp.interp(l_vals, ells, cl, left=0.0, right=0.0)


# -----------------------------------------------------------------------
# QE reconstruction noise N_0
# -----------------------------------------------------------------------

def compute_n0_eb(Ls: jnp.ndarray,
                  spectra: LensingSpectra,
                  nl_ee: jnp.ndarray,
                  nl_bb: jnp.ndarray,
                  l_min: int = 2,
                  l_max: int = 3000,
                  n_phi: int = 128,
                  fullsky: bool = False) -> jnp.ndarray:
    """N_0^{EB}(L) — QE reconstruction noise for the EB estimator.

    Hu & Okamoto (2002) Eq. 11 with Table 1 weight functions.

    The response uses **unlensed** C_EE; the filter denominator uses
    **total** (lensed + noise) C_EE and C_BB.

    Args:
        Ls:      1-D array of reconstruction multipoles L.
        spectra: LensingSpectra with unlensed/lensed CMB spectra.
        nl_ee:   Noise N_ℓ^EE array indexed by ell (same length as spectra.ells).
        nl_bb:   Noise N_ℓ^BB array indexed by ell.
        l_min:   Minimum ell for internal sum over l₁.
        l_max:   Maximum ell for internal sum.
        n_phi:   Number of GL quadrature nodes for φ integral (flat-sky only).
        fullsky: Use full-sky Wigner 3j coupling (default False = flat-sky).

    Returns:
        N_0^{EB}(L), array of shape (n_L,).
    """
    if fullsky:
        return _compute_n0_eb_fullsky(Ls, spectra, nl_ee, nl_bb, l_min, l_max)
    phi, w_phi = _gl_nodes(n_phi)

    # Total observed spectra (lensed + noise)
    cl_ee_tot = spectra.cl_ee_len + nl_ee
    cl_bb_tot = spectra.cl_bb_len + nl_bb

    # Response uses unlensed C_EE
    cl_ee_unl = spectra.cl_ee_unl

    l1_vals = jnp.arange(l_min, l_max + 1, dtype=float)

    def scan_fn(integral_acc, l1):
        """Accumulate the N_0 integrand over l₁ values."""
        l2, _cos2phi12, sin2phi12 = _triangle_geometry(Ls, l1, phi)

        # Response: f_EB = C_{l₁}^{EE,unl} × (L · l₁) × sin(2φ_{l₁,l₂})
        # L · l₁ = L * l₁ * cos(φ), where φ is the angle between L and l₁
        Ldotl1 = Ls[:, None] * l1 * jnp.cos(phi[None, :])  # (n_L, n_phi)
        f_eb = _interp_at(cl_ee_unl, l1) * Ldotl1 * sin2phi12

        # Filter: F_EB = f_EB / (C_{l₁}^{EE,tot} × C_{l₂}^{BB,tot})
        denom = _interp_at(cl_ee_tot, l1) * _interp_at(cl_bb_tot, l2)
        # Avoid division by zero for out-of-range l₂
        safe_denom = jnp.where(denom > 0, denom, 1.0)
        F_eb = f_eb / safe_denom
        F_eb = jnp.where(denom > 0, F_eb, 0.0)

        # Integrand: f × F × l₁ / (2π)²  weighted by φ quadrature
        # d²l₁ = l₁ dl₁ dφ, so integrand per dl₁ is l₁ × ∫dφ (f×F) / (2π)²
        integrand_phi = f_eb * F_eb * w_phi[None, :]  # (n_L, n_phi)
        integral_l1 = jnp.sum(integrand_phi, axis=1)  # (n_L,)

        # Multiply by l₁ / (2π)² (the dl₁ is implicit: step size = 1)
        contribution = integral_l1 * l1 / (2.0 * jnp.pi)**2
        return integral_acc + contribution, None

    total_integral, _ = lax.scan(scan_fn, jnp.zeros_like(Ls), l1_vals)

    # N_0^{φφ} = 1 / integral.
    # HO02 Eq. 11 gives A(L) = L²/∫ for the deflection field d = ∇φ.
    # Since C_L^{dd} = L² C_L^{φφ}, we need N_0^{φφ} = A/L² = 1/integral.
    return jnp.where(total_integral > 0, 1.0 / total_integral, jnp.inf)


def compute_n0_tb(Ls: jnp.ndarray,
                  spectra: LensingSpectra,
                  nl_tt: jnp.ndarray,
                  nl_bb: jnp.ndarray,
                  l_min: int = 2,
                  l_max: int = 3000,
                  n_phi: int = 128,
                  fullsky: bool = False) -> jnp.ndarray:
    """N_0^{TB}(L) — QE reconstruction noise for the TB estimator.

    Response: f_TB = C_{l₁}^{TE,unl} × (L·l₁) × sin(2φ_{l₁,l₂})
    Filter:   F_TB = f_TB / (C_{l₁}^{TT,tot} × C_{l₂}^{BB,tot})
    """
    if fullsky:
        return _compute_n0_tb_fullsky(Ls, spectra, nl_tt, nl_bb, l_min, l_max)
    phi, w_phi = _gl_nodes(n_phi)
    cl_tt_tot = spectra.cl_tt_len + nl_tt
    cl_bb_tot = spectra.cl_bb_len + nl_bb
    cl_te_unl = spectra.cl_te_unl
    l1_vals = jnp.arange(l_min, l_max + 1, dtype=float)

    def scan_fn(acc, l1):
        l2, _cos2phi12, sin2phi12 = _triangle_geometry(Ls, l1, phi)
        Ldotl1 = Ls[:, None] * l1 * jnp.cos(phi[None, :])
        f = _interp_at(cl_te_unl, l1) * Ldotl1 * sin2phi12
        denom = _interp_at(cl_tt_tot, l1) * _interp_at(cl_bb_tot, l2)
        safe_denom = jnp.where(denom > 0, denom, 1.0)
        F = jnp.where(denom > 0, f / safe_denom, 0.0)
        contrib = jnp.sum(f * F * w_phi[None, :], axis=1) * l1 / (2 * jnp.pi)**2
        return acc + contrib, None

    total, _ = lax.scan(scan_fn, jnp.zeros_like(Ls), l1_vals)
    return jnp.where(total > 0, 1.0 / total, jnp.inf)


def compute_n0_tt(Ls: jnp.ndarray,
                  spectra: LensingSpectra,
                  nl_tt: jnp.ndarray,
                  l_min: int = 2,
                  l_max: int = 3000,
                  n_phi: int = 128,
                  fullsky: bool = False) -> jnp.ndarray:
    """N_0^{TT}(L) — QE reconstruction noise for the TT estimator.

    Response: f_TT = C_{l₁}^{TT,unl} (L·l₁) + C_{l₂}^{TT,unl} (L·l₂)
    Filter:   F_TT = f_TT / (2 × C_{l₁}^{TT,tot} × C_{l₂}^{TT,tot})
    Factor of 2 from same-field estimator.
    """
    if fullsky:
        return _compute_n0_tt_fullsky(Ls, spectra, nl_tt, l_min, l_max)
    phi, w_phi = _gl_nodes(n_phi)
    cl_tt_tot = spectra.cl_tt_len + nl_tt
    cl_tt_unl = spectra.cl_tt_unl
    l1_vals = jnp.arange(l_min, l_max + 1, dtype=float)

    def scan_fn(acc, l1):
        l2, _cos2phi12, _sin2phi12 = _triangle_geometry(Ls, l1, phi)
        Ldotl1 = Ls[:, None] * l1 * jnp.cos(phi[None, :])
        # L · l₂: need cos of angle between L and l₂
        # l₂ = (L - l₁ cos φ, -l₁ sin φ), L = (L, 0)
        # L · l₂ = L × (L - l₁ cos φ)
        Ldotl2 = Ls[:, None] * (Ls[:, None] - l1 * jnp.cos(phi[None, :]))

        f = (_interp_at(cl_tt_unl, l1) * Ldotl1
             + _interp_at(cl_tt_unl, l2) * Ldotl2)
        denom = 2.0 * _interp_at(cl_tt_tot, l1) * _interp_at(cl_tt_tot, l2)
        safe_denom = jnp.where(denom > 0, denom, 1.0)
        F = jnp.where(denom > 0, f / safe_denom, 0.0)
        contrib = jnp.sum(f * F * w_phi[None, :], axis=1) * l1 / (2 * jnp.pi)**2
        return acc + contrib, None

    total, _ = lax.scan(scan_fn, jnp.zeros_like(Ls), l1_vals)
    return jnp.where(total > 0, 1.0 / total, jnp.inf)


def compute_n0_ee(Ls: jnp.ndarray,
                  spectra: LensingSpectra,
                  nl_ee: jnp.ndarray,
                  l_min: int = 2,
                  l_max: int = 3000,
                  n_phi: int = 128,
                  fullsky: bool = False) -> jnp.ndarray:
    """N_0^{EE}(L) — QE reconstruction noise for the EE estimator.

    Response: f_EE = [C_{l₁}^{EE,unl} (L·l₁) + C_{l₂}^{EE,unl} (L·l₂)] cos(2φ_{l₁,l₂})
    Filter:   F_EE = f_EE / (2 × C_{l₁}^{EE,tot} × C_{l₂}^{EE,tot})
    """
    if fullsky:
        return _compute_n0_ee_fullsky(Ls, spectra, nl_ee, l_min, l_max)
    phi, w_phi = _gl_nodes(n_phi)
    cl_ee_tot = spectra.cl_ee_len + nl_ee
    cl_ee_unl = spectra.cl_ee_unl
    l1_vals = jnp.arange(l_min, l_max + 1, dtype=float)

    def scan_fn(acc, l1):
        l2, cos2phi12, _sin2phi12 = _triangle_geometry(Ls, l1, phi)
        Ldotl1 = Ls[:, None] * l1 * jnp.cos(phi[None, :])
        Ldotl2 = Ls[:, None] * (Ls[:, None] - l1 * jnp.cos(phi[None, :]))

        f = (_interp_at(cl_ee_unl, l1) * Ldotl1
             + _interp_at(cl_ee_unl, l2) * Ldotl2) * cos2phi12
        denom = 2.0 * _interp_at(cl_ee_tot, l1) * _interp_at(cl_ee_tot, l2)
        safe_denom = jnp.where(denom > 0, denom, 1.0)
        F = jnp.where(denom > 0, f / safe_denom, 0.0)
        contrib = jnp.sum(f * F * w_phi[None, :], axis=1) * l1 / (2 * jnp.pi)**2
        return acc + contrib, None

    total, _ = lax.scan(scan_fn, jnp.zeros_like(Ls), l1_vals)
    return jnp.where(total > 0, 1.0 / total, jnp.inf)


def compute_n0_te(Ls: jnp.ndarray,
                  spectra: LensingSpectra,
                  nl_tt: jnp.ndarray,
                  nl_ee: jnp.ndarray,
                  l_min: int = 2,
                  l_max: int = 3000,
                  n_phi: int = 128,
                  fullsky: bool = False,
                  te_filter: str = 'ho02_diag_approx') -> jnp.ndarray:
    """N_0^{TE}(L) — QE reconstruction noise for the TE estimator.

    Follows Hu & Okamoto 2002 (astro-ph/0111606) Table 1 with α = ΘE:
    the T-field sits at l₁ and the E-field at l₂. The response is
      f_TE(l₁, l₂) = C_TE(l₁) (L·l₁) cos(2φ) + C_TE(l₂) (L·l₂).
    cos(2φ) attaches to the C_TE(l₁) term because the Wick contraction
    that generates it matches Θ(l₁) with Ẽ(-l₁); the E-field's spin-2
    deflection kernel then brings in cos(2φ_{l₁,l₂}) at position l₁.
    (The E-field is still what's being deflected — it's just evaluated
    against the l₁ momentum.)

    The exact filter requires a 2×2 covariance inversion at each
    (l₁, l₂) pair (HO02 Eq. 13). The default ``te_filter`` is augr's
    diagonal approximation with denominator
    ``C_TT(l₁) C_EE(l₂) + C_TE(l₁) C_TE(l₂)``. Unlike the full HO02
    denominator (always positive by Cauchy-Schwarz), this form can
    flip sign at (l₁, l₂) where C_TE(l₁)C_TE(l₂) is negative and
    large — hence the abs() guard in the full-sky variant. Since TE
    contributes ~1-2% to N_0^{MV} at space-experiment noise levels,
    the approximation is adequate for production.

    Parameters
    ----------
    te_filter : {'ho02_diag_approx', 'strict_diagonal'}
        ONLY affects the full-sky path (``fullsky=True``); the flat-sky
        path always uses HO02 Eq. 13's diagonal approximation, which is
        validated against the closed-form constant-Cl test.
        ``'ho02_diag_approx'`` (default) reproduces the production
        filter described above. ``'strict_diagonal'`` drops the
        ``C_TE * C_TE`` cross term, giving ``C_TT(l₁) C_EE(l₂)``; this
        matches plancklens with ``fal['te']=0`` for the apples-to-apples
        N_0 validation harness in ``scripts/n0_validation/``.
    """
    if fullsky:
        return _compute_n0_te_fullsky(Ls, spectra, nl_tt, nl_ee, l_min, l_max,
                                      te_filter=te_filter)
    phi, w_phi = _gl_nodes(n_phi)
    cl_tt_tot = spectra.cl_tt_len + nl_tt
    cl_ee_tot = spectra.cl_ee_len + nl_ee
    cl_te_tot = spectra.cl_te_len  # noise is uncorrelated between T and E
    cl_te_unl = spectra.cl_te_unl
    l1_vals = jnp.arange(l_min, l_max + 1, dtype=float)

    def scan_fn(acc, l1):
        l2, cos2phi12, _sin2phi12 = _triangle_geometry(Ls, l1, phi)
        Ldotl1 = Ls[:, None] * l1 * jnp.cos(phi[None, :])
        Ldotl2 = Ls[:, None] * (Ls[:, None] - l1 * jnp.cos(phi[None, :]))

        # Response
        f = (_interp_at(cl_te_unl, l1) * Ldotl1 * cos2phi12
             + _interp_at(cl_te_unl, l2) * Ldotl2)

        # Diagonal approximation to the TE filter (see docstring).
        denom = (_interp_at(cl_tt_tot, l1) * _interp_at(cl_ee_tot, l2)
                 + _interp_at(cl_te_tot, l1) * _interp_at(cl_te_tot, l2))
        safe_denom = jnp.where(jnp.abs(denom) > 0, denom, 1.0)
        F = jnp.where(jnp.abs(denom) > 0, f / safe_denom, 0.0)
        contrib = jnp.sum(f * F * w_phi[None, :], axis=1) * l1 / (2 * jnp.pi)**2
        return acc + contrib, None

    total, _ = lax.scan(scan_fn, jnp.zeros_like(Ls), l1_vals)
    # TE denominator C_TT*C_EE + C_TE^2 is always non-negative
    return jnp.where(total > 0, 1.0 / total, jnp.inf)


def compute_n0_mv(Ls: jnp.ndarray,
                  spectra: LensingSpectra,
                  nl_tt: jnp.ndarray,
                  nl_ee: jnp.ndarray,
                  nl_bb: jnp.ndarray,
                  l_min: int = 2,
                  l_max: int = 3000,
                  n_phi: int = 128,
                  fullsky: bool = False) -> jnp.ndarray:
    """Minimum-variance combination of all five QE estimators.

    1/N_0^{MV}(L) = Σ_α 1/N_0^α(L)    (HO02 Eq. 22)

    This is the diagonal approximation: it ignores cross-correlations
    between estimators. Parity-even {TT, TE, EE} and parity-odd {EB, TB}
    are exactly uncorrelated; within each sector, cross-correlations are
    a percent-level correction. For low-noise space experiments, EB
    dominates the MV and the approximation has negligible impact.
    """
    n0_tt = compute_n0_tt(Ls, spectra, nl_tt, l_min, l_max, n_phi,
                          fullsky=fullsky)
    n0_ee = compute_n0_ee(Ls, spectra, nl_ee, l_min, l_max, n_phi,
                          fullsky=fullsky)
    n0_te = compute_n0_te(Ls, spectra, nl_tt, nl_ee, l_min, l_max, n_phi,
                          fullsky=fullsky)
    n0_eb = compute_n0_eb(Ls, spectra, nl_ee, nl_bb, l_min, l_max, n_phi,
                          fullsky=fullsky)
    n0_tb = compute_n0_tb(Ls, spectra, nl_tt, nl_bb, l_min, l_max, n_phi,
                          fullsky=fullsky)

    inv_n0_mv = (1.0 / n0_tt + 1.0 / n0_ee + 1.0 / n0_te
                 + 1.0 / n0_eb + 1.0 / n0_tb)
    return 1.0 / inv_n0_mv


# -----------------------------------------------------------------------
# Full-sky N_0 via Wigner 3j coupling
# -----------------------------------------------------------------------

def _compute_n0_eb_fullsky(Ls: jnp.ndarray,
                           spectra: LensingSpectra,
                           nl_ee: jnp.ndarray,
                           nl_bb: jnp.ndarray,
                           l_min: int = 2,
                           l_max: int = 3000) -> jnp.ndarray:
    """Full-sky N_0^{EB}(L) using Smith et al. (2012) Eq. 6-7.

    [N_0(L)]^{-1} = 1/(2L+1) × sum_{l_E, l_B, odd}
        [C_{l_E}^{EE,unl}]^2 × |f^{EB}(l_B, l_E, L)|^2
        / (C_{l_E}^{EE,tot} × C_{l_B}^{BB,tot})

    where the parity-odd coupling is (Smith Eq. 6-7, odd l_E+l_B+L only):
    |f^{EB}|^2 = [-l_B(l_B+1)+l_E(l_E+1)+L(L+1)]^2
                 × (2l_B+1)(2l_E+1)(2L+1)/(16π)
                 × (l_B, l_E, L; 2, -2, 0)^2

    Computed via cyclic 3j identity:
        (l_B, l_E, L; 2, -2, 0) = (l_E, L, l_B; -2, 0, 2)
    """
    from augr.wigner import wigner3j_vectorized

    cl_ee_tot = np.asarray(spectra.cl_ee_len + nl_ee)
    cl_bb_tot = np.asarray(spectra.cl_bb_len + nl_bb)
    cl_ee_unl = np.asarray(spectra.cl_ee_unl)

    Ls_np = np.asarray(Ls)
    l_E_arr = np.arange(l_min, l_max + 1, dtype=float)

    L_samples = _fullsky_L_samples(Ls_np)
    n0_inv_samples = np.zeros(len(L_samples))

    for i_L, L in enumerate(L_samples):
        L = int(L)

        # Cyclic: (l_B, l_E, L; 2,-2,0) = (l_E, L, l_B; -2, 0, 2)
        l_B_grid, w3j = wigner3j_vectorized(L, l_E_arr, m1=-2, m2=0)

        # Smith's three-term geometric factor
        l_E_ll = l_E_arr * (l_E_arr + 1)
        l_B_ll = l_B_grid * (l_B_grid + 1)
        L_LL = L * (L + 1)
        geom = -l_B_ll[None, :] + l_E_ll[:, None] + L_LL

        prefactor = np.sqrt((2 * l_E_arr + 1)[:, None]
                            * (2 * l_B_grid + 1)[None, :]
                            * (2 * L + 1) / (16.0 * np.pi))

        # Parity-odd mask
        parity_sum = (l_E_arr.astype(int)[:, None]
                      + l_B_grid.astype(int)[None, :] + L)
        odd_mask = (parity_sum % 2 == 1).astype(float)

        f_eb_sq = (prefactor * w3j * geom) ** 2 * odd_mask

        # Spectrum weights: C_EE^2 / C_EE_tot (indexed by l_E)
        ee_unl_sq = cl_ee_unl[l_min:l_max + 1] ** 2
        ee_tot = cl_ee_tot[l_min:l_max + 1]
        safe_ee_tot = np.where(ee_tot > 0, ee_tot, 1.0)
        l_E_weight = np.where(ee_tot > 0, ee_unl_sq / safe_ee_tot, 0.0)

        # 1/C_BB (indexed by l_B)
        inv_bb = _fullsky_inv_spectrum(cl_bb_tot, l_B_grid)

        l_B_sum = f_eb_sq @ inv_bb  # (n_l_E,)
        n0_inv_samples[i_L] = np.sum(l_E_weight * l_B_sum) / (2 * L + 1)

    # Interpolate N_0^{-1} to the requested L grid (log-space for smoothness)
    log_n0_inv = np.log(np.maximum(n0_inv_samples, 1e-300))
    n0_inv_interp = np.exp(np.interp(Ls_np, L_samples.astype(float),
                                     log_n0_inv))

    n0_inv_jax = jnp.array(n0_inv_interp)
    return jnp.where(n0_inv_jax > 0, 1.0 / n0_inv_jax, jnp.inf)


def _fullsky_L_samples(Ls_np: np.ndarray) -> np.ndarray:
    """Generate L sample points for the full-sky N_0 evaluation.

    The full-sky path computes the (l1, l2) sum at these sample L values
    and log-interpolates onto the requested ``Ls``. To keep the interp a
    no-op at every requested point, the requested ``Ls`` are *included*
    in the sample grid; an internal log-spaced grid then fills in any
    gaps so monotone interp at intermediate user-queried L values still
    works smoothly.

    A previous version capped ``n_sample`` by ``len(Ls_np)``, which
    silently collapsed the internal grid when the user passed sparse
    Ls (e.g. 7 points) and gave ~10-20% interp error at intermediate L.
    The fix is to (a) drop that cap, (b) always include the input Ls in
    the sample grid.
    """
    L_min = max(2, int(Ls_np.min()))
    L_max = int(Ls_np.max())
    n_sample = max(50, L_max // 20)
    return np.unique(np.concatenate([
        np.arange(L_min, min(20, L_max + 1)),
        np.geomspace(max(20, L_min), L_max, n_sample).astype(int),
        np.asarray(Ls_np, dtype=int),
    ]).clip(L_min, L_max).astype(int))


def _fullsky_spin2_coupling(L: int, l_E_arr: np.ndarray):
    """Full-sky spin-2 lensing coupling |f^{EB}|^2 for fixed L.

    Implements Smith et al. (2012) Eq. 6-7 coupling via cyclic 3j identity:
        (l_B, l_E, L; 2, -2, 0) = (l_E, L, l_B; -2, 0, 2)

    This allows efficient vectorized computation: recurse on l_B (column)
    with l_E (row) as the input array.

    Args:
        L:        Fixed lensing multipole.
        l_E_arr:  Array of l values (rows; E-mode side for EB/TB).

    Returns (l_B_grid, f_odd_sq, f_even_sq):
        l_B_grid:      1-D array of l_B values (columns).
        f_odd_sq[i,j]: Parity-odd |f|^2 at (l_E[i], l_B[j], L). For EB/TB.
        f_even_sq[i,j]: Parity-even |f|^2.  For EE.
    """
    from augr.wigner import wigner3j_vectorized

    # Cyclic: (l_B, l_E, L; 2,-2,0) = (l_E, L, l_B; -2, 0, 2)
    l_B_grid, w3j = wigner3j_vectorized(L, l_E_arr, m1=-2, m2=0)

    l_E_ll = l_E_arr * (l_E_arr + 1)
    l_B_ll = l_B_grid * (l_B_grid + 1)
    L_LL = L * (L + 1)

    # Smith's three-term geometric factor
    geom = -l_B_ll[None, :] + l_E_ll[:, None] + L_LL

    # Prefactor: sqrt[(2l_E+1)(2l_B+1)(2L+1)/(16π)]
    pf = np.sqrt((2 * l_E_arr + 1)[:, None]
                 * (2 * l_B_grid + 1)[None, :]
                 * (2 * L + 1) / (16.0 * np.pi))

    # Parity masks: EB/TB use odd l_E+l_B+L, EE uses even
    parity_sum = (l_E_arr.astype(int)[:, None]
                  + l_B_grid.astype(int)[None, :] + L)
    odd_mask = (parity_sum % 2 == 1).astype(float)

    coupling_sq = (pf * w3j * geom) ** 2
    f_odd_sq = coupling_sq * odd_mask
    f_even_sq = coupling_sq * (1.0 - odd_mask)

    return l_B_grid, f_odd_sq, f_even_sq


def _fullsky_inv_spectrum(cl_tot: np.ndarray, l2_grid: np.ndarray) -> np.ndarray:
    """Safe 1/C_l at l2_grid positions."""
    l2_int = l2_grid.astype(int)
    valid = (l2_int >= 0) & (l2_int < len(cl_tot))
    inv_cl = np.zeros(len(l2_grid))
    inv_cl[valid] = np.where(cl_tot[l2_int[valid]] > 0,
                             1.0 / cl_tot[l2_int[valid]], 0.0)
    return inv_cl


def _compute_n0_tb_fullsky(Ls, spectra, nl_tt, nl_bb, l_min, l_max):
    """Full-sky N_0^{TB}: same parity-odd coupling as EB, but C^{TE}/C^{TT} weights."""
    cl_tt_tot = np.asarray(spectra.cl_tt_len + nl_tt)
    cl_bb_tot = np.asarray(spectra.cl_bb_len + nl_bb)
    cl_te_unl = np.asarray(spectra.cl_te_unl)

    Ls_np = np.asarray(Ls)
    l_E_arr = np.arange(l_min, l_max + 1, dtype=float)
    L_samples = _fullsky_L_samples(Ls_np)
    n0_inv_samples = np.zeros(len(L_samples))

    for i_L, L in enumerate(L_samples):
        l_B_grid, f_odd_sq, _ = _fullsky_spin2_coupling(int(L), l_E_arr)
        inv_bb = _fullsky_inv_spectrum(cl_bb_tot, l_B_grid)

        te_unl_sq = cl_te_unl[l_min:l_max + 1] ** 2
        tt_tot = cl_tt_tot[l_min:l_max + 1]
        safe_tt = np.where(tt_tot > 0, tt_tot, 1.0)
        l1_weight = np.where(tt_tot > 0, te_unl_sq / safe_tt, 0.0)

        l2_sum = f_odd_sq @ inv_bb
        n0_inv_samples[i_L] = np.sum(l1_weight * l2_sum) / (2 * L + 1)

    log_n0_inv = np.log(np.maximum(n0_inv_samples, 1e-300))
    n0_inv_interp = np.exp(np.interp(Ls_np, L_samples.astype(float), log_n0_inv))
    n0_inv_jax = jnp.array(n0_inv_interp)
    return jnp.where(n0_inv_jax > 0, 1.0 / n0_inv_jax, jnp.inf)


def _compute_n0_tt_fullsky(Ls, spectra, nl_tt, l_min, l_max):
    """Full-sky N_0^{TT} using vectorized (l1 l2 L; 0 0 0)."""
    from augr.wigner import wigner3j_000_vectorized

    cl_tt_tot = np.asarray(spectra.cl_tt_len + nl_tt)
    cl_tt_unl = np.asarray(spectra.cl_tt_unl)

    Ls_np = np.asarray(Ls)
    l1_arr = np.arange(l_min, l_max + 1, dtype=int)
    L_samples = _fullsky_L_samples(Ls_np)
    n0_inv_samples = np.zeros(len(L_samples))

    for i_L, L in enumerate(L_samples):
        L = int(L)
        L_LL = L * (L + 1)
        l1_ll1 = l1_arr * (l1_arr + 1)

        l2_grid, w000 = wigner3j_000_vectorized(L, l1_arr, l2_min=l_min,
                                                 l2_max=l_max)
        l2_ll2 = l2_grid * (l2_grid + 1)

        # alpha(l1,l2,L) = [L(L+1)+l1(l1+1)-l2(l2+1)] / 2
        alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :]) / 2.0
        alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None]) / 2.0

        pf = np.sqrt((2*l1_arr+1)[:, None] * (2*l2_grid+1)[None, :] *
                     (2*L+1) / (4.0 * np.pi))

        # TT response: [C_TT(l1)*alpha1 + C_TT(l2)*alpha2] × pf × w000
        tt_l1 = cl_tt_unl[l1_arr]
        tt_l2 = np.zeros(len(l2_grid))
        valid = (l2_grid >= 0) & (l2_grid < len(cl_tt_unl))
        tt_l2[valid] = cl_tt_unl[l2_grid[valid]]

        f_sq = (tt_l1[:, None] * alpha1 + tt_l2[None, :] * alpha2) ** 2 \
               * pf**2 * w000**2

        # Filter: 1 / (2 × C_TT_tot(l1) × C_TT_tot(l2))
        inv_tt_l1 = np.where(cl_tt_tot[l1_arr] > 0,
                             1.0 / cl_tt_tot[l1_arr], 0.0)
        inv_tt_l2 = _fullsky_inv_spectrum(cl_tt_tot, l2_grid.astype(float))

        integrand = f_sq * inv_tt_l1[:, None] * inv_tt_l2[None, :] / 2.0
        n0_inv_samples[i_L] = np.sum(integrand) / (2 * L + 1)

    log_n0_inv = np.log(np.maximum(n0_inv_samples, 1e-300))
    n0_inv_interp = np.exp(np.interp(Ls_np, L_samples.astype(float), log_n0_inv))
    n0_inv_jax = jnp.array(n0_inv_interp)
    return jnp.where(n0_inv_jax > 0, 1.0 / n0_inv_jax, jnp.inf)


def _compute_n0_ee_fullsky(Ls, spectra, nl_ee, l_min, l_max):
    """Full-sky N_0^{EE} using parity-even spin-2 coupling.

    Implements Okamoto & Hu 2003 (astro-ph/0301031) Eq. 14 for the EE
    estimator (Table I, EE row): each leg is the spin-2 building block

        _2F_{l_j L l_i} = [L(L+1) + l_i(l_i+1) - l_j(l_j+1)]
                        × sqrt[(2L+1)(2 l_i + 1)(2 l_j + 1)/(16π)]
                        × (l_j L l_i; +2 0 -2)

    so the response is f^EE(l1, l2, L)
        = C_EE(l1) · _2F_{l2 L l1}  +  C_EE(l2) · _2F_{l1 L l2}.

    The bracket factor is the same for all spins; only the Wigner-3j
    m-values change between spin-0 and spin-2 (see OkaHu 2003 Eq. 14
    and `scripts/n0_validation/derivation.md`).
    """
    from augr.wigner import wigner3j_vectorized

    cl_ee_tot = np.asarray(spectra.cl_ee_len + nl_ee)
    cl_ee_unl = np.asarray(spectra.cl_ee_unl)

    Ls_np = np.asarray(Ls)
    l1_arr = np.arange(l_min, l_max + 1, dtype=float)
    L_samples = _fullsky_L_samples(Ls_np)
    n0_inv_samples = np.zeros(len(L_samples))

    for i_L, L in enumerate(L_samples):
        L = int(L)
        L_LL = L * (L + 1)

        # Cyclic: (l1, l2, L; 2,-2,0) = (l2, L, l1; -2, 0, 2)
        l2_grid, w3j = wigner3j_vectorized(L, l1_arr, m1=-2, m2=0)

        l1_ll1 = l1_arr * (l1_arr + 1)
        l2_ll2 = l2_grid * (l2_grid + 1)

        # OkaHu 2003 Eq. 14 spin-2 bracket. NOT divided by 2 -- the /2
        # that appears in the spin-0 (TT, TE) paths is an artefact of
        # combining bracket/2 with the 4π prefactor; with the 16π
        # prefactor here the bracket itself is the right factor.
        alpha1 = L_LL + l1_ll1[:, None] - l2_ll2[None, :]
        alpha2 = L_LL + l2_ll2[None, :] - l1_ll1[:, None]

        pf = np.sqrt((2 * l1_arr + 1)[:, None]
                     * (2 * l2_grid + 1)[None, :]
                     * (2 * L + 1) / (16.0 * np.pi))

        # Parity-even mask: per OkaHu 2003 Eq. 22 + Table I (EE row), the
        # ε factor restricts the EE estimator to L+l1+l2 even.
        parity_sum = (l1_arr.astype(int)[:, None]
                      + l2_grid.astype(int)[None, :] + L)
        even_mask = (parity_sum % 2 == 0).astype(float)

        # EE spectra at l2 positions
        l2_int = l2_grid.astype(int)
        valid = (l2_int >= l_min) & (l2_int < len(cl_ee_unl))
        ee_at_l2 = np.zeros(len(l2_grid))
        ee_at_l2[valid] = cl_ee_unl[l2_int[valid]]

        ee_l1 = cl_ee_unl[l_min:l_max + 1]

        # Response: [C_EE(l1)·α1 + C_EE(l2)·α2] · pf · w_2 · ε
        f_sq = (ee_l1[:, None] * alpha1 + ee_at_l2[None, :] * alpha2) ** 2 \
               * pf**2 * w3j**2 * even_mask

        # Filter: 1 / (2 × C_EE_tot(l1) × C_EE_tot(l2))
        ee_tot = cl_ee_tot[l_min:l_max + 1]
        safe_ee = np.where(ee_tot > 0, ee_tot, 1.0)
        inv_ee_l1 = np.where(ee_tot > 0, 1.0 / safe_ee, 0.0)
        inv_ee_l2 = _fullsky_inv_spectrum(cl_ee_tot, l2_grid)

        integrand = f_sq * inv_ee_l1[:, None] * inv_ee_l2[None, :] / 2.0
        n0_inv_samples[i_L] = np.sum(integrand) / (2 * L + 1)

    log_n0_inv = np.log(np.maximum(n0_inv_samples, 1e-300))
    n0_inv_interp = np.exp(np.interp(Ls_np, L_samples.astype(float), log_n0_inv))
    n0_inv_jax = jnp.array(n0_inv_interp)
    return jnp.where(n0_inv_jax > 0, 1.0 / n0_inv_jax, jnp.inf)


def _compute_n0_te_fullsky(Ls, spectra, nl_tt, nl_ee, l_min, l_max,
                           te_filter='ho02_diag_approx'):
    """Full-sky N_0^{TE} using OkaHu Table I spin-mixed coupling.

    Implements the spin-mixed response per Okamoto & Hu 2003 Table I:

        f^TE(l1, l2, L) = C^TE(l1) * _2F_{l2 L l1} * eps_TE
                        + C^TE(l2) * _0F_{l1 L l2}

    where _2F uses the spin-2 Wigner-3j (l1, L, l2; -2, 0, 2) and _0F
    uses the spin-0 Wigner-3j (l1, L, l2; 0, 0, 0). The spin-2 leg
    carries the parity-even mask eps_TE (per OkaHu Eq. 22); the spin-0
    (000) Wigner-3j vanishes for L+l1+l2 odd by symmetry, so no explicit
    mask is needed on the C^TE(l2) leg. With mixed spins the squared
    response carries an interference cross term

        2 * (C^TE(l1) alpha1 pf w2F) * (C^TE(l2) alpha2 pf w000)

    that pure-spin codes hide via flat-sky phi-integration. (f_2 + f_0)**2
    captures it correctly.

    Bracket / prefactor convention: we use the spin-0 form uniformly
    (alpha = bracket/2, pf = sqrt(...(2L+1)/(4 pi))) on BOTH legs.
    pf * alpha is numerically identical in the spin-0 and spin-2
    conventions of OkaHu Eq. 14 (cf. _compute_n0_ee_fullsky lines
    761-770), so the only spin dependence enters through the
    Wigner-3j building block.

    Residual vs plancklens 'p_te'
    -----------------------------
    The form above implements OkaHu Table I's *single-projection* TE
    response (T-at-l1, E-at-l2). Plancklens's ``'p_te'`` is the
    *symmetric* estimator ``g_pte + g_pet`` whose variance carries an
    additional cross-Wick term ``2 * Cov(pte, pet)`` that the single-
    projection form does not. With ``te_filter='strict_diagonal'``
    (matching plancklens's ``fal['te']=0``) the structural residual is
    ~5% across mid-L; it goes to 10-20% at the C_TE zero-crossings
    near l~1850 where the response amplitude vanishes and the relative
    error explodes. Per this function's caller's docstring, TE
    contributes ~1-2% to N_0^MV at space-experiment noise levels, so a
    5% TE residual is sub-1-permille on N_0^MV. Capturing the cross
    term cleanly requires porting ``plancklens.nhl._get_nhl``'s leg-
    pair Wick logic to harmonic space (``augr/_qe.py`` is the bit-
    exact-validated leg-construction reference); deferred. The
    ``TestN0TEAgainstPlancklens`` slow test in ``tests/test_delensing.py``
    locks the 5% structural floor in at ``TOL_FULLSKY_TE_BULK = 6e-2``
    in bulk-L = (10, 1800).

    Parameters
    ----------
    te_filter : {'ho02_diag_approx', 'strict_diagonal'}
        Filter denominator. Default 'ho02_diag_approx' matches the
        production flat-sky path: HO02 Eq. 13 diagonal approximation
        ``C_TT(l1)*C_EE(l2) + C_TE(l1)*C_TE(l2)``. The combination can
        flip sign at (l1, l2) where C_TE(l1)*C_TE(l2) is negative and
        large, hence the abs() guard. 'strict_diagonal' uses
        ``C_TT(l1)*C_EE(l2)`` only -- matches plancklens with
        ``fal['te']=0`` for the apples-to-apples validation harness.
    """
    from augr.wigner import wigner3j_000_vectorized, wigner3j_vectorized

    if te_filter not in ('ho02_diag_approx', 'strict_diagonal'):
        raise ValueError(
            f"te_filter must be 'ho02_diag_approx' or 'strict_diagonal', "
            f"got {te_filter!r}"
        )

    cl_tt_tot = np.asarray(spectra.cl_tt_len + nl_tt)
    cl_ee_tot = np.asarray(spectra.cl_ee_len + nl_ee)
    cl_te_tot = np.asarray(spectra.cl_te_len)
    cl_te_unl = np.asarray(spectra.cl_te_unl)

    Ls_np = np.asarray(Ls)
    l1_arr = np.arange(l_min, l_max + 1, dtype=int)
    L_samples = _fullsky_L_samples(Ls_np)
    n0_inv_samples = np.zeros(len(L_samples))

    for i_L, L in enumerate(L_samples):
        L = int(L)
        L_LL = L * (L + 1)
        l1_ll1 = l1_arr * (l1_arr + 1)

        # Both Wigner-3j building blocks on the same l2 grid.
        # wigner3j_vectorized internally clamps l2_min to max(l2_min, |m3|);
        # for m1=-2, m2=0 we have m3=2 so the clamp is a no-op when l_min>=2.
        l2_grid, w000 = wigner3j_000_vectorized(
            L, l1_arr, l2_min=l_min, l2_max=l_max,
        )
        l2_grid_2, w2F = wigner3j_vectorized(
            L, l1_arr, m1=-2, m2=0,
            l2_min_global=l_min, l2_max_global=l_max,
        )
        assert l2_grid.shape == l2_grid_2.shape and np.array_equal(
            l2_grid, l2_grid_2
        ), "w000 and w2F l2 grids disagree"
        l2_ll2 = l2_grid * (l2_grid + 1)

        alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :]) / 2.0
        alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None]) / 2.0

        pf = np.sqrt((2*l1_arr+1)[:, None] * (2*l2_grid+1)[None, :] *
                     (2*L+1) / (4.0 * np.pi))

        te_l1 = cl_te_unl[l1_arr]
        te_l2 = np.zeros(len(l2_grid))
        valid = (l2_grid >= 0) & (l2_grid < len(cl_te_unl))
        te_l2[valid] = cl_te_unl[l2_grid[valid]]

        # Parity-even mask for the spin-2 leg (OkaHu Eq. 22 eps_TE).
        # The spin-0 (000) Wigner-3j already vanishes for L+l1+l2 odd
        # by column-permutation symmetry, so w000 carries the parity
        # restriction implicitly on the spin-0 leg.
        parity_sum = (l1_arr.astype(int)[:, None]
                      + l2_grid.astype(int)[None, :] + L)
        even_mask = (parity_sum % 2 == 0).astype(float)

        # Two response terms; sum then square to capture the cross term.
        f_2 = te_l1[:, None] * alpha1 * pf * w2F * even_mask
        f_0 = te_l2[None, :] * alpha2 * pf * w000
        f_total_sq = (f_2 + f_0) ** 2

        # Filter denominator -- dispatched on te_filter.
        tt_l1 = cl_tt_tot[l1_arr]
        ee_l2 = np.zeros(len(l2_grid))
        ee_l2[valid] = cl_ee_tot[l2_grid[valid]]

        if te_filter == 'ho02_diag_approx':
            te_tot_l1 = cl_te_tot[l1_arr]
            te_tot_l2 = np.zeros(len(l2_grid))
            te_tot_l2[valid] = cl_te_tot[l2_grid[valid]]
            denom = (tt_l1[:, None] * ee_l2[None, :]
                     + te_tot_l1[:, None] * te_tot_l2[None, :])
            safe_denom = np.where(np.abs(denom) > 0, denom, 1.0)
            inv_denom = np.where(np.abs(denom) > 0, 1.0 / safe_denom, 0.0)
        else:  # 'strict_diagonal' -- always positive (auto-spectra)
            denom = tt_l1[:, None] * ee_l2[None, :]
            safe_denom = np.where(denom > 0, denom, 1.0)
            inv_denom = np.where(denom > 0, 1.0 / safe_denom, 0.0)

        n0_inv_samples[i_L] = np.sum(f_total_sq * inv_denom) / (2 * L + 1)

    log_n0_inv = np.log(np.maximum(np.abs(n0_inv_samples), 1e-300))
    n0_inv_interp = np.exp(np.interp(Ls_np, L_samples.astype(float), log_n0_inv))
    n0_inv_jax = jnp.array(n0_inv_interp)
    return jnp.where(n0_inv_jax > 0, 1.0 / n0_inv_jax, jnp.inf)


def _lensing_kernel_fullsky(ls: jnp.ndarray, Ls: jnp.ndarray,
                            spectra: LensingSpectra,
                            l_min: int = 2,
                            l_max: int = 3000,
                            *,
                            w_ee: jnp.ndarray | None = None) -> jnp.ndarray:
    """Full-sky lensing kernel K(l, L) using Smith et al. (2012) coupling.

    C_l^{BB,lens} = Σ_L K(l,L) C_L^{φφ}

    K(l_B, L) = 1/(2l_B+1) × Σ_{l_E, odd} C_{l_E}^{EE,unl}
                × |f^{EB}(l_B, l_E, L)|^2

    where |f^{EB}|^2 is the Smith et al. parity-odd coupling (Eq. 6-7).
    Computed via cyclic 3j: (l_B, l_E, L; 2,-2,0) = (l_E, L, l_B; -2, 0, 2).

    If w_ee is provided, C_EE is multiplied by W_EE(ℓ_E) inside the sum --
    needed for the exact Smith+ Eq. 12 residual BB (see residual_cl_bb).
    """
    from augr.wigner import wigner3j_vectorized

    cl_ee_unl = np.asarray(spectra.cl_ee_unl)
    if w_ee is not None:
        cl_ee_unl = cl_ee_unl * np.asarray(w_ee)
    ls_np = np.asarray(ls, dtype=float)  # target l_B values
    Ls_np = np.asarray(Ls, dtype=float)
    n_l = len(ls_np)
    n_L = len(Ls_np)

    # l_E range for the sum
    l_E_arr = np.arange(l_min, l_max + 1, dtype=float)

    # Sample L values and interpolate (kernel is smooth in L)
    L_min_int = max(2, int(Ls_np.min()))
    L_max_int = int(Ls_np.max())
    n_L_sample = min(n_L, max(50, L_max_int // 20))
    L_samples = np.unique(np.concatenate([
        np.arange(L_min_int, min(20, L_max_int + 1)),
        np.geomspace(max(20, L_min_int), L_max_int, n_L_sample).astype(int),
    ]).clip(L_min_int, L_max_int).astype(int))

    K_samples = np.zeros((n_l, len(L_samples)))

    for i_L, L in enumerate(L_samples):
        L = int(L)
        if L < 2:
            continue

        # Cyclic: (l_B, l_E, L; 2,-2,0) = (l_E, L, l_B; -2, 0, 2)
        # Recurse on l_B with l_E as input array
        l_B_grid, w3j = wigner3j_vectorized(L, l_E_arr, m1=-2, m2=0)

        # Smith's geometric factor and prefactor
        l_E_ll = l_E_arr * (l_E_arr + 1)
        l_B_ll = l_B_grid * (l_B_grid + 1)
        L_LL = L * (L + 1)
        geom = -l_B_ll[None, :] + l_E_ll[:, None] + L_LL

        pf = np.sqrt((2 * l_E_arr + 1)[:, None]
                     * (2 * l_B_grid + 1)[None, :]
                     * (2 * L + 1) / (16.0 * np.pi))

        # Parity-odd mask
        parity_sum = (l_E_arr.astype(int)[:, None]
                      + l_B_grid.astype(int)[None, :] + L)
        odd_mask = (parity_sum % 2 == 1).astype(float)

        f_eb_sq = (pf * w3j * geom) ** 2 * odd_mask  # (n_l_E, n_l_B)

        # C_EE weights (indexed by l_E)
        l_E_int = l_E_arr.astype(int)
        valid_E = (l_E_int >= l_min) & (l_E_int < len(cl_ee_unl))
        ee = np.zeros(len(l_E_arr))
        ee[valid_E] = cl_ee_unl[l_E_int[valid_E]]

        # For each target l_B, extract its column and sum over l_E
        l_B_map = {int(v): j for j, v in enumerate(l_B_grid)}
        for i_l in range(n_l):
            j = l_B_map.get(int(ls_np[i_l]))
            if j is not None:
                K_samples[i_l, i_L] = np.sum(ee * f_eb_sq[:, j]) / (2 * ls_np[i_l] + 1)

    # Interpolate K to the requested L grid (per-l, log-space)
    K = np.zeros((n_l, n_L))
    L_samp_f = L_samples.astype(float)
    for i_l in range(n_l):
        log_K = np.log(np.maximum(K_samples[i_l, :], 1e-300))
        K[i_l, :] = np.exp(np.interp(Ls_np, L_samp_f, log_K))

    return jnp.array(K)


# -----------------------------------------------------------------------
# Lensing kernel and residual BB
# -----------------------------------------------------------------------

def lensing_kernel(ls: jnp.ndarray, Ls: jnp.ndarray,
                   spectra: LensingSpectra,
                   l_min: int = 2,
                   l_max: int = 3000,
                   n_phi: int = 128,
                   fullsky: bool = False,
                   *,
                   w_ee: jnp.ndarray | None = None) -> jnp.ndarray:
    """Lensing kernel K(l, L) such that C_l^{BB,lens} = Σ_L K(l,L) C_L^{φφ}.

    The kernel encodes how lensing power at multipole L generates B-mode
    power at multipole l, by deflecting the E-mode gradient:

      K(l_B, L) = Σ_{l_E} C_{l_E}^{EE,unl} × |coupling(l_B, l_E, L)|^2

    In flat-sky: the coupling involves (L·(l-L)) sin(2φ) from the spin-2
    geometry. In full-sky: it uses Smith et al. (2012) Eq. 6-7 Wigner 3j
    coupling (first-order gradient approximation, ~1% accurate at ℓ ≤ 20,
    degrades at higher ℓ due to the perturbative expansion).

    The same kernel enters the residual BB calculation after delensing:
      C_l^{BB,res} = Σ_L K(l,L) × C_L^{φφ} × N_0/(C_φφ + N_0)

    Args:
        ls:      1-D array of BB multipoles (output).
        Ls:      1-D array of lensing multipoles (input).
        spectra: LensingSpectra.
        l_min:   Minimum ell for internal E-mode sum.
        l_max:   Maximum ell for internal E-mode sum.
        n_phi:   GL quadrature nodes (flat-sky only).
        fullsky: Use Wigner 3j coupling instead of flat-sky.
        w_ee:    Optional E-mode Wiener filter W_EE(ℓ_E) = C_EE/(C_EE+N_EE),
                 indexed on the same grid as spectra.cl_ee_unl.  When
                 provided, the kernel integrand carries an extra factor
                 W_EE(ℓ_E) on top of C_EE(ℓ_E) -- needed for the exact
                 Smith+ 2012 Eq. 12 residual BB when the E map is not
                 signal-dominated (see residual_cl_bb).

    Returns:
        K: array of shape (n_l, n_L).
    """
    if fullsky:
        return _lensing_kernel_fullsky(ls, Ls, spectra, l_min, l_max,
                                        w_ee=w_ee)

    phi, w_phi = _gl_nodes(n_phi)
    cl_ee_unl = spectra.cl_ee_unl
    cl_ee_weighted = cl_ee_unl if w_ee is None else cl_ee_unl * w_ee

    # Flat-sky lensing BB at first order in the gradient expansion:
    #   C_l^{BB} = ∫ d²L/(2π)² C_L^{φφ} C_{|l-L|}^{EE,unl}
    #              × [L·(l-L)]² sin²(2φ_{l-L})
    #
    # We factor as C_l^{BB} = Σ_L K(l,L) C_L^{φφ} and compute K by GL
    # quadrature over the azimuthal angle ψ between l and L.
    # Geometry: l along x-axis, L = L(cos ψ, sin ψ), so
    #   |l-L| = √(l² + L² - 2lL cos ψ)
    #   L·(l-L) = lL cos ψ - L²
    #   φ_{l-L} = atan2(-L sin ψ, l - L cos ψ)

    # Scan over L and compute K(l, L) for all l at once
    def compute_K_column(L_val):
        """Compute K(l, L_val) for all l via φ quadrature."""
        # For each (l, ψ): compute |l - L|, L·(l-L), sin(2φ_{l-L})
        # Shapes: ls is (n_l,), phi is (n_phi,)
        ls_ = ls[:, None]       # (n_l, 1)
        psi = phi[None, :]      # (1, n_phi)

        lmL_x = ls_ - L_val * jnp.cos(psi)    # (n_l, n_phi)
        lmL_y = -L_val * jnp.sin(psi)          # (1, n_phi) broadcasts
        lmL_mag = jnp.sqrt(lmL_x**2 + lmL_y**2)

        # L · (l - L)
        LdotlmL = L_val * (ls_ * jnp.cos(psi) - L_val)

        # sin(2φ_{l-L}) where φ_{l-L} = atan2(lmL_y, lmL_x)
        phi_lmL = jnp.arctan2(lmL_y, lmL_x)
        sin2phi = jnp.sin(2.0 * phi_lmL)

        # C_{|l-L|}^{EE,unl} × W_EE(|l-L|) if w_ee was provided
        cl_ee_at_lmL = _interp_at(cl_ee_weighted, lmL_mag)

        # Integrand: C_EE [× W_EE] × (L·(l-L))² × sin²(2φ) × L / (2π)²
        integrand = cl_ee_at_lmL * LdotlmL**2 * sin2phi**2

        # Integrate over ψ with GL weights, factor of L/(2π)²
        K_col = jnp.sum(integrand * w_phi[None, :], axis=1) * L_val / (2 * jnp.pi)**2
        return K_col  # shape (n_l,)

    # Scan over L values
    def scan_K(_, L_val):
        return None, compute_K_column(L_val)

    _, K_columns = lax.scan(scan_K, None, Ls)
    # K_columns shape: (n_L, n_l) -> transpose to (n_l, n_L)
    return K_columns.T


def residual_cl_bb(ls: jnp.ndarray, Ls: jnp.ndarray,
                   spectra: LensingSpectra,
                   n0_mv: jnp.ndarray,
                   l_min: int = 2,
                   l_max: int = 3000,
                   n_phi: int = 128,
                   fullsky: bool = False,
                   *,
                   nl_ee: jnp.ndarray | None = None) -> jnp.ndarray:
    """Residual lensing BB after QE delensing (Smith et al. 2012, Eq. 12).

    The full Smith+ 2012 formula is
        C_l^{BB,res} ∝ ∫ [1 - W_EE(ℓ_E) W_φφ(L)] C_EE(ℓ_E) C_φφ(L) × (kernel)
    with W_EE(ℓ) = C_EE / (C_EE + N_EE) and W_φφ(L) = C_φφ / (C_φφ + N_0).
    This factors as
        C_l^{BB,res} = K @ [C_φφ (1 - W_φφ)] + (K - K_WEE) @ [C_φφ W_φφ]
    where K is the standard kernel and K_WEE carries an extra W_EE(ℓ_E)
    factor inside the C_EE integrand.  The first term is the "simple"
    W_EE=1 approximation; the second is the correction that becomes
    non-negligible when the E map is not signal-dominated.

    When nl_ee is None we use the W_EE=1 simplification
        C_l^{BB,res} = K @ [C_φφ N_0 / (C_φφ + N_0)]
    which is fine for space missions (C_EE/N_EE ~ 10²-10³ at the
    lensing peak) but optimistic for ground experiments.

    Args:
        ls:      BB multipoles to compute residual at.
        Ls:      Lensing multipoles for the kernel sum.
        spectra: LensingSpectra.
        n0_mv:   MV reconstruction noise N_0(L), same length as Ls.
        l_min, l_max, n_phi: passed to lensing_kernel().
        nl_ee:   EE noise spectrum on the same ℓ grid as spectra.cl_ee_unl.
                 If provided, the exact Eq. 12 form is used; if None, the
                 W_EE=1 approximation.

    Returns:
        C_l^{BB,res} at each l in ls.
    """
    cl_pp_at_L = _interp_at(spectra.cl_pp, Ls)
    w_pp = cl_pp_at_L / (cl_pp_at_L + n0_mv)
    cl_pp_res = cl_pp_at_L * (1.0 - w_pp)           # = C_φφ N_0 / (C_φφ + N_0)

    K = lensing_kernel(ls, Ls, spectra, l_min, l_max, n_phi, fullsky=fullsky)

    if nl_ee is None:
        return K @ cl_pp_res

    # Exact Smith+ 2012 Eq. 12: build K_WEE with the extra W_EE(ℓ_E) factor
    # and add the W_EE-correction term.
    cl_ee = spectra.cl_ee_unl
    nl_ee_arr = jnp.asarray(nl_ee)
    w_ee = cl_ee / (cl_ee + nl_ee_arr)
    K_wee = lensing_kernel(ls, Ls, spectra, l_min, l_max, n_phi,
                           fullsky=fullsky, w_ee=w_ee)
    return K @ cl_pp_res + (K - K_wee) @ (cl_pp_at_L * w_pp)


# -----------------------------------------------------------------------
# Iterative delensing
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class DelensedSpectra:
    """Output of the iterative delensing procedure.

    Attributes:
        ls:         BB multipoles at which residual is computed.
        cl_bb_res:  Residual lensing BB, shape (n_l,) [μK²].
        n0_mv:      MV reconstruction noise at final iteration, shape (n_L,).
        Ls:         Lensing multipoles, shape (n_L,).
        n_iter:     Number of iterations performed.
        A_lens_eff: Effective A_lens = sum(cl_bb_res) / sum(cl_bb_lens) at ls.
    """
    ls: jnp.ndarray
    cl_bb_res: jnp.ndarray
    n0_mv: jnp.ndarray
    Ls: jnp.ndarray
    n_iter: int
    A_lens_eff: float


def iterate_delensing(spectra: LensingSpectra,
                      nl_tt: jnp.ndarray,
                      nl_ee: jnp.ndarray,
                      nl_bb: jnp.ndarray,
                      ls: jnp.ndarray | None = None,
                      L_max: int = 3000,
                      l_min_qe: int = 2,
                      l_max_qe: int = 3000,
                      n_phi: int = 128,
                      n_iter: int = 5,
                      verbose: bool = False,
                      fullsky: bool = False) -> DelensedSpectra:
    """Iterative QE delensing: compute residual lensing BB self-consistently.

    The key insight (Smith et al. 2012 §3.1): lensed B-mode power acts as
    noise for the EB lens reconstruction, so after one round of delensing
    the reduced BB can be fed back into the QE to get a better φ estimate,
    and so on. Converges in 3-5 iterations for typical space experiments.

    Procedure (CLASS_delens-inspired, Trendafilova, Hotinli & Meyers 2024):
      1. Start with C_l^{BB,tot} = C_l^{BB,lensed} + N_l^{BB}
      2. Compute MV N_0(L) using current C_l^{BB,tot} in EB/TB filters
      3. Compute residual C_l^{BB,res} via lensing kernel × Wiener filter
      4. Update C_l^{BB,tot} = C_l^{BB,res} + N_l^{BB}
      5. Repeat until converged

    The response functions always use **unlensed** spectra (not updated).
    Only the filter denominators change between iterations — this is what
    makes the iteration converge rather than diverge.

    Each iteration now builds two lensing kernels (the standard K and the
    W_EE-weighted K_WEE for the exact Smith+ 2012 Eq. 12 residual), so
    the per-iteration cost is ~2× the W_EE=1 version.  For fullsky=True
    this is the dominant runtime: expect ~15-25 min for n_iter=5 on a
    typical workstation, versus ~7-12 min pre-W_EE.

    Args:
        spectra:    LensingSpectra with unlensed/lensed CMB and C_L^{φφ}.
        nl_tt:      Combined TT noise, indexed by ell.
        nl_ee:      Combined EE noise, indexed by ell.
        nl_bb:      Combined BB noise, indexed by ell.
        ls:         BB multipoles for output (default: 2..300).
        L_max:      Maximum lensing multipole for reconstruction.
        l_min_qe:   Minimum ell for QE integrals.
        l_max_qe:   Maximum ell for QE integrals.
        n_phi:      GL quadrature nodes (flat-sky only).
        n_iter:     Number of iterations (3-5 typically sufficient).
        verbose:    Print A_lens_eff at each iteration.
        fullsky:    Use full-sky Wigner 3j coupling.

    Returns:
        DelensedSpectra with residual BB, final N_0, and effective A_lens.
    """
    if ls is None:
        ls = jnp.arange(2, 301, dtype=float)

    Ls = jnp.arange(2, L_max + 1, dtype=float)

    # The iteration updates the BB spectrum used in the EB and TB filter
    # denominators. Start with full lensed BB.
    cl_bb_current = spectra.cl_bb_len.copy()
    cl_bb_res = n0 = None  # set on first iteration

    for iteration in range(n_iter):
        # Build a modified spectra-like object with updated BB for filters
        # We do this by modifying nl_bb_eff: the total BB in filters is
        # cl_bb_current + nl_bb, but the estimator functions take
        # spectra.cl_bb_len + nl_bb as total. So we pass an effective noise
        # that makes the total come out right:
        # nl_bb_eff = cl_bb_current + nl_bb - spectra.cl_bb_len
        nl_bb_eff = cl_bb_current - spectra.cl_bb_len + nl_bb

        # Compute MV N_0
        n0 = compute_n0_mv(Ls, spectra, nl_tt, nl_ee, nl_bb_eff,
                           l_min_qe, l_max_qe, n_phi, fullsky=fullsky)

        # Compute residual BB (exact Smith+ Eq. 12 with W_EE Wiener filter)
        cl_bb_res = residual_cl_bb(ls, Ls, spectra, n0,
                                   l_min_qe, l_max_qe, n_phi,
                                   fullsky=fullsky, nl_ee=nl_ee)

        # Interpolate residual onto the full ell grid for the next
        # iteration's filters.  Outside the QE ls range we fall back to
        # the full lensed BB -- i.e. "no delensing applied where we
        # didn't reconstruct."  Flat-constant extrapolation (the prior
        # behaviour at ell > ls[-1]) silently under-counted BB in the EB
        # filter denominator past ls[-1], producing an artificially
        # small N_0 and an over-optimistic residual at ell in ls.
        cl_bb_res_interp = jnp.interp(spectra.ells, ls, cl_bb_res,
                                       left=0.0, right=0.0)
        in_range = (spectra.ells >= ls[0]) & (spectra.ells <= ls[-1])
        cl_bb_current = jnp.where(in_range, cl_bb_res_interp,
                                  spectra.cl_bb_len)

        if verbose:
            cl_bb_lens_at_ls = _interp_at(spectra.cl_bb_len, ls)
            a_lens = float(jnp.sum(cl_bb_res) / jnp.sum(cl_bb_lens_at_ls))
            print(f"  Iteration {iteration + 1}: A_lens_eff = {a_lens:.4f}")

    # Final effective A_lens
    cl_bb_lens_at_ls = _interp_at(spectra.cl_bb_len, ls)
    A_lens_eff = float(jnp.sum(cl_bb_res) / jnp.sum(cl_bb_lens_at_ls))

    return DelensedSpectra(
        ls=ls,
        cl_bb_res=cl_bb_res,
        n0_mv=n0,
        Ls=Ls,
        n_iter=n_iter,
        A_lens_eff=A_lens_eff,
    )
