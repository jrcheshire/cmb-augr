"""
delensing_fullsky_jax.py -- pure-jnp full-sky QE N_0 and lensing kernel.

Differentiable (jax.jit / jax.grad) counterparts of the numpy full-sky
drivers in ``delensing.py`` (issue #45 Stage 3). Each per-L body is
``jnp`` throughout and uses the traced-L Wigner cores in ``wigner_jax``
(``spin2_body`` / ``spin0_body``); the per-L sweep is a ``lax.map`` over the
static ``_fullsky_L_samples`` grid (sequential -> one Wigner table live at a
time), replacing the numpy ProcessPool. The log-interp onto the requested Ls
is a differentiable ``jnp.interp``.

Shape contract: ``L`` is traced inside ``lax.map``; the l2 grid bounds are
static (a *global* l2_max = l_max + max(L_sample) for the spin-2 estimators,
so the (n_l1, n_l2) table shape is uniform across L -- extra columns fall
outside the per-L triangle and are zeroed by the Wigner mask). Validated
bit-for-bit against the numpy drivers in ``tests/test_delensing.py``.

Math (couplings, parity masks, filters, weights) is identical to the numpy
per-L workers ``_per_L_{eb,tb,tt,ee,te}`` and ``_per_L_lensing_kernel``;
only the backend changes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from augr.delensing import LensingSpectra, _fullsky_L_samples, _interp_at
from augr.wigner_jax import spin0_body, spin2_body

_PI = float(np.pi)


# -----------------------------------------------------------------------
# Spectrum gather helpers (safe out-of-range -> 0), jnp
# -----------------------------------------------------------------------

def _inv_spectrum_jax(cl_tot: jnp.ndarray, l_grid: jnp.ndarray) -> jnp.ndarray:
    """Safe 1/C_l at integer positions ``l_grid`` (0 outside [0, len))."""
    n = cl_tot.shape[0]
    li = l_grid.astype(int)
    valid = (li >= 0) & (li < n)
    cl = cl_tot[jnp.clip(li, 0, n - 1)]
    ok = valid & (cl > 0)
    return jnp.where(ok, 1.0 / jnp.where(cl > 0, cl, 1.0), 0.0)


def _gather_spectrum_jax(cl: jnp.ndarray, l_grid: jnp.ndarray,
                         l_min_valid: int = 0) -> jnp.ndarray:
    """C_l gathered at integer positions ``l_grid`` (0 outside range)."""
    n = cl.shape[0]
    li = l_grid.astype(int)
    valid = (li >= l_min_valid) & (li < n)
    return jnp.where(valid, cl[jnp.clip(li, 0, n - 1)], 0.0)


def _odd_mask(l_row: jnp.ndarray, l_col: jnp.ndarray, L_f) -> jnp.ndarray:
    parity = jnp.round(l_row[:, None] + l_col[None, :] + L_f).astype(int) % 2
    return (parity == 1).astype(float)


def _even_mask(l_row: jnp.ndarray, l_col: jnp.ndarray, L_f) -> jnp.ndarray:
    parity = jnp.round(l_row[:, None] + l_col[None, :] + L_f).astype(int) % 2
    return (parity == 0).astype(float)


def _interp_n0(n0_inv_samples: jnp.ndarray, L_samples: np.ndarray,
               Ls_np: np.ndarray, use_abs: bool = False) -> jnp.ndarray:
    """Log-interp N_0^{-1} samples onto Ls, then invert -> N_0(L)."""
    vals = jnp.abs(n0_inv_samples) if use_abs else n0_inv_samples
    log_n0 = jnp.log(jnp.maximum(vals, 1e-300))
    Ls_f = jnp.asarray(np.asarray(Ls_np), dtype=float)
    L_s = jnp.asarray(L_samples, dtype=float)
    n0_inv = jnp.exp(jnp.interp(Ls_f, L_s, log_n0))
    return jnp.where(n0_inv > 0, 1.0 / n0_inv, jnp.inf)


# -----------------------------------------------------------------------
# EB
# -----------------------------------------------------------------------

def compute_n0_eb_fullsky_jax(Ls, spectra: LensingSpectra,
                              nl_ee, nl_bb,
                              l_min: int = 2, l_max: int = 3000) -> jnp.ndarray:
    """jnp full-sky N_0^{EB}(L). Mirrors ``_per_L_eb`` / ``_compute_n0_eb_fullsky``."""
    Ls_np = np.asarray(Ls)
    L_samples = _fullsky_L_samples(Ls_np)
    Lmax_s = int(L_samples.max())

    l_E_arr = jnp.arange(l_min, l_max + 1, dtype=float)
    cl_ee_tot = spectra.cl_ee_len + nl_ee
    cl_bb_tot = spectra.cl_bb_len + nl_bb

    ee_unl = spectra.cl_ee_unl[l_min:l_max + 1]
    ee_tot = cl_ee_tot[l_min:l_max + 1]
    l_E_weight = jnp.where(ee_tot > 0, ee_unl ** 2 / jnp.where(ee_tot > 0, ee_tot, 1.0), 0.0)

    l2_min = max(0, abs(-(-2 + 0)))         # |m3| for (m1,m2)=(-2,0) -> 2
    l2_max = l_max + Lmax_s
    l_B_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    l_E_ll = l_E_arr * (l_E_arr + 1)
    l_B_ll = l_B_grid * (l_B_grid + 1)
    inv_bb = _inv_spectrum_jax(cl_bb_tot, l_B_grid)

    def body(L_f):
        w3j = spin2_body(L_f, l_E_arr, -2, 0, 2, l2_min, l2_max)
        L_LL = L_f * (L_f + 1.0)
        geom = -l_B_ll[None, :] + l_E_ll[:, None] + L_LL
        pf = jnp.sqrt((2 * l_E_arr + 1)[:, None] * (2 * l_B_grid + 1)[None, :]
                      * (2 * L_f + 1) / (16.0 * _PI))
        odd_L = _odd_mask(l_E_arr, l_B_grid, L_f)
        f_eb_sq = (pf * w3j * geom) ** 2 * odd_L
        l_B_sum = f_eb_sq @ inv_bb
        return jnp.sum(l_E_weight * l_B_sum) / (2 * L_f + 1.0)

    n0_inv = lax.map(body, jnp.asarray(L_samples, dtype=float))
    return _interp_n0(n0_inv, L_samples, Ls_np)


# -----------------------------------------------------------------------
# TB
# -----------------------------------------------------------------------

def compute_n0_tb_fullsky_jax(Ls, spectra: LensingSpectra,
                              nl_tt, nl_bb,
                              l_min: int = 2, l_max: int = 3000) -> jnp.ndarray:
    """jnp full-sky N_0^{TB}(L). Same parity-odd coupling as EB, C^TE/C^TT weights."""
    Ls_np = np.asarray(Ls)
    L_samples = _fullsky_L_samples(Ls_np)
    Lmax_s = int(L_samples.max())

    l_E_arr = jnp.arange(l_min, l_max + 1, dtype=float)
    cl_tt_tot = spectra.cl_tt_len + nl_tt
    cl_bb_tot = spectra.cl_bb_len + nl_bb

    te_unl = spectra.cl_te_unl[l_min:l_max + 1]
    tt_tot = cl_tt_tot[l_min:l_max + 1]
    l1_weight = jnp.where(tt_tot > 0, te_unl ** 2 / jnp.where(tt_tot > 0, tt_tot, 1.0), 0.0)

    l2_min = 2
    l2_max = l_max + Lmax_s
    l_B_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    l_E_ll = l_E_arr * (l_E_arr + 1)
    l_B_ll = l_B_grid * (l_B_grid + 1)
    inv_bb = _inv_spectrum_jax(cl_bb_tot, l_B_grid)

    def body(L_f):
        w3j = spin2_body(L_f, l_E_arr, -2, 0, 2, l2_min, l2_max)
        L_LL = L_f * (L_f + 1.0)
        geom = -l_B_ll[None, :] + l_E_ll[:, None] + L_LL
        pf = jnp.sqrt((2 * l_E_arr + 1)[:, None] * (2 * l_B_grid + 1)[None, :]
                      * (2 * L_f + 1) / (16.0 * _PI))
        odd_L = _odd_mask(l_E_arr, l_B_grid, L_f)
        f_odd_sq = (pf * w3j * geom) ** 2 * odd_L
        l2_sum = f_odd_sq @ inv_bb
        return jnp.sum(l1_weight * l2_sum) / (2 * L_f + 1.0)

    n0_inv = lax.map(body, jnp.asarray(L_samples, dtype=float))
    return _interp_n0(n0_inv, L_samples, Ls_np)


# -----------------------------------------------------------------------
# TT
# -----------------------------------------------------------------------

def compute_n0_tt_fullsky_jax(Ls, spectra: LensingSpectra, nl_tt,
                              l_min: int = 2, l_max: int = 3000) -> jnp.ndarray:
    """jnp full-sky N_0^{TT}(L) via (l1 l2 L; 0 0 0). Mirrors ``_per_L_tt``."""
    Ls_np = np.asarray(Ls)
    L_samples = _fullsky_L_samples(Ls_np)

    l1_arr = jnp.arange(l_min, l_max + 1, dtype=float)
    cl_tt_tot = spectra.cl_tt_len + nl_tt
    cl_tt_unl = spectra.cl_tt_unl

    l1_ll1 = l1_arr * (l1_arr + 1)
    tt_l1 = cl_tt_unl[l_min:l_max + 1]
    tot_l1 = cl_tt_tot[l_min:l_max + 1]
    inv_tt_l1 = jnp.where(tot_l1 > 0, 1.0 / jnp.where(tot_l1 > 0, tot_l1, 1.0), 0.0)

    l2_min, l2_max = l_min, l_max
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    l2_ll2 = l2_grid * (l2_grid + 1)
    tt_l2 = _gather_spectrum_jax(cl_tt_unl, l2_grid)
    inv_tt_l2 = _inv_spectrum_jax(cl_tt_tot, l2_grid)

    def body(L_f):
        w000 = spin0_body(L_f, l1_arr, l2_min, l2_max)
        L_LL = L_f * (L_f + 1.0)
        alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :]) / 2.0
        alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None]) / 2.0
        pf = jnp.sqrt((2 * l1_arr + 1)[:, None] * (2 * l2_grid + 1)[None, :]
                      * (2 * L_f + 1) / (4.0 * _PI))
        f_sq = (tt_l1[:, None] * alpha1 + tt_l2[None, :] * alpha2) ** 2 \
            * pf ** 2 * w000 ** 2
        integrand = f_sq * inv_tt_l1[:, None] * inv_tt_l2[None, :] / 2.0
        return jnp.sum(integrand) / (2 * L_f + 1.0)

    n0_inv = lax.map(body, jnp.asarray(L_samples, dtype=float))
    return _interp_n0(n0_inv, L_samples, Ls_np)


# -----------------------------------------------------------------------
# EE
# -----------------------------------------------------------------------

def compute_n0_ee_fullsky_jax(Ls, spectra: LensingSpectra, nl_ee,
                              l_min: int = 2, l_max: int = 3000) -> jnp.ndarray:
    """jnp full-sky N_0^{EE}(L) (parity-even spin-2). Mirrors ``_per_L_ee``."""
    Ls_np = np.asarray(Ls)
    L_samples = _fullsky_L_samples(Ls_np)
    Lmax_s = int(L_samples.max())

    l1_arr = jnp.arange(l_min, l_max + 1, dtype=float)
    cl_ee_tot = spectra.cl_ee_len + nl_ee
    cl_ee_unl = spectra.cl_ee_unl

    l1_ll1 = l1_arr * (l1_arr + 1)
    ee_l1 = cl_ee_unl[l_min:l_max + 1]
    tot_l1 = cl_ee_tot[l_min:l_max + 1]
    inv_ee_l1 = jnp.where(tot_l1 > 0, 1.0 / jnp.where(tot_l1 > 0, tot_l1, 1.0), 0.0)

    l2_min, l2_max = 2, l_max + Lmax_s
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    l2_ll2 = l2_grid * (l2_grid + 1)
    ee_at_l2 = _gather_spectrum_jax(cl_ee_unl, l2_grid, l_min_valid=l_min)
    inv_ee_l2 = _inv_spectrum_jax(cl_ee_tot, l2_grid)

    def body(L_f):
        w3j = spin2_body(L_f, l1_arr, -2, 0, 2, l2_min, l2_max)
        L_LL = L_f * (L_f + 1.0)
        alpha1 = L_LL + l1_ll1[:, None] - l2_ll2[None, :]
        alpha2 = L_LL + l2_ll2[None, :] - l1_ll1[:, None]
        pf = jnp.sqrt((2 * l1_arr + 1)[:, None] * (2 * l2_grid + 1)[None, :]
                      * (2 * L_f + 1) / (16.0 * _PI))
        even_L = _even_mask(l1_arr, l2_grid, L_f)
        f_sq = (ee_l1[:, None] * alpha1 + ee_at_l2[None, :] * alpha2) ** 2 \
            * pf ** 2 * w3j ** 2 * even_L
        integrand = f_sq * inv_ee_l1[:, None] * inv_ee_l2[None, :] / 2.0
        return jnp.sum(integrand) / (2 * L_f + 1.0)

    n0_inv = lax.map(body, jnp.asarray(L_samples, dtype=float))
    return _interp_n0(n0_inv, L_samples, Ls_np)


# -----------------------------------------------------------------------
# TE
# -----------------------------------------------------------------------

def compute_n0_te_fullsky_jax(Ls, spectra: LensingSpectra, nl_tt, nl_ee,
                              l_min: int = 2, l_max: int = 3000,
                              te_filter: str = "ho02_diag_approx") -> jnp.ndarray:
    """jnp full-sky N_0^{TE}(L) (spin-mixed). Mirrors ``_per_L_te``."""
    if te_filter not in ("ho02_diag_approx", "strict_diagonal"):
        raise ValueError(
            f"te_filter must be 'ho02_diag_approx' or 'strict_diagonal', "
            f"got {te_filter!r}")
    Ls_np = np.asarray(Ls)
    L_samples = _fullsky_L_samples(Ls_np)

    l1_arr = jnp.arange(l_min, l_max + 1, dtype=float)
    cl_tt_tot = spectra.cl_tt_len + nl_tt
    cl_ee_tot = spectra.cl_ee_len + nl_ee
    cl_te_tot = spectra.cl_te_len
    cl_te_unl = spectra.cl_te_unl

    l1_ll1 = l1_arr * (l1_arr + 1)
    te_l1 = cl_te_unl[l_min:l_max + 1]
    tt_l1 = cl_tt_tot[l_min:l_max + 1]
    te_tot_l1 = cl_te_tot[l_min:l_max + 1]

    l2_min, l2_max = l_min, l_max
    l2_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    l2_ll2 = l2_grid * (l2_grid + 1)
    te_l2 = _gather_spectrum_jax(cl_te_unl, l2_grid)
    ee_l2 = _gather_spectrum_jax(cl_ee_tot, l2_grid)
    te_tot_l2 = _gather_spectrum_jax(cl_te_tot, l2_grid)
    use_ho02 = te_filter == "ho02_diag_approx"

    def body(L_f):
        w000 = spin0_body(L_f, l1_arr, l2_min, l2_max)
        w2F = spin2_body(L_f, l1_arr, -2, 0, 2, l2_min, l2_max)
        L_LL = L_f * (L_f + 1.0)
        alpha1 = (L_LL + l1_ll1[:, None] - l2_ll2[None, :]) / 2.0
        alpha2 = (L_LL + l2_ll2[None, :] - l1_ll1[:, None]) / 2.0
        pf = jnp.sqrt((2 * l1_arr + 1)[:, None] * (2 * l2_grid + 1)[None, :]
                      * (2 * L_f + 1) / (4.0 * _PI))
        even_L = _even_mask(l1_arr, l2_grid, L_f)
        f_2 = te_l1[:, None] * alpha1 * pf * w2F * even_L
        f_0 = te_l2[None, :] * alpha2 * pf * w000
        f_total_sq = (f_2 + f_0) ** 2
        if use_ho02:
            denom = (tt_l1[:, None] * ee_l2[None, :]
                     + te_tot_l1[:, None] * te_tot_l2[None, :])
            inv_denom = jnp.where(jnp.abs(denom) > 0,
                                  1.0 / jnp.where(jnp.abs(denom) > 0, denom, 1.0), 0.0)
        else:
            denom = tt_l1[:, None] * ee_l2[None, :]
            inv_denom = jnp.where(denom > 0,
                                  1.0 / jnp.where(denom > 0, denom, 1.0), 0.0)
        return jnp.sum(f_total_sq * inv_denom) / (2 * L_f + 1.0)

    n0_inv = lax.map(body, jnp.asarray(L_samples, dtype=float))
    return _interp_n0(n0_inv, L_samples, Ls_np, use_abs=True)


# -----------------------------------------------------------------------
# Lensing kernel
# -----------------------------------------------------------------------

def lensing_kernel_fullsky_jax(ls, Ls, spectra: LensingSpectra,
                               l_min: int = 2, l_max: int = 3000,
                               *, w_ee=None) -> jnp.ndarray:
    """jnp full-sky lensing kernel K(l, L). Mirrors ``_per_L_lensing_kernel``.

    Returns (n_l, n_L). C_l^{BB,lens} = sum_L K(l,L) C_L^{phiphi}.
    """
    ls_np = np.asarray(ls)
    Ls_np = np.asarray(Ls)
    n_L = len(Ls_np)

    cl_ee_unl = spectra.cl_ee_unl
    if w_ee is not None:
        cl_ee_unl = cl_ee_unl * w_ee

    l_E_arr = jnp.arange(l_min, l_max + 1, dtype=float)
    l_E_ll = l_E_arr * (l_E_arr + 1)
    ee = _gather_spectrum_jax(cl_ee_unl, l_E_arr, l_min_valid=l_min)

    # L sample grid (same construction as the numpy kernel driver).
    L_min_int = max(2, int(Ls_np.min()))
    L_max_int = int(Ls_np.max())
    n_L_sample = min(n_L, max(50, L_max_int // 20))
    L_samples = np.unique(np.concatenate([
        np.arange(L_min_int, min(20, L_max_int + 1)),
        np.geomspace(max(20, L_min_int), L_max_int, n_L_sample).astype(int),
    ]).clip(L_min_int, L_max_int).astype(int))
    Lmax_s = int(L_samples.max())

    l2_min, l2_max = 2, l_max + Lmax_s
    l_B_grid = jnp.arange(l2_min, l2_max + 1, dtype=float)
    l_B_ll = l_B_grid * (l_B_grid + 1)
    # Columns of f_eb_sq we actually need: the target l_B == ls values.
    ls_idx = jnp.asarray(np.asarray(ls_np, dtype=int) - l2_min)   # into l_B_grid
    ls_f = jnp.asarray(np.asarray(ls_np), dtype=float)

    def body(L_f):
        w3j = spin2_body(L_f, l_E_arr, -2, 0, 2, l2_min, l2_max)
        L_LL = L_f * (L_f + 1.0)
        geom = -l_B_ll[None, :] + l_E_ll[:, None] + L_LL
        pf = jnp.sqrt((2 * l_E_arr + 1)[:, None] * (2 * l_B_grid + 1)[None, :]
                      * (2 * L_f + 1) / (16.0 * _PI))
        odd_L = _odd_mask(l_E_arr, l_B_grid, L_f)
        f_eb_sq = (pf * w3j * geom) ** 2 * odd_L        # (n_lE, n_l2)
        # K_col[i_l] = sum_lE ee * f_eb_sq[:, col(ls[i_l])] / (2 ls + 1)
        cols = f_eb_sq[:, ls_idx]                       # (n_lE, n_l)
        K_col = jnp.sum(ee[:, None] * cols, axis=0) / (2.0 * ls_f + 1.0)
        return jnp.where(L_f >= 2.0, K_col, 0.0)        # (n_l,)

    K_samples = lax.map(body, jnp.asarray(L_samples, dtype=float))  # (n_Lsamp, n_l)
    # Per-l log-interp onto Ls.
    log_K = jnp.log(jnp.maximum(K_samples.T, 1e-300))   # (n_l, n_Lsamp)
    Ls_f = jnp.asarray(np.asarray(Ls_np), dtype=float)
    L_s = jnp.asarray(L_samples, dtype=float)
    K = jax.vmap(lambda row: jnp.exp(jnp.interp(Ls_f, L_s, row)))(log_K)
    return K


# -----------------------------------------------------------------------
# MV combination + residual BB (the entry points iterate_delensing calls)
# -----------------------------------------------------------------------

def compute_n0_mv_fullsky_jax(Ls, spectra: LensingSpectra,
                              nl_tt, nl_ee, nl_bb,
                              l_min: int = 2, l_max: int = 3000) -> jnp.ndarray:
    """jnp full-sky MV N_0(L) = 1 / sum_alpha 1/N_0^alpha. Mirrors ``_compute_n0_mv_body``."""
    n0_tt = compute_n0_tt_fullsky_jax(Ls, spectra, nl_tt, l_min, l_max)
    n0_ee = compute_n0_ee_fullsky_jax(Ls, spectra, nl_ee, l_min, l_max)
    n0_te = compute_n0_te_fullsky_jax(Ls, spectra, nl_tt, nl_ee, l_min, l_max)
    n0_eb = compute_n0_eb_fullsky_jax(Ls, spectra, nl_ee, nl_bb, l_min, l_max)
    n0_tb = compute_n0_tb_fullsky_jax(Ls, spectra, nl_tt, nl_bb, l_min, l_max)
    inv_n0_mv = (1.0 / n0_tt + 1.0 / n0_ee + 1.0 / n0_te
                 + 1.0 / n0_eb + 1.0 / n0_tb)
    return 1.0 / inv_n0_mv


def residual_cl_bb_fullsky_jax(ls, Ls, spectra: LensingSpectra, n0_mv,
                               l_min: int = 2, l_max: int = 3000,
                               *, nl_ee=None) -> jnp.ndarray:
    """jnp full-sky residual lensing BB (Smith+ 2012 Eq. 12). Mirrors ``residual_cl_bb``."""
    cl_pp_at_L = _interp_at(spectra.cl_pp, Ls)
    w_pp = cl_pp_at_L / (cl_pp_at_L + n0_mv)
    cl_pp_res = cl_pp_at_L * (1.0 - w_pp)

    if nl_ee is None:
        K = lensing_kernel_fullsky_jax(ls, Ls, spectra, l_min, l_max)
        return K @ cl_pp_res

    cl_ee = spectra.cl_ee_unl
    w_ee = cl_ee / (cl_ee + jnp.asarray(nl_ee))
    K = lensing_kernel_fullsky_jax(ls, Ls, spectra, l_min, l_max)
    K_wee = lensing_kernel_fullsky_jax(ls, Ls, spectra, l_min, l_max, w_ee=w_ee)
    return K @ cl_pp_res + (K - K_wee) @ (cl_pp_at_L * w_pp)
