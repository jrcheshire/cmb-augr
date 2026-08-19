"""JAX-native MASTER pseudo-Cl for cut-sky BB bandpowers.

The differentiable counterpart to :mod:`augr.pseudo_cl`. That module wraps
NaMaster, which is C code behind an opaque ``NmtWorkspace`` handle -- correct,
well-tested, and impossible to put inside a ``jax.grad``. This one rebuilds the
same estimator out of pieces augr already owns, so the sky mask can be a
continuous design parameter rather than a fixed choice::

    mask (traced)  --map2alm(spin=0)--> W_l
                   --Wigner 3j sum-->   M+, M-        [this module, S1]
                   --bin + 2x2 solve--> C_b^BB        [S2]

**The EE/BB sector closes on itself.** For a spin-2 field the mask couples the
four pseudo-spectra (EE, EB, BE, BB), but the 4x4 system is block diagonal as
{EE, BB} + {EB, BE} -- verified against ``NmtWorkspace.get_coupling_matrix()``,
whose cross blocks are identically zero. Within the closed block

    <C~_EE> = M+ C_EE + M- C_BB
    <C~_BB> = M- C_EE + M+ C_BB

so BB recovery needs both parities and both pseudo-spectra. (The {EB, BE} block
reuses the same two matrices, with ``EB<-BE = -M-``.)

**Conventions.** ``M+-`` follow Hivon et al. 2002 / Brown, Castro & Taylor 2005:

    M+-(l1,l2) = (2 l2 + 1)/(4 pi)
                 * sum_l3 (2 l3 + 1) W_l3 [3j(l1,l2,l3; 2,-2,0)]^2
                 * [1 +- (-1)^(l1+l2+l3)] / 2

No beam and no pixel window are folded in here: the surrounding pipeline already
absorbs the common-resolution beam ``B_c^2`` into the multiplicative transfer
``F_b`` (see :func:`augr.spectrum_stages.beamed_prior`), so applying it twice
would double-count.

**The l3 grid runs to ``2 * lmax``, not to ``lmax_mask``.**
:func:`augr.wigner_jax.spin2_body` normalizes by ``sum_j (2j+1) w^2 = 1`` over
whatever grid it is handed, so a row whose support runs past the end is
renormalized against a partial sum (measured: 73-162% error). ``W_l`` is
zero-padded above ``lmax_mask`` instead. The extra columns cost 0.6 MB at
lmax=192.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .sht import alm2cl, map2alm, synthesis_pol
from .wigner_jax import spin2_body

__all__ = [
    "DecoupleOperators",
    "MasterBBJax",
    "bandpower_windows_bb",
    "bin_matrices",
    "coupling_matrices",
    "decouple_bb",
    "decouple_operators",
    "mask_power_spectrum",
    "pseudo_cl_eb",
]


def _parity(x: jnp.ndarray) -> jnp.ndarray:
    """``(-1)^x`` for integer-valued float input."""
    return jnp.where(jnp.mod(jnp.round(x), 2.0) == 0.0, 1.0, -1.0)


def mask_power_spectrum(mask, *, nside: int, lmax_mask: int,
                        n_iter: int = 3) -> jnp.ndarray:
    """``W_l`` of the mask, differentiable in ``mask``.

    ``n_iter`` is load-bearing, not a tuning knob: it must match the Jacobi
    iteration count NaMaster's ``NmtField`` uses, which is 3. Measured error
    against ``healpy.anafast`` runs 2.2e-3 at ``n_iter=0`` and 3.0e-16 at 3.

    Returns ``(lmax_mask + 1,)``.
    """
    m = jnp.asarray(mask)[None, :]
    alm = map2alm(m, spin=0, lmax=int(lmax_mask), nside=int(nside),
                  n_iter=int(n_iter))
    return alm2cl(alm[0], int(lmax_mask))


def coupling_matrices(w_ell, *, lmax: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parity-split spin-2 mode-coupling matrices ``(M+, M-)``.

    Both are ``(lmax + 1, lmax + 1)`` indexed ``[l1, l2]``. Rows and columns
    below l=2 are identically zero -- the ``|m1| <= l1`` and ``|m2| <= l2``
    constraints on the 3j symbol -- matching NaMaster.

    ``w_ell`` is the mask power spectrum from :func:`mask_power_spectrum`; it is
    zero-padded (or truncated) onto the internal ``[0, 2*lmax]`` l3 grid.

    Linear in ``w_ell``, and the Wigner symbols carry no mask dependence, so the
    whole mask gradient flows through this one argument.
    """
    lmax = int(lmax)
    l3_max = 2 * lmax

    w_ell = jnp.asarray(w_ell)
    w = jnp.zeros(l3_max + 1, dtype=w_ell.dtype)
    n_take = min(int(w_ell.shape[0]), l3_max + 1)
    w = w.at[:n_take].set(w_ell[:n_take])

    l1 = jnp.arange(lmax + 1, dtype=float)
    l3 = jnp.arange(l3_max + 1, dtype=float)
    pre3 = (2.0 * l3 + 1.0) * w
    pre3_signed = pre3 * _parity(l3)

    def body(l2):
        # 3j(l1[i], l2, l3[j]; 2, -2, 0) -- the m=0 slot is the MASK multipole.
        w3j_sq = spin2_body(l2, l1, 2, -2, 0, 0, l3_max) ** 2
        # One pass, two reductions: (-1)^(l1+l2+l3) factorizes as
        # (-1)^(l1+l2) * (-1)^l3, so the l3 sum is done once for each parity.
        s_tot = w3j_sq @ pre3
        s_sgn = (w3j_sq @ pre3_signed) * _parity(l1 + l2)
        pref = (2.0 * l2 + 1.0) / (4.0 * jnp.pi)
        return 0.5 * pref * (s_tot + s_sgn), 0.5 * pref * (s_tot - s_sgn)

    # lax.map stacks over l2, giving [l2, l1]; transpose to the [l1, l2] the
    # MASTER convention wants. Sequential by construction -- one Wigner table
    # is live at a time, 0.6 MB at lmax=192.
    m_plus, m_minus = lax.map(body, jnp.arange(lmax + 1, dtype=float))
    return m_plus.T, m_minus.T


# ---------------------------------------------------------------------------
# Binning and the 2x2 decoupling
# ---------------------------------------------------------------------------


class DecoupleOperators(NamedTuple):
    """Binned coupling operators for the closed {EE, BB} sector.

    ``s = B_w (M+ + M-) B_s^T`` and ``d = B_w (M+ - M-) B_s^T`` are the sum and
    difference combinations that diagonalize the 2x2 block system. Working with
    these instead of ``P = B_w M+ B_s^T`` and ``Q = B_w M- B_s^T`` separately
    turns the block inverse

        [P Q]^-1   [A C]              A = (s^-1 + d^-1)/2
        [Q P]    = [C A]      with    C = (s^-1 - d^-1)/2

    into two ordinary solves, so no matrix is ever explicitly inverted.
    """

    s: jnp.ndarray          # (n_bins, n_bins)
    d: jnp.ndarray          # (n_bins, n_bins)
    bin_weight: jnp.ndarray  # (n_bins, lmax+1)
    g_plus: jnp.ndarray     # B_w (M+ + M-),  (n_bins, lmax+1)
    g_minus: jnp.ndarray    # B_w (M+ - M-),  (n_bins, lmax+1)


def bin_matrices(bin_edges, lmax: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(B_w, B_s)``: the binning and the piecewise-constant unbinning.

    ``B_w`` averages over each inclusive ``(lo, hi)`` bin with uniform weight --
    NaMaster's convention with ``weights=ones``. ``B_s`` is its 0/1 support,
    which is what un-bins a bandpower back onto the l grid (the same
    piecewise-constant choice as
    :func:`augr.bandpower_windows.unbin_bandpower_template`, and for the same
    reason: it is the unique reconstruction that round-trips).
    """
    lmax = int(lmax)
    edges = [(int(lo), int(hi)) for lo, hi in bin_edges]
    b_w = np.zeros((len(edges), lmax + 1))
    b_s = np.zeros((len(edges), lmax + 1))
    for i, (lo, hi) in enumerate(edges):
        if hi > lmax:
            raise ValueError(f"bin ({lo}, {hi}) runs past lmax={lmax}.")
        b_w[i, lo:hi + 1] = 1.0 / (hi - lo + 1)
        b_s[i, lo:hi + 1] = 1.0
    return jnp.asarray(b_w), jnp.asarray(b_s)


def decouple_operators(m_plus, m_minus, bin_weight, bin_select) -> DecoupleOperators:
    """Assemble the binned sum/difference operators from the coupling matrices."""
    g_plus = bin_weight @ (m_plus + m_minus)
    g_minus = bin_weight @ (m_plus - m_minus)
    return DecoupleOperators(
        s=g_plus @ bin_select.T,
        d=g_minus @ bin_select.T,
        bin_weight=bin_weight,
        g_plus=g_plus,
        g_minus=g_minus,
    )


def bandpower_windows_bb(ops: DecoupleOperators) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(W_BB<-BB, W_BB<-EE)``, each ``(n_bins, lmax+1)``.

    **There are two windows, and dropping the second is a real error.** For a
    B-only sky the mask still produces a nonzero pseudo-EE through ``M-``, so
    the truth a recovery test compares against is

        C_b^truth = W_BB<-BB @ C_l^BB + W_BB<-EE @ C_l^EE

    ``MasterBB.window`` is ``W_BB<-BB`` alone, which is the whole story only
    when C_l^EE is identically zero. Measured on an apodized 25 deg cut at
    nside=32: ``max|W_BB<-EE| / max|W_BB<-BB| = 0.169``.
    """
    x = jnp.linalg.solve(ops.s, ops.g_plus)
    y = jnp.linalg.solve(ops.d, ops.g_minus)
    return 0.5 * (x + y), 0.5 * (x - y)


def decouple_bb(ops: DecoupleOperators, ctilde_ee, ctilde_bb) -> jnp.ndarray:
    """Decoupled ``C_b^BB`` from the two coupled pseudo-spectra.

    Needs pseudo-EE as well as pseudo-BB: the {EE, BB} block is 2x2, and
    treating it as ``solve(P, ...)`` on BB alone is wrong by 3.5% of the
    bandpower-window peak (measured at nside=32; pinned by
    ``test_naive_single_block_inverse_is_insufficient``).
    """
    u = ops.bin_weight @ (jnp.asarray(ctilde_ee) + jnp.asarray(ctilde_bb))
    v = ops.bin_weight @ (jnp.asarray(ctilde_bb) - jnp.asarray(ctilde_ee))
    return 0.5 * (jnp.linalg.solve(ops.s, u) + jnp.linalg.solve(ops.d, v))


def pseudo_cl_eb(mask, qu, *, nside: int, lmax: int,
                 n_iter: int = 3) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Coupled pseudo-spectra ``(C~_EE, C~_BB)`` of a masked Q/U map.

    The spin-2 analysis of ``mask * (Q, U)``; both spectra come from the single
    transform, so pseudo-EE costs nothing extra.
    """
    masked = jnp.asarray(mask)[None, :] * jnp.asarray(qu)
    eb = map2alm(masked, spin=2, lmax=int(lmax), nside=int(nside),
                 n_iter=int(n_iter))
    return alm2cl(eb[0], int(lmax)), alm2cl(eb[1], int(lmax))


# ---------------------------------------------------------------------------
# The estimator object
# ---------------------------------------------------------------------------


class MasterBBJax(eqx.Module):
    """MASTER BB estimator on a traced mask.

    Mirrors :class:`augr.pseudo_cl.MasterBB`'s property names so it is a drop-in
    wherever that object's outputs are consumed, with one addition
    (:attr:`window_ee`) and one deliberate reversal.

    **It is a pytree, unlike MasterBB.** ``MasterBB`` documents itself as an
    exception to the frozen-dataclass rule because it owns an ``NmtWorkspace`` C
    handle: not hashable, not picklable, must be rebuilt inside every pool
    worker. Nothing here is opaque, so this one is an ordinary ``eqx.Module`` --
    picklable, poolable, and usable as a traced argument. ``mask`` is a leaf, so
    ``jax.grad`` reaches it; ``bin_edges``, ``nside``, ``lmax`` and ``lmax_mask``
    are static.

    ``purify_b`` has no analogue here. For a NILC-cleaned map there is nothing to
    purify (the cleaner discards E), and the cost of omitting it elsewhere is
    variance, not bias -- which the MC covariance measures rather than assumes.
    """

    mask: jax.Array
    _ops: DecoupleOperators
    _window: jax.Array
    _window_ee: jax.Array
    _bin_edges: tuple = eqx.field(static=True)
    _nside: int = eqx.field(static=True)
    _lmax: int = eqx.field(static=True)
    _lmax_mask: int = eqx.field(static=True)
    _n_iter: int = eqx.field(static=True)

    @classmethod
    def build(cls, mask, *, bin_edges, nside: int, lmax: int,
              lmax_mask: int | None = None, n_iter: int = 3) -> MasterBBJax:
        """Compute W_l, the coupling matrices and both windows for ``mask``."""
        nside, lmax = int(nside), int(lmax)
        lmax_mask = 3 * nside - 1 if lmax_mask is None else int(lmax_mask)
        edges = tuple((int(lo), int(hi)) for lo, hi in bin_edges)
        if not edges:
            raise ValueError("bin_edges is empty.")
        mask = jnp.asarray(mask)
        w_ell = mask_power_spectrum(mask, nside=nside, lmax_mask=lmax_mask,
                                    n_iter=n_iter)
        m_plus, m_minus = coupling_matrices(w_ell, lmax=lmax)
        b_w, b_s = bin_matrices(edges, lmax)
        ops = decouple_operators(m_plus, m_minus, b_w, b_s)
        window, window_ee = bandpower_windows_bb(ops)
        return cls(mask=mask, _ops=ops, _window=window, _window_ee=window_ee,
                   _bin_edges=edges, _nside=nside, _lmax=lmax,
                   _lmax_mask=lmax_mask, _n_iter=int(n_iter))

    # --- geometry -------------------------------------------------------
    @property
    def bin_edges(self) -> list[tuple[int, int]]:
        return [tuple(e) for e in self._bin_edges]

    @property
    def n_bins(self) -> int:
        return len(self._bin_edges)

    @property
    def nside(self) -> int:
        return self._nside

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def lmax_mask(self) -> int:
        return self._lmax_mask

    @property
    def window_ells(self) -> np.ndarray:
        return np.arange(self._lmax + 1)

    @property
    def window(self) -> jax.Array:
        """``W_BB<-BB``, ``(n_bins, lmax+1)`` -- the MasterBB.window analogue."""
        return self._window

    @property
    def window_ee(self) -> jax.Array:
        """``W_BB<-EE``. No MasterBB analogue; see :func:`bandpower_windows_bb`."""
        return self._window_ee

    @property
    def bin_centers(self) -> jax.Array:
        """Effective centres, matching ``NmtBin.get_effective_ells()``.

        The mean of l over the **bin weights** -- for the uniform weights used
        here, exactly the bin midpoint (verified against NaMaster). Note this is
        *not* the bandpower-window centroid ``sum_l l W_b(l) / sum_l W_b(l)``,
        which is a different and slightly shifted quantity (0.13 apart at
        nside=32/lmax=48). The window is right there in :attr:`window` if a
        caller wants the centroid instead.

        Kept faithful to NaMaster because these values get handed to template
        consumers, where a mis-stated bin coordinate is a known, expensive
        failure mode (see ``augr.bandpower_windows.unbin_bandpower_template``).
        """
        ell = jnp.arange(self._lmax + 1, dtype=float)
        b_w = self._ops.bin_weight
        return (b_w @ ell) / jnp.sum(b_w, axis=1)

    # --- mask moments ---------------------------------------------------
    @property
    def f_sky_eff(self) -> jax.Array:
        """``<w^2>^2 / <w^4>`` -- the Knox mode count for an apodized mask.

        Differentiable, unlike ``masking.f_sky_of``'s bare ``<w>`` (which is only
        correct for a binary mask anyway).
        """
        w2 = jnp.mean(self.mask ** 2)
        w4 = jnp.mean(self.mask ** 4)
        return w2 ** 2 / w4

    # --- estimation -----------------------------------------------------
    def decouple(self, ctilde_ee, ctilde_bb) -> jax.Array:
        return decouple_bb(self._ops, ctilde_ee, ctilde_bb)

    def bb(self, qu) -> jax.Array:
        """Decoupled ``C_b^BB`` of a Q/U map ``(2, npix)``."""
        c_ee, c_bb = pseudo_cl_eb(self.mask, qu, nside=self._nside,
                                  lmax=self._lmax, n_iter=self._n_iter)
        return self.decouple(c_ee, c_bb)

    def bb_from_b_alm(self, b_alm) -> jax.Array:
        """Decoupled ``C_b^BB`` of a B-only alm, mirroring MasterBB."""
        b_alm = jnp.asarray(b_alm)
        zero = jnp.zeros_like(b_alm)
        _t, q, u = synthesis_pol(zero, zero, b_alm, lmax=self._lmax,
                                 nside=self._nside)
        return self.bb(jnp.stack([q, u], axis=0))

    # --- diagnostics ----------------------------------------------------
    def summary(self) -> dict:
        """Conditioning of the two decoupling solves.

        Bin width and sky fraction **interact**; neither alone tells the story.
        Measured at nside=32/lmax=48 on a sigmoid mask (cond(s)):

            f_sky     dl=1     dl=5    dl=10    dl=20
            0.654     1.45     1.30     1.11     1.09
            0.143     4.4e2    3.7      1.4      1.3
            0.048     2.3e4    1.0e1    2.0      1.7

        At generous f_sky even per-l bins are fine; the blow-up is a small-f_sky
        phenomenon that narrow bins expose. Operationally ``delta_l >= 5`` is
        safe across this whole range and ``delta_l >= 10`` has wide margin, so
        bound the bin width and let the mask move -- but do not read that as
        "f_sky does not matter", because at dl=1 it moves cond by four orders.
        """
        cond_s = float(jnp.linalg.cond(self._ops.s))
        cond_d = float(jnp.linalg.cond(self._ops.d))
        warnings = []
        if max(cond_s, cond_d) > 1e3:
            warnings.append(
                f"WARNING: decoupling is ill-conditioned (cond(s)={cond_s:.3e}, "
                f"cond(d)={cond_d:.3e}). Widen the bins before narrowing the mask."
            )
        return {"cond_s": cond_s, "cond_d": cond_d, "n_bins": self.n_bins,
                "f_sky_eff": float(self.f_sky_eff), "warnings": warnings}
