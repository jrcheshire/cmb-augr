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

import jax.numpy as jnp
from jax import lax

from .sht import alm2cl, map2alm
from .wigner_jax import spin2_body

__all__ = [
    "coupling_matrices",
    "mask_power_spectrum",
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
