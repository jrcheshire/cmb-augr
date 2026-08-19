"""Gates for the JAX-native MASTER coupling matrices (augr.pseudo_cl_jax).

The reference is NaMaster's own ``NmtWorkspace.get_coupling_matrix()`` --
deterministic and sim-free, so these are exact-agreement tests rather than
Monte-Carlo recovery tests. NaMaster is imported per-test rather than at module
scope so the mask-power and structural tests still run in the aarch64
environments, which deliberately carry no namaster (and whose lack of it is
half the motivation for this module existing).
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from augr.pseudo_cl_jax import coupling_matrices, mask_power_spectrum  # noqa: E402

NSIDE, LMAX = 16, 24
SPIN2_ORDER = ["EE", "EB", "BE", "BB"]


def _smooth_mask(nside=NSIDE, b_cut_deg=20.0, width_deg=8.0):
    """Sigmoid |b| cut -- smooth, so it needs no separate apodization step."""
    npix = healpy.nside2npix(nside)
    lat = 90.0 - np.degrees(healpy.pix2ang(nside, np.arange(npix))[0])
    return 1.0 / (1.0 + np.exp(-(np.abs(lat) - b_cut_deg) / width_deg))


def _namaster_coupling(mask, nside=NSIDE, lmax=LMAX):
    """NaMaster's unbinned spin-2 coupling matrix, reshaped to [l1, i, l2, j]."""
    nmt = pytest.importorskip("pymaster")
    z = np.zeros(healpy.nside2npix(nside))
    fld = nmt.NmtField(mask, [z, z], spin=2, lmax=lmax)
    bpws = -np.ones(lmax + 1, dtype=int)
    bpws[2:] = (np.arange(2, lmax + 1) - 2) // 6
    bins = nmt.NmtBin(bpws=bpws, ells=np.arange(lmax + 1),
                      weights=np.ones(lmax + 1), lmax=lmax)
    wsp = nmt.NmtWorkspace.from_fields(fld, fld, bins)
    n = lmax + 1
    return np.asarray(wsp.get_coupling_matrix()).reshape(n, 4, n, 4)


def _augr_coupling(mask, nside=NSIDE, lmax=LMAX):
    w_ell = mask_power_spectrum(jnp.asarray(mask), nside=nside,
                                lmax_mask=3 * nside - 1)
    m_plus, m_minus = coupling_matrices(w_ell, lmax=lmax)
    return np.asarray(m_plus), np.asarray(m_minus)


def test_coupling_matrix_matches_namaster_elementwise():
    """All four spin-2 blocks, element by element, against NaMaster.

    Measured worst relative deviation over all 16 blocks: 1.27e-15, i.e. fp64
    round-off. Gated well above that so ordinary reassociation cannot trip it.

    Note EB<-BE and BE<-EB carry ``-M_minus``: the sign is a free correctness
    check on the parity split, since M_plus and M_minus are built from the same
    3j table and only the ``(-1)^(l1+l2+l3)`` selector distinguishes them.
    """
    mask = _smooth_mask()
    m_plus, m_minus = _augr_coupling(mask)
    ref = _namaster_coupling(mask)
    expected = {(0, 0): m_plus, (1, 1): m_plus, (2, 2): m_plus, (3, 3): m_plus,
                (0, 3): m_minus, (3, 0): m_minus,
                (1, 2): -m_minus, (2, 1): -m_minus}
    for (i, j), want in expected.items():
        got = ref[:, i, :, j]
        np.testing.assert_allclose(
            want, got, rtol=1e-11, atol=1e-13,
            err_msg=f"{SPIN2_ORDER[i]}<-{SPIN2_ORDER[j]}")


def test_ee_bb_sector_is_closed():
    """{EE,BB} and {EB,BE} are separate closed sectors -- cross blocks are zero.

    This is what makes the 2x2 restriction in the decoupling exact rather than
    an approximation, so it is worth pinning independently of the block values.
    """
    ref = _namaster_coupling(_smooth_mask())
    coupled = {(0, 0), (1, 1), (2, 2), (3, 3), (0, 3), (3, 0), (1, 2), (2, 1)}
    for i in range(4):
        for j in range(4):
            if (i, j) in coupled:
                continue
            blk = ref[:, i, :, j]
            assert np.all(blk == 0.0), (
                f"{SPIN2_ORDER[i]}<-{SPIN2_ORDER[j]} not identically zero: "
                f"max|.| = {np.abs(blk).max():.3e}")


def test_coupling_is_null_below_ell_two():
    """Rows and columns below l=2 vanish, from |m1| <= l1 and |m2| <= l2.

    The column half is the one that needed the spin2_body guard: without it the
    l2 = 0, 1 columns come back as a normalized-but-meaningless table rather
    than zero (see test_wigner.test_spin2_body_zeroes_L_below_abs_m2).
    """
    m_plus, m_minus = _augr_coupling(_smooth_mask())
    for name, m in (("M+", m_plus), ("M-", m_minus)):
        assert np.all(m[:2, :] == 0.0), f"{name} rows l1<2 nonzero"
        assert np.all(m[:, :2] == 0.0), f"{name} cols l2<2 nonzero"


@pytest.mark.parametrize("n_iter,tol", [(0, 3e-4), (3, 1e-12)])
def test_mask_power_spectrum_matches_anafast(n_iter, tol):
    """W_l against healpy.anafast, and n_iter shown to be load-bearing.

    healpy.anafast itself defaults to iter=3, so at n_iter=3 this compares the
    same Jacobi scheme at the same iteration count and bit-matches rather than
    making an accuracy claim: measured 5.9e-16 (smooth mask) / 1.0e-15 (sharp).
    Lowering the count degrades it smoothly -- measured 1.24e-3 / 5.3e-4 /
    1.3e-4 at n_iter = 0 / 1 / 2 on the smooth mask -- which is why n_iter is
    pinned in mask_power_spectrum rather than exposed as a tuning knob.

    Compared with a (2l+1) weighting, NOT per-l ratios: an equator-symmetric
    mask has W_l ~ 1e-33 at odd l, where a ratio is pure round-off noise.
    """
    mask = _smooth_mask()
    lmax_mask = 3 * NSIDE - 1
    mine = np.asarray(mask_power_spectrum(jnp.asarray(mask), nside=NSIDE,
                                          lmax_mask=lmax_mask, n_iter=n_iter))
    ref = healpy.anafast(mask, lmax=lmax_mask)
    ell = np.arange(lmax_mask + 1)
    wt = 2.0 * ell + 1.0
    rel = np.abs(wt * (mine - ref)).sum() / np.abs(wt * ref).sum()
    if n_iter == 3:
        assert rel < tol, f"n_iter=3 should bit-match anafast, got {rel:.2e}"
    else:
        assert rel > tol, (
            f"n_iter=0 should be visibly worse than anafast, got {rel:.2e} -- "
            "if this fails, n_iter has stopped mattering and the pinning in "
            "mask_power_spectrum needs re-deriving")


def test_coupling_matrices_jit_and_are_linear_in_w_ell():
    """Compiles under jit, and M is linear in W_l -- the whole mask dependence.

    Linearity is not decoration: it is why the Wigner symbols can be treated as
    mask-independent constants, and why the reverse pass is one extra lax.map
    rather than a custom VJP.
    """
    w_ell = mask_power_spectrum(jnp.asarray(_smooth_mask()), nside=NSIDE,
                                lmax_mask=3 * NSIDE - 1)
    fn = jax.jit(lambda w: coupling_matrices(w, lmax=LMAX))
    mp1, mm1 = fn(w_ell)
    mp2, mm2 = fn(2.5 * w_ell)
    np.testing.assert_allclose(np.asarray(mp2), 2.5 * np.asarray(mp1),
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(mm2), 2.5 * np.asarray(mm1),
                               rtol=1e-12, atol=1e-14)
