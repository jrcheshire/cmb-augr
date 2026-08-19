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

from augr.pseudo_cl_jax import (  # noqa: E402
    MasterBBJax,
    bandpower_windows_bb,
    bin_matrices,
    coupling_matrices,
    decouple_operators,
    mask_power_spectrum,
    pseudo_cl_eb,
)

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


# ---------------------------------------------------------------------------
# S2: binning, the 2x2 decoupling, and both bandpower windows
# ---------------------------------------------------------------------------


def _edges(lmax, delta_ell=8, low_hi=9):
    out = [(2, low_hi)] if low_hi >= 2 else []
    lo = low_hi + 1 if low_hi >= 2 else 2
    while lo <= lmax:
        out.append((lo, min(lo + delta_ell - 1, lmax)))
        lo += delta_ell
    return out


def _namaster_workspace(mask, edges, nside, lmax):
    nmt = pytest.importorskip("pymaster")
    z = np.zeros(healpy.nside2npix(nside))
    fld = nmt.NmtField(mask, [z, z], spin=2, lmax=lmax)
    bpws = -np.ones(lmax + 1, dtype=int)
    for i, (lo, hi) in enumerate(edges):
        bpws[lo:hi + 1] = i
    bins = nmt.NmtBin(bpws=bpws, ells=np.arange(lmax + 1),
                      weights=np.ones(lmax + 1), lmax=lmax)
    return nmt, nmt.NmtWorkspace.from_fields(fld, fld, bins)


def _ops_for(mask, edges, nside, lmax):
    w_ell = mask_power_spectrum(jnp.asarray(mask), nside=nside,
                                lmax_mask=3 * nside - 1)
    m_plus, m_minus = coupling_matrices(w_ell, lmax=lmax)
    b_w, b_s = bin_matrices(edges, lmax)
    return m_plus, m_minus, b_w, b_s, decouple_operators(m_plus, m_minus, b_w, b_s)


@pytest.mark.parametrize("nside,lmax", [(16, 24), (32, 48)])
def test_bandpower_windows_match_namaster(nside, lmax):
    """Both windows, element by element, against get_bandpower_windows().

    Measured: W_BB<-BB agrees to 1.9e-15 of peak, W_BB<-EE to 4.9e-17.
    """
    mask = _smooth_mask(nside)
    edges = _edges(lmax)
    _, _, _, _, ops = _ops_for(mask, edges, nside, lmax)
    w_bb, w_ee = (np.asarray(x) for x in bandpower_windows_bb(ops))
    _, wsp = _namaster_workspace(mask, edges, nside, lmax)
    ref = np.asarray(wsp.get_bandpower_windows())
    np.testing.assert_allclose(w_bb, ref[3, :, 3, :], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(w_ee, ref[3, :, 0, :], rtol=1e-10, atol=1e-13)


def test_naive_single_block_inverse_is_insufficient():
    """Pins the error this module exists to avoid.

    Treating the decoupling as ``solve(B_w M+ B_s^T, B_w M+)`` -- i.e. ignoring
    that {EE, BB} is a 2x2 block -- is wrong by 3.5% of the bandpower-window
    peak, which is far too small to notice by eye and far too large to ship.
    The test asserts the naive form *fails*, so it stays a live tripwire even if
    the correct path is later refactored.
    """
    nside, lmax = 32, 48
    mask, edges = _smooth_mask(nside), _edges(lmax)
    m_plus, _, b_w, b_s, ops = _ops_for(mask, edges, nside, lmax)
    _, wsp = _namaster_workspace(mask, edges, nside, lmax)
    ref = np.asarray(wsp.get_bandpower_windows())[3, :, 3, :]

    correct = np.asarray(bandpower_windows_bb(ops)[0])
    naive = np.linalg.solve(np.asarray(b_w @ m_plus @ b_s.T),
                            np.asarray(b_w @ m_plus))
    peak = np.abs(ref).max()
    assert np.abs(correct - ref).max() / peak < 1e-10
    assert np.abs(naive - ref).max() / peak > 1e-3, (
        "the naive single-block inverse no longer deviates -- either the block "
        "structure changed or this tripwire has gone stale"
    )


def test_bpwf_rows_sum_to_unity_and_null_monopole():
    """W_BB<-BB rows sum to 1; W_BB<-EE rows sum to 0; l=0,1 columns vanish.

    The second is the physical statement that a *constant* EE spectrum must not
    leak into BB once the mask coupling is deconvolved -- a sharper check on the
    parity bookkeeping than the BB<-BB normalization, which is largely fixed by
    construction.
    """
    nside, lmax = 32, 48
    _, _, _, _, ops = _ops_for(_smooth_mask(nside), _edges(lmax), nside, lmax)
    w_bb, w_ee = (np.asarray(x) for x in bandpower_windows_bb(ops))
    np.testing.assert_allclose(w_bb.sum(axis=1), 1.0, rtol=1e-10)
    np.testing.assert_allclose(w_ee.sum(axis=1), 0.0, atol=1e-12)
    assert np.all(w_bb[:, :2] == 0.0) and np.all(w_ee[:, :2] == 0.0)


def test_decouple_matches_namaster_decouple_cell():
    """decouple_bb against wsp.decouple_cell(...)[3] on synthetic coupled spectra.

    Uses a random non-smooth pair so the comparison is not accidentally
    degenerate -- a flat input would make the EE and BB columns interchangeable
    and hide a swapped operator. Measured agreement 8.2e-16 relative.
    """
    nside, lmax = 32, 48
    mask, edges = _smooth_mask(nside), _edges(lmax)
    _, _, _, _, ops = _ops_for(mask, edges, nside, lmax)
    _, wsp = _namaster_workspace(mask, edges, nside, lmax)

    rng = np.random.default_rng(0)
    c_ee = np.abs(rng.normal(size=lmax + 1))
    c_bb = np.abs(rng.normal(size=lmax + 1))
    c_ee[:2] = 0.0
    c_bb[:2] = 0.0
    ref = np.asarray(wsp.decouple_cell(
        np.array([c_ee, np.zeros(lmax + 1), np.zeros(lmax + 1), c_bb])))[3]
    from augr.pseudo_cl_jax import decouple_bb
    mine = np.asarray(decouple_bb(ops, jnp.asarray(c_ee), jnp.asarray(c_bb)))
    np.testing.assert_allclose(mine, ref, rtol=1e-11, atol=1e-14)


def test_pseudo_cl_eb_matches_compute_coupled_cell():
    """The masked-map spin-2 analysis against nmt.compute_coupled_cell."""
    nmt = pytest.importorskip("pymaster")
    nside, lmax = 32, 48
    mask = _smooth_mask(nside)
    cl = np.concatenate([[0.0, 0.0], 1.0 / np.arange(2, lmax + 1) ** 2.0])
    b_alm = healpy.synalm(cl, lmax=lmax, new=True)
    zero = np.zeros_like(b_alm)
    q, u = healpy.alm2map_spin([zero, b_alm], nside, 2, lmax)

    c_ee, c_bb = pseudo_cl_eb(jnp.asarray(mask), jnp.asarray(np.stack([q, u])),
                              nside=nside, lmax=lmax)
    fld = nmt.NmtField(mask, [q, u], spin=2, lmax=lmax)
    ref = np.asarray(nmt.compute_coupled_cell(fld, fld))
    np.testing.assert_allclose(np.asarray(c_ee), ref[0], rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(np.asarray(c_bb), ref[3], rtol=1e-10, atol=1e-20)


def test_masterbb_jax_is_drop_in_for_masterbb():
    """Property-for-property against the NaMaster-backed MasterBB.

    Includes bin_centers, which is NaMaster's bin-weight mean (the bin midpoint
    for uniform weights) and NOT the bandpower-window centroid -- those differ
    by 0.13 here, and the centroid is the wrong one to hand a template consumer.
    """
    pytest.importorskip("pymaster")
    from augr.masking import galactic_mask
    from augr.pseudo_cl import MasterBB, apodize_mask

    nside, lmax = 32, 48
    mask = np.asarray(apodize_mask(np.asarray(galactic_mask(nside, 0.7)), 8.0))
    edges = _edges(lmax)
    ref = MasterBB.build(mask, bin_edges=edges, nside=nside, lmax=lmax)
    mine = MasterBBJax.build(jnp.asarray(mask), bin_edges=edges, nside=nside,
                             lmax=lmax)

    assert mine.bin_edges == ref.bin_edges
    assert mine.n_bins == ref.n_bins and mine.window_ells.shape == ref.window_ells.shape
    np.testing.assert_allclose(np.asarray(mine.window), np.asarray(ref.window),
                               rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(np.asarray(mine.bin_centers),
                               np.asarray(ref.bin_centers), rtol=1e-10)
    np.testing.assert_allclose(float(mine.f_sky_eff), ref.f_sky_eff, rtol=1e-12)

    cl = np.concatenate([[0.0, 0.0], 1.0 / np.arange(2, lmax + 1) ** 2.0])
    b_alm = healpy.synalm(cl, lmax=lmax, new=True)
    np.testing.assert_allclose(np.asarray(mine.bb_from_b_alm(jnp.asarray(b_alm))),
                               ref.bb_from_b_alm(b_alm), rtol=1e-10)


def test_masterbb_jax_is_a_pytree():
    """Unlike MasterBB it is picklable and traceable -- the point of the rewrite."""
    import pickle
    nside, lmax = 16, 24
    m = MasterBBJax.build(jnp.asarray(_smooth_mask(nside)), bin_edges=_edges(lmax),
                          nside=nside, lmax=lmax)
    leaves = jax.tree_util.tree_leaves(m)
    assert leaves, "no traced leaves -- mask would not receive a gradient"
    np.testing.assert_allclose(np.asarray(pickle.loads(pickle.dumps(m)).window),
                               np.asarray(m.window), rtol=0, atol=0)


def test_summary_reports_and_warns_on_conditioning():
    """cond() diagnostics, and the warning fires where it was measured to.

    Measured at nside=32/lmax=48 on a sigmoid mask: delta_l=1 gives cond(s)=1.45
    at f_sky 0.65 but 2.3e4 at f_sky 0.048. Bin width and sky fraction interact
    -- it is not a pure bin-width effect.
    """
    nside, lmax = 32, 48
    npix = healpy.nside2npix(nside)
    lat = 90.0 - np.degrees(healpy.pix2ang(nside, np.arange(npix))[0])
    tight = 1.0 / (1.0 + np.exp(-(np.abs(lat) - 75.0) / 5.0))

    wide_bins = MasterBBJax.build(jnp.asarray(tight),
                                  bin_edges=_edges(lmax, delta_ell=20, low_hi=21),
                                  nside=nside, lmax=lmax).summary()
    assert wide_bins["cond_s"] < 10.0 and not wide_bins["warnings"]

    per_ell = MasterBBJax.build(jnp.asarray(tight),
                                bin_edges=[(x, x) for x in range(2, lmax + 1)],
                                nside=nside, lmax=lmax).summary()
    assert per_ell["cond_s"] > 1e3 and per_ell["warnings"]


# ---------------------------------------------------------------------------
# Non-circular recovery: does the estimator return the spectrum that went in?
# ---------------------------------------------------------------------------


def _recover(mask, edges, nside, lmax, ee_over_bb, n_sims, seed0=1000):
    """Mean recovered C_b^BB over sims, with its SEM, plus both truths."""
    mb = MasterBBJax.build(jnp.asarray(mask), bin_edges=edges, nside=nside,
                           lmax=lmax)
    w_bb, w_ee = np.asarray(mb.window), np.asarray(mb.window_ee)
    cl_bb = np.concatenate([[0.0, 0.0], 1.0 / np.arange(2, lmax + 1) ** 2.0])
    cl_ee = ee_over_bb * cl_bb

    recs = []
    for s in range(n_sims):
        np.random.seed(seed0 + s)  # noqa: NPY002 - healpy.synalm uses the global RNG
        alm_e = (healpy.synalm(cl_ee, lmax=lmax, new=True) if ee_over_bb > 0
                 else np.zeros(healpy.Alm.getsize(lmax), dtype=complex))
        alm_b = healpy.synalm(cl_bb, lmax=lmax, new=True)
        q, u = healpy.alm2map_spin([alm_e, alm_b], nside, 2, lmax)
        recs.append(np.asarray(mb.bb(jnp.asarray(np.stack([q, u])))))
    recs = np.asarray(recs)
    return (recs.mean(axis=0), recs.std(axis=0, ddof=1) / np.sqrt(n_sims),
            w_bb @ cl_bb + w_ee @ cl_ee, w_bb @ cl_bb)


@pytest.mark.slow
def test_recovers_input_bb_b_only():
    """Mean over sims equals the windowed input spectrum, on a B-only sky.

    Truth is ``W @ C_l^theory`` with an analytic input the estimator never sees
    -- not a transfer function derived from the recovery itself, which would be
    circular (the trap flagged in tests/test_masking.py's own docstring).

    Measured at 120 sims: mean/truth 0.969-1.045, max |mean-truth|/SEM = 1.66.
    """
    nside, lmax = 32, 48
    mean, sem, truth, _ = _recover(_smooth_mask(nside, 25.0, 8.0), _edges(lmax),
                                   nside, lmax, ee_over_bb=0.0, n_sims=120)
    dev = np.abs(mean - truth) / sem
    assert dev.max() < 4.0, f"max deviation {dev.max():.2f} SEM: {mean / truth}"


@pytest.mark.slow
def test_recovers_input_bb_with_e_present_needs_both_windows():
    """With E on the sky, the truth is TWO windows -- and one is not close enough.

    A masked B/E sky couples EE into pseudo-BB through ``M-``, so the unbiased
    comparison is ``W_BB<-BB @ C_BB + W_BB<-EE @ C_EE``. Measured at EE/BB = 300
    over 120 sims: against that truth the estimator is unbiased (max 1.25 SEM),
    while against the single-window truth the two lowest bins come out 9.0x and
    -11.5x off -- including a sign flip.

    This is the gate for dropping the ``C @ Ctilde_EE`` term in decouple_bb or
    the ``W_BB<-EE`` return from bandpower_windows_bb. Both would leave every
    deterministic NaMaster comparison in this file passing.
    """
    nside, lmax = 32, 48
    mean, sem, truth, naive = _recover(_smooth_mask(nside, 25.0, 8.0),
                                       _edges(lmax), nside, lmax,
                                       ee_over_bb=300.0, n_sims=120)
    dev = np.abs(mean - truth) / sem
    assert dev.max() < 4.0, f"max deviation {dev.max():.2f} SEM: {mean / truth}"
    assert np.abs(mean[:2] / naive[:2] - 1.0).max() > 0.5, (
        "the single-window truth is no longer distinguishable -- this test has "
        f"stopped guarding anything (mean/naive = {mean[:2] / naive[:2]})"
    )
