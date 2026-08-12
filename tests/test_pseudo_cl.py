"""Gate for augr.pseudo_cl: MASTER BB bandpowers on an apodized mask.

The acceptance bar: the estimator must recover the **known input** BB bandpower in
the mean, on a cut and apodized sky, with the mask's mode coupling deconvolved.

Truth here is ``W @ C_ℓ^{BB,theory}`` -- the workspace's own bandpower window applied
to an analytic CAMB spectrum the estimator never sees. Nothing is divided by a
quantity derived from the recovery, which is the failure mode of
``test_masking.py::test_namaster_mean_bandpower_crosscheck`` (see the note there):
that test's ``transfer_function(rec, true)`` returns ``rec/true``, so its "debiased"
vector equals truth identically and the comparison is vacuous.

Sims are **B-only**, entering through :meth:`MasterBB.field_from_b_alm` -- both
because that is the physically correct stand-in for a B-only NILC output (E is
discarded upstream, so the cleaned sky has identically zero E) and because it
exercises the exact code path a consumer uses.

Fast tests run at nside 64 / lmax 128; the ``slow`` twin repeats the recovery at
nside 128 / lmax 256. Everything needs pymaster (importorskip'd at module level).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("pymaster")

import healpy as hp

from augr.compsep_sims import cmb_b_alm
from augr.masking import galactic_mask
from augr.pseudo_cl import (
    MasterBB,
    apodize_mask,
    mask_moments,
    master_bin_edges,
)
from augr.spectra import CMBSpectra

R_IN = 0.01
APO_DEG = 2.0
F_SKY = 0.7

# ---------------------------------------------------------------------------
# fast: mask preparation and binning algebra (no NaMaster workspace)
# ---------------------------------------------------------------------------


def test_apodize_mask_zero_is_passthrough():
    """aposize_deg=0 returns the mask unchanged, so the taper can be a plain knob."""
    m = galactic_mask(64, F_SKY)
    out = apodize_mask(m, 0.0)
    np.testing.assert_allclose(out, np.asarray(m, dtype=np.float64), rtol=0, atol=0)


def test_apodize_mask_reduces_sky_and_smooths():
    """A taper strictly loses sky and introduces intermediate weights."""
    m = np.asarray(galactic_mask(64, F_SKY), dtype=np.float64)
    ap = apodize_mask(m, APO_DEG, "C2")
    assert ap.mean() < m.mean()
    assert np.all(ap >= 0.0) and np.all(ap <= 1.0 + 1e-12)
    # the input is binary; the output must not be
    interior = (ap > 1e-6) & (ap < 1.0 - 1e-6)
    assert interior.sum() > 0


def test_mask_moments_binary_reduces_to_mean():
    """For a BINARY mask w^i = w, so w2**2/w4 == <w>.

    This is why augr.masking.f_sky_of returning the bare mean is self-consistent for
    sharp masks -- and why an apodized mask needs f_sky_eff instead. Guards against
    swapping one for the other.
    """
    m = np.asarray(galactic_mask(64, F_SKY), dtype=np.float64)
    mom = mask_moments(m)
    assert mom.f_sky_eff == pytest.approx(mom.w1, rel=1e-12)


def test_mask_moments_apodized_is_below_mean():
    """Apodizing makes w2**2/w4 < <w>: modes are lost beyond the area lost."""
    m = np.asarray(galactic_mask(128, F_SKY), dtype=np.float64)
    mom = mask_moments(apodize_mask(m, APO_DEG, "C2"))
    assert mom.f_sky_eff < mom.w1


def test_master_bin_edges_structure():
    """One wide low bin then uniform blocks, contiguous, never exceeding ell_max."""
    edges = master_bin_edges(2, 256, 20, 29)
    assert edges[0] == (2, 29)
    assert edges[-1][1] == 256
    for (_, hi), (lo2, _) in itertools.pairwise(edges):
        assert lo2 == hi + 1
    assert all(lo <= hi for lo, hi in edges)


def test_master_bin_edges_uniform_when_low_bin_disabled():
    edges = master_bin_edges(2, 41, 20, low_bin_hi=0)
    assert edges[0] == (2, 21)
    assert edges[-1][1] == 41


def test_master_bin_edges_rejects_bad_input():
    with pytest.raises(ValueError, match="ell_max"):
        master_bin_edges(30, 10)
    with pytest.raises(ValueError, match="delta_ell"):
        master_bin_edges(2, 100, 0)


# ---------------------------------------------------------------------------
# fast: workspace, BPWF, and the recovery gate
# ---------------------------------------------------------------------------


def _build(nside, lmax, *, purify_b=False, lmax_mask=None):
    m = np.asarray(galactic_mask(nside, F_SKY), dtype=np.float64)
    ap = apodize_mask(m, APO_DEG, "C2")
    edges = master_bin_edges(2, lmax, 20, 29)
    return MasterBB.build(
        ap,
        bin_edges=edges,
        nside=nside,
        lmax=lmax,
        purify_b=purify_b,
        lmax_mask=lmax_mask,
    )


def test_build_rejects_nside_mismatch():
    m = np.asarray(galactic_mask(32, F_SKY), dtype=np.float64)
    with pytest.raises(ValueError, match="nside"):
        MasterBB.build(m, bin_edges=master_bin_edges(2, 64, 20, 29), nside=64, lmax=64)


def test_build_rejects_bins_past_lmax():
    m = np.asarray(galactic_mask(64, F_SKY), dtype=np.float64)
    with pytest.raises(ValueError, match="lmax"):
        MasterBB.build(m, bin_edges=[(2, 200)], nside=64, lmax=128)


def test_bpwf_shape_rows_sum_to_unity_and_null_monopole():
    """The BPWF preconditions the forecast leg depends on.

    Row sums of 1 and exactly-zero ell=0,1 columns are what make
    ``SignalModel``'s per-row ``np.interp`` onto the integer grid an identity --
    user-supplied windows are deliberately NOT re-normalized there, so if NaMaster
    ever changed this the Fisher would silently under-predict every bandpower.
    """
    master = _build(64, 128)
    w = master.window
    assert w.shape == (master.n_bins, master.lmax + 1)
    assert master.window_ells.shape == (master.lmax + 1,)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(w[:, :2], 0.0, rtol=0, atol=0)


def test_save_window_roundtrips_through_augr_loader():
    """save_window must be readable by augr.bandpower_windows unchanged."""
    from augr.bandpower_windows import load_bandpower_window

    master = _build(64, 128)
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "bpwf_bb.npz"
        master.save_window(p)
        ells, w = load_bandpower_window(str(p))
    np.testing.assert_allclose(ells, master.window_ells)
    np.testing.assert_allclose(w, master.window)


def test_window_feeds_forecast_from_spectra():
    """The seam: MasterBB.window -> forecast_from_spectra(bandpower_window=...).

    Asserts the BPWF actually reaches SignalModel (has_measured_bpwf True, and the
    bin count comes from the window rather than delta_ell), and that the whole
    A_res + external-covariance forecast runs on it and returns finite numbers.
    This is what makes the ilc consumer a config change rather than new wiring.
    """
    from augr.forecast import forecast_from_spectra

    lmax = 128
    master = _build(64, lmax)
    spec = CMBSpectra()
    ell = np.arange(lmax + 1, dtype=float)
    cl_bb = np.array(spec.cl_bb(ell, R_IN), dtype=np.float64)
    cl_bb[:2] = 0.0

    # a plausible residual template + a positive-definite MC-like covariance
    template = 0.05 * cl_bb
    bp = master.window @ cl_bb
    cov = np.diag((0.1 * bp) ** 2)

    res = forecast_from_spectra(
        template_ells=ell,
        template_cl=template,
        f_sky=master.f_sky_eff,
        r_fid=R_IN,
        ell_min=2,
        ell_max=lmax,
        external_covariance=cov,
        bandpower_window=master.window,
        bandpower_window_ells=master.window_ells,
    )
    assert np.isfinite(res.sigma_r_flat) and res.sigma_r_flat > 0
    assert np.isfinite(res.sigma_r_gauss) and res.sigma_r_gauss > 0
    # a Gaussian prior on A_res cannot loosen the r constraint
    assert res.sigma_r_gauss <= res.sigma_r_flat * (1 + 1e-9)
    assert np.isfinite(res.delta_r)


def test_forecast_bpwf_requires_both_args():
    """One of the pair without the other is a silent-misbinning footgun."""
    from augr.forecast import forecast_from_spectra

    master = _build(64, 128)
    with pytest.raises(ValueError, match="bandpower_window"):
        forecast_from_spectra(
            template_ells=np.arange(129, dtype=float),
            template_cl=np.ones(129),
            f_sky=0.7,
            external_covariance=np.eye(master.n_bins),
            bandpower_window=master.window,
        )


def test_purify_b_requires_matching_lmax_mask():
    """purify_b defaults lmax_mask to lmax because NaMaster's own default raises.

    The mask alm is built at 3*nside-1 and mismatches the purification, which works
    at lmax. Pinning this so the purified cross-check does not blow up later.
    """
    master = _build(64, 128, purify_b=True)
    assert master.purify_b is True
    assert master.lmax_mask == 128
    # the mismatched combination is what NaMaster rejects
    m = np.asarray(galactic_mask(64, F_SKY), dtype=np.float64)
    ap = apodize_mask(m, APO_DEG, "C2")
    import pymaster as nmt

    # numpy raises on the alm-length mismatch (73920 for 3*nside-1 vs 33153 for 256)
    with pytest.raises(ValueError, match="broadcast"):
        nmt.NmtField(ap, [np.zeros(hp.nside2npix(64))] * 2, spin=2,
                     purify_b=True, lmax=128, lmax_mask=3 * 64 - 1)


def _recovery(nside, lmax, n_sims, seed0=0):
    """Mean recovered bandpower and its SEM against W @ C_ell^theory."""
    master = _build(nside, lmax)
    spec = CMBSpectra()
    ell = np.arange(lmax + 1, dtype=float)
    cl_theory = np.array(spec.cl_bb(ell, R_IN), dtype=np.float64)  # copy: jax arrays are read-only
    cl_theory[:2] = 0.0
    truth = master.window @ cl_theory

    rec = np.array(
        [
            master.bb_from_b_alm(cmb_b_alm(spec, R_IN, lmax, seed=seed0 + i))
            for i in range(n_sims)
        ]
    )
    return master, truth, rec


def _assert_recovers(master, truth, rec):
    n = rec.shape[0]
    mean = rec.mean(axis=0)
    sem = rec.std(axis=0, ddof=1) / np.sqrt(n)
    centers = master.bin_centers
    hi = centers >= 30
    assert hi.sum() >= 3, "need several well-measured bins to test"

    ratio = mean[hi] / truth[hi]
    # per-bin: consistent with unity at 4 sigma of the measured ensemble error
    tol = 4.0 * sem[hi] / truth[hi]
    assert np.all(np.abs(ratio - 1.0) < tol), (ratio, tol)
    # band-aggregate absolute normalization
    assert abs(np.mean(ratio) - 1.0) < 0.01, np.mean(ratio)
    # the reionization bin is few-mode; assert only that it is not broken
    assert 0.7 < mean[0] / truth[0] < 1.3, mean[0] / truth[0]


def test_master_bb_recovers_input_bb():
    """MASTER recovers the known input BB in the mean on an apodized cut sky.

    Measured margin at this config: band-mean ratio 0.99799, max per-bin |dev|
    0.0063, mean SEM/truth 0.0065.

    Catches (verified by mutation): dropping the MASTER deconvolution entirely
    drives the band-mean ratio to 0.687 -- i.e. the <w^2> suppression -- so a
    missing or doubled normalization fails hard. Also catches an omitted
    NmtField(lmax=), a transposed or mis-sliced BPWF, and a Q/U spin-convention
    error in field_from_b_alm (BB power would land in EE and the ratio go to zero).

    Does NOT catch a missing apodization: a sharp mask recovers the mean just as
    well (measured 0.9981, max|dev| 0.0062), because MASTER deconvolves the mean
    either way. The taper buys variance control and purify_b stability, not mean
    normalization -- so the guards for it are test_apodize_mask_reduces_sky_and_smooths
    and test_mask_moments_apodized_is_below_mean, plus recording w1/w2/w4 per run.
    """
    master, truth, rec = _recovery(64, 128, 32)
    _assert_recovers(master, truth, rec)


@pytest.mark.slow
def test_master_bb_recovers_input_bb_production_resolution():
    """The nside 128 / lmax 256 twin -- the configuration the ilc campaign runs.

    Measured: band-mean ratio 0.99924, max per-bin |dev| 0.0107, mean SEM/truth
    0.0046. Per-bin margins are tighter here than at nside 64, so a regression is
    likelier to surface in this test first.
    """
    master, truth, rec = _recovery(128, 256, 32)
    _assert_recovers(master, truth, rec)


@pytest.mark.slow
def test_purified_and_unpurified_agree_on_a_b_only_sky():
    """With E identically zero there is nothing to purify, so the arms must agree.

    Both instances pin lmax_mask=lmax so they differ ONLY in purify_b. A large gap
    here would mean purification is doing something other than removing E leakage.
    Measured max fractional difference over 8 sims: 0.039 (the bound is 0.10, ~2.5x
    that; it is loose because 8 sims of a per-bin max is a noisy statistic).
    """
    spec = CMBSpectra()
    plain = _build(64, 128, purify_b=False, lmax_mask=128)
    pure = _build(64, 128, purify_b=True, lmax_mask=128)
    a, b = [], []
    for i in range(8):
        alm = cmb_b_alm(spec, R_IN, 128, seed=100 + i)
        a.append(plain.bb_from_b_alm(alm))
        b.append(pure.bb_from_b_alm(alm))
    ma, mb = np.mean(a, axis=0), np.mean(b, axis=0)
    hi = plain.bin_centers >= 30
    np.testing.assert_allclose(ma[hi], mb[hi], rtol=0.10)
