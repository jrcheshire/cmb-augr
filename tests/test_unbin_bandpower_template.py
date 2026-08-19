"""Un-binning a natively binned template, and the guard against not doing it.

The bug this closes: a component-separation residual template arrives as one
number per bandpower bin, and handing its bin CENTRES to a consumer that
interpolates silently reinterprets it as a function sampled on an ell grid.
With a wide reionization bin followed by narrow ones, linear interpolation
across the resulting cliff overshoots the next bin by an order of magnitude.
"""
from __future__ import annotations

import numpy as np
import pytest

from augr.bandpower_windows import unbin_bandpower_template
from augr.foregrounds import ResidualTemplateForegroundModel

# A CMB-analysis binning: one wide reionization bin, then delta_ell = 20.
BIN_LO = np.array([2, *range(30, 250, 20)])
BIN_HI = np.array([29, *range(49, 269, 20)])
ELLS = np.arange(257, dtype=float)


def _steep_template():
    """Dust-like: ~100x drop from the wide first bin to the next."""
    return np.array([1.08e-6, 9.93e-9, 1.99e-8, 1.34e-8, 1.66e-8, 1.52e-8,
                     1.16e-8, 9.18e-9, 8.29e-9, 7.71e-9, 7.02e-9, 6.21e-9])


def test_roundtrip_through_tophat_is_exact():
    """THE defining property: re-binning the un-binned template returns it."""
    cl_b = _steep_template()
    cl_ell = unbin_bandpower_template(cl_b, BIN_LO, BIN_HI, ELLS)
    for value, lo, hi in zip(cl_b, BIN_LO, BIN_HI, strict=True):
        band = (lo <= ELLS) & (hi >= ELLS)
        assert np.allclose(cl_ell[band].mean(), value, rtol=1e-12)


def test_interpolating_bin_centres_overshoots_the_narrow_bin():
    """The bug itself, pinned: centres + interp vs the correct un-binning.

    Re-binned through a top-hat, the interpolated version overshoots the bin
    after the wide one by ~10x while its neighbours come out low. If this test
    ever goes quiet, the failure mode has changed and the guard needs revisiting.
    """
    cl_b = _steep_template()
    centres = 0.5 * (BIN_LO + BIN_HI)
    interpolated = np.interp(ELLS, centres, cl_b)
    correct = unbin_bandpower_template(cl_b, BIN_LO, BIN_HI, ELLS)

    def rebin(x):
        return np.array([x[(lo <= ELLS) & (hi >= ELLS)].mean()
                         for lo, hi in zip(BIN_LO, BIN_HI, strict=True)])

    ratio = rebin(interpolated) / rebin(correct)
    assert ratio[1] > 5.0, f"expected a large overshoot in bin 1, got {ratio[1]}"
    assert ratio[0] < 0.95 and ratio[2] < 0.95
    assert np.allclose(ratio[4:], 1.0, rtol=0.1)


def test_ends_extend_by_nearest_neighbour():
    cl_b = _steep_template()
    ells = np.arange(-5, 400, dtype=float)
    out = unbin_bandpower_template(cl_b, BIN_LO, BIN_HI, ells)
    assert np.all(out[ells < BIN_LO[0]] == cl_b[0])
    assert np.all(out[ells > BIN_HI[-1]] == cl_b[-1])


def test_constant_template_is_constant_everywhere():
    """A controlled-input check that is NOT measure-zero on its own, so it is
    paired with the steep cases above."""
    cl_b = np.full(len(BIN_LO), 3.0)
    out = unbin_bandpower_template(cl_b, BIN_LO, BIN_HI, ELLS)
    assert np.allclose(out, 3.0)


def test_shape_and_ordering_validation():
    cl_b = _steep_template()
    with pytest.raises(ValueError, match="must match"):
        unbin_bandpower_template(cl_b, BIN_LO[:-1], BIN_HI, ELLS)
    with pytest.raises(ValueError, match="bin_hi must be"):
        unbin_bandpower_template(cl_b, BIN_HI, BIN_LO, ELLS)
    with pytest.raises(ValueError, match="1-D"):
        unbin_bandpower_template(cl_b[:, None], BIN_LO, BIN_HI, ELLS)


def test_bin_order_does_not_matter():
    cl_b = _steep_template()
    perm = np.array([3, 0, 5, 1, 2, 4, 6, 7, 8, 9, 10, 11])
    a = unbin_bandpower_template(cl_b, BIN_LO, BIN_HI, ELLS)
    b = unbin_bandpower_template(cl_b[perm], BIN_LO[perm], BIN_HI[perm], ELLS)
    assert np.array_equal(a, b)


def test_from_bandpowers_matches_manual_unbinning():
    cl_b = _steep_template()
    model = ResidualTemplateForegroundModel.from_bandpowers(
        cl_b, BIN_LO, BIN_HI, ELLS)
    manual = ResidualTemplateForegroundModel(
        unbin_bandpower_template(cl_b, BIN_LO, BIN_HI, ELLS), ELLS)
    params = np.array([1.0])
    assert np.allclose(model.cl_bb(100.0, 100.0, ELLS, params),
                       manual.cl_bb(100.0, 100.0, ELLS, params))


def test_from_bandpowers_does_not_warn():
    cl_b = _steep_template()
    with warnings_as_errors():
        ResidualTemplateForegroundModel.from_bandpowers(
            cl_b, BIN_LO, BIN_HI, ELLS)


def test_passing_bin_centres_warns():
    """The guard fires on exactly the mistake that caused the bug."""
    cl_b = _steep_template()
    centres = 0.5 * (BIN_LO + BIN_HI)
    with pytest.warns(UserWarning, match="looks like BANDPOWERS"):
        ResidualTemplateForegroundModel(cl_b, centres)


def test_smooth_template_on_a_coarse_grid_does_not_warn():
    """Guard must not cry wolf: coarse sampling alone is fine."""
    ells = np.arange(2, 300, 25, dtype=float)
    smooth = 1e-8 * (ells / 80.0) ** -0.4
    with warnings_as_errors():
        ResidualTemplateForegroundModel(smooth, ells)


def test_noisy_unit_grid_template_does_not_warn():
    """A big jump on a UNIT grid is not the bandpower signature."""
    rng = np.random.default_rng(0)
    ells = np.arange(2, 200, dtype=float)
    noisy = 1e-8 * rng.lognormal(sigma=1.5, size=ells.size)
    with warnings_as_errors():
        ResidualTemplateForegroundModel(noisy, ells)


def warnings_as_errors():
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            yield
    return ctx()
