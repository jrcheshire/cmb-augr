"""Gates for augr.cleaning: the Cleaner / CleanerResult protocol + factory adapters.

The protocol-conformance checks need no ducc0 (they inspect a constructed
NILCResult). The factory-parity checks run a tiny real clean, so they importorskip
ducc0 and assert the adapter is bit-identical to calling the cleaner directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.cleaning import CleanerResult, cmilc_cleaner, nilc_cleaner
from augr.nilc import NILCResult

NSIDE, LMAX = 32, 48
FREQS = (30.0, 90.0, 150.0)
BEAMS = (40.0, 20.0, 10.0)
PEAKS = [8, 24, LMAX]


def _dummy_nilc_result() -> NILCResult:
    """A NILCResult with placeholder arrays (no clean run → no ducc0 needed)."""
    return NILCResult(
        cleaned_b_alm=jnp.zeros(3),
        weights=jnp.zeros((2, 3)),
        needlet_bands=jnp.zeros((2, LMAX + 1)),
        beam_fwhm_arcmin=BEAMS,
        common_fwhm_arcmin=10.0,
        lmax=LMAX,
        nside=NSIDE,
        n_iter=3,
    )


def test_nilc_result_satisfies_cleaner_result_protocol() -> None:
    assert isinstance(_dummy_nilc_result(), CleanerResult)


def test_incomplete_object_is_not_a_cleaner_result() -> None:
    class Partial:  # has the data fields but no .project method
        cleaned_b_alm = None
        lmax = nside = n_iter = 0
        beam_fwhm_arcmin = ()
        common_fwhm_arcmin = 0.0

    assert not isinstance(Partial(), CleanerResult)


def test_factories_return_callables() -> None:
    assert callable(nilc_cleaner())
    assert callable(cmilc_cleaner(FREQS))


def _random_band_qu(seed: int = 0):
    npix = 12 * NSIDE * NSIDE
    return jax.random.normal(jax.random.PRNGKey(seed), (len(BEAMS), 2, npix))


def test_nilc_cleaner_matches_nilc_clean() -> None:
    pytest.importorskip("ducc0")
    from augr.nilc import nilc_clean

    band_qu = _random_band_qu()
    direct = nilc_clean(band_qu, BEAMS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    via = nilc_cleaner(needlet_peaks=PEAKS)(band_qu, BEAMS, lmax=LMAX, nside=NSIDE)
    assert isinstance(via, CleanerResult)
    np.testing.assert_array_equal(np.asarray(via.cleaned_b_alm), np.asarray(direct.cleaned_b_alm))


def test_cmilc_cleaner_matches_cmilc_clean() -> None:
    pytest.importorskip("ducc0")
    from augr.cmilc import cmilc_clean

    band_qu = _random_band_qu()
    direct = cmilc_clean(band_qu, BEAMS, FREQS, lmax=LMAX, nside=NSIDE, needlet_peaks=PEAKS)
    via = cmilc_cleaner(FREQS, needlet_peaks=PEAKS)(band_qu, BEAMS, lmax=LMAX, nside=NSIDE)
    assert isinstance(via, CleanerResult)
    np.testing.assert_array_equal(np.asarray(via.cleaned_b_alm), np.asarray(direct.cleaned_b_alm))
