"""Tests for spectra.py."""

import jax.numpy as jnp
import numpy as np
import pytest
from augr.spectra import CMBSpectra


@pytest.fixture(scope="module")
def spectra():
    return CMBSpectra()


def test_load(spectra):
    """Templates load without error and cover ell 2..500."""
    assert spectra.ell_min <= 2
    assert spectra.ell_max >= 500


def test_cl_bb_r0_A1(spectra):
    """r=0, A_lens=1 returns lensing spectrum only."""
    ells = jnp.arange(2, 301)
    cl = spectra.cl_bb(ells, r=0.0, A_lens=1.0)
    cl_lens = spectra.cl_lensing(ells)
    assert jnp.allclose(cl, cl_lens, rtol=1e-5)


def test_cl_bb_r1_A0(spectra):
    """r=1, A_lens=0 returns tensor spectrum only."""
    ells = jnp.arange(2, 301)
    cl = spectra.cl_bb(ells, r=1.0, A_lens=0.0)
    cl_tensor = spectra.cl_tensor_r1(ells)
    assert jnp.allclose(cl, cl_tensor, rtol=1e-5)


def test_cl_bb_linearity(spectra):
    """cl_bb is linear in r and A_lens."""
    ells = jnp.arange(2, 301)
    cl_half = spectra.cl_bb(ells, r=0.5, A_lens=0.5)
    cl_full = spectra.cl_bb(ells, r=1.0, A_lens=1.0)
    assert jnp.allclose(2 * cl_half, cl_full, rtol=1e-5)


def test_cl_bb_positive(spectra):
    """CMB BB power is non-negative."""
    ells = jnp.arange(2, 301)
    cl = spectra.cl_bb(ells, r=0.1, A_lens=1.0)
    assert jnp.all(cl >= 0)


def test_lensing_peak_ell(spectra):
    """Lensing BB peaks around ell ~ 100-200 (recombination)."""
    ells = jnp.arange(2, 400)
    cl = spectra.cl_lensing(ells)
    peak_ell = int(ells[jnp.argmax(cl)])
    assert 80 <= peak_ell <= 250, f"Lensing BB peak at unexpected ell={peak_ell}"


def test_tensor_reion_bump(spectra):
    """Tensor BB reionization bump (ell < 15) exceeds recombination peak for r=1."""
    ells_low = jnp.arange(2, 15)
    ells_high = jnp.arange(50, 120)
    cl_tensor = spectra.cl_tensor_r1(jnp.arange(2, 301))
    reion = float(cl_tensor[:13].max())   # ell 2..14
    recomb = float(cl_tensor[48:118].max())  # ell 50..119
    assert reion > recomb, f"Reion bump ({reion:.3e}) should exceed recomb peak ({recomb:.3e})"


def test_out_of_range_returns_zero(spectra):
    """Ells outside template range return zero."""
    ells = jnp.array([0.0, 1.0, 600.0])
    cl = spectra.cl_bb(ells, r=1.0, A_lens=1.0)
    assert float(cl[0]) == 0.0
    assert float(cl[1]) == 0.0
    assert float(cl[2]) == 0.0


def test_interpolation_at_integer_ells(spectra):
    """Interpolation at exact integer ells matches stored values."""
    ells = jnp.array([50.0, 100.0, 150.0])
    cl_lens_interp = spectra.cl_lensing(ells)
    # Load raw data to cross-check
    import numpy as np
    raw = np.loadtxt("data/camb_lens_nobb.dat", comments="#")
    for i, ell in enumerate([50, 100, 150]):
        raw_val = raw[raw[:, 0] == ell, 1][0]
        assert abs(float(cl_lens_interp[i]) - raw_val) / (raw_val + 1e-30) < 1e-4
