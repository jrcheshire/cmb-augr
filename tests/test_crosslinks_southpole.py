"""Tests for ``augr.crosslinks_southpole``."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr._chi2alpha import chi2alpha
from augr.crosslinks_southpole import h_k_boresight


# -- single deck: closed form match --------------------------------------------

@pytest.mark.parametrize("deck", [0.0, 45.0, 113.0, -68.0, 200.0])
@pytest.mark.parametrize("chi", [0.0, 11.0, -45.0])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_single_deck_closed_form(deck, chi, k):
    """Single-deck h_k = exp(-i k alpha)."""
    decks = jnp.array([deck])
    actual = complex(h_k_boresight(decks, chi_deg=chi, k=k))
    alpha_rad = np.deg2rad(-90.0 + chi + deck)
    expected = np.exp(-1j * k * alpha_rad)
    assert np.isclose(actual, expected, atol=1e-12)


# -- BK pipeline cross-check via chi2alpha -------------------------------------

def test_consistency_with_chi2alpha_r0():
    """h_k_boresight must use the same alpha as chi2alpha at r=0."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    chi = 17.0
    k = 2

    # chi2alpha at r=0 with thetaref = each deck angle.
    alpha_per_deck = jnp.array([
        float(chi2alpha(0.0, -55.0, 0.0, 0.0, chi, float(d))) for d in decks
    ])
    expected = complex(np.mean(np.exp(-1j * k * np.deg2rad(np.asarray(alpha_per_deck)))))

    actual = complex(h_k_boresight(decks, chi_deg=chi, k=k))
    assert np.isclose(actual, expected, atol=1e-12)


# -- symmetry: 4 evenly spaced decks kill k = 1, 2, 3 --------------------------

@pytest.mark.parametrize("k,expected_zero", [(1, True), (2, True), (3, True), (4, False)])
def test_uniform_4deck_symmetry(k, expected_zero):
    """4 equally-weighted decks at {0, 90, 180, 270} null h_1, h_2, h_3 but
    leave |h_4| = 1 (all four 4*alpha values land on the same angle mod 2 pi)."""
    decks = jnp.array([0.0, 90.0, 180.0, 270.0])
    h = complex(h_k_boresight(decks, chi_deg=0.0, k=k))
    if expected_zero:
        assert abs(h) < 1e-12, f"k={k}: expected ~0, got {h}"
    else:
        assert np.isclose(abs(h), 1.0, atol=1e-12), f"k={k}: expected |h|=1, got {abs(h)}"


# -- Schwarz inequality --------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_modulus_bounded_by_one(k):
    """|h_k| <= 1 for any deck distribution (Schwarz / Jensen)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 12))
        decks = jnp.asarray(rng.uniform(-180, 180, size=n))
        weights = jnp.asarray(rng.uniform(0.1, 5.0, size=n))
        h = complex(h_k_boresight(decks, weights=weights, k=k))
        assert abs(h) <= 1.0 + 1e-12, f"|h|={abs(h)} > 1"


# -- chi acts as a pure phase --------------------------------------------------

@pytest.mark.parametrize("chi", [0.0, 17.0, -45.0, 90.0])
def test_chi_is_pure_phase(chi):
    """|h_k| is invariant under chi; chi only rotates the complex phase."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    h0 = complex(h_k_boresight(decks, chi_deg=0.0, k=2))
    hc = complex(h_k_boresight(decks, chi_deg=chi, k=2))
    assert np.isclose(abs(h0), abs(hc), atol=1e-12)
    # Phase shift should be exactly -2 * chi (in degrees), modulo 360.
    if abs(h0) > 1e-12:
        dphi_deg = (np.angle(hc, deg=True) - np.angle(h0, deg=True)) % 360.0
        expected = (-2.0 * chi) % 360.0
        assert np.isclose(dphi_deg, expected, atol=1e-9)


# -- weights normalisation -----------------------------------------------------

def test_weights_normalisation():
    """Unnormalised weights must give the same result as normalised ones."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])
    raw_weights = jnp.array([2.0, 3.0, 5.0, 7.0])
    norm_weights = raw_weights / raw_weights.sum()
    h_raw = complex(h_k_boresight(decks, weights=raw_weights, k=2))
    h_norm = complex(h_k_boresight(decks, weights=norm_weights, k=2))
    assert np.isclose(h_raw, h_norm, atol=1e-12)


# -- 5-line direct MC validation ----------------------------------------------

def test_against_direct_mc():
    """Closed form vs. drawing samples from the deck distribution and
    averaging exp(-i k alpha) directly. The two should agree to MC
    sampling noise (~1/sqrt(N))."""
    decks = np.array([68.0, 113.0, 248.0, 293.0])
    weights_raw = np.array([2.0, 3.0, 5.0, 7.0])
    weights = weights_raw / weights_raw.sum()
    chi = 11.0
    k = 2

    # Direct MC: draw N samples from the discrete deck distribution, compute
    # alpha at each via the chi2alpha r=0 closed form, accumulate the
    # exp(-i k alpha) average.
    rng = np.random.default_rng(0)
    n = 200_000
    sampled_decks = rng.choice(decks, size=n, p=weights)
    alpha_rad = np.deg2rad(-90.0 + chi + sampled_decks)
    h_mc = np.mean(np.exp(-1j * k * alpha_rad))

    h_closed = complex(h_k_boresight(jnp.asarray(decks),
                                     weights=jnp.asarray(weights),
                                     chi_deg=chi, k=k))

    # Discrete-distribution MC has zero bias; the only error is from finite N.
    # For 200k samples and |h| <= 1, the per-sample variance is at most 1, so
    # rms error ~ 1/sqrt(N) ~ 2e-3.
    assert abs(h_mc - h_closed) < 5e-3, (
        f"closed form {h_closed} vs MC {h_mc}: |diff|={abs(h_mc - h_closed):.3e}"
    )


# -- JAX differentiability ----------------------------------------------------

def test_grad_through_weights():
    """|h_k|^2 should be jax.grad-able through weights."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])

    def loss(weights):
        h = h_k_boresight(decks, weights=weights, chi_deg=0.0, k=2)
        return jnp.real(h * jnp.conj(h))

    g = jax.grad(loss)(jnp.array([2.0, 3.0, 5.0, 7.0]))
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


def test_grad_through_decks():
    """|h_k|^2 differentiable in deck angles (use case: schedule design)."""
    decks = jnp.array([68.0, 113.0, 248.0, 293.0])

    def loss(d):
        h = h_k_boresight(d, chi_deg=0.0, k=4)
        return jnp.real(h * jnp.conj(h))

    g = jax.grad(loss)(decks)
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


# -- input validation ----------------------------------------------------------

def test_k_must_be_positive_integer():
    decks = jnp.array([0.0, 90.0])
    with pytest.raises(ValueError):
        h_k_boresight(decks, k=0)
    with pytest.raises(ValueError):
        h_k_boresight(decks, k=-1)
    with pytest.raises(ValueError):
        h_k_boresight(decks, k=1.5)
