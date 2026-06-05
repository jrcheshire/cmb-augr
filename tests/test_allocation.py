"""Gates for augr.allocation: grouped focal-plane allocation → (beams, w_inv).

Validates the budget-conserving softmax allocation: baseline recovery, the
area/detector conservation laws, frozen within-group ratios, beam ∝ 1/D, the
correct sign of the noise response to a reallocation, and differentiability in
the group logits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from augr.allocation import band_params, grouped_allocation
from augr.config import pico_like
from augr.instrument import white_noise_power

# PICO's 21 channels in the 6-tier grouping (data; mirrors the JPL-side
# PICO_TIER_FREQS the study driver passes in).
PICO_TIERS = (
    (21.0, 25.0, 30.0),
    (36.0, 43.0, 52.0, 62.0),
    (75.0, 90.0, 108.0),
    (129.0, 155.0, 186.0),
    (223.0, 268.0, 321.0, 385.0, 462.0),
    (555.0, 666.0, 799.0),
)

D_REF = 1.4  # PICO reference aperture [m]


def _alloc(constraint="area"):
    return grouped_allocation(pico_like(), PICO_TIERS, constraint=constraint)


def test_baseline_logits_recover_reference_instrument() -> None:
    """At baseline logits the allocation reproduces the reference n_det and w_inv."""
    inst = pico_like()
    alloc = grouped_allocation(inst, PICO_TIERS)
    n_det = np.asarray(alloc.n_det(alloc.baseline_logits))
    np.testing.assert_allclose(n_det, np.asarray(alloc.n_det_baseline), rtol=1e-12)

    _, _, w_inv = band_params(alloc, alloc.baseline_logits, D_REF)
    w_ref = np.array(
        [white_noise_power(ch, inst.mission_duration_years, inst.f_sky) for ch in inst.channels]
    )
    np.testing.assert_allclose(np.asarray(w_inv), w_ref, rtol=1e-12)


def test_area_constraint_conserves_focal_plane_area() -> None:
    """Total area Σ n_det/ν² is invariant under any reallocation (area constraint)."""
    alloc = _alloc("area")
    nu2 = np.asarray(alloc.n_det_baseline) * 0 + np.array(alloc.freqs_ghz) ** 2
    base_area = np.sum(np.asarray(alloc.n_det_baseline) / nu2)
    rng = np.random.default_rng(0)
    for _ in range(5):
        logits = alloc.baseline_logits + jnp.asarray(rng.normal(size=alloc.n_groups))
        area = np.sum(np.asarray(alloc.n_det(logits)) / nu2)
        np.testing.assert_allclose(area, base_area, rtol=1e-10)


def test_detector_constraint_conserves_total_count() -> None:
    """Total detector count is invariant under any reallocation (detector constraint)."""
    alloc = _alloc("detectors")
    total = float(np.sum(np.asarray(alloc.n_det_baseline)))
    rng = np.random.default_rng(1)
    for _ in range(5):
        logits = alloc.baseline_logits + jnp.asarray(rng.normal(size=alloc.n_groups))
        np.testing.assert_allclose(
            float(np.sum(np.asarray(alloc.n_det(logits)))), total, rtol=1e-10
        )


def test_within_group_ratio_frozen() -> None:
    """n_det / n_det_baseline is constant within each group for any logits."""
    alloc = _alloc("area")
    gidx = np.asarray(alloc.group_index)
    rng = np.random.default_rng(2)
    logits = alloc.baseline_logits + jnp.asarray(rng.normal(size=alloc.n_groups))
    multiplier = np.asarray(alloc.n_det(logits)) / np.asarray(alloc.n_det_baseline)
    for g in range(alloc.n_groups):
        in_g = multiplier[gidx == g]
        assert np.ptp(in_g) < 1e-10  # flat within the group


def test_beam_scales_inverse_aperture() -> None:
    """Doubling the aperture halves every beam FWHM."""
    alloc = _alloc()
    _, beams_d, _ = band_params(alloc, alloc.baseline_logits, D_REF)
    _, beams_2d, _ = band_params(alloc, alloc.baseline_logits, 2 * D_REF)
    np.testing.assert_allclose(np.asarray(beams_2d), np.asarray(beams_d) / 2.0, rtol=1e-12)


def test_reallocation_lowers_noise_in_favoured_group() -> None:
    """Raising one group's logit adds detectors there → lower w_inv in its bands."""
    alloc = _alloc("area")
    gidx = np.asarray(alloc.group_index)
    _, _, w_base = band_params(alloc, alloc.baseline_logits, D_REF)
    bumped = alloc.baseline_logits.at[2].add(1.0)  # favour the CMB-core group
    _, _, w_bumped = band_params(alloc, bumped, D_REF)
    in_g = gidx == 2
    assert np.all(np.asarray(w_bumped)[in_g] < np.asarray(w_base)[in_g])
    assert np.all(np.asarray(w_bumped)[~in_g] > np.asarray(w_base)[~in_g])  # others lose budget


def test_w_inv_differentiable_in_logits() -> None:
    """jax.grad of a scalar of w_inv w.r.t. the group logits is finite and nonzero."""
    alloc = _alloc("area")

    def loss(logits):
        _, _, w_inv = band_params(alloc, logits, D_REF)
        return jnp.sum(1.0 / w_inv)  # inverse-variance "sensitivity" proxy

    g = jax.grad(loss)(alloc.baseline_logits)
    g = np.asarray(g)
    assert np.all(np.isfinite(g))
    assert np.any(np.abs(g) > 0)


def test_grouping_validation() -> None:
    """Unmatched channels and empty groups raise."""
    inst = pico_like()
    with pytest.raises(ValueError, match="not in any group"):
        grouped_allocation(inst, PICO_TIERS[:-1])  # drops the high-ν tier
    with pytest.raises(ValueError, match="matched no instrument channels"):
        grouped_allocation(inst, (*PICO_TIERS, (1000.0, 1200.0)))  # phantom group
