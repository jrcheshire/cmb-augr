"""allocation.py — grouped focal-plane allocation → per-band beams + white noise.

A differentiable map from a small vector of per-group allocation logits to the
per-band sensitivity ``(beams, w_inv)`` that the map-based component-separation
path (:mod:`augr.compsep_sims` / :mod:`augr.nilc`) consumes. It is the design
lever for the aperture / synchrotron-bias study: hold the telescope aperture
fixed, vary how a constrained focal-plane budget is split across frequency
*groups*, and read off how σ(r) and the FG-leakage bias Δr respond.

Mechanism
---------
The channels are partitioned into ``n_groups`` frequency groups (e.g. PICO's
sync-monitor / low-CMB / CMB-core / CMB-high / dust-monitor / high-ν tiers). The
**within-group split is frozen at the reference instrument's baseline ratios**;
only the per-group *totals* are free, parameterised as a softmax over a length-
``n_groups`` logit vector so the budget constraint holds by construction:

    n_det_b(logits) = n_det_baseline_b · softmax(logits)_g(b) / φ_g(b)

where ``φ_g`` is the baseline group fraction *of the conserved budget*. At
``logits = baseline_logits`` (``= log φ``) this returns the baseline instrument
exactly, so the absolute σ(r) calibration is inherited from the reference design.

Two budgets are conserved depending on ``constraint``:

* ``"area"`` (default, the plan's choice) — total **focal-plane area** is fixed.
  Pixel cell area scales as ``1/ν²`` (feedhorn diameter ∝ λ), so the area weight
  of a band is ``n_det/ν²`` and ``φ_g ∝ Σ_{b∈g} n_det_b/ν_b²``. Moving area into a
  high-ν group buys *more* (smaller) detectors there — the physically correct
  "fixed real estate" lever.
* ``"detectors"`` — total **detector count** is fixed (``φ_g ∝ Σ_{b∈g} n_det_b``).
  This reproduces the analytic D-mirror showcase's ``n_det_from_z`` allocation and
  is kept as a cross-check.

Only ``w_inv`` (through ``n_det``) carries the logit gradient; the beams depend on
the aperture, not the allocation. The output is exactly the ``(beam_fwhm_arcmin,
w_inv)`` pair :func:`augr.compsep_sims.generate_band_sky` /
:func:`augr.compsep_sims.assemble_band_maps` expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .instrument import Instrument, white_noise_power_continuous
from .telescope import beam_fwhm_arcmin

_CONSTRAINTS = ("area", "detectors")


def _group_index_for_freqs(
    freqs_ghz: tuple[float, ...],
    groups: tuple[tuple[float, ...], ...],
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """Map each channel frequency to its group index, validating the partition.

    ``groups`` is a sequence of frequency tuples (matched to channel frequencies by
    value). Every channel must land in exactly one group; every group must be
    non-empty. Raises ``ValueError`` otherwise.
    """
    gidx = np.full(len(freqs_ghz), -1, dtype=np.int64)
    for ci, nu in enumerate(freqs_ghz):
        hits = [gi for gi, gfreqs in enumerate(groups) if any(abs(nu - f) < atol for f in gfreqs)]
        if len(hits) == 0:
            raise ValueError(f"channel frequency {nu} GHz is not in any group.")
        if len(hits) > 1:
            raise ValueError(f"channel frequency {nu} GHz matches multiple groups {hits}.")
        gidx[ci] = hits[0]
    empty = [gi for gi in range(len(groups)) if not np.any(gidx == gi)]
    if empty:
        raise ValueError(f"groups {empty} matched no instrument channels.")
    return gidx


@dataclass(frozen=True)
class GroupedAllocation:
    """Static pieces of a grouped focal-plane allocation, built once per instrument.

    Carries everything :func:`band_params` needs that does not depend on the
    allocation logits, so the per-logit forward (and its gradient) is cheap.

    Attributes
    ----------
    freqs_ghz
        Channel center frequencies [GHz], length ``n_chan``.
    group_index
        Group index ``0..n_groups-1`` per channel, shape ``(n_chan,)``.
    n_det_baseline
        Reference per-channel detector counts, shape ``(n_chan,)``.
    net, eta
        Per-channel NET [μK√s] and total efficiency, shape ``(n_chan,)``. Held at
        the reference values (allocation moves detectors, not per-detector noise).
    group_fraction_baseline
        Baseline group fraction ``φ_g`` of the conserved budget, shape
        ``(n_groups,)`` (sums to 1).
    baseline_logits
        ``log φ_g``; passing this as ``group_logits`` recovers the reference
        instrument exactly.
    mission_years, f_sky
        Survey integration time and sky fraction used in the white-noise power.
    constraint
        ``"area"`` or ``"detectors"`` (which budget is conserved).
    """

    freqs_ghz: tuple[float, ...]
    group_index: jnp.ndarray
    n_det_baseline: jnp.ndarray
    net: jnp.ndarray
    eta: jnp.ndarray
    group_fraction_baseline: jnp.ndarray
    baseline_logits: jnp.ndarray
    mission_years: float
    f_sky: float
    constraint: str

    @property
    def n_groups(self) -> int:
        return int(self.group_fraction_baseline.shape[0])

    @property
    def n_chan(self) -> int:
        return len(self.freqs_ghz)

    def n_det(self, group_logits: jax.Array) -> jax.Array:
        """Per-channel detector counts for a given length-``n_groups`` logit vector.

        ``n_det_b = n_det_baseline_b · softmax(logits)_g(b) / φ_g(b)``. Differentiable
        in ``group_logits``; conserves the budget set by ``constraint``.
        """
        frac = jax.nn.softmax(group_logits)
        multiplier = frac / self.group_fraction_baseline  # per group
        return self.n_det_baseline * multiplier[self.group_index]


def grouped_allocation(
    instrument: Instrument,
    groups: tuple[tuple[float, ...], ...],
    *,
    constraint: str = "area",
) -> GroupedAllocation:
    """Build a :class:`GroupedAllocation` from a reference instrument and grouping.

    Parameters
    ----------
    instrument
        Reference design (e.g. ``config.pico_like()``); supplies the baseline
        per-channel detector counts, NETs, efficiencies, ``f_sky``, and mission
        duration. The within-group detector ratios are frozen at these values.
    groups
        Frequency groups as a sequence of GHz tuples (matched to channel
        frequencies by value). Every channel must fall in exactly one group.
    constraint
        ``"area"`` (default) conserves total focal-plane area (band weight
        ``n_det/ν²``); ``"detectors"`` conserves total detector count (band weight
        ``n_det``).
    """
    if constraint not in _CONSTRAINTS:
        raise ValueError(f"constraint must be one of {_CONSTRAINTS}, got {constraint!r}.")

    freqs = tuple(float(ch.nu_ghz) for ch in instrument.channels)
    gidx = _group_index_for_freqs(freqs, tuple(tuple(g) for g in groups))

    n_det_base = jnp.array([float(ch.n_detectors) for ch in instrument.channels])
    net = jnp.array([float(ch.net_per_detector) for ch in instrument.channels])
    eta = jnp.array([float(ch.efficiency.total) for ch in instrument.channels])
    nu = jnp.array(freqs)

    # Per-band weight in the conserved budget: area ∝ n_det · cell_area ∝ n_det/ν².
    band_weight = n_det_base / nu**2 if constraint == "area" else n_det_base

    n_groups = len(groups)
    gidx_j = jnp.asarray(gidx)
    group_weight = jnp.zeros(n_groups).at[gidx_j].add(band_weight)
    group_fraction = group_weight / jnp.sum(group_weight)

    return GroupedAllocation(
        freqs_ghz=freqs,
        group_index=gidx_j,
        n_det_baseline=n_det_base,
        net=net,
        eta=eta,
        group_fraction_baseline=group_fraction,
        baseline_logits=jnp.log(group_fraction),
        mission_years=float(instrument.mission_duration_years),
        f_sky=float(instrument.f_sky),
        constraint=constraint,
    )


def band_params(
    alloc: GroupedAllocation,
    group_logits: jax.Array,
    aperture_m,
    *,
    illumination_factor: float = 1.22,
) -> tuple[tuple[float, ...], jax.Array, jax.Array]:
    """Allocation logits + aperture → ``(freqs_ghz, beams_arcmin, w_inv)``.

    Differentiable in ``group_logits`` (and in ``aperture_m``). The beams follow the
    diffraction limit at ``aperture_m``; ``w_inv`` is the polarization white-noise
    power [μK²·sr] per band, the beam-free input
    :func:`augr.compsep_sims.assemble_band_maps` adds as noise.

    Parameters
    ----------
    alloc
        Static allocation context from :func:`grouped_allocation`.
    group_logits
        Length-``n_groups`` logit vector; ``alloc.baseline_logits`` reproduces the
        reference instrument.
    aperture_m
        Primary mirror diameter [m]; sets every band's beam FWHM.
    illumination_factor
        ``FWHM = factor · λ/D`` (1.22 for the Airy disk).

    Returns
    -------
    ``(freqs_ghz, beams_arcmin, w_inv)`` — frequencies as a static tuple, beams and
    ``w_inv`` as JAX arrays of shape ``(n_chan,)``.
    """
    n_det = alloc.n_det(group_logits)
    beams = jnp.stack(
        [beam_fwhm_arcmin(nu, aperture_m, illumination_factor) for nu in alloc.freqs_ghz]
    )
    w_inv = white_noise_power_continuous(
        alloc.net, n_det, alloc.eta, alloc.mission_years, alloc.f_sky
    )
    return alloc.freqs_ghz, beams, w_inv
