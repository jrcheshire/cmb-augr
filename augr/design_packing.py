"""design_packing.py -- z-space reparam for the horn-packing instrument design.

Maps a flat standardized vector ``z`` to the physical telescope design knobs the EIG
objective consumes (:func:`augr.eig.physical_design_objective`), enforcing the
constraints the focal-plane-allocation tradespace requires:

- **Area allocation is a simplex.** A fixed total cold focal-plane area is allocated
  across pixel groups, so the per-group fractions must sum to 1. We softmax per-group
  logits with one group held as a gauge reference (its logit is fixed), giving
  ``n_groups - 1`` identifiable free allocation knobs.
- **f-number is bounded** to a buildable range via a smooth sigmoid into
  ``[f_min, f_max]``. The sigmoid gradient vanishes at the rails, so a railed f# reads
  as *low activity* in the active subspace -- the honest behavior -- whereas a hard clip
  would break differentiability at the boundary.
- **Aperture and mission years are positive**, so they use log (multiplicative)
  standardization ``xi = xi_fid * exp(z)``.
- **fp_diameter (the total focal-plane area) is fixed** -- the "spend a fixed cold focal
  plane" story -- so it is design context, not a knob.

``z = 0`` maps to the fiducial design. :meth:`design_pytree` returns the dict
``{aperture_m, f_number, area_fractions, mission_years}``; ``n_dim`` / ``knob_labels``
feed the active-subspace machinery (:mod:`augr.active_subspace`) exactly as
:class:`augr.active_subspace.DesignSpec` does for the simpler log/affine knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def _sigmoid(x):
    return 0.5 * (jnp.tanh(0.5 * x) + 1.0)


@dataclass(frozen=True)
class PackingDesignSpec:
    """Fiducial horn-packing design + the ``z`` <-> design-pytree reparam.

    Attributes:
        freqs_per_group: Per-group frequency tuples (1 or 2 bands each), e.g.
            ``((20.,), (35.,), (80., 115.), (160., 225.), (315., 440.), (615.,))``.
        frac_fid:        Fiducial focal-plane area fractions, ``(n_groups,)``, summing to 1.
        aperture_fid:    Fiducial aperture [m].
        f_number_fid:    Fiducial focal ratio (must lie strictly inside ``f_bounds``).
        years_fid:       Fiducial mission duration [yr].
        fp_diameter_m:   Fixed usable focal-plane diameter [m] (sets the total area).
        f_bounds:        ``(f_min, f_max)`` buildable f-number range.
        ref_group:       Index of the gauge-fixed allocation reference group.
        eta_total:       Total optical efficiency carried to the objective.
    """

    freqs_per_group: tuple[tuple[float, ...], ...]
    frac_fid: np.ndarray
    aperture_fid: float
    f_number_fid: float
    years_fid: float
    fp_diameter_m: float
    f_bounds: tuple[float, float] = (1.4, 3.0)
    ref_group: int = 0
    eta_total: float = 0.5

    def __post_init__(self):
        f = np.asarray(self.frac_fid, dtype=float)
        if f.ndim != 1 or f.size != len(self.freqs_per_group):
            raise ValueError("frac_fid must be 1-D with one entry per pixel group.")
        if not np.isclose(f.sum(), 1.0):
            raise ValueError(f"frac_fid must sum to 1, got {f.sum():.6f}.")
        f_lo, f_hi = self.f_bounds
        if not (f_lo < self.f_number_fid < f_hi):
            raise ValueError(
                f"f_number_fid {self.f_number_fid} must lie strictly inside {self.f_bounds}."
            )
        if not (0 <= self.ref_group < f.size):
            raise ValueError(
                f"ref_group {self.ref_group} out of range for {f.size} groups."
            )
        object.__setattr__(self, "frac_fid", f)  # store the normalized float array

    @property
    def n_groups(self) -> int:
        return len(self.freqs_per_group)

    @property
    def n_dim(self) -> int:
        # (n_groups - 1) free allocation logits + log aperture + z_f + log mission years.
        return (self.n_groups - 1) + 3

    @property
    def free_groups(self) -> tuple[int, ...]:
        return tuple(g for g in range(self.n_groups) if g != self.ref_group)

    @property
    def knob_labels(self) -> tuple[str, ...]:
        def _lab(g: int) -> str:
            return "+".join(f"{f:.0f}" for f in self.freqs_per_group[g])

        alloc = tuple(f"alloc@{_lab(g)}" for g in self.free_groups)
        return (*alloc, "aperture", "f_number", "mission_years")

    @property
    def _f_z0(self) -> float:
        """z_f offset so ``f_number(z_f=0) == f_number_fid`` (inverse sigmoid)."""
        f_lo, f_hi = self.f_bounds
        u = (self.f_number_fid - f_lo) / (f_hi - f_lo)
        return float(np.log(u / (1.0 - u)))

    def design_pytree(self, z) -> dict:
        """Standardized ``z`` -> ``{aperture_m, f_number, area_fractions, mission_years}``.

        ``z`` layout: ``[free allocation logits (n_groups - 1), log aperture, z_f,
        log mission_years]``. Differentiable; ``z = 0`` returns the fiducial design.
        """
        z = jnp.asarray(z)
        n_alloc = self.n_groups - 1
        z_alloc = z[:n_alloc]
        z_ap, z_f, z_yr = z[n_alloc], z[n_alloc + 1], z[n_alloc + 2]

        # Softmax allocation: base logits are the fiducial log-fractions (softmax(log p) = p
        # when sum(p) = 1), so z = 0 reproduces frac_fid exactly. Only the free groups move;
        # the reference group's logit is gauge-fixed.
        logits = jnp.log(jnp.asarray(self.frac_fid))
        logits = logits.at[jnp.asarray(self.free_groups)].add(z_alloc)
        fracs = jax.nn.softmax(logits)

        f_lo, f_hi = self.f_bounds
        f_number = f_lo + (f_hi - f_lo) * _sigmoid(z_f + self._f_z0)
        aperture = self.aperture_fid * jnp.exp(z_ap)
        years = self.years_fid * jnp.exp(z_yr)
        return {
            "aperture_m": aperture,
            "f_number": f_number,
            "area_fractions": fracs,
            "mission_years": years,
        }

    @property
    def freqs_flat(self) -> tuple[float, ...]:
        """Channel frequencies in flattened-group order (matches design_to_channels)."""
        return tuple(float(f) for grp in self.freqs_per_group for f in grp)
