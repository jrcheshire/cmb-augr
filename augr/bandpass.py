"""
bandpass.py — Per-band instrument bandpass representation.

A ``Bandpass`` is a frequency grid + response weights describing how a detector
band integrates emission across frequency. It feeds two consumers:

1. **The forward foreground sky** (``compsep_sims``): the grid + weights are
   handed to ``pysm3.Sky.get_emission(nu_grid, weights=...)`` so the simulated
   maps are bandpass-integrated rather than evaluated at a single band center.
2. **cMILC color-corrected SEDs** (``units.color_correct`` → ``cmilc``): the
   foreground deprojection SED columns are band-averaged over the same bandpass,
   so cMILC deprojects the *effective* (color-corrected) SED that the sky
   actually presents.

Weighting convention: ``weights`` is the raw bandpass response on ``nu_ghz``;
the Rayleigh-Jeans-power normalization that PySM applies internally
(``pysm3.utils.normalize_weights`` / ``bandpass_unit_conversion``) is reproduced
on the analytic side in ``units.color_correct`` — see that function for the
kernel and its PySM cross-check.

Differentiability scope. ``smooth_tophat`` builds the grid on a *fixed relative*
axis ``xi`` scaled by ``nu_center`` (``nu = nu_center * xi``) with smooth
(sigmoid-edged) weights, so a band-averaged SED is differentiable w.r.t. band
center and fractional bandwidth — ``d(SED_band)/d(nu_center)`` flows in JAX.
This delivers a differentiable color-correction *primitive*; it does NOT by
itself make end-to-end map-based component-separation optimization
differentiable (the MC spectrum stage + Fisher reporting are forward-only).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Default quadrature resolution. The forward sky evaluates a full PySM emission
# map at every grid point, so a top-hat defaults to a modest count typical of
# bandpass integration (convergence is cheap to widen per call). The photon-noise
# NET grid in telescope.py keeps its own (denser) n_quad=512 — unrelated path.
N_QUAD_TOPHAT = 16
N_QUAD_SMOOTH = 64

# Static relative half-extent of the smooth_tophat grid (xi in [1-W, 1+W]).
# Fixed (not derived from fractional_bandwidth) so the grid stays put while the
# band edges move through the weights — the key to clean frac_bw gradients.
SMOOTH_GRID_HALF_WIDTH = 0.5


@dataclass(frozen=True)
class Bandpass:
    """A per-band frequency response.

    Attributes:
        nu_ghz: Absolute frequency grid [GHz], shape ``(n_quad,)``.
        weights: Bandpass response on ``nu_ghz``, shape ``(n_quad,)`` (raw; not
            pre-normalized — ``units.color_correct`` applies the RJ-power norm).
        nu_center_ghz: Nominal band center [GHz] (scalar; may be a JAX tracer
            when built differentiably).

    A single-point grid (``n_quad == 1``) is the monochromatic / delta-function
    limit and reproduces the legacy band-center evaluation exactly.
    """

    nu_ghz: jax.Array
    weights: jax.Array
    nu_center_ghz: jax.Array | float

    @property
    def is_monochromatic(self) -> bool:
        """True for a single-point (delta-function) grid."""
        return self.nu_ghz.shape[0] == 1

    @classmethod
    def monochromatic(cls, nu_center_ghz: jax.Array | float) -> Bandpass:
        """Delta-function bandpass at ``nu_center_ghz`` (legacy band-center limit)."""
        nu = jnp.atleast_1d(jnp.asarray(nu_center_ghz, dtype=float))
        return cls(nu_ghz=nu, weights=jnp.ones_like(nu), nu_center_ghz=nu_center_ghz)

    @classmethod
    def tophat(
        cls,
        nu_center_ghz: jax.Array | float,
        fractional_bandwidth: float,
        n_quad: int = N_QUAD_TOPHAT,
    ) -> Bandpass:
        """Uniform top-hat over ``[nu_c(1 - f/2), nu_c(1 + f/2)]``.

        Matches the band-edge convention of the photon-noise NET grid
        (``telescope.photon_noise_net``). ``fractional_bandwidth <= 0`` returns
        the monochromatic limit. The grid scales with ``nu_center_ghz`` so
        ``d/d(nu_center)`` flows; the edges move with ``fractional_bandwidth``.
        This is the forward-sky default.
        """
        if fractional_bandwidth <= 0.0:
            return cls.monochromatic(nu_center_ghz)
        half = 0.5 * fractional_bandwidth
        xi = jnp.linspace(1.0 - half, 1.0 + half, n_quad)
        nu = jnp.asarray(nu_center_ghz, dtype=float) * xi
        return cls(nu_ghz=nu, weights=jnp.ones(n_quad), nu_center_ghz=nu_center_ghz)

    @classmethod
    def smooth_tophat(
        cls,
        nu_center_ghz: jax.Array | float,
        fractional_bandwidth: jax.Array | float,
        *,
        edge_softness: float = 0.05,
        n_quad: int = N_QUAD_SMOOTH,
        grid_half_width: float = SMOOTH_GRID_HALF_WIDTH,
    ) -> Bandpass:
        """Differentiable smooth-edged top-hat on a fixed relative grid.

        The grid ``xi = linspace(1 - W, 1 + W, n_quad)`` is *static* (``W =
        grid_half_width``, ``n_quad`` fixed), and ``nu = nu_center * xi``. The
        band is encoded entirely in smooth weights — a product of two opposed
        sigmoids with edge width ``edge_softness * fractional_bandwidth`` — so
        both ``nu_center`` and ``fractional_bandwidth`` may be JAX tracers and
        ``d(SED_band)/d(nu_center)``, ``d(SED_band)/d(frac_bw)`` are finite
        everywhere (no grid point ever crosses a hard cutoff). Use this when the
        bandpass is a design variable. ``grid_half_width`` must comfortably
        exceed ``fractional_bandwidth / 2`` plus a few edge widths.
        """
        xi = jnp.linspace(1.0 - grid_half_width, 1.0 + grid_half_width, n_quad)
        half = 0.5 * fractional_bandwidth
        s = edge_softness * fractional_bandwidth
        weights = jax.nn.sigmoid((xi - (1.0 - half)) / s) * jax.nn.sigmoid(((1.0 + half) - xi) / s)
        nu = jnp.asarray(nu_center_ghz, dtype=float) * xi
        return cls(nu_ghz=nu, weights=weights, nu_center_ghz=nu_center_ghz)

    @classmethod
    def from_profile(
        cls,
        nu_ghz: jax.Array,
        weights: jax.Array,
        nu_center_ghz: jax.Array | float | None = None,
    ) -> Bandpass:
        """Explicit measured / asymmetric profile.

        ``nu_center_ghz`` defaults to the response-weighted mean frequency.
        """
        nu = jnp.asarray(nu_ghz, dtype=float)
        w = jnp.asarray(weights, dtype=float)
        if nu_center_ghz is None:
            nu_center_ghz = jnp.sum(nu * w) / jnp.sum(w)
        return cls(nu_ghz=nu, weights=w, nu_center_ghz=nu_center_ghz)
