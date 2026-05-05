"""crosslinks.py - Year-averaged spin coefficients h_k(n_hat) for L2 scans.

Produces the per-pixel "cross-link factor" h_k = <exp(-i k psi)>_pixel,
where psi is the scan-direction-vs-north angle in the pixel's local
tangent frame and the average is over hits within the pixel. This is the
load-bearing quantity for differential-systematic propagation in CMB
satellite missions (Wallis et al. 2017 arXiv:1604.02290 Eqs. 20-22 give
the B-mode bias from differential gain, pointing, and ellipticity in
terms of h_1, h_2, h_4).

For an L2 satellite with anti-sun rotating around the ecliptic at 1
rev/year, spin axis precessing around antisun at half-angle
``precession_angle_deg``, and boresight scanning around the spin axis
at half-angle ``spin_angle_deg``, the year-averaged h_k is azimuthally
symmetric around the ecliptic pole and reduces to a 1-D function
h_k(theta_ecl).

Closed form (derivation in scripts/falcons_validation/derivation_and_lit_search.md):

    h_k(theta_ecl) = (i)^k * <cos(k A)>_w

with weight (precession-Jacobian * spin-Jacobian product)

    w(theta_S) = sin(theta_S) / sqrt(D_prec * D_spin)

and integrand argument the spherical-triangle vertex angle at the
boresight in the triangle (ecliptic pole, spin axis, boresight):

    cos A = (cos(theta_S) - cos(theta_ecl) cos(spin)) / (sin(theta_ecl) sin(spin))

where:
    D_prec = sin(prec)^2 - cos(theta_S)^2
              [precession turning-point Jacobian; vanishes when the
               spin axis is at the extreme of its precession cone]
    D_spin = (sin(theta_S) sin(spin))^2 - (cos(theta_ecl) - cos(theta_S) cos(spin))^2
              [spin-circle tangent-to-parallel Jacobian; vanishes when
               the spin circle is tangent to the target colatitude]

Both vanish linearly at the support endpoints, giving 1/sqrt-distance
integrable singularities. The implementation uses a Chebyshev
substitution u = u_m + u_a cos(s) over u = cos(theta_S) so that
uniform-grid trapezoid in s in (0, pi) absorbs both singularities into
sin(s) factors and converges with O(100) points. This keeps the
quadrature differentiable under jax.grad with respect to theta_ecl,
spin_angle_deg, and precession_angle_deg.

The closed form is the **ergodic phase-space limit**; it applies when
the orbital frequencies (year, precession, spin) are mutually
incommensurate. Specific commensurate parameter choices give
quasi-periodic orbits with finite-time averages that deviate from the
ergodic limit by a few percent in the bulk and up to ~30% near the
ecliptic poles. Validated against Falcons.jl
(https://github.com/yusuke-takase/Falcons.jl) at the LiteBIRD-standard
config (precession 45 deg, spin 50 deg) to within 0.008 absolute
everywhere in the bulk for k in {1, 2, 4}; at Planck-default Falcons
preset (precession 7.5 deg, spin 85 deg) the bulk passes but pole
regions expose the non-ergodicity of that preset's near-2:1
T_year:T_prec ratio. See ``scripts/falcons_validation/`` for the full
validation pipeline.

Convention notes:

* augr ``spin_angle_deg`` and ``precession_angle_deg`` follow the
  Wallis 2017 convention used by ``augr.hit_maps.l2_hit_map``:
  spin = boresight-to-spin half-angle, precession = spin-to-antisun
  half-angle. **This is opposite the Takase 2025 / Falcons.jl
  convention, where alpha = precession opening and beta = spin
  opening.** Be careful when cross-comparing.
* h_k = <exp(-i k psi)> with psi the crossing angle east of north.
  Falcons.jl's internal psi convention turns out to be the conjugate
  of a direct east-of-north computation (verified empirically); the
  ``(+i)^k`` phase prefactor in this module matches the Falcons sign,
  not a direct east-of-north derivation. For the standard downstream
  use (|h_k|^2 in differential-bias formulas like Wallis 2017
  Eqs. 20-22), the difference is invisible. If you need the
  east-of-north sign convention, take the complex conjugate.
* The closed-form 1-D quadrature with the precession * spin Jacobian
  split appears not to be in the published CMB scan-strategy
  literature surveyed (Wallis et al. 2017, McCallum et al. 2021,
  Takase 2025, Falcons.jl, plus the pseudo-Cl asymmetric-beam
  stream). See
  ``scripts/falcons_validation/derivation_and_lit_search.md`` for the
  search trail before claiming novelty publicly.

Out of scope:

* Wallis 2017 Eqs. 20-22 propagation of h_k to the B-mode bias.
* HWP-modulated h_{n,m} for m != 0; the closed-form generalization is
  straightforward (extra HWP-phase integration) but not derived here.
* Per-detector focal-plane offsets; this module returns a single map
  for the boresight detector. Per-channel maps would require shifting
  the (theta_S, ksi) parametrization by the focal-plane angles.

Public API:
    h_k_map         - HEALPix complex map of h_k for given scan params.
    yearavg_h_k_1d  - Underlying 1-D h_k(theta_ecl), JAX-differentiable.
    pack_cos_sin    - Convert real (cos, sin) pair to complex h_k
                      (Takase 2025 convention -> Wallis convention).
"""

from __future__ import annotations

import healpy as hp
import jax.numpy as jnp
import numpy as np

__all__ = ["h_k_map", "pack_cos_sin", "yearavg_h_k_1d"]


_ALLOWED_COORDS = ("G", "E", "C")
_DEFAULT_N_QUAD = 200


def yearavg_h_k_1d(
    theta_ecl: jnp.ndarray,
    spin_angle_deg: float = 50.0,
    precession_angle_deg: float = 45.0,
    k: int = 2,
    n_quad: int = _DEFAULT_N_QUAD,
) -> jnp.ndarray:
    """Year-averaged ergodic h_k(theta_ecl), evaluated on an array of colatitudes.

    Returns the complex year-averaged spin coefficient at each input
    ecliptic colatitude. JAX-differentiable with respect to ``theta_ecl``,
    ``spin_angle_deg``, and ``precession_angle_deg`` (use ``jax.grad``).

    Args:
        theta_ecl: Ecliptic colatitude(s) in radians. Scalar or array.
        spin_angle_deg: Boresight-to-spin-axis half-angle (Wallis
            convention; equals beta in Takase / Falcons.jl).
        precession_angle_deg: Spin-axis-to-antisun half-angle (Wallis
            convention; equals alpha in Takase / Falcons.jl).
        k: Spin order (positive integer; typically 1, 2, or 4).
        n_quad: Number of Chebyshev quadrature points. 200 converges
            to ~1e-4 absolute on standard configs; raise for higher
            precision at extreme parameters.

    Returns:
        ``jnp.complex128`` array same shape as ``theta_ecl``. Pixels
        outside the support (intersection of the precession band
        ``[pi/2 - precession, pi/2 + precession]`` in theta_S with the
        spherical-triangle inequality on ``(theta_S, theta_ecl, spin)``)
        return ``NaN + 0j``.
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    spin_rad = jnp.radians(spin_angle_deg)
    prec_rad = jnp.radians(precession_angle_deg)

    theta_ecl = jnp.asarray(theta_ecl, dtype=jnp.float64)

    cos_te = jnp.cos(theta_ecl)
    sin_te = jnp.sin(theta_ecl)
    cos_b = jnp.cos(spin_rad)
    sin_b = jnp.sin(spin_rad)
    sin_a = jnp.sin(prec_rad)

    # Support endpoints in u = cos(theta_S):
    #   precession bound: theta_S in [pi/2 - prec, pi/2 + prec] -> u in [-sin_a, +sin_a]
    #   triangle bound:   |theta_ecl - spin| <= theta_S <= theta_ecl + spin
    #                     -> u in [cos(theta_ecl + spin), cos|theta_ecl - spin|]
    u_alpha_lo = -sin_a
    u_alpha_hi = sin_a
    u_beta_lo = jnp.cos(theta_ecl + spin_rad)
    u_beta_hi = jnp.cos(jnp.abs(theta_ecl - spin_rad))
    u_min = jnp.maximum(u_alpha_lo, u_beta_lo)
    u_max = jnp.minimum(u_alpha_hi, u_beta_hi)

    # Chebyshev sub: u = u_m + u_a cos(s), s in (0, pi). Both endpoint
    # 1/sqrt singularities (whether from D_prec or D_spin) are absorbed
    # into the sin(s) du/ds factor, leaving a smooth integrand in s.
    u_m = 0.5 * (u_min + u_max)
    u_a = 0.5 * (u_max - u_min)

    # Midpoint sampling avoids exact s = 0, pi.
    s = (jnp.arange(n_quad, dtype=jnp.float64) + 0.5) * jnp.pi / n_quad
    sin_s = jnp.sin(s)

    # Broadcast theta_ecl-shape to (..., n_quad) over the s axis.
    u = u_m[..., None] + u_a[..., None] * jnp.cos(s)
    sin_ts_sq = jnp.maximum(1.0 - u * u, 0.0)

    cos_te_b = cos_te[..., None]
    sin_te_b = sin_te[..., None]

    D_prec = sin_a * sin_a - u * u
    D_spin = sin_ts_sq * sin_b * sin_b - (cos_te_b - u * cos_b) ** 2

    # cos A; clip protects arccos and the rare cos_te=0 / sin_te=0 limit.
    cos_A = (u - cos_te_b * cos_b) / (sin_te_b * sin_b)
    cos_A = jnp.clip(cos_A, -1.0, 1.0)
    A = jnp.arccos(cos_A)

    # Numerator and denominator integrands. The leading factors that
    # cancel between num and den (u_a, ds) are dropped. Floor on
    # D_prec * D_spin guards against numerical underflow at the
    # midpoints closest to the boundary; integrable singularities
    # have already been absorbed by sin(s).
    safe_sqrt = jnp.sqrt(jnp.maximum(D_prec * D_spin, 1e-300))
    den_int = sin_s / safe_sqrt
    num_int = jnp.cos(k * A) * den_int

    avg_cos_kA = jnp.sum(num_int, axis=-1) / jnp.sum(den_int, axis=-1)

    h_k = ((1j) ** k) * avg_cos_kA

    # NaN out unphysical (empty-support) inputs.
    valid = u_max > u_min
    h_k = jnp.where(valid, h_k, jnp.nan + 0j)

    return h_k.astype(jnp.complex128)


def h_k_map(
    nside: int,
    spin_angle_deg: float = 50.0,
    precession_angle_deg: float = 45.0,
    k: int = 2,
    coord: str = "G",
    n_quad: int = _DEFAULT_N_QUAD,
) -> jnp.ndarray:
    """HEALPix complex map of year-averaged h_k for an L2 scan.

    Parallel structure to ``augr.hit_maps.l2_hit_map``: builds the
    map in the ecliptic frame analytically (h_k depends only on
    ecliptic colatitude after year-averaging), then optionally rotates
    to galactic or celestial via ``hp.Rotator``.

    Args:
        nside: HEALPix nside (power of 2).
        spin_angle_deg: Boresight-to-spin-axis half-angle. Default 50.0
            (LiteBIRD value). Wallis convention (NOT Takase/Falcons).
        precession_angle_deg: Spin-axis-to-antisun half-angle. Default
            45.0 (LiteBIRD).
        k: Spin order (positive integer; standard set is 1, 2, 4).
        coord: Output coordinate frame. "G" (galactic, default to match
            ``l2_hit_map``), "E" (ecliptic), or "C" (celestial / J2000).
        n_quad: Chebyshev quadrature points; 200 is plenty for the
            standard configs.

    Returns:
        ``jnp.complex128`` HEALPix map of length ``12 * nside**2``
        (RING ordering). NaN+0j at unobserved pixels (below the
        precession+triangle support intersection).
    """
    if coord not in _ALLOWED_COORDS:
        raise ValueError(f"coord={coord!r} not in {_ALLOWED_COORDS}")

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    if coord == "E":
        theta_ecl = theta
    else:
        rot = hp.Rotator(coord=[coord, "E"])
        theta_ecl, _ = rot(theta, phi)

    return yearavg_h_k_1d(
        jnp.asarray(theta_ecl),
        spin_angle_deg=spin_angle_deg,
        precession_angle_deg=precession_angle_deg,
        k=k,
        n_quad=n_quad,
    )


def pack_cos_sin(
    cos_map: np.ndarray | jnp.ndarray,
    sin_map: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Pack a real (cos, sin) pair into a complex h_k.

    Takase 2025 (Sec. 4.2 / App. A.2) and some downstream code provide
    h_k as a real-valued pair (<cos(k psi)>, <sin(k psi)>) instead of
    the complex h_k = <exp(-i k psi)>. This helper performs the
    conversion to the Wallis 2017 / Falcons.jl complex convention used
    internally by ``h_k_map``::

        h_k_complex = <cos(k psi)> - i <sin(k psi)>

    Args:
        cos_map: HEALPix map of <cos(k psi)> per pixel.
        sin_map: HEALPix map of <sin(k psi)> per pixel. Same shape as
            ``cos_map``.

    Returns:
        ``jnp.complex128`` HEALPix map of h_k.

    Raises:
        ValueError: if ``cos_map`` and ``sin_map`` shapes differ.
    """
    cm = jnp.asarray(cos_map)
    sm = jnp.asarray(sin_map)
    if cm.shape != sm.shape:
        raise ValueError(
            f"cos_map shape {cm.shape} != sin_map shape {sm.shape}"
        )
    return (cm - 1j * sm).astype(jnp.complex128)
