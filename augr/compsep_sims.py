"""compsep_sims.py — per-band sky + noise sims for the differentiable NILC.

Builds the per-frequency-band Q/U maps a blind map-based component separation
sees: a beam-smoothed sky (CMB B-mode realization + PySM Galactic foregrounds)
plus instrument noise. The split between a *fixed* beamed sky and a
*differentiable* noise term is the load-bearing design choice for the
common-random-number (CRN) gradients used downstream:

* The beamed sky depends on the aperture ``D`` (through the per-band beam) and
  the fixed sky realization, but **not** on the focal-plane allocation. So it is
  computed once (with numpy / healpy), stored in :class:`BandSky`, and reused
  across allocations.
* The noise depends on the allocation through the per-band white-noise power
  ``w_inv`` (more detectors in a band → lower ``w_inv``). It is added at
  evaluation time via :func:`assemble_band_maps`, which is differentiable in
  ``w_inv`` under fixed random numbers (Stage 1 :func:`augr.noise_sims.noise_maps`).

This keeps the σ(r) / Δr gradient flowing only through the noise term, which is
the physically meaningful lever (the sky is held fixed across allocations).

Conventions
-----------
* **B-only, full-sky v1.** The CMB is a Gaussian B-mode realization drawn from
  ``C_ℓ^{BB}(r) = r·C_ℓ^{tensor} + C_ℓ^{lensing}`` (E set to zero). Full-sky with
  no mask means E and B do not mix, so the CMB E-modes are irrelevant to a B-mode
  NILC and are omitted. The lensing BB is treated as a Gaussian field with the
  right ``C_ℓ`` — the standard power-spectrum-level sim approximation; it does not
  capture lensing non-Gaussianity or the E/φ correlation (out of scope for v1).
* **Q/U are HEALPix-internal** (not IAU); negate U at a boundary if an IAU
  consumer needs it. The CMB and the foregrounds are transformed with the same
  beam/SHT path, so the B-mode is internally self-consistent regardless of the
  absolute U sign.
* **Beam.** Each band is smoothed by ``instrument.beam_bl(ℓ, FWHM_band)`` applied
  identically to E and B in harmonic space (CMB) or after a healpy ``map2alm``
  analysis (PySM foregrounds). The foreground analysis uses healpy quadrature
  (non-differentiable) because the sky is allocation-independent; the NILC's own
  map→alm step (Stage 3) is the differentiable one.
* **Units.** μK_CMB throughout. PySM emission (μK_RJ) is converted with
  ``pysm3.units.cmb_equivalencies`` at each band center.

Requires the optional ``[compsep]`` extra (``pysm3``) only for the foreground
path; the CMB-only and noise paths need just healpy + ducc0.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .instrument import beam_bl
from .noise_sims import noise_maps
from .sht import almxfl, check_band_limit, synthesis
from .spectra import CMBSpectra

# PySM preset-string combinations. d1s1 is the simple baseline (matches the
# Carones 2025 / BROOM driver). d10 is the GNILC-2023 modified-blackbody dust.
# For synchrotron, s5 is a fixed power-law template with a spatially-varying
# index but NO injected small-scale power -- its polarization morphology is
# smooth below the ~1 deg WMAP-K resolution that sets the angle template.
# s6 (PowerLawRealization, PySM 3.4 / arXiv:2502.20452) injects stochastic
# small scales to high ell; it is the model to use when the small-scale
# synchrotron structure is itself under test (e.g. the aperture / D_min study,
# where a coarse low-nu beam must fail to resolve real small-scale synch).
_FG_PRESETS: dict[str, tuple[str, ...]] = {
    "d1s1": ("d1", "s1"),
    "d10s5": ("d10", "s5"),
    "d10s6": ("d10", "s6"),
}


def _require_pysm():
    """Import pysm3 or raise a helpful error pointing at the [compsep] extra."""
    try:
        import pysm3
        import pysm3.units as u

        return pysm3, u
    except ImportError as exc:  # pragma: no cover - exercised only without pysm3
        raise ImportError(
            "augr.compsep_sims foreground generation requires 'pysm3', which "
            "ships with the component-separation extra. Install it with:\n"
            "    pip install 'cmb-augr[compsep]'\n"
            "or, in the development env:\n"
            "    pixi add pysm3"
        ) from exc


# ---------------------------------------------------------------------------
# fixed (allocation-independent) beamed sky
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandSky:
    """Beam-smoothed, allocation-independent per-band sky maps [μK_CMB].

    Holds the fixed pieces of a CRN sim: the CMB B-mode realization and the PySM
    foregrounds, each beam-smoothed per band. Noise is added separately by
    :func:`assemble_band_maps`. ``cmb_qu`` and ``fg_qu`` double as the CMB-only
    and FG-only maps needed for the Stage-4 transfer / leakage projections.

    Attributes
    ----------
    freqs_ghz
        Band center frequencies [GHz], length ``n_band``.
    beam_fwhm_arcmin
        Per-band Gaussian beam FWHM [arcmin], length ``n_band``. Set by the
        aperture ``D`` upstream.
    nside, lmax
        HEALPix resolution and band limit of the maps / transforms.
    r_in
        Input tensor-to-scalar ratio of the CMB realization.
    cmb_qu
        Beamed CMB Q/U, shape ``(n_band, 2, npix)``.
    fg_qu
        Beamed foreground Q/U, shape ``(n_band, 2, npix)``; all-zero when no
        foreground model was requested.
    """

    freqs_ghz: tuple[float, ...]
    beam_fwhm_arcmin: tuple[float, ...]
    nside: int
    lmax: int
    r_in: float
    cmb_qu: jax.Array
    fg_qu: jax.Array

    @property
    def n_band(self) -> int:
        return len(self.freqs_ghz)

    @property
    def npix(self) -> int:
        return 12 * self.nside**2


def cmb_b_alm(spectra: CMBSpectra, r_in: float, lmax: int, *, seed: int = 0) -> jax.Array:
    """Draw a Gaussian CMB B-mode alm from ``C_ℓ^{BB}(r_in)`` [healpy packing].

    The realization is fixed by ``seed`` (CRN). Returns a 1-D complex alm of
    length ``alm_size(lmax)``.
    """
    import healpy as hp

    ells = np.arange(lmax + 1)
    cl_bb = np.asarray(spectra.cl_bb(jnp.asarray(ells, dtype=float), float(r_in)))
    cl_bb = np.clip(cl_bb, 0.0, None)  # guard tiny negative interpolation undershoot
    np.random.seed(int(seed) & 0xFFFFFFFF)  # noqa: NPY002 - healpy.synalm uses the global RNG
    b_alm = hp.synalm(cl_bb, lmax=lmax, new=True)
    return jnp.asarray(b_alm)


def _beam_qu_from_eb(
    alm_e: jax.Array, alm_b: jax.Array, fwhm_arcmin: float, lmax: int, nside: int
) -> jax.Array:
    """E/B alm → beam-smoothed Q/U map, shape ``(2, npix)`` [HEALPix-internal]."""
    bl = beam_bl(jnp.arange(lmax + 1, dtype=float), fwhm_arcmin)
    e_beamed = almxfl(jnp.asarray(alm_e), bl, lmax)
    b_beamed = almxfl(jnp.asarray(alm_b), bl, lmax)
    return synthesis(jnp.stack([e_beamed, b_beamed], axis=0), 2, lmax, nside)


def cmb_band_qu(
    b_alm: jax.Array, beam_fwhm_arcmin: tuple[float, ...], lmax: int, nside: int
) -> jax.Array:
    """Beam the shared CMB B-mode alm per band → ``(n_band, 2, npix)`` Q/U."""
    zero_e = jnp.zeros_like(b_alm)
    return jnp.stack(
        [_beam_qu_from_eb(zero_e, b_alm, fw, lmax, nside) for fw in beam_fwhm_arcmin],
        axis=0,
    )


def _seeded_component_config(fg_model: str, fg_seed: int) -> dict:
    """Build a PySM ``component_config`` for ``fg_model`` with seeded small scales.

    Stochastic small-scale components (PySM ``*Realization`` classes, e.g. ``s6``
    = ``PowerLawRealization``) reseed ``numpy``'s global RNG from entropy at
    construction when their ``seeds`` field is left ``None`` (the preset default).
    That makes the foreground realization both non-reproducible and not held fixed
    across apertures, which breaks the common-random-number assumption the Δr(D)
    sweep relies on (the CMB and noise are CRN'd by ``cmb_seed`` / the JAX key, so
    the foreground must be too). We therefore deep-copy each preset dict and inject
    an explicit ``seeds`` list for any ``*Realization`` component, keyed off
    ``fg_seed``. Non-realization components (fixed-template ``PowerLaw`` /
    ``ModifiedBlackBody`` such as ``d1``/``d10``/``s1``/``s5``) take no ``seeds``
    kwarg, so they are passed through untouched and are deterministic regardless of
    ``fg_seed``.
    """
    from pysm3.sky import PRESET_MODELS

    seed0 = int(fg_seed) % (2**31)  # keep inside numpy's [0, 2**32) seed range
    config: dict[str, dict] = {}
    for k, preset in enumerate(_FG_PRESETS[fg_model]):
        cfg = copy.deepcopy(PRESET_MODELS[preset])
        if "Realization" in cfg.get("class", ""):
            # Offset per component so two stochastic components never share a seed
            # (none do in the current presets, but a future dust-realization preset
            # would). PowerLawRealization reads seeds[0:2]; a 3rd is for a future
            # ModifiedBlackBodyRealization and is harmlessly ignored otherwise.
            base = (seed0 + 1000 * k) % (2**31)
            cfg["seeds"] = [base, base + 1, base + 2]
        config[preset] = cfg
    return config


def pysm_fg_iqu(
    freqs_ghz: tuple[float, ...], fg_model: str, nside: int, *, fg_seed: int = 0
) -> np.ndarray:
    """PySM Galactic foreground I/Q/U per band [μK_CMB], shape ``(n_band, 3, npix)``.

    Maps are at native HEALPix resolution (unbeamed); beaming happens in
    :func:`fg_band_qu`. ``fg_model`` is a key of :data:`_FG_PRESETS`. ``fg_seed``
    fixes any stochastic small-scale realization (e.g. ``s6``) for reproducibility
    and common random numbers; see :func:`_seeded_component_config`.
    """
    if fg_model not in _FG_PRESETS:
        raise ValueError(f"unknown fg_model {fg_model!r}; expected one of {sorted(_FG_PRESETS)}")
    pysm3, u = _require_pysm()
    sky = pysm3.Sky(nside=int(nside), component_config=_seeded_component_config(fg_model, fg_seed))
    npix = 12 * int(nside) ** 2
    out = np.empty((len(freqs_ghz), 3, npix), dtype=np.float64)
    for i, nu in enumerate(freqs_ghz):
        emission = sky.get_emission(nu * u.GHz)
        emission = emission.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu * u.GHz))
        out[i] = np.asarray(emission.value, dtype=np.float64)
    return out


def fg_band_qu(
    freqs_ghz: tuple[float, ...],
    beam_fwhm_arcmin: tuple[float, ...],
    fg_model: str,
    lmax: int,
    nside: int,
    *,
    fg_seed: int = 0,
) -> jax.Array:
    """PySM foregrounds, analyzed and beamed per band → ``(n_band, 2, npix)`` Q/U."""
    import healpy as hp

    iqu = pysm_fg_iqu(freqs_ghz, fg_model, nside, fg_seed=fg_seed)
    bands = []
    for i, fw in enumerate(beam_fwhm_arcmin):
        # healpy quadrature analysis (allocation-independent → need not be diff'able)
        _alm_t, alm_e, alm_b = hp.map2alm(iqu[i], lmax=lmax, pol=True)
        bands.append(_beam_qu_from_eb(alm_e, alm_b, fw, lmax, nside))
    return jnp.stack(bands, axis=0)


def generate_band_sky(
    freqs_ghz: tuple[float, ...],
    beam_fwhm_arcmin: tuple[float, ...],
    *,
    spectra: CMBSpectra,
    r_in: float,
    nside: int,
    lmax: int,
    fg_model: str | None = "d1s1",
    cmb_seed: int = 0,
    fg_seed: int | None = None,
) -> BandSky:
    """Build the fixed beamed sky (CMB + optional PySM FG) for all bands.

    Parameters
    ----------
    freqs_ghz, beam_fwhm_arcmin
        Per-band center frequencies [GHz] and beam FWHM [arcmin] (same length).
    spectra
        CMB BB template provider.
    r_in
        Input tensor-to-scalar ratio of the CMB realization.
    nside, lmax
        HEALPix resolution and band limit.
    fg_model
        PySM preset key (``"d1s1"`` / ``"d10s5"`` / ``"d10s6"``) or ``None`` for a
        CMB-only sky (foreground maps all zero).
    cmb_seed
        Seed for the CMB B-mode realization (CRN).
    fg_seed
        Seed for any stochastic small-scale foreground realization (e.g. ``s6``).
        Defaults to ``cmb_seed`` so that a single per-sim index CRN's the CMB,
        foreground, and (via the JAX key downstream) noise together — fixed across
        apertures, varied across sims. Fixed-template foreground models are
        deterministic and ignore this. See :func:`_seeded_component_config`.
    """
    if len(freqs_ghz) != len(beam_fwhm_arcmin):
        raise ValueError("freqs_ghz and beam_fwhm_arcmin must have the same length.")
    check_band_limit(lmax, nside)
    if fg_seed is None:
        fg_seed = cmb_seed

    b_alm = cmb_b_alm(spectra, r_in, lmax, seed=cmb_seed)
    cmb_qu = cmb_band_qu(b_alm, beam_fwhm_arcmin, lmax, nside)

    if fg_model is None:
        fg_qu = jnp.zeros_like(cmb_qu)
    else:
        fg_qu = fg_band_qu(freqs_ghz, beam_fwhm_arcmin, fg_model, lmax, nside, fg_seed=fg_seed)

    return BandSky(
        freqs_ghz=tuple(float(f) for f in freqs_ghz),
        beam_fwhm_arcmin=tuple(float(f) for f in beam_fwhm_arcmin),
        nside=int(nside),
        lmax=int(lmax),
        r_in=float(r_in),
        cmb_qu=cmb_qu,
        fg_qu=fg_qu,
    )


# ---------------------------------------------------------------------------
# differentiable assembly (sky + allocation-dependent noise)
# ---------------------------------------------------------------------------


def assemble_band_maps(
    band_sky: BandSky,
    w_inv: jax.Array,
    hit_map: jax.Array,
    *,
    noise_key: jax.Array,
) -> jax.Array:
    """Per-band total Q/U maps = beamed sky + anisotropic noise [μK_CMB].

    Differentiable in ``w_inv`` (hence the focal-plane allocation) under common
    random numbers: ``hit_map`` and ``noise_key`` are held fixed, so the noise
    enters only through its ``sqrt(w_inv)`` amplitude. Q and U get independent
    noise realizations (split keys) at the same per-band ``w_inv`` — the
    polarization white-noise power, for which each of Q, U carries pixel variance
    ``w_inv / Ω_pix`` and the map gives ``N_ℓ^{BB} = w_inv``.

    Parameters
    ----------
    band_sky
        Fixed beamed sky from :func:`generate_band_sky`.
    w_inv
        Per-band polarization white-noise power [μK²·sr], shape ``(n_band,)``,
        e.g. from :func:`augr.instrument.white_noise_power`.
    hit_map
        Shared relative exposure per pixel, shape ``(npix,)``.
    noise_key
        JAX PRNG key; fixed across allocations for CRN gradients.

    Returns
    -------
    Total maps, shape ``(n_band, 2, npix)`` (axis 1 = Q, U).
    """
    key_q, key_u = jax.random.split(noise_key)
    noise_q = noise_maps(hit_map, w_inv, key_q)  # (n_band, npix)
    noise_u = noise_maps(hit_map, w_inv, key_u)
    noise_qu = jnp.stack([noise_q, noise_u], axis=1)  # (n_band, 2, npix)
    return band_sky.cmb_qu + band_sky.fg_qu + noise_qu
