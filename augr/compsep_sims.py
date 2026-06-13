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
from .noise_sims import correlated_noise_maps, noise_maps
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
    # Single-component skies, to attribute the aperture dependence to dust vs synch:
    # "d10" (fixed GNILC dust) is traced by the fine high-nu beams (resolution-
    # insensitive); "s6" (stochastic small-scale synch) by the coarse low-nu beams
    # (the hypothesized D_min driver).
    "d10": ("d10",),
    "s6": ("s6",),
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


@dataclass(frozen=True)
class HarmonicSky:
    """Aperture-independent harmonic-space sky: CMB B alm + per-band FG E/B alm.

    The expensive, aperture-independent part of a sim -- the CMB realization and
    the PySM foreground emission + analysis -- factored out of the per-band beaming
    so it can be built **once** and beamed at many apertures. The per-band beam is
    the only aperture-dependent step, so an aperture sweep at fixed sim builds one
    ``HarmonicSky`` and calls :func:`beam_harmonic_sky` per aperture, rather than
    regenerating the (multi-GB, multi-minute) PySM sky for every diameter. Holding
    the E/B alm (``(n_band, 2, n_alm)``, ~1.4 GB at nside=1024 / 21 bands) across
    the sweep is far cheaper than re-running PySM.

    Attributes
    ----------
    freqs_ghz
        Band center frequencies [GHz], length ``n_band``.
    nside, lmax
        HEALPix resolution and band limit.
    r_in
        Input tensor-to-scalar ratio of the CMB realization.
    cmb_b_alm
        CMB B-mode alm, shape ``(n_alm,)`` [healpy packing].
    fg_eb_alm
        Per-band foreground E/B alm, shape ``(n_band, 2, n_alm)``, or ``None`` for
        a CMB-only sky.
    cmb_e_alm
        CMB E-mode alm, shape ``(n_alm,)``, or ``None`` (default) for a B-only
        CMB. Set it (via ``harmonic_sky(cl_ee=...)``) when the cleaned map must
        carry realistic E-modes for cut-sky E→B leakage — e.g. the masked-Wiener
        forecast. The full-sky B-only forecasts leave it ``None`` (E and B do not
        mix full-sky, so CMB E is irrelevant there).
    """

    freqs_ghz: tuple[float, ...]
    nside: int
    lmax: int
    r_in: float
    cmb_b_alm: jax.Array
    fg_eb_alm: jax.Array | None
    cmb_e_alm: jax.Array | None = None

    @property
    def n_band(self) -> int:
        return len(self.freqs_ghz)


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


def cmb_e_alm(cl_ee: jax.Array, lmax: int, *, seed: int = 0) -> jax.Array:
    """Draw a Gaussian CMB E-mode alm from ``cl_ee`` [healpy packing].

    Companion to :func:`cmb_b_alm` for the cut-sky validation sims (E→B leakage
    purity / fidelity), where a realistic *lensed* EE source dominates the
    ambiguous-mode leakage. ``cl_ee`` is the EE power on ``ℓ = 0..lmax`` — use
    ``augr.delensing.load_lensing_spectra().cl_ee_len``. The realization is fixed
    by ``seed`` (CRN); E and B are drawn independently (TE/EB correlation is out
    of scope, matching the B-only sim's Gaussian-field approximation).
    """
    import healpy as hp

    cl = np.clip(np.asarray(cl_ee)[: lmax + 1], 0.0, None)
    np.random.seed(int(seed) & 0xFFFFFFFF)  # noqa: NPY002 - healpy.synalm uses the global RNG
    e_alm = hp.synalm(cl, lmax=lmax, new=True)
    return jnp.asarray(e_alm)


def _beam_qu_from_eb(
    alm_e: jax.Array, alm_b: jax.Array, fwhm_arcmin: float, lmax: int, nside: int
) -> jax.Array:
    """E/B alm → beam-smoothed Q/U map, shape ``(2, npix)`` [HEALPix-internal]."""
    bl = beam_bl(jnp.arange(lmax + 1, dtype=float), fwhm_arcmin)
    e_beamed = almxfl(jnp.asarray(alm_e), bl, lmax)
    b_beamed = almxfl(jnp.asarray(alm_b), bl, lmax)
    return synthesis(jnp.stack([e_beamed, b_beamed], axis=0), 2, lmax, nside)


def cmb_eb_qu(
    alm_e: jax.Array, alm_b: jax.Array, fwhm_arcmin: float, lmax: int, nside: int
) -> jax.Array:
    """Beam a single E/B alm pair → ``(2, npix)`` Q/U [HEALPix-internal].

    Single-map (one effective beam) CMB realization for the cut-sky estimator
    validation, as opposed to the per-band :func:`cmb_band_qu`. Pass a zero
    ``alm_b`` for an **E-only** sky (the E→B leakage / purity-null source) or a
    zero ``alm_e`` for a **B-only** sky (the transfer-function source).
    """
    return _beam_qu_from_eb(alm_e, alm_b, fwhm_arcmin, lmax, nside)


def cmb_band_qu(
    b_alm: jax.Array,
    beam_fwhm_arcmin: tuple[float, ...],
    lmax: int,
    nside: int,
    *,
    e_alm: jax.Array | None = None,
) -> jax.Array:
    """Beam the shared CMB E/B alm per band → ``(n_band, 2, npix)`` Q/U.

    ``e_alm`` defaults to zero (B-only CMB, the full-sky-forecast convention);
    pass a CMB E-mode alm to include E (cut-sky E→B leakage forecasts).
    """
    alm_e = jnp.zeros_like(b_alm) if e_alm is None else jnp.asarray(e_alm)
    return jnp.stack(
        [_beam_qu_from_eb(alm_e, b_alm, fw, lmax, nside) for fw in beam_fwhm_arcmin],
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


def _fg_eb_alm(
    freqs_ghz: tuple[float, ...], fg_model: str, lmax: int, nside: int, *, fg_seed: int = 0
) -> jax.Array:
    """PySM foregrounds analyzed per band → E/B alm ``(n_band, 2, n_alm)``.

    The aperture-independent half of the foreground path: PySM emission + healpy
    quadrature analysis. The beam (aperture-dependent) is applied later by
    :func:`_beam_fg_eb`.
    """
    import healpy as hp

    iqu = pysm_fg_iqu(freqs_ghz, fg_model, nside, fg_seed=fg_seed)
    eb = []
    for i in range(len(freqs_ghz)):
        # healpy quadrature analysis (allocation-independent → need not be diff'able)
        _alm_t, alm_e, alm_b = hp.map2alm(iqu[i], lmax=lmax, pol=True)
        eb.append(jnp.stack([jnp.asarray(alm_e), jnp.asarray(alm_b)], axis=0))
    return jnp.stack(eb, axis=0)


def _beam_fg_eb(
    fg_eb_alm: jax.Array, beam_fwhm_arcmin: tuple[float, ...], lmax: int, nside: int
) -> jax.Array:
    """Beam precomputed per-band FG E/B alm → ``(n_band, 2, npix)`` Q/U."""
    return jnp.stack(
        [
            _beam_qu_from_eb(fg_eb_alm[i, 0], fg_eb_alm[i, 1], fw, lmax, nside)
            for i, fw in enumerate(beam_fwhm_arcmin)
        ],
        axis=0,
    )


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
    fg_eb = _fg_eb_alm(freqs_ghz, fg_model, lmax, nside, fg_seed=fg_seed)
    return _beam_fg_eb(fg_eb, beam_fwhm_arcmin, lmax, nside)


def harmonic_sky(
    freqs_ghz: tuple[float, ...],
    *,
    spectra: CMBSpectra,
    r_in: float,
    nside: int,
    lmax: int,
    fg_model: str | None = "d1s1",
    cmb_seed: int = 0,
    fg_seed: int | None = None,
    cl_ee: jax.Array | None = None,
) -> HarmonicSky:
    """Build the aperture-independent harmonic sky (CMB B alm + per-band FG E/B alm).

    This is the expensive, aperture-independent part of a sim: the CMB realization
    and the PySM foreground emission + analysis. Beam it at one or more apertures
    with :func:`beam_harmonic_sky`. See :class:`HarmonicSky` for why the split
    matters for an aperture sweep.

    Parameters mirror :func:`generate_band_sky` (minus ``beam_fwhm_arcmin``, which
    is supplied per aperture at the beaming step).

    ``cl_ee`` (optional): when given, also draw a CMB E-mode realization from this
    EE spectrum (``ℓ = 0..lmax``; use ``delensing.load_lensing_spectra().cl_ee_len``)
    so the beamed sky carries E for cut-sky E→B leakage. The E draw is seeded by
    ``cmb_seed + 1`` so E and B are independent realizations sharing the per-sim
    CRN index. Left ``None`` (default) gives a B-only CMB, unchanged.
    """
    check_band_limit(lmax, nside)
    if fg_seed is None:
        fg_seed = cmb_seed

    b_alm = cmb_b_alm(spectra, r_in, lmax, seed=cmb_seed)
    e_alm = None if cl_ee is None else cmb_e_alm(cl_ee, lmax, seed=cmb_seed + 1)
    fg_eb = (
        None if fg_model is None else _fg_eb_alm(freqs_ghz, fg_model, lmax, nside, fg_seed=fg_seed)
    )

    return HarmonicSky(
        freqs_ghz=tuple(float(f) for f in freqs_ghz),
        nside=int(nside),
        lmax=int(lmax),
        r_in=float(r_in),
        cmb_b_alm=b_alm,
        fg_eb_alm=fg_eb,
        cmb_e_alm=e_alm,
    )


def beam_harmonic_sky(hsky: HarmonicSky, beam_fwhm_arcmin: tuple[float, ...]) -> BandSky:
    """Beam an aperture-independent :class:`HarmonicSky` at one aperture → :class:`BandSky`.

    The only aperture-dependent step. Cheap relative to :func:`harmonic_sky`, so an
    aperture sweep at fixed sim calls this once per diameter on a shared ``hsky``.
    """
    if len(beam_fwhm_arcmin) != hsky.n_band:
        raise ValueError(
            f"beam_fwhm_arcmin has {len(beam_fwhm_arcmin)} entries but the harmonic "
            f"sky has {hsky.n_band} bands."
        )
    cmb_qu = cmb_band_qu(
        hsky.cmb_b_alm, beam_fwhm_arcmin, hsky.lmax, hsky.nside, e_alm=hsky.cmb_e_alm
    )
    if hsky.fg_eb_alm is None:
        fg_qu = jnp.zeros_like(cmb_qu)
    else:
        fg_qu = _beam_fg_eb(hsky.fg_eb_alm, beam_fwhm_arcmin, hsky.lmax, hsky.nside)
    return BandSky(
        freqs_ghz=hsky.freqs_ghz,
        beam_fwhm_arcmin=tuple(float(f) for f in beam_fwhm_arcmin),
        nside=hsky.nside,
        lmax=hsky.lmax,
        r_in=hsky.r_in,
        cmb_qu=cmb_qu,
        fg_qu=fg_qu,
    )


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
    cl_ee: jax.Array | None = None,
) -> BandSky:
    """Build the fixed beamed sky (CMB + optional PySM FG) for all bands.

    Convenience wrapper = :func:`harmonic_sky` then :func:`beam_harmonic_sky` at a
    single aperture. For an aperture sweep at fixed sim, call those two directly so
    the (expensive) :func:`harmonic_sky` runs once and only the beaming repeats.

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
    cl_ee
        Optional CMB EE spectrum (ℓ=0..lmax) to also draw CMB E-modes for cut-sky
        E→B leakage forecasts; ``None`` (default) gives a B-only CMB. See
        :func:`harmonic_sky`.
    """
    if len(freqs_ghz) != len(beam_fwhm_arcmin):
        raise ValueError("freqs_ghz and beam_fwhm_arcmin must have the same length.")
    hsky = harmonic_sky(
        freqs_ghz,
        spectra=spectra,
        r_in=r_in,
        nside=nside,
        lmax=lmax,
        fg_model=fg_model,
        cmb_seed=cmb_seed,
        fg_seed=fg_seed,
        cl_ee=cl_ee,
    )
    return beam_harmonic_sky(hsky, beam_fwhm_arcmin)


# ---------------------------------------------------------------------------
# differentiable assembly (sky + allocation-dependent noise)
# ---------------------------------------------------------------------------


def assemble_band_maps(
    band_sky: BandSky,
    w_inv: jax.Array,
    hit_map: jax.Array,
    *,
    noise_key: jax.Array,
    knee_ell: jax.Array | None = None,
    alpha_knee: jax.Array = 1.0,
) -> jax.Array:
    """Per-band total Q/U maps = beamed sky + anisotropic noise [μK_CMB].

    Differentiable in ``w_inv`` (hence the focal-plane allocation) under common
    random numbers: ``hit_map`` and ``noise_key`` are held fixed, so the noise
    enters only through its ``sqrt(w_inv)`` amplitude. Q and U get independent
    noise realizations (split keys) at the same per-band ``w_inv`` — the
    polarization white-noise power, for which each of Q, U carries pixel variance
    ``w_inv / Ω_pix`` and the map gives ``N_ℓ^{BB} = w_inv``.

    With ``knee_ell`` supplied, the noise picks up a 1/f tilt ``N_ℓ = w_inv · (1 +
    (ℓ_knee/ℓ)^α)`` via :func:`augr.noise_sims.correlated_noise_maps` (drawn in
    harmonic space at ``band_sky.lmax`` / ``band_sky.nside``), differentiable
    additionally in ``knee_ell`` / ``alpha_knee``. Q and U get independent 1/f draws
    (isotropic; no scan-direction structure — see the ``noise_sims`` docstring). Left
    as ``None`` (default) the fast white pixel-domain path is used.

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
    knee_ell, alpha_knee
        Per-band 1/f knee multipole and slope (scalars broadcast to all bands). When
        ``knee_ell`` is ``None`` the noise is pure white; otherwise the correlated
        1/f draw is used. ``knee_ell = 0`` per band reduces to white for that band.

    Returns
    -------
    Total maps, shape ``(n_band, 2, npix)`` (axis 1 = Q, U).
    """
    key_q, key_u = jax.random.split(noise_key)
    if knee_ell is None:
        noise_q = noise_maps(hit_map, w_inv, key_q)  # (n_band, npix)
        noise_u = noise_maps(hit_map, w_inv, key_u)
    else:
        noise_q = correlated_noise_maps(
            hit_map, w_inv, knee_ell, alpha_knee, key_q,
            lmax=band_sky.lmax, nside=band_sky.nside,
        )
        noise_u = correlated_noise_maps(
            hit_map, w_inv, knee_ell, alpha_knee, key_u,
            lmax=band_sky.lmax, nside=band_sky.nside,
        )
    noise_qu = jnp.stack([noise_q, noise_u], axis=1)  # (n_band, 2, npix)
    return band_sky.cmb_qu + band_sky.fg_qu + noise_qu
