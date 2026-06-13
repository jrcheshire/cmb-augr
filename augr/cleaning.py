"""cleaning.py — the component-separation *cleaner* contract.

A "cleaner" turns per-band Q/U maps into a single cleaned B-mode map and knows
how to re-apply its weights to passive (FG-only / noise-only / CMB-only) maps so
the forecast can extract post-separation spectra. :mod:`augr.nilc` (blind NILC)
and :mod:`augr.cmilc` (constrained-moment ILC) are both cleaners in this sense
and already return the same :class:`augr.nilc.NILCResult`; this module names that
contract as a structural :class:`~typing.Protocol` (mirroring
:class:`augr.foregrounds.ForegroundModel`) so they are formally interchangeable
behind one seam, and provides factory adapters that give them a uniform call
site.

What is *not* a cleaner here, deliberately:

* :func:`augr.gnilc.build_gnilc` / ``gnilc_residual_template`` — GNILC takes a
  *second* (nuisance = CMB+noise) map and produces a foreground-residual
  *template*, not a cleaned CMB map. It is a distinct stage (a residual-template
  source), not a ``map → cleaned map`` operation. Keep it explicit.
* :func:`augr.masking.masked_wiener_bb` — a cut-sky *spectrum* estimator that
  consumes a single already-cleaned Q/U map and returns a power spectrum. It sits
  downstream of a cleaner, in the spectrum stage.

Future extension (the deferred masking / E-B work): a spin-2 **Q/U cleaner** that
outputs a cut-sky cleaned Q/U map for :func:`augr.masking.masked_wiener_bb` will
extend :class:`CleanerResult` *additively* — a ``cleaned_qu`` accessor plus a
Q/U-consuming spectrum stage — so this protocol is the seam that work slots into,
not something it has to redesign.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax

from .cmilc import CMILC08_MOMENTS, cmilc_clean
from .config import FIDUCIAL_BK15
from .nilc import nilc_clean

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CleanerResult(Protocol):
    """Structural type for the output of a :class:`Cleaner`.

    The consumed surface — exactly what :func:`augr.nilc_forecast.nilc_spectra`
    reads — so any object exposing it works as a cleaner result with no
    inheritance. :class:`augr.nilc.NILCResult` satisfies it as-is (and
    :func:`augr.cmilc.cmilc_clean` returns a ``NILCResult``).

    Attributes
    ----------
    cleaned_b_alm
        Cleaned B-mode alm at the common resolution, shape ``(Nlm,)``.
    lmax, nside, n_iter
        Transform configuration (so passive maps project identically).
    beam_fwhm_arcmin
        Per-band input beam FWHMs [arcmin], length ``n_band``.
    common_fwhm_arcmin
        Common resolution the bands were brought to (the deconvolved beam).
    """

    cleaned_b_alm: jax.Array
    lmax: int
    nside: int
    n_iter: int
    beam_fwhm_arcmin: tuple[float, ...]
    common_fwhm_arcmin: float

    def project(self, passive_band_qu: jax.Array) -> jax.Array:
        """Apply the stored weights to another map set → its cleaned B alm."""
        ...


@runtime_checkable
class Cleaner(Protocol):
    """Structural type for a component-separation cleaner: per-band Q/U → result.

    The uniform call site shared by NILC and cMILC. Cleaner-specific
    configuration (cMILC's band-center ``freqs`` and moment set, localization,
    needlet peaks, ...) is bound ahead of time by the :func:`nilc_cleaner` /
    :func:`cmilc_cleaner` factories, leaving only the data and the transform
    band-limit at the call site.
    """

    def __call__(
        self,
        band_qu: jax.Array,
        beam_fwhm_arcmin,
        *,
        lmax: int,
        nside: int,
    ) -> CleanerResult:
        """Clean per-band Q/U maps ``(n_band, 2, npix)`` → :class:`CleanerResult`."""
        ...


# ---------------------------------------------------------------------------
# Factory adapters — bind cleaner-specific config into a conforming callable
# ---------------------------------------------------------------------------


def nilc_cleaner(
    *,
    needlet_peaks=None,
    localization_fwhm_arcmin: float | None = None,
    common_fwhm_arcmin: float | None = None,
    n_iter: int = 3,
    ridge: float = 1e-10,
    beam_band_limit: float = 0.1,
) -> Cleaner:
    """A blind-NILC :class:`Cleaner`; kwargs are forwarded to :func:`augr.nilc.nilc_clean`.

    The returned callable is bit-identical to calling ``nilc_clean`` directly with
    the same arguments — it only fixes the uniform ``(band_qu, beams, *, lmax,
    nside)`` call site so NILC and cMILC are interchangeable in a driver.
    """

    def _cleaner(band_qu, beam_fwhm_arcmin, *, lmax, nside):
        return nilc_clean(
            band_qu,
            beam_fwhm_arcmin,
            lmax=lmax,
            nside=nside,
            needlet_peaks=needlet_peaks,
            localization_fwhm_arcmin=localization_fwhm_arcmin,
            common_fwhm_arcmin=common_fwhm_arcmin,
            n_iter=n_iter,
            ridge=ridge,
            beam_band_limit=beam_band_limit,
        )

    return _cleaner


def cmilc_cleaner(
    freqs,
    *,
    moments: tuple[str, ...] = CMILC08_MOMENTS,
    fiducial=FIDUCIAL_BK15,
    needlet_peaks=None,
    localization_fwhm_arcmin: float | None = None,
    common_fwhm_arcmin: float | None = None,
    n_iter: int = 3,
    ridge: float = 1e-10,
    beam_band_limit: float = 0.1,
) -> Cleaner:
    """A cMILC :class:`Cleaner`; ``freqs``/``moments`` are bound here, the rest forwarded.

    cMILC is *not* blind — it needs the band-center ``freqs`` [GHz] and the moment
    set to deproject. Binding them in the factory is what lets cMILC share NILC's
    ``(band_qu, beams, *, lmax, nside)`` call site. The returned callable is
    bit-identical to :func:`augr.cmilc.cmilc_clean` with the same arguments
    (``return_diagnostics`` is left ``False`` so it yields a bare
    :class:`augr.nilc.NILCResult`).
    """

    def _cleaner(band_qu, beam_fwhm_arcmin, *, lmax, nside):
        return cmilc_clean(
            band_qu,
            beam_fwhm_arcmin,
            freqs,
            lmax=lmax,
            nside=nside,
            moments=moments,
            fiducial=fiducial,
            needlet_peaks=needlet_peaks,
            localization_fwhm_arcmin=localization_fwhm_arcmin,
            common_fwhm_arcmin=common_fwhm_arcmin,
            n_iter=n_iter,
            ridge=ridge,
            beam_band_limit=beam_band_limit,
        )

    return _cleaner
