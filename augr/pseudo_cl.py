"""pseudo_cl.py — MASTER pseudo-Cℓ BB bandpowers on an apodized mask, via NaMaster.

A cut-sky BB estimator for pipelines that already produce a *cleaned* map and need
its bandpowers with the mask's mode coupling deconvolved. This is the complement to
:mod:`augr.masking`: that module estimates B from Q/U with a noise-aware Wiener
filter (differentiable, prior-regularized, sharp mask); this one applies the
standard MASTER deconvolution (non-differentiable, prior-free, apodized mask) and
additionally exposes the resulting **bandpower window function**, which
:class:`augr.signal.SignalModel` consumes directly via ``bandpower_window=``.

Estimator chain::

    B alm  --synthesis_pol(0, 0, b)-->  Q/U
           --NmtField(mask_apo, spin=2)-->  coupled pseudo-Cℓ
           --NmtWorkspace.decouple_cell-->  C_b^{BB}

Use this when the map is already component-separated and you want an unbiased mean
bandpower plus a BPWF; use :mod:`augr.masking` when you need a differentiable
estimator on the ``jax.grad`` path. **NaMaster is C code and not differentiable**,
so nothing here can sit inside a ``jax.grad`` of σ(r); the intended consumption is
frozen BPWFs plus a Monte-Carlo covariance
(:func:`augr.covariance.mc_bandpower_covariance`) fed to
``FisherForecast(external_covariance=...)``.

Conventions
-----------
* **Apodized masks, by necessity.** Unlike :mod:`augr.masking` there is no prior to
  regularize the E/B-ambiguous boundary modes, so the taper is the only thing
  suppressing them and ``purify_b`` requires a smooth window. Default 2° ``C2``;
  measured cost on Planck GAL070 at nside 128 is ``w2²/w4`` 0.7005 → 0.6895 (1.6%).
* **Masks are galactic-frame, RING**, matching :mod:`augr.masking` and PySM-native
  maps. Load with :func:`augr.masking.load_mask`, which casts to float64 *before*
  ``ud_grade`` — the Planck plane masks are ``uint8`` and a raw ``hp.ud_grade`` of
  one truncates 2048 → 128 to an **all-zero map**.
* **Mode counting is ``w2²/w4``, not ``⟨w⟩``.** For a sharp binary mask the two
  coincide (``w^i = w``), which is why :func:`augr.masking.f_sky_of` returns the bare
  mean; for an apodized mask they differ and the Knox mode count needs
  :attr:`MaskMoments.f_sky_eff`. See :func:`augr.covariance.knox_sigma_from_measured_spectrum`,
  whose ``f_sky`` argument delegates exactly this to the caller.
* **Pure-B is opt-in, not the default.** A map whose E component is identically zero
  by construction (e.g. a B-only NILC output) has no E→B leakage to purify, and
  purification on a near-binary mask is numerically fragile. Turn it on only for
  input that genuinely carries E.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MaskMoments",
    "MasterBB",
    "apodize_mask",
    "mask_moments",
    "master_bin_edges",
]


def _require_pymaster():
    """Import pymaster or raise a helpful error naming the conda-only constraint."""
    try:
        import pymaster

        return pymaster
    except ImportError as exc:  # pragma: no cover - exercised only without pymaster
        raise ImportError(
            "augr.pseudo_cl requires NaMaster (conda package 'namaster', import "
            "name 'pymaster'). It is deliberately NOT a pip extra: the PyPI "
            "'pymaster' builds from source against gsl/fftw/cfitsio. It ships in "
            "the pixi 'host' feature only -- the linux-aarch64 'gpu' and 'cpu' "
            "environments exclude it on purpose. Install with:\n"
            "    pixi add namaster"
        ) from exc


# ---------------------------------------------------------------------------
# mask preparation
# ---------------------------------------------------------------------------


def apodize_mask(mask, aposize_deg: float = 2.0, apotype: str = "C2") -> np.ndarray:
    """Apodize a HEALPix mask with a smooth taper (NaMaster ``mask_apodization``).

    ``aposize_deg = 0`` returns the input unchanged (as float64), so callers can
    expose the taper as a tunable without branching. ``apotype`` is ``"C1"``,
    ``"C2"`` or ``"Smooth"``; ``C2`` is the default because it is the smoother of
    the two cosine tapers and so kinder to ``purify_b``.

    The taper is measured from the mask's exact-zero set, so a mask carrying
    *fractional* boundary pixels (as ``ud_grade`` of a high-resolution binary mask
    does) tapers from the outer edge of that fringe. That is intentional and is the
    path the measured ``w2²/w4 = 0.6895`` for GAL070 refers to; binarizing first
    gives 0.6847 instead.
    """
    m = np.asarray(mask, dtype=np.float64)
    if float(aposize_deg) <= 0.0:
        return m
    nmt = _require_pymaster()
    return np.asarray(nmt.mask_apodization(m, float(aposize_deg), apotype=str(apotype)))


@dataclass(frozen=True)
class MaskMoments:
    """Mask moments ``w_i = ⟨w^i⟩`` and the effective sky fraction for mode counting.

    Attributes
    ----------
    w1, w2, w4
        ``⟨w⟩``, ``⟨w²⟩``, ``⟨w⁴⟩`` over the full sphere.
    """

    w1: float
    w2: float
    w4: float

    @property
    def f_sky_eff(self) -> float:
        """``w2²/w4`` — the apodization-corrected sky fraction for the Knox mode count.

        Reduces to ``w1`` for a binary mask. Pass this (not ``⟨w⟩``) wherever an
        ``f_sky`` is used to count modes on an apodized mask.
        """
        return float(self.w2**2 / self.w4)


def mask_moments(mask) -> MaskMoments:
    """Compute :class:`MaskMoments` for a HEALPix mask (no NaMaster needed)."""
    w = np.asarray(mask, dtype=np.float64)
    return MaskMoments(
        w1=float(np.mean(w)), w2=float(np.mean(w**2)), w4=float(np.mean(w**4))
    )


def master_bin_edges(
    ell_min: int = 2,
    ell_max: int = 256,
    delta_ell: int = 20,
    low_bin_hi: int = 29,
) -> list[tuple[int, int]]:
    """Inclusive ``(lo, hi)`` bin edges: one wide low bin, then uniform ``delta_ell``.

    The single ``[ell_min, low_bin_hi]`` bin collects the reionization bump, where
    per-ℓ bandpowers on a cut sky have too few modes to be useful; above it the bins
    are uniform. Set ``low_bin_hi < ell_min`` to get uniform bins throughout.

    Bin count matters downstream: :func:`augr.covariance.mc_bandpower_covariance`
    requires ``n_sims > n_bins + 2`` and its Hartlap factor degrades as the two
    approach, so widening ``delta_ell`` is the cheap lever when sims are scarce.
    """
    ell_min, ell_max = int(ell_min), int(ell_max)
    delta_ell, low_bin_hi = int(delta_ell), int(low_bin_hi)
    if ell_max < ell_min:
        raise ValueError(f"ell_max ({ell_max}) must be >= ell_min ({ell_min}).")
    if delta_ell < 1:
        raise ValueError(f"delta_ell must be >= 1, got {delta_ell}.")

    edges: list[tuple[int, int]] = []
    lo = ell_min
    if low_bin_hi >= ell_min:
        edges.append((ell_min, min(low_bin_hi, ell_max)))
        lo = min(low_bin_hi, ell_max) + 1
    while lo <= ell_max:
        edges.append((lo, min(lo + delta_ell - 1, ell_max)))
        lo += delta_ell
    return edges


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


class MasterBB:
    """MASTER BB estimator on a fixed mask + binning. **Process-local, not frozen.**

    Deliberately breaks two house conventions, for one reason: this object owns an
    opaque ``NmtWorkspace`` (a C handle), so it is **not** a frozen dataclass, not
    hashable, and not picklable. Do not send one to a process-pool worker — have
    each worker call :meth:`build` itself. That is cheap: the workspace costs ~19 ms
    at nside 128 / lmax 256, which is why there is no disk cache here either.

    ``purify_b`` is part of the object's identity rather than a per-call argument,
    because the coupling matrix differs between the purified and unpurified cases;
    a cross-check needs two instances.

    Build once per (mask, binning, purify_b), then reuse across realizations::

        master = MasterBB.build(apodize_mask(mask), bin_edges=master_bin_edges(),
                                nside=128, lmax=256)
        cl_bb = master.bb(qu)

    For several spectra off the same maps, build the fields once and cross them --
    :meth:`field` then :meth:`decouple` -- rather than calling :meth:`bb` per pair.
    """

    def __init__(self, *, mask, bin_edges, nside, lmax, purify_b, lmax_mask, _bins, _wsp):
        self._mask = mask
        self._bin_edges = bin_edges
        self._nside = nside
        self._lmax = lmax
        self._purify_b = purify_b
        self._lmax_mask = lmax_mask
        self._bins = _bins
        self._wsp = _wsp
        self._moments = mask_moments(mask)
        self._window: np.ndarray | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        mask,
        *,
        bin_edges: list[tuple[int, int]],
        nside: int,
        lmax: int,
        purify_b: bool = False,
        lmax_mask: int | None = None,
    ) -> MasterBB:
        """Build the workspace for ``mask`` at ``bin_edges``.

        ``lmax_mask`` bounds the mask power spectrum entering the coupling matrix.
        ``None`` leaves NaMaster's default (``3·nside − 1``), which is the more
        accurate choice. **With ``purify_b=True`` the default raises** (the mask alm
        is built at ``3·nside − 1`` and mismatches the purification, which works at
        ``lmax``), so ``None`` is promoted to ``lmax`` in that case. A strict
        purified-vs-unpurified comparison should therefore pass ``lmax_mask=lmax``
        to *both* instances, so the arms differ only in ``purify_b``.
        """
        nmt = _require_pymaster()
        m = np.asarray(mask, dtype=np.float64)
        nside, lmax = int(nside), int(lmax)
        purify_b = bool(purify_b)

        import healpy as hp

        if hp.get_nside(m) != nside:
            raise ValueError(
                f"mask nside ({hp.get_nside(m)}) != requested nside ({nside}); "
                "ud_grade it first (augr.masking.load_mask(..., nside=...))."
            )
        if not bin_edges:
            raise ValueError("bin_edges is empty.")
        if max(hi for _, hi in bin_edges) > lmax:
            raise ValueError(
                f"bin_edges reach ell={max(hi for _, hi in bin_edges)} > lmax={lmax}."
            )

        if lmax_mask is None and purify_b:
            lmax_mask = lmax
        lmax_mask = None if lmax_mask is None else int(lmax_mask)

        # Explicit NmtBin(bpws=...): NmtBin.from_edges infers its own lmax from the
        # last edge, and from_fields then rejects the field/bin lmax mismatch.
        bpws = np.full(lmax + 1, -1, dtype=int)
        for ib, (lo, hi) in enumerate(bin_edges):
            bpws[int(lo) : int(hi) + 1] = ib
        bins = nmt.NmtBin(
            bpws=bpws,
            ells=np.arange(lmax + 1),
            weights=np.ones(lmax + 1),
            lmax=lmax,
        )

        # A zero field only fixes the mask-dependent coupling matrix, which is all
        # the workspace needs; the data never enters it.
        npix = hp.nside2npix(nside)
        zero = np.zeros((2, npix))
        fld = cls._make_field(nmt, m, zero, lmax, purify_b, lmax_mask)
        wsp = nmt.NmtWorkspace.from_fields(fld, fld, bins)

        return cls(
            mask=m,
            bin_edges=[(int(lo), int(hi)) for lo, hi in bin_edges],
            nside=nside,
            lmax=lmax,
            purify_b=purify_b,
            lmax_mask=lmax_mask,
            _bins=bins,
            _wsp=wsp,
        )

    @staticmethod
    def _make_field(nmt, mask, qu, lmax, purify_b, lmax_mask):
        # NmtField(lmax=) is load-bearing: it otherwise defaults to 3*nside-1 and
        # mismatches the NmtBin lmax.
        kw = dict(spin=2, purify_b=purify_b, lmax=int(lmax))
        if lmax_mask is not None:
            kw["lmax_mask"] = int(lmax_mask)
        return nmt.NmtField(mask, [np.asarray(qu[0]), np.asarray(qu[1])], **kw)

    # -- read-only surface ---------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def bin_edges(self) -> list[tuple[int, int]]:
        return list(self._bin_edges)

    @property
    def bin_centers(self) -> np.ndarray:
        """Effective bin centers ``Σ_ℓ ℓ W_b(ℓ) / Σ_ℓ W_b(ℓ)`` from NaMaster."""
        return np.asarray(self._bins.get_effective_ells())

    @property
    def n_bins(self) -> int:
        return len(self._bin_edges)

    @property
    def nside(self) -> int:
        return self._nside

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def purify_b(self) -> bool:
        return self._purify_b

    @property
    def lmax_mask(self) -> int | None:
        return self._lmax_mask

    @property
    def moments(self) -> MaskMoments:
        return self._moments

    @property
    def f_sky_eff(self) -> float:
        """``w2²/w4`` of the mask — use this as the Knox ``f_sky``."""
        return self._moments.f_sky_eff

    @property
    def window_ells(self) -> np.ndarray:
        """Integer ℓ grid the BPWF is defined on, ``0 .. lmax``."""
        return np.arange(self._lmax + 1)

    @property
    def window(self) -> np.ndarray:
        """BB→BB bandpower window ``(n_bins, lmax+1)``, augr's BPWF convention.

        Feeds :class:`augr.signal.SignalModel` as
        ``bandpower_window=..., bandpower_window_ells=...`` unchanged. It is already
        on the integer ℓ grid, so ``SignalModel``'s per-row ``np.interp`` is the
        identity and no row normalization is lost -- which matters because a
        user-supplied BPWF is deliberately *not* re-normalized there.
        """
        if self._window is None:
            # (n_cl_out, n_bins, n_cl_in, n_ells); spin-2 auto ordering is
            # EE, EB, BE, BB, so BB<-BB is [3, :, 3, :].
            w4d = np.asarray(self._wsp.get_bandpower_windows())
            if w4d.ndim != 4 or w4d.shape[0] != 4 or w4d.shape[2] != 4:
                raise RuntimeError(
                    "unexpected get_bandpower_windows() shape "
                    f"{w4d.shape}; expected (4, n_bins, 4, n_ells) for spin-2."
                )
            w = np.ascontiguousarray(w4d[3, :, 3, :], dtype=np.float64)
            if w.shape != (self.n_bins, self._lmax + 1):
                raise RuntimeError(
                    f"BPWF shape {w.shape} != (n_bins, lmax+1) = "
                    f"{(self.n_bins, self._lmax + 1)}."
                )
            self._window = w
        return self._window

    def save_window(self, path) -> None:
        """Write the BPWF as ``.npz`` readable by :func:`augr.bandpower_windows.load_bandpower_window`."""
        np.savez(str(path), ells=self.window_ells, window=self.window)

    # -- estimation ----------------------------------------------------------

    def field(self, qu):
        """Build an ``NmtField`` from a ``(2, npix)`` Q/U map (opaque handle)."""
        nmt = _require_pymaster()
        qu = np.asarray(qu, dtype=np.float64)
        if qu.shape[0] != 2:
            raise ValueError(f"qu must be (2, npix), got {qu.shape}.")
        return self._make_field(
            nmt, self._mask, qu, self._lmax, self._purify_b, self._lmax_mask
        )

    def field_from_b_alm(self, b_alm):
        """Build a field from a **B-only** alm, synthesizing Q/U with E = T = 0.

        Mirrors :func:`augr.spectrum_stages._cleaned_b_qu`. The resulting sky has
        identically zero E, which is why ``purify_b=False`` is the right default for
        such input: there is no E power to leak into B.
        """
        import jax.numpy as jnp

        from .sht import synthesis_pol

        b = jnp.asarray(b_alm)
        _t, q, u = synthesis_pol(
            jnp.zeros_like(b), jnp.zeros_like(b), b, lmax=self._lmax, nside=self._nside
        )
        return self.field(np.stack([np.asarray(q), np.asarray(u)], axis=0))

    def decouple(self, field_a, field_b=None) -> np.ndarray:
        """Mode-coupling-corrected ``C_b^{BB}``, shape ``(n_bins,)``.

        ``field_b=None`` gives the auto-spectrum. The spin-2 × spin-2 coupled cell
        has four components (EE, EB, BE, BB); index 3 is BB.
        """
        nmt = _require_pymaster()
        other = field_a if field_b is None else field_b
        coupled = nmt.compute_coupled_cell(field_a, other)
        return np.asarray(self._wsp.decouple_cell(coupled)[3])

    def bb(self, qu) -> np.ndarray:
        """Convenience: ``decouple(field(qu))`` for a single Q/U map."""
        return self.decouple(self.field(qu))

    def bb_from_b_alm(self, b_alm) -> np.ndarray:
        """Convenience: ``decouple(field_from_b_alm(b_alm))``."""
        return self.decouple(self.field_from_b_alm(b_alm))
