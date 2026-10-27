"""masking.py — cut-sky B-mode bandpowers via jht's masked Wiener filter.

The in-house compsep sim path (:mod:`augr.compsep_sims`) is full-sky and forms
spectra with ``alm2cl`` and a scalar ``1/f_sky`` correction, so it does not
model realistic sky coverage: the spin-2 E/B split is ambiguous on a cut sky,
and since ``C_ℓ^{EE} ≫ C_ℓ^{BB}`` the leaked E *cosmic variance* inflates the B
error bars by a mask-dependent factor — contaminating instrument *design
comparisons*, not just absolute σ(r). This module replaces that path with a
masked, noise-aware **Wiener-filter** B-mode estimator built on the JAX-native
SHT package `jht` (PyPI ``jaxht``).

Estimator chain (all differentiable; ``jht`` runs its CG in fp64 because augr
enables JAX x64 at import):

    Q/U map  --jht.wiener(spin=2, prior=(C_EE,C_BB)@r=0, N^-1)-->  E/B alm
             --jht.bandpower(spin=2)[1]-->  raw C_ℓ^{BB}
             --bin_spectrum-->  binned bandpowers

The Wiener mean is a *suppressed* estimate of the sky (the prior bounds the
near-null E/B-ambiguous modes — a bias-for-variance trade), so the raw
bandpower is **biased** and must be debiased:

* a **multiplicative transfer** ``F_b`` from B-only sims (corrects the filter's
  suppression of true B), and
* an **always-subtracted additive** E→B leakage template ``⟨C_b^{BB,leak}⟩``
  from E-only (B=0) sims (E→B leakage is additive; ``F_b`` does not remove it).

``C_b^{true} = (C_b^{rec} − leak_b) / F_b`` (:func:`debias_bandpower`). The two
sim sets separate the two corrections cleanly: B-only inputs carry no E, so
``F_b`` is leakage-free; E-only inputs carry no B, so the recovered power is
pure leakage.

The debias assumes the filter is a fixed linear operator, which holds only for
a **converged** solve: a truncated CG leaves a data-dependent filter.
:func:`wiener_preconditioner` makes convergence affordable, and
``return_info=True`` is the only way to see a truncated solve (the
:mod:`augr.spectrum_stages` MC drivers use both and raise on truncation).

**The per-bin debias is exact only for the lensing shape it is calibrated on.** The
sharp-mask filter couples ℓ, so a spectrum of another shape (the tensor, for
``∂/∂r``) is mis-scaled by ``F_b``. Model the estimator with its exact response instead:
:func:`wiener_normal_matrix` → :func:`wiener_bb_response` (``K``, dense, NumPy/LAPACK)
→ :func:`debiased_window` (``W``, a measured bandpower window with ``J_r = W @
C_tensor``). :func:`analytic_transfer` is the noise-free ``F_b`` from ``K``, a check
against the MC.

Conventions
-----------
* **Signal prior fixed at r=0** (LCDM): ``C_EE`` = lensed EE, ``C_BB`` = lensing
  BB. The filter is an estimator choice, not the signal model — fixing it at
  r=0 keeps ``∂(filter)/∂r`` out of the σ(r) derivative (a production pipeline
  fixes the filter before knowing r). Use :func:`augr.delensing.load_lensing_spectra`
  (``cl_ee_len`` / ``cl_bb_len``) for the priors.
* **Maps are in the galactic frame** (PySM native; matches
  ``hit_maps.l2_hit_map(coord="G")``), so a ``|b|``-cut mask thresholds pixel
  latitude directly with no rotation. RING ordering, HEALPix-internal Q/U.
* **Sharp masks, no apodization — for *this* estimator.** The Wiener prior, not an
  apodization taper, controls the E/B-ambiguous modes here, so a taper would only
  throw away sky. This does **not** generalize: :mod:`augr.pseudo_cl`'s MASTER
  pseudo-Cℓ estimator has no prior, so the taper is the only thing suppressing the
  ambiguous boundary modes, and ``purify_b`` there requires a smooth window
  outright. Measured cost of a 2° ``C2`` taper on Planck GAL070 at nside 128:
  effective ``w2²/w4`` goes 0.7005 → 0.6895, i.e. 1.6%.
* **Mode counting.** :func:`f_sky_of` returns the bare ``⟨w⟩``, which is correct
  *because* the masks here are binary (``w^i = w``, so ``w2²/w4 = ⟨w⟩``). An
  apodized mask needs ``w2²/w4`` instead — see
  :class:`augr.pseudo_cl.MaskMoments`.
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _require_jht():
    """Import jht or raise a helpful error pointing at the [masking] extra."""
    try:
        import jht

        return jht
    except ImportError as exc:  # pragma: no cover - exercised only without jht
        raise ImportError(
            "augr.masking requires 'jht' (PyPI distribution 'jaxht'), which "
            "ships with the masking extra. Install it with:\n"
            "    pip install 'cmb-augr[masking]'\n"
            "or, in the development env:\n"
            "    pixi add --pypi 'jaxht>=0.4.0'"
        ) from exc


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def gal_cut_mask(nside: int, b_cut_deg: float) -> jax.Array:
    """Sharp galactic ``|b| > b_cut`` binary mask (1 = observed, 0 = cut), RING.

    Thresholds pixel latitude directly; assumes galactic-frame maps (see module
    docstring). ``b_cut_deg = 0`` is full sky.
    """
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    theta = hp.pix2ang(int(nside), np.arange(npix))[0]  # colatitude [rad]
    lat_deg = 90.0 - np.degrees(theta)
    mask = (np.abs(lat_deg) >= float(b_cut_deg)).astype(np.float64)
    return jnp.asarray(mask)


def galactic_mask(nside: int, f_sky: float) -> jax.Array:
    """``|b|``-cut binary mask hitting a target ``f_sky``.

    For a symmetric ``|b| > b_cut`` cut over both hemispheres, the observed sky
    fraction is ``f_sky = 1 − sin(b_cut)``, so ``b_cut = arcsin(1 − f_sky)``.
    The realized ``f_sky`` matches the target up to HEALPix pixelization.
    """
    b_cut_deg = float(np.degrees(np.arcsin(np.clip(1.0 - float(f_sky), -1.0, 1.0))))
    return gal_cut_mask(nside, b_cut_deg)


def _pixel_abs_lat_deg(nside: int) -> np.ndarray:
    """|galactic latitude| of each RING pixel centre [deg]. Fixed geometry."""
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    theta = hp.pix2ang(int(nside), np.arange(npix))[0]
    return np.abs(90.0 - np.degrees(theta))


def smooth_gal_cut_mask(nside: int, b_cut_deg, width_deg) -> jax.Array:
    """Sigmoid ``|b|`` cut -- the differentiable counterpart to :func:`gal_cut_mask`.

    ``w(p) = sigmoid((|b_p| - b_cut) / width)``, differentiable in **both**
    ``b_cut_deg`` and ``width_deg``. The pixel geometry is fixed, so nothing is
    gridded: the mask is a closed-form function of the two design parameters
    evaluated at fixed pixel centres, and ``jax.grad`` reaches them.

    ``gal_cut_mask`` and :func:`galactic_mask` threshold with a numpy comparison
    and therefore carry no gradient at all, which is why a sky-coverage design
    axis was not previously possible.

    **The width is the apodization.** A MASTER estimator needs a tapered mask, and
    rather than applying a separate (non-differentiable) apodization step this
    family is smooth by construction. Keep ``width_deg`` at or above roughly one
    pixel scale: measured at nside=128, the fraction of ``sum (2l+1) W_l`` above
    l = 192 is 1.7e-8 at width 1 deg but 1.6e-3 for a hard step, and that tail is
    what a truncated mask band-limit throws away.
    """
    lat = jnp.asarray(_pixel_abs_lat_deg(nside))
    return jax.nn.sigmoid((lat - jnp.asarray(b_cut_deg)) / jnp.asarray(width_deg))


def admission_rank(masks, *, check: bool = True) -> jax.Array:
    """Per-pixel admission rank of a **nested** family of masks.

    Given masks ordered from most to least aggressive (or the reverse), the sum
    ``A(p) = sum_k M_k(p)`` counts how many of them admit pixel ``p``. For a
    nested family the individual masks are exactly the level sets of ``A``, so a
    single scalar threshold on ``A`` reproduces any of them -- and a *smooth*
    threshold interpolates continuously between them. See :func:`level_set_mask`.

    The Planck ``HFI_Mask_GalPlane`` stack is the motivating instance (its eight
    fields are exactly nested, verified: zero subset violations).

    ``check`` verifies nestedness and raises if it fails. Leaving it on is
    recommended: on a non-nested family ``A`` is still a perfectly well-formed
    array and ``level_set_mask`` still returns something plausible, but the level
    sets are no longer the input masks and the interpolation means nothing.
    """
    m = jnp.asarray(masks)
    if m.ndim != 2:
        raise ValueError(f"masks must be (n_masks, npix), got shape {m.shape}.")
    if check:
        binary = np.asarray(m) > 0.5
        order = np.argsort(binary.sum(axis=1))
        for a, b in itertools.pairwise(order):
            if not np.all(binary[b][binary[a]]):
                raise ValueError(
                    f"masks are not nested: mask {a} admits pixels that mask {b} "
                    "cuts, so the summed rank field has no level-set meaning. "
                    "Pass check=False only if you know why that is acceptable."
                )
    return jnp.sum(m, axis=0)


def level_set_mask(rank, t, width: float = 0.15) -> jax.Array:
    """Smoothly threshold an admission-rank field: ``sigmoid((A - t) / width)``.

    Continuous and differentiable in ``t``, which is the sky-coverage design
    coordinate. At integer ``t`` and small ``width`` this reproduces the
    corresponding member of the nested family; between integers it interpolates.

    ``f_sky`` is monotone in ``t`` by construction (``A`` is fixed and sigmoid is
    monotone), so the coordinate has no folds for an optimizer to get caught in.
    """
    return jax.nn.sigmoid((jnp.asarray(rank) - jnp.asarray(t)) / float(width))


def load_mask(path: str, *, nside: int | None = None, field: int = 0) -> jax.Array:
    """Load a HEALPix mask FITS (RING). Optionally ``ud_grade`` to ``nside``."""
    import healpy as hp

    m = np.asarray(hp.read_map(path, field=field), dtype=np.float64)
    if nside is not None and hp.get_nside(m) != int(nside):
        m = hp.ud_grade(m, int(nside))
    return jnp.asarray(m)


def f_sky_of(mask: jax.Array) -> float:
    """Effective sky fraction ``⟨mask⟩`` (mean of the per-pixel weight)."""
    return float(jnp.mean(jnp.asarray(mask)))


# ---------------------------------------------------------------------------
# per-pixel inverse-noise
# ---------------------------------------------------------------------------


def inv_noise_map(
    hit_map: jax.Array, var_pix_ref: float, *, mask: jax.Array | None = None
) -> jax.Array:
    """Per-pixel inverse-noise ``N^-1`` [μK⁻²] for :func:`masked_wiener_bb`.

    The polarization pixel noise variance scales inversely with exposure:
    ``σ²(p) = var_pix_ref · max(H) / H(p)``, so ``N^-1(p) = H(p) / (var_pix_ref ·
    max(H))``. Zero-hit pixels and (if ``mask`` is given) cut pixels are set to
    0 — the mask must be folded into ``inv_noise`` for jht's Wiener solve (it
    does not apply a separate mask when ``inv_noise`` is supplied).

    Parameters
    ----------
    hit_map
        Relative exposure per pixel ``(npix,)`` (any positive scaling; only the
        shape matters, e.g. ``hit_maps.l2_hit_map``).
    var_pix_ref
        Per-pixel polarization noise variance [μK²] at the maximum-exposure
        pixel (the cleaned-map white level there).
    mask
        Optional binary/weight mask ``(npix,)``; pixels with ``mask <= 0`` are
        zeroed.
    """
    h = jnp.asarray(hit_map)
    hmax = jnp.max(h)
    inv = jnp.where(h > 0, h / (float(var_pix_ref) * hmax), 0.0)
    if mask is not None:
        inv = inv * (jnp.asarray(mask) > 0)
    return inv


def var_pix_ref_from_noise(
    noise_qu: jax.Array, hit_map: jax.Array, mask: jax.Array
) -> float:
    """``var_pix_ref`` [μK²] for :func:`inv_noise_map` from a noise Q/U realization.

    Each observed pixel's ``(Q² + U²)/2`` is scaled by ``H(p)/max(H)`` before averaging,
    so a noise map with ``σ²(p) = v · max(H)/H(p)`` returns ``v`` -- the level at the
    maximum-exposure pixel, which is what :func:`inv_noise_map` expects. Under uniform
    hits the scale is exactly 1 and this is the plain observed-pixel mean.

    Parameters
    ----------
    noise_qu
        Noise-only Q/U maps ``(2, npix)`` (e.g. a cleaned-noise realization).
    hit_map
        Relative exposure ``(npix,)``, the one passed to :func:`inv_noise_map`.
    mask
        Binary/weight mask ``(npix,)``; pixels with ``mask <= 0`` or zero hits are
        excluded.
    """
    h = jnp.asarray(hit_map)
    obs = (jnp.asarray(mask) > 0) & (h > 0)
    p = (noise_qu[0] ** 2 + noise_qu[1] ** 2) * (h / jnp.max(h))
    return float(jnp.mean(p[obs]) / 2.0)


# ---------------------------------------------------------------------------
# masked-Wiener B-mode estimator
# ---------------------------------------------------------------------------


def wiener_preconditioner(
    inv_noise: jax.Array,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    nside: int,
    lmax: int,
    ell_split: int | None = 32,
):
    """``jht.WienerPreconditioner`` for :func:`masked_wiener_eb_alm` at fixed ``N⁻¹`` and prior.

    Spin-2 wrapper of ``jht.wiener_preconditioner``: ℓ-diagonal ``1 + C_ℓ n̄``, with the
    exact operator block on ``ℓ ≤ ell_split`` (``None`` = diagonal only). It changes the
    iteration count, not the solution. Build eagerly, once per ``(inv_noise, prior)``;
    the block costs ``2((ell_split+1)² − 4)`` operator applications (2170 at 32, ~3 min
    at nside 128).
    """
    jht = _require_jht()
    return jht.wiener_preconditioner(
        (jnp.asarray(cl_ee_prior), jnp.asarray(cl_bb_prior)),
        int(nside),
        int(lmax),
        spin=2,
        inv_noise=jnp.asarray(inv_noise),
        ell_split=ell_split,
    )


def masked_wiener_eb_alm(
    qu_map: jax.Array,
    inv_noise: jax.Array,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    nside: int,
    lmax: int,
    max_iter: int = 1000,
    tol: float = 1e-8,
    precond=None,
    return_info: bool = False,
):
    """Wiener-filtered E/B alm from a masked Q/U map → ``(2, K)`` ``[E, B]``.

    Solves ``(SᵀN⁻¹S + C⁻¹) a = SᵀN⁻¹ m`` (jht's prior-whitened CG). The signal
    prior is fixed at r=0: ``cl_ee_prior`` = lensed EE, ``cl_bb_prior`` = lensing
    BB (each ``(lmax+1,)``). ``inv_noise`` ``(npix,)`` must have the mask folded
    in (see :func:`inv_noise_map`).

    ``precond`` is a :func:`wiener_preconditioner` built for the same ``inv_noise``
    and priors. ``return_info=True`` also returns ``jht.SolveInfo(n_iter,
    rel_residual)``; ``rel_residual > tol`` means the solve stopped at ``max_iter``
    unconverged, which nothing else reports.
    """
    jht = _require_jht()
    cl_ee = jnp.asarray(cl_ee_prior)
    cl_bb = jnp.asarray(cl_bb_prior)
    return jht.wiener(
        jnp.asarray(qu_map),
        (cl_ee, cl_bb),
        int(nside),
        int(lmax),
        spin=2,
        inv_noise=jnp.asarray(inv_noise),
        max_iter=max_iter,
        tol=tol,
        precond=precond,
        return_info=return_info,
    )


def masked_wiener_bb(
    qu_map: jax.Array,
    inv_noise: jax.Array,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    nside: int,
    lmax: int,
    max_iter: int = 1000,
    tol: float = 1e-8,
    precond=None,
    return_info: bool = False,
):
    """Raw per-ℓ ``C_ℓ^{BB}`` (length ``lmax+1``) of the Wiener-filtered map.

    This is the **filter-suppressed, leakage-contaminated** B power — apply
    :func:`transfer_function` and subtract :func:`leakage_template` (via
    :func:`debias_bandpower`) for an unbiased estimate. Differentiable in the
    map and the noise level. ``precond`` / ``return_info`` as in
    :func:`masked_wiener_eb_alm` (``return_info`` gives ``(cl, SolveInfo)``).
    """
    jht = _require_jht()
    out = masked_wiener_eb_alm(
        qu_map,
        inv_noise,
        cl_ee_prior,
        cl_bb_prior,
        nside=nside,
        lmax=lmax,
        max_iter=max_iter,
        tol=tol,
        precond=precond,
        return_info=return_info,
    )
    eb_alm, info = out if return_info else (out, None)
    cl_bb = jht.bandpower(eb_alm, int(lmax), spin=2)[1]  # C_BB channel
    return (cl_bb, info) if return_info else cl_bb


def bin_spectrum(cl: jax.Array, bin_matrix: jax.Array, ell_min: int) -> jax.Array:
    """Bin a per-ℓ spectrum (indexed from ℓ=0) onto SignalModel bins.

    ``bin_matrix`` is ``(n_bins, n_ells)`` over ``[ell_min, ell_max]`` (e.g.
    ``SignalModel.bin_matrix``); the spectrum is sliced to that range, so
    ``ell_min`` is the SignalModel's ``ell_min``. Returns ``(n_bins,)``.
    """
    bm = jnp.asarray(bin_matrix)
    n_ells = bm.shape[1]
    seg = jnp.asarray(cl)[int(ell_min) : int(ell_min) + n_ells]
    return bm @ seg


# ---------------------------------------------------------------------------
# debias: multiplicative transfer (B-only) + additive E→B leakage (E-only)
# ---------------------------------------------------------------------------


def transfer_function(rec_binned_b_only: jax.Array, true_binned_bb: jax.Array) -> jax.Array:
    """Multiplicative transfer ``F_b = ⟨C_b^{rec}⟩ / C_b^{true}`` from B-only sims.

    Parameters
    ----------
    rec_binned_b_only
        Recovered binned BB bandpowers from B-only (E=0) inputs, ``(n_sims,
        n_bins)`` (averaged over sims) or already-averaged ``(n_bins,)``.
    true_binned_bb
        The binned input BB truth ``(n_bins,)`` (same binning).
    """
    rec = jnp.asarray(rec_binned_b_only)
    mean_rec = jnp.mean(rec, axis=0) if rec.ndim == 2 else rec
    return mean_rec / jnp.asarray(true_binned_bb)


def leakage_template(rec_binned_e_only: jax.Array) -> jax.Array:
    """Additive E→B leakage template ``⟨C_b^{BB,leak}⟩`` from E-only (B=0) sims.

    ``rec_binned_e_only`` is ``(n_sims, n_bins)`` (averaged) or ``(n_bins,)``.
    """
    rec = jnp.asarray(rec_binned_e_only)
    return jnp.mean(rec, axis=0) if rec.ndim == 2 else rec


def debias_bandpower(rec_binned: jax.Array, transfer: jax.Array, leakage: jax.Array) -> jax.Array:
    """Unbiased binned BB: ``(C_b^{rec} − leakage_b) / F_b``.

    Inverts the bandpower model ``C^{rec} = F · C^{true,B} + leakage`` (the
    multiplicative filter suppression plus the additive E→B leakage).
    """
    return (jnp.asarray(rec_binned) - jnp.asarray(leakage)) / jnp.asarray(transfer)


# ---------------------------------------------------------------------------
# exact bandpower window: the filter's dense B→B response
# ---------------------------------------------------------------------------


def _dof_ell(lmax: int) -> np.ndarray:
    """Multipole of each spin-2 real DOF in jht's ``[E, B]`` layout (length ``n_dof``)."""
    import healpy as hp

    jht = _require_jht()
    ell, m = hp.Alm.getlm(int(lmax))
    unit = np.where(m > 0, 1.0 + 1.0j, 1.0 + 0.0j)

    def to_real(v):
        return np.asarray(jht.alm_to_real(jnp.asarray(np.stack([v, v])), int(lmax), spin=2))

    return np.rint(to_real(ell * unit) / to_real(unit)).astype(int)


def _symmetrize_inplace(a: np.ndarray, block: int = 2048) -> None:
    """``a <- (a + aᵀ)/2`` block by block, without a full-size temporary."""
    n = a.shape[0]
    for i in range(0, n, block):
        ii = slice(i, i + block)
        a[ii, ii] = 0.5 * (a[ii, ii] + a[ii, ii].T)
        for j in range(i + block, n, block):
            jj = slice(j, j + block)
            u = 0.5 * (a[ii, jj] + a[jj, ii].T)
            a[ii, jj] = u
            a[jj, ii] = u.T


def _normal_apply(x, w, nside: int, lmax: int):
    """``A x = T Sᵀ diag(w) S T⁻¹ x`` on one real-DOF vector; ``w`` is ``(1, npix)``."""
    jht = _require_jht()
    a = jht.real_to_alm(x, lmax, spin=2)
    mp = w * jht.synthesis(a, nside, lmax, spin=2)
    return jht.alm_to_real(jht.adjoint_synthesis(mp, nside, lmax, spin=2), lmax, spin=2)


def _prior_pw(cl_ee_prior, cl_bb_prior, ell: np.ndarray, lmax: int) -> np.ndarray:
    """Per-DOF ``√C_prior`` in the ``[E, B]`` layout; 0 where the prior is not positive (as jht)."""
    n_e = ell.size // 2
    cl_ee = np.clip(np.asarray(cl_ee_prior, dtype=np.float64)[: lmax + 1], 0.0, None)
    cl_bb = np.clip(np.asarray(cl_bb_prior, dtype=np.float64)[: lmax + 1], 0.0, None)
    return np.sqrt(np.concatenate([cl_ee[ell[:n_e]], cl_bb[ell[n_e:]]]))


def wiener_normal_matrix(
    inv_noise: jax.Array, *, nside: int, lmax: int, batch: int = 64, verbose: bool = False
) -> np.ndarray:
    """Dense spin-2 normal matrix ``A = T Sᵀ diag(N⁻¹) S T⁻¹`` of the masked-Wiener solve.

    ``(n_dof, n_dof)`` float64 in jht's real-DOF x-space (``[E, B]``, ℓ < 2 dropped;
    ``n_dof = jht.n_dof(lmax, 2)``), built by applying the operator to ``batch`` unit
    vectors at a time and symmetrized. Input to :func:`wiener_bb_response`. Costs
    ``n_dof`` operator applications and ``8 n_dof²`` bytes: 74,490 and 44 GB at lmax 192.
    ``A`` is linear in ``inv_noise``, so a uniform-hits filter at a different
    ``var_pix_ref`` needs no rebuild (``wiener_bb_response(scale=)``). ``verbose`` prints
    the column rate and ETA about every 5%.
    """
    import time
    jht = _require_jht()
    nside, lmax, batch = int(nside), int(lmax), int(batch)
    n = int(jht.n_dof(lmax, 2))
    w = jnp.asarray(inv_noise)[None, :]

    @jax.jit
    def rows(start):
        idx = start + jnp.arange(batch)
        units = (idx[:, None] == jnp.arange(n)[None, :]).astype(jnp.float64)
        return jax.vmap(lambda x: _normal_apply(x, w, nside, lmax))(units)

    out = np.empty((n, n))
    starts = range(0, n, batch)
    every = max(1, len(starts) // 20)
    t0 = time.perf_counter()
    for i, s in enumerate(starts):
        out[s : s + batch] = np.asarray(rows(s))[: n - s]  # A e_j is row j (A = Aᵀ)
        if verbose and (i + 1) % every == 0:
            done = min(s + batch, n)
            dt = time.perf_counter() - t0
            print(f"wiener_normal_matrix: {done}/{n} columns, {dt / done * 1e3:.1f} ms/column, "
                  f"ETA {dt / done * (n - done) / 60:.1f} min", flush=True)
    _symmetrize_inplace(out)
    return out


def _whitened_cholesky(
    normal, cl_ee_prior, cl_bb_prior, *, lmax, scale, overwrite, block, clean, stage=None
):
    """Host LAPACK Cholesky of ``M + I``, ``M = P (scale·A) P``: ``(chol, pw, ell)``.

    ``chol`` is the Fortran-ordered lower factor (``clean=False`` leaves the input's
    values above the diagonal), ``pw`` the per-DOF ``√C_prior`` (0 where the prior is
    not positive, as in jht), ``ell`` the per-DOF multipole. ``overwrite=True`` factors
    ``normal`` in place (C-contiguous float64 only), otherwise a copy.
    """
    from scipy.linalg import lapack

    lmax = int(lmax)
    ell = _dof_ell(lmax)
    n = ell.size
    if np.shape(normal) != (n, n):
        raise ValueError(f"normal must be ({n}, {n}) for lmax={lmax}, got {np.shape(normal)}")
    pw = _prior_pw(cl_ee_prior, cl_bb_prior, ell, lmax)

    if overwrite:
        m = normal
        if not (isinstance(m, np.ndarray) and m.dtype == np.float64 and m.flags.c_contiguous):
            raise ValueError("overwrite=True needs a C-contiguous float64 ndarray")
    else:
        m = np.array(normal, dtype=np.float64, order="C")
    for s in range(0, n, block):
        m[s : s + block] *= (float(scale) * pw[s : s + block, None]) * pw[None, :]
    m.flat[:: n + 1] += 1.0

    # m is symmetric and C-ordered, so m.T is the same matrix in Fortran order: LAPACK
    # factors it in place.
    if stage is not None:
        stage("M + I")
    chol, info = lapack.dpotrf(m.T, lower=1, overwrite_a=1, clean=int(clean))
    if stage is not None:
        stage("dpotrf")
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotrf failed (info={info}): M + I is not positive definite")
    return chol, pw, ell


def wiener_bb_response(
    normal: np.ndarray,
    cl_ee_prior: jax.Array,
    cl_bb_prior: jax.Array,
    *,
    lmax: int,
    scale: float = 1.0,
    overwrite: bool = False,
    block: int = 2048,
    verbose: bool = False,
) -> np.ndarray:
    """Exact B→B response ``K`` ``(lmax+1, lmax+1)`` of the converged masked-Wiener estimator.

    ``E[Ĉ_ℓ'^BB] = Σ_ℓ K[ℓ', ℓ] C_ℓ^BB`` for a noise-free B-only map with spectrum
    ``C^BB`` at the filter input (beamed, like the priors), where ``Ĉ^BB`` is
    :func:`masked_wiener_bb` with the same ``N⁻¹`` and priors. ``normal`` is
    :func:`wiener_normal_matrix`; ``scale`` multiplies it (``N⁻¹ -> scale·N⁻¹``).

    With ``M = P (scale·A) P`` and ``P = diag(√C_prior)`` the map-to-alm operator is
    ``P [I − (M+I)⁻¹] P⁻¹``; its BB block needs only ``[(M+I)⁻¹]_BB``, which is
    ``(L_BB L_BBᵀ)⁻¹`` from the Cholesky factor of ``M + I`` (B is the trailing half).
    The B prior must be positive on ℓ = 2..lmax. NumPy/LAPACK, not traced: about
    ``n_dof³/3`` flops, and ``8 n_dof²`` bytes for the factor (``overwrite=True`` factors
    ``normal`` in place, which must then be C-contiguous) plus ``2 n_dof²`` for the BB
    inverse. Use as a measured bandpower window via :func:`debiased_window`. ``verbose``
    prints the time of each stage.
    """
    import time

    from scipy.linalg import lapack

    t0 = time.perf_counter()

    def stage(name):
        if verbose:
            print(f"wiener_bb_response: {name} done at {time.perf_counter() - t0:.0f} s", flush=True)

    lmax = int(lmax)
    cl_bb = np.asarray(cl_bb_prior, dtype=np.float64)[: lmax + 1]
    if np.any(cl_bb[2:] <= 0.0):
        raise ValueError("cl_bb_prior must be positive on ell = 2..lmax (the response divides by it)")
    chol, _pw, ell = _whitened_cholesky(
        normal, cl_ee_prior, cl_bb, lmax=lmax, scale=scale, overwrite=overwrite, block=block,
        clean=False, stage=stage,
    )
    n_e = ell.size // 2
    inv_bb, info = lapack.dpotri(chol[n_e:, n_e:], lower=1, overwrite_c=1)
    stage("dpotri")
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotri failed (info={info})")

    # S[ℓ', ℓ] = Σ_{d'∈ℓ', d∈ℓ} G²_{d'd}, G = I − inv_bb, from the lower triangle only:
    # S = T + Tᵀ − D, T the d' >= d part, D the diagonal (counted in both).
    ell_b = ell[n_e:]
    n_b = ell_b.size
    hot = np.zeros((n_b, lmax + 1))
    hot[np.arange(n_b), ell_b] = 1.0
    t = np.zeros((lmax + 1, lmax + 1))
    for s in range(0, n_b, block):
        e = min(s + block, n_b)
        g = -np.tril(inv_bb[s:e, :e], k=s)
        g[np.arange(e - s), np.arange(s, e)] += 1.0
        t += hot[s:e].T @ ((g * g) @ hot[:e])
    d = np.zeros(lmax + 1)
    np.add.at(d, ell_b, (1.0 - np.diagonal(inv_bb)) ** 2)
    s_sum = t + t.T - np.diag(d)
    stage("accumulate")

    ells = np.arange(lmax + 1)
    ok = ells >= 2
    ratio = np.zeros((lmax + 1, lmax + 1))
    ratio[np.ix_(ok, ok)] = cl_bb[ok][:, None] / cl_bb[ok][None, :]
    return s_sum * ratio / (2.0 * ells + 1.0)[:, None]


def debiased_window(
    K: np.ndarray,
    bin_matrix: jax.Array,
    ell_min: int,
    *,
    common_fwhm_arcmin: float,
    transfer: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpower window of the debiased Wiener bandpowers: ``(W, ells)``.

    ``W[b, ℓ] = Σ_ℓ' bin_matrix[b, ℓ'] K[ℓ', ℓ] B_c,ℓ² / F_b``, so ``E[debiased_b] = W @ C``
    for a beam-free input B spectrum ``C`` on ``ells = ell_min..lmax`` (wider than the
    bins: power above the last bin still leaks in). ``K`` is :func:`wiener_bb_response`;
    ``transfer`` must be the ``F_b`` that debiased the data, so ``J_r = W @ C_tensor``
    and the covariance share it. Feeds ``SignalModel(bandpower_window=W,
    bandpower_window_ells=ells)`` with ``ell_max = lmax``.
    """
    from .instrument import beam_bl

    k = np.asarray(K)
    lmax = k.shape[0] - 1
    bm = np.asarray(bin_matrix)
    n_ells = bm.shape[1]
    ells = np.arange(int(ell_min), lmax + 1)
    bl2 = np.asarray(beam_bl(jnp.asarray(ells, dtype=float), float(common_fwhm_arcmin))) ** 2
    rows = bm @ k[int(ell_min) : int(ell_min) + n_ells, int(ell_min) :]
    return rows * bl2[None, :] / np.asarray(transfer)[:, None], ells


def analytic_transfer(
    K: np.ndarray,
    bin_matrix: jax.Array,
    ell_min: int,
    *,
    common_fwhm_arcmin: float,
    true_cl_bb: jax.Array,
) -> np.ndarray:
    """Noise-free ``F_b = bin(K B_c² C) / bin(C)`` -- the exact counterpart of the MC transfer.

    ``true_cl_bb`` is the beam-free input B spectrum on ℓ = 0..lmax, the same truth the
    MC :func:`transfer_function` divides by. A check on ``K`` against the MC; the debias
    itself keeps the MC ``F_b``.
    """
    from .instrument import beam_bl

    k = np.asarray(K)
    lmax = k.shape[0] - 1
    cl = np.asarray(true_cl_bb, dtype=np.float64)[: lmax + 1]
    bl2 = np.asarray(beam_bl(jnp.arange(lmax + 1, dtype=float), float(common_fwhm_arcmin))) ** 2
    rec = k @ (bl2 * cl)
    num = np.asarray(bin_spectrum(rec, bin_matrix, ell_min))
    return num / np.asarray(bin_spectrum(cl, bin_matrix, ell_min))


# ---------------------------------------------------------------------------
# dense direct solve: one Cholesky per filter in place of a CG per map
# ---------------------------------------------------------------------------


class WienerFactor(eqx.Module):
    """Cholesky factor of the prior-whitened masked-Wiener system, from :func:`wiener_factor`.

    ``chol_t`` is the **transpose** of the lower Cholesky factor ``L`` of ``M + I``
    (``M + I = L Lᵀ``, ``M = P (A / var_pix_ref) P``, ``A`` the unit-noise
    :func:`wiener_normal_matrix`, ``P = diag(pw)``, ``pw`` the per-DOF ``√C_prior``); the
    other triangle is zero. Stored transposed because the triangular solvers (LAPACK,
    cuBLAS) take column-major input: ``chol_t.T`` reaches them without a copy, where ``L``
    itself would be copied (``8 n_dof²`` bytes, 44 GB at lmax 192) on every solve. A pytree
    whose arrays are leaves: hand it to a jitted function as an argument, since a
    closed-over array is compiled in as a constant.
    """

    chol_t: jax.Array
    pw: jax.Array
    var_pix_ref: float = eqx.field(static=True)
    lmax: int = eqx.field(static=True)


def wiener_factor(
    normal,
    cl_ee_prior,
    cl_bb_prior,
    *,
    lmax: int,
    var_pix_ref: float,
    on_device: bool = False,
    overwrite: bool = False,
    block: int = 2048,
) -> WienerFactor:
    """Factor the masked-Wiener system once, for :func:`masked_wiener_eb_alm_dense`.

    ``normal`` is :func:`wiener_normal_matrix` at **unit** noise
    (``inv_noise_map(hits, 1.0, mask=...)``). ``A`` is linear in ``N⁻¹``, so the filter of
    ``inv_noise_map(hits, var_pix_ref, mask=...)`` is ``A / var_pix_ref``. The priors are
    the (beamed) ones the CG path takes.

    ``on_device=False`` factors on the host with LAPACK (in place under ``overwrite=True``,
    which needs a C-contiguous float64 ndarray) and copies the factor to the default JAX
    device. ``on_device=True`` moves ``normal`` to the default device and factors there
    (the GPU path; transiently ~2-3 copies of ``A`` in device memory). Either way the cost
    is ``n_dof³/3`` flops and ``8 n_dof²`` bytes per copy -- 44 GB at lmax 192, ~140 GB at
    lmax 256. Memory grows as ``lmax⁴``, so this serves low-ℓ filters.
    """
    lmax = int(lmax)
    scale = 1.0 / float(var_pix_ref)
    if not on_device:
        chol, pw, _ell = _whitened_cholesky(
            normal, cl_ee_prior, cl_bb_prior, lmax=lmax, scale=scale, overwrite=overwrite,
            block=block, clean=True,
        )
        # chol is the Fortran-ordered L, so chol.T is C-ordered Lᵀ: no copy on the host.
        chol_t = jnp.asarray(chol.T)
        del chol
        return WienerFactor(chol_t, jnp.asarray(pw), var_pix_ref=float(var_pix_ref), lmax=lmax)

    ell = _dof_ell(lmax)
    n = ell.size
    if np.shape(normal) != (n, n):
        raise ValueError(f"normal must be ({n}, {n}) for lmax={lmax}, got {np.shape(normal)}")
    pw = jnp.asarray(_prior_pw(cl_ee_prior, cl_bb_prior, ell, lmax))
    idx = jnp.arange(n)

    def whiten(a, pw):
        return (a * (scale * pw[:, None]) * pw[None, :]).at[idx, idx].add(1.0)

    m = jax.jit(whiten, donate_argnums=0)(jnp.array(normal, dtype=jnp.float64), pw)
    # Transposing inside the Cholesky's jit costs no extra buffer (measured on CPU).
    chol_t = jax.jit(lambda m: jnp.linalg.cholesky(m).T, donate_argnums=0)(m)
    if not bool(jnp.all(jnp.isfinite(jnp.diagonal(chol_t)))):
        raise np.linalg.LinAlgError("device Cholesky failed: M + I is not positive definite")
    return WienerFactor(chol_t, pw, var_pix_ref=float(var_pix_ref), lmax=lmax)


def _dense_solve(factor: WienerFactor, b: jax.Array) -> jax.Array:
    """``(M + I)⁻¹`` applied to each row of ``b`` ``(n_rhs, n_dof)``: two triangular solves.

    ``factor.chol_t.T`` is ``L`` in the solvers' column-major layout, so neither solve
    copies the factor (``tests/test_masking.py`` gates the compiled scratch size).
    """
    from jax.scipy.linalg import solve_triangular

    low = jax.lax.stop_gradient(factor.chol_t).T
    z = solve_triangular(low, b.T, lower=True)
    y = solve_triangular(low, z, lower=True, trans="T")
    return y.T


def masked_wiener_eb_alm_dense(
    qu_maps: jax.Array,
    inv_noise: jax.Array,
    factor: WienerFactor,
    *,
    nside: int,
    lmax: int,
    return_info: bool = False,
):
    """Wiener-filtered E/B alm of a stack of Q/U maps from a :func:`wiener_factor`.

    The direct counterpart of :func:`masked_wiener_eb_alm`: the same system, solved for all
    maps at once by two triangular solves. ``qu_maps`` is ``(n, 2, npix)`` (or one
    ``(2, npix)`` map) → ``(n, 2, K)`` (or ``(2, K)``). ``inv_noise`` must be the filter the
    factor encodes, ``inv_noise_map(hits, factor.var_pix_ref, mask=...)``. Differentiable
    in the maps; the factor is held fixed.

    ``return_info=True`` also returns ``jht.SolveInfo`` with ``n_iter`` 0 and the relative
    residual ``‖(M + I) y − P b‖ / ‖P b‖`` re-evaluated with the matrix-free operator (one
    extra operator application per map). That check does not use the factor, so a factor
    built for another mask, hit map or ``var_pix_ref`` shows up as an O(1) residual.
    """
    jht = _require_jht()
    nside, lmax = int(nside), int(lmax)
    if lmax != factor.lmax:
        raise ValueError(f"factor was built at lmax={factor.lmax}, not {lmax}")
    qu = jnp.asarray(qu_maps)
    single = qu.ndim == 2
    legs = qu[None] if single else qu
    w = jnp.asarray(inv_noise)[None, :]
    pw = factor.pw

    def rhs(m):
        return pw * jht.alm_to_real(jht.adjoint_synthesis(w * m, nside, lmax, spin=2), lmax, spin=2)

    b = jax.vmap(rhs)(legs)
    y = _dense_solve(factor, b)
    alm = jax.vmap(lambda v: jht.real_to_alm(pw * v, lmax, spin=2))(y)
    alm = alm[0] if single else alm
    if not return_info:
        return alm
    r = jax.vmap(lambda v: pw * _normal_apply(pw * v, w, nside, lmax) + v)(y) - b
    bn = jnp.linalg.norm(b, axis=1)
    rel = jnp.linalg.norm(r, axis=1) / jnp.where(bn > 0, bn, 1.0)
    rel = rel[0] if single else rel
    info = jht.SolveInfo(n_iter=jnp.zeros_like(rel, dtype=jnp.int32), rel_residual=rel)
    return alm, info


def masked_wiener_bb_dense(
    qu_maps: jax.Array,
    inv_noise: jax.Array,
    factor: WienerFactor,
    *,
    nside: int,
    lmax: int,
    return_info: bool = False,
):
    """Raw per-ℓ ``C_ℓ^{BB}`` of each Wiener-filtered map, via :func:`masked_wiener_eb_alm_dense`.

    ``(n, lmax+1)`` for a ``(n, 2, npix)`` stack, ``(lmax+1,)`` for one map; the same
    filter-suppressed, leakage-contaminated power as :func:`masked_wiener_bb`.
    """
    jht = _require_jht()
    out = masked_wiener_eb_alm_dense(
        qu_maps, inv_noise, factor, nside=nside, lmax=lmax, return_info=return_info
    )
    eb_alm, info = out if return_info else (out, None)

    def bb(a):
        return jht.bandpower(a, int(lmax), spin=2)[1]

    cl = bb(eb_alm) if eb_alm.ndim == 2 else jax.vmap(bb)(eb_alm)
    return (cl, info) if return_info else cl


def normal_matrix_mismatch(normal, inv_noise: jax.Array, *, nside: int, lmax: int,
                           seed: int = 0) -> float:
    """``‖A v − op(v)‖ / ‖op(v)‖`` for one random ``v``: does ``normal`` describe ``inv_noise``?

    ``op`` is the matrix-free normal operator of ``inv_noise``, the one
    :func:`wiener_normal_matrix` samples column by column, so a matrix built for another
    mask or hit map, or at another noise scale, gives O(1). One matrix-vector product and
    one operator application.
    """
    nside, lmax = int(nside), int(lmax)
    n = np.shape(normal)[0]
    v = np.random.default_rng(seed).standard_normal(n)
    av = np.asarray(normal @ v) if isinstance(normal, np.ndarray) else np.asarray(
        jnp.asarray(normal) @ jnp.asarray(v))
    ov = np.asarray(_normal_apply(jnp.asarray(v), jnp.asarray(inv_noise)[None, :], nside, lmax))
    return float(np.linalg.norm(av - ov) / np.linalg.norm(ov))
