"""mc_calibrated.py — a KS calibration of the cleaned-bandpower ensemble + a
moment-matched offset-lognormal cross-check likelihood.

The analytic Hamimeche-Lewis likelihood (:mod:`augr.likelihood.hl`) assumes the
per-bin bandpower estimator is a scaled chi-squared (offset by the noise) — the
right low-mode-count shape *if* the cleaned bandpowers actually follow it. After
masked-Wiener component separation that is an assumption, not a theorem: the
mask-mode coupling and E->B leakage can distort the per-bin distribution. So
rather than assume HL is adequate, this module **tests** it.

:func:`bandpower_ks` runs a per-bin Kolmogorov-Smirnov test of the Monte-Carlo
debiased-bandpower ensemble (``CutskyMC.debiased_bandpowers``) against (a) the
Gaussian ``N(mean_b, var_b)`` that the Knox/Fisher path assumes and (b) the
scaled-chi-squared ``(mean_b / nu_b) * chi2_{nu_b}`` with ``nu_b = 2 mean_b^2 /
var_b`` effective modes that the HL likelihood implies. Whichever has the smaller
KS statistic per bin is the better-calibrated form; the low-ell (reionization
bump) bins are where it matters for sigma(r). This is the *decider* for whether
the analytic HL headline is adequate or the MC-calibrated form is needed
(cf. the "show the PDF / run a KS test, don't compare means-vs-error-bar"
discipline).

:class:`MCCalibratedLikelihood` is the lightweight cross-check it decides on: a
moment-matched per-bin offset-lognormal (Bond, Jaffe & Knox 2000) built directly
from the ensemble's mean + covariance, with the noise floor as the offset. It is
a genuine non-Gaussian likelihood (asymmetric in ``C_b``) but, unlike analytic
HL, calibrated to the realized ensemble rather than the chi-squared idealization.
It is *single-field* only (one cleaned BB map): the log transform of a per-bin
cross-spectrum matrix is ill-defined for ``n_field > 1``.

``mean_bandpower`` is the **total** debiased bandpower ``S + N + residual`` (the
masked-Wiener debias does not subtract a noise bias — see
``augr.masking.debias_bandpower``), so the debiased ensemble is the total
bandpower and the chi-squared / lognormal reference needs no separate offset for
the *distributional* (KS) test; the noise offset only enters the *likelihood*,
where the model signal varies while the noise floor is held fixed.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from augr.likelihood.hl import _dense_cov_inv
from augr.likelihood.ordering import SpectrumLayout
from augr.likelihood.protocols import BinnedSpectra

# Floor on the model total inside the log: keeps log_prob + its gradient finite
# if a sampler step pushes the modelled total bandpower non-positive (well away
# from the bulk, where the model total = signal + noise floor is comfortably
# positive). Mirrors the differentiate-through-floor pattern in hl._safe_g.
_LOG_FLOOR = 1e-30

# The chi^2/HL shape counts as a *materially* better description of the bump only
# when its summed KS statistic is at least this much smaller than the Gaussian's.
# At high effective mode count the two shapes coincide (chi^2 -> Gaussian) and the
# per-bin KS difference is pure draw noise (measured ratio 0.89-1.26 on Gaussian
# ensembles vs 0.48 on a genuinely chi^2 ensemble), so the margin stops an unlucky
# bin from escalating the verdict away from "gaussian".
_HL_BETTER_MARGIN = 0.7


def bandpower_ks(ensemble, mean, cov, *, alpha: float = 0.05, n_bump_bins: int = 3) -> dict:
    """Per-bin KS calibration of the cleaned-bandpower ensemble (Gaussian vs HL chi^2).

    Parameters
    ----------
    ensemble
        ``(n_sims, n_bins)`` debiased bandpowers (``CutskyMC.debiased_bandpowers``).
    mean
        ``(n_bins,)`` ensemble-mean total bandpower (``CutskyMC.mean_bandpower``).
    cov
        ``(n_bins, n_bins)`` bandpower covariance; only its diagonal (per-bin
        variance) is used for the marginal KS tests.
    alpha
        Rejection level for the bump-band verdict (default 0.05), Bonferroni-corrected
        to ``alpha / n_bump_bins`` so a single noisy bin cannot flip the headline.
    n_bump_bins
        How many lowest-ell bins count as the reionization-bump band for the
        headline verdict (default 3): these are the low-mode-count bins where the
        non-Gaussian widening lives and where the HL-vs-Gaussian choice moves
        sigma(r).

    Returns
    -------
    dict with per-bin arrays ``ks_gauss`` / ``p_gauss`` / ``ks_chi2`` / ``p_chi2``
    / ``nu_eff`` / ``hl_preferred`` (chi^2 KS < Gaussian KS), the bump-band
    summaries, and a ``recommend`` string in ``{"gaussian", "hl",
    "mc_calibrated"}``:

    * ``"gaussian"`` — the Gaussian is *not* rejected in the bump band, or the
      chi^2/HL form is no better there (high effective mode count): the
      non-Gaussian widening is negligible and Knox/Fisher suffices.
    * ``"hl"`` — the Gaussian is rejected, the chi^2/HL form is materially better
      *and* survives (not rejected) in the bump band: the analytic HL headline is
      the right call.
    * ``"mc_calibrated"`` — the Gaussian is rejected, chi^2/HL is the better shape
      but is *also* rejected in the bump band: the idealized HL shape is
      inadequate, prefer the ensemble-calibrated :class:`MCCalibratedLikelihood`.
    """
    from scipy import stats

    ens = np.asarray(ensemble)
    m = np.asarray(mean, dtype=float)
    var = np.diag(np.asarray(cov, dtype=float))
    n_bins = ens.shape[1]
    if m.shape[0] != n_bins or var.shape[0] != n_bins:
        raise ValueError(
            f"shape mismatch: ensemble has {n_bins} bins, mean has {m.shape[0]}, "
            f"cov diag has {var.shape[0]}."
        )

    ks_gauss = np.full(n_bins, np.nan)
    p_gauss = np.full(n_bins, np.nan)
    ks_chi2 = np.full(n_bins, np.nan)
    p_chi2 = np.full(n_bins, np.nan)
    nu_eff = np.full(n_bins, np.nan)

    for b in range(n_bins):
        sd = np.sqrt(var[b]) if var[b] > 0 else np.nan
        if np.isfinite(sd) and sd > 0:
            ks_gauss[b], p_gauss[b] = stats.kstest(ens[:, b], "norm", args=(m[b], sd))
        # Scaled chi^2 with nu_b effective modes: mean m_b, var 2 m_b^2 / nu_b, so
        # nu_b = 2 m_b^2 / var_b and scale = m_b / nu_b. The debiased ensemble is the
        # total bandpower (debias does not subtract noise), so no loc offset.
        if var[b] > 0 and m[b] > 0:
            nu_b = 2.0 * m[b] ** 2 / var[b]
            nu_eff[b] = nu_b
            ks_chi2[b], p_chi2[b] = stats.kstest(ens[:, b], "chi2", args=(nu_b, 0.0, m[b] / nu_b))

    hl_preferred = ks_chi2 < ks_gauss
    n_bump = min(n_bump_bins, n_bins)
    bump = slice(0, n_bump)
    # Bonferroni-correct the bump-band rejection so a single noisy bin can't flip the
    # headline verdict (3 bump bins => a true-Gaussian ensemble rejects ~14% of the
    # time at a naive per-bin alpha=0.05; alpha/n_bump protects against that).
    alpha_bump = alpha / max(n_bump, 1)
    gauss_rejected_bump = bool(np.nanmin(p_gauss[bump]) < alpha_bump)
    chi2_rejected_bump = bool(np.nanmin(p_chi2[bump]) < alpha_bump)
    # chi^2/HL must be the materially-better description of the bump (clearly smaller
    # summed KS) to escalate beyond Gaussian -- otherwise the two shapes coincide.
    sum_ks_gauss = float(np.nansum(ks_gauss[bump]))
    sum_ks_chi2 = float(np.nansum(ks_chi2[bump]))
    hl_better_bump = sum_ks_chi2 < _HL_BETTER_MARGIN * sum_ks_gauss

    if not (gauss_rejected_bump and hl_better_bump):
        # Gaussian is adequate at the bump (or chi^2 is no better) -> Gaussian headline.
        recommend = "gaussian"
    elif chi2_rejected_bump:
        # Both idealized forms fail -> use the ensemble-calibrated form.
        recommend = "mc_calibrated"
    else:
        # Gaussian rejected, chi^2/HL is materially better and survives -> analytic HL.
        recommend = "hl"

    return {
        "ks_gauss": ks_gauss,
        "p_gauss": p_gauss,
        "ks_chi2": ks_chi2,
        "p_chi2": p_chi2,
        "nu_eff": nu_eff,
        "hl_preferred": hl_preferred,
        "hl_preferred_fraction": float(np.mean(hl_preferred)),
        "gauss_rejected_bump": gauss_rejected_bump,
        "chi2_rejected_bump": chi2_rejected_bump,
        "hl_better_bump": hl_better_bump,
        "sum_ks_gauss_bump": sum_ks_gauss,
        "sum_ks_chi2_bump": sum_ks_chi2,
        "n_bump_bins": int(n_bump),
        "alpha": float(alpha),
        "recommend": recommend,
    }


class MCCalibratedLikelihood(eqx.Module):
    """Moment-matched per-bin offset-lognormal (BJK00) over a single cleaned BB map.

    The Bond-Jaffe-Knox 2000 offset-lognormal: ``z_b = ln(C_b^total) `` is treated
    as Gaussian, so the log-likelihood is a quadratic form in
    ``z_b = ln(mean_b) - ln(model_total_b)`` with the log-space covariance
    ``Sigma_z = diag(1/mean) Sigma diag(1/mean)`` (the delta-method covariance of
    ``ln C`` at the fiducial). ``model_total_b = data_vector(theta)_b + N_b`` adds
    the held-fixed noise floor to the modelled signal, exactly as
    :class:`~augr.likelihood.hl.HLLikelihood` does, so at the fiducial
    ``model_total = mean`` and ``z = 0`` — the likelihood peaks at the fiducial.

    Unlike the analytic HL (which assumes a scaled chi^2 per bin), this is
    calibrated to the realized ensemble's mean + covariance; it is the cross-check
    that :func:`bandpower_ks` promotes to the headline only when the chi^2/HL form
    is rejected. Single-field only (``n_field == 1``): ``ln`` of a cross-spectrum
    matrix is ill-defined.
    """

    mean: jax.Array  # (n_bins,) total Asimov bandpower
    noise: jax.Array  # (n_bins,) fiducial noise floor N_b (offset added to the model)
    cov_z_inv: jax.Array  # (n_bins, n_bins) inverse log-space covariance
    log_mean: jax.Array  # (n_bins,) ln(mean), precomputed
    layout: SpectrumLayout = eqx.field(static=True)

    def residual_vector(self, prediction: BinnedSpectra) -> jax.Array:
        model_total = prediction.as_vector() + self.noise
        model_total = jnp.maximum(model_total, _LOG_FLOOR)
        return self.log_mean - jnp.log(model_total)

    def log_prob(self, prediction: BinnedSpectra) -> jax.Array:
        z = self.residual_vector(prediction)
        return -0.5 * z @ self.cov_z_inv @ z

    @classmethod
    def from_external(
        cls,
        signal_model,
        fiducial_params: jax.Array,
        total_bandpower: jax.Array,
        covariance: jax.Array,
    ) -> MCCalibratedLikelihood:
        """Build from external MC outputs (``CutskyMC.mean_bandpower`` / ``.covariance``).

        ``total_bandpower`` is the total (S + N + residual) Asimov bandpower
        ``(n_bins,)``; the noise floor is ``N_b = total - data_vector(fid)``
        (see module docstring). Raises if the signal model is not single-field.
        """
        fid = jnp.asarray(fiducial_params)
        total = jnp.asarray(total_bandpower)
        layout = SpectrumLayout.from_freq_pairs(signal_model.freq_pairs, signal_model.n_bins)
        if layout.n_field != 1:
            raise ValueError(
                "MCCalibratedLikelihood supports single-field (n_field=1) cleaned maps "
                f"only; got n_field={layout.n_field}."
            )
        noise = total - signal_model.data_vector(fid)
        cov = jnp.asarray(covariance)
        inv_mean = 1.0 / total
        cov_z = cov * inv_mean[:, None] * inv_mean[None, :]
        cov_z_inv = _dense_cov_inv(cov_z)
        return cls(
            mean=total,
            noise=noise,
            cov_z_inv=cov_z_inv,
            log_mean=jnp.log(total),
            layout=layout,
        )
