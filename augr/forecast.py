"""forecast.py — post-component-separation spectra → σ(r) variants + the FG bias Δr.

The cleaner-agnostic core shared by every post-separation forecast: given the
beam-free B-mode spectra of a *single cleaned map* — post-separation noise
``N_ℓ^{BB}`` and a residual-foreground template ``ΔC_ℓ`` — build
``cleaned_map_instrument`` + ``NullForegroundModel`` + a residual-template
``SignalModel`` and run the three :class:`augr.fisher.FisherForecast` variants
(baseline / flat-``A_res`` / Gaussian-``A_res``) plus the debias-OFF
``bias_from_truth_model`` Δr.

It takes plain arrays, not a particular cleaner's result object, so the same
function backs the in-house NILC/cMILC path (via
:func:`augr.nilc_forecast.nilc_forecast`), the BROOM-npy consumer
(``scripts/validate_carones.py``), and the :mod:`augr.pipeline` driver — instead
of each re-deriving the ``SignalModel``/``FisherForecast`` wiring. Imports stay on
the light Fisher path (no ``[compsep]`` deps), so a Fisher-only consumer with just
loaded spectra never pulls the cleaner/SHT modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np

from .config import DEFAULT_PRIORS_POST_COMPSEP, cleaned_map_instrument
from .fisher import FisherForecast
from .foregrounds import NullForegroundModel
from .signal import SignalModel
from .spectra import CMBSpectra


@dataclass(frozen=True)
class ForecastResult:
    """σ(r) variants + the FG-leakage bias Δr from a cleaned-map spectrum set.

    ``sigma_r_baseline`` is the residual-unmodelled fit; ``sigma_r_flat`` /
    ``sigma_r_gauss`` add the ``A_res`` residual-template amplitude with a flat /
    Gaussian prior. ``delta_r`` is the debias-OFF linear bias on the baseline fit
    induced by the unmodelled residual. ``cond_r_Ares`` is the marginalized
    (r, A_res) condition number (``inf`` if A_res is unconstrained, e.g. a zero
    residual template). :meth:`as_dict` reproduces the legacy ``nilc_forecast``
    dict for back-compat.
    """

    sigma_r_baseline: float
    sigma_r_flat: float
    sigma_r_gauss: float
    sigma_A_res_flat: float
    sigma_A_res_gauss: float
    delta_r: float
    transfer_mean: float
    cond_r_Ares: float
    a_res_prior: float
    r_fid: float

    def as_dict(self) -> dict:
        """The result as a plain dict (legacy ``nilc_forecast`` return shape)."""
        return asdict(self)


def forecast_from_spectra(
    *,
    nl_ells=None,
    nl_post=None,
    template_ells,
    template_cl,
    f_sky: float,
    transfer=None,
    r_fid: float = 0.0,
    ell_min: int = 2,
    ell_max: int = 180,
    delta_ell: int = 5,
    ell_per_bin_below: int = 30,
    a_res_prior: float | None = None,
    apply_transfer: bool = False,
    delensed_bb=None,
    delensed_bb_ells=None,
    external_covariance=None,
) -> ForecastResult:
    """σ(r) variants + Δr from beam-free cleaned-map noise + residual-template spectra.

    The noise and the residual template may live on *different* ℓ grids (e.g. the
    in-house path has both on ``0..lmax`` while the BROOM-npy consumer has each on
    its own bandpower bin centres): the noise is interpolated onto the
    ``SignalModel`` grid, and the template is handed to ``SignalModel`` which does
    its own (nearest-neighbour-extrapolated) interpolation.

    Parameters
    ----------
    nl_ells, nl_post
        Beam-free post-separation noise ``N_ℓ^{BB}`` and its multipoles →
        ``external_noise_bb`` (interpolated onto the ``SignalModel`` grid). Required
        for the analytic-Knox path; ignored (and may be ``None``) when
        ``external_covariance`` is supplied.
    external_covariance
        Optional full ``(n_bins, n_bins)`` bandpower covariance (e.g. the cut-sky
        masked-Wiener Monte-Carlo covariance from
        :func:`augr.spectrum_stages.mc_cutsky_bandpowers`). When given, the analytic
        Knox / ``external_noise_bb`` path is bypassed for *all three* Fisher variants
        (and the Δr bias) — the same covariance applies to the baseline and the
        ``A_res`` models since they share the binning. The masked-Wiener transfer is
        already folded into the debiased bandpowers' covariance, so ``transfer`` /
        ``apply_transfer`` are ignored in this mode (no double-counting).
    template_ells, template_cl
        Beam-free residual-foreground ``ΔC_ℓ`` and its multipoles → the ``A_res``
        template and the Δr truth-vs-fit mismatch.
    f_sky
        Observed sky fraction (Knox mode count via ``cleaned_map_instrument``).
    transfer
        Optional signal-loss transfer on the ``nl_ells`` grid; its band-mean is
        reported and (if ``apply_transfer``) divided through ``nl_post`` /
        ``template_cl``. ``None`` → ``transfer_mean = 1.0`` (no signal loss
        assumed; the case for externally loaded spectra without a transfer).
    delensed_bb, delensed_bb_ells
        Optional self-consistent residual lensing ``C_ℓ^{BB}`` replacing the
        ``A_lens`` multiplier; must be supplied together and span
        ``[ell_min, ell_max]``.
    """
    if a_res_prior is None:
        a_res_prior = DEFAULT_PRIORS_POST_COMPSEP["A_res"]
    if (delensed_bb is None) != (delensed_bb_ells is None):
        raise ValueError("delensed_bb and delensed_bb_ells must be supplied together.")
    use_external_cov = external_covariance is not None
    if not use_external_cov and nl_post is None:
        raise ValueError(
            "forecast_from_spectra needs nl_post (the post-separation noise spectrum) "
            "for the analytic-Knox path, or an external_covariance for the cut-sky "
            "Monte-Carlo path."
        )

    tcl = np.asarray(template_cl)
    if use_external_cov:
        # The masked-Wiener transfer is already in the MC covariance; do not apply a
        # (separate) signal-loss transfer here, and there is no nl to interpolate.
        transfer_mean = 1.0
    else:
        nl_ells_arr = np.asarray(nl_ells)
        nl = np.asarray(nl_post)
        if transfer is None:
            transfer_mean = 1.0
        else:
            band = (nl_ells_arr >= ell_min) & (nl_ells_arr <= ell_max)
            transfer_mean = float(np.mean(np.asarray(transfer)[band]))
        if apply_transfer and transfer_mean > 0:
            nl = nl / transfer_mean
            tcl = tcl / transfer_mean

    inst = cleaned_map_instrument(f_sky=f_sky)
    cmb = CMBSpectra()

    def _signal(with_template):
        kw = dict(
            instrument=inst,
            foreground_model=NullForegroundModel(),
            cmb_spectra=cmb,
            ell_min=ell_min,
            ell_max=ell_max,
            delta_ell=delta_ell,
            ell_per_bin_below=ell_per_bin_below,
        )
        if with_template:
            kw["residual_template_cl"] = jnp.asarray(tcl)
            kw["residual_template_ells"] = jnp.asarray(template_ells, dtype=float)
        if delensed_bb is not None:
            kw["delensed_bb"] = jnp.asarray(delensed_bb)
            kw["delensed_bb_ells"] = jnp.asarray(delensed_bb_ells, dtype=float)
        return SignalModel(**kw)

    baseline = _signal(with_template=False)
    if use_external_cov:
        # Same (n_bins, n_bins) covariance for baseline and A_res models (shared bins);
        # FisherForecast validates the shape against the data-vector length.
        cov_kw = {"external_covariance": jnp.asarray(external_covariance)}
    else:
        nl_interp = jnp.interp(
            baseline.ells, jnp.asarray(nl_ells_arr, dtype=float), jnp.asarray(nl)
        )
        cov_kw = {"external_noise_bb": nl_interp[None, :]}

    fid_base = {"r": r_fid, "A_lens": 1.0}
    fisher_baseline = FisherForecast(
        baseline, inst, fid_base, priors={}, fixed_params=[], **cov_kw
    )
    fisher_baseline.compute()

    signal = _signal(with_template=True)
    fid = {**fid_base, "A_res": 1.0}
    fisher_flat = FisherForecast(signal, inst, fid, priors={}, fixed_params=[], **cov_kw)
    fisher_flat.compute()
    fisher_gauss = FisherForecast(
        signal,
        inst,
        fid,
        priors={"A_res": a_res_prior},
        fixed_params=[],
        **cov_kw,
    )
    fisher_gauss.compute()

    # Δr (debias-OFF): bias on the baseline fit from the unmodelled residual.
    # truth = baseline + residual template at A_res=1; ΔD = the binned residual.
    delta = fisher_baseline.bias_from_truth_model(signal, fid)

    # (r, A_res) marginalized condition number. A zero residual template (no FG)
    # leaves A_res unconstrained -> F singular -> inf.
    F = np.asarray(fisher_flat.fisher_matrix)
    names = fisher_flat.free_parameter_names
    ix = np.array([names.index("r"), names.index("A_res")])
    try:
        cond = float(np.linalg.cond(np.linalg.inv(F)[np.ix_(ix, ix)]))
    except np.linalg.LinAlgError:
        cond = float("inf")

    return ForecastResult(
        sigma_r_baseline=fisher_baseline.sigma("r"),
        sigma_r_flat=fisher_flat.sigma("r"),
        sigma_r_gauss=fisher_gauss.sigma("r"),
        sigma_A_res_flat=fisher_flat.sigma("A_res"),
        sigma_A_res_gauss=fisher_gauss.sigma("A_res"),
        delta_r=delta["r"],
        transfer_mean=transfer_mean,
        cond_r_Ares=cond,
        a_res_prior=a_res_prior,
        r_fid=r_fid,
    )
