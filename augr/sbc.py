"""sbc.py -- simulation-based calibration / frequentist coverage of a sigma(r) interval.

The reusable core shared by the coverage drivers (``scripts/validate_hl_coverage*.py``).
Given the real augr Gaussian + Hamimeche-Lewis likelihood objects built at a fiducial, a
grid over ``r`` that marginalizes the ``A_lens`` / ``A_res`` nuisances against Gaussian
priors, and an iterable of TOTAL-bandpower data realizations, it computes, per realization,
the probability-integral transform ``PIT = F_post(r_true)`` and the frequentist coverage
of the credible interval (two-sided central + one-sided upper limit).

A calibrated interval has ``PIT ~ Uniform(0, 1)``; two-sided central-``L`` coverage =
fraction with ``|PIT - 0.5| <= L/2``; one-sided upper-limit coverage at level ``u`` =
fraction with ``PIT <= u``. The CRN-paired Gaussian-vs-HL difference is the clean statistic.

The two drivers differ only in how the data realizations are produced -- the analytic
``validate_hl_coverage.py`` draws them from an exact scaled-chi^2 oracle, the MC
``validate_hl_coverage_mc.py`` takes them from a real cut-sky masked-Wiener ensemble -- and
in how the likelihood prep (covariance, fiducial bandpower, residual template) is sourced.
Everything common lives here.

The single-cleaned-BB model is **linear** in ``(r, A_lens, A_res)`` (no foreground-index
banana), so the prediction over the ``(r, A_lens, A_res)`` grid is one tensor contraction
of three fixed basis spectra -- precomputed once, trial-independent.

Dependency-light: numpy + jax + equinox only (no ``[sampling]`` extra).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from augr.likelihood.ordering import spectra_to_matrices
from augr.likelihood.protocols import BinnedSpectra
from augr.signal import flatten_params


def _cumtrapz(y, x):
    """Cumulative trapezoid with a leading 0 (same length as ``x``); numpy."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(seg)])


def mode_counts(bin_edges, f_sky):
    """nu_b = sum_{l in [lo, hi]} (2l + 1) * f_sky for each (lo, hi) bin (single field)."""
    return np.array([f_sky * np.sum(2.0 * np.arange(lo, hi + 1) + 1.0) for (lo, hi) in bin_edges])


def data_vector_at(signal_model, names, r, a_lens, a_res):
    """``signal_model.data_vector`` at ``(r, A_lens, A_res)`` as a numpy array."""
    return np.asarray(
        signal_model.data_vector(flatten_params({"r": r, "A_lens": a_lens, "A_res": a_res}, names))
    )


def linear_basis(signal_model, names):
    """The linear-model basis ``(base, t_r, t_L, t_res)`` (model = base + r t_r + A_lens t_L + A_res t_res)."""
    base = data_vector_at(signal_model, names, 0.0, 0.0, 0.0)
    t_r = data_vector_at(signal_model, names, 1.0, 0.0, 0.0) - base
    t_l = data_vector_at(signal_model, names, 0.0, 1.0, 0.0) - base
    t_res = data_vector_at(signal_model, names, 0.0, 0.0, 1.0) - base
    return base, t_r, t_l, t_res


@dataclass(frozen=True)
class NuisanceGrid:
    """Marginalization grid + Gaussian log-priors over ``(A_lens, A_res)``.

    A nuisance not in ``floated`` collapses to a single grid point at its fiducial 1.0, so
    the downstream ``logsumexp`` over its axis is a no-op (conditional case).
    """

    al_axis: np.ndarray  # (n_al,)
    ares_axis: np.ndarray  # (n_ar,)
    logprior_grid: jax.Array  # (n_al, n_ar)

    @property
    def n_al(self) -> int:
        return int(self.al_axis.size)

    @property
    def n_ar(self) -> int:
        return int(self.ares_axis.size)

    @classmethod
    def build(cls, *, floated, prior_sig, n_nuis_grid, n_sigma_nuis) -> NuisanceGrid:
        def axis(name):
            if name in floated:
                s = prior_sig[name]
                return np.linspace(1.0 - n_sigma_nuis * s, 1.0 + n_sigma_nuis * s, n_nuis_grid)
            return np.array([1.0])

        def log_prior(ax, name):
            return -0.5 * ((ax - 1.0) / prior_sig[name]) ** 2 if name in floated else np.zeros_like(ax)

        al, ar = axis("A_lens"), axis("A_res")
        lpg = jnp.asarray(log_prior(al, "A_lens")[:, None] + log_prior(ar, "A_res")[None, :])
        return cls(al_axis=al, ares_axis=ar, logprior_grid=lpg)


def build_pred_grid(base, t_r, t_l, t_res, *, r_grid, al_axis, ares_axis):
    """Linear prediction grid flattened to ``(n_grid * n_al * n_ar, n_bins)`` (trial-independent)."""
    nnn = (None, None, None, slice(None))
    pred = (
        base[nnn]
        + np.asarray(r_grid)[:, None, None, None] * t_r[nnn]
        + np.asarray(al_axis)[None, :, None, None] * t_l[nnn]
        + np.asarray(ares_axis)[None, None, :, None] * t_res[nnn]
    )
    return jnp.asarray(pred.reshape(-1, base.size))


def marginal_sigma_r(*, t_r, t_l, t_res, cov_diag, floated, prior_sig) -> float:
    """Fisher marginal sigma(r) over (r + floated nuisances) on a diagonal covariance.

    Used only to size the r-grid; the diagonal cov is the analytic Knox diagonal or
    ``diag(MC covariance)``.
    """
    cov_diag = np.asarray(cov_diag)
    cols, prior_diag = [t_r], [0.0]
    if "A_lens" in floated:
        cols.append(t_l)
        prior_diag.append(1.0 / prior_sig["A_lens"] ** 2)
    if "A_res" in floated:
        cols.append(t_res)
        prior_diag.append(1.0 / prior_sig["A_res"] ** 2)
    design = np.stack(cols, axis=1)
    fisher = design.T @ (design / cov_diag[:, None]) + np.diag(prior_diag)
    return float(np.sqrt(np.linalg.inv(fisher)[0, 0]))


def make_marginal_logpost(
    *, gauss0, hl0, noise_floor, layout, pred_flat, logprior_grid, n_grid, n_al, n_ar
) -> Callable[[jax.Array], jax.Array]:
    """Return a JITted ``trial_core(realization_total) -> (n_grid, 2)`` marginal r log-posterior.

    ``realization_total`` is the TOTAL bandpower ``(n_bins,)`` (S + N + residual). The
    realization is swapped into the pre-built likelihoods via ``eqx.tree_at`` -- the Gaussian
    works on the noise-debiased signal estimate (``data = total - noise_floor``, matching its
    signal-only mean), HL on the total (``data_matrices``). The marginal over the
    ``(A_lens, A_res)`` grid is ``logsumexp`` against the Gaussian priors. Columns are
    ``[Gaussian, HL]``.
    """
    n_b_j = jnp.asarray(noise_floor)

    @jax.jit
    def trial_core(realization_total):
        gauss = eqx.tree_at(lambda g: g.data, gauss0, realization_total - n_b_j)
        hl = eqx.tree_at(lambda h: h.data_matrices, hl0, spectra_to_matrices(realization_total, layout))

        def both(cl):
            pred_bs = BinnedSpectra(cl=cl, layout=layout)
            return jnp.array([gauss.log_prob(pred_bs), hl.log_prob(pred_bs)])

        ll = jax.vmap(both)(pred_flat).reshape(n_grid, n_al, n_ar, 2)
        return logsumexp(ll + logprior_grid[None, :, :, None], axis=(1, 2))

    return trial_core


@dataclass(frozen=True)
class CoverageResult:
    pit: dict[str, np.ndarray]  # {label: (n_trials,)}
    edge_hits: dict[str, int]  # grid-edge PIT pile-up counts per label
    r_grid: np.ndarray
    n_trials: int
    labels: tuple[str, ...]


def run_coverage(
    marginal_logpost: Callable[[jax.Array], jax.Array],
    realizations: Iterable,
    *,
    r_true: float,
    r_grid: np.ndarray,
    n_trials: int,
    labels: tuple[str, ...] = ("gauss", "hl"),
    edge_tol: float = 1e-6,
) -> CoverageResult:
    """The PIT loop: for each realization, normalize the marginal r log-post to a CDF and
    record ``PIT = F_post(r_true)``. ``realizations`` is any iterable of ``(n_bins,)`` totals
    (analytic chi^2 draws or real MC test bandpowers)."""
    pit = {tag: np.empty(n_trials) for tag in labels}
    edge_hits = {tag: 0 for tag in labels}
    for i, real in enumerate(realizations):
        if i >= n_trials:
            break
        marg = np.asarray(marginal_logpost(jnp.asarray(real)))  # (n_grid, n_labels)
        for k, tag in enumerate(labels):
            w = np.exp(marg[:, k] - marg[:, k].max())
            cdf = _cumtrapz(w, r_grid)
            cdf /= cdf[-1]
            p = float(np.interp(r_true, r_grid, cdf))
            pit[tag][i] = p
            edge_hits[tag] += int(p <= edge_tol or p >= 1 - edge_tol)
    return CoverageResult(
        pit=pit, edge_hits=edge_hits, r_grid=np.asarray(r_grid), n_trials=n_trials, labels=tuple(labels)
    )


def coverage_table(result: CoverageResult, *, levels=(0.68, 0.95), upper_level=0.95):
    """Rows ``[(label, {tag: (coverage, mc_err)})]``: two-sided central + one-sided UL."""
    n = result.n_trials

    def mc_err(p):
        return float(np.sqrt(p * (1 - p) / n))

    rows = []
    for level in levels:
        cov = {tag: float(np.mean(np.abs(result.pit[tag] - 0.5) <= level / 2)) for tag in result.labels}
        rows.append((f"two-sided {level:.2f}", {tag: (cov[tag], mc_err(cov[tag])) for tag in result.labels}))
    cov = {tag: float(np.mean(result.pit[tag] <= upper_level)) for tag in result.labels}
    rows.append((f"one-sided UL {upper_level:.2f}", {tag: (cov[tag], mc_err(cov[tag])) for tag in result.labels}))
    return rows


def print_coverage_table(rows, *, header=("Gaussian", "HL"), keys=("gauss", "hl")):
    """Print the coverage table block (header + rule + rows), shared by both drivers."""
    print(f"\n  {'quantity':>26} | {header[0]:>17} | {header[1]:>17}")
    print(f"  {'-' * 26}-+-{'-' * 17}-+-{'-' * 17}")
    for label, cov in rows:
        g_c, g_e = cov[keys[0]]
        h_c, h_e = cov[keys[1]]
        print(f"  {label:>26} | {g_c:>8.3f} +/-{g_e:.3f} | {h_c:>8.3f} +/-{h_e:.3f}")


def make_coverage_plot(path, result: CoverageResult, *, title: str | None = None):
    """Two-panel coverage-curve + PIT-histogram figure (optional suptitle)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pit = result.pit
    fig, (ax_cov, ax_pit) = plt.subplots(1, 2, figsize=(11, 4.4))
    if title:
        fig.suptitle(title, fontsize=11)
    nominal = np.linspace(0.02, 0.98, 49)
    styles = (("gauss", "C3", "Gaussian / Knox"), ("hl", "C0", "Hamimeche-Lewis"))
    for tag, color, label in styles:
        emp = [np.mean(np.abs(pit[tag] - 0.5) <= level / 2) for level in nominal]
        ax_cov.plot(nominal, emp, color=color, label=label, lw=2)
    ax_cov.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
    ax_cov.set(
        xlabel="nominal credible level",
        ylabel="empirical coverage",
        title="Coverage of the r interval",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax_cov.legend(loc="upper left", fontsize=9)

    bins = np.linspace(0, 1, 21)
    for tag, color, label in styles:
        ax_pit.hist(pit[tag], bins=bins, histtype="step", color=color, lw=2, density=True, label=label)
    ax_pit.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
    ax_pit.set(
        xlabel="PIT = F_post(r_true)",
        ylabel="density",
        title="Probability-integral transform",
        xlim=(0, 1),
    )
    ax_pit.legend(loc="upper center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    fig.savefig(path, dpi=130)
