"""Stochastic design optimization with held-out validation.

Shared harness for optimizing an instrument design *through* the differentiable
map-based Monte-Carlo component-separation forward
(:func:`augr.spectrum_stages.mc_cutsky_cov_traced`), where the objective is a
function of a *sample* covariance and therefore carries Monte-Carlo noise.

Fixed common-random-numbers (CRN) full-batch descent run to convergence overfits
the finite sample: the optimizer walks to a design that exploits *that* sample's
empirical-ILC-bias / covariance-estimation noise rather than one that helps on
average. The fix here is **stochastic approximation with re-randomized CRN** -- a
fresh sim ensemble every ``resample_every`` steps under an Adam-style optimizer,
so the optimizer never sees the same realization twice and cannot fit any single
one. Validation tracking on a *separate fixed* ensemble + best-on-val early
stopping (:class:`DesignDescentResult`) gives honest model selection on top.

The objective is supplied as ``loss_fn(params, ctx) -> scalar``, differentiable
in ``params`` (argument 0); ``ctx`` is the traced Monte-Carlo ensemble (a PR #28
:class:`~augr.spectrum_stages.CutskyMCContext`, an ``eqx.Module``). Because
``loss_fn`` is wrapped in ``eqx.filter_jit`` over ``(params, ctx)``, re-drawing
the ensemble at fixed ``(n_sims, nside, lmax)`` reuses the *one* compiled
executable -- re-randomizing the CRN every step is then nearly free.

Two consumers share this loop: the allocation-descent diagnostic
(``scripts/mapbased_grad_characterization.py``) and the B.3 cost-constrained EIG
design driver -- both expose the same overfit risk through the same forward.

Usage:
    import optax
    from augr.design_opt import build_design_objectives, stochastic_design_descent, held_out_gain

    def loss_fn(params, ctx):
        # params -> instrument design -> sigma(r) / -EIG through mc_cutsky_cov_traced(ctx)
        ...

    value_fn, vg_fn = build_design_objectives(loss_fn)
    result = stochastic_design_descent(
        value_fn, vg_fn, init_params,
        make_train_ctx=lambda i: build_ensemble(train_base_seed(i)),  # disjoint seeds
        val_ctx=build_ensemble(val_base_seed),
        optimizer=optax.adam(5e-2), steps=60, resample_every=1)
    gains = held_out_gain(value_fn, test_ctxs, baseline_params, result.params_best)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class DesignDescentResult:
    """Outcome of :func:`stochastic_design_descent`.

    Attributes:
        params_final:  The design after the last optimizer step.
        params_best:   The best-on-validation design (early stopping) -- the one
                       to report; the final iterate may sit past the validation
                       minimum.
        n_to_best:     Step index at which ``val_curve`` was minimal.
        steps:         ``arange(n_steps)``.
        train_curve:   Objective on the (re-randomized) train ensemble per step.
        val_curve:     Objective on the fixed validation ensemble per step (same
                       ``params`` as ``train_curve`` at that step).
        per_eval_s:    Steady-state value+grad wall time (median past the first,
                       compile-bearing, eval).
    """

    params_final: Any
    params_best: Any
    n_to_best: int
    steps: np.ndarray
    train_curve: np.ndarray
    val_curve: np.ndarray
    per_eval_s: float


def build_design_objectives(
    loss_fn: Callable[[Any, Any], jnp.ndarray],
) -> tuple[Callable[[Any, Any], jnp.ndarray], Callable[[Any, Any], tuple[jnp.ndarray, Any]]]:
    """``eqx.filter_jit`` the objective into ``(value_fn, value_and_grad_fn)``.

    Both are jitted over ``(params, ctx)`` so swapping in a fresh CRN ensemble of
    the same static shape reuses the one compiled executable. The gradient is
    w.r.t. ``params`` (argument 0) only; ``ctx`` flows through untouched. Build
    them once and reuse ``value_fn`` for validation *and* :func:`held_out_gain`
    so every evaluation shares the same compiled trace.
    """
    value_fn = eqx.filter_jit(loss_fn)
    vg_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
    return value_fn, vg_fn


def stochastic_design_descent(
    value_fn: Callable[[Any, Any], jnp.ndarray],
    vg_fn: Callable[[Any, Any], tuple[jnp.ndarray, Any]],
    init_params: Any,
    *,
    make_train_ctx: Callable[[int], Any],
    val_ctx: Any,
    optimizer: optax.GradientTransformation,
    steps: int,
    resample_every: int = 1,
    on_step: Callable[[int, Any, float, float], None] | None = None,
) -> DesignDescentResult:
    """Adam-style descent with re-randomized CRN + validation-gated selection.

    Args:
        value_fn, vg_fn:  From :func:`build_design_objectives` (the jitted
            objective and its value-and-grad).
        init_params:      Initial design pytree (all inexact arrays; optax + the
            ``eqx.filter`` gradient both key on the inexact-array leaves).
        make_train_ctx:   ``resample_index -> ctx``. Called once per resample
            (every ``resample_every`` steps) with ``step // resample_every``; the
            caller maps the index to a *disjoint* CRN ensemble (fresh seed block,
            disjoint from ``val_ctx`` and any held-out test ensembles).
        val_ctx:          Fixed validation ensemble (its own disjoint seed block).
        optimizer:        An ``optax`` transformation, e.g. ``optax.adam(lr)``.
        steps:            Number of optimizer steps.
        resample_every:   Re-draw the train ensemble every ``K`` steps (1 = fresh
            CRN each step, the strongest anti-overfit setting).
        on_step:          Optional ``callback(step, params, train_loss, val_loss)``.

    Returns:
        A :class:`DesignDescentResult`. Report ``params_best`` (best-on-val), not
        ``params_final``.
    """
    params = init_params
    state = optimizer.init(params)
    train_curve: list[float] = []
    val_curve: list[float] = []
    eval_times: list[float] = []
    best_val, best_params = np.inf, params
    train_ctx = None

    for step in range(steps):
        if step % resample_every == 0:
            train_ctx = make_train_ctx(step // resample_every)
        t0 = time.time()
        s_tr, grad = vg_fn(params, train_ctx)
        eval_times.append(time.time() - t0)
        s_tr = float(s_tr)
        s_va = float(value_fn(params, val_ctx))
        train_curve.append(s_tr)
        val_curve.append(s_va)
        if s_va < best_val:
            best_val, best_params = s_va, params
        if on_step is not None:
            on_step(step, params, s_tr, s_va)
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)

    per_eval_s = float(np.median(eval_times[1:])) if len(eval_times) > 1 else float(eval_times[0])
    return DesignDescentResult(
        params_final=params,
        params_best=best_params,
        n_to_best=int(np.argmin(np.asarray(val_curve))),
        steps=np.arange(steps),
        train_curve=np.asarray(train_curve),
        val_curve=np.asarray(val_curve),
        per_eval_s=per_eval_s,
    )


def held_out_gain(
    value_fn: Callable[[Any, Any], jnp.ndarray],
    test_ctxs: list[Any],
    baseline_params: Any,
    params: Any,
) -> np.ndarray:
    """Per-test-ensemble percent improvement of ``params`` over ``baseline_params``.

    ``gain = 100 * (loss(baseline) - loss(params)) / loss(baseline)`` on each
    disjoint held-out ensemble in ``test_ctxs``. Positive on every ensemble means
    the design generalizes; a negative entry means it does not improve there. Use
    the same ``value_fn`` returned by :func:`build_design_objectives` so the
    evaluations reuse the compiled trace.
    """
    gains = []
    for ctx in test_ctxs:
        s_base = float(value_fn(baseline_params, ctx))
        s_opt = float(value_fn(params, ctx))
        gains.append(100.0 * (s_base - s_opt) / s_base)
    return np.asarray(gains)
