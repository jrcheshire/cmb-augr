"""Fast unit tests for the stochastic design-optimization helper.

A synthetic quadratic loss stands in for the heavy map-based sigma(r) / EIG
objective: ``loss(params, ctx) = ||params - ctx||^2`` where ``ctx`` plays the
role of the per-CRN Monte-Carlo ensemble (a noisy target for training, the clean
target for validation). This exercises the loop mechanics -- optax integration,
re-randomized-CRN resampling, validation tracking, best-on-val selection, and the
``eqx.filter_jit`` over ``(params, ctx)`` -- without the cut-sky compsep forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from augr.design_opt import (
    build_design_objectives,
    held_out_gain,
    stochastic_design_descent,
)


def _quadratic_loss(params, ctx):
    return jnp.sum((params - ctx) ** 2)


def test_descent_recovers_target_and_selects_best_on_val():
    target = jnp.array([1.0, -2.0, 0.5])
    value_fn, vg_fn = build_design_objectives(_quadratic_loss)

    def make_train_ctx(i):
        # Fresh "ensemble" each resample: the clean target plus per-index noise.
        return target + 0.1 * jax.random.normal(jax.random.PRNGKey(i), target.shape)

    result = stochastic_design_descent(
        value_fn,
        vg_fn,
        jnp.zeros(3),
        make_train_ctx=make_train_ctx,
        val_ctx=target,  # clean validation
        optimizer=optax.adam(0.2),
        steps=200,
        resample_every=1,
    )

    # Validation loss drops substantially and the best-on-val design is near target.
    assert result.val_curve[-1] < 0.5 * result.val_curve[0]
    assert float(jnp.linalg.norm(result.params_best - target)) < 0.1
    # Bookkeeping shapes.
    assert result.train_curve.shape == (200,)
    assert result.val_curve.shape == (200,)
    assert 0 <= result.n_to_best < 200
    assert result.per_eval_s >= 0.0


def test_best_on_val_tracks_validation_minimum():
    # Deterministic train ctx that pulls params PAST the validation optimum, so the
    # final iterate overshoots and best-on-val must pick an earlier step.
    val_target = jnp.zeros(2)
    train_target = jnp.array([5.0, 5.0])  # training pulls away from the val optimum
    value_fn, vg_fn = build_design_objectives(_quadratic_loss)

    result = stochastic_design_descent(
        value_fn,
        vg_fn,
        jnp.zeros(2),
        make_train_ctx=lambda i: train_target,  # fixed: a steady pull toward [5, 5]
        val_ctx=val_target,
        optimizer=optax.sgd(0.1),
        steps=80,
        resample_every=1,
    )

    # Started at the val optimum (zeros), so any motion only worsens val: best is step 0.
    assert result.n_to_best == int(np.argmin(result.val_curve))
    assert result.val_curve[-1] >= result.val_curve[result.n_to_best]
    assert float(jnp.linalg.norm(result.params_best - val_target)) <= float(
        jnp.linalg.norm(result.params_final - val_target)
    )


def test_held_out_gain_sign():
    value_fn, _ = build_design_objectives(_quadratic_loss)
    target = jnp.zeros(3)
    test_ctxs = [target, target + 0.01]
    baseline = jnp.array([1.0, 1.0, 1.0])  # far from target
    better = jnp.array([0.1, 0.1, 0.1])  # closer to target
    worse = jnp.array([2.0, 2.0, 2.0])  # farther than baseline

    assert np.all(held_out_gain(value_fn, test_ctxs, baseline, better) > 0)
    assert np.all(held_out_gain(value_fn, test_ctxs, baseline, worse) < 0)
