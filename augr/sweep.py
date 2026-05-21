"""augr.sweep -- ready-made jax.vmap wrappers over the differentiable forward.

The Tier 1 (`sigma_r_from_channels`) and Tier 2 (`sigma_r_from_design`)
forwards in :mod:`augr.optimize` already take leaf jnp scalars/arrays
and are pure (no Python side effects, no .item() calls, no
incompatible JIT traces). They vmap cleanly without any wrapping.
This module exposes ready-made vmapped callables for the most common
single-axis design sweeps so callers don't have to write
``jax.vmap(..., in_axes=(0, None, None, None, None))`` boilerplate
inline.

Public callables (all ``jax.vmap``-of-the-underlying):

Channel-level (vmaps a leading axis of one of the four per-channel
arrays passed to :func:`augr.optimize.sigma_r_from_channels`):

* :data:`sigma_r_over_n_det`     -- vmap n_det
* :data:`sigma_r_over_net`       -- vmap NET
* :data:`sigma_r_over_beam`      -- vmap beam_fwhm
* :data:`sigma_r_over_eta`       -- vmap eta_total

Design-level (vmaps a scalar/array positional arg of
:func:`augr.optimize.sigma_r_from_design`):

* :data:`sigma_r_over_aperture`        -- vmap aperture_m
* :data:`sigma_r_over_f_number`        -- vmap f_number
* :data:`sigma_r_over_fp_diameter`     -- vmap fp_diameter_m
* :data:`sigma_r_over_area_fractions`  -- vmap area_fractions (leading axis)

Plus two factories for the less-common axis or when the caller wants to
construct the vmap inline:

* :func:`vmap_channels(axis)` -- factory for channel-level sweeps.
* :func:`vmap_design(axis)`   -- factory for design-level sweeps.

**Kwargs gotcha.** ``jax.vmap`` maps over *both* positional args and
keyword args by default, with the global default ``in_axes=0`` applied
to every leaf. The vmaps in this module set ``in_axes`` only for the
positional args, so any kwarg array you pass when invoking the
returned callable will *also* be mapped along its leading axis and
will fail if the size disagrees with the swept axis. The fix is to
bind those kwargs with :func:`functools.partial` *before* the vmap is
constructed: pass them to :func:`vmap_channels` / :func:`vmap_design`
as keyword arguments and they'll be baked in. Scalars (Python floats)
in kwargs are fine and need no binding.

Out of scope on purpose:

* **Multi-axis grids**. For an aperture × f_number grid, call
  :data:`sigma_r_over_aperture` inside a Python loop over f_number
  values (or wrap a nested vmap yourself; memory profile is worse).
  A first-class multi-axis API will land when a concrete caller wants
  one.
* **BROOM-driven sweeps**. BROOM is subprocess + Julia + disk I/O;
  not JAX-traceable. Stay on :func:`augr.parallel.process_pool`.
* **iterate_delensing sweeps**. The full-sky path has numpy +
  side-effects + its own process pool; not vmap-able. Stay on
  :func:`augr.parallel.process_pool` for those too.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax

from augr.optimize import sigma_r_from_channels, sigma_r_from_design

# Positional-arg arity baked into each vmap. sigma_r_from_channels takes
# (n_det, net, beam_fwhm, eta_total, ctx) = 5 positional args before kwargs;
# sigma_r_from_design takes (aperture_m, f_number, fp_diameter_m,
# area_fractions, ctx, freqs_per_group) = 6 positional args before kwargs.
_N_POS_CHANNELS = 5
_N_POS_DESIGN = 6

_CHANNEL_AXES: dict[str, int] = {"n_det": 0, "net": 1, "beam": 2, "eta": 3}
_DESIGN_AXES: dict[str, int] = {
    "aperture_m": 0,
    "f_number": 1,
    "fp_diameter_m": 2,
    "area_fractions": 3,
}


def _in_axes(n_args: int, swept_idx: int) -> tuple:
    axes: list[int | None] = [None] * n_args
    axes[swept_idx] = 0
    return tuple(axes)


def vmap_channels(axis: str, **bound_kwargs) -> Callable:
    """Return ``jax.vmap`` of :func:`sigma_r_from_channels` over the named arg.

    Parameters
    ----------
    axis
        One of ``"n_det"``, ``"net"``, ``"beam"``, ``"eta"``.
    **bound_kwargs
        Kwargs of :func:`sigma_r_from_channels` to bind via
        :func:`functools.partial` *before* the vmap is constructed.
        Use this for any kwarg you want held fixed across the swept
        axis (especially arrays -- see the kwargs gotcha in the
        module docstring).

    Returns
    -------
    Callable
        Equivalent to ``jax.vmap(partial(sigma_r_from_channels,
        **bound_kwargs), in_axes=...)`` with the chosen positional
        arg mapped along its leading axis.

    Examples
    --------
    >>> f = vmap_channels("n_det")
    >>> result = f(n_det_grid, net, beam, eta, ctx)
    """
    if axis not in _CHANNEL_AXES:
        raise ValueError(
            f"axis must be one of {sorted(_CHANNEL_AXES)}; got {axis!r}"
        )
    fn = partial(sigma_r_from_channels, **bound_kwargs) if bound_kwargs else sigma_r_from_channels
    return jax.vmap(
        fn,
        in_axes=_in_axes(_N_POS_CHANNELS, _CHANNEL_AXES[axis]),
    )


def vmap_design(axis: str, **bound_kwargs) -> Callable:
    """Return ``jax.vmap`` of :func:`sigma_r_from_design` over the named arg.

    Parameters
    ----------
    axis
        One of ``"aperture_m"``, ``"f_number"``, ``"fp_diameter_m"``,
        ``"area_fractions"``.
    **bound_kwargs
        Kwargs of :func:`sigma_r_from_design` to bind via
        :func:`functools.partial` *before* the vmap is constructed.
        Use this for any kwarg you want held fixed across the swept
        axis (especially arrays like ``net_override`` /
        ``eta_total`` -- see the kwargs gotcha in the module
        docstring).

    Returns
    -------
    Callable
        Equivalent to ``jax.vmap(partial(sigma_r_from_design,
        **bound_kwargs), in_axes=...)`` with the chosen positional
        arg mapped along its leading axis.

    Examples
    --------
    >>> f = vmap_design("aperture_m", net_override=net, eta_total=eta)
    >>> sigmas = f(jnp.linspace(1.0, 5.0, 9), f_num, fp_diam, area_frac,
    ...            ctx, freqs_per_group)
    """
    if axis not in _DESIGN_AXES:
        raise ValueError(
            f"axis must be one of {sorted(_DESIGN_AXES)}; got {axis!r}"
        )
    fn = partial(sigma_r_from_design, **bound_kwargs) if bound_kwargs else sigma_r_from_design
    return jax.vmap(
        fn,
        in_axes=_in_axes(_N_POS_DESIGN, _DESIGN_AXES[axis]),
    )


sigma_r_over_n_det = vmap_channels("n_det")
sigma_r_over_net = vmap_channels("net")
sigma_r_over_beam = vmap_channels("beam")
sigma_r_over_eta = vmap_channels("eta")

sigma_r_over_aperture = vmap_design("aperture_m")
sigma_r_over_f_number = vmap_design("f_number")
sigma_r_over_fp_diameter = vmap_design("fp_diameter_m")
sigma_r_over_area_fractions = vmap_design("area_fractions")
