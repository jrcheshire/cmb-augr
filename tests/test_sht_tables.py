"""Scoped stored jht tables (``sht.legendre_tables``) leave every result unchanged.

A ``jht.LegendreTables`` holds the values jht's recursion computes per transform, so
with the tables scoped the jht backend returns the recursion's results; measured
bitwise on CPU (2026-11-21) for the transforms, ``map2alm``, their gradients and the
NILC / cMILC08 cleaners. Gated at ``ALG_TOL`` to allow another backend to fuse the
same arithmetic differently. Each equality is paired with a mutation (tables scaled
by 1.01) that must move the result, so a scope that silently falls back to the
recursion fails.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jht = pytest.importorskip("jht")
if not hasattr(jht, "legendre_tables"):
    pytest.skip("jht without legendre_tables", allow_module_level=True)

from augr import sht  # noqa: E402

ALG_TOL = 1e-12
NSIDE, LMAX = 32, 48


@pytest.fixture(autouse=True)
def _jht_backend():
    with sht.sht_backend("jht"):
        yield


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _scaled(tabs, spins=(0, 2)):
    return tuple(
        dataclasses.replace(t, lam=tuple(1.01 * a for a in t.lam)) if t.spin in spins else t
        for t in tabs
    )


def _alm(key):
    k1, k2 = jax.random.split(key)
    n = sht.alm_size(LMAX)
    a = jax.random.normal(k1, (2, n)) + 1j * jax.random.normal(k2, (2, n))
    return a.at[:, : LMAX + 1].set(a[:, : LMAX + 1].real)


def _loss(a, tabs=None):
    with sht.legendre_tables(tabs):
        qu = sht.synthesis(a, 2, LMAX, NSIDE)
        eb = sht.map2alm(qu, 2, LMAX, NSIDE, 3)
        t = sht.synthesis(a[:1], 0, LMAX, NSIDE)
        return jnp.sum(qu**2) + jnp.sum(jnp.abs(eb) ** 2) + jnp.sum(t**3)


def test_transforms_and_gradients_match_without_tables():
    a = _alm(jax.random.PRNGKey(0))
    tabs = sht.build_legendre_tables(NSIDE, LMAX)
    vg = jax.jit(jax.value_and_grad(_loss))
    v0, g0 = vg(a)
    v1, g1 = vg(a, tabs)
    assert _rel(v1, v0) <= ALG_TOL
    assert _rel(g1, g0) <= ALG_TOL
    for spins in ((0,), (2,)):  # each table is live
        vb, _ = vg(a, _scaled(tabs, spins))
        assert _rel(vb, v0) > 1e-6, f"spin {spins} table not used"


def test_cleaners_match_without_tables():
    from augr.cmilc import cmilc_clean
    from augr.nilc import nilc_clean

    npix = 12 * NSIDE**2
    freqs = (40.0, 90.0, 150.0, 220.0, 350.0)
    beams = (60.0, 40.0, 30.0, 25.0, 20.0)
    k = jax.random.split(jax.random.PRNGKey(3), 2)
    cmb = jnp.broadcast_to(jax.random.normal(k[0], (1, 2, npix)), (5, 2, npix))
    sed = jnp.asarray([(f / 150.0) ** 1.5 for f in freqs])[:, None, None]
    total = cmb + 3.0 * sed * jax.random.normal(k[1], (1, 2, npix))
    tabs = sht.build_legendre_tables(NSIDE, LMAX)
    for name in ("nilc", "cmilc08"):
        def clean(x, t=None, name=name):
            with sht.legendre_tables(t):
                if name == "nilc":
                    r = nilc_clean(x, beams, lmax=LMAX, nside=NSIDE)
                else:
                    r = cmilc_clean(x, beams, freqs, lmax=LMAX, nside=NSIDE)
                return r.cleaned_b_alm, r.project(cmb)

        c0, p0 = jax.jit(clean)(total)
        c1, p1 = jax.jit(clean)(total, tabs)
        assert _rel(c1, c0) <= ALG_TOL, name
        assert _rel(p1, p0) <= ALG_TOL, name
        cb, _ = jax.jit(clean)(total, _scaled(tabs))
        assert _rel(cb, c0) > 1e-6, f"{name}: tables not used"


def test_scope_is_keyed_and_restored():
    tabs = sht.build_legendre_tables(NSIDE, LMAX, spins=(2,))
    assert sht.tables_for(NSIDE, LMAX, 2) is None
    with sht.legendre_tables(tabs):
        assert sht.tables_for(NSIDE, LMAX, 2) is tabs[0]
        assert sht.tables_for(NSIDE, LMAX, 0) is None  # no spin-0 table: recursion
        assert sht.tables_for(NSIDE, LMAX + 1, 2) is None
        assert sht.scoped_tables() == tabs
    assert sht.tables_for(NSIDE, LMAX, 2) is None
    with sht.legendre_tables(None):
        assert sht.scoped_tables() == ()


def test_concrete_tables_inside_a_trace_raise():
    """Opening the scope around a jitted call closes over concrete tables: refused."""
    a = _alm(jax.random.PRNGKey(1))
    tabs = sht.build_legendre_tables(NSIDE, LMAX)
    with sht.legendre_tables(tabs), pytest.raises(ValueError, match="constants"):
        jax.jit(lambda a: sht.synthesis(a, 2, LMAX, NSIDE))(a)
