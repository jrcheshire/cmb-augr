"""Tests for augr._qe (port of plancklens QE-leg machinery).

Compares augr._qe.get_qes against plancklens.qresp.get_qes qe-by-qe
for each estimator in scope: ptt, pee, p_eb, p_te, p_tb, p_p, p.

plancklens is not installed in the pixi env (it lives in the user's
``n0val`` conda env). The whole module is skipped if plancklens is
not importable, so this file is CI-safe; the user must run it
manually in ``n0val`` to validate the port:

    conda activate n0val
    pytest tests/test_qe.py -v
"""

from __future__ import annotations

# ruff: noqa: I001, E402  -- importorskip must run before the
# plancklens / augr imports, breaking ruff's import-order rule.

import numpy as np
import pytest

plancklens = pytest.importorskip("plancklens")
from plancklens import qresp as pl_qresp

from augr import _qe as ag_qe


QE_KEYS = ("ptt", "pee", "p_eb", "p_te", "p_tb", "p_p", "p")


def _make_cls(lmax: int) -> dict:
    """Smooth, non-degenerate weights so qe co-additions are exercised."""
    ell = np.arange(lmax + 1, dtype=float)
    safe = np.where(ell > 0, ell, 1.0)
    return {
        "tt": 1.0 / safe**2,
        "ee": 0.3 / safe**2,
        "bb": 0.05 / safe**2,
        "te": 0.4 / safe**2,
    }


def _canonical_signature(q) -> tuple:
    """Hashable identifier for a qe by spin signature only."""
    return (q.leg_a.spin_in, q.leg_a.spin_ou, q.leg_b.spin_in, q.leg_b.spin_ou)


@pytest.mark.parametrize("qe_key", QE_KEYS)
def test_qe_count_matches(qe_key):
    """Number of qes after simplify+proj matches plancklens."""
    lmax = 100
    cls = _make_cls(lmax)
    pl_qes = pl_qresp.get_qes(qe_key, lmax, cls)
    ag_qes = ag_qe.get_qes(qe_key, lmax, cls)
    assert len(ag_qes) == len(pl_qes), (
        f"qe count mismatch for {qe_key!r}: augr={len(ag_qes)}, plancklens={len(pl_qes)}"
    )


@pytest.mark.parametrize("qe_key", QE_KEYS)
def test_qe_signatures_match(qe_key):
    """Spin signatures (with multiplicity) match plancklens."""
    lmax = 100
    cls = _make_cls(lmax)
    pl_qes = pl_qresp.get_qes(qe_key, lmax, cls)
    ag_qes = ag_qe.get_qes(qe_key, lmax, cls)
    pl_sigs = sorted(_canonical_signature(q) for q in pl_qes)
    ag_sigs = sorted(_canonical_signature(q) for q in ag_qes)
    assert ag_sigs == pl_sigs, (
        f"qe signature multiset differs for {qe_key!r}:\n"
        f"  augr:       {ag_sigs}\n"
        f"  plancklens: {pl_sigs}"
    )


@pytest.mark.parametrize("qe_key", QE_KEYS)
def test_qe_legcls_match(qe_key):
    """Per-qe leg cls match plancklens to bit-precision after grouping by sig.

    The internal sweep order of qe_simplify is sensitive to numpy
    arithmetic so we don't compare element-by-element; instead we
    group qes by canonical signature and compare the multisets of
    (lega.cl, legb.cl) within each signature bucket.
    """
    lmax = 200
    cls = _make_cls(lmax)
    pl_qes = pl_qresp.get_qes(qe_key, lmax, cls)
    ag_qes = ag_qe.get_qes(qe_key, lmax, cls)

    def _bucket(qes):
        out: dict = {}
        for q in qes:
            sig = _canonical_signature(q)
            out.setdefault(sig, []).append((np.asarray(q.leg_a.cl), np.asarray(q.leg_b.cl)))
        return out

    pl_bk = _bucket(pl_qes)
    ag_bk = _bucket(ag_qes)
    assert set(ag_bk) == set(pl_bk)

    for sig, ag_pairs in ag_bk.items():
        pl_pairs = pl_bk[sig]
        assert len(ag_pairs) == len(pl_pairs), (
            f"{qe_key} sig {sig}: augr {len(ag_pairs)} vs plancklens {len(pl_pairs)}"
        )

        # Within a signature bucket, every augr pair must match exactly
        # one plancklens pair. Use a greedy match.
        unmatched = list(range(len(pl_pairs)))
        for ag_la, ag_lb in ag_pairs:
            for j in unmatched:
                pl_la, pl_lb = pl_pairs[j]
                if (
                    ag_la.shape == pl_la.shape
                    and ag_lb.shape == pl_lb.shape
                    and np.allclose(ag_la, pl_la, atol=1e-15, rtol=1e-12)
                    and np.allclose(ag_lb, pl_lb, atol=1e-15, rtol=1e-12)
                ):
                    unmatched.remove(j)
                    break
            else:
                raise AssertionError(
                    f"{qe_key} sig {sig}: augr leg pair "
                    f"(la_norm={np.linalg.norm(ag_la):.4e}, "
                    f"lb_norm={np.linalg.norm(ag_lb):.4e}) has no match in "
                    f"plancklens bucket of {len(pl_pairs)} pairs"
                )


@pytest.mark.parametrize("qe_key", QE_KEYS)
def test_qe_cL_match(qe_key):
    """qe.cL(L) returns the same array as plancklens for L in [0, lmax]."""
    lmax = 200
    cls = _make_cls(lmax)
    pl_qes = pl_qresp.get_qes(qe_key, lmax, cls)
    ag_qes = ag_qe.get_qes(qe_key, lmax, cls)
    Ls = np.arange(2 * lmax + 1)
    pl_cL_set = sorted(np.asarray(q.cL(Ls)).round(decimals=14).tobytes() for q in pl_qes)
    ag_cL_set = sorted(np.asarray(q.cL(Ls)).round(decimals=14).tobytes() for q in ag_qes)
    assert ag_cL_set == pl_cL_set, (
        f"{qe_key}: cL multiset differs. augr {len(ag_qes)}, plancklens {len(pl_qes)}"
    )


# ---------------------------------------------------------------------
# Direct unit checks on spin helpers.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("s", [-2, 0, 2])
def test_get_spin_raise_matches(s):
    from plancklens.utils_spin import get_spin_raise as pl_raise

    lmax = 50
    np.testing.assert_allclose(
        ag_qe.get_spin_raise(s, lmax),
        pl_raise(s, lmax),
        atol=1e-15,
    )


@pytest.mark.parametrize("s", [-2, 0, 2])
def test_get_spin_lower_matches(s):
    from plancklens.utils_spin import get_spin_lower as pl_lower

    lmax = 50
    np.testing.assert_allclose(
        ag_qe.get_spin_lower(s, lmax),
        pl_lower(s, lmax),
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "s1,s2",
    [
        (0, 0),
        (0, 2),
        (0, -2),
        (2, 0),
        (-2, 0),
        (2, 2),
        (-2, -2),
        (2, -2),
        (-2, 2),
    ],
)
def test_spin_cls_matches(s1, s2):
    from plancklens.utils_spin import spin_cls as pl_spin_cls

    lmax = 200
    cls = _make_cls(lmax)
    ag = ag_qe.spin_cls(s1, s2, cls)
    pl = pl_spin_cls(s1, s2, cls)
    np.testing.assert_allclose(ag, pl, atol=1e-15, rtol=1e-12)
