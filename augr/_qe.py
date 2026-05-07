"""Internal QE-leg machinery for full-sky N_0 validation.

This is a numpy-only port of plancklens's QE-leg representation,
narrowed to the lensing-gradient source ('p') and the keys we
validate against in scripts/n0_validation:

    'ptt', 'pee', 'p_eb', 'p_te', 'p_tb', 'p_p', 'p'

The port preserves plancklens's sign conventions exactly. In
particular get_spin_lower returns a NEGATIVE sqrt, matching
plancklens/utils_spin.py:106-114; this is load-bearing for the
nhl GG_N0 = 0.5*R_sutv + 0.5*(-1)^(to+so)*R_msmtuv combination
that recovers the spin-lowering branch of the lensing source action.

References (plancklens commit at /Users/jamie/cmb/plancklens):
    plancklens/utils_qe.py    -- qeleg, qe, qe_simplify, qe_proj
    plancklens/qresp.py       -- get_qes, get_resp_legs, get_covresp
    plancklens/utils_spin.py  -- get_spin_raise, get_spin_lower, spin_cls

Not ported (out of scope for N_0 validation):
    sources other than 'p' (curl 'x', modulation 'f', polarization
    rotation 'a', point source 'stt', noise 'ntt'); bias-hardened
    ('_bh_') estimators; transfer-function args.
"""

# SIM118 / SIM910: "k in d.keys()" / ".get(k, None)" idioms are
# intentionally retained for fidelity to plancklens.utils_qe /
# utils_spin / qresp (any cosmetic divergence makes future re-syncs
# harder to audit). E741 ambiguous-variable-name 'l' / 'I' likewise
# matches plancklens upstream.
# ruff: noqa: SIM118, SIM910

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Spin-weighted spectra helpers (port of plancklens/utils_spin.py).
# ---------------------------------------------------------------------


def get_spin_raise(s: int, lmax: int) -> np.ndarray:
    """Spin-raising eigenvalue: +sqrt((l-s)(l+s+1)) for |s| <= l <= lmax.

    Matches plancklens.utils_spin.get_spin_raise.
    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s) :] = np.sqrt(
        np.arange(abs(s) - s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2)
    )
    return ret


def get_spin_lower(s: int, lmax: int) -> np.ndarray:
    """Spin-lowering eigenvalue: -sqrt((l+s)(l-s+1)) for |s| <= l <= lmax.

    Note the leading minus -- plancklens.utils_spin.get_spin_lower
    returns the NEGATIVE sqrt, and the response uses
    ``-0.5 * get_spin_lower(s)`` so the two negatives cancel and the
    effective coefficient on the lowering leg is +0.5 sqrt(...).
    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s) :] = -np.sqrt(
        np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2)
    )
    return ret


def _dict_transpose(cls: dict) -> dict:
    """Helper for spin_cls's s1<0 recursion: swap key letters."""
    ret = {}
    for k in cls.keys():
        if len(k) == 1:
            ret[k + k] = np.copy(cls[k])
        else:
            assert len(k) == 2
            ret[k[1] + k[0]] = np.copy(cls[k])
    return ret


def spin_cls(s1: int, s2: int, cls: dict) -> np.ndarray:
    """Spin-weighted cross-spectrum _{s1}X_{lm} _{s2}X*_{lm}.

    Port of plancklens.utils_spin.spin_cls. For (s1, s2) in
    {-2, 0, 2}^2; returns real or complex array as required. Reads
    'tt', 'te' / 'et', 'ee', 'bb', 'tb' / 'bt', 'eb' / 'be' from
    ``cls``; missing keys treated as zero except where noted.
    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * np.conjugate(spin_cls(-s1, -s2, _dict_transpose(cls)))
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2, "not implemented")
    if s1 == 0:
        if s2 == 0:
            return cls["tt"]
        tb = cls.get("tb", None)
        assert "te" in cls.keys() or "et" in cls.keys()
        te = cls.get("te", cls.get("et"))
        return -te if tb is None else -te + 1j * np.sign(s2) * tb
    elif s1 == 2:
        if s2 == 0:
            assert "te" in cls.keys() or "et" in cls.keys()
            tb = cls.get("bt", cls.get("tb", None))
            et = cls.get("et", cls.get("te"))
            return -et if tb is None else -et - 1j * tb
        elif s2 == 2:
            return cls["ee"] + cls["bb"]
        elif s2 == -2:
            eb = cls.get("be", cls.get("eb", None))
            return cls["ee"] - cls["bb"] if eb is None else cls["ee"] - cls["bb"] + 2j * eb
    raise AssertionError((s1, s2))


# ---------------------------------------------------------------------
# qeleg / qe -- port of plancklens/utils_qe.py.
# ---------------------------------------------------------------------


class qeleg:
    """One leg of a quadratic estimator.

    spin_in: spin-weight that this leg consumes from the IVF map.
    spin_ou: spin-weight of the resulting position-space product.
    cl:      a per-ell weight array (the 'W' in HO02 notation).

    Mutable: qe_proj re-flips spin_in in place per
    utils_qe.py:135-176; we preserve that semantic.
    """

    def __init__(self, spin_in: int, spin_out: int, cl: np.ndarray):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def __eq__(self, other) -> bool:
        if not isinstance(other, qeleg):
            return NotImplemented
        if (
            self.spin_in != other.spin_in
            or self.spin_ou != other.spin_ou
            or self.get_lmax() != other.get_lmax()
        ):
            return False
        return np.all(self.cl == other.cl)

    def __mul__(self, other) -> qeleg:
        return qeleg(self.spin_in, self.spin_ou, self.cl * other)

    def __add__(self, other: qeleg) -> qeleg:
        assert self.spin_in == other.spin_in and self.spin_ou == other.spin_ou
        lmax = max(self.get_lmax(), other.get_lmax())
        cl = np.zeros(lmax + 1, dtype=self.cl.dtype)
        cl[: len(self.cl)] += self.cl
        cl[: len(other.cl)] += other.cl
        return qeleg(self.spin_in, self.spin_ou, cl)

    def copy(self) -> qeleg:
        return qeleg(self.spin_in, self.spin_ou, np.copy(self.cl))

    def get_lmax(self) -> int:
        return len(self.cl) - 1


class qe:
    """One quadratic estimator: lega(n) * legb(n), reduced via cL."""

    def __init__(self, leg_a: qeleg, leg_b: qeleg, cL):
        assert leg_a.spin_ou + leg_b.spin_ou >= 0
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.cL = cL

    def get_lmax_a(self) -> int:
        return self.leg_a.get_lmax()

    def get_lmax_b(self) -> int:
        return self.leg_b.get_lmax()


def qe_simplify(qe_list: list[qe], _swap: bool = False) -> list[qe]:
    """Co-add qes with identical (lega, legb-spins, cL).

    Mirrors plancklens.utils_qe.qe_simplify. Sweeps once with
    legs in given order, then once with legs swapped, then unswaps,
    so co-additions are caught from either side.
    """
    skip: list[int] = []
    qes_ret: list[qe] = []
    qes = [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qe_list] if _swap else qe_list
    for i, qe1 in enumerate(qes):
        if i in skip:
            continue
        leg_a = qe1.leg_a.copy()
        leg_b = qe1.leg_b.copy()
        for j, qe2 in enumerate(qes[i + 1 :]):
            if qe2.leg_a == leg_a:
                if (
                    qe2.leg_b.spin_in == qe1.leg_b.spin_in
                    and qe2.leg_b.spin_ou == qe1.leg_b.spin_ou
                ):
                    Ls = np.arange(max(qe1.leg_b.get_lmax(), qe2.leg_b.get_lmax()) + 1)
                    if np.all(qe1.cL(Ls) == qe2.cL(Ls)):
                        leg_b += qe2.leg_b
                        skip.append(j + i + 1)
        if np.any(leg_a.cl) and np.any(leg_b.cl):
            qes_ret.append(qe(leg_a, leg_b, qe1.cL))
    if not _swap:
        return qe_simplify(qes_ret, _swap=True)
    return [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qes_ret]


def qe_proj(qe_list: list[qe], a: str, b: str) -> list[qe]:
    """Project a list of qes onto the (a, b) field combination.

    a, b in {'t', 'e', 'b'}. Mirrors plancklens.utils_qe.qe_proj. For
    spin-2 sides, splits each input qe into 2 or 4 pieces with sign
    factors that combine to project E-only or B-only out of the
    spin-2 (E +/- iB) basis.
    """
    assert a in ["t", "e", "b"] and b in ["t", "e", "b"]
    l_in = [0] if a == "t" else [-2, 2]
    r_in = [0] if b == "t" else [-2, 2]
    qes_ret: list[qe] = []
    for q in qe_list:
        si, ri = (q.leg_a.spin_in, q.leg_b.spin_in)
        if si not in l_in or ri not in r_in:
            continue
        leg_a = q.leg_a.copy()
        leg_b = q.leg_b.copy()
        if si == 0 and ri == 0:
            qes_ret.append(qe(leg_a, leg_b, q.cL))
        elif si == 0 and abs(ri) > 0:
            sgn = 1 if b == "e" else -1
            qes_ret.append(qe(leg_a, leg_b * 0.5, q.cL))
            leg_b.spin_in *= -1
            qes_ret.append(qe(leg_a, leg_b * 0.5 * sgn, q.cL))
        elif ri == 0 and abs(si) > 0:
            sgn = 1 if a == "e" else -1
            qes_ret.append(qe(leg_a * 0.5, leg_b, q.cL))
            leg_a.spin_in *= -1
            qes_ret.append(qe(leg_a * 0.5 * sgn, leg_b, q.cL))
        elif abs(ri) > 0 and abs(si) > 0:
            sgna = 1 if a == "e" else -1
            sgnb = 1 if b == "e" else -1
            qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5, q.cL))
            leg_b.spin_in *= -1
            qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5 * sgnb, q.cL))
            leg_a.spin_in *= -1
            qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5 * sgnb, q.cL))
            leg_b.spin_in *= -1
            qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5, q.cL))
        else:
            raise AssertionError((si, ri))
    return qe_simplify(qes_ret)


# ---------------------------------------------------------------------
# Lensing-source response legs (port of plancklens/qresp.py:104-147,
# narrowed to source='p').
# ---------------------------------------------------------------------


def get_resp_legs(source: str, lmax: int) -> dict:
    """Per-spin response of a CMB map to anisotropy source.

    Returns {s: (s_source, prR, mrR, cL_scal)} for s in {-2, 0, 2}.

    For lensing ('p'): _{s}X -> _{s}X - 1/2 alpha_{1} eth (_{s}X)
                                   - 1/2 alpha_{-1} ethbar (_{s}X)

    With:
        prR = -0.5 * get_spin_lower(s)   (so prR is positive sqrt; lowering)
        mrR = -0.5 * get_spin_raise(s)   (negative sqrt; raising)
        cL_scal = lambda L: sqrt(L(L+1))
    """
    if source != "p":
        raise NotImplementedError(f"only source='p' is ported (got {source!r})")
    return {
        s: (
            1,
            -0.5 * get_spin_lower(s, lmax),
            -0.5 * get_spin_raise(s, lmax),
            lambda ell: get_spin_raise(0, np.max(ell))[ell],
        )
        for s in [0, -2, 2]
    }


def get_covresp(source: str, s1: int, s2: int, cls: dict, lmax: int):
    """Covariance-level response: get_resp_legs piece * spin_cls coupling.

    Port of plancklens.qresp.get_covresp narrowed to source='p'. The
    s1<0 sign-folding in plancklens.utils_spin.spin_cls IS preserved
    (we simply call our ported spin_cls).
    """
    if source != "p":
        raise NotImplementedError(f"only source='p' is ported (got {source!r})")
    s_source, prR, mrR, cL_scal = get_resp_legs(source, lmax)[s1]
    coupl = spin_cls(s1, s2, cls)[: lmax + 1]
    return s_source, prR * coupl, mrR * coupl, cL_scal


# ---------------------------------------------------------------------
# get_qes (port of plancklens/qresp.py:50-101 narrowed to source='p').
# ---------------------------------------------------------------------

# QE keys that are pure spin-0 (TT-only on both legs).
_QE_KEYS_TT = ["ptt"]
# QE keys that are pure spin-2 (P only on both legs).
_QE_KEYS_PP = ["p_p"]

# Symmetrized split-by-letter keys ('p_te', 'p_tb', 'p_eb').
_QE_KEYS_PROJ_SYM = {"p_te", "p_tb", "p_eb"}
# Non-symmetrized split-by-letter keys.
_QE_KEYS_PROJ_NONSYM = {"pte", "pet", "ptb", "pbt", "pee", "peb", "pbe", "pbb"}


def get_qes(qe_key: str, lmax: int, cls_weight: dict, lmax2: int | None = None) -> list[qe]:
    """Build the QE leg-list for ``qe_key``.

    Port of plancklens.qresp.get_qes narrowed to source='p' and
    qe_keys in {ptt, pee, p_eb, p_te, p_tb, p_p, p, pte, peb, ...}.
    The cls_weight dict supplies 'tt', 'te', 'ee', 'bb' (plus 'tb',
    'eb' if present) entering the QE response weights.

    Returns a simplified+projected qe list. Each element of the
    returned list is a ``qe(leg_a, leg_b, cL)`` whose lega/legb
    carry per-ell ``cl`` arrays, ``spin_in``, and ``spin_ou`` ready
    for consumption by ``_get_nhl`` / ``get_response`` ports.
    """
    if lmax2 is None:
        lmax2 = lmax
    if not qe_key or qe_key[0] != "p":
        raise NotImplementedError(f"only lensing-gradient keys ('p...') ported, got {qe_key!r}")

    # Pick the input-spin sweep by qe_key family, mirroring qresp.py:63-69.
    if qe_key in _QE_KEYS_TT:
        s_lefts = [0]
    elif qe_key in _QE_KEYS_PP:
        s_lefts = [-2, 2]
    else:
        s_lefts = [0, -2, 2]
    s_rights_in = s_lefts

    qes: list[qe] = []
    for s_left in s_lefts:
        for sin in s_rights_in:
            sout = -s_left
            s_qe, _irr1, cl_sosi, cL_out = get_covresp(qe_key[0], sout, sin, cls_weight, lmax2)
            if not np.any(cl_sosi):
                continue
            lega = qeleg(
                s_left,
                s_left,
                0.5 * (1.0 + (s_left == 0)) * np.ones(lmax + 1, dtype=float),
            )
            legb = qeleg(
                sin,
                sout + s_qe,
                0.5 * (1.0 + (sin == 0)) * 2.0 * cl_sosi,
            )
            qes.append(qe(lega, legb, cL_out))

    # Project + simplify per qe_key suffix.
    if len(qe_key) == 1 or qe_key[1:] in ("tt", "_p"):
        return qe_simplify(qes)
    suffix = qe_key[1:]
    if suffix in {"te", "et", "tb", "bt", "ee", "eb", "be", "bb"}:
        return qe_simplify(qe_proj(qes, suffix[0], suffix[1]))
    if suffix in {"_te", "_tb", "_eb"}:
        return qe_simplify(qe_proj(qes, suffix[1], suffix[2]) + qe_proj(qes, suffix[2], suffix[1]))
    raise AssertionError(f"qe key {qe_key!r} not recognized")
