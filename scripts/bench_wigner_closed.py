"""
bench_wigner_closed.py -- before/after timings for the issue #48 Wigner-3j work.

Every row reports wall time, process CPU time summed over all threads, and
their ratio ``eff`` (the number of cores the row effectively kept busy). A
row with ``eff`` ~ 1 on a many-core node is serial by construction, not
"unlucky"; the Schulten-Gordon recursion (a ``lax.scan`` over l2 with a
vector of ``n_l1`` elements per step) is the known example.

Rows (one machine, jit-warm, median of ``--repeat``):

  1. per-L tables: ``spin2_body`` closed form vs the retained Schulten-Gordon
     scan ``_spin2_body_sg``; ``spin0_body`` g-table vs ``_spin0_body_gammaln``.
  2. full estimators: ``compute_n0_{tt,ee,te,eb,tb}_fullsky_jax`` and the
     lensing kernel, SG vs closed form, on the production sampled L grid
     (``--dense`` for the exact every-L grid: ~13x the sweeps at
     l_max_qe=1500, and the SG rows there are serial -- that is the shape
     that ran for two hours on a 144-core node).
  3. ``iterate_delensing(fullsky=True)`` end to end, JAX backend, SG vs closed
     on the sampled grid, plus the exact grid and the numpy ProcessPool backend
     on request.
  4. ``pseudo_cl_jax.coupling_matrices`` (MASTER M+, M-).
  5. ``DelensCoupling`` build + design gradient, flat-sky vs full-sky sampled.

Harness rules (each one was violated by the first version of this script and
produced a table that measured nothing -- see the commit message):

  * The swept quantity (``L``) is a traced jit ARGUMENT. A Python scalar
    closed over by ``jax.jit(lambda: ...)`` is constant-folded and the timed
    call is a memcpy of a precomputed table.
  * Each variant is a fresh function object and ``jax.clear_caches()`` runs
    between variants: ``jax.jit`` caches executables by the function object,
    so timing one ``f`` under a monkeypatch and again without it re-runs the
    first executable (every row reads 1.0x).
  * The implementation swap is verified live: the patched-in Wigner core is a
    counting wrapper and the row asserts it was invoked during tracing.
  * Every print is flushed, so a killed or cancelled job leaves its rows.

Usage:
    pixi run python scripts/bench_wigner_closed.py [--l-max 1500 3000]
        [--repeat 3] [--dense] [--numpy] [--skip 1 2 3 4 5]
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from augr.config import pico_like
from augr.delensing import default_n_L_sample, iterate_delensing, load_lensing_spectra
from augr.instrument import combined_noise_nl


def _print(*a):
    print(*a, flush=True)


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _threads():
    try:
        import psutil
        return psutil.Process().num_threads()
    except Exception:
        return -1


class Row:
    """Wall / CPU / eff for one timed callable (median over repeats)."""

    def __init__(self, wall, cpu):
        self.wall, self.cpu = wall, cpu

    @property
    def eff(self):
        return self.cpu / self.wall if self.wall > 0 else float("nan")

    def __format__(self, spec):
        unit = "ms" if self.wall < 1 else "s"
        w = self.wall * (1e3 if unit == "ms" else 1)
        return f"{w:8.2f} {unit} (eff {self.eff:5.1f})"


def _time(fn, repeat):
    """Compile/warm once, then median wall + CPU over ``repeat`` calls.

    Returns ``(Row, value)`` so callers can check that two variants agree.
    """
    value = jax.block_until_ready(fn())
    walls, cpus = [], []
    for _ in range(repeat):
        c0, w0 = time.process_time(), time.perf_counter()
        value = jax.block_until_ready(fn())
        cpus.append(time.process_time() - c0)
        walls.append(time.perf_counter() - w0)
    i = int(np.argsort(walls)[len(walls) // 2])
    return Row(walls[i], cpus[i]), value


def _check_agree(name, a, b, rtol=1e-6):
    """The two variants must compute the same thing, or the ratio is meaningless."""
    a = jax.tree.leaves(a)
    b = jax.tree.leaves(b)
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.array_equal(np.isfinite(x), np.isfinite(y)):
            raise RuntimeError(f"{name}: SG and closed form differ in finite pattern")
        scale = np.maximum(np.abs(x[finite]), np.abs(y[finite]))
        rel = np.abs(x[finite] - y[finite]) / np.where(scale > 0, scale, 1.0)
        worst = max(worst, float(rel.max()) if rel.size else 0.0)
    if worst > rtol:
        raise RuntimeError(f"{name}: SG vs closed form disagree, worst rel {worst:.2e} > {rtol:.0e}")
    return worst


# ---------------------------------------------------------------------------
# Wigner implementation switch, verified live
# ---------------------------------------------------------------------------

_SG_EXTENDED = [0]   # how many SG calls had their l2 grid extended to cover the triangle


def _sg_covering(j2, l1, m1, m2, m3, l2_min, l2_max):
    """Schulten-Gordon spin-2 table, made correct on a truncated l2 grid.

    The recursion is seeded at ``l2 = l1 + j2`` and sum-rule normalized over
    the grid it is given, so a grid that stops below ``max(l1) + j2`` seeds
    rows off-grid and returns garbage (the pre-#48 full-sky TE defect: the
    TE and TT estimators sum over the square ``[l_min, l_max]`` grid, and TE's
    spin-2 leg inherited it). Timing that garbage against the closed form is
    not a before/after, so the SG reference here extends a truncated grid to
    one that covers the triangle and slices the result back; the closed form
    is grid-independent and needs nothing. Grids that already cover the
    triangle (EE / EB / TB / kernel: ``l2_max = l_max + max(L)``) are left
    alone so their SG timing is the genuine pre-#48 cost.

    Truncation test: every production caller builds ``l1 = arange(2, l_max+1)``,
    so ``l_max = len(l1) + 1`` and a grid with ``l2_max <= l_max`` is the
    square one. ``j2 <= l_max`` always, so ``l2_max + l_max`` covers.
    """
    from augr import wigner_jax as wj
    l_max = int(l1.shape[0]) + 1
    if l2_max <= l_max:
        _SG_EXTENDED[0] += 1
        w = wj._spin2_body_sg(j2, l1, m1, m2, m3, l2_min, l2_max + l_max)
        return w[:, : l2_max - l2_min + 1]
    return wj._spin2_body_sg(j2, l1, m1, m2, m3, l2_min, l2_max)


@contextmanager
def wigner_impl(name: str):
    """Route every production Wigner call through ``name`` ('closed' | 'sg').

    Yields a one-element call counter. The caller must compile inside the
    block and check ``calls[0] > 0`` -- a zero means the patch never reached
    the traced code and the row would be measuring the other variant.
    """
    import augr.delensing_fullsky_jax as dj
    import augr.pseudo_cl_jax as pj
    from augr import wigner_jax as wj
    if name == "closed":
        s2, s0 = wj.spin2_body, wj.spin0_body
    elif name == "sg":
        s2, s0 = _sg_covering, wj._spin0_body_gammaln
    else:
        raise ValueError(name)
    calls = [0]

    def w2(*a, **k):
        calls[0] += 1
        return s2(*a, **k)

    def w0(*a, **k):
        calls[0] += 1
        return s0(*a, **k)

    jax.clear_caches()
    with mock.patch.object(dj, "spin2_body", w2), \
         mock.patch.object(dj, "spin0_body", w0), \
         mock.patch.object(pj, "spin2_body", w2):
        yield calls
    jax.clear_caches()


def _timed_variant(make_fn, impl, repeat):
    """``make_fn()`` builds a FRESH callable; time it under ``impl``."""
    with wigner_impl(impl) as calls:
        row, value = _time(make_fn(), repeat)
        if calls[0] == 0:
            raise RuntimeError(
                f"Wigner implementation {impl!r} was never invoked during tracing; "
                "the row would be measuring the wrong variant")
    return row, value


def _pair(name, make_fn, repeat):
    """SG -> closed pair on one fresh callable per variant; values must agree."""
    _SG_EXTENDED[0] = 0
    t_sg, v_sg = _timed_variant(make_fn, "sg", repeat)
    ext = "  (SG grid extended)" if _SG_EXTENDED[0] else ""
    t_new, v_new = _timed_variant(make_fn, "closed", repeat)
    worst = _check_agree(name, v_sg, v_new)
    _print(f"    {name:7s} {t_sg:>26} -> {t_new:>26}   ({t_sg.wall / t_new.wall:4.1f}x)"
           f"  agree {worst:.0e}{ext}   [{_now()}]")
    return t_sg, t_new


# ---------------------------------------------------------------------------
# [1] per-L tables
# ---------------------------------------------------------------------------

def bench_tables(l_max, repeat):
    from augr.wigner_jax import _spin0_body_gammaln, _spin2_body_sg, spin0_body, spin2_body
    l1 = jnp.arange(2, l_max + 1, dtype=float)
    l2_min, l2_max = 2, l_max + l_max // 2
    L = jnp.asarray(float(l_max // 5))          # traced argument, never closed over
    _print(f"\n[1] per-L tables, l1=2..{l_max}, l2={l2_min}..{l2_max}, L={int(L)}")
    rows = [
        ("spin2 SG scan     ", lambda: jax.jit(lambda L: _spin2_body_sg(L, l1, -2, 0, 2, l2_min, l2_max))),
        ("spin2 closed form ", lambda: jax.jit(lambda L: spin2_body(L, l1, -2, 0, 2, l2_min, l2_max))),
        ("spin0 gammaln     ", lambda: jax.jit(lambda L: _spin0_body_gammaln(L, l1, l2_min, l2_max))),
        ("spin0 g-table     ", lambda: jax.jit(lambda L: spin0_body(L, l1, l2_min, l2_max))),
    ]
    for name, make in rows:
        jax.clear_caches()
        f = make()
        row, _ = _time(lambda f=f: f(L), repeat)
        _print(f"    {name} {row:>26}")


# ---------------------------------------------------------------------------
# [2] full estimators
# ---------------------------------------------------------------------------

def bench_estimators(l_max, repeat, spec, nl, n_L_sample):
    import augr.delensing_fullsky_jax as dj
    nl_tt, nl_ee, nl_bb = nl
    Ls = np.arange(2, l_max + 1, dtype=float)
    ls = np.arange(2, 301, dtype=float)
    kw = dict(n_L_sample=n_L_sample)
    grid = "exact every-L grid" if n_L_sample is None else f"sampled grid, n_L_sample={n_L_sample}"
    fns = {
        "N0 TT": lambda: dj.compute_n0_tt_fullsky_jax(Ls, spec, nl_tt, 2, l_max, **kw),
        "N0 EE": lambda: dj.compute_n0_ee_fullsky_jax(Ls, spec, nl_ee, 2, l_max, **kw),
        "N0 TE": lambda: dj.compute_n0_te_fullsky_jax(Ls, spec, nl_tt, nl_ee, 2, l_max, **kw),
        "N0 EB": lambda: dj.compute_n0_eb_fullsky_jax(Ls, spec, nl_ee, nl_bb, 2, l_max, **kw),
        "N0 TB": lambda: dj.compute_n0_tb_fullsky_jax(Ls, spec, nl_tt, nl_bb, 2, l_max, **kw),
        "kernel": lambda: dj.lensing_kernel_fullsky_jax(ls, Ls, spec, 2, l_max),
    }
    _print(f"\n[2] full-sky estimators, l_max_qe={l_max}, {grid}: SG -> closed")
    for name, f in fns.items():
        # jax.jit of a fresh lambda per variant; f itself is re-traced by jit
        _pair(name, lambda f=f: jax.jit(lambda: f()), repeat)


# ---------------------------------------------------------------------------
# [3] iterate_delensing end to end
# ---------------------------------------------------------------------------

def bench_iterate(l_max, spec, nl, n_L_sample, *, dense, numpy_backend):
    nl_tt, nl_ee, nl_bb = nl
    kw = dict(L_max=l_max, l_max_qe=l_max, n_iter=2, fullsky=True)
    _print(f"\n[3] iterate_delensing(fullsky=True, n_iter=2, l_max_qe={l_max})")

    def run(backend, n_L):
        return lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend=backend,
                                         n_L_sample=n_L, **kw).cl_bb_res

    # Eager lax.map callers re-trace per call, so one call per row is honest.
    t, v_sg = _timed_variant(lambda: run("jax", n_L_sample), "sg", 1)
    _print(f"    jax, n_L_sample={n_L_sample:<4d} SG tables (pre-#48)   {t:>26}   [{_now()}]")
    t, v_new = _timed_variant(lambda: run("jax", n_L_sample), "closed", 1)
    worst = _check_agree("iterate_delensing", v_sg, v_new)
    _print(f"    jax, n_L_sample={n_L_sample:<4d} closed form           {t:>26}"
           f"  agree {worst:.0e}   [{_now()}]")
    if dense:
        t, _ = _timed_variant(lambda: run("jax", None), "closed", 1)
        _print(f"    jax, exact every-L grid, closed form      {t:>26}   [{_now()}]")
    if numpy_backend:
        workers = os.environ.get("AUGR_DELENS_WORKERS", "cpu_count")
        t, v_np = _time(run("numpy", n_L_sample), 1)
        worst = _check_agree("numpy vs jax", v_np, v_new)
        _print(f"    numpy, n_L_sample={n_L_sample:<4d} (ProcessPool {workers} workers) {t:>26}"
               f"  agree {worst:.0e}   [{_now()}]")


# ---------------------------------------------------------------------------
# [4] MASTER coupling matrices
# ---------------------------------------------------------------------------

def bench_master(repeat):
    import healpy as hp

    from augr.pseudo_cl_jax import coupling_matrices, mask_power_spectrum
    _print("\n[4] pseudo_cl_jax.coupling_matrices (MASTER M+, M-): SG -> closed")
    for nside, lmax in ((128, 192), (256, 512)):
        theta = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0]
        mask = jnp.asarray((np.abs(np.cos(theta)) > 0.3).astype(float))
        w_ell = mask_power_spectrum(mask, nside=nside, lmax_mask=2 * lmax)
        def make(w=w_ell, lmax=lmax):
            f = jax.jit(lambda w: coupling_matrices(w, lmax=lmax))
            return lambda: f(w)
        _pair(f"lmax={lmax:4d}", make, repeat)


# ---------------------------------------------------------------------------
# [5] DelensCoupling build + design gradient
# ---------------------------------------------------------------------------

def bench_coupling(spec, n_L_sample, l_max_qes):
    from augr.optimize import DelensCoupling
    d = dict(n_det=jnp.asarray((200.0, 400.0, 200.0)), net=jnp.asarray((60.0, 50.0, 80.0)),
             beam=jnp.asarray((40.0, 30.0, 20.0)), eta=jnp.asarray((0.5, 0.5, 0.5)),
             mission_years=4.0, f_sky=0.6)
    _print("\n[5] DelensCoupling (3-band design): build + jit'd design gradient")
    for l_max_qe in l_max_qes:
        for fullsky, nL in ((False, None), (True, n_L_sample)):
            jax.clear_caches()
            c0, w0 = time.process_time(), time.perf_counter()
            c = DelensCoupling.build(lensing_spectra=spec, l_max_qe=l_max_qe, n_iter=2,
                                     fullsky=fullsky, n_L_sample=nL, **d)
            jax.block_until_ready(c.cl_bb_res0)
            tb = Row(time.perf_counter() - w0, time.process_time() - c0)

            def total(s, c=c):
                return jnp.sum(c.residual(d["n_det"], d["net"], d["beam"] * jnp.exp(s),
                                          d["eta"], d["mission_years"], d["f_sky"]))
            g = jax.jit(jax.grad(total))
            tg, _ = _time(lambda g=g: g(jnp.asarray(0.0)), 1)
            tag = f"full-sky n_L={nL}" if fullsky else "flat-sky"
            _print(f"    l_max_qe={l_max_qe}  {tag:18s} build {tb:>26}   grad {tg:>26}   [{_now()}]")


# ---------------------------------------------------------------------------

def _env_banner():
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_NUM_CPU_DEVICES",
                                          "AUGR_DELENS_WORKERS", "SLURM_CPUS_PER_TASK")}
    _print(f"{platform.node()}  {_now()}  python {sys.version.split()[0]}  jax {jax.__version__}")
    _print(f"cpu_count {os.cpu_count()}  schedulable {aff}  jax devices {jax.device_count()}  "
           f"os threads at start {_threads()}")
    _print("env " + "  ".join(f"{k}={v}" for k, v in env.items() if v is not None))
    _print("columns: wall (eff = process CPU time / wall, i.e. cores kept busy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-max", type=int, nargs="+", default=[1500, 3000])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n-L-sample", type=int, default=None,
                    help="sampled grid for [2]/[3]/[5]; default = the production default at each l_max")
    ap.add_argument("--dense", action="store_true",
                    help="[2]: run on the exact every-L grid instead of the sampled one; "
                         "[3]: add the exact-grid row. Serial SG rows, ~13x the sweeps.")
    ap.add_argument("--numpy", action="store_true", help="[3]: add the numpy ProcessPool backend row")
    ap.add_argument("--skip", nargs="*", default=[], choices=["1", "2", "3", "4", "5"])
    args = ap.parse_args()

    _env_banner()
    spec = load_lensing_spectra()
    inst = pico_like()
    nl = tuple(combined_noise_nl(inst, spec.ells, s) for s in ("TT", "EE", "BB"))

    def n_L_for(l_max):
        return args.n_L_sample if args.n_L_sample is not None else default_n_L_sample(l_max)

    for l_max in args.l_max:
        if "1" not in args.skip:
            bench_tables(l_max, args.repeat)
        if "2" not in args.skip:
            bench_estimators(l_max, args.repeat, spec, nl, None if args.dense else n_L_for(l_max))
    if "3" not in args.skip:
        l_max = min(args.l_max)
        bench_iterate(l_max, spec, nl, n_L_for(l_max), dense=args.dense, numpy_backend=args.numpy)
    if "4" not in args.skip:
        bench_master(args.repeat)
    if "5" not in args.skip:
        bench_coupling(spec, n_L_for(min(args.l_max)),
                       sorted({min(lm, 1600) for lm in args.l_max} | {800}))
    _print(f"\ndone {_now()}")


if __name__ == "__main__":
    main()
