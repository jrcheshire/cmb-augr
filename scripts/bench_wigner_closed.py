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

``--sweep`` runs the L-batching / device-sharding grid: one CHILD PROCESS per
``(l_batch, devices)`` pair, because ``JAX_NUM_CPU_DEVICES`` is read once at
import and the sharding decision is baked in at trace time -- never toggle
either knob inside one process. Children run ``--closed-only`` against the
``(1,1)`` child's values, and every row carries ``n -> n_pad``: padding repeats
the largest L sample, so ``(16,16)`` at 125 samples does 256 samples' work and
cannot win on wall time unless per-L efficiency more than doubles. ``eff``
over-counts once N > 1 (the replicated outer work is charged to every device),
so **wall time decides**, not eff.

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
    pixi run python scripts/bench_wigner_closed.py --sweep --l-max 1500 3000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
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

#: Parsed CLI, set once in ``main``; the timing helpers read the knobs off it.
ARGS = argparse.Namespace(l_batch=1, closed_only=False, agree_tol=1e-9)

#: name -> flat value array, written by ``--values-out`` and read as the
#: cross-process agreement baseline by ``--baseline-values``.
_VALUES: dict[str, np.ndarray] = {}
_BASELINE: dict[str, np.ndarray] = {}


def _bn_pair(text):
    """CLI \"B,N\" -> (l_batch, devices)."""
    b, n = text.split(",")
    return int(b), int(n)


def _print(*a):
    print(*a, flush=True)


def _result(key, row, **extra):
    """Machine-readable row for the sweep parent to collect."""
    payload = dict(key=key, wall=row.wall, eff=row.eff, **extra)
    print("#RESULT " + json.dumps(payload), flush=True)


def _peak_rss_gb():
    """Peak RSS of THIS process. RUSAGE_CHILDREN is a max over reaped
    children, not per child, so each child must report its own."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6   # bytes vs KB


def _flat(value):
    leaves = [np.asarray(x, dtype=float).ravel() for x in jax.tree.leaves(value)]
    return np.concatenate(leaves) if leaves else np.zeros(0)


def _agree_with_baseline(name, value):
    """Relative departure from the ``(1,1)`` child's value, or nan if unknown."""
    got = _flat(value)
    _VALUES[name] = got
    want = _BASELINE.get(name)
    if want is None or want.shape != got.shape:
        return float("nan")
    finite = np.isfinite(got) & np.isfinite(want)
    scale = np.maximum(np.abs(got[finite]), np.abs(want[finite]))
    rel = np.abs(got[finite] - want[finite]) / np.where(scale > 0, scale, 1.0)
    return float(rel.max()) if rel.size else 0.0


def _agree_tag(worst):
    if not np.isfinite(worst):
        return "agree    n/a"
    flag = "  MISMATCH" if worst > ARGS.agree_tol else ""
    return f"agree {worst:.0e}{flag}"


def _pad_note(n, l_batch, n_dev):
    """``n -> n_pad`` for an L grid of length n under (l_batch, devices)."""
    m = l_batch * n_dev
    n_pad = -(-n // m) * m
    return f"{n}->{n_pad}"


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


def _pair(name, make_fn, repeat, *, pad=""):
    """SG -> closed pair on one fresh callable per variant; values must agree.

    Under ``--closed-only`` the SG variant is skipped entirely (its ``lax.scan``
    under a ``vmap`` batch is not a quantity anyone would run) and the closed
    row is checked against the ``(1,1)`` baseline values instead.
    """
    if ARGS.closed_only:
        t_new, v_new = _timed_variant(make_fn, "closed", repeat)
        worst = _agree_with_baseline(name, v_new)
        _print(f"    {name:7s} {t_new:>26}  {pad:>10}  {_agree_tag(worst)}   [{_now()}]")
        _result(name, t_new, pad=pad, agree=worst)
        return None, t_new
    _SG_EXTENDED[0] = 0
    t_sg, v_sg = _timed_variant(make_fn, "sg", repeat)
    ext = "  (SG grid extended)" if _SG_EXTENDED[0] else ""
    t_new, v_new = _timed_variant(make_fn, "closed", repeat)
    worst = _check_agree(name, v_sg, v_new)
    _agree_with_baseline(name, v_new)
    _print(f"    {name:7s} {t_sg:>26} -> {t_new:>26}   ({t_sg.wall / t_new.wall:4.1f}x)"
           f"  agree {worst:.0e}{ext}   [{_now()}]")
    _result(name, t_new, pad=pad, agree=worst, sg_wall=t_sg.wall)
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
    from augr.delensing import _fullsky_L_samples
    nl_tt, nl_ee, nl_bb = nl
    Ls = np.arange(2, l_max + 1, dtype=float)
    ls = np.arange(2, 301, dtype=float)
    kw = dict(n_L_sample=n_L_sample, l_batch=ARGS.l_batch)
    grid = "exact every-L grid" if n_L_sample is None else f"sampled grid, n_L_sample={n_L_sample}"
    pad = _pad_note(len(_fullsky_L_samples(Ls, n_L_sample)), ARGS.l_batch, _devices())
    fns = {
        "N0 TT": lambda: dj.compute_n0_tt_fullsky_jax(Ls, spec, nl_tt, 2, l_max, **kw),
        "N0 EE": lambda: dj.compute_n0_ee_fullsky_jax(Ls, spec, nl_ee, 2, l_max, **kw),
        "N0 TE": lambda: dj.compute_n0_te_fullsky_jax(Ls, spec, nl_tt, nl_ee, 2, l_max, **kw),
        "N0 EB": lambda: dj.compute_n0_eb_fullsky_jax(Ls, spec, nl_ee, nl_bb, 2, l_max, **kw),
        "N0 TB": lambda: dj.compute_n0_tb_fullsky_jax(Ls, spec, nl_tt, nl_bb, 2, l_max, **kw),
        "kernel": lambda: dj.lensing_kernel_fullsky_jax(ls, Ls, spec, 2, l_max,
                                                       l_batch=ARGS.l_batch),
    }
    if ARGS.closed_only:
        fns = {k: v for k, v in fns.items() if k in ARGS.rows}
    _print(f"\n[2] full-sky estimators, l_max_qe={l_max}, {grid}, "
           f"l_batch={ARGS.l_batch} devices={_devices()} n_pad {pad}")
    for name, f in fns.items():
        # jax.jit of a fresh lambda per variant; f itself is re-traced by jit
        _pair(f"[2]{name}@{l_max}", lambda f=f: jax.jit(lambda: f()), repeat, pad=pad)


# ---------------------------------------------------------------------------
# [3] iterate_delensing end to end
# ---------------------------------------------------------------------------

def bench_iterate(l_max, spec, nl, n_L_sample, *, dense, numpy_backend):
    nl_tt, nl_ee, nl_bb = nl
    kw = dict(L_max=l_max, l_max_qe=l_max, n_iter=2, fullsky=True)
    _print(f"\n[3] iterate_delensing(fullsky=True, n_iter=2, l_max_qe={l_max})")

    def run(backend, n_L):
        return lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend=backend,
                                         n_L_sample=n_L, l_batch=ARGS.l_batch,
                                         **kw).cl_bb_res

    if ARGS.closed_only:
        t, v = _timed_variant(lambda: run("jax", n_L_sample), "closed", 1)
        worst = _agree_with_baseline("[3]iterate", v)
        _print(f"    jax, n_L_sample={n_L_sample:<4d} l_batch={ARGS.l_batch:<4d}"
               f"{t:>26}  {_agree_tag(worst)}   [{_now()}]")
        _result("[3]iterate", t, agree=worst)
        return

    # Eager lax.map callers re-trace per call, so one call per row is honest.
    t, v_sg = _timed_variant(lambda: run("jax", n_L_sample), "sg", 1)
    _print(f"    jax, n_L_sample={n_L_sample:<4d} SG tables (pre-#48)   {t:>26}   [{_now()}]")
    t, v_new = _timed_variant(lambda: run("jax", n_L_sample), "closed", 1)
    worst = _check_agree("iterate_delensing", v_sg, v_new)
    _agree_with_baseline("[3]iterate", v_new)
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
        _pair(f"[4]lmax={lmax}", make, repeat)


# ---------------------------------------------------------------------------
# [5] DelensCoupling build + design gradient
# ---------------------------------------------------------------------------

def bench_coupling(spec, n_L_sample, l_max_qes):
    """The production quantity: one delensing solve, then the design gradient."""
    from augr.delensing import _fullsky_L_samples
    from augr.optimize import DelensCoupling
    d = dict(n_det=jnp.asarray((200.0, 400.0, 200.0)), net=jnp.asarray((60.0, 50.0, 80.0)),
             beam=jnp.asarray((40.0, 30.0, 20.0)), eta=jnp.asarray((0.5, 0.5, 0.5)),
             mission_years=4.0, f_sky=0.6)
    _print("\n[5] DelensCoupling (3-band design): build + jit'd design gradient"
           f"  l_batch={ARGS.l_batch} devices={_devices()}")
    # flat-sky ignores l_batch and sharding entirely -- it is the serial
    # baseline, so the sweep children only pay for it once, at (1, 1).
    arms = [(True, n_L_sample)] if ARGS.closed_only else [(False, None), (True, n_L_sample)]
    for l_max_qe in l_max_qes:
        for fullsky, nL in arms:
            lb = ARGS.l_batch if fullsky else 1
            jax.clear_caches()
            c0, w0 = time.process_time(), time.perf_counter()
            c = DelensCoupling.build(lensing_spectra=spec, l_max_qe=l_max_qe, n_iter=2,
                                     fullsky=fullsky, n_L_sample=nL, l_batch=lb, **d)
            jax.block_until_ready(c.cl_bb_res0)
            tb = Row(time.perf_counter() - w0, time.process_time() - c0)

            def total(s, c=c):
                return jnp.sum(c.residual(d["n_det"], d["net"], d["beam"] * jnp.exp(s),
                                          d["eta"], d["mission_years"], d["f_sky"]))
            g = jax.jit(jax.grad(total))
            tg, gval = _time(lambda g=g: g(jnp.asarray(0.0)), 1)
            tag = f"full-sky n_L={nL}" if fullsky else "flat-sky"
            pad = (_pad_note(len(_fullsky_L_samples(np.arange(2, l_max_qe + 1), nL)),
                             lb, _devices()) if fullsky else "-")
            worst = _agree_with_baseline(f"[5]grad@{l_max_qe}:{tag}", gval)
            _print(f"    l_max_qe={l_max_qe}  {tag:18s} build {tb:>26}   grad {tg:>26}"
                   f"  {pad:>10}  {_agree_tag(worst)}   [{_now()}]")
            _result(f"[5]grad@{l_max_qe}:{tag}", tg, pad=pad, agree=worst,
                    build_wall=tb.wall)


# ---------------------------------------------------------------------------

def _env_banner():
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_NUM_CPU_DEVICES",
                                          "AUGR_DELENS_WORKERS", "SLURM_CPUS_PER_TASK")}
    _print(f"{platform.node()}  {_now()}  python {sys.version.split()[0]}  jax {jax.__version__}")
    _print(f"cpu_count {os.cpu_count()}  schedulable {aff}  jax devices {jax.device_count()}  "
           f"os threads at start {_threads()}")
    _print("env " + "  ".join(f"{k}={v}" for k, v in env.items() if v is not None))
    import augr.delensing_fullsky_jax as dj
    _print(f"l_batch {ARGS.l_batch}  _shard_devices() {dj._shard_devices()}  "
           f"{dj._NO_SHARD_ENV}={os.environ.get(dj._NO_SHARD_ENV, '')!r}")
    _print("columns: wall (eff = process CPU time / wall, i.e. cores kept busy). "
           "eff over-counts at devices > 1: wall decides.")


def _devices():
    """Devices the per-L map will actually shard over in THIS process."""
    import augr.delensing_fullsky_jax as dj
    return dj._shard_devices()


#: (l_batch, devices) pairs. (16,16) pads 125 -> 256 L at l_max_qe=1500, so it
#: is carried at 1500 only and labelled by its n_pad; it cannot win on wall
#: time unless per-L efficiency more than doubles.
_SWEEP_GRID = [(1, 1), (4, 1), (16, 1), (1, 4), (1, 16), (4, 4), (16, 16)]


def _child_cmd(args, l_batch, devices, *, values_out=None, baseline=None):
    cmd = [sys.executable, os.path.abspath(__file__),
           "--l-max", *[str(x) for x in args.l_max],
           "--repeat", str(args.repeat),
           "--l-batch", str(l_batch),
           "--rows", *args.rows,
           "--agree-tol", repr(args.agree_tol),
           "--no-lmax-cap",
           # [1] and [4] do not touch the per-L map; the caller's own --skip
           # rides along so a smoke can drop [3] as well.
           "--skip", *sorted({"1", "4"} | set(args.skip))]
    if devices > 1:
        cmd += ["--devices", str(devices)]
    if values_out:
        cmd += ["--values-out", values_out]
    else:
        cmd += ["--closed-only"]
    if baseline:
        cmd += ["--baseline-values", baseline]
    return cmd


def _sweep(args):
    """One fresh child per (l_batch, devices): both knobs are import/trace time."""
    grid = args.sweep_grid or _SWEEP_GRID
    baseline = os.path.abspath(args.sweep_baseline)
    rows = []
    _print(f"\n=== sweep {grid} at l_max {args.l_max}; baseline values -> {baseline}")
    for l_batch, devices in grid:
        if l_batch * devices >= 256 and max(args.l_max) > 1500:
            _print(f"\n--- skip (B={l_batch}, N={devices}) above l_max 1500: "
                   "padding would dominate")
            continue
        first = (l_batch, devices) == (1, 1)
        cmd = _child_cmd(args, l_batch, devices,
                         values_out=baseline if first else None,
                         baseline=None if first else baseline)
        _print(f"\n--- child B={l_batch} N={devices}: {' '.join(cmd[1:])}   [{_now()}]")
        env = dict(os.environ)
        env.pop("AUGR_DELENS_NO_SHARD", None)
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        mine, peak = [], float("nan")
        for line in proc.stdout.splitlines():
            if line.startswith("#RESULT "):
                r = json.loads(line[len("#RESULT "):])
                if r["key"] == "peak_rss":
                    peak = r["peak_rss_gb"]
                    continue
                r.update(l_batch=l_batch, devices=devices)
                mine.append(r)
            else:
                _print("    | " + line)
        for r in mine:                       # the child's own RSS, not RUSAGE_CHILDREN
            r["peak_rss_gb"] = peak
        rows.extend(mine)
        if proc.returncode != 0:
            _print(f"    FAILED rc={proc.returncode}")
            _print("    stderr: " + proc.stderr[-2000:])
    _print("\n=== sweep table (wall seconds; eff over-counts at N>1, so wall decides)")
    _print("    agree: (1,1) rows are SG vs closed form; every other row is that "
           "row's value vs the (1,1) baseline")
    _print(f"    {'key':28s} {'B':>3s} {'N':>3s} {'n->n_pad':>12s} "
           f"{'wall':>10s} {'eff':>7s} {'peakRSS':>9s} {'agree':>10s}")
    for r in rows:
        _print(f"    {r['key']:28s} {r['l_batch']:3d} {r['devices']:3d} "
               f"{r.get('pad', '-'):>12s} {r['wall']:10.3f} {r['eff']:7.1f} "
               f"{r.get('peak_rss_gb', float('nan')):8.2f}G "
               f"{r.get('agree', float('nan')):10.1e}")
    _print(f"\nsweep done {_now()}")


def _relaunch_with_devices(args):
    """Re-exec self with JAX_NUM_CPU_DEVICES set; JAX refuses it post-init."""
    env = dict(os.environ)
    env["JAX_NUM_CPU_DEVICES"] = str(args.devices)
    if args.no_shard:
        env["AUGR_DELENS_NO_SHARD"] = "1"
    env["_AUGR_BENCH_CHILD"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    return subprocess.run(cmd, env=env).returncode


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
    ap.add_argument("--l-batch", type=int, default=1,
                    help="L values vmapped into each step of the full-sky per-L map")
    ap.add_argument("--devices", type=int, default=1,
                    help="CPU devices to shard the L grid over; re-execs self with "
                         "JAX_NUM_CPU_DEVICES (JAX reads it once, at import)")
    ap.add_argument("--no-shard", action="store_true",
                    help="with --devices: set AUGR_DELENS_NO_SHARD=1 in the child "
                         "(the opt-out control -- same device count, no shard_map)")
    ap.add_argument("--closed-only", action="store_true",
                    help="skip the Schulten-Gordon variants and check values against "
                         "--baseline-values instead (SG under a vmap batch is not a "
                         "quantity anyone runs)")
    ap.add_argument("--rows", nargs="*", default=["N0 TT", "N0 EB", "kernel"],
                    help="[2] rows to keep under --closed-only")
    ap.add_argument("--values-out", help="write this run's values as the sweep baseline")
    ap.add_argument("--baseline-values", help="npz of (1,1) values to check against")
    ap.add_argument("--agree-tol", type=float, default=4e-13,
                    help="flag a row MISMATCH above this relative departure from the "
                         "baseline; default = the measured l_batch gradient gate")
    ap.add_argument("--no-lmax-cap", action="store_true",
                    help="[5]: keep the full l_max list instead of capping at 1600, so "
                         "the 3000 gradient row exists")
    ap.add_argument("--sweep", action="store_true",
                    help="run the (l_batch, devices) grid, one child process each")
    ap.add_argument("--sweep-grid", type=_bn_pair, nargs="*", default=None,
                    help='pairs like "1,1 4,2"; default is the full grid')
    ap.add_argument("--sweep-baseline", default="bench_sweep_baseline.npz")
    args = ap.parse_args()

    global ARGS
    ARGS = args
    if args.sweep:
        return _sweep(args)
    if args.devices > 1 and not os.environ.get("_AUGR_BENCH_CHILD"):
        return sys.exit(_relaunch_with_devices(args))
    if args.devices > 1:
        if jax.device_count() != args.devices:
            raise RuntimeError(f"sharding not engaged: jax.device_count() "
                               f"{jax.device_count()} != --devices {args.devices}")
        want = 1 if args.no_shard else args.devices
        if _devices() != want:
            raise RuntimeError(f"sharding not engaged: _shard_devices() {_devices()} "
                               f"!= {want}")
    if args.baseline_values:
        _BASELINE.update(dict(np.load(args.baseline_values)))

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
        cap = (lambda lm: lm) if args.no_lmax_cap else (lambda lm: min(lm, 1600))
        bench_coupling(spec, n_L_for(min(args.l_max)),
                       sorted({cap(lm) for lm in args.l_max} | {800}))
    if args.values_out:
        np.savez(args.values_out, **_VALUES)
        _print(f"\nbaseline values -> {args.values_out} ({len(_VALUES)} keys)")
    _print(f"peak RSS {_peak_rss_gb():.2f} GB")
    print("#RESULT " + json.dumps(dict(key="peak_rss", wall=float("nan"),
                                       eff=float("nan"),
                                       peak_rss_gb=_peak_rss_gb())), flush=True)
    _print(f"\ndone {_now()}")


if __name__ == "__main__":
    main()
