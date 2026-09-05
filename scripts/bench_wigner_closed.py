"""
bench_wigner_closed.py -- before/after timings for the issue #48 Wigner-3j work.

Rows (one machine, jit-warm, median of ``--repeat``):

  1. per-L tables: ``spin2_body`` closed form vs the retained Schulten-Gordon
     scan ``_spin2_body_sg``; ``spin0_body`` g-table vs ``_spin0_body_gammaln``.
  2. full estimators: ``compute_n0_{tt,ee,te,eb,tb}_fullsky_jax`` and the
     lensing kernel on the dense L grid, closed form vs SG (monkeypatched).
  3. ``iterate_delensing(fullsky=True)`` end to end: JAX backend dense-L vs the
     sampled grid, and the numpy backend (ProcessPool) for reference.
  4. ``pseudo_cl_jax.coupling_matrices`` at lmax 192 and 512.
  5. ``DelensCoupling`` build + design gradient: flat-sky vs full-sky sampled.

Usage:  pixi run python scripts/bench_wigner_closed.py [--l-max 1500 3000] [--repeat 3]
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from augr.config import pico_like
from augr.delensing import iterate_delensing, load_lensing_spectra
from augr.instrument import combined_noise_nl


def _t(fn, repeat):
    fn()  # warm / compile
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


@contextmanager
def _sg_tables():
    """Swap the JAX cores back to the SG / gammaln reference implementations."""
    import augr.delensing_fullsky_jax as dj
    import augr.pseudo_cl_jax as pj
    from augr.wigner_jax import _spin0_body_gammaln, _spin2_body_sg
    with mock.patch.object(dj, "spin2_body", _spin2_body_sg), \
         mock.patch.object(dj, "spin0_body", _spin0_body_gammaln), \
         mock.patch.object(pj, "spin2_body", _spin2_body_sg):
        yield


def bench_tables(l_max, repeat):
    from augr.wigner_jax import _spin0_body_gammaln, _spin2_body_sg, spin0_body, spin2_body
    l1 = jnp.arange(2, l_max + 1, dtype=float)
    l2_min, l2_max = 2, l_max + l_max // 2
    L = float(l_max // 5)
    print(f"\n[1] per-L tables, l1=2..{l_max}, l2={l2_min}..{l2_max}, L={int(L)}")
    for name, f in [
        ("spin2 SG scan     ", jax.jit(lambda: _spin2_body_sg(L, l1, -2, 0, 2, l2_min, l2_max))),
        ("spin2 closed form ", jax.jit(lambda: spin2_body(L, l1, -2, 0, 2, l2_min, l2_max))),
        ("spin0 gammaln     ", jax.jit(lambda: _spin0_body_gammaln(L, l1, l2_min, l2_max))),
        ("spin0 g-table     ", jax.jit(lambda: spin0_body(L, l1, l2_min, l2_max))),
    ]:
        print(f"    {name} {1e3 * _t(f, repeat):8.1f} ms")


def bench_estimators(l_max, repeat, spec, nl):
    import augr.delensing_fullsky_jax as dj
    nl_tt, nl_ee, nl_bb = nl
    Ls = np.arange(2, l_max + 1, dtype=float)
    ls = np.arange(2, 301, dtype=float)
    fns = {
        "N0 TT": lambda: dj.compute_n0_tt_fullsky_jax(Ls, spec, nl_tt, 2, l_max),
        "N0 EE": lambda: dj.compute_n0_ee_fullsky_jax(Ls, spec, nl_ee, 2, l_max),
        "N0 TE": lambda: dj.compute_n0_te_fullsky_jax(Ls, spec, nl_tt, nl_ee, 2, l_max),
        "N0 EB": lambda: dj.compute_n0_eb_fullsky_jax(Ls, spec, nl_ee, nl_bb, 2, l_max),
        "N0 TB": lambda: dj.compute_n0_tb_fullsky_jax(Ls, spec, nl_tt, nl_bb, 2, l_max),
        "kernel": lambda: dj.lensing_kernel_fullsky_jax(ls, Ls, spec, 2, l_max),
    }
    print(f"\n[2] full-sky estimators, dense L grid (2..{l_max}), l_max_qe={l_max}: SG -> closed")
    for name, f in fns.items():
        jf = jax.jit(f)
        with _sg_tables():
            t_sg = _t(jax.jit(f), repeat)
        t_new = _t(jf, repeat)
        print(f"    {name:7s} {t_sg:7.2f} s -> {t_new:7.2f} s   ({t_sg / t_new:4.1f}x)")


def bench_iterate(l_max, repeat, spec, nl, n_L_sample):
    nl_tt, nl_ee, nl_bb = nl
    kw = dict(L_max=l_max, l_max_qe=l_max, n_iter=2, fullsky=True)
    print(f"\n[3] iterate_delensing(fullsky=True, n_iter=2, l_max_qe={l_max})")
    with _sg_tables():
        t = _t(lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend="jax", **kw).cl_bb_res, 1)
    print(f"    jax, dense L, SG tables (pre-#48)      {t:7.1f} s")
    t = _t(lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend="jax", **kw).cl_bb_res, 1)
    print(f"    jax, dense L, closed form              {t:7.1f} s")
    t = _t(lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend="jax",
                                     n_L_sample=n_L_sample, **kw).cl_bb_res, 1)
    print(f"    jax, n_L_sample={n_L_sample:<4d} closed form      {t:7.1f} s")
    if os.environ.get("BENCH_NUMPY", "1") == "1":
        t = _t(lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend="numpy", **kw).cl_bb_res, 1)
        print(f"    numpy, dense L (ProcessPool {os.environ.get('AUGR_DELENS_WORKERS', 'cpu_count')} workers) {t:7.1f} s")
        t = _t(lambda: iterate_delensing(spec, nl_tt, nl_ee, nl_bb, backend="numpy",
                                         n_L_sample=n_L_sample, **kw).cl_bb_res, 1)
        print(f"    numpy, n_L_sample={n_L_sample:<4d} (ProcessPool)      {t:7.1f} s")


def bench_master(repeat):
    import healpy as hp

    from augr.pseudo_cl_jax import coupling_matrices, mask_power_spectrum
    print("\n[4] pseudo_cl_jax.coupling_matrices (MASTER M+, M-): SG -> closed")
    for nside, lmax in ((128, 192), (256, 512)):
        theta = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0]
        mask = jnp.asarray((np.abs(np.cos(theta)) > 0.3).astype(float))
        w_ell = mask_power_spectrum(mask, nside=nside, lmax_mask=2 * lmax)
        f = jax.jit(lambda w=w_ell, lmax=lmax: coupling_matrices(w, lmax=lmax))
        with _sg_tables():
            t_sg = _t(jax.jit(lambda w=w_ell, lmax=lmax: coupling_matrices(w, lmax=lmax)), repeat)
        t_new = _t(f, repeat)
        print(f"    lmax={lmax:4d}  {t_sg:6.2f} s -> {t_new:6.2f} s   ({t_sg / t_new:4.1f}x)")


def bench_coupling(spec, n_L_sample):
    from augr.optimize import DelensCoupling
    d = dict(n_det=jnp.asarray((200.0, 400.0, 200.0)), net=jnp.asarray((60.0, 50.0, 80.0)),
             beam=jnp.asarray((40.0, 30.0, 20.0)), eta=jnp.asarray((0.5, 0.5, 0.5)),
             mission_years=4.0, f_sky=0.6)
    print("\n[5] DelensCoupling (3-band design): build + jit'd design gradient")
    for l_max_qe in (800, 1600):
        for fullsky, nL in ((False, None), (True, n_L_sample)):
            t0 = time.perf_counter()
            c = DelensCoupling.build(lensing_spectra=spec, l_max_qe=l_max_qe, n_iter=2,
                                     fullsky=fullsky, n_L_sample=nL, **d)
            jax.block_until_ready(c.cl_bb_res0)
            tb = time.perf_counter() - t0

            def total(s, c=c):
                return jnp.sum(c.residual(d["n_det"], d["net"], d["beam"] * jnp.exp(s),
                                          d["eta"], d["mission_years"], d["f_sky"]))
            g = jax.jit(jax.grad(total))
            g(0.0).block_until_ready()
            t0 = time.perf_counter()
            g(0.0).block_until_ready()
            tg = time.perf_counter() - t0
            tag = f"full-sky n_L={nL}" if fullsky else "flat-sky"
            print(f"    l_max_qe={l_max_qe}  {tag:18s} build {tb:6.1f} s   grad {tg:6.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-max", type=int, nargs="+", default=[1500, 3000])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n-L-sample", type=int, default=None,
                    help="sampled grid for [3]/[5]; default = the production default")
    ap.add_argument("--skip", nargs="*", default=[], choices=["1", "2", "3", "4", "5"])
    args = ap.parse_args()
    spec = load_lensing_spectra()
    inst = pico_like()
    nl = tuple(combined_noise_nl(inst, spec.ells, s) for s in ("TT", "EE", "BB"))
    n_L = args.n_L_sample
    if n_L is None:
        from augr.delensing import default_n_L_sample
        n_L = default_n_L_sample(max(args.l_max))
    print(f"jax {jax.__version__}, devices {jax.devices()}, PICO-like noise")
    for l_max in args.l_max:
        if "1" not in args.skip:
            bench_tables(l_max, args.repeat)
        if "2" not in args.skip:
            bench_estimators(l_max, args.repeat, spec, nl)
    if "3" not in args.skip:
        bench_iterate(min(args.l_max), args.repeat, spec, nl, n_L)
    if "4" not in args.skip:
        bench_master(args.repeat)
    if "5" not in args.skip:
        bench_coupling(spec, n_L)


if __name__ == "__main__":
    main()
