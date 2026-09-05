"""
l_sampling_convergence.py -- how many L samples does the full-sky N_0 need?

Issue #48 follow-up. ``iterate_delensing(fullsky=True)`` passes ``Ls = 2..L_max``
and the five N_0 estimators used to evaluate a Wigner sweep at every one of
them (~L_max sweeps per estimator). ``n_L_sample=n`` instead evaluates every
L < 20 plus ``n`` log-spaced samples and log-interpolates ``N_0^{-1}`` -- the
construction the lensing kernel has always used. This script measures, against
the dense grid, what that approximation costs on

  * the VALUE: ``N_0^MV(L)`` (max / median relative error over L),
    ``cl_bb_res(l)`` and ``A_lens_eff``;
  * the DERIVATIVE: ``d A_lens_eff / d ln(noise)`` for the BB and EE noise
    (repo rule: converge the derivative, not the value -- an ell cap validated
    on a value has twice been badly wrong for the design derivative).

Both the JAX backend (differentiable; remat on) and, for the values, the numpy
backend share ``_fullsky_L_samples`` so one table covers both. Runtimes of the
dense and sampled forwards are printed alongside.

Usage (from the cmb-augr checkout; ~30-40 min at the default settings on an M4 Max):

    pixi run python scripts/n0_validation/l_sampling_convergence.py \
        --l-max-qe 1000 1500 --n-samples 25 50 75 100 150 200 --out results.json
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from augr.config import litebird_like, pico_like
from augr.delensing import _delens_core, load_lensing_spectra
from augr.instrument import combined_noise_nl

N_ITER = 2
LS = jnp.arange(2, 301, dtype=float)


def _noise(inst, ells):
    return tuple(combined_noise_nl(inst, ells, s) for s in ("TT", "EE", "BB"))


def _core(spec, nl, l_max_qe, n_L, log_scale_bb=0.0, log_scale_ee=0.0):
    nl_tt, nl_ee, nl_bb = nl
    Ls = np.arange(2, l_max_qe + 1, dtype=float)  # concrete: sizes the L grid under jit
    return _delens_core(
        spec, nl_tt, nl_ee * jnp.exp(log_scale_ee), nl_bb * jnp.exp(log_scale_bb),
        LS, Ls, n_iter=N_ITER, l_min_qe=2, l_max_qe=l_max_qe, n_phi=32,
        fullsky=True, backend="jax", remat=True, n_L_sample=n_L)


def evaluate(spec, nl, l_max_qe, n_L):
    """Value + derivative bundle for one (l_max_qe, n_L) cell, jit-compiled."""
    def a_lens(s_bb, s_ee):
        return _core(spec, nl, l_max_qe, n_L, s_bb, s_ee)[2]

    fwd = jax.jit(lambda: _core(spec, nl, l_max_qe, n_L))
    grad = jax.jit(jax.grad(a_lens, argnums=(0, 1)))
    t0 = time.perf_counter()
    cl_res, n0, a_l, _ = fwd()
    jax.block_until_ready(cl_res)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_bb, g_ee = grad(0.0, 0.0)
    jax.block_until_ready(g_bb)
    t_grad = time.perf_counter() - t0
    return dict(cl_res=np.asarray(cl_res), n0=np.asarray(n0), A_L=float(a_l),
                dA_dlnNbb=float(g_bb), dA_dlnNee=float(g_ee),
                t_fwd=t_fwd, t_grad=t_grad)


def _rel(a, b):
    return np.abs(np.asarray(a) / np.asarray(b) - 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--l-max-qe", type=int, nargs="+", default=[1000, 1500])
    ap.add_argument("--n-samples", type=int, nargs="+",
                    default=[25, 50, 75, 100, 150, 200])
    ap.add_argument("--instruments", nargs="+", default=["pico_like", "litebird_like"])
    ap.add_argument("--out", default=None, help="JSON results file")
    args = ap.parse_args()

    spec = load_lensing_spectra()
    insts = {"pico_like": pico_like, "litebird_like": litebird_like}
    results = {}
    for name in args.instruments:
        nl = _noise(insts[name](), spec.ells)
        for l_max_qe in args.l_max_qe:
            key = f"{name}/l_max_qe={l_max_qe}"
            print(f"\n=== {key}: dense reference ...", flush=True)
            dense = evaluate(spec, nl, l_max_qe, None)
            print(f"    A_L={dense['A_L']:.6f}  dA/dlnN_bb={dense['dA_dlnNbb']:+.5e}  "
                  f"dA/dlnN_ee={dense['dA_dlnNee']:+.5e}  fwd {dense['t_fwd']:.0f}s  "
                  f"grad {dense['t_grad']:.0f}s", flush=True)
            rows = []
            hdr = (f"    {'n':>4s} {'n_L':>5s} | {'N0 max':>8s} {'N0 med':>8s} | "
                   f"{'clres max':>9s} {'A_L':>8s} | {'dA/dNbb':>8s} {'dA/dNee':>8s} | "
                   f"{'fwd s':>6s} {'grad s':>6s}")
            print(hdr, flush=True)
            from augr.delensing import _fullsky_L_samples
            Ls_np = np.arange(2, l_max_qe + 1)
            for n in args.n_samples:
                r = evaluate(spec, nl, l_max_qe, n)
                row = dict(
                    n=n, n_L=len(_fullsky_L_samples(Ls_np, n)),
                    n0_max=float(_rel(r["n0"], dense["n0"]).max()),
                    n0_med=float(np.median(_rel(r["n0"], dense["n0"]))),
                    clres_max=float(_rel(r["cl_res"], dense["cl_res"]).max()),
                    A_L=float(_rel(r["A_L"], dense["A_L"])),
                    dA_dlnNbb=float(_rel(r["dA_dlnNbb"], dense["dA_dlnNbb"])),
                    dA_dlnNee=float(_rel(r["dA_dlnNee"], dense["dA_dlnNee"])),
                    t_fwd=r["t_fwd"], t_grad=r["t_grad"])
                rows.append(row)
                print(f"    {n:4d} {row['n_L']:5d} | {row['n0_max']:8.1e} {row['n0_med']:8.1e} | "
                      f"{row['clres_max']:9.1e} {row['A_L']:8.1e} | {row['dA_dlnNbb']:8.1e} "
                      f"{row['dA_dlnNee']:8.1e} | {row['t_fwd']:6.1f} {row['t_grad']:6.1f}",
                      flush=True)
            results[key] = dict(
                dense=dict(A_L=dense["A_L"], dA_dlnNbb=dense["dA_dlnNbb"],
                           dA_dlnNee=dense["dA_dlnNee"], n_L=int(l_max_qe - 1),
                           t_fwd=dense["t_fwd"], t_grad=dense["t_grad"]),
                sampled=rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(n_iter=N_ITER, results=results), f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
