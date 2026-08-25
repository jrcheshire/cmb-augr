"""What does crediting delensing actually cost the design forward?

The map-based EIG forward calls ``DelensCoupling.residual`` once per design
evaluation, so its cost is amortized over ``n_sims`` Monte-Carlo sims -- but
reverse-mode AD holds the per-step ``lax.scan`` residuals of the iterative QE
solve, and that is the shape that blows memory up. The Vista sizing (8 workers
per node at 25-30 GB each, ~1.85 h per design gradient) was measured with NO
delensing in the trace, so both halves of the budget are unverified with it on.

This isolates the delensing leg: no MC forward, no compsep, no ``nside``
dependence. The QE cost is set by ``l_max_qe`` and the residual grid alone, so a
measurement here transfers to production even though the MC part does not.

Reports, per ``l_max_qe``:

  build      one reference solve (paid once per worker at startup)
  value      forward-only residual (jitted, steady state)
  grad       jax.grad through the solve (jitted, steady state) -- the number
             that matters, and the one that carries the AD tape
  peak RSS   high-water mark, from the cgroup where available (sacct's MaxRSS
             is polled and under-reports; see reference_sacct_maxrss_undercounts)

Run under Slurm. On albireo every job goes through sbatch -- deneb is both the
submit host and a compute node, so a bare `pixi run` there is a rogue job.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from augr.delensing import load_lensing_spectra
from augr.design_packing import PackingDesignSpec
from augr.optimize import DelensCoupling, design_to_channels

# Mirrors scripts/active_subspace_hl_eig.py so the numbers transfer to the run.
FREQS_PER_GROUP = ((20.0,), (35.0,), (80.0, 115.0), (160.0, 225.0),
                   (315.0, 440.0), (615.0,))
FRAC_FID = (1 / 6,) * 6
APERTURE_FID, F_NUMBER_FID, YEARS_FID = 1.5, 1.8, 4.0
FP_DIAMETER_M, ETA_TOTAL, F_SKY = 0.3, 0.5, 0.6


def _peak_rss_gb() -> float:
    """High-water mark in GB. Prefer the cgroup; ru_maxrss is the fallback."""
    for p in ("/sys/fs/cgroup/memory.peak",
              "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(p) as fh:
                return int(fh.read().strip()) / 1024**3
        except (OSError, ValueError):
            continue
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS bytes.
    return ru / 1024**2 if os.uname().sysname == "Linux" else ru / 1024**3


def _timeit(fn, *args, repeat=3):
    """Compile-inclusive first call, then the best of `repeat` steady-state calls."""
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0

    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return t_compile, best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--l-max-qe", type=int, nargs="+", default=[500, 1000, 1500])
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=str, default=None, help="write JSON here")
    args = p.parse_args()

    spec = PackingDesignSpec(
        freqs_per_group=FREQS_PER_GROUP, frac_fid=FRAC_FID,
        aperture_fid=APERTURE_FID, f_number_fid=F_NUMBER_FID,
        years_fid=YEARS_FID, fp_diameter_m=FP_DIAMETER_M,
        f_bounds=(1.2, 3.0), ref_group=0, eta_total=ETA_TOTAL,
    )
    d0 = spec.design_pytree(jnp.zeros(spec.n_dim))
    n_det, net, beam = design_to_channels(
        d0["aperture_m"], d0["f_number"], spec.fp_diameter_m,
        d0["area_fractions"], spec.freqs_per_group,
    )
    ls_spec = load_lensing_spectra()

    print(f"jax {jax.__version__}  devices={jax.devices()}", flush=True)
    print(f"design: {len(np.asarray(n_det))} channels, "
          f"n_det={float(jnp.sum(n_det)):.0f}, "
          f"beam {float(jnp.min(beam)):.1f}-{float(jnp.max(beam)):.1f} arcmin",
          flush=True)
    print(f"n_iter={args.n_iter}  repeat={args.repeat}\n", flush=True)

    hdr = (f"{'l_max_qe':>9} {'build/s':>9} {'value/s':>9} {'grad/s':>9} "
           f"{'g/v':>6} {'compile/s':>10} {'peakRSS/GB':>11}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    for lmq in args.l_max_qe:
        gc.collect()
        t0 = time.perf_counter()
        dc = DelensCoupling.build(
            lensing_spectra=ls_spec, n_det=n_det, net=net, beam=beam,
            eta=ETA_TOTAL, mission_years=YEARS_FID, f_sky=F_SKY,
            l_max_qe=lmq, n_iter=args.n_iter,
        )
        t_build = time.perf_counter() - t0

        # Beam is the design direction that matters: aperture enters through it,
        # and it is the direction the geometric-domain fix corrected.
        def value(b, _dc=dc):  # bind: _dc is a loop variable (ruff B023)
            return jnp.sum(_dc.residual(n_det, net, b, ETA_TOTAL,
                                        YEARS_FID, F_SKY))

        jv = jax.jit(value)
        jg = jax.jit(jax.grad(value))

        _, t_value = _timeit(jv, beam, repeat=args.repeat)
        t_gc, t_grad = _timeit(jg, beam, repeat=args.repeat)
        peak = _peak_rss_gb()

        print(f"{lmq:>9d} {t_build:>9.2f} {t_value:>9.2f} {t_grad:>9.2f} "
              f"{t_grad / t_value:>6.2f} {t_gc:>10.2f} {peak:>11.2f}",
              flush=True)
        rows.append(dict(l_max_qe=lmq, n_iter=args.n_iter, build_s=t_build,
                         value_s=t_value, grad_s=t_grad, compile_s=t_gc,
                         peak_rss_gb=peak))

    if len(rows) > 1:
        def expo(key):
            x = np.log([r["l_max_qe"] for r in rows])
            y = np.log([r[key] for r in rows])
            return np.polyfit(x, y, 1)[0]
        print(f"\nscaling in l_max_qe:  grad ~ L^{expo('grad_s'):.2f}   "
              f"peakRSS ~ L^{expo('peak_rss_gb'):.2f}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
