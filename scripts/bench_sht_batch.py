"""Batched-vs-looped jht spherical-harmonic transforms, and a GPU bring-up check.

Two questions, one job:

1. **Does this machine run JAX at all, and how fast is fp64 here?** Blackwell
   (GB200) gives double precision materially less hardware than Hopper, and augr
   is fp64 throughout, so the realized number matters more than the vendor one.
   A missing sm_100 build would fail here rather than three stages later.

2. **Does batching independent transforms share their Legendre recursion?**
   ``jht.synthesis`` takes one map at a time, and the cleaner calls it
   ``J * n_band`` times per sim (126 at J=6, n_band=21) on independent inputs.
   The recursion is ~the whole cost of a transform and does **not** depend on the
   input alm, so ``jax.vmap`` should compute it once per batch rather than once
   per transform. Measured 7.5x at batch 32 on CPU; this asks the GPU.

Both the timing and the *structure* are reported: a jaxpr check reads whether the
recursion carry stayed unbatched, which is the mechanism the speedup is claimed
to come from. A wall-clock win with a batched carry would mean something else is
going on and the batch-size story would not transfer.

Traps this harness is built to avoid (see reference_jax_benchmark_traps):
the swept quantity is a **traced argument**, never closed over, so XLA cannot
constant-fold the answer; each variant gets a **fresh function object** plus
``jax.clear_caches()``, because ``jax.jit`` caches on the function object and
would otherwise re-run the first executable for every row; every print is
flushed, because a redirected log that shows nothing gets a running job killed;
and the two arms are **asserted to agree** before either timing is believed.

Usage
-----
    pixi run -e gpu python scripts/bench_sht_batch.py --nsides 128 256 --repeat 3
    pixi run python scripts/bench_sht_batch.py --nsides 64 --repeat 1   # CPU smoke
"""

from __future__ import annotations

import argparse
import gc
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

import augr  # noqa: F401  -- enables x64 at import, which jht requires


def _p(*a):
    print(*a, flush=True)


def _peak_gb():
    """Peak device bytes if the backend reports them, else None (CPU)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for k in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
        if k in stats:
            return stats[k] / 1e9
    return None


def bring_up(matmul_n: int) -> None:
    """Report the backend and a realized fp64 matmul rate."""
    _p("=== bring-up ===")
    _p(f"  platform      : {platform.platform()}")
    _p(f"  jax           : {jax.__version__}")
    _p(f"  backend       : {jax.default_backend()}")
    for d in jax.devices():
        _p(f"  device        : {d} kind={getattr(d, 'device_kind', '?')}")
    x64 = jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64
    _p(f"  x64 enabled   : {x64}")
    if not x64:
        raise SystemExit("x64 is off; every augr number would be wrong. Aborting.")

    n = matmul_n
    a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.float64)
    mm = jax.jit(lambda x: x @ x)
    jax.block_until_ready(mm(a))
    t0 = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(mm(a))
    dt = (time.perf_counter() - t0) / 3
    _p(f"  fp64 matmul   : n={n}  {dt * 1e3:.1f} ms  ->  {2 * n**3 / dt / 1e12:.2f} TFLOP/s")
    _p("")


def _scan_carries(jaxpr, out=None):
    """Every scan's carry avals, recursively."""
    if out is None:
        out = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "scan":
            n_carry = eqn.params.get("num_carry", 0)
            inner = eqn.params["jaxpr"]
            inner = getattr(inner, "jaxpr", inner)
            n_const = eqn.params.get("num_consts", 0)
            avals = [v.aval for v in inner.invars[n_const:n_const + n_carry]]
            if avals:
                out.append(avals)
        for v in eqn.params.values():
            sub = getattr(v, "jaxpr", None)
            if sub is not None:
                _scan_carries(getattr(sub, "jaxpr", sub), out)
            elif type(v).__name__ == "Jaxpr":
                _scan_carries(v, out)
    return out


def carry_is_shared(nside: int, lmax: int, spin: int, batch: int) -> bool | None:
    """True if the Legendre recursion state stayed unbatched under vmap.

    This is the *mechanism*, read off the jaxpr rather than inferred from a
    timing. The claim is that batching shares one recursion across the batch, so
    in the batched scan the real-valued recursion carry must keep its unbatched
    rank while the complex accumulators pick the batch axis up. A wall-clock win
    without this would be coming from somewhere else, and the batch-size story
    would not transfer to another shape.

    Returns None if no scan carrying both kinds of buffer is found.
    """
    import jht

    n_alm = jht.alm_size(lmax)
    shape = (batch, n_alm) if spin == 0 else (batch, 2, n_alm)
    alm = jnp.zeros(shape, dtype=jnp.complex128)
    fn = jax.vmap(lambda a: jht.synthesis(a, nside=nside, lmax=lmax, spin=spin))
    for avals in _scan_carries(jax.make_jaxpr(fn)(alm).jaxpr):
        real = [a for a in avals if a.dtype == jnp.float64]
        cplx = [a for a in avals if a.dtype == jnp.complex128]
        if not (real and cplx):
            continue
        # Batched accumulators carry the batch as a leading axis; the recursion
        # state should not.
        acc_batched = any(a.ndim >= 3 and a.shape[0] == batch for a in cplx)
        rec_batched = any(a.ndim >= 3 and a.shape[0] == batch for a in real)
        if acc_batched:
            return not rec_batched
    return None


def sweep(nsides, lmaxes, batches, spins, repeat: int) -> None:
    import jht

    for nside, lmax in zip(nsides, lmaxes, strict=True):
        for spin in spins:
            n_alm = jht.alm_size(lmax)
            _p(f"=== nside={nside} lmax={lmax} spin={spin} (n_alm={n_alm}) ===")
            _p(f"  {'B':>4} {'loop ms':>10} {'vmap ms':>10} {'loop/B':>9} "
               f"{'vmap/B':>9} {'speedup':>8} {'peakGB':>8}  {'agree':>10} carry")
            for B in batches:
                shape = (B, n_alm) if spin == 0 else (B, 2, n_alm)
                key = jax.random.PRNGKey(B)
                alm = (jax.random.normal(key, shape)
                       + 1j * jax.random.normal(key, shape)).astype(jnp.complex128)

                # Fresh function objects per variant; jax.jit caches on the object,
                # so reusing one would silently time the first executable twice.
                jax.clear_caches()
                gc.collect()

                def _one(a, nside=nside, lmax=lmax, spin=spin):
                    return jht.synthesis(a, nside=nside, lmax=lmax, spin=spin)

                loop_fn = jax.jit(lambda x: jnp.stack([_one(x[i]) for i in range(x.shape[0])]))
                vmap_fn = jax.jit(jax.vmap(_one))

                try:
                    o_loop = jax.block_until_ready(loop_fn(alm))
                    o_vmap = jax.block_until_ready(vmap_fn(alm))
                except Exception as exc:  # OOM at large B is a result, not a crash
                    _p(f"  {B:>4}  FAILED: {type(exc).__name__}: {str(exc)[:60]}")
                    continue

                d = float(np.max(np.abs(np.asarray(o_loop) - np.asarray(o_vmap))))
                scale = float(np.max(np.abs(np.asarray(o_loop)))) or 1.0
                agree = d / scale

                t0 = time.perf_counter()
                for _ in range(repeat):
                    jax.block_until_ready(loop_fn(alm))
                t_loop = (time.perf_counter() - t0) / repeat * 1e3
                t0 = time.perf_counter()
                for _ in range(repeat):
                    jax.block_until_ready(vmap_fn(alm))
                t_vmap = (time.perf_counter() - t0) / repeat * 1e3

                peak = _peak_gb()
                shared = carry_is_shared(nside, lmax, spin, B) if B > 1 else None
                _p(f"  {B:>4} {t_loop:>10.2f} {t_vmap:>10.2f} {t_loop / B:>9.3f} "
                   f"{t_vmap / B:>9.3f} {t_loop / t_vmap:>7.2f}x "
                   f"{(peak if peak is not None else float('nan')):>8.2f}  "
                   f"{agree:>10.2e} {shared}")
                if agree > 1e-10:
                    _p(f"       MISMATCH at B={B}: rel {agree:.3e} -- timings above are "
                       f"not comparable")
            _p("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsides", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--lmax-factor", type=float, default=1.5,
                    help="lmax = factor * nside (jht's validated band).")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--spins", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--matmul-n", type=int, default=4096)
    ap.add_argument("--fft-mode", default="looped",
                    help="jht azimuth FFT mode; 'looped' is mandatory at nside>=1024 "
                         "and is what production uses, so measure under it.")
    args = ap.parse_args()

    _p(f"# bench_sht_batch  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    bring_up(args.matmul_n)

    import jht

    if hasattr(jht, "set_azimuth_fft_mode"):
        jht.set_azimuth_fft_mode(args.fft_mode)
        _p(f"  jht azimuth FFT mode: {args.fft_mode}\n")

    lmaxes = [int(args.lmax_factor * n) for n in args.nsides]
    sweep(args.nsides, lmaxes, args.batches, args.spins, args.repeat)
    _p("# done")


if __name__ == "__main__":
    main()
