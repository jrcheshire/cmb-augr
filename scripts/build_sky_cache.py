"""build_sky_cache.py -- pre-generate FG sky caches for the pysm3-less GPU run.

Runs the PySM foreground generation (needs pysm3) on a capable machine and writes one
sky cache per CRN ensemble the active-subspace driver will request, so the aarch64 GPU
run loads them via ``--sky-cache-dir`` and never touches pysm3. The index scheme MUST
match ``scripts/active_subspace_hl_eig.py`` (``make_ctx(idx) -> base_seed =
SEED_STRIDE * (idx + 1)``), and ``--nside/--lmax/--n-sims/--fg-model`` must match the
driver run that consumes the caches.

Run on a pysm3 machine (the Mac ``default`` env, or an x86 node), then scp the dir to
Vista:

    pixi run python scripts/build_sky_cache.py --fg-model d1s1 --nside 128 --lmax 192 \
        --n-sims 64 --n-designs 24 --n-crn 1 --out-dir $SCRATCH/sky_cache_d1s1_n128

Storage note: the full per-sim ensemble is stored per cache (so stochastic models like
d10/d12 are captured). For a deterministic d1s1 sky the foreground alm are identical
across caches -- redundant but correct; a future optimization could store one FG map and
regenerate CMB+noise on the GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from active_subspace_hl_eig import (
    SEED_STRIDE,
    _build_mc_ctx,
    _spec,
    _static_pieces,
)

from augr.spectrum_stages import save_sky_cache


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fg-model", type=str, default="d1s1")
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--lmax", type=int, default=192)
    p.add_argument("--n-sims", type=int, default=64)
    p.add_argument("--n-designs", type=int, default=24)
    p.add_argument("--n-crn", type=int, default=1)
    p.add_argument(
        "--eval-index", type=int, default=9999, help="held-out scan ensemble index"
    )
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    fg_model = None if args.fg_model.lower() == "none" else args.fg_model
    spec = _spec()
    static = _static_pieces(args.nside, args.lmax)
    os.makedirs(args.out_dir, exist_ok=True)
    indices = [*range(args.n_designs * args.n_crn), args.eval_index]
    print(
        f"generating {len(indices)} sky caches "
        f"(fg={fg_model}, nside={args.nside}, lmax={args.lmax}, n_sims={args.n_sims}) "
        f"-> {args.out_dir}"
    )
    t0 = time.time()
    for k, idx in enumerate(indices):
        base_seed = SEED_STRIDE * (idx + 1)
        ctx = _build_mc_ctx(
            static,
            spec,
            base_seed=base_seed,
            n_sims=args.n_sims,
            nside=args.nside,
            lmax=args.lmax,
            fg_model=fg_model,
        )
        save_sky_cache(
            os.path.join(args.out_dir, f"sky_{idx}.npz"),
            ctx,
            fg_model=str(fg_model),
            base_seed=base_seed,
        )
        print(
            f"  [{k + 1}/{len(indices)}] [{time.time() - t0:.0f}s] wrote sky_{idx}.npz"
        )


if __name__ == "__main__":
    main()
