#!/bin/bash
# TACC Vista Grace-Grace (GPU-less, dual-Grace aarch64, ~144 cores/node) launcher for the
# design active-subspace + HL-EIG scan.
#
# Venue + sizing are from MEASURED verify runs (2026-07-02, nside=128/lmax=192/n_sims=30):
#   - jit compile ~6 min/process (jaxht>=0.2.0 looped FFT; the old unrolled mode was the
#     42-min/uncompilable wall). H200 GPU loses to Grace CPU on this graph (2.4 h vs 1.85 h
#     per grad eval -- small-lmax SHTs are launch-latency-bound), so CPU is the venue.
#   - steady-state grad eval ~1.85 h/design at ~1.6 cores/process.
#   - memory ~25-30 GB/worker (autodiff residency) -> ~8 workers per ~237 GB GG node.
#     n_sims=100 is ~3x time AND ~3x memory (~2-3 workers/node): NOT single-node-viable
#     inside the 48 h wall -- needs the --design-range array pattern below, or the remat
#     memory work in the traced forward.
#
# The driver is INCREMENTAL AND RESUMABLE: every design gradient / scan point lands in its
# own npz under --parts-dir as it finishes; reruns skip finished parts, so a timed-out job
# resumes where it died and this script can simply be resubmitted. Multi-node: submit
# disjoint --design-range slices into the SAME --parts-dir/--out (sbatch array), then one
# finishing run (full range, all cached) continues through the eigendecomp + scan.
#
# Parallelism: the driver fans design-gradients AND scan points over an augr.parallel
# process pool (--workers); each worker pins ducc/BLAS to 1 thread. Do NOT set
# OMP_NUM_THREADS high (144 workers x N threads oversubscribes).
#
#SBATCH -J augr_subspace
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1                          # single task; parallelism is the in-process pool
#SBATCH -t 30:00:00                   # 96 designs / 8 workers x ~1.85 h + scan ~4 h + margin
#SBATCH -A JPL-PUB
#SBATCH -o subspace_%j.out
#SBATCH -e subspace_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Let pin_blas own the thread count (OMP=1 per worker). Clear any inherited cap.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS 2>/dev/null || true

cd "$WORK/vista/cmb-augr"

# lmax=192 is jht's HEALPix band-limit ceiling at nside=128 (lmax <= 1.5*nside, enforced by
# jht.wiener even under --backend ducc), still clean through the recomb bump. M=96 designs
# (>> D=8) gives a robust C. n_sims=30 clears the Hartlap floor (n_bins+2 = 26 at these
# bins); n_sims=100 needs the array pattern (see header). pysm3 runs on the Grace cores
# (--fg-model d1s1 generates inline), but the TEMPLATES must be pre-cached from the LOGIN
# node (compute nodes have no internet):
#   pixi run -e cpu python -c "import pysm3, pysm3.units as u; \
#       pysm3.Sky(nside=128, preset_strings=['d1','s1']).get_emission(100*u.GHz)"
pixi run -e cpu python scripts/active_subspace_hl_eig.py \
    --backend ducc --fft-mode looped --nside 128 --lmax 192 \
    --n-sims 30 --n-designs 96 --n-crn 1 \
    --workers 8 \
    --scan-points 11 --n-outer 1024 \
    --fg-model d1s1 \
    --compile-cache "$SCRATCH/xla_cache_cpu" \
    --out "$SCRATCH/subspace_d1s1_n128"
    # parts land in $SCRATCH/subspace_d1s1_n128_parts; resubmit this script to resume.
    # --design-range I0 I1                             # multi-node slice (shared --out)
    # --sky-cache-dir "$SCRATCH/sky_cache_d1s1_n128"   # optional: pin FG / decouple PySM

# First run on Vista: `pixi install --locked -e cpu`. The committed lock already contains
# the cpu env (pixi solves linux-aarch64 cross-platform; no on-node regen needed).
#
# Fallback venue if Grace-Grace is unavailable: Stampede3 SKX (x86) -- same command with the
# `default` env (ducc/pysm3 are in the host feature there) instead of `-e cpu`.
