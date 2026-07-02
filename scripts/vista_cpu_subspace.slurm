#!/bin/bash
# TACC Vista Grace-Grace (GPU-less, dual-Grace aarch64, ~144 cores/node) launcher for the
# design active-subspace + HL-EIG scan.
#
# Why CPU on Grace-Grace (not the GH200 GPU): the gradient collect + HL scan are ~M
# independent design evaluations, which the driver fans over an augr.parallel process pool
# (--workers) -- a many-core CPU node is the natural venue, and it does not depend on the
# (still unverified) post-jaxht-0.2.0 GPU executable size. --backend ducc keeps the
# needlet-cleaner SHTs on the opaque ducc custom_call; the masked-Wiener stage stays jht
# regardless of backend (masking.py calls jht.wiener / jht.bandpower directly), so
# --fft-mode looped (jaxht>=0.2.0 chirp-z lax.scan; the driver default) is load-bearing
# here too: unrolled was uncompilable at nside>=64 (~42 min / 12 GB per process on Grace).
# conda-forge ships ducc0 + pysm3 for linux-aarch64, so the whole forward runs natively on
# the Grace cores (the `cpu` pixi env) -- same Vista allocation, no SKX/x86 detour.
#
# Parallelism: the driver fans the M design-gradients AND the HL scan across an
# augr.parallel process pool (--workers); each worker pins ducc/BLAS to 1 thread, so
# parallelism is from the pool, not threading. Do NOT set OMP_NUM_THREADS high (it would
# oversubscribe 144 workers x N threads).
#
#SBATCH -J augr_subspace
#SBATCH -p gg                         # Vista Grace-Grace CPU queue (confirm: sinfo -o '%P')
#SBATCH -N 1
#SBATCH -n 1                          # single task; parallelism is the in-process pool
#SBATCH -t 04:00:00
#SBATCH -A <JPL-ALLOCATION>           # <-- set your allocation (or override: sbatch -A ...)
#SBATCH -o subspace_%j.out
#SBATCH -e subspace_%j.err

set -euo pipefail

# Let pin_blas own the thread count (OMP=1 per worker). Clear any inherited cap.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS 2>/dev/null || true

cd "$WORK/cmb-augr"

# lmax=192 is jht's HEALPix band-limit ceiling at nside=128 (lmax <= 1.5*nside, enforced by
# jht.wiener even under --backend ducc) and is still clean through the recomb bump. M=96
# designs (>> D=8) gives a robust C and saturates ~96 cores (1 design/worker). pysm3 runs on
# the Grace cores, so --fg-model d1s1 generates inline -- the sky cache is OPTIONAL here --
# but the TEMPLATES must be pre-cached from the LOGIN node (compute nodes have no internet):
#   pixi run -e cpu python -c "import pysm3, pysm3.units as u; \
#       pysm3.Sky(nside=128, preset_strings=['d1','s1']).get_emission(100*u.GHz)"
pixi run -e cpu python scripts/active_subspace_hl_eig.py \
    --backend ducc --fft-mode looped --nside 128 --lmax 192 \
    --n-sims 100 --n-designs 96 --n-crn 1 \
    --workers 96 \
    --scan-points 11 --n-outer 1024 \
    --fg-model d1s1 \
    --out "$SCRATCH/subspace_d1s1_n128"
    # --sky-cache-dir "$SCRATCH/sky_cache_d1s1_n128"   # optional: pin FG / decouple PySM

# First run on Vista: `pixi install --locked -e cpu`. The committed lock already contains
# the cpu env (pixi solves linux-aarch64 cross-platform; no on-node regen needed).
#
# Fallback venue if Grace-Grace is unavailable: Stampede3 SKX (x86) -- same command with the
# `default` env (ducc/pysm3 are in the host feature there) instead of `-e cpu`.
