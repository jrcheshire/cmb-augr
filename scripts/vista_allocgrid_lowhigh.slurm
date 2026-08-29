#!/bin/bash
# TACC Vista Grace-Grace launcher for the focal-plane ALLOCATION-plane grid --
# value-only 2-D EIG surface over the low- and high-frequency FG-monitor allocation knobs
# (alloc@35 x alloc@615), everything else at fiducial. The poster/paper
# figure for "how should the cold focal plane be divided across bands?" -- the
# substitution question this map-based forward measures well (unlike aperture,
# whose response is delensing-only here and carries an open sign anomaly).
#
# Same machinery/venue as vista_isocost_grid.slurm (which see): fixed held-out
# ensemble, budget penalty off with per-point cost stored (the plane is nearly
# cost-neutral: +/-6% in the toy), zg_<k>.npz parts, resumable by resubmission.
# Measured per-point time on the iso grid: ~475 s value-only at 8 workers.
#
#SBATCH -J augr_allocgrid_lh
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1                          # single task; parallelism is the in-process pool
#SBATCH -t 8:00:00                    # 81 value-only points / 8 workers + compile + margin
#SBATCH -A JPL-PUB
#SBATCH -o allocgrid_lh_%j.out
#SBATCH -e allocgrid_lh_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS 2>/dev/null || true

cd "$WORK/cmb-augr"   # $WORK is already .../vista on Vista

# z ranges +/-0.75 in the allocation logits = area fractions ~0.10-0.26 per group
# around the 1/6 equal-area fiducial (verified in the toy smoke).
"$HOME/.pixi/bin/pixi" run --frozen -e cpu python scripts/active_subspace_hl_eig.py \
    --backend ducc --fft-mode looped --nside 128 --lmax 192 \
    --n-sims 30 \
    --workers 8 \
    --fg-model d1s1 \
    --grid-knob1 "alloc@35" -0.75 0.75 9 \
    --grid-knob2 "alloc@615" -0.75 0.75 9 \
    --compile-cache "$SCRATCH/xla_cache_cpu" \
    --out "$SCRATCH/allocgrid_lowhigh_d1s1_n128"
    # parts land in $SCRATCH/allocgrid_lowhigh_d1s1_n128_parts; resubmit this script to resume.
