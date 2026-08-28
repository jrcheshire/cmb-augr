#!/bin/bash
# TACC Vista Grace-Grace launcher for the iso-cost (aperture x detector-count) grid --
# the value-only 2-D surface behind the poster/paper iso-cost figure
# (eig_production_scoping.md: at fixed budget the design volume collapses to a 2-D
# iso-cost surface; this evaluates the surface's EIG and stores every point's cost).
#
# Same venue + machinery as vista_cpu_subspace.slurm (which see for the sizing
# provenance). Differences:
#   - value-only (no jax.grad), so per-point time is well under the ~2030 s/design the
#     gradient stage measured in job 947183; 81 points / 8 workers fits in a few hours.
#   - one fixed held-out CRN ensemble (--eval-index default 9999) for every point, so
#     neighboring grid points differ by design, not by realization.
#   - budget penalty is OFF in the objective (pure Gaussian EIG surface); the dollar
#     cost of each point is stored in its part for the contour overlay.
#
# INCREMENTAL AND RESUMABLE like the subspace run: parts land in --parts-dir as
# iso_<k>.npz; resubmit this script to resume after a timeout.
#
#SBATCH -J augr_isocost
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1                          # single task; parallelism is the in-process pool
#SBATCH -t 8:00:00                    # 81 value-only points / 8 workers + compile + margin
#SBATCH -A JPL-PUB
#SBATCH -o isocost_%j.out
#SBATCH -e isocost_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Let pin_blas own the thread count (OMP=1 per worker). Clear any inherited cap.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS 2>/dev/null || true

cd "$WORK/cmb-augr"   # $WORK is already .../vista on Vista

# Axes: aperture 1.0-2.4 m spans the deployable-shield wall (1.45 m) and the F9
# ocean-recovery mass wall (2.26 m); fp diameter 0.19-0.48 m is x0.4-x2.5 in total
# detector count around the 0.3 m fiducial (n_det ~ fp_d^2). Both geometric, 9 points,
# geometric centers at the fiducials (1.55 m, 0.302 m). d1s1 templates must already be
# in ~/.cache/astropy (see vista_cpu_subspace.slurm).
"$HOME/.pixi/bin/pixi" run --frozen -e cpu python scripts/active_subspace_hl_eig.py \
    --backend ducc --fft-mode looped --nside 128 --lmax 192 \
    --n-sims 30 \
    --workers 8 \
    --fg-model d1s1 \
    --grid-aperture 1.0 2.4 9 \
    --grid-fpd 0.19 0.48 9 \
    --compile-cache "$SCRATCH/xla_cache_cpu" \
    --out "$SCRATCH/isocost_d1s1_n128"
    # parts land in $SCRATCH/isocost_d1s1_n128_parts; resubmit this script to resume.
