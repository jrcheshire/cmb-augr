#!/bin/bash
# albireo (deneb+antares): profile what crediting delensing costs the design forward.
#
# Runs on ANTARES deliberately -- it is the 124 GB node, and the open question is a
# memory ceiling, so the measurement needs headroom to find the true peak rather
# than being cgroup-killed at it. The cost is wall time: antares is 5-9x slower per
# task than deneb on identical work (measured 2026-08-17), so treat the ABSOLUTE
# seconds here as an upper bound and read the l_max_qe SCALING and the peak RSS as
# the transferable results.
#
# --mem is mandatory on this cluster: DefMemPerNode=UNLIMITED, so a job that omits
# it is allocated the whole node and blocks everything else. 100G leaves headroom
# under antares's 124000 MB for Jamie's refcat-lite jobs.
#
# --time is deliberately generous: the partition has no limit, and a tight guess
# only buys the chance to kill your own job mid-flight.
#
#SBATCH -J delens_cost
#SBATCH -p compute
#SBATCH -w antares
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH -o delens_cost_%j.out
#SBATCH -e delens_cost_%j.err

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1

# Pin the thread count to the allocation: OMP_NUM_THREADS caps ducc0/healpy and
# OVERRIDES per-call nthreads, and an unpinned value oversubscribes the node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export DUCC0_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# The QE solve here is a single in-process trace; no inner pool.
export AUGR_DELENS_WORKERS=1

cd "$HOME/cmb-augr"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "commit=$(git rev-parse --short HEAD)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB"
free -g | head -2
echo

# Two legs, ACCEPTANCE FIRST. The control leg peaks at 91.5 GB against this job's
# 100G cgroup cap (92%), and peak RSS is polled, so it can be cgroup-killed at a
# peak the previous run merely failed to observe. Under `set -e` that kill would
# abort the job before the acceptance leg -- the whole point of the exercise --
# ever ran. So the deliverable goes first, and the control is non-fatal.

# The acceptance run. Expect peakRSS roughly flat, gradtemp linear in l_max_qe
# (~0.033 MB per unit L, extrapolating to ~134 MB at 4000), grad/s about 2x the
# --no-remat figure at the same L, and the gradtemp exponent dropping from ~1.99
# to ~1. l_max_qe=4000 is the point of the exercise: it is where N_0^MV
# saturation stops under-crediting small-beam / large-aperture designs.
pixi run python scripts/delens_cost_profile.py \
    --remat \
    --l-max-qe 500 1000 1500 2000 3000 4000 \
    --n-iter 5 \
    --repeat 3 \
    --out "delens_cost_remat_${SLURM_JOB_ID:-local}.json"

echo
echo "=========== control leg (--no-remat, non-fatal) ==========="
echo

# The control: it must reproduce the pre-fix numbers (23.9 GB / 91.5 GB peak RSS,
# 20.7 s / 68.2 s grad at 500 / 1000) or the harness changed under us and the
# comparison is not interpretable. Capped at 1000 deliberately -- 1500 OOMs a
# 124 GB node with a 217 GB allocation. `|| echo` because a cgroup kill here is a
# plausible outcome that must not discard the acceptance results above.
pixi run python scripts/delens_cost_profile.py \
    --no-remat \
    --l-max-qe 500 1000 \
    --n-iter 5 \
    --repeat 3 \
    --out "delens_cost_noremat_${SLURM_JOB_ID:-local}.json" \
  || echo "CONTROL LEG FAILED (exit $?) -- acceptance results above still stand"

echo
echo "--- cgroup peak ---"
cat /sys/fs/cgroup/memory.peak 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null \
  || echo "(cgroup peak unavailable)"
