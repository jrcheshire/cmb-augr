#!/bin/bash
# Wall-clock table for the full-sky per-L map's two parallelism knobs
# (l_batch, CPU device sharding) -- the measurement that decides their
# defaults. Generic: account, partition, time and cores go on the command
# line, so this file carries no site policy.
#
#   sbatch -A <alloc> -p <queue> -t 03:00:00 -c 144 \
#          scripts/bench_fullsky_parallel.slurm [extra bench args]
#
# Estimated ~15 min per child x 7 children on a 144-core node; ask for 3 h.
# Run it from a cmb-augr checkout, or set AUGR=<path to one>.
#
# Do NOT set OMP_NUM_THREADS here: it was measured not to cap XLA's CPU
# thread pool, and pinning it would only mislead the next reader.
#SBATCH -J fullsky_parallel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o fullsky_parallel.%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.pixi/bin:$PATH"

# sbatch spools a COPY of this script, so $0's directory is not the repo.
cd "${AUGR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f pyproject.toml || { echo "not a cmb-augr checkout: $PWD" >&2; exit 2; }

echo "host      $(hostname)"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "nproc     $(nproc)"
echo "affinity  $(python3 -c 'import os;print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo '?')"
echo "slurm     cpus_on_node=${SLURM_CPUS_ON_NODE:-} cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "omp       OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "started   $(date -Is)"
echo

pixi run --frozen -e cpu python scripts/bench_wigner_closed.py \
    --sweep --l-max 1500 3000 \
    --sweep-baseline "bench_sweep_baseline.${SLURM_JOB_ID:-local}.npz" "$@"

echo
echo "finished  $(date -Is)"
# Peak RSS of the whole job step. sacct's MaxRSS is polled and under-reports;
# the cgroup counter is not.
for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
    [ -r "$f" ] && echo "cgroup peak $(awk '{printf "%.2f GB\n", $1/1e9}' "$f")" && break
done
