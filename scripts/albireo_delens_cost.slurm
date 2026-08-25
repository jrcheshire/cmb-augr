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

pixi run python scripts/delens_cost_profile.py \
    --l-max-qe 500 1000 1500 2000 \
    --n-iter 5 \
    --repeat 3 \
    --out "delens_cost_${SLURM_JOB_ID:-local}.json"

echo
echo "--- cgroup peak ---"
cat /sys/fs/cgroup/memory.peak 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null \
  || echo "(cgroup peak unavailable)"
