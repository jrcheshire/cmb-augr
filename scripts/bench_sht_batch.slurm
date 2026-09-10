#!/bin/bash
# GPU bring-up + the batched-vs-looped SHT curve that sizes the batch knob.
# Generic: account, partition and time go on the command line, so this file
# carries no site policy.
#
#   sbatch -A <alloc> -p <gpu queue> -t 00:40:00 \
#          scripts/bench_sht_batch.slurm [extra bench args]
#
# Uses the `gpu` pixi environment (jax[cuda12] + jaxht, no ducc0/pysm3), so it
# needs a GPU node. Run it from a cmb-augr checkout, or set AUGR=<path to one>.
#
# The bring-up half is the load-bearing part on a new card: a jaxlib without
# kernels for this compute capability fails here, in 40 minutes, rather than
# inside a production run.
#SBATCH -J sht_batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o sht_batch.%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.pixi/bin:$PATH"

# sbatch spools a COPY of this script, so $0's directory is not the repo.
cd "${AUGR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f pyproject.toml || { echo "not a cmb-augr checkout: $PWD" >&2; exit 2; }

echo "host      $(hostname)"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "partition ${SLURM_JOB_PARTITION:-unknown}"
echo "started   $(date -Is)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
               --format=csv,noheader || true
else
    echo "nvidia-smi absent"
fi
echo

pixi run --frozen -e gpu python scripts/bench_sht_batch.py "$@"

echo
echo "finished  $(date -Is)"
for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
    if [ -r "$f" ]; then
        echo "cgroup peak $(awk '{printf "%.2f GB\n", $1/1e9}' "$f")"
        break
    fi
done
# Neither counter exists on every site, and under `set -e` a failing final test
# would fail the whole job after the table is already printed.
exit 0
