#!/bin/bash
# albireo/deneb: does the delensing gradient scale with core count, and does it
# stay off the memory bound while doing it?
#
# The production question is Vista's gg node: 144 Grace cores, ~260 GB. At the
# measured 25-30 GB/worker the packing is ~8 workers/node, i.e. 18 cores each --
# so the thing to know is whether the delensing leg uses 18 cores or plateaus
# well before them. Job 706 is pinned at 8 threads throughout and gives one point.
#
# Runs on DENEB, not antares: this needs core RANGE (64 vs 28) and almost no
# memory (~1.3 GB measured at l_max_qe=1000), and antares is busy with 706.
#
# ARCHITECTURE CAVEAT: deneb is x86, Grace is aarch64. Read the SHAPE of the
# curve -- where it plateaus -- not the absolute seconds. JC notes an M4 Max is
# the closer proxy to Grace clock-for-clock; a laptop leg would cover 1-16 cores
# on the right ISA family and is worth pairing with this.
#
#SBATCH -J delens_threads
#SBATCH -p compute
#SBATCH -w deneb
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH -o delens_threads_%j.out
#SBATCH -e delens_threads_%j.err

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export AUGR_DELENS_WORKERS=1     # single in-process trace; no inner pool

cd "$HOME/cmb-augr"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "commit=$(git rev-parse --short HEAD)"
echo "alloc cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB"
echo

L=1000   # fixed: big enough that per-step work dominates dispatch overhead

for N in 1 2 4 8 16 32; do
  echo "================ threads=$N ================"

  export OMP_NUM_THREADS=$N
  export DUCC0_NUM_THREADS=$N
  export OPENBLAS_NUM_THREADS=$N
  export MKL_NUM_THREADS=$N

  # VERIFY THE KNOB IS LIVE. A dead affinity setting and a genuine plateau are
  # indistinguishable in the timings alone, so assert the restriction took
  # before trusting any number from this leg. JAX/XLA's CPU pool sizes itself
  # from sched_getaffinity, which is what taskset restricts.
  taskset -c 0-$((N - 1)) python3 -c "
import os
n = len(os.sched_getaffinity(0))
print(f'  affinity_cpus={n}  expected={$N}', flush=True)
assert n == $N, f'AFFINITY NOT APPLIED: got {n}, wanted {$N}'
"

  taskset -c 0-$((N - 1)) pixi run python scripts/delens_cost_profile.py \
      --remat \
      --l-max-qe $L \
      --n-iter 5 \
      --repeat 2 \
      --out "delens_threads_${N}_${SLURM_JOB_ID:-local}.json"
  echo
done

echo "--- cgroup peak ---"
cat /sys/fs/cgroup/memory.peak 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null \
  || echo "(cgroup peak unavailable)"
