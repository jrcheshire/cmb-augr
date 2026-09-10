#!/bin/bash
# Phase profile of the map-based design gradient: where the time and the memory
# go, and whether the GPU forward is launch-bound or arithmetic-bound.
# Generic: account, partition and time go on the command line, so this file
# carries no site policy.
#
#   sbatch -A <alloc> -p <gpu queue> -t 01:00:00 \
#          scripts/mapbased_profile.slurm [extra profile args]
#
# Uses the `gpu` pixi environment (jax[cuda12] + jaxht, no ducc0/pysm3), so it
# needs a GPU node -- and, because that env has no pysm3, a CMB-only config or a
# pre-built sky cache. The defaults below are CMB-only.
#
# The trace leg is the point of running this on a GPU at all: on CPU there is no
# device stream to form a gap histogram from, since augr's ducc transforms are
# pure_callbacks and never appear as device kernels.
#SBATCH -J mb_profile
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o mb_profile.%j.out

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

# nside/lmax and the two n_sims rungs are the knobs worth changing per run; the
# rungs must straddle a real difference or the intercept is unresolvable.
pixi run --frozen -e gpu python scripts/mapbased_grad_characterization.py \
    --mode profile --backend jht \
    --nside "${MB_NSIDE:-128}" --lmax "${MB_LMAX:-192}" \
    --profile-n-sims ${MB_NSIMS:-8 32} \
    --repeat "${MB_REPEAT:-3}" \
    --trace-dir "mb_profile_trace.${SLURM_JOB_ID:-local}" \
    --out-prefix "mb_profile.${SLURM_JOB_ID:-local}" "$@"

echo
echo "finished  $(date -Is)"
for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
    if [ -r "$f" ]; then
        echo "cgroup peak $(awk '{printf "%.2f GB\n", $1/1e9}' "$f")"
        break
    fi
done
# Neither counter exists on every site, and under `set -e` a failing final test
# would fail the whole job after the profile is already printed.
exit 0
