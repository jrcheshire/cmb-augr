#!/bin/bash
# TACC Vista Grace-Grace SMOKE for the design active-subspace + HL-EIG run: ONE production
# design, ONE worker, so the job measures what the production sizing needs and nothing else:
#
#   * per-worker peak RSS at production shape (nside=128 / lmax=192 / n_sims=30 / d1s1 /
#     delensing credited at l_max_qe=4000). The inherited 25-30 GB/worker figure predates
#     delensing being wired in; N_max ~ 0.85 * 237 GB / RSS sets --workers for production.
#   * the real jit-compile + design-gradient wall time at 1 ducc thread (production workers
#     are pinned to 1 thread each by augr.parallel, so this is the per-worker rate).
#
# The gradient part lands in --parts-dir as grad_0000.npz, indices referring to the
# seed-fixed 96-design LHS, so it is a genuine production part (copy it into the production
# parts dir to skip design 0 there). With --design-range the driver exits 0 after its slice.
#
# Memory is sampled in the background every 30 s (cgroup v2 memory.current) and the cgroup
# memory.peak is printed on exit -- including on a wall-clock SIGTERM -- so a timeout still
# yields the number. /usr/bin/time does not exist on Vista compute nodes; sacct MaxRSS
# under-reports. Batch, not idev: a design gradient is measured in hours.
#
# Prereqs (done 2026-08-26): $WORK/cmb-augr on main with `pixi install --locked -e cpu`
# (cpu env = ducc0 + pysm3 on linux-aarch64; the gpu env omits both), and the 9 PySM d1s1
# template files in ~/.cache/astropy/download/url/ (astropy >= 8 reads ~/.cache/astropy,
# NOT ~/.astropy/cache; copied from the laptop -- compute nodes have no network and pixi
# is not usable on login nodes, so the pre-warm the production launcher describes is not
# runnable on Vista).
#
#SBATCH -J augr_smoke1
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A JPL-PUB
#SBATCH -o smoke1_%j.out
#SBATCH -e smoke1_%j.err

set -uo pipefail
export PYTHONUNBUFFERED=1

REPO="$WORK/cmb-augr"
PIXI="$HOME/.pixi/bin/pixi"
OUT="$SCRATCH/augr_smoke"
PARTS="$SCRATCH/augr_smoke_parts"
MEMLOG="$REPO/smoke1_${SLURM_JOB_ID}_mem.log"
L_MAX_QE=4000   # JC 2026-08-26: >=3000, target 4000; the driver default (1000) under-credits
                # delensing for small-beam designs. Pass it explicitly until the default moves.

# One worker => the driver runs serially in this process. Pin the numerics to ONE thread so
# the timing is the production per-worker rate (augr.parallel pins workers to 1 thread), and
# so the full-sky delensing pool, if anything reaches it, cannot fan out to 144 processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DUCC0_NUM_THREADS=1 AUGR_SHT_NTHREADS=1 AUGR_DELENS_WORKERS=1

cd "$REPO"
mkdir -p "$PARTS"

CG="/sys/fs/cgroup$(grep '^0::' /proc/self/cgroup | cut -d: -f3)"
echo "job $SLURM_JOB_ID on $(hostname)  cgroup=$CG  start=$(date -Is)"
echo "repo HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"

report() {
    echo "--- memory report $(date -Is)"
    echo "cgroup memory.peak    = $(cat "$CG/memory.peak" 2>/dev/null || echo n/a) bytes"
    echo "cgroup memory.current = $(cat "$CG/memory.current" 2>/dev/null || echo n/a) bytes"
    if [ -s "$MEMLOG" ]; then
        awk 'NR>1 && $2>m {m=$2} END {printf "sampler max memory.current = %.2f GB over %d samples\n", m/1e9, NR-1}' "$MEMLOG"
    fi
}
trap 'report' EXIT
trap 'echo "SIGTERM (wall clock?) at $(date -Is)"; kill $SAMPLER 2>/dev/null; kill $DRIVER 2>/dev/null; wait $DRIVER 2>/dev/null; exit 143' TERM

# Background sampler: epoch  memory.current(bytes)
( echo "t_epoch memory_current_bytes" > "$MEMLOG"
  while true; do echo "$(date +%s) $(cat "$CG/memory.current")" >> "$MEMLOG"; sleep 30; done ) &
SAMPLER=$!

# Production shape, ONE design, ONE worker. --frozen: use the installed env as-is, no lock
# check, no network.
"$PIXI" run --frozen -e cpu python scripts/active_subspace_hl_eig.py \
    --backend ducc --fft-mode looped --nside 128 --lmax 192 \
    --n-sims 30 --n-designs 96 --n-crn 1 \
    --workers 1 \
    --scan-points 11 --n-outer 1024 \
    --fg-model d1s1 \
    --delens-l-max-qe "$L_MAX_QE" \
    --compile-cache "$SCRATCH/xla_cache_cpu" \
    --design-range 0 1 \
    --parts-dir "$PARTS" \
    --out "$OUT" &
DRIVER=$!
wait $DRIVER
RC=$?
kill $SAMPLER 2>/dev/null
echo "driver exit code $RC at $(date -Is)"
ls -la "$PARTS"
exit $RC
