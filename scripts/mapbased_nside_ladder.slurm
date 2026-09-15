#!/bin/bash
# High-nside timing of the map-based design gradient, one program per process.
# Generic: account, partition and time go on the command line, so this file
# carries no site policy.
#
#   MB_NSIDE=2048 MB_LEGS="2:40:90 2:80:180 1:40:180" \
#   sbatch -A <alloc> -p <gpu queue> -t 08:00:00 \
#          scripts/mapbased_nside_ladder.slurm [extra nside-ladder args]
#
# MB_LEGS is a list of sim_batch:n_sims:minutes. Each leg is its own process
# with its own time budget, for two measured reasons (job 997868, nside=2048,
# 4x GB200, sim_batch=2):
#   * One value+grad nearly fills a card -- a ~134 GB temporary block plus the
#     full sim ensemble, which every device receives. A second program in the
#     same process then failed on GPU 0, whose pool the first had fragmented.
#   * An out-of-memory on one GPU leaves the others waiting in a collective
#     forever; that job hung ~3 h until the wall clock. `timeout` ends the leg.
# n_sims should be a multiple of devices x sim_batch, or padded slots cost
# full sims. Per-sim slope = difference of two legs' steady times at the same
# sim_batch, read from their JSONs.
#
# Uses the `gpu` pixi environment (jax[cuda12] + jaxht, no pysm3): CMB-only.
#SBATCH -J mb_ladder
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o mb_ladder.%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.pixi/bin:$PATH"
# JAX's default pool is 75% of the card; the 80-sim leg needs ~146 GB of 189.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

# sbatch spools a COPY of this script, so $0's directory is not the repo.
cd "${AUGR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -f pyproject.toml || { echo "not a cmb-augr checkout: $PWD" >&2; exit 2; }

echo "host      $(hostname)"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "partition ${SLURM_JOB_PARTITION:-unknown}"
echo "mem frac  $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "started   $(date -Is)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
pixi install --locked -e gpu || { echo "gpu env install failed" >&2; exit 2; }
pixi run --frozen -e gpu python -c "import jht; print('jht', jht.__version__)"

nside="${MB_NSIDE:-2048}"
tag_base="mb_ladder.${SLURM_JOB_ID:-local}"
for leg in ${MB_LEGS:-2:40:90 2:80:180 1:40:180}; do
    IFS=: read -r b n minutes <<< "$leg"
    tag="${tag_base}.n${nside}_b${b}_s${n}"
    echo; echo "########## leg sim_batch=$b n_sims=$n budget=${minutes}m $(date -Is)"
    timeout --kill-after=60 "${minutes}m" \
        pixi run --frozen -e gpu python scripts/mapbased_grad_characterization.py \
        --mode nside-ladder --backend jht --fft-mode looped \
        --nside-ladder "$nside" --lmax-factor 1.5 --ell-max-like 300 --repeat 1 \
        --sim-batch "$b" --ladder-n-sims "$n" --out-prefix "$tag" "$@"
    rc=$?
    [ "$rc" -eq 124 ] && echo "leg TIMED OUT after ${minutes}m"
    echo "leg exit $rc $(date -Is)"
done

echo
echo "finished  $(date -Is)"
# A missing file must not fail the job after every leg has printed.
exit 0
