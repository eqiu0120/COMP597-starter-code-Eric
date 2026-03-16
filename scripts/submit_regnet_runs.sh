#!/bin/bash
# submit_regnet_runs.sh
# ---------------------
# Submit N independent RegNet training jobs to SLURM, each with a different
# run number so their CodeCarbon output files do not overwrite each other.
# Results from all runs can then be averaged with plot_measurements.py.
#
# Usage:
#   ./scripts/submit_regnet_runs.sh [NUM_RUNS] [BATCH_SIZE]
#
# Default: 3 runs, batch size 8.
#   ./scripts/submit_regnet_runs.sh 3 32

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")
NUM_RUNS="${1:-3}"
BATCH_SIZE="${2:-8}"

# Load the same SLURM configuration that scripts/sbatch.sh uses so we get
# the correct partition, account, QOS, memory, time limit, etc.
DEFAULT_CONFIG_FILE="${REPO_DIR}/config/default_sbatch_config.sh"
. "${DEFAULT_CONFIG_FILE}"
if [[ -f "${COMP597_SLURM_CONFIG}" ]]; then
    . "${COMP597_SLURM_CONFIG}"
fi

module load slurm

echo "Submitting ${NUM_RUNS} RegNet run(s) [batch_size=${BATCH_SIZE}] to partition=${COMP597_SLURM_PARTITION} account=${COMP597_SLURM_ACCOUNT}..."
echo ""

for RUN in $(seq 0 $((NUM_RUNS - 1))); do
    JOB_ID=$(
        sbatch \
            --output="${REPO_DIR}/comp597-regnet-run${RUN}-%N-%j.log" \
            --partition=${COMP597_SLURM_PARTITION} \
            --mem=${COMP597_SLURM_MIN_MEM} \
            --time=${COMP597_SLURM_TIME_LIMIT} \
            --ntasks=${COMP597_SLURM_NTASKS} \
            --account=${COMP597_SLURM_ACCOUNT} \
            --nodelist=${COMP597_SLURM_NODELIST} \
            --cpus-per-task=${COMP597_SLURM_CPUS_PER_TASK} \
            --qos=${COMP597_SLURM_QOS} \
            --gpus=${COMP597_SLURM_NUM_GPUS} \
            --export=COMP597_SLURM_SCRIPTS_DIR=${COMP597_SLURM_SCRIPTS_DIR},REGNET_RUN_NUM=${RUN},REGNET_BATCH_SIZE=${BATCH_SIZE} \
            "${COMP597_SLURM_JOB_SCRIPT}" \
        | awk '{print $NF}'
    )
    echo "  Run ${RUN} -> SLURM job ${JOB_ID}"
    echo "           log : ${REPO_DIR}/comp597-regnet-run${RUN}-*-${JOB_ID}.log"

    if [[ ${RUN} -lt $((NUM_RUNS - 1)) ]]; then
        echo "           waiting for job ${JOB_ID} to complete before submitting run $((RUN + 1))..."
        while squeue -j "${JOB_ID}" -h 2>/dev/null | grep -q "${JOB_ID}"; do
            sleep 30
        done
        echo "           job ${JOB_ID} finished."
        echo ""
    fi
done

echo ""
echo "Monitor with:  squeue -u ${USER}"
echo ""
echo "Once all jobs finish, generate averaged plots with:"
echo "  python GPU_result/plot_measurements.py \\"
echo "    --cc_dir   /home/slurm/comp597/students/${USER}/regnet_measurements \\"
echo "    --out_dir  ~/COMP597-starter-code-Eric/GPU_result/plots \\"
echo "    --num_runs ${NUM_RUNS} \\"
echo "    --log_files \$(ls comp597-regnet-run*-*.log 2>/dev/null | sort)"
