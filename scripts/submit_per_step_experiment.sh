#!/bin/bash
# Submit one per_step run for each batch size (32, 16, 8) to compare overhead
# against Exp1 (noop) and Exp3 (combined).
#
# Usage:
#   ./scripts/submit_per_step_experiment.sh

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

DEFAULT_CONFIG_FILE="${REPO_DIR}/config/default_sbatch_config.sh"
. "${DEFAULT_CONFIG_FILE}"
if [[ -f "${COMP597_SLURM_CONFIG}" ]]; then
    . "${COMP597_SLURM_CONFIG}"
fi

module load slurm

for BS in 32 16 8; do
    JOB_ID=$(
        sbatch \
            --output="${REPO_DIR}/comp597-regnet-run0-per_step-bs${BS}-%N-%j.log" \
            --partition=${COMP597_SLURM_PARTITION} \
            --mem=${COMP597_SLURM_MIN_MEM} \
            --time=${COMP597_SLURM_TIME_LIMIT} \
            --ntasks=${COMP597_SLURM_NTASKS} \
            --account=${COMP597_SLURM_ACCOUNT} \
            --nodelist=${COMP597_SLURM_NODELIST} \
            --cpus-per-task=${COMP597_SLURM_CPUS_PER_TASK} \
            --qos=${COMP597_SLURM_QOS} \
            --gpus=${COMP597_SLURM_NUM_GPUS} \
            --export=COMP597_SLURM_SCRIPTS_DIR=${COMP597_SLURM_SCRIPTS_DIR},REGNET_RUN_NUM=0,REGNET_BATCH_SIZE=${BS},REGNET_TRAINER_STATS=per_step \
            "${COMP597_SLURM_JOB_SCRIPT}" \
        | awk '{print $NF}'
    )
    echo "bs=${BS} -> SLURM job ${JOB_ID}"
    echo "       log: ${REPO_DIR}/comp597-regnet-run0-per_step-bs${BS}-*-${JOB_ID}.log"
done

echo ""
echo "Monitor with: squeue -u ${USER}"
