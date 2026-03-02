#!/bin/bash
set -euo pipefail

# Required paths

SCRIPTS_DIR=${COMP597_SLURM_SCRIPTS_DIR:-$(readlink -f -n $(dirname $0))}
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

DEFAULT_CONFIG_FILE=${REPO_DIR}/config/default_job_config.sh

# Load dependencies

. ${DEFAULT_CONFIG_FILE}

if [[ -f ${COMP597_JOB_CONFIG} ]]; then
	. ${COMP597_JOB_CONFIG}
fi

. ${SCRIPTS_DIR}/env_table.sh # Basics to print an environment variables table

# Set up the Conda environment

. ${SCRIPTS_DIR}/conda_init.sh
conda activate ${COMP597_JOB_CONDA_ENV_PREFIX}

# Logs

if [[ $COMP597_JOB_CONFIG_LOG = true ]]; then
	env_table "^SLURM_([CG]PU|MEM)" "SLURM Hardware Configuration"
	echo
	echo
	env_table "^SLURM_JOB" "SLURM Job Info"
	echo
	echo
	env_table "^COMP597_JOB_" "SLURM Job Configuration"
	echo
	echo
	env_table "^CONDA" "Conda Environment Variables"
	echo
	echo
fi

# Change the working directory if configured

if [[ -d "${COMP597_JOB_WORKING_DIRECTORY}" ]]; then
	cd ${COMP597_JOB_WORKING_DIRECTORY}
fi


# Optional GPU logging (only when COMP597_LOG_GPU=1 and GPU is allocated)
OUTDIR=/home/slurm/comp597/students/$USER/gpu_measurements
mkdir -p "$OUTDIR"
GPU_LOG="$OUTDIR/gpu_${SLURM_JOB_ID}.csv"

GPU_PID=""
if [[ "${COMP597_LOG_GPU:-0}" == "1" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
	echo "Starting GPU logging -> $GPU_LOG"
	nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm \
  	--format=csv -l 0.1 > "$GPU_LOG" & 
	GPU_PID=$!
fi

cleanup() {
	if [[ -n "${GPU_PID}" ]]; then
		kill "${GPU_PID}" 2>/dev/null || true
		# Delete empty logs (e.g., if job ended too fast)
		if [[ ! -s "$GPU_LOG" ]]; then
			rm -f "$GPU_LOG"
		else
			echo "Stopped GPU logging. Saved: $GPU_LOG"
		fi
	fi
}
trap cleanup EXIT INT TERM

# Run the job (original behavior)
echo "PRECHECK: hostname=$(hostname)"
echo "PRECHECK: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"
python -c "import torch; print('PRECHECK: torch', torch.__version__); print('PRECHECK: cuda_available', torch.cuda.is_available()); print('PRECHECK: device_count', torch.cuda.device_count())"
eval "${COMP597_JOB_COMMAND} $@"
