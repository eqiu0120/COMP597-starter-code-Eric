
# This if-statement is not required, it is used to make sure no one accidently
# tries to execute the config file.
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# -----------------------------------------------------------------------
# RegNet experiment job configuration
# This is sourced by scripts/job.sh after default_job_config.sh, so
# COMP597_JOB_COMMAND is already set to scripts/launch.sh here.
# -----------------------------------------------------------------------

# Enable nvidia-smi GPU logging in job.sh
export COMP597_LOG_GPU=1

# Results directory on the SLURM storage partition
REGNET_OUT_DIR="/home/slurm/comp597/students/zqiu6/regnet_measurements"

# Run number: override by setting REGNET_RUN_NUM before sourcing this file
# (or by passing --export=ALL,REGNET_RUN_NUM=N to sbatch).
# Defaults to 0 so a plain srun/sbatch still works.
REGNET_RUN_NUM="${REGNET_RUN_NUM:-0}"

# Extend the default launch command with RegNet-specific arguments:
#   --model regnet          : use the RegNet y-128GF model
#   --data  regnet          : load FakeImageNet from the student storage
#   --trainer_stats combined: enable GPU resource + CodeCarbon tracking
#   --trainer_stats_configs.combined.*: pass CodeCarbon output settings
REGNET_BATCH_SIZE="${REGNET_BATCH_SIZE:-8}"
REGNET_TRAINER_STATS="${REGNET_TRAINER_STATS:-combined}"

# Output dir encodes both batch size and experiment type (combined keeps legacy name)
if [[ "${REGNET_TRAINER_STATS}" == "combined" ]]; then
    REGNET_OUT_DIR="/home/slurm/comp597/students/zqiu6/regnet_measurements/bs_${REGNET_BATCH_SIZE}"
else
    REGNET_OUT_DIR="/home/slurm/comp597/students/zqiu6/regnet_measurements/bs_${REGNET_BATCH_SIZE}_${REGNET_TRAINER_STATS}"
fi

# Base command (same for all experiments)
REGNET_BASE_CMD="--model regnet \
  --data regnet \
  --model_configs.regnet.batch_size ${REGNET_BATCH_SIZE} \
  --model_configs.regnet.duration_seconds 300 \
  --trainer_stats ${REGNET_TRAINER_STATS}"

# Stats-specific config args
if [[ "${REGNET_TRAINER_STATS}" == "combined" ]]; then
    REGNET_STATS_CMD="--trainer_stats_configs.combined.run_num ${REGNET_RUN_NUM} \
  --trainer_stats_configs.combined.project_name regnet-energy \
  --trainer_stats_configs.combined.output_dir ${REGNET_OUT_DIR}"
elif [[ "${REGNET_TRAINER_STATS}" == "codecarbon_full" ]]; then
    REGNET_STATS_CMD="--trainer_stats_configs.codecarbon_full.run_num ${REGNET_RUN_NUM} \
  --trainer_stats_configs.codecarbon_full.project_name regnet-energy \
  --trainer_stats_configs.codecarbon_full.output_dir ${REGNET_OUT_DIR}"
else
    # noop and others: no stats config args needed
    REGNET_STATS_CMD=""
fi

export COMP597_JOB_COMMAND="${COMP597_JOB_COMMAND} \
  ${REGNET_BASE_CMD} \
  ${REGNET_STATS_CMD}"
