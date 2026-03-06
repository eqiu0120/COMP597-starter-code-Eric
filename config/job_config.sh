
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
REGNET_OUT_DIR="/home/slurm/comp597/students/${USER}/regnet_measurements"

# Run number: override by setting REGNET_RUN_NUM before sourcing this file
# (or by passing --export=ALL,REGNET_RUN_NUM=N to sbatch).
# Defaults to 0 so a plain srun/sbatch still works.
REGNET_RUN_NUM="${REGNET_RUN_NUM:-0}"

# Extend the default launch command with RegNet-specific arguments:
#   --model regnet          : use the RegNet y-128GF model
#   --data  regnet          : load FakeImageNet from the student storage
#   --trainer_stats combined: enable GPU resource + CodeCarbon tracking
#   --trainer_stats_configs.combined.*: pass CodeCarbon output settings
export COMP597_JOB_COMMAND="${COMP597_JOB_COMMAND} \
  --model regnet \
  --data regnet \
  --trainer_stats combined \
  --trainer_stats_configs.combined.run_num ${REGNET_RUN_NUM} \
  --trainer_stats_configs.combined.project_name regnet-energy \
  --trainer_stats_configs.combined.output_dir ${REGNET_OUT_DIR}"
