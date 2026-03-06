
# This if-statement is not required, it is used to make sure no one accidently
# tries to execute the config file.
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# -----------------------------------------------------------------------
# RegNet experiment SLURM configuration
# Override defaults from default_slurm_config.sh as needed.
# -----------------------------------------------------------------------

# Run on the GPU teaching node
export COMP597_SLURM_NODELIST="gpu-teach-03"

# RegNet y-128GF is large; request enough RAM
export COMP597_SLURM_MIN_MEM="24GB"

# Allow enough wall-clock time for ~250 steps over 2000 samples
export COMP597_SLURM_TIME_LIMIT="30:00"

# One GPU is sufficient for a single-process run
export COMP597_SLURM_NUM_GPUS=1

# 4 CPU workers match the DataLoader num_workers in regnet/model.py
export COMP597_SLURM_CPUS_PER_TASK=4
