#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={TASKS_PER_NODE}
#SBATCH --gpus-per-node={GPUS_PER_NODE}
#SBATCH --time=02:00:00
#SBATCH --output={JOB_NAME}_%j.out
#SBATCH -A pilotgpu

# See config/*yaml and submit_nemo*.py files
{MODULE_LOADS}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS

srun bash -c ' 
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID

  apptainer exec --nv \
    -B {DATA_MOUNT}:/workspace/data \
    -B {OUTPUT_MOUNT}:/workspace/output \
    # -B $(pwd):/workspace/output \
    {CONTAINER} bash -c '\''
    uv sync
    source ../../.venv/bin/activate
    python {TRAIN_SCRIPT} \
      --config-path {CONFIG_PATH} \
      --config-name {CONFIG_NAME} \
      {YAML_OVERRIDES}
    '\''
'
