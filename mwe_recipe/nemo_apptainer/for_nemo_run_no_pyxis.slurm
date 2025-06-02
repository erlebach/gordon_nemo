#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "finetune_lora_experiment"
#SBATCH -t 00:30:00
#SBATCH -A pilotgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --output=nemo_experiment_%j.out

set -euo pipefail

echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Define container and host paths
# SIF_IMAGE="$HOME/containers/cuda_nvidia_12.sif"
SIF_IMAGE="$HOME/containers/cuda_uv_12.sif
PROJECT_DIR="$HOME/src/gordon_nemo"                      # contains pyproject.toml, configs/
EXPERIMENT_CONFIG="configs/finetune_lora.yaml"
EXPERIMENT_OUTPUT="$HOME/nemo_experiments/job_${SLURM_JOB_ID}"

# Paths to cuDNN (bind individually to avoid /lib64 override)
HOST_CUDNN_LIBS=(
  "/lib64/libcudnn.so.9"
  "/lib64/libcudnn_ops.so.9"
)
BIND_FLAGS=""
for lib in "${HOST_CUDNN_LIBS[@]}"; do
  [[ -e "$lib" ]] && BIND_FLAGS+=" --bind $lib:$lib"
done

# Bind project directory and output path
BIND_FLAGS+=" --bind $PROJECT_DIR:/workspace"
BIND_FLAGS+=" --bind $EXPERIMENT_OUTPUT:/workspace/output"

# Create output directory for experiment
mkdir -p "$EXPERIMENT_OUTPUT"

# Run experiment using nemo-run inside the container
apptainer exec --nv $BIND_FLAGS "$SIF_IMAGE" bash -c '
  set -e
  cd /workspace
  uv venv
  source .venv/bin/activate
  uv sync
  nemo-run --config='"$EXPERIMENT_CONFIG"' --output_dir=/workspace/output
'
