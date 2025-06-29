#!/bin/bash -x
#SBATCH --job-name=apptainer_python_job
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH -A pilotgpu

# --- 1. Get Python script from command-line argument ---
if [ -z "$1" ]; then
    echo "Error: No Python script provided."
    echo "Usage: sbatch $0 your_script.py"
    exit 1
fi
PYTHON_SCRIPT=$1

# --- 2. Set up cache directories in scratch space ---
SCRATCH_CACHE="${TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"
export UV_CACHE_DIR="$SCRATCH_CACHE/uv"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

# --- 3. Set up paths ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_VENV="$PROJECT_ROOT/.venv"
CONTAINER_IMAGE="$PROJECT_ROOT/new_slurm_apptainer_template/enhanced_nemo_container.sif"

# --- 4. Run the container with shared environment ---
echo "Using shared environment: $SHARED_VENV"
echo "Running: $PYTHON_SCRIPT"

apptainer exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$SHARED_VENV:$SHARED_VENV" \
    --bind "${TMPDIR:-/tmp}:${TMPDIR:-/tmp}" \
    "$CONTAINER_IMAGE" \
    bash -c "source $SHARED_VENV/bin/activate && python $PYTHON_SCRIPT"
