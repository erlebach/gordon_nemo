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

# --- 3. Set up paths using SLURM_SUBMIT_DIR ---
PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"  # Parent of submission directory
CONTAINER_IMAGE="$HOME/containers/cuda_uv_12.sif"

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Shared environment: $SHARED_VENV"
echo "Container image: $CONTAINER_IMAGE"


# --- 4. Run the container and create a unique venv in /tmp ---
apptainer exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "${TMPDIR:-/tmp}:${TMPDIR:-/tmp}" \
    "$CONTAINER_IMAGE" \
    bash -c "
        set -x
		TMPDIR="/tmp/uvjob_${SLURM_JOB_ID:-$USER}"
		mkdir -p "$TMPDIR"
		cp "$PROJECT_ROOT/../pyproject.toml" "$TMPDIR/"
		cd "$TMPDIR"
		uv venv
		source .venv/bin/activate
		uv pip install --upgrade pip
		uv sync
        cd "$SLURM_SUBMIT_DIR"
		python "$PYTHON_SCRIPT" "${SCRIPT_ARGS[@]}"
    "

