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
CONTAINER_IMAGE="$HOME/containers/cuda_uv_git_12.sif"

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Shared environment: $SHARED_VENV"
echo "Container image: $CONTAINER_IMAGE"

module load webproxy

# --- 4. Run the container and create a unique venv in /tmp ---
apptainer exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "${TMPDIR:-/tmp}:${TMPDIR:-/tmp}" \
    "$CONTAINER_IMAGE" \
    bash -c '
        set -x
	TMPDIR="/tmp/gerlebacher"
	mkdir -p "$TMPDIR"
	cp "'"$PROJECT_ROOT"'/pyproject.toml" "$TMPDIR/"
	cd "$TMPDIR"
	uv venv
	source .venv/bin/activate
	uv pip install --upgrade pip

    # Install core dependencies first to avoid platform issues
	uv pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	uv pip install nemo-toolkit>=2.1.0
	uv pip install "lightning>=2.2.1,<=2.4.0"
	uv pip install transformers>=4.51.3
	uv pip install numpy>=2.2.6 jaxtyping>=0.3.2
    uv pip install transformer-engine
    
    # Install apex after torch is available
    uv pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation --nodeps
	# Try to install remaining dependencies, but don'\''t fail on problematic ones
	uv pip install -r <(grep -E "^[[:space:]]*\"[^\"]+>=" pyproject.toml | sed "s/^[[:space:]]*\"//;s/\",$//") || true
    uv pip list
        cd "'"$SLURM_SUBMIT_DIR"'"
	python "'"$PYTHON_SCRIPT"'" "${SCRIPT_ARGS[@]}"
    '
