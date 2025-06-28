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

# --- 2. Define paths and environment variables ---
CONTAINER_IMG="$HOME/containers/cuda_uv_12.sif"
SUBMIT_DIR="$SLURM_SUBMIT_DIR"

# --- 3. Load modules for host-side tools (like nvidia-smi) ---
module load cuda/12.1

# --- 4. Log job details ---
echo "===== Job Details ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "SLURM Submit Directory: $SUBMIT_DIR"
echo "Python Script: $PYTHON_SCRIPT"
echo "Apptainer Image: $CONTAINER_IMG"
echo "======================="

# --- 5. Run the Python script inside the Apptainer container ---
echo "Launching Apptainer container..."

apptainer exec --nv \
    -B "$SUBMIT_DIR":/app \
    --pwd /app \
    "$CONTAINER_IMG" \
    bash -c "
        set -e
        echo '--- Inside container ---'
        echo 'Working directory:' \$(pwd)
        echo 'Available files:'
        ls -la
        
        echo 'Step 1: Synchronizing Python environment with uv...'
        uv sync --quiet
        
        echo 'Step 2: Activating virtual environment...'
        source .venv/bin/activate

        echo 'Step 3: Executing Python script...'
        export PYTHONUNBUFFERED=1
        python -u $PYTHON_SCRIPT
        echo '--- Python script finished ---'
    "

CONTAINER_EXIT_CODE=$?
echo "Container exited with code: $CONTAINER_EXIT_CODE"

# --- 6. Final job summary ---
echo "===== Job Ended at $(date) ====="
echo "Final job status:"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS
