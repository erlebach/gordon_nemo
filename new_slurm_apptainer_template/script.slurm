#!/bin/bash

#SBATCH --job-name=apptainer_python_job
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00

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
SCRATCH_DIR="/scratch/${USER}/${SLURM_JOB_ID}"

# --- 3. Set up scratch directory ---
mkdir -p "$SCRATCH_DIR"
echo "Job will run in scratch directory: $SCRATCH_DIR"

# --- 4. Load modules for host-side tools (like nvidia-smi) ---
module load cuda/12.1

# --- 5. Start GPU monitoring in the background ---
echo "Starting GPU monitoring... log will be at $SCRATCH_DIR/gpu_usage.log"
(
    while true; do
        echo "--- $(date) ---"
        nvidia-smi
        sleep 15
    done
) > "$SCRATCH_DIR/gpu_usage.log" 2>&1 &
GPU_LOGGER_PID=$!

# --- 6. Log job details ---
echo "===== Job Details ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "SLURM Submit Directory: $SUBMIT_DIR"
echo "Python Script: $PYTHON_SCRIPT"
echo "Apptainer Image: $CONTAINER_IMG"
echo "======================="

# --- 7. Run the Python script inside the Apptainer container ---
echo "Launching Apptainer container..."

apptainer exec --nv \
    -B "$SUBMIT_DIR":/app \
    -B "$SCRATCH_DIR":/scratch \
    --pwd /app \
    "$CONTAINER_IMG" \
    bash -c "
        set -e
        echo '--- Inside container ---'
        echo 'Step 1: Synchronizing Python environment with uv...'
        uv sync --quiet
        
        echo 'Step 2: Activating virtual environment...'
        source .venv/bin/activate

        echo 'Step 3: Changing to scratch directory...'
        cd /scratch

        echo 'Step 4: Executing Python script...'
        python /app/$PYTHON_SCRIPT
        echo '--- Python script finished ---'
    "

# --- 8. Stop GPU monitoring ---
echo "Stopping GPU monitoring (PID: $GPU_LOGGER_PID)..."
kill "$GPU_LOGGER_PID"

# --- 9. Final job summary ---
echo "===== Job Ended at $(date) ====="
echo "Final job status:"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS

echo "===== GPU Usage Summary (first 20 lines) ====="
head -n 20 "$SCRATCH_DIR/gpu_usage.log"
