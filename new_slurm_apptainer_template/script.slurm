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
SCRATCH_DIR="/tmp/scratch/gerlebacher/${SLURM_JOB_ID}"

# --- 3. Set up scratch directory ---
echo "Creating scratch directory: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

# Verify scratch directory was created successfully
if [ ! -d "$SCRATCH_DIR" ]; then
    echo "ERROR: Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi

# Test write permissions
if ! touch "$SCRATCH_DIR/test_file" 2>/dev/null; then
    echo "ERROR: Cannot write to scratch directory: $SCRATCH_DIR"
    exit 1
fi
rm -f "$SCRATCH_DIR/test_file"

echo "Scratch directory created successfully: $SCRATCH_DIR"

# --- 4. Load modules for host-side tools ---
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

echo "GPU monitoring started with PID: $GPU_LOGGER_PID"

# --- 6. Log job details ---
echo "===== Job Details ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "SLURM Submit Directory: $SUBMIT_DIR"
echo "Python Script: $PYTHON_SCRIPT"
echo "Apptainer Image: $CONTAINER_IMG"
echo "Scratch Directory: $SCRATCH_DIR"
echo "======================="

# --- 7. Run the Python script inside the Apptainer container ---
echo "Launching Apptainer container..."

apptainer exec --nv \
    -B "$SUBMIT_DIR":/app \
    -B "$SCRATCH_DIR":/scratch \
    --pwd /scratch \
    "$CONTAINER_IMG" \
    bash -c "
        set -e
        echo '--- Inside container ---'
        echo 'Working directory:' \$(pwd)
        echo 'Container scratch directory contents:'
        ls -la /scratch
        echo 'Submit directory contents:'
        ls -la /app
        
        echo 'Step 1: Copy project files to scratch (avoiding pyproject.toml conflicts)...'
        # Copy the Python script
        cp /app/$PYTHON_SCRIPT .
        
        # Copy pyproject.toml if it exists in the submit directory
        if [ -f /app/pyproject.toml ]; then
            cp /app/pyproject.toml .
            echo 'Copied pyproject.toml to scratch directory'
        else
            echo 'No pyproject.toml found in submit directory'
        fi
        
        # Copy uv.lock if it exists
        if [ -f /app/uv.lock ]; then
            cp /app/uv.lock .
            echo 'Copied uv.lock to scratch directory'
        fi
        
        # Copy any other necessary files (add as needed)
        # cp /app/requirements.txt . 2>/dev/null || true
        
        echo 'Scratch directory contents after copying:'
        ls -la
        
        echo 'Step 2: Synchronizing Python environment with uv...'
        export PYTHONUNBUFFERED=1
        uv sync --quiet
        
        echo 'Step 3: Activating virtual environment...'
        source .venv/bin/activate
        
        echo 'Step 4: Executing Python script...'
        echo \"About to run: python -u $PYTHON_SCRIPT\"
        python -u $PYTHON_SCRIPT
        PYTHON_EXIT_CODE=\$?
        echo \"Python script finished with exit code: \$PYTHON_EXIT_CODE\"
        echo '--- Python script finished ---'
    "

CONTAINER_EXIT_CODE=$?
echo "Container exited with code: $CONTAINER_EXIT_CODE"

# --- 8. Stop GPU monitoring ---
echo "Stopping GPU monitoring (PID: $GPU_LOGGER_PID)..."
if kill -0 "$GPU_LOGGER_PID" 2>/dev/null; then
    kill "$GPU_LOGGER_PID"
    echo "GPU monitoring stopped"
else
    echo "GPU monitoring process was not running"
fi

# --- 9. Final job summary ---
echo "===== Job Ended at $(date) ====="
echo "Final job status:"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS

echo "===== GPU Usage Summary (first 20 lines) ====="
if [ -f "$SCRATCH_DIR/gpu_usage.log" ]; then
    head -n 20 "$SCRATCH_DIR/gpu_usage.log"
else
    echo "GPU usage log file not found: $SCRATCH_DIR/gpu_usage.log"
fi

# --- 10. Cleanup scratch directory (optional) ---
echo "Cleaning up scratch directory: $SCRATCH_DIR"
# I only wish to delete scratch if there are no errors. 
#rm -rf "$SCRATCH_DIR"
echo "Cleanup completed"
