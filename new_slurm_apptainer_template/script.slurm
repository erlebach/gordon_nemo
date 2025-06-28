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

# Set cache environment variables - let apps create the directories automatically
export UV_CACHE_DIR="$SCRATCH_CACHE/uv"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

echo "Cache base directory: $SCRATCH_CACHE"
echo "UV cache: $UV_CACHE_DIR"
echo "PIP cache: $PIP_CACHE_DIR"
echo "HuggingFace cache: $HF_HOME"

# --- 3. Define paths and environment variables ---
CONTAINER_IMG="$HOME/containers/cuda_uv_12.sif"
SUBMIT_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_DIR="/tmp/scratch/gerlebacher/${SLURM_JOB_ID}"

# --- 4. Set up scratch directory ---
echo "Creating scratch directory: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

if [ ! -d "$SCRATCH_DIR" ]; then
    echo "ERROR: Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi

echo "Scratch directory created successfully: $SCRATCH_DIR"

# --- 5. Load modules ---
module load cuda/12.1

# --- 6. Start GPU monitoring in the background ---
echo "Starting GPU monitoring... log will be at $SCRATCH_DIR/gpu_usage.log"
(
    while true; do
        echo "--- $(date) ---"
        nvidia-smi
        sleep 15
    done
) > "$SCRATCH_DIR/gpu_usage.log" 2>&1 &
GPU_LOGGER_PID=$!

# --- 7. Log job details ---
echo "===== Job Details ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "SLURM Submit Directory: $SUBMIT_DIR"
echo "Python Script: $PYTHON_SCRIPT"
echo "Apptainer Image: $CONTAINER_IMG"
echo "Scratch Directory: $SCRATCH_DIR"
echo "======================="

# --- 8. Run the Python script inside the Apptainer container ---
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
        
        # Pass cache environment variables to container
        export UV_CACHE_DIR='$UV_CACHE_DIR'
        export PIP_CACHE_DIR='$PIP_CACHE_DIR'
        export HF_HOME='$HF_HOME'
        export XDG_CACHE_HOME='$XDG_CACHE_HOME'
        
        echo 'Cache configuration:'
        echo \"UV cache: \$UV_CACHE_DIR\"
        echo \"PIP cache: \$PIP_CACHE_DIR\"
        echo \"HF cache: \$HF_HOME\"
        
        echo 'Step 1: Copy project files to scratch...'
        cp /app/$PYTHON_SCRIPT .
        
        echo 'Step 2: Copy pre-built virtual environment...'
        if [ -d /app/.venv ]; then
            cp -r /app/.venv .
            echo 'Virtual environment copied successfully'
        else
            echo 'ERROR: No .venv found in submit directory!'
            echo 'Please run prepare_environment_host.sh on the frontend first.'
            exit 1
        fi
        
        echo 'Step 3: Activating virtual environment...'
        source .venv/bin/activate
        
        echo 'Step 4: Installing additional packages if needed...'
        
        # Check if we need to install NeMo toolkit (if not in base environment)
        if ! python -c 'import nemo' 2>/dev/null; then
            echo 'Installing NeMo toolkit...'
            pip install nemo-toolkit --no-warn-script-location || echo 'Warning: NeMo installation failed'
        fi
        
        # Check if local CUDA packages are available
        if [ -d /app/cuda_packages ]; then
            echo 'Found local CUDA packages, copying to scratch...'
            cp -r /app/cuda_packages .
            
            echo 'Installing apex from local source...'
            cd cuda_packages/apex
            pip install -v --disable-pip-version-check --no-build-isolation \
                --config-settings \"--build-option=--cpp_ext\" \
                --config-settings \"--build-option=--cuda_ext\" ./ || echo 'Warning: apex installation failed'
            cd ../..
            
            echo 'Installing TransformerEngine from local source...'
            cd cuda_packages/TransformerEngine
            pip install -v --disable-pip-version-check --no-build-isolation ./ || echo 'Warning: TransformerEngine installation failed'
            cd ../..
            
        else
            echo 'No local CUDA packages found, trying online installation...'
            pip install apex --no-build-isolation || echo 'Warning: apex installation failed'
            pip install git+https://github.com/NVIDIA/TransformerEngine.git || echo 'Warning: TransformerEngine installation failed'
        fi
        
        echo 'Step 5: Verify complete environment...'
        python -c 'import sys; print(f\"Python: {sys.version}\")'
        python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")' || echo 'PyTorch not available'
        python -c 'import nemo; print(f\"NeMo: {nemo.__version__}\")' || echo 'NeMo not available'
        python -c 'import apex; print(f\"Apex available\")' || echo 'Apex not available'
        python -c 'import transformer_engine; print(f\"TransformerEngine available\")' || echo 'TransformerEngine not available'
        
        echo 'Step 6: Show cache usage...'
        echo \"UV cache size: \$(du -sh \$UV_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')\"
        echo \"PIP cache size: \$(du -sh \$PIP_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')\"
        echo \"HF cache size: \$(du -sh \$HF_HOME 2>/dev/null | cut -f1 || echo 'empty')\"
        
        echo 'Step 7: Executing Python script...'
        export PYTHONUNBUFFERED=1
        python -u $PYTHON_SCRIPT
        PYTHON_EXIT_CODE=\$?
        echo \"Python script finished with exit code: \$PYTHON_EXIT_CODE\"
        echo '--- Python script finished ---'
    "

CONTAINER_EXIT_CODE=$?
echo "Container exited with code: $CONTAINER_EXIT_CODE"

# --- 9. Stop GPU monitoring ---
echo "Stopping GPU monitoring (PID: $GPU_LOGGER_PID)..."
if kill -0 "$GPU_LOGGER_PID" 2>/dev/null; then
    kill "$GPU_LOGGER_PID"
    echo "GPU monitoring stopped"
else
    echo "GPU monitoring process was not running"
fi

# --- 10. Final job summary ---
echo "===== Job Ended at $(date) ====="
echo "Final job status:"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS

echo "===== GPU Usage Summary (first 20 lines) ====="
if [ -f "$SCRATCH_DIR/gpu_usage.log" ]; then
    head -n 20 "$SCRATCH_DIR/gpu_usage.log"
else
    echo "GPU usage log file not found: $SCRATCH_DIR/gpu_usage.log"
fi

# --- 11. Show final cache sizes ---
echo "===== Final Cache Usage ====="
echo "UV cache size: $(du -sh $UV_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo "PIP cache size: $(du -sh $PIP_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo "HF cache size: $(du -sh $HF_HOME 2>/dev/null | cut -f1 || echo 'empty')"

# --- 12. Cleanup scratch directory ---
echo "Cleaning up scratch directory: $SCRATCH_DIR"
rm -rf "$SCRATCH_DIR"

# Note: Cache directories in $SCRATCH_CACHE will be automatically cleaned up
# when the job ends since they're in /tmp or $TMPDIR
echo "Cleanup completed"
