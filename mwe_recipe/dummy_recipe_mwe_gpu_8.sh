#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=%x_%j.out
#SBATCH -J "dummy_recipe_mwe_gpu_8"
#SBATCH -t 00:30:00
#SBATCH -A pilotgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB

echo "===== Job started at $(date) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Running on node(s): $SLURM_NODELIST"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "==================================="

# Load modules and environment
module load cuda/12.1
module load python-uv
uv sync
source ../.venv/bin/activate
uv pip show nemo-run

# Start GPU usage logging in background
(while true; do 
    echo "--- $(date) ---" >> gpu_usage_%x_$SLURM_JOB_ID.log
    nvidia-smi >> gpu_usage_%x_$SLURM_JOB_ID.log
    sleep 10
done) &

# Save background PID so we can kill it later
GPU_LOGGER_PID=$!

# Run your script with resource timing
echo "----- Running Python job -----"
/usr/bin/time -v python dummy_recipe_mwe_gpu_8.py

# Kill the GPU monitoring loop
kill $GPU_LOGGER_PID

# Final job summary
echo "===== Job ended at $(date) ====="
sacct -j $SLURM_JOB_ID --format=JobID,JobName%20,Elapsed,MaxRSS,State

# Optionally print top of GPU usage log
echo "===== GPU Usage Summary ====="
head -n 20 gpu_usage_%x_$SLURM_JOB_ID.log

