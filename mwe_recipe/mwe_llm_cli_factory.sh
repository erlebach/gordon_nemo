#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output %x_%j.txt
#SBATCH -J "mwe_llm_cli_factory"
#SBATCH -t 00:30:00
#SBATCH -A pilotgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB

module load cuda/12.1
module load python-uv
uv sync
source ../.venv/bin/activate
uv pip show nemo-run

watch -n 10 (/usr/bin/date; nvidia-smi)

# export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

/usr/bin/time python mwe_llm_cli_factory.py

echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
sacct -j $SLURM_JOB_ID --format=JobID,JobName%20,Elapsed,MaxRSS,State


(nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 10 > gpu_usage_$SLURM_JOB_ID.csv) &

or 

(while true; do echo "--- $(date) ---" >> gpu_usage_$SLURM_JOB_ID.log; nvidia-smi >> gpu_usage_$SLURM_JOB_ID.log; sleep 10; done) &


/usr/bin/time -v python mwe_llm_cli_factory.py

echo "Job started at $(date)"
/usr/bin/time -v python mwe_llm_cli_factory.py
echo "Job ended at $(date)"

trap "kill 0" EXIT
