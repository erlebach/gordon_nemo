#!/bin/bash
#SBATCH -J llama3_peft
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o peft_%j.out

module load cuda/12.1
module load python-uv

apptainer exec --nv \
  --bind $PWD:/workspace \
  gordon_nemo.sif \
  bash -c "cd /workspace && uv sync && source .venv/bin/activate && python submit_peft_experiment.py"
