#!/bin/bash
#SBATCH -J llama3_full_ft
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o full_ft_%j.out

module load cuda/12.1
module load python-uv

apptainer exec --nv \
  --bind $PWD:/workspace \
  gordon_nemo.sif \
  bash -c "cd /workspace && uv sync && source .venv/bin/activate && python submit_full_finetune_from_peft.py"

