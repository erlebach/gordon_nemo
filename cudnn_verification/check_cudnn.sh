#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "finetune_lora_script"
#SBATCH -t 00:30:00
#SBATCH -A pilotgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB

module load python-uv
module load cuda/12.1
uv sync
source ../.venv/bin/activate

# export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

# echo "-------------"
# echo "module avail"
# echo `module avail`
# echo "-------------"
# echo $LD_LIBRARY_PATH
# echo "-------------"
# echo "ls /usr/local/cuda-12.1/lib64"
# echo `ls /usr/local/cuda-12.1/lib64`
# echo "-------------"
# echo "ls /usr/local/cuda-12.1/lib64/stubs"
# echo `ls /usr/local/cuda-12.1/lib64/stubs`
# echo "-------------"
# echo "ls /opt/rcc/precompiled/lib64"
# echo `ls /opt/rcc/precompiled/lib64`
# echo "------------"
# echo "ls /lib64"
# echo `ls /lib64`
rpm -qa | grep cudnn9
python check_cudnn.py

