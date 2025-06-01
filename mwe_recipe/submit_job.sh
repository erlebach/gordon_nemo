#!/bin/bash
# submit_job.sh

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <slurm_template.sh> <python_script_base_name>"
    echo "Example: $0 templatized_slurm_script.sh dummy_recipe_mwe_gpu_8"
    exit 1
fi

SLURM_TEMPLATE=$1
PYTHON_SCRIPT_BASE=$2

# Generate timestamp
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Submit with job name and pass script base name as argument to template
# sbatch -J "${PYTHON_SCRIPT_BASE}_${TIMESTAMP}" "$SLURM_TEMPLATE" "$PYTHON_SCRIPT_BASE"
echo sbatch -J "${PYTHON_SCRIPT_BASE}_${TIMESTAMP}" "$SLURM_TEMPLATE" "$PYTHON_SCRIPT_BASE"
