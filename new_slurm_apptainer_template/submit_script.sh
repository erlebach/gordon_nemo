#!/bin/bash -x

SCRIPT=$1
PYTHON=$2
sbatch $1 $2
# run_in_apptainer.slurm my_script.py
