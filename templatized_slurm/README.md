# run a SLURM job with a python program used as an argument. 

PYTHON_CODE  = dummy_recipe_mwe_gpu_8.py
sbatch templatized_slurm_script.sh ${PYTHON_CODE}
