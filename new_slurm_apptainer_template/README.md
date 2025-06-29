# New Slurm Templatized Tempalte
- Demonstrate a fancy templatized template 
- handles apptainer
- handles a single python script
- sets up uv project
----------------------------------------------------------------------
# File structure
gordon_nemo/                          # Root directory
├── pyproject.toml                    # Single shared dependencies
├── .venv/                           # Single shared virtual environment
├── prepare_environment_host.sh       # Single setup script
├── new_slurm_apptainer_template/
│   ├── script.slurm                 # Project-specific SLURM script
│   └── hello.py                     # Project files
├── project2/
│   ├── script.slurm                 # Different SLURM script
│   └── other_files.py
└── project3/
    ├── script.slurm
    └── more_files.py

# One-time setup from root directory
cd gordon_nemo
bash prepare_environment_host.sh

# Use from any project
cd new_slurm_apptainer_template
sbatch script.slurm hello.py

cd ../project2  
sbatch script.slurm other_script.py
