#!/bin/bash

set -e

echo "Preparing Python environment using existing container..."

CONTAINER_IMG="$HOME/containers/cuda_uv_12.sif"
PROJECT_DIR="$PWD"

# Check if container exists
if [ ! -f "$CONTAINER_IMG" ]; then
    echo "ERROR: Container not found: $CONTAINER_IMG"
    exit 1
fi

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in current directory"
    echo "Please create pyproject.toml with your dependencies."
    exit 1
fi

echo "Container: $CONTAINER_IMG"
echo "Project directory: $PROJECT_DIR"
echo ""

# Clean up any existing .venv to start fresh
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create virtual environment using container in isolated directory
echo "Creating virtual environment with uv (avoiding parent pyproject.toml conflicts)..."
apptainer exec --nv \
    -B "$PROJECT_DIR":/app \
    --pwd /tmp \
    "$CONTAINER_IMG" \
    bash -c "
        set -e
        echo 'Inside container, working in isolated directory:' \$(pwd)
        
        # Create isolated work directory
        WORK_DIR=/tmp/isolated_project_$$
        mkdir -p \$WORK_DIR
        cd \$WORK_DIR
        echo 'Isolated work directory:' \$(pwd)
        
        # Copy project files to isolated directory
        echo 'Copying project files to isolated directory...'
        cp /app/pyproject.toml .
        cp /app/uv.lock . 2>/dev/null || echo 'No uv.lock found (will be created)'
        
        echo 'Files in isolated directory:'
        ls -la
        
        echo 'Creating virtual environment in isolation...'
        export UV_CACHE_DIR=/tmp/uv_cache_$$
        export PYTHONUNBUFFERED=1
        
        # Use --isolated flag to ignore any parent pyproject.toml files
        uv sync --isolated
        
        echo 'Verifying installation...'
        source .venv/bin/activate
        python -c 'import sys; print(f\"Python: {sys.version}\")'
        python -c 'import numpy; print(f\"NumPy: {numpy.__version__}\")' || echo 'NumPy not available'
        python -c 'import matplotlib; print(f\"Matplotlib: {matplotlib.__version__}\")' || echo 'Matplotlib not available'
        
        echo 'Copying virtual environment back to project directory...'
        cp -r .venv /app/
        
        # Also copy back uv.lock if it was created/updated
        cp uv.lock /app/ 2>/dev/null || echo 'No uv.lock to copy back'
        
        echo 'Cleaning up temporary files...'
        cd /tmp
        rm -rf \$WORK_DIR
        rm -rf /tmp/uv_cache_$$
    "

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: Virtual environment created!"
    echo "Location: $PROJECT_DIR/.venv"
    echo ""
    echo "You can now submit SLURM jobs that will use this environment."
    echo "Run: sbatch script.slurm hello.py"
else
    echo ""
    echo "ERROR: Failed to create virtual environment!"
    exit 1
fi
