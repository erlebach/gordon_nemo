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
        
        # Install git temporarily in the container session
        apt-get update && apt-get install -y git
        
        # Create isolated work directory
        WORK_DIR=/tmp/isolated_project_$$
        mkdir -p \$WORK_DIR
        cd \$WORK_DIR
        
        # Copy project files
        cp /app/pyproject.toml .
        cp /app/uv.lock . 2>/dev/null || true
        
        # Now uv sync will work with git dependencies
        export UV_CACHE_DIR=/tmp/uv_cache_$$
        uv sync --no-config
        
        # Use --isolated flag to ignore any parent pyproject.toml files
        uv sync --no-config
        
        echo 'Verifying installation...'
        source .venv/bin/activate
        python -c 'import sys; print(f\"Python: {sys.version}\")'
        python -c 'import numpy; print(f\"NumPy: {numpy.__version__}\")' || echo 'NumPy not available'
        python -c 'import matplotlib; print(f\"Matplotlib: {matplotlib.__version__}\")' || echo 'Matplotlib not available'
        
        echo 'Copying virtual environment back to project directory...'
        cp -r .venv /app/
        
        # Cleanup
        cd /tmp && rm -rf \$WORK_DIR && rm -rf /tmp/uv_cache_$$
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
