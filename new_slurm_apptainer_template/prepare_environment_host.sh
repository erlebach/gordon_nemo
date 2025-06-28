#!/bin/bash

set -e

echo "Preparing base Python environment on frontend (excluding CUDA packages)..."

PROJECT_DIR="$PWD"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in current directory"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"

# Clean up UV cache first
echo "Cleaning up UV cache to free space..."
rm -rf ~/.cache/uv 2>/dev/null || echo "No UV cache to clean"

# Check disk usage
echo "Checking disk usage:"
df -h "$HOME" | head -2

# Clean up any existing .venv
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Please install uv first."
    exit 1
fi

# Create isolated work directory in /tmp (not in home directory)
WORK_DIR="/tmp/isolated_project_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Creating base environment in temporary location..."
cp "$PROJECT_DIR/pyproject.toml" .

# Create a minimal pyproject.toml without heavy packages
cat > pyproject_minimal.toml << 'EOF'
[project]
name = "slurm-apptainer-template"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "matplotlib>=3.8.0",
    "torch>=2.0.0",
    "pandas>=2.0.0",
    "tqdm",
    "hydra-core",
    # Try minimal NeMo or exclude it entirely for now
    # "nemo-toolkit>=2.1.0",
]
EOF

mv pyproject_minimal.toml pyproject.toml

# Use /tmp for UV cache instead of home directory
export UV_CACHE_DIR="/tmp/uv_cache_$$"
echo "Using temporary UV cache: $UV_CACHE_DIR"

uv sync --no-config

echo "Verifying base installation..."
source .venv/bin/activate
python -c 'import sys; print(f"Python: {sys.version}")'
python -c 'import numpy; print(f"NumPy: {numpy.__version__}")'
python -c 'import torch; print(f"PyTorch: {torch.__version__}")'

echo "Moving environment to project directory..."
mv .venv "$PROJECT_DIR/"

cd "$PROJECT_DIR"

# Clean up temporary files
rm -rf "$WORK_DIR"
rm -rf "/tmp/uv_cache_$$"

echo ""
echo "SUCCESS: Minimal base environment created!"
echo "Location: $PROJECT_DIR/.venv"
echo "Size: $(du -sh .venv | cut -f1)"
echo ""
echo "NeMo and CUDA packages will be installed on compute nodes." 