#!/bin/bash

set -e

echo "Setting up complete Python environment on frontend (including CUDA packages)..."

PROJECT_DIR="$PWD"

# Clear any existing virtual environment variables to avoid conflicts
unset VIRTUAL_ENV
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in current directory"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"

# Load CUDA module for build headers (only needed for Apex compilation)
echo "Loading CUDA module for build headers..."
module load cuda/12.1

# Set up persistent cache in scratch space
SCRATCH_CACHE="/tmp/scratch/gerlebacher/.cache"
export UV_CACHE_DIR="$SCRATCH_CACHE/uv"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

echo "Using cache directory: $SCRATCH_CACHE"

# Create virtual environment and install base dependencies
echo "Installing base dependencies with uv..."
uv sync --no-config

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install build dependencies (only needed for Apex)
echo "Installing build dependencies for Apex..."
uv pip install pybind11

# Install CUDA packages
echo "Installing Apex (compiling from source - may take a few minutes)..."
uv pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation

echo "Installing TransformerEngine (pre-built wheel - should be fast)..."
uv pip install transformer-engine

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import apex; print(f'Apex installed at: {apex.__file__}')"
python -c "import transformer_engine; print(f'TransformerEngine: {transformer_engine.__version__}')"

echo ""
echo "✅ Environment setup complete!"
echo "📁 Virtual environment: $PROJECT_DIR/.venv"
echo "💾 Cache location: $SCRATCH_CACHE"
echo ""
echo "To use this environment:"
echo "  source .venv/bin/activate"
echo "  python your_script.py"
echo ""
echo "Or in SLURM jobs, the container will use this pre-built environment." 