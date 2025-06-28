#!/bin/bash

set -e

echo "Updating Python environment on frontend (adding CUDA packages)..."

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

# Load CUDA module for build headers (not runtime)
echo "Loading CUDA module for build headers..."
module load cuda/12.1

# Set up persistent cache in scratch space
SCRATCH_CACHE_BASE="/tmp/scratch/gerlebacher"
PERSISTENT_CACHE_DIR="$SCRATCH_CACHE_BASE/.cache"

# Create cache directories if they don't exist
mkdir -p "$PERSISTENT_CACHE_DIR"

# Set cache environment variables to persistent locations
export UV_CACHE_DIR="$PERSISTENT_CACHE_DIR/uv"
export PIP_CACHE_DIR="$PERSISTENT_CACHE_DIR/pip"
export HF_HOME="$PERSISTENT_CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$PERSISTENT_CACHE_DIR"

echo "Using persistent cache directories:"
echo "  UV cache: $UV_CACHE_DIR"

# Show CUDA environment
echo "CUDA environment:"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found')"

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "Found existing .venv - will update it incrementally"
    echo "Current size: $(du -sh .venv | cut -f1)"
else
    echo "No existing .venv found - will create new one"
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Please install uv first."
    exit 1
fi

# Create isolated work directory in /tmp
WORK_DIR="/tmp/isolated_project_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Setting up isolated work environment..."
cp "$PROJECT_DIR/pyproject.toml" .
cp "$PROJECT_DIR/uv.lock" . 2>/dev/null || echo "No existing uv.lock found (will be created)"

# Create virtual environment first
echo "Creating virtual environment..."
uv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install setuptools and build tools first
echo "Installing build dependencies..."
uv pip install setuptools wheel

# Copy existing .venv content if it exists (after we have a working venv)
if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "Copying packages from existing .venv..."
    # Copy site-packages from existing environment
    cp -r "$PROJECT_DIR/.venv/lib/python3.10/site-packages/"* .venv/lib/python3.10/site-packages/ 2>/dev/null || echo "Some packages may not have copied"
fi

# Set environment variables for CUDA package builds
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA="1"
export MAX_JOBS="4"

echo "Running uv sync with build isolation disabled..."
uv sync --no-build-isolation

echo "Verifying updated installation..."
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")'
python -c 'import nemo; print(f"NeMo: {nemo.__version__}")'
python -c 'import apex; print(f"Apex: available")' || echo 'Apex: FAILED to import'
python -c 'import transformer_engine; print(f"TransformerEngine: available")' || echo 'TransformerEngine: FAILED to import'

echo "Moving updated environment back to project directory..."
# Remove old .venv only after successful build
if [ -d "$PROJECT_DIR/.venv" ]; then
    rm -rf "$PROJECT_DIR/.venv"
fi
mv .venv "$PROJECT_DIR/"
mv uv.lock "$PROJECT_DIR/" 2>/dev/null || echo "No uv.lock to move"

cd "$PROJECT_DIR"
rm -rf "$WORK_DIR"

echo ""
echo "SUCCESS: Environment updated!"
echo "Location: $PROJECT_DIR/.venv"
echo "Lock file: $PROJECT_DIR/uv.lock"
echo "Final size: $(du -sh .venv | cut -f1)"
echo ""
echo "You can now run: uv run python your_script.py" 