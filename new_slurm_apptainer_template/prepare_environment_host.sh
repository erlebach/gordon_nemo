#!/bin/bash

set -e

echo "Preparing FULL Python environment on frontend (including CUDA packages)..."

PROJECT_DIR="$PWD"

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
echo "  PIP cache: $PIP_CACHE_DIR"
echo "  HF cache: $HF_HOME"

# Show CUDA environment
echo "CUDA environment:"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found')"

# Remove any existing .venv
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
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

echo "Creating FULL environment (including CUDA packages)..."
cp "$PROJECT_DIR/pyproject.toml" .
cp "$PROJECT_DIR/uv.lock" . 2>/dev/null || echo "No existing uv.lock found (will be created)"

# Set environment variables for CUDA package builds
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"  # Support multiple GPU architectures
export FORCE_CUDA="1"  # Force CUDA build for apex
export MAX_JOBS="4"    # Limit parallel builds to avoid memory issues

echo "Building full environment with CUDA packages..."
echo "This may take several minutes for first-time builds..."

# Use --no-build-isolation for packages that need torch during build
uv sync --no-config

echo "Verifying FULL installation..."
source .venv/bin/activate

echo "Testing base packages..."
python -c 'import sys; print(f"Python: {sys.version}")'
python -c 'import numpy; print(f"NumPy: {numpy.__version__}")'
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")'
python -c 'import nemo; print(f"NeMo: {nemo.__version__}")'

echo "Testing CUDA packages..."
python -c 'import apex; print(f"Apex: available")' || echo 'Apex: FAILED'
python -c 'import transformer_engine; print(f"TransformerEngine: available")' || echo 'TransformerEngine: FAILED'

echo "Moving environment and lock file to project directory..."
mv .venv "$PROJECT_DIR/"
mv uv.lock "$PROJECT_DIR/" 2>/dev/null || echo "No uv.lock to move"

cd "$PROJECT_DIR"
rm -rf "$WORK_DIR"

echo ""
echo "SUCCESS: FULL environment created!"
echo "Location: $PROJECT_DIR/.venv"
echo "Lock file: $PROJECT_DIR/uv.lock"
echo "Size: $(du -sh .venv | cut -f1)"
echo ""
echo "Cache sizes:"
echo "  UV: $(du -sh $UV_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo ""
echo "You can now run: uv run python your_script.py"
echo "Or activate with: source .venv/bin/activate" 