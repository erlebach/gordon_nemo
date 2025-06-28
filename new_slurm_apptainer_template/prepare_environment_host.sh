#!/bin/bash

set -e

echo "Updating Python environment on frontend (adding CUDA packages)..."

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

# Copy existing .venv if it exists (much faster than rebuilding)
if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "Copying existing .venv to work directory..."
    cp -r "$PROJECT_DIR/.venv" .
    echo "Copied successfully"
fi

# Set environment variables for CUDA package builds
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"  # Support multiple GPU architectures
export FORCE_CUDA="1"  # Force CUDA build for apex
export MAX_JOBS="4"    # Limit parallel builds to avoid memory issues

echo "Running uv sync with --no-build-isolation for CUDA packages..."
uv sync --no-config --no-build-isolation

echo "Verifying updated installation..."
source .venv/bin/activate

echo "Testing packages..."
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