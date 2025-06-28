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

# Check current cache sizes
echo "Current cache sizes:"
echo "  UV: $(du -sh $UV_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo "  PIP: $(du -sh $PIP_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo "  HF: $(du -sh $HF_HOME 2>/dev/null | cut -f1 || echo 'empty')"

# Check disk usage
echo "Checking disk usage:"
df -h "$HOME" | head -2
df -h "$SCRATCH_CACHE_BASE" | head -2 2>/dev/null || echo "Scratch space not yet available"

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

echo "Running uv sync (this will populate the cache)..."
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

# Show final cache sizes
echo ""
echo "Cache populated! Final sizes:"
echo "  UV: $(du -sh $UV_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo "  PIP: $(du -sh $PIP_CACHE_DIR 2>/dev/null | cut -f1 || echo 'empty')"
echo "  HF: $(du -sh $HF_HOME 2>/dev/null | cut -f1 || echo 'empty')"

echo ""
echo "SUCCESS: Minimal base environment created!"
echo "Location: $PROJECT_DIR/.venv"
echo "Size: $(du -sh .venv | cut -f1)"
echo ""
echo "Cache is now populated for fast job execution."
echo "NeMo and CUDA packages will be installed on compute nodes." 