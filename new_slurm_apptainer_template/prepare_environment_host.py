#!/bin/bash

set -e

echo "Preparing base Python environment on frontend host (without CUDA packages)..."

PROJECT_DIR="$PWD"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in current directory"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"

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

# Create isolated work directory
WORK_DIR="/tmp/isolated_project_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Creating base environment (excluding CUDA packages)..."
cp "$PROJECT_DIR/pyproject.toml" .
cp "$PROJECT_DIR/uv.lock" . 2>/dev/null || echo "No uv.lock found"

# Install base dependencies only
uv sync --no-config

echo "Verifying base installation..."
source .venv/bin/activate
python -c 'import sys; print(f"Python: {sys.version}")'
python -c 'import numpy; print(f"NumPy: {numpy.__version__}")'
python -c 'import torch; print(f"PyTorch: {torch.__version__}")'
python -c 'import nemo; print(f"NeMo: {nemo.__version__}")' || echo 'NeMo not available'

echo "Moving environment to project directory..."
mv .venv "$PROJECT_DIR/"
mv uv.lock "$PROJECT_DIR/" 2>/dev/null || true

cd "$PROJECT_DIR"
rm -rf "$WORK_DIR"

echo ""
echo "SUCCESS: Base environment created!"
echo "Location: $PROJECT_DIR/.venv"
echo "Size: $(du -sh .venv | cut -f1)"
echo ""
echo "CUDA packages (apex, transformer-engine) will be installed on compute nodes."
