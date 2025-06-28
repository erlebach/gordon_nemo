#!/bin/bash -x

set -e

echo "Building enhanced CUDA + UV container with pre-installed dependencies..."
echo $HOME

# Container names
BASE_CONTAINER="$HOME/containers/cuda_uv_12.sif"
NEW_CONTAINER="$HOME/containers/cuda_uv_12_with_deps.sif"
DEF_FILE="cuda_uv_with_deps.def"

# Check if base container exists
if [ ! -f "$BASE_CONTAINER" ]; then
    echo "ERROR: Base container not found: $BASE_CONTAINER"
    echo "Please ensure the base container exists before building."
    exit 1
fi

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    echo "Please ensure $DEF_FILE exists in the current directory."
    exit 1
fi

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in current directory"
    echo "Please create pyproject.toml with your dependencies."
    exit 1
fi

echo "Base container: $BASE_CONTAINER"
echo "New container: $NEW_CONTAINER"
echo "Definition file: $DEF_FILE"
echo ""

# Build the container
echo "Starting build process..."
apptainer build --fakeroot "$NEW_CONTAINER" "$DEF_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: Container built successfully!"
    echo "New container: $NEW_CONTAINER"
    echo ""
    echo "Testing the container..."
    apptainer exec --nv "$NEW_CONTAINER" python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch not available')
try:
    import matplotlib
    print(f'Matplotlib: {matplotlib.__version__}')
except ImportError:
    print('Matplotlib not available')
"
    echo ""
    echo "Container is ready for use!"
    echo "Update your SLURM script to use: $NEW_CONTAINER"
else
    echo "ERROR: Container build failed!"
    exit 1
fi
