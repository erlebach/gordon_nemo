#!/bin/bash

set -e

echo "Downloading CUDA packages for offline compilation..."

PROJECT_DIR="$PWD"
PACKAGES_DIR="$PROJECT_DIR/cuda_packages"

# Create directory for CUDA packages
mkdir -p "$PACKAGES_DIR"
cd "$PACKAGES_DIR"

echo "Downloading apex..."
if [ ! -d "apex" ]; then
    git clone https://github.com/NVIDIA/apex.git
    echo "Apex downloaded successfully"
else
    echo "Apex already exists, updating..."
    cd apex && git pull && cd ..
fi

echo "Downloading TransformerEngine..."
if [ ! -d "TransformerEngine" ]; then
    git clone https://github.com/NVIDIA/TransformerEngine.git
    echo "TransformerEngine downloaded successfully"
else
    echo "TransformerEngine already exists, updating..."
    cd TransformerEngine && git pull && cd ..
fi

cd "$PROJECT_DIR"

echo ""
echo "SUCCESS: CUDA packages downloaded!"
echo "Location: $PACKAGES_DIR"
echo "Contents:"
ls -la "$PACKAGES_DIR"
echo ""
echo "These will be available on compute nodes for compilation."
