        # Set up build environment for transformer-engine
        export NVTE_FRAMEWORK=pytorch
        export NVTE_WITH_USERBUFFERS=0
        export CUDA_HOME=/usr/local/cuda
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        
        # Try multiple approaches to install transformer-engine
        echo "Attempting to install transformer-engine..."
        
        # Method 1: Try with specific version
        pip install "transformer-engine[pytorch]==0.12.0" --no-cache-dir --force-reinstall || \
        
        # Method 2: Try without build isolation
        pip install "transformer-engine[pytorch]" --no-build-isolation --no-cache-dir --force-reinstall || \
        
        # Method 3: Try from NVIDIA's wheel index
        pip install "transformer-engine[pytorch]" --index-url https://pypi.ngc.nvidia.com --no-cache-dir --force-reinstall || \
        
        # Method 4: Try basic installation without extras
        pip install "transformer-engine" --no-cache-dir --force-reinstall
        
        echo "Transformer Engine installation completed (or failed gracefully)"

# The key changes:
#   Added --no-build-isolation to prevent the build environment conflict
#   Added fallback methods in case the first approach fails
#   Added specific version (0.12.0) which is known to work well
#   Added NVIDIA's wheel index as an alternative source
#   Try the first approach first (just adding --no-build-isolation). If that doesn't work, we can use the more comprehensive fallback approach.
