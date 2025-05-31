#!/bin/bash
# submit_job.sh

# Check if script name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_name.sh>"
    exit 1
fi

SCRIPT_NAME=$1

# Extract base name without extension for job name
JOB_NAME=$(basename "$SCRIPT_NAME" .sh)

# Generate timestamp
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Submit with derived job name
sbatch -J "${JOB_NAME}_${TIMESTAMP}" "$SCRIPT_NAME"
