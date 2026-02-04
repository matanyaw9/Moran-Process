#!/bin/bash
#BSUB -q short
#BSUB -J master_graph_gen
#BSUB -R "rusage[mem=8192]"
#BSUB -o logs/master_%J.out
#BSUB -e logs/master_%J.err

# Create logs directory if it doesn't exist
mkdir -p logs

# FIXED: Bash uses 'echo', not 'print'
echo "Starting Master Submission Job..."

# Run the python script
uv run python -u main.py