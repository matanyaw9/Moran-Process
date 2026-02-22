#!/bin/bash
#BSUB -q short
#BSUB -J master_graph_gen
#BSUB -R "rusage[mem=8192]"
#BSUB -o logs/master_%J.out
#BSUB -e logs/master_%J.err

mkdir -p logs
echo "Starting Master Submission Job..."
uv run python -u main.py