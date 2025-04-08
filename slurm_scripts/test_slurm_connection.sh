#!/bin/bash
#SBATCH -p performance
#SBATCH -t 00:05:00
#SBATCH --job-name=slurm-test
#SBATCH --output=slurm-test-%j.out

echo "Hello from Slurm cluster!"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Available memory: $(free -h)"
echo "User: $(whoami)"
echo "Date: $(date)"

# Test GPU if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "nvidia-smi not available, no CUDA GPU detected"
fi
