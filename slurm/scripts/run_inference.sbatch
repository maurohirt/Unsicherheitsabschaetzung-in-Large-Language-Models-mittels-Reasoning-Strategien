#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=cot_inference
#SBATCH --output=slurm/logs/slurm_inference_%j.out
#SBATCH --error=slurm/logs/slurm_inference_%j.err

# Environment setup
# Hard-code project root to the actual location on the cluster
PROJECT_ROOT="/home2/mauro.hirt/CoT-UQ"
echo "Project root: $PROJECT_ROOT"

# Verify that main script exists
if [ -f "$PROJECT_ROOT/inference_refining.py" ]; then
  echo "✓ Found inference_refining.py at $PROJECT_ROOT"
else
  echo "✗ ERROR: inference_refining.py not found at $PROJECT_ROOT"
  echo "Searching for it elsewhere..."
  find /home2/mauro.hirt -name "inference_refining.py" -type f
fi

export PYTHONPATH=$PROJECT_ROOT
export HOST_HF_CACHE="/home2/mauro.hirt/hf-cache"
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"

# Define parameters (can be overridden by command line arguments)
MODEL_ENGINE=${1:-"llama3-1_8B"}
DATASET=${2:-"gsm8k"}
OUTPUT_PATH=${3:-"${PROJECT_ROOT}/output/slurm-test/${DATASET}"}
TEMP=${4:-1.0}
TRY_TIMES=${5:-20}

# Echo parameters for logging
echo "Running with parameters:"
echo "MODEL_ENGINE: ${MODEL_ENGINE}"
echo "DATASET: ${DATASET}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "TEMPERATURE: ${TEMP}"
echo "TRY_TIMES: ${TRY_TIMES}"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_PATH}

# Run inside singularity container
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -lc "
    # Enable error handling and debug output inside container
    set -e
    set -x
    
    # Debug filesystem mounting
    echo \"===== ENVIRONMENT & DIRECTORY DEBUGGING =====\"
    echo \"Current working directory: \$(pwd)\"
    echo \"Contents of current directory: \"
    ls -la
    echo \"Python version:\"
    python --version
    echo \"PATH:\"
    echo \$PATH
    
    # Navigate to the mounted code directory
    cd /app/CoT-UQ || {
      echo \"FATAL: Could not change to /app/CoT-UQ directory\"
      exit 1
    }
    
    echo \"Contents of /app/CoT-UQ:\"
    ls -la /app/CoT-UQ
    
    # Check Python files in the project
    echo \"===== EXAMINING PROJECT STRUCTURE =====\"
    echo \"Listing main directory:\"
    ls -la
    
    # Check where the code files are
    echo \"Looking for Python files in current dir:\"
    find . -maxdepth 1 -name \"*.py\" || echo \"No Python files in current directory\"
    
    echo \"Checking if code was copied during container build:\"
    ls -la /app || echo \"No /app directory\"
    
    # Set up environment
    export HF_HOME=\"${CONTAINER_HF_CACHE}\"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export PYTHONPATH=/home2/mauro.hirt/CoT-UQ
    echo \"PYTHONPATH: \$PYTHONPATH\"
    
    # Use the exact path from the successful test (line 55 in the MD file)
    MODEL_PATH_INSIDE=\"/root/.cache/huggingface/models/llama3-8B\"
    
    echo \"===== MODEL PATH DEBUGGING =====\"
    echo \"Using model path: \${MODEL_PATH_INSIDE}\"
    if [ -d \"\${MODEL_PATH_INSIDE}\" ]; then
      echo \"Model directory exists. Contents:\"
      ls -la \"\${MODEL_PATH_INSIDE}\"
      
      # Create a symlink for llama3-1_8B to point to the model directory
      echo \"===== CREATING SYMLINK FOR MODEL =====\"
      mkdir -p /root/.cache/huggingface/symlinks || echo \"Failed to create symlinks directory\"
      ln -sfn \"\${MODEL_PATH_INSIDE}\" /root/.cache/huggingface/symlinks/llama3-1_8B
      echo \"Created symlink: llama3-1_8B -> \${MODEL_PATH_INSIDE}\"
      ls -la /root/.cache/huggingface/symlinks/
      
      # Update model path to use the symlink name that matches config options
      MODEL_PATH_INSIDE=\"llama3-1_8B\"
      echo \"Using model name: \${MODEL_PATH_INSIDE} (via symlink)\"
    else
      echo \"WARNING: Model directory not found. Listing alternate locations:\"
      echo \"Cached models directory:\"
      ls -la /root/.cache/huggingface || echo \"No huggingface cache directory\"
      
      if [ -d \"/root/.cache/huggingface/models\" ]; then
        echo \"Models subdirectory exists. Contents:\"
        ls -la /root/.cache/huggingface/models
      else
        echo \"Models subdirectory doesn't exist\"
      fi
    fi
    
    echo \"===== PYTHON IMPORTS TEST =====\"
    python -c \"
import sys
print('Python path:', sys.path)
try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU device count:', torch.cuda.device_count())
        print('GPU device name:', torch.cuda.get_device_name(0))
except Exception as e:
    print('Error importing torch:', e)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('Transformers imported successfully')
except Exception as e:
    print('Error importing transformers:', e)
\" || echo \"Python import test failed\"

    echo \"===== STARTING INFERENCE =====\"
    # Check if the entrypoint file exists in the mounted directory
    echo \"Checking for inference_refining.py:\"
    ls -la /home2/mauro.hirt/CoT-UQ/inference_refining.py || echo \"File not found at expected location\"
    
    # Change directory to the home location where we know the code exists
    cd /home2/mauro.hirt/CoT-UQ
    echo \"Changed to directory: \$(pwd)\"
    echo \"Files in this directory:\"
    ls -la
    
    # Run inference with clearer error handling from the home directory
    # Note: We've modified config.py to accept absolute paths for model_path
    python /home2/mauro.hirt/CoT-UQ/inference_refining.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --temperature ${TEMP} \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || {
        echo \"FATAL: inference_refining.py execution failed with exit code \$?\"
        exit 1
      }
    
    echo \"===== INFERENCE COMPLETED SUCCESSFULLY =====\"
  "

# Capture the exit status
EXIT_STATUS=$?

# Output job completion status
if [ $EXIT_STATUS -eq 0 ]; then
  echo "✅ Inference job completed successfully"
else
  echo "❌ Inference job failed with exit status $EXIT_STATUS"
fi

# Output information about the log files
echo "Log files:"
echo "- Standard output: slurm/logs/slurm_inference_${SLURM_JOB_ID}.out"
echo "- Standard error:  slurm/logs/slurm_inference_${SLURM_JOB_ID}.err"

exit $EXIT_STATUS