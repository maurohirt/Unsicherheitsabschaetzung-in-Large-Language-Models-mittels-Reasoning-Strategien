#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=cot_inference_13B
#SBATCH --output=slurm/logs/slurm_inference_llama13B_%j.out
#SBATCH --error=slurm/logs/slurm_inference_llama13B_%j.err

# Environment setup
PROJECT_ROOT="/home2/mauro.hirt/CoT-UQ"
echo "Project root: $PROJECT_ROOT"

export PYTHONPATH=$PROJECT_ROOT
export HOST_HF_CACHE="/home2/mauro.hirt/hf-cache"
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"

# Define parameters (can be overridden by command line arguments)
MODEL_ENGINE=${1:-"llama2-13b"}
DATASET=${2:-"gsm8k"}
OUTPUT_PATH=${3:-"${PROJECT_ROOT}/output/slurm-test-13B/${DATASET}"}
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
    # Navigate to the code directory
    cd /home2/mauro.hirt/CoT-UQ
    
    # Set up environment
    export HF_HOME=\"${CONTAINER_HF_CACHE}\"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export PYTHONPATH=/home2/mauro.hirt/CoT-UQ
    
    # Use the exact path from the successful test
    MODEL_PATH_INSIDE=\"/root/.cache/huggingface/models/llama2-13b\"
    
    echo \"===== MODEL SETUP =====\"
    echo \"Using model path: \${MODEL_PATH_INSIDE}\"
    
    # Create a symlink for llama2-13b to point to the model directory
    mkdir -p /root/.cache/huggingface/symlinks
    ln -sfn \"\${MODEL_PATH_INSIDE}\" /root/.cache/huggingface/symlinks/llama2-13b
    echo \"Created symlink: llama2-13b -> \${MODEL_PATH_INSIDE}\"
    
    # Update model path to use the symlink name that matches config options
    MODEL_PATH_INSIDE=\"llama2-13b\"
    
    # GPU Information before loading model
    echo \"===== GPU INFO BEFORE MODEL LOADING =====\"
    nvidia-smi --query-gpu=memory.total,memory.used,gpu_name --format=csv
    
    echo \"===== STARTING INFERENCE =====\"
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
    
    # GPU Information after inference
    echo \"===== GPU INFO AFTER INFERENCE =====\"
    nvidia-smi --query-gpu=memory.total,memory.used,gpu_name --format=csv
    
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
echo "- Standard output: slurm/logs/slurm_inference_llama13B_${SLURM_JOB_ID}.out"
echo "- Standard error:  slurm/logs/slurm_inference_llama13B_${SLURM_JOB_ID}.err"

exit $EXIT_STATUS