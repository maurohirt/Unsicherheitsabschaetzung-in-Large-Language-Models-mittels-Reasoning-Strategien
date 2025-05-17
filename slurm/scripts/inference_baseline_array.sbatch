#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:rtxA4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --job-name=baseline_inference
#SBATCH --output=output/logs/baseline/baseline_inference_%A_dataset_%a.out
#SBATCH --error=output/logs/baseline/baseline_inference_%A_dataset_%a.err
#SBATCH --export=ALL,HUGGINGFACE_HUB_TOKEN
#SBATCH --array=1-4 # test run later All 5 datasets (ASDiv, 2WikimhQA, hotpotQA, svamp, gsm8k)

# Environment setup
PROJECT_ROOT="/home2/mauro.hirt/CoT-UQ"
echo "Project root: $PROJECT_ROOT"

export PYTHONPATH=$PROJECT_ROOT
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"

# Hugging Face caching
export HOST_HF_CACHE="${PROJECT_ROOT}/hf_cache"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"
mkdir -p ${HOST_HF_CACHE}
# Hugging Face token (must be set in submission environment)
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-""}

# List of datasets to process
DATASETS=("ASDiv" "2WikimhQA" "hotpotQA" "svamp" "gsm8k")

# Define parameters
MODEL_ENGINE=${1:-"llama3-1_8B"}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_PATH="${PROJECT_ROOT}/output/${MODEL_ENGINE}/${DATASET}"
TRY_TIMES=5
MAX_LENGTH_COT=1024

# Create output directories
mkdir -p ${OUTPUT_PATH}/baseline
mkdir -p output/logs/baseline

# Echo parameters for logging
echo "Running with parameters:"
echo "MODEL_ENGINE: ${MODEL_ENGINE}"
echo "DATASET: ${DATASET}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "TRY_TIMES: ${TRY_TIMES}"
echo "MAX_LENGTH_COT: ${MAX_LENGTH_COT}"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"

# Show GPU info before starting
echo "===== GPU INFO BEFORE INFERENCE ====="
nvidia-smi --query-gpu=memory.total,memory.used --format=csv

# Run inference in container
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:${PROJECT_ROOT} \
  "${SIF_PATH}" \
  bash -lc "
    cd ${PROJECT_ROOT}
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    export PYTHONPATH=${PROJECT_ROOT}
    
    # Add verbose debugging info
    echo \"Python executable: \$(which python)\"
    echo \"Working directory: \$(pwd)\"
    echo \"PYTHONPATH: \$PYTHONPATH\"
    echo \"HF_HOME: \$HF_HOME\"
    
    # Run inference
    echo \"===== STARTING BASELINE INFERENCE =====\"
    echo \"Dataset: ${DATASET}\"
    
    python ${PROJECT_ROOT}/inference_baseline.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} \
      --max_length_cot ${MAX_LENGTH_COT} || {
        echo \"FATAL: inference_baseline.py execution failed with code \$?\"
        exit 1
      }
    
    echo \"Baseline inference completed. Output written to: ${OUTPUT_PATH}/baseline/output_v1.json\"
  "

# Show GPU info after completion
echo "===== GPU INFO AFTER INFERENCE ====="
nvidia-smi --query-gpu=memory.total,memory.used --format=csv

echo "===== BASELINE INFERENCE COMPLETED FOR DATASET: ${DATASET} ====="
echo "Job completed for dataset: ${DATASET}"
