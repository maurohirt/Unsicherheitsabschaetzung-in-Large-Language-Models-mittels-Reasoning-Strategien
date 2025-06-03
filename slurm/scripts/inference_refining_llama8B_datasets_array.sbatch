#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:rtxA4500:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=25:00:00
#SBATCH --job-name=cot_llama8B_array
#SBATCH --output=output/logs/llama8B/llama8B_%A_dataset_%a.out
#SBATCH --error=output/logs/llama8B/llama8B_%A_dataset_%a.err
#SBATCH --export=ALL,HUGGINGFACE_HUB_TOKEN
#SBATCH --array=0-3

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

# Array of datasets
DATASETS=("ASDiv" "2WikimhQA" "hotpotQA" "svamp" "gsm8k")

# Define parameters
MODEL_ENGINE="llama3-1_8B"
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_PATH="${PROJECT_ROOT}/output/${MODEL_ENGINE}/${DATASET}"
TEMP=1.0
TRY_TIMES=20
MAX_LENGTH_COT=128

# Echo parameters for logging
echo "Running with parameters:"
echo "MODEL_ENGINE: ${MODEL_ENGINE}"
echo "DATASET: ${DATASET}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "TEMPERATURE: ${TEMP}"
echo "TRY_TIMES: ${TRY_TIMES}"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "DATASET_SELECTED: ${DATASET}"

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
    
    # Configure Hugging Face env inside container
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}

    # Ensure models are cached (download if missing)
    python - <<PY
import os
from huggingface_hub import snapshot_download

repo_map = {
    'llama3-1_8B': 'meta-llama/Llama-3.1-8B',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
}

model = os.environ.get('MODEL_PATH_INSIDE')
repo = repo_map.get(model)
if repo:
    print(f'Ensuring model {repo} is cached...')
    snapshot_download(repo_id=repo, token=os.getenv('HUGGINGFACE_HUB_TOKEN'), resume_download=True)
PY
  
    # GPU Information before loading model
    echo \"===== GPU INFO BEFORE MODEL LOADING =====\"
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv
    
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
    
    echo \"===== INFERENCE COMPLETED SUCCESSFULLY =====\"
  "
