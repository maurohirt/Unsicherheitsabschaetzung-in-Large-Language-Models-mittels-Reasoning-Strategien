#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:rtxA4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:35:00
#SBATCH --job-name=stepuq_all_metrics_bl
#SBATCH --output=output/logs/stepuq/stepuq_all_metrics_bl_%A_dataset_%a.out
#SBATCH --error=output/logs/stepuq/stepuq_all_metrics_bl_%A_dataset_%a.err
#SBATCH --export=ALL,HUGGINGFACE_HUB_TOKEN
#SBATCH --array=0-4 # 0 for Test ;5 datasets (ASDiv, 2WikimhQA, hotpotQA, svamp, gsm8k)

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

# Array of all datasets to process
DATASETS=("ASDiv" "2WikimhQA" "hotpotQA" "svamp" "gsm8k")

# List of all UQ baseline methods to run
UQ_METHODS=(
  #  "probas-mean-bl"
    "probas-min-bl"
  #  "token-sar-bl"
)

# Define parameters
MODEL_ENGINE=${1:-"llama3-1_8B"}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_PATH="${PROJECT_ROOT}/output/${MODEL_ENGINE}/${DATASET}"
TRY_TIMES=5

# Echo parameters for logging
echo "Running with parameters:"
echo "MODEL_ENGINE: ${MODEL_ENGINE}"
echo "DATASET: ${DATASET}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "TRY_TIMES: ${TRY_TIMES}"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "UQ METHODS: ${UQ_METHODS[*]}"

# Check if inference output exists
if [ ! -f "${OUTPUT_PATH}/output_v1.json" ]; then
    echo "ERROR: Inference output file not found at ${OUTPUT_PATH}/output_v1.json"
    echo "Please run inference_refining.py first for this dataset/model combination"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p ${OUTPUT_PATH}/confidences
mkdir -p output/logs/stepuq

# Show initial GPU status
echo "===== GPU INFO BEFORE ANY PROCESSING ====="
nvidia-smi --query-gpu=memory.total,memory.used --format=csv

# Run all UQ methods in a single container execution
echo "Processing all UQ methods for dataset: ${DATASET}"
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
    
    # Add verbose debugging info once at the start
    echo \"Python executable: \$(which python)\"
    echo \"Working directory: \$(pwd)\"
    echo \"PYTHONPATH: \$PYTHONPATH\"
    
    # Pass UQ_METHODS array to container
    UQ_METHODS_STRING=\"${UQ_METHODS[*]}\"
    echo \"Processing methods: \$UQ_METHODS_STRING\"
    
    # Process each UQ method in a loop
    for UQ_ENGINE in \$UQ_METHODS_STRING; do
      echo \"===== STARTING \$UQ_ENGINE UQ PROCESSING =====\"
      echo \"Dataset: ${DATASET}, UQ Method: \$UQ_ENGINE\"
      
      # Show GPU info for self-probing (GPU-based method)
      if [ \"\$UQ_ENGINE\" == \"self-probing\" ]; then
        echo \"GPU INFO BEFORE \$UQ_ENGINE:\"
        nvidia-smi --query-gpu=memory.total,memory.used --format=csv
      fi
      
      # Run the UQ method
      python ${PROJECT_ROOT}/stepuq.py \
        --dataset ${DATASET} \
        --model_engine ${MODEL_ENGINE} \
        --model_path \${MODEL_PATH_INSIDE} \
        --uq_engine \$UQ_ENGINE \
        --output_path ${OUTPUT_PATH} \
        --try_times ${TRY_TIMES} || {
          echo \"FATAL: stepuq.py execution failed for \$UQ_ENGINE with code \$?\"
          
          # For self-probing, check if there's an error log
          if [ \"\$UQ_ENGINE\" == \"self-probing\" ] && [ -f \"${OUTPUT_PATH}/error.log\" ]; then
            echo \"Error log contents:\"
            cat \"${OUTPUT_PATH}/error.log\"
          fi
          
          # Continue with other methods instead of exiting
          echo \"Continuing with next method...\"
          continue
        }
      
      echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_\$UQ_ENGINE.json\"
      
      # Show GPU info after self-probing
      if [ \"\$UQ_ENGINE\" == \"self-probing\" ]; then
        echo \"GPU INFO AFTER \$UQ_ENGINE:\"
        nvidia-smi --query-gpu=memory.total,memory.used --format=csv
      fi
      
      echo \"\"
    done
  "

echo "===== ALL UQ METHODS PROCESSING COMPLETED FOR DATASET: ${DATASET} ====="
echo "Job completed for dataset: ${DATASET}"
