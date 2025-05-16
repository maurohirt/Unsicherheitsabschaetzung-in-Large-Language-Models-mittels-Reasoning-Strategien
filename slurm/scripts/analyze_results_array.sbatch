#!/bin/bash

#SBATCH -p performance
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=35:00:00
#SBATCH --job-name=analyze_results
#SBATCH --output=output/logs/analysis/analyze_results_%A_dataset_%a.out
#SBATCH --error=output/logs/analysis/analyze_results_%A_dataset_%a.err
#SBATCH --export=ALL,OPENAI_API_KEY
#SBATCH --array=2 # All 5 datasets (ASDiv, 2WikimhQA, hotpotQA, svamp, gsm8k)

# Environment setup
PROJECT_ROOT="/home2/mauro.hirt/CoT-UQ"
echo "Project root: $PROJECT_ROOT"

export PYTHONPATH=$PROJECT_ROOT
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"

# OpenAI API key (must be set in submission environment)
export OPENAI_API_KEY=${OPENAI_API_KEY:-""}

# Array of all datasets to process
DATASETS=("ASDiv" "2WikimhQA" "hotpotQA" "svamp" "gsm8k")

# List of all UQ methods to analyze
# Modify this array to change the order or run only specific methods
UQ_METHODS=("probas-mean" "probas-min" "token-sar" "p-true" "self-probing")

# Define parameters
MODEL_ENGINE=${1:-"llama3-1_8B"}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_PATH="${PROJECT_ROOT}/output/${MODEL_ENGINE}/${DATASET}"

# Echo parameters for logging
echo "Running analysis with parameters:"
echo "MODEL_ENGINE: ${MODEL_ENGINE}"
echo "DATASET: ${DATASET}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "UQ METHODS: ${UQ_METHODS[*]}"

# Check if inference output exists
if [ ! -f "${OUTPUT_PATH}/output_v1.json" ]; then
    echo "ERROR: Inference output file not found at ${OUTPUT_PATH}/output_v1.json"
    echo "Please run inference_refining.py first for this dataset/model combination"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p output/logs/analysis

# Check if OpenAI API key is set for datasets that need it
if [[ "${DATASET}" == "2WikimhQA" || "${DATASET}" == "hotpotQA" ]] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    echo "OpenAI API key is required for ${DATASET} dataset"
    exit 1
elif [[ "${DATASET}" == "2WikimhQA" || "${DATASET}" == "hotpotQA" ]]; then
    echo "OpenAI API key is set. Length: ${#OPENAI_API_KEY} characters"
else
    echo "OpenAI API key not required for ${DATASET} dataset"
fi

# Run inside singularity container
singularity exec --nv \
  -B ${PROJECT_ROOT}:${PROJECT_ROOT} \
  "${SIF_PATH}" \
  bash -lc "
    cd ${PROJECT_ROOT}
    export OPENAI_API_KEY=${OPENAI_API_KEY}
    export PYTHONPATH=${PROJECT_ROOT}
    
    # Add verbose debugging info
    echo \"Python executable: \$(which python)\"
    echo \"Working directory: \$(pwd)\"
    echo \"PYTHONPATH: \$PYTHONPATH\"
    
    # Process each UQ method in a loop
    UQ_METHODS_STRING=\"${UQ_METHODS[*]}\"
    echo \"Processing methods: \$UQ_METHODS_STRING\"
    
    for UQ_ENGINE in \$UQ_METHODS_STRING; do
      # Check if UQ output exists for this method
      if [ ! -f \"${OUTPUT_PATH}/confidences/output_v1_\$UQ_ENGINE.json\" ]; then
        echo \"WARNING: UQ output file not found at ${OUTPUT_PATH}/confidences/output_v1_\$UQ_ENGINE.json\"
        echo \"Skipping analysis for UQ method: \$UQ_ENGINE\"
        continue
      fi
      
      echo \"===== STARTING RESULT ANALYSIS =====\"
      echo \"Dataset: ${DATASET}, UQ Method: \$UQ_ENGINE\"
      
      # Run the analysis script
      python ${PROJECT_ROOT}/analyze_result.py \
        --dataset ${DATASET} \
        --model_engine ${MODEL_ENGINE} \
        --uq_engine \$UQ_ENGINE \
        --output_path ${OUTPUT_PATH} || {
          echo \"FATAL: analyze_result.py execution failed for \$UQ_ENGINE with exit code \$?\"
          echo \"Continuing with next method...\"
          continue
        }
      
      echo \"===== RESULT ANALYSIS COMPLETED SUCCESSFULLY =====\"
      echo \"Output written to: ${OUTPUT_PATH}/output_v1_w_labels.json\"
      echo \"AUROC calculation completed for UQ method: \$UQ_ENGINE\"
      echo \"\"
    done
  "

echo "===== ALL ANALYSES COMPLETED FOR DATASET: ${DATASET} =====" 
echo "Job completed for dataset: ${DATASET} with all available UQ methods"
