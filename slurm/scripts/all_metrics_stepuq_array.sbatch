#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:rtxA4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=16:00:00
#SBATCH --job-name=stepuq_all_metrics
#SBATCH --output=output/logs/stepuq/stepuq_all_metrics_%A_dataset_%a.out
#SBATCH --error=output/logs/stepuq/stepuq_all_metrics_%A_dataset_%a.err
#SBATCH --export=ALL,HUGGINGFACE_HUB_TOKEN
#SBATCH --array=0  # 5 datasets (ASDiv, 2WikimhQA, hotpotQA, svamp, gsm8k)

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

# List of all UQ methods to run (both GPU and CPU-based)
# We'll handle each one separately to avoid array passing issues in singularity

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
echo "UQ METHODS: self-probing probas-mean probas-entropy probas-max probas-perplexity token-entropy"

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

# Run each UQ method separately to avoid array passing issues in singularity

# 1. Self-probing (GPU-based)
echo "Processing UQ method: self-probing"
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -c "
    cd /home2/mauro.hirt/CoT-UQ
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    
    echo \"===== STARTING self-probing UQ PROCESSING =====\"
    echo \"Dataset: ${DATASET}, UQ Method: self-probing\"
    
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv
    
    python /home2/mauro.hirt/CoT-UQ/stepuq.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --uq_engine self-probing \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || echo \"ERROR: stepuq.py failed for self-probing with code \$?\"
    
    echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_self-probing.json\"
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv
  "

# 2. Probas-mean (CPU-based)
echo "Processing UQ method: probas-mean"
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -c "
    cd /home2/mauro.hirt/CoT-UQ
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    
    echo \"===== STARTING probas-mean UQ PROCESSING =====\"
    echo \"Dataset: ${DATASET}, UQ Method: probas-mean\"
    
    python /home2/mauro.hirt/CoT-UQ/stepuq.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --uq_engine probas-mean \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || echo \"ERROR: stepuq.py failed for probas-mean with code \$?\"
    
    echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_probas-mean.json\"
  "

# 3. Probas-entropy (CPU-based)
echo "Processing UQ method: probas-entropy"
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -c "
    cd /home2/mauro.hirt/CoT-UQ
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    
    echo \"===== STARTING probas-entropy UQ PROCESSING =====\"
    echo \"Dataset: ${DATASET}, UQ Method: probas-entropy\"
    
    python /home2/mauro.hirt/CoT-UQ/stepuq.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --uq_engine probas-entropy \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || echo \"ERROR: stepuq.py failed for probas-entropy with code \$?\"
    
    echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_probas-entropy.json\"
  "

# 4. Probas-max (CPU-based)
echo "Processing UQ method: probas-max"
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -c "
    cd /home2/mauro.hirt/CoT-UQ
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    
    echo \"===== STARTING probas-max UQ PROCESSING =====\"
    echo \"Dataset: ${DATASET}, UQ Method: probas-max\"
    
    python /home2/mauro.hirt/CoT-UQ/stepuq.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --uq_engine probas-max \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || echo \"ERROR: stepuq.py failed for probas-max with code \$?\"
    
    echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_probas-max.json\"
  "

# 5. Probas-perplexity (CPU-based)
echo "Processing UQ method: probas-perplexity"
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -c "
    cd /home2/mauro.hirt/CoT-UQ
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    
    echo \"===== STARTING probas-perplexity UQ PROCESSING =====\"
    echo \"Dataset: ${DATASET}, UQ Method: probas-perplexity\"
    
    python /home2/mauro.hirt/CoT-UQ/stepuq.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --uq_engine probas-perplexity \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || echo \"ERROR: stepuq.py failed for probas-perplexity with code \$?\"
    
    echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_probas-perplexity.json\"
  "

# 6. Token-entropy (CPU-based)
echo "Processing UQ method: token-entropy"
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  -B ${PROJECT_ROOT}:/app/CoT-UQ \
  "${SIF_PATH}" \
  bash -c "
    cd /home2/mauro.hirt/CoT-UQ
    export HF_HOME=${CONTAINER_HF_CACHE}
    export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    export MODEL_PATH_INSIDE=${MODEL_ENGINE}
    
    echo \"===== STARTING token-entropy UQ PROCESSING =====\"
    echo \"Dataset: ${DATASET}, UQ Method: token-entropy\"
    
    python /home2/mauro.hirt/CoT-UQ/stepuq.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --uq_engine token-entropy \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} || echo \"ERROR: stepuq.py failed for token-entropy with code \$?\"
    
    echo \"Output written to: ${OUTPUT_PATH}/confidences/output_v1_token-entropy.json\"
  "

echo "===== ALL UQ METHODS PROCESSING COMPLETED FOR DATASET: ${DATASET} ====="
echo "Job completed for dataset: ${DATASET}"
