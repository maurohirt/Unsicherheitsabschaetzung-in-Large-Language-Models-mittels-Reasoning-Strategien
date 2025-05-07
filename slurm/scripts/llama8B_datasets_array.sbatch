#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=llama8B_array
#SBATCH --output=slurm/logs/llama8B/llama8B_%A_dataset_%a.out
#SBATCH --error=slurm/logs/llama8B/llama8B_%A_dataset_%a.err
#SBATCH --array=0-4
#SBATCH --nodelist=server0094

# Environment setup
PROJECT_ROOT="/home2/mauro.hirt/CoT-UQ"
echo "Project root: $PROJECT_ROOT"

export PYTHONPATH=$PROJECT_ROOT
export HOST_HF_CACHE="/home2/mauro.hirt/hf-cache"
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"

# Array of datasets (including gsm8k)
DATASETS=("ASDiv" "2WikimhQA" "hotpotQA" "svamp" "gsm8k")

# Select dataset based on array index
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL_ENGINE="llama3-1_8B"
OUTPUT_PATH="${PROJECT_ROOT}/output/${MODEL_ENGINE}/${DATASET}"
TEMP=1.0
TRY_TIMES=20                   # ← paper value
MAX_COT_LEN=128   # add this after --try_times

# Create a symlink to the log file with the dataset name
LOG_DIR="$PROJECT_ROOT/slurm/logs/llama8B"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_DIR/llama8B_${SLURM_ARRAY_JOB_ID}_dataset_${SLURM_ARRAY_TASK_ID}.out" "$LOG_DIR/llama8B_${SLURM_ARRAY_JOB_ID}_${DATASET}.out"
ln -sf "$LOG_DIR/llama8B_${SLURM_ARRAY_JOB_ID}_dataset_${SLURM_ARRAY_TASK_ID}.err" "$LOG_DIR/llama8B_${SLURM_ARRAY_JOB_ID}_${DATASET}.err"

# Echo parameters for logging
echo "Running with parameters:"
echo "MODEL_ENGINE: ${MODEL_ENGINE}"
echo "DATASET: ${DATASET}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "TEMPERATURE: ${TEMP}"
echo "TRY_TIMES: ${TRY_TIMES}"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "JOB_ID: ${SLURM_ARRAY_JOB_ID}"

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
    MODEL_PATH_INSIDE=\"/root/.cache/huggingface/models/llama3-8B\"
    
    echo \"===== MODEL SETUP =====\"
    echo \"Using model path: \${MODEL_PATH_INSIDE}\"
    
    # Create a symlink for llama3-1_8B to point to the model directory
    mkdir -p /root/.cache/huggingface/symlinks
    ln -sfn "\${MODEL_PATH_INSIDE}" /root/.cache/huggingface/symlinks/llama3-1_8B
    echo \"Created symlink: llama3-1_8B -> \${MODEL_PATH_INSIDE}\"
    
    # Update model path to use the symlink name that matches config options
    MODEL_PATH_INSIDE=\"llama3-1_8B\"
    
    # GPU Information before loading model
    echo \"===== GPU INFO BEFORE MODEL LOADING =====\"
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv
    
    echo \"===== STARTING INFERENCE FOR DATASET: ${DATASET} =====\"
    python /home2/mauro.hirt/CoT-UQ/inference_refining.py \
      --dataset ${DATASET} \
      --model_engine ${MODEL_ENGINE} \
      --model_path \${MODEL_PATH_INSIDE} \
      --temperature ${TEMP} \
      --output_path ${OUTPUT_PATH} \
      --try_times ${TRY_TIMES} \
      --max_length_cot ${MAX_COT_LEN} || {
        echo \"FATAL: inference_refining.py execution failed with exit code \$?\"
        exit 1
      }
    
    # GPU Information after inference
    echo \"===== GPU INFO AFTER INFERENCE =====\"
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv
    
    echo \"===== INFERENCE COMPLETED SUCCESSFULLY FOR DATASET: ${DATASET} =====\"
  "

# Capture the exit status
EXIT_STATUS=$?

# Output job completion status
if [ $EXIT_STATUS -eq 0 ]; then
  echo "✅ Inference job completed successfully for dataset: ${DATASET}"
else
  echo "❌ Inference job failed with exit status $EXIT_STATUS for dataset: ${DATASET}"
fi

# Output information about the log files
echo "Log files:"
echo "- Standard output: slurm/logs/llama8B/llama8B_${SLURM_ARRAY_JOB_ID}_${DATASET}.out"
echo "- Standard error:  slurm/logs/llama8B/llama8B_${SLURM_ARRAY_JOB_ID}_${DATASET}.err"

exit $EXIT_STATUS