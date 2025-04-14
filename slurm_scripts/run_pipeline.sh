#!/bin/bash
#SBATCH --job-name=cot_uq_run
#SBATCH --output=../../outputs/cot_uq_%j.log
#SBATCH --error=../../outputs/cot_uq_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=performance
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Load singularity
module load singularity

# Container information
DOCKER_REGISTRY="docker.io"
IMAGE_NAME="maurohirtfhnw/cot-uq"
IMAGE_TAG="latest"
SIF_FILE="../../containers/cot-uq_${IMAGE_TAG}.sif"

# Ensure the container exists
if [ ! -f "$SIF_FILE" ]; then
    echo "Container not found, pulling from Docker registry..."
    singularity pull "$SIF_FILE" "docker://${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    if [ $? -ne 0 ]; then
        echo "Error: Singularity pull failed!"
        exit 1
    fi
fi

# Parameters
MODEL_ENGINE="llama3-1_8B"  # Can be llama3-1_8B or llama2-13b
UQ_ENGINE="probas-mean"     # Uncertainty quantification engine
DATASET="gsm8k"             # Dataset choice: ASDiv, 2WikimhQA, gsm8k, hotpotQA, svamp, etc.
OUTPUT_PATH="output/${MODEL_ENGINE}_${DATASET}"
TEMP=1.0
TRY_TIMES=5                 # Start with fewer tries for testing

# Test subset (add these parameters to run on just part of the dataset)
TEST_START=0
TEST_END=100                # Start with a small subset for testing

echo "Running CoT-UQ pipeline with:"
echo "- Model: $MODEL_ENGINE"
echo "- Dataset: $DATASET"
echo "- Samples: $TEST_START to $TEST_END"

# Create output directory
mkdir -p ../../${OUTPUT_PATH}

# Run the pipeline
singularity exec --nv "$SIF_FILE" bash -c "
    cd /app/CoT-UQ &&
    export PYTHONPATH=./ &&
    python inference_refining.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path ${OUTPUT_PATH} --try_times ${TRY_TIMES} --test_start ${TEST_START} --test_end ${TEST_END} &&
    python stepuq.py --dataset ${DATASET} --uq_engine ${UQ_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path ${OUTPUT_PATH} --try_times ${TRY_TIMES} --test_start ${TEST_START} --test_end ${TEST_END}
"

echo "Job completed!"