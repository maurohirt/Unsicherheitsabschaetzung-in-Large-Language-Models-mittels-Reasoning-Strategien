#!/bin/bash
#SBATCH --job-name=cot_uq_inference
#SBATCH --output=$HOME/../../outputs/cot_uq_inference_%j.log
#SBATCH --error=$HOME/../../outputs/cot_uq_inference_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=performance
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Load singularity
module load singularity

# Pfade
PROJECT_DIR="$HOME/cot-uq"
CONTAINER_DIR="$HOME/containers"
SIF_FILE="$CONTAINER_DIR/cot-uq_latest.sif"
OUTPUT_PATH="$HOME/outputs/llama3-1_8B_gsm8k"
MODEL_LOCAL_PATH="$HOME/models/Llama-3.1-8B"  # Absoluter Pfad zum lokalen Modell

# Singularity Pull-Konfiguration
export SINGULARITY_CACHEDIR=/tmp/$USER-singularity-cache
mkdir -p "$SINGULARITY_CACHEDIR" "$CONTAINER_DIR" "$OUTPUT_PATH"

# Container-Infos
DOCKER_IMAGE="docker://docker.io/maurohirtfhnw/cot-uq:latest"

# Container ziehen, falls nicht vorhanden
if [ ! -f "$SIF_FILE" ]; then
    echo "🔄 Container nicht gefunden – pulling $DOCKER_IMAGE"
    singularity pull "$SIF_FILE" "$DOCKER_IMAGE"
    if [ $? -ne 0 ]; then
        echo "❌ Fehler: Singularity pull schlug fehl!"
        exit 1
    fi
    echo "✅ Container gespeichert unter: $SIF_FILE"
else
    echo "✅ Container bereits vorhanden: $SIF_FILE"
fi

# Pipeline-Parameter
MODEL_ENGINE="llama3-1_8B"
DATASET="gsm8k"
TEMP=1.0
TRY_TIMES=5
TEST_START=0
TEST_END=5  # Reduziert auf 5 für ersten Test

echo "▶️ Starte Inference-Refining"
echo "- Model:     $MODEL_ENGINE (lokaler Pfad: $MODEL_LOCAL_PATH)"
echo "- Dataset:   $DATASET"
echo "- Samples:   $TEST_START to $TEST_END"
echo "- Output to: $OUTPUT_PATH"

# Nur Inference-Refining ausführen mit lokalem Modellpfad
singularity exec --nv --no-home --containall \
  --bind "$PROJECT_DIR:/app" \
  --bind "$HOME/models:/app/models" \
  --bind "$OUTPUT_PATH:/app/output/${MODEL_ENGINE}_${DATASET}" \
  "$SIF_FILE" bash -c "
    cd /app &&
    export PYTHONPATH=./ &&
    mkdir -p /app/output/${MODEL_ENGINE}_${DATASET} &&
    
    # Set the absolute path to the model within the container
    MODEL_PATH_IN_CONTAINER=\"/app/models/Llama-3.1-8B\"
    echo \"Using model at container path: \${MODEL_PATH_IN_CONTAINER}\"
    
    # Run with absolute path to model
    python inference_refining.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} --model_path \${MODEL_PATH_IN_CONTAINER} --temperature ${TEMP} --output_path output/${MODEL_ENGINE}_${DATASET} --try_times ${TRY_TIMES} --test_start ${TEST_START} --test_end ${TEST_END}
"

echo "✅ Job abgeschlossen."