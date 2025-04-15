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

# Erstelle temporäre Datei mit angepasstem Code für die lokale Modellladung
TEMP_SCRIPT="/tmp/fixed_model_load.py"
cat > $TEMP_SCRIPT << 'EOF'
# Ersetze diese Zeile in llama2_predict.py
def model_init(args):
    model_path = args.model_path  # Wird nun als absoluter Pfad zum Modell erwartet
    device = torch.device("cuda:0")
    print(f"Loading model from path: {model_path}")
    
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_path,  # Direkter Pfad zum Modell
            torch_dtype=torch.bfloat16,
            local_files_only=True  # Erzwinge lokales Laden
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True  # Erzwinge lokales Laden
        )
        print(f"✅ Model successfully loaded from: {model_path}")
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise
EOF

# Nur Inference-Refining ausführen mit lokalem Modellpfad
singularity exec --nv --no-home --containall \
  --bind "$PROJECT_DIR:/app" \
  --bind "$HOME/models:/app/models" \
  --bind "$OUTPUT_PATH:/app/output/${MODEL_ENGINE}_${DATASET}" \
  "$SIF_FILE" bash -c "
    cd /app &&
    export PYTHONPATH=./ &&
    mkdir -p /app/output/${MODEL_ENGINE}_${DATASET} &&
    
    # Run with standard model_path parameter - our updated code will handle paths properly
    python inference_refining.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} --model_path ${MODEL_ENGINE} --temperature ${TEMP} --output_path output/${MODEL_ENGINE}_${DATASET} --try_times ${TRY_TIMES} --test_start ${TEST_START} --test_end ${TEST_END}
"

echo "✅ Job abgeschlossen."