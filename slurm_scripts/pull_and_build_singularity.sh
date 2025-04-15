#!/bin/bash
#SBATCH --job-name=singularity_pull_test
#SBATCH --output=$HOME/../../outputs/singularity_pull_test_%j.log
#SBATCH --error=$HOME/../../outputs/singularity_pull_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=performance
#SBATCH --mem=32G

module load singularity

# Safe Container Output
SIF_FILE="$HOME/containers/cot-uq_latest.sif"
mkdir -p "$(dirname "$SIF_FILE")"

# Optional: Use clean cache location
export SINGULARITY_CACHEDIR=/tmp/$USER-singularity-cache
mkdir -p "$SINGULARITY_CACHEDIR"

# Pull only if image does not exist
DOCKER_IMAGE="docker://docker.io/maurohirtfhnw/cot-uq:latest"

if [ ! -f "$SIF_FILE" ]; then
    echo "Singularity image not found, pulling: $DOCKER_IMAGE → $SIF_FILE"
    singularity pull "$SIF_FILE" "$DOCKER_IMAGE"
    if [ $? -ne 0 ]; then
        echo "❌ Fehler: Singularity pull schlug fehl!"
        exit 1
    fi
    echo "✅ Image erfolgreich gespeichert unter: $SIF_FILE"
else
    echo "✅ Image bereits vorhanden: $SIF_FILE"
fi

# Run test
echo "Starte Container-Test mit PyTorch..."
singularity exec --nv "$SIF_FILE" python -c 'import torch; print("PyTorch version:", torch.__version__)'