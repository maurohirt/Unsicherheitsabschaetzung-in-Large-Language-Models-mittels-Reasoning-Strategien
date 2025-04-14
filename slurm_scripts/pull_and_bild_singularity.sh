#!/bin/bash
#SBATCH --job-name=singularity_pull_test
#SBATCH --output=../../outputs/singularity_pull_test_%j.log
#SBATCH --error=../../outputs/singularity_pull_test_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=performance
#SBATCH --mem=32G  # Erhöhe dies auf einen Wert, der ausreichend ist

module load singularity

DOCKER_REGISTRY="docker.io"
IMAGE_NAME="maurohirtfhnw/cot-uq"
IMAGE_TAG="latest"
DOCKER_IMAGE="docker://${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
SIF_FILE="../../containers/cot-uq_${IMAGE_TAG}.sif"

echo "Pulling Singularity image from ${DOCKER_IMAGE} ..."
singularity pull "$SIF_FILE" "$DOCKER_IMAGE"
if [ $? -ne 0 ]; then
    echo "Fehler: Singularity pull schlug fehl!"
    exit 1
fi

echo "Image erfolgreich heruntergeladen als ${SIF_FILE}."
echo "Starte Testbefehl im Container ..."
singularity exec "$SIF_FILE" python -c 'import torch; print("PyTorch version:", torch.__version__)'
echo "Pull- und Exec-Test abgeschlossen."