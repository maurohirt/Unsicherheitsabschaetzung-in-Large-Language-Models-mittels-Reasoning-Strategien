#!/bin/bash

#====== SLURM Optionen ======
#SBATCH --job-name=pull_container
#SBATCH --output=slurm-pull-%j.out
#SBATCH --error=slurm-pull-%j.err
#SBATCH --partition=performance
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00

echo "Container-Pull Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

# Erstelle Verzeichnis für Container
mkdir -p containers

# GitLab Registry Konfiguration
REGISTRY_URL="registry.gitlab.com"
IMAGE_PATH="your-group/your-project/cot-uq"  # Passe an dein GitLab Repo an
IMAGE_TAG="latest"

CONTAINER_PATH="./containers/cot_uq_${IMAGE_TAG}.sif"

echo "Pulling container from: ${REGISTRY_URL}/${IMAGE_PATH}:${IMAGE_TAG}"
echo "Saving to: ${CONTAINER_PATH}"

# Pull von GitLab Registry mit Authentifizierung
# Option 1: Mit Docker-Login (falls Singularity das unterstützt)
singularity pull --docker-login ${CONTAINER_PATH} docker://${REGISTRY_URL}/${IMAGE_PATH}:${IMAGE_TAG}

# Option 2: Ohne Login für öffentliche Images
# singularity pull ${CONTAINER_PATH} docker://${REGISTRY_URL}/${IMAGE_PATH}:${IMAGE_TAG}

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Container erfolgreich heruntergeladen: ${CONTAINER_PATH}"
    
    # Einfacher Test des Containers
    echo "Teste Container mit Python-Import..."
    singularity exec ${CONTAINER_PATH} python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
    
    # Zeige Container-Info an
    echo "Container Info:"
    singularity inspect ${CONTAINER_PATH}
else
    echo "FEHLER beim Herunterladen des Containers! Exit code: ${EXIT_CODE}"
fi

echo "Job beendet am: $(date)"
exit ${EXIT_CODE}