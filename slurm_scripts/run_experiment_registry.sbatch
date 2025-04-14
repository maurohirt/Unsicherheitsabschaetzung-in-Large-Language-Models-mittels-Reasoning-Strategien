#!/bin/bash

#====== SLURM Optionen für GPU-Job ======
#SBATCH --job-name=cot_uq_experiment
#SBATCH --output=slurm-run-%j.out
#SBATCH --error=slurm-run-%j.err
#SBATCH --partition=performance       # GPU-Partition anpassen falls nötig
#SBATCH --gpus=1                      # 1 GPU anfordern
#SBATCH --cpus-per-task=4             # 4 CPU Kerne
#SBATCH --mem=16G                     # 16 GB RAM
#SBATCH --time=04:00:00               # 4 Stunden Laufzeit

#====== Umgebung ======
echo "Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)"
echo "Aktuelles Verzeichnis: $(pwd)"

# GPU Information anzeigen
nvidia-smi

#====== Container-Konfiguration ======
# Ändere diese Variablen entsprechend deinem Setup
REGISTRY_URL="registry.gitlab.com"
IMAGE_PATH="your-group/your-project/cot-uq"
IMAGE_TAG="latest"  # Oder spezifischer Tag wie "v1.0.0"

CONTAINER_NAME="${IMAGE_PATH##*/}_${IMAGE_TAG}.sif"
CONTAINER_PATH="./containers/${CONTAINER_NAME}"

#====== Pfade definieren ======
HOST_PROJECT_DIR=$(pwd)
CONTAINER_PROJECT_DIR="/app/CoT-UQ"  # Pfad im Container, siehe Dockerfile

# Container-Verzeichnis erstellen falls nötig
mkdir -p containers

echo "Host Projekt Pfad: ${HOST_PROJECT_DIR}"
echo "Container wird nach Bedarf heruntergeladen: ${CONTAINER_PATH}"

#====== Container ziehen falls notwendig ======
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "Container wird heruntergeladen..."
    singularity pull --docker-login ${CONTAINER_PATH} docker://${REGISTRY_URL}/${IMAGE_PATH}:${IMAGE_TAG}
    
    if [ $? -ne 0 ]; then
        echo "Fehler beim Herunterladen des Containers!"
        exit 1
    fi
fi

#====== Ausführung im Container ======
echo "Starte Singularity Container mit GPU-Unterstützung..."

# Run llama pipeline mit GPU-Unterstützung
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":"${CONTAINER_PROJECT_DIR}" \
    "${CONTAINER_PATH}" \
    bash "${CONTAINER_PROJECT_DIR}/run_llama_pipeline.sh"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Experiment finished successfully."
else
    echo "Experiment failed with exit code ${EXIT_CODE}."
fi

echo "Job beendet am: $(date)"
exit ${EXIT_CODE}