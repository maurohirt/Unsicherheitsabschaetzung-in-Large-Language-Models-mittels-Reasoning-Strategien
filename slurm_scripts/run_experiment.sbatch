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

#====== Pfade definieren ======
HOST_PROJECT_DIR=$(pwd)
HOST_SIF_PATH="${HOST_PROJECT_DIR}/containers/cot_uq_env.sif"
CONTAINER_PROJECT_DIR="/project"

echo "Host Projekt Pfad: ${HOST_PROJECT_DIR}"
echo "Container Image: ${HOST_SIF_PATH}"

# Check ob Container existiert
if [ ! -f "${HOST_SIF_PATH}" ]; then
    echo "ERROR: Container nicht gefunden: ${HOST_SIF_PATH}"
    echo "Bitte zuerst build_container.sbatch ausführen"
    exit 1
fi

#====== Ausführung im Container ======
echo "Starte Singularity Container mit GPU-Unterstützung..."

# Run llama pipeline mit GPU-Unterstützung
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":"${CONTAINER_PROJECT_DIR}" \
    "${HOST_SIF_PATH}" \
    bash "${CONTAINER_PROJECT_DIR}/run_llama_pipeline.sh"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Experiment finished successfully."
else
    echo "Experiment failed with exit code ${EXIT_CODE}."
fi

echo "Job beendet am: $(date)"
exit ${EXIT_CODE}