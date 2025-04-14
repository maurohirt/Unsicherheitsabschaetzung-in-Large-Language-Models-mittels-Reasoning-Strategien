#!/bin/bash

#====== SLURM Optionen ======
#SBATCH --job-name=container_test
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err
#SBATCH --partition=performance
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

echo "Test-Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

# Pfade definieren
HOST_PROJECT_DIR=$(pwd)
CONTAINER_PATH="./containers/cot_uq_latest.sif"  # Passe an, wenn ein anderer Dateiname verwendet wurde
CONTAINER_PROJECT_DIR="/app/CoT-UQ"  # Container-Pfad aus Dockerfile

# Prüfen, ob Container existiert
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container nicht gefunden: ${CONTAINER_PATH}"
    echo "Bitte zuerst pull_test_container.sbatch ausführen"
    exit 1
fi

echo "Container gefunden: ${CONTAINER_PATH}"

# Teste Python-Setup im Container
echo "Teste Python-Imports..."
singularity exec \
    --bind "${HOST_PROJECT_DIR}":"${CONTAINER_PROJECT_DIR}" \
    "${CONTAINER_PATH}" \
    python -c "
import sys
import torch
import transformers
import nltk
import os

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('Transformers version:', transformers.__version__)
print('Current directory:', os.getcwd())
print('Directory contents:', os.listdir('.'))
"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Python-Test erfolgreich abgeschlossen."
    
    # Optionaler Test: Führe eine kleine Funktion aus dem Projekt aus
    echo "Teste test_basic_setup.py (falls vorhanden)..."
    if [ -f "${HOST_PROJECT_DIR}/test_basic_setup.py" ]; then
        singularity exec \
            --bind "${HOST_PROJECT_DIR}":"${CONTAINER_PROJECT_DIR}" \
            "${CONTAINER_PATH}" \
            python "${CONTAINER_PROJECT_DIR}/test_basic_setup.py"
    else
        echo "test_basic_setup.py nicht gefunden, überspringe diesen Test."
    fi
else
    echo "Python-Test fehlgeschlagen mit Exit-Code ${EXIT_CODE}"
fi

echo "Job beendet am: $(date)"
exit ${EXIT_CODE}