#!/bin/bash

#====== SLURM Optionen (Minimal für CPU-Test) ======
#SBATCH --job-name=simple_container_test # Name des Jobs
#SBATCH --output=slurm_logs/%x_%j.out # Ausgabe-Datei
#SBATCH --error=slurm_logs/%x_%j.err  # Fehler-Datei
#SBATCH --partition=performance              # *** ANPASSEN: Name einer CPU-Partition auf deinem Cluster! ***
#SBATCH --cpus-per-task=1             # 1 CPU-Kern sollte reichen
#SBATCH --mem=1G                      # 1 GB Speicher sollte reichen
#SBATCH --time=00:05:00               # 5 Minuten sollten locker reichen


#====== Umgebung ======
echo "Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)"
echo "Aktuelles Verzeichnis: $(pwd)" # Zeigt, wo sbatch ausgeführt wurde

# Evtl. Singularity-Modul laden (falls auf deinem Cluster nötig)
# module load singularity

#====== Pfade definieren (HOST = auf dem Cluster-Dateisystem) ======
# ANPASSEN an deine Pfade!
HOST_HOME="/home/mauro.hirt"
HOST_PROJECT_DIR="${HOST_HOME}" # Haupt-Projektordner
HOST_SIF_PATH="${HOST_PROJECT_DIR}/containers/cot_uq_env.sif" # Pfad zum SIF Image

#====== Container-Pfade definieren (Wie sie IM Container heissen sollen) ======
# Wähle EINEN Mountpoint für den Code. /project ist oft neutraler.
# Wenn du /opt/CoT-UQ nimmst, passt es zu deinem PYTHONPATH in der .def-Datei.
CONTAINER_PROJECT_DIR="/project"
# CONTAINER_PROJECT_DIR="/opt/CoT-UQ" # Alternative

echo "Host Projekt Pfad: ${HOST_PROJECT_DIR}"
echo "Container Image: ${HOST_SIF_PATH}"
echo "Projekt wird nach ${CONTAINER_PROJECT_DIR} im Container gemountet."

#====== Ausführung im Container ======
echo "Starte Singularity Container (CPU only)..."

singularity exec \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_PROJECT_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_PROJECT_DIR}/test_basic_setup.py" # Führe das neue Test-Skript aus

# Überprüfe den Exit-Code des Python-Skripts
EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Python script finished successfully."
else
    echo "Python script failed with exit code ${EXIT_CODE}."
fi

echo "Job beendet am: $(date)"

exit ${EXIT_CODE} # Beende den Slurm-Job mit dem gleichen Exit-Code