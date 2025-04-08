#!/bin/bash -l
# Use login shell to potentially source profile files

#====== SLURM Optionen für den Build-Job ======
#SBATCH --job-name=build_singularity   # Name des Jobs
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm_logs/%x_%j.err  # Fehler-Datei
#SBATCH --partition=performance              # *** ANPASSEN: Eine CPU-Partition wählen! ***
                                       # (Build braucht keine GPU, 'performance' könnte gehen, aber eine reine CPU-Partition ist oft besser)
#SBATCH --cpus-per-task=2              # 2 CPUs sollten reichen
#SBATCH --mem=8G                       # 8 GB Speicher (manche Builds brauchen etwas mehr)
#SBATCH --time=00:15:00                # 45 Minuten (sollte für den Build reichen)

#====== Umgebung ======
echo "Build-Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)" # Zeigt den Rechenknoten an, auf dem gebaut wird
echo "Working directory: $(pwd)" # Sollte der Ordner sein, von dem aus submitted wurde

#====== Modul laden & Bauen ======
# Explizit das Modul laden (BESTE PRAXIS!)
# Versuche 'apptainer' oder 'singularity' basierend auf deiner Recherche
echo "Versuche Modul zu laden..."
module load apptainer # ODER singularity, oder spezifische Version
# TODO: Den korrekten Modulnamen hier eintragen, sobald bekannt!
# Wenn kein Modul explizit geladen werden muss, kann die Zeile weg.

echo "Prüfe, ob Befehl gefunden wird..."
which singularity || which apptainer # Zeigt den Pfad an, wenn gefunden

# Wechsel ins Verzeichnis, wo die .def-Datei liegt (wichtig!)
# $SLURM_SUBMIT_DIR ist das Verzeichnis, von dem aus sbatch aufgerufen wurde
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Aktuelles Verzeichnis nach cd: $(pwd)"

# Pfad zur Definitionsdatei und zum Ziel-Image
DEF_FILE="singularity.def"
SIF_FILE="./containers/cot_uq_env.sif"

echo "Starte Build für ${SIF_FILE} aus ${DEF_FILE}"

# Führe den Build-Befehl aus
# Verwende singularity oder apptainer, je nachdem was verfügbar ist
singularity build --fakeroot "${SIF_FILE}" "${DEF_FILE}"
# ODER: apptainer build --fakeroot "${SIF_FILE}" "${DEF_FILE}"

# Überprüfe den Exit-Code des Build-Prozesses
EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Singularity Image erfolgreich gebaut: ${SIF_FILE}"
else
    echo "FEHLER beim Bauen des Singularity Images! Exit code: ${EXIT_CODE}."
    # Bei Fehlern: Unfertiges Image löschen, falls vorhanden? (Optional)
    # rm -f "${SIF_FILE}"
fi

echo "Build-Job beendet am: $(date)"
exit ${EXIT_CODE}