#!/bin/bash -l
# Use login shell to potentially source profile files

#====== SLURM Optionen für den Build-Job ======
#SBATCH --job-name=build_singularity   # Name des Jobs
#SBATCH --output=slurm-build-%j.out
#SBATCH --error=slurm-build-%j.err  # Fehler-Datei
#SBATCH --partition=performance        # CPU-Partition
#SBATCH --cpus-per-task=2              # 2 CPUs sollten reichen
#SBATCH --mem=8G                       # 8 GB Speicher
#SBATCH --time=00:30:00                # 30 Minuten

#====== Umgebung ======
echo "Build-Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

# Erstelle Verzeichnisse falls nötig
mkdir -p containers

# OPTION 1: Docker Image ziehen (aus dem Cheatsheet)
# CUDA 11.8 mit PyTorch Container verwenden
echo "Pulling Docker image to Singularity container..."
singularity pull containers/cot_uq_env.sif docker://pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Aktuelles Arbeitsverzeichnis
WORK_DIR=$(pwd)

# Requirements in Container kopieren
echo "Creating setup script for requirements..."
cat > setup_container.sh << 'EOL'
#!/bin/bash
# Install additional requirements from the project
pip install -r /requirements.txt
python -c "import nltk; nltk.download('punkt')"
EOL

chmod +x setup_container.sh

# Container mit Requirements ausführen
echo "Installing requirements inside container..."
singularity exec --bind ${WORK_DIR}/requirements.txt:/requirements.txt containers/cot_uq_env.sif ./setup_container.sh

# Überprüfe den Exit-Code
EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Singularity container successfully built and set up: containers/cot_uq_env.sif"
else
    echo "ERROR setting up Singularity container! Exit code: ${EXIT_CODE}."
fi

echo "Build-Job beendet am: $(date)"
exit ${EXIT_CODE}