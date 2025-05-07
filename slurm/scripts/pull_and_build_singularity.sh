#!/bin/bash
#SBATCH --job-name=build_cotuq_sif
#SBATCH --partition=performance     # oder ein spezieller „build“-Partition
#SBATCH --gres=gpu:0                # GPU nicht nötig
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_sif_%j.out

module load singularity

# damit große Downloads/Build-Temporaries nicht ins HOME gehen
export SINGULARITY_CACHEDIR=/tmp/$USER-singularity-cache
mkdir -p "$SINGULARITY_CACHEDIR"

# Zielpfad für die fertige SIF
SIF=$HOME/containers/cot-uq_latest.sif
mkdir -p "$(dirname "$SIF")"

# --- Variante A: Pull direkt vom Docker-Registry (schnell) ---
if [ ! -f "$SIF" ]; then
  echo "🔽 Pulling SIF from Docker Hub → $SIF"
  singularity pull "$SIF" docker://docker.io/maurohirtfhnw/cot-uq:latest
else
  echo "ℹ️  SIF already exists: $SIF"
fi

# --- Variante B: (statt Pull) Build aus lokalem Dockerfile ---
# Uncomment diesen Block, falls ihr aus eurem eigenen Dockerfile bauen wollt.
#
# if [ ! -f "$SIF" ]; then
#   echo "🛠️  Building SIF from Dockerfile → $SIF"
#   singularity build --fakeroot "$SIF" ~/cot-uq/Dockerfile
# fi

echo "✅ SIF ready at $SIF"