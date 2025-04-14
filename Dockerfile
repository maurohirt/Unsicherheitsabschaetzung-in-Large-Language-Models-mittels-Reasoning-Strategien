FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Kopiere requirements.txt
COPY requirements.txt .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# NLTK Daten
RUN python -c "import nltk; nltk.download('punkt')"

# Setze Arbeitsverzeichnis
WORKDIR /app/CoT-UQ

# Container bleibt offen für Kommandos
CMD ["/bin/bash"]