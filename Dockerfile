# -------------------- Basis passend zu CUDA-12.8 Treiber --------------------
    FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

    # -------------------- Minimales System-Setup --------------------------------
    RUN apt-get update && \
        apt-get install -y --no-install-recommends git build-essential && \
        rm -rf /var/lib/apt/lists/*
    
    # -------------------- Python-Abhängigkeiten ---------------------------------
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # -------------------- Projekt-Code ------------------------------------------
    COPY . /app/CoT-UQ
    ENV PYTHONPATH=/app/CoT-UQ:$PYTHONPATH
    WORKDIR /app/CoT-UQ
    
    CMD ["/bin/bash"]