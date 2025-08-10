# -------------------- CUDA 11.7.1 BASE IMAGE for PyTorch 2.0 --------------------
    FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

    # Avoid interactive prompts during package installation
    ARG DEBIAN_FRONTEND=noninteractive
    ENV DEBIAN_FRONTEND=noninteractive \
        TZ=Europe/Zurich
    
    # -------------------- Minimal System Setup --------------------------------
    RUN apt-get update && \
        apt-get install -y software-properties-common curl && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            git \
            build-essential \
            tzdata \
            python3.9 \
            python3.9-distutils \
            python3.9-venv && \
        python3.9 -m ensurepip && \
        python3.9 -m pip install --upgrade pip && \
        ln -sf /usr/bin/python3.9 /usr/bin/python && \
        ln -sf /usr/local/bin/pip /usr/bin/pip && \
        rm -rf /var/lib/apt/lists/*
    
    # -------------------- Python Dependencies ----------------------------------
    WORKDIR /app
    COPY requirements.txt .
    
    # Install PyTorch 2.2.2 wheels (CUDA 11.8)
    RUN python3.9 -m pip install torch==2.2.2+cu118 \
        torchvision==0.17.2+cu118 \
        torchaudio==2.2.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118
    
    # Install Transformers and other dependencies
    RUN python3.9 -m pip install --no-cache-dir -r requirements.txt
    
    # -------------------- Project Code -----------------------------------------
    ENV PYTHONPATH=/app/CoT-UQ:$PYTHONPATH
    WORKDIR /app/CoT-UQ
    
    CMD ["/bin/bash"]
    