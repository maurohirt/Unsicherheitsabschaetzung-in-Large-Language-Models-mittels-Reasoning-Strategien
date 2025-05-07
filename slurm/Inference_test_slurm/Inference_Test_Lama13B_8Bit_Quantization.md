# Inference-Test für LLaMA 2 13B (8-bit quantisiert) mit Singularity und SLURM

## 🎯 Ziel

Ein reproduzierbarer Inference-Workflow für das lokal gespeicherte LLaMA‑2‑13B-Modell unter Verwendung von bitsandbytes 8‑bit Quantisierung im Offline‑Modus (kein Internetzugriff im Cluster). Zusätzlich wird das GPU‑Speicher‑Logging vor und nach wichtigen Schritten protokolliert.

---

## 📦 Voraussetzungen

* **SLURM-Partition**: `performance`
* **GPU**: 1 (z. B. A4500 mit 20 GB VRAM)
* **CPU**: 8 Threads
* **RAM**: used 64 GB (can be lowered to about 8GB)
* **Singularity-Image**: `cot-uq_latest.sif` mit vorinstallierten Python‑Paketen (`transformers`, `accelerate`, `torch`)
* **bitsandbytes** (wird im Container installiert)

---

## 🔧 Pfad-Setup (Host)

```bash
export HOST_HF_CACHE="/home2/mauro.hirt/hf-cache"
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"

# Optional: Verzeichnisse prüfen
ds -ld "${HOST_HF_CACHE}/models/llama2-13b"
ls -l "${SIF_PATH}"
```

---

## 🚀 Singularity-Ausführung

Leitet das gesamte Skript via stdin in den Container. Das Skript führt folgende Schritte aus:

1. **Offline-Modus aktivieren** (`TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`)
2. **bitsandbytes** installieren
3. **GPU‑Speicher vor Modell-Ladung** mit `nvidia-smi` loggen
4. **Laden des Tokenizers** und Modell mit 8‑bit Quantisierung (`BitsAndBytesConfig`)
5. **GPU-Speicher nach Modell-Ladung** und **vor/nach Token-Generierung** per `torch.cuda.memory_*` loggen
6. **Ausgabe** des generierten Textes
7. **GPU‑Speicher nach Container-Exit** loggen

```bash
singularity exec --nv \
  -B "${HOST_HF_CACHE}:${CONTAINER_HF_CACHE}" \
  "${SIF_PATH}" \
  bash << 'EOF'
set -e

# Environment
export HF_HOME="${CONTAINER_HF_CACHE}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 8-bit Quantisierung
pip install --no-cache-dir bitsandbytes

# Modellpfad
export MODEL_PATH_INSIDE="/root/.cache/huggingface/models/llama2-13b"
ls -ld "$MODEL_PATH_INSIDE"

# GPU-Speicher vor Modell-Ladung
echo "[INFO] GPU-Speicher vor Modell-Ladung:" >&2
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits >&2

# Python-Inferenz
python3 << 'PYCODE'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging, os, sys

logging.basicConfig(level=logging.INFO)
model_id = os.environ['MODEL_PATH_INSIDE']

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

# Speicherlog nach Modell-Ladung
print(f"[INFO] Nach Modell-Ladung: Allocated={{torch.cuda.memory_allocated()/1e9:.2f}} GB, Max Allocated={{torch.cuda.max_memory_allocated()/1e9:.2f}} GB", file=sys.stderr)

device = model.device
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(device)

# Speicherlog vor Generierung
print(f"[INFO] Vor Generierung: Allocated={{torch.cuda.memory_allocated()/1e9:.2f}} GB, Reserved={{torch.cuda.memory_reserved()/1e9:.2f}} GB", file=sys.stderr)

outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

# Speicherlog nach Generierung
print(f"[INFO] Nach Generierung: Allocated={{torch.cuda.memory_allocated()/1e9:.2f}} GB, Reserved={{torch.cuda.memory_reserved()/1e9:.2f}} GB", file=sys.stderr)

# Ausgabe
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
PYCODE

# GPU-Speicher nach Exit
echo "[INFO] GPU-Speicher nach Container-Exit:" >&2
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits >&2
EOF
```

---

## 📊 Erwartetes Logging

```text
[INFO] GPU-Speicher vor Modell-Ladung:
20470, 2
...
[INFO] Nach Modell-Ladung: Allocated=13.37 GB, Max Allocated=13.55 GB
[INFO] Vor Generierung: Allocated=13.37 GB, Reserved=14.23 GB
[INFO] Nach Generierung: Allocated=13.38 GB, Reserved=14.25 GB
...
[INFO] GPU-Speicher nach Container-Exit:
20470, 2
```

---

### 🚩 Hinweise

* Passe `max_new_tokens`, `device_map` oder `bnb_config`-Parameter je nach Bedarf an.
* Bei Speicherausfällen (`OOM`) kann das Setzen von `max_memory` in der Accelerate-Konfiguration helfen.
