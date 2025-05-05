# 🔪 Inference-Test mit LLaMA 3 8B auf SLURM (Singularity)

## ✅ Ziel

Ein erfolgreich replizierbarer Inference-Test mit einem lokal gespeicherten LLaMA-3-8B-Modell unter Verwendung von Singularity-Containern und Offline-Modus (kein Internetzugriff im Cluster).

---

## 📦 Ressourcen-Anforderungen (SLURM)

```bash
srun -p performance \
     --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=00:20:00 \
     --pty bash -l
```

* `gpu:1`: genau eine GPU (z. B. A4500 mit 20 GB VRAM)
* `mem=64G`: sicherer Wert, evtl. testweise reduzierbar auf 48G
* `cpus-per-task=8`: ausreichend Threads
* `partition=performance`: für GPU-Jobs

---

## 💠 Pfad-Setup (Host-Seite)

```bash
export HOST_HF_CACHE="/home2/mauro.hirt/hf-cache"
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"
```

Optional prüfen:

```bash
ls -ld "${HOST_HF_CACHE}/models/llama3-8B"
ls -l "${SIF_PATH}"
```

---

## 🚀 Singularity-Start

```bash
singularity exec --nv \
  -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
  "${SIF_PATH}" \
  bash -lc '
    export HF_HOME="${CONTAINER_HF_CACHE}"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1

    MODEL_PATH_INSIDE="/root/.cache/huggingface/models/llama3-8B"
    ls -ld "$MODEL_PATH_INSIDE" || exit 1

    python3 - <<EOF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging, os, sys

logging.basicConfig(level=logging.INFO)
model_id = "${MODEL_PATH_INSIDE}"

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    local_files_only=True
)
device = model.device
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
'
```

---

## 🔄 Änderungen gegenüber ursprünglichem Code

### 💡 Modellinitialisierung (`model_init()`)

* **Ersetzt**: `device = torch.device("cuda:0")` & `.to(device)`
* **Neu**:

  ```python
  device_map="auto"
  local_files_only=True
  torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  ```

### 📁 Pfadlogik

* Modellpfad direkt übergeben (z. B. `args.model_path` → symlink wie `llama3-8B`)
* Kein Lookup in HF\_NAMES nötig

### 💾 Offline-Modus

* `TRANSFORMERS_OFFLINE=1`
* `HF_HUB_OFFLINE=1`
* `local_files_only=True` in allen `from_pretrained(...)`-Aufrufen

---

## 📉 Ressourcenverbrauch (laut Logs)

* **GPU RAM belegt**: \~14.96 GB
* **Modelldatentyp**: `torch.bfloat16`
* **Lademethode**: `device_map="auto"` erfolgreich verwendet

---

## ✅ Fazit

Mit diesen Einstellungen funktioniert ein vollständiger Offline-Inference-Test für LLaMA 3 8B mit Singularity und SLURM zuverlässig. Die getestete Konfiguration bildet einen stabilen Startpunkt für weitere Tests, Benc
