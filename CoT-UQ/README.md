# CoT-UQ: End-to-End Pipeline README

This repository contains an end-to-end pipeline to run CoT-UQ on Llama-family models across multiple datasets on an HPC cluster using Singularity and SLURM. It also includes a Dockerfile for building and publishing a container image that can be converted into a Singularity Image File (SIF).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Containers (Docker and Singularity)](#containers-docker-and-singularity)
- [Pipeline Runners (SLURM)](#pipeline-runners-slurm)
  - [Full pipeline (single job)](#full-pipeline-single-job)
  - [Full pipeline (array job, one dataset per task)](#full-pipeline-array-job-one-dataset-per-task)
  - [Stage-by-stage runners (array jobs)](#stage-by-stage-runners-array-jobs)
- [Key Python Scripts](#key-python-scripts)
- [Configuration (YAML)](#configuration-yaml)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Environment variables](#environment-variables)
- [Notes on logging and defaults](#notes-on-logging-and-defaults)
- [Useful paths and scripts](#useful-paths-and-scripts)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- GPU-enabled HPC cluster with SLURM and Singularity/Apptainer.
- CUDA 11.8-capable nodes (the Docker image targets PyTorch 2.2.2 + cu118).
- Docker (optional) if you want to build/push the image yourself.
- Environment variables:
  - `HUGGINGFACE_HUB_TOKEN` (required for model downloads)
  - `OPENAI_API_KEY` only if you run analysis on logical QA datasets (2WikimhQA, hotpotQA).
- Outbound internet access on the cluster to pull images and HF models, or pre-provision them via a local registry/cache.

## Quickstart

1) Export required tokens before any command (both local and SLURM jobs):
```bash
export HUGGINGFACE_HUB_TOKEN="<your_hf_token>"
# Required only for analysis on logical QA datasets (2WikimhQA, hotpotQA)
export OPENAI_API_KEY="<your_openai_api_key>"
```

2) Pull/create the Singularity image on the cluster via SLURM:
```bash
sbatch slurm/scripts/pull_and_build_singularity.sbatch
```

3) Run the full pipeline (inference → UQ → analysis) using a YAML config:
```bash
# Minimal test
sbatch slurm/scripts/run_full_pipeline.sbatch configs/pipeline_config_minimal.yaml

# Full run (example)
sbatch slurm/scripts/run_full_pipeline.sbatch configs/pipeline_config_full_lama3.yaml
```

4) Or run the array version to parallelize per-dataset:
```bash
# Processes one dataset per array task based on the config's datasets list
sbatch slurm/scripts/run_full_pipeline_array.sbatch configs/pipeline_config_full_lama3.yaml

```

## Containers (Docker and Singularity)

### Build Docker image locally (optional)
Use the root `Dockerfile` to build a CUDA-enabled image with PyTorch 2.2.2 (CUDA 11.8):
```bash
# From repository root
docker build -t maurohirtfhnw/cot-uq:latest .
# Optional: push to Docker Hub
# docker push maurohirtfhnw/cot-uq:latest
```

### Pull Singularity image on the cluster (recommended)
The SLURM helper script pulls a SIF from Docker Hub and stores it under `$HOME/containers/cot-uq_latest.sif`:
- Script: `slurm/scripts/pull_and_build_singularity.sbatch`
- It uses: `singularity pull docker://docker.io/maurohirtfhnw/cot-uq:latest`

Run it as:
```bash
sbatch slurm/scripts/pull_and_build_singularity.sbatch
```


## Pipeline Runners (SLURM)

### Full pipeline (single job)
- Script: `slurm/scripts/run_full_pipeline.sbatch`
- Reads YAML config and runs, per dataset:
  - `inference_refining.py` → `stepuq.py` (for each UQ method) → `analyze_result.py`
- Pre-downloads HF models using `huggingface_hub.snapshot_download`
- Logging is cleaned to focus on progress/results and ignore known noisy warnings

Usage:
```bash
# With default config path inside the script
sbatch slurm/scripts/run_full_pipeline.sbatch

# With explicit config file
sbatch slurm/scripts/run_full_pipeline.sbatch configs/pipeline_config_minimal.yaml
```

### Full pipeline (array job, one dataset per task)
- Script: `slurm/scripts/run_full_pipeline_array.sbatch`
- Same behavior as single-job runner, but processes only the dataset corresponding to the array index.

Examples:
```bash
# Use the #SBATCH --array declared in the script (e.g., 0-4)
sbatch slurm/scripts/run_full_pipeline_array.sbatch configs/pipeline_config_full_lama3.yaml

# Override to a specific dataset index at submit time (e.g., gsm8k = 4)
sbatch --array=4 slurm/scripts/run_full_pipeline_array.sbatch configs/pipeline_config_full_lama3.yaml
```

Dataset index mapping used by the array scripts:
```
0 → ASDiv
1 → 2WikimhQA
2 → hotpotQA
3 → svamp
4 → gsm8k
```

### Stage-by-stage runners (array jobs)
Run individual pipeline stages as separate array jobs. These expect prior stage outputs to exist.

1) Inference only
- Script: `slurm/scripts/inference_refining_llama8B_datasets_array.sbatch`
- Default model: `llama3-1_8B` (change in the script if needed)
- Try times: uses `--try_times` from CLI (typically 20)

Commands:
```bash
# All datasets declared in the script's array
sbatch slurm/scripts/inference_refining_llama8B_datasets_array.sbatch

# Only one dataset by index (e.g., hotpotQA = 2)
sbatch --array=2 slurm/scripts/inference_refining_llama8B_datasets_array.sbatch
```

2) UQ metrics only (baselines and variants)
- Script: `slurm/scripts/all_metrics_stepuq_array.sbatch`
- Pass model engine as first argument (defaults to llama3-1_8B)
- Internally runs `stepuq.py` for each selected `UQ_METHODS`

Commands:
```bash
# Run for all datasets, with default model
a. sbatch slurm/scripts/all_metrics_stepuq_array.sbatch
# Or specify model explicitly
b. sbatch slurm/scripts/all_metrics_stepuq_array.sbatch llama3-1_8B
# Single dataset (e.g., svamp = 3)
c. sbatch --array=3 slurm/scripts/all_metrics_stepuq_array.sbatch llama3-1_8B
```

3) Analysis only
- Script: `slurm/scripts/analyze_results_array.sbatch`
- Requires `OPENAI_API_KEY` only for 2WikimhQA/hotpotQA
- Runs `analyze_result.py` per UQ method and computes AUROC

Commands:
```bash
# Run for all datasets, with default model
sbatch slurm/scripts/analyze_results_array.sbatch
# Specific dataset (e.g., 2WikimhQA = 1)
sbatch --array=1 slurm/scripts/analyze_results_array.sbatch llama3-1_8B
```


## Key Python Scripts

- `inference_refining.py`
  - Loads dataset and Llama model; performs CoD/CoT prompting to produce reasoning traces and final answers.
  - Saves `output_v1.json` with: answer-token probabilities, step-wise keywords, keyword probabilities and contributions, COT token IDs/probabilities, per-token entropies/confidences, etc.
  - Uses CLI flags from `config.py` (e.g., `--dataset`, `--model_engine`, `--model_path`, `--temperature`, `--try_times`, `--max_length_cot`, `--test_start`, `--test_end`).
  - Typical `--try_times` is 20.

- `stepuq.py`
  - Computes uncertainty with multiple engines:
    - AP: `probas-mean`, `probas-min`, `token-sar` (+ `-bl` baselines, and `*-alltokens` variants)
    - P(True): `p-true-*` family (prompts an LLM for answer correctness probability)
    - Self-Probing: `self-probing-*` family
  - For speed, pipeline runners invoke it with `--try_times 5` by design.
  - Writes `confidences/output_v1_<uq_engine>.json` under the dataset output path.

- `analyze_result.py`
  - Labels predictions vs ground truth, then computes AUROC over confidences.
  - Numeric datasets (e.g., gsm8k) use string/number matching. Logical QA (2WikimhQA, hotpotQA) uses OpenAI (`gpt-4o-mini`) via `OPENAI_API_KEY`.
  - Produces/updates `output_v1_w_labels.json` and prints AUROC.

## Configuration (YAML)

The full pipeline runners read a YAML config. See examples in `configs/`:
- `configs/pipeline_config_minimal.yaml` (single dataset, small slice)
- `configs/pipeline_config_full_lama3.yaml` (full run, Llama3.1-8B)
- `configs/pipeline_config_full_lama3_cod.yaml` (variant)

Typical structure:
```yaml
model_engine: "llama3-1_8B"  # or "llama2-13b"
datasets:
  - "ASDiv"
  - "2WikimhQA"
  - "hotpotQA"
  - "svamp"
  - "gsm8k"
output_path_base: "/home2/<user>/CoT-UQ/output"
temperature: 1.0
try_times: 20
max_length_cot: 128
# Optional subrange for testing
test_start: "0"
test_end: "full"
# UQ methods (choose any supported in config.py)
uq_methods:
  - "probas-mean-bl"
  - "probas-min-bl"
  - "token-sar-bl"
```


## Outputs

For each dataset under `<output_path_base>/<model_engine>/<dataset>/`:
- `output_v1.json` from inference
- `confidences/output_v1_<uq_engine>.json` from UQ
- `output_v1_w_labels.json` from analysis
- Logs under `output/logs/...` per SLURM script


## Requirements

Pinned dependencies used by the image and pipeline (from `requirements.txt`):
```txt
torch==2.2.2+cu118 # needs torch.compiler submodule for flex_attention
--extra-index-url https://download.pytorch.org/whl/cu118
transformers>=4.45.0
accelerate>=0.25.0
safetensors>=0.4.0
sentencepiece>=0.1.99
torchmetrics>=1.0.0
openai>=1.6.0

# Existing dependencies
peft>=0.4.0
trl>=0.4.7
jieba
rouge-chinese
nltk
gradio>=3.36.0
uvicorn
pydantic==1.10.11
fastapi==0.95.1
sse-starlette
sentence_transformers
datasets>=2.12.0
```


## Environment variables

- `HUGGINGFACE_HUB_TOKEN` (required): used to download HF models via `huggingface_hub.snapshot_download`
- `OPENAI_API_KEY` (conditionally required): only for analyzing `2WikimhQA` and `hotpotQA`

The SLURM scripts propagate these into the container. If `OPENAI_API_KEY` is missing, analysis is skipped for those datasets.


## Notes on logging and defaults

- `inference_refining.py` uses full config
- `stepuq.py` is invoked with `--try_times=5` for speed; includes `--temperature`.
- `analyze_result.py` does not use `try_times`.
- SLURM pipeline scripts filter noisy logs (duplicate experiment args, known CrossEncoder warnings, verbose parameter dumps) to keep output focused on progress and results.


## Useful paths and scripts

- Dockerfile: `Dockerfile`
- Pipeline runners: `slurm/scripts/run_full_pipeline.sbatch`, `slurm/scripts/run_full_pipeline_array.sbatch`
- Stage runners: `slurm/scripts/inference_refining_llama8B_datasets_array.sbatch`, `slurm/scripts/all_metrics_stepuq_array.sbatch`, `slurm/scripts/analyze_results_array.sbatch`
- Singularity pull helper: `slurm/scripts/pull_and_build_singularity.sbatch`
- Additional doc: `slurm/scripts/README_pipeline.md`


## Troubleshooting

- Missing HF token: set `HUGGINGFACE_HUB_TOKEN` and resubmit. The pipeline ensures HF models are cached prior to running.
- Analysis errors on 2WikimhQA/hotpotQA: ensure `OPENAI_API_KEY` is exported; the array analysis script validates this.
- Array indices: verify the dataset order shown above; you can override the `--array` range when submitting.
- GPU memory: scripts print `nvidia-smi` summaries; use fewer concurrent array tasks if constrained.

