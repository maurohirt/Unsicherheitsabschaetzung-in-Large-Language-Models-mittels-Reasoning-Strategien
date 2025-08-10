# Results_Analysis

Utilities to compute, aggregate, and visualize uncertainty estimation results across datasets and runs. Includes scripts for AUROC, ECE, Brier score, and Accuracy, plus plotting utilities to build “leaderboards” and reliability diagrams.

- Python: 3.9+ recommended
- Run all commands from the repository root: `Results_Analysis/`

## Table of Contents
- [Prerequisites](#prerequisites)
- [Folder structure](#folder-structure)
- [Installation](#installation)
- [Data layout](#data-layout)
- [Configuration (YAML)](#configuration-yaml)
- [Switching between CoT and CoD paths](#switching-between-cot-and-cod-paths)
- [Quickstart](#quickstart)
- [Visualization](#visualization)
- [Notes & Troubleshooting](#notes--troubleshooting)
- [Extending](#extending)

## Prerequisites

- Python 3.9+ (recommended)
- pip installed
- Optional: a virtual environment for reproducibility

## Folder structure

```text
Results_Analysis/
├─ configs/
│  ├─ accuracy_config.yaml, auroc_config.yaml, ece_config.yaml, brier_config.yaml
│  ├─ *_cod.yaml, *_cot.yaml
│  └─ datasets.yaml, uq_methods_cot.yaml, uq_methods_cod.yaml
├─ scripts/
│  ├─ calculation/ (calculate_*.py)
│  └─ visualization/ (plot_*_leaderboard.py, plot_reliability_*.py)
├─ src/
│  ├─ data/loader.py
│  ├─ metrics/ (auroc.py, ece.py, brier.py, accuracy.py)
│  └─ utils/config_loader.py
├─ Data/
└─ results/
```

- `configs/`
  - Metric configs: `accuracy_config.yaml`, `auroc_config.yaml`, `ece_config.yaml`, `brier_config.yaml`
  - Variants: `*_cod.yaml`, `*_cot.yaml`
  - Shared: `datasets.yaml`, `uq_methods_cot.yaml`, `uq_methods_cod.yaml`
- `scripts/`
  - `calculation/`: `calculate_auroc.py`, `calculate_ece.py`, `calculate_brier.py`, `calculate_accuracy.py` (plus a focused README)
  - `visualization/`: plotting scripts (e.g., `plot_ece_leaderboard.py`, reliability plots)
- `src/`
  - `data/loader.py`: loads raw outputs and confidences per run/dataset/model
  - `metrics/`: implementations for `auroc.py`, `ece.py`, `brier.py`, `accuracy.py`
  - `utils/config_loader.py`: YAML loader with imports and Jinja2 templating; group resolution with `@group` syntax
- `Data/`: raw data expected by the loaders (see below)
- `results/`: per-run and aggregated outputs and figures

## Installation

Install with a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
```

## Data layout

The calculation scripts expect run-structured raw data, e.g. CoT (or CoD) results under `Data/CoT/raw` or `Data/CoD/raw`:

```
Data/CoT/raw/
  run_{RUN}/
    {MODEL}/
      {DATASET}/
        output_v1_w_labels.json
        confidences/
          output_v1_{METRIC}.json
```

- `output_v1_w_labels.json` (JSONL):
  - `question`: input text
  - `llm answer`: model output
  - `correct answer`: ground truth
  - `label`: boolean (True if correct)
- `confidences/output_v1_{METRIC}.json` (JSONL):
  - `question`
  - `confidence` in [0, 1]

The loader joins confidences to examples by matching `question` strings.

## Configuration (YAML)

All scripts read a YAML config in `configs/`. Common fields:

- Paths
  - `data_dir` or `data_path`: base directory for raw data (e.g., `Data/CoT/raw`)
  - `model` or `model_dir`: model subdirectory (e.g., `llama3-1_8B`)
  - `output_dir`, `results_path`: where to save JSON and reports
- Scope
  - `datasets`: list, imported from `datasets.yaml`
  - `runs`: list of run indices, e.g. `[0, 1, 2, 3, 4]`
- Metrics
  - `metrics`: list of metric names or group references via `@group`
  - Groups are provided by imported files like `uq_methods_cot.yaml` / `uq_methods_cod.yaml`
- Metric-specific
  - `n_bins` (ECE)
  - `roc_curve.n_points`, `confidence_interval.level` (AUROC)

YAML supports imports and simple templating via `src/utils/config_loader.py`. The `imports:` key merges other YAMLs; local values override imported ones. The `@group` syntax in `metrics` expands to the group’s `name` entries.

Examples:

```yaml
# configs/ece_config.yaml
imports:
  - datasets.yaml
  - uq_methods_cot.yaml

data_dir: Data/CoT/raw
model: llama3-1_8B
results_path: results/cot/ece
runs: [0, 1, 2, 3, 4]
n_bins: 10
metrics:
  - '@baseline_methods'
  - '@cot_methods'
  - '@true_probability_methods'
  - '@self_probing_methods'
```

```yaml
# configs/auroc_config.yaml
imports:
  - datasets.yaml
  - uq_methods_cot.yaml

data_dir: Data/CoT/raw
model_dir: llama3-1_8B
results_path: results/cot/auroc
runs: [0, 1, 2, 3, 4]
roc_curve: { n_points: 100 }
confidence_interval: { level: 0.95 }
# Optional grouping for reporting
grouped_metrics:
  baselines: "{{ baseline_methods | map(attribute='name') | list }}"
  cot_methods: "{{ cot_methods | map(attribute='name') | list }}"
  ptrue_methods: "{{ true_probability_methods | map(attribute='name') | list }}"
  self_probing: "{{ self_probing_methods | map(attribute='name') | list }}"
```

## Switching between CoT and CoD paths

To switch between Chain-of-Thought (CoT) and Chain-of-Decision (CoD) results use the provided `*_cot.yaml` and `*_cod.yaml` config variants to avoid manual edits.

Example (CoD ECE):

```yaml
# configs/ece_config_cod.yaml
imports:
  - datasets.yaml
  - uq_methods_cod.yaml

data_dir: Data/CoD/raw
model: llama3-1_8B
results_path: results/cod/ece
runs: [0, 1, 2, 3, 4]
n_bins: 10
metrics:
  - '@baseline_methods'
  - '@cot_methods'
  - '@true_probability_methods'
  - '@self_probing_methods'
```

## Quickstart

From the project root (`Results_Analysis/`):

1) Adjust a config in `configs/` to point to your data and desired datasets/metrics.
2) Run metric calculations (each script has a `--config` flag):

```bash
python scripts/calculation/calculate_accuracy.py  --config configs/accuracy_config_cot.yaml
python scripts/calculation/calculate_auroc.py     --config configs/auroc_config_cot.yaml
python scripts/calculation/calculate_ece.py       --config configs/ece_config_cot.yaml
python scripts/calculation/calculate_brier.py     --config configs/brier_config_cot.yaml
```

Each script:
- Loads config (and imports)
- Processes each dataset × run
- Saves per-run JSON to `results/.../{dataset}/run_{i}/{dataset}_{task}.json`
- Saves aggregated JSON to `results/.../{dataset}/aggregated/{dataset}_{task}.json`
- Writes a Markdown summary to `results/.../{dataset}/{dataset}_{task}_report.md`

Tip: Use the `*_cod.yaml` or `*_cot.yaml` config variants to switch between CoD/CoT result paths.

## Visualization

Example: ECE leaderboard across datasets.

```bash
python scripts/visualization/plot_ece_leaderboard.py --config configs/ece_config.yaml
```

- Reads per-run and aggregated ECE from `results_path`
- Saves a figure `ece_leaderboard.png`
  - By default it writes into a `figures/` directory under the results family; to avoid overwriting CoT figures, the script may default to `results/cod/figures` when `results_path` doesn’t include `cod`.

Other useful plots (see `scripts/visualization/`):
- `plot_auroc_leaderboard.py`, `plot_brier_leaderboard.py`, `plot_ece_leaderboard_split.py`
- Reliability diagrams: `plot_reliability_ap.py`, `plot_reliability_ptrue.py`, `plot_reliability_selfprobing.py`

## Notes & Troubleshooting

- Paths are resolved relative to this folder; scripts prepend the project root so relative paths in configs work when run from `Results_Analysis`.
- If a metric is missing for a dataset/run, you’ll see warnings like "Metric '<name>' not found" and the run will be skipped for that metric.
- Aggregation requires at least one successful run per metric; otherwise aggregated files won’t be written.
- Ensure dataset names in `configs/datasets.yaml` match your directory casing (e.g., `ASDiv`, `svamp`).

## Extending

- Add a new uncertainty metric: implement it in `src/metrics/`, expose a calculator API similar to `AUROC`/`ECE`.
- Add a new group or method: edit `uq_methods_*.yaml` and reference in a config via `@your_group`.
- Create new plots: use the patterns in `scripts/visualization/` and read from the aggregated JSON.
