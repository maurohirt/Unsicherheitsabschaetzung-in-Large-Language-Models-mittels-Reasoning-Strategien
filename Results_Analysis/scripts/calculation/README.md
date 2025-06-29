# Calculation Scripts for Uncertainty Metrics

This directory contains scripts to compute evaluation metrics for uncertainty quantification across multiple experimental runs:

- **AUROC** (Area Under the ROC Curve)
- **ECE** (Expected Calibration Error)
- **Brier Score**
- **Accuracy** (classification accuracy; not a UQ metric)

## Overview

Each script processes raw model outputs per run, calculates the specified metric for each uncertainty method, and aggregates results across runs:

- `calculate_auroc.py`: Generates per-run ROC curves and AUROC scores, aggregates mean, std, and confidence intervals.
- `calculate_ece.py`: Computes calibration error by binning confidences, collects bin-level accuracy vs. confidence, aggregates ECE scores.
- `calculate_brier.py`: Calculates Brier Score (mean squared error between confidence and true label) per run and aggregates statistics.
- `calculate_accuracy.py`: Computes classification accuracy (percentage correct) per run and aggregates across runs.

### Accuracy

The `calculate_accuracy.py` script computes classification accuracy (percentage of correct predictions) per run and per dataset. It loads data via `DataLoader`, counts true labels, aggregates across runs (mean, std, 95% CI), saves per-run and aggregated JSON, and generates a markdown report.

## Prerequisites

- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Key packages: numpy, scikit-learn, scipy, tqdm, pyyaml, jinja2

## Configuration

Each script automatically loads its YAML config from the `configs/` folder (no flags required):

- `configs/auroc_config.yaml`
- `configs/ece_config.yaml`
- `configs/brier_config.yaml`

Typical config fields:
- `datasets`: List of dataset names to process (e.g., ASDiv)
- `data_dir`: Path to raw data (e.g., `Data/CoT/raw`)
- `model`: Model identifier or path
- `runs`: Runs to include (e.g., `[0, 1, 2, 3, 4]`)
- `metrics`: List of metrics or groups (use `@group_name` to reference groups in imported UQ methods YAML)
- `n_bins` (ECE only): Number of bins for calibration error

Edit these YAML files to point to your data and desired metrics.

## Usage

From the project root, run:
```bash
python scripts/calculation/calculate_auroc.py
python scripts/calculation/calculate_ece.py
python scripts/calculation/calculate_brier.py
python scripts/calculation/calculate_accuracy.py
```

Each script will:
1. Load its config
2. Initialize `DataLoader` with `data_dir` and `model`
3. Process each dataset and run
4. Save per-run JSON in `results/<task>/<dataset>/run_X/`
5. Save aggregated JSON in `results/<task>/<dataset>/aggregated/`
6. Generate a Markdown report in `results/<task>/<dataset>/`

## Raw Data Format

Raw data in `{data_dir}` must be organized as:

```
{data_dir}/run_{run_id}/{model}/{dataset}/output_v1_w_labels.json
{data_dir}/run_{run_id}/{model}/{dataset}/confidences/output_v1_<metric>.json
```

Within each dataset folder:
- `output_v1_w_labels.json`: JSON lines with keys:
  - `question`: input question
  - `llm answer`: model's response
  - `correct answer`: ground truth answer
  - `label`: boolean indicating correctness
- `confidences/`: directory containing files `output_v1_<metric>.json` with JSON lines containing:
  - `question`
  - `confidence`

The `DataLoader` matches each `question` in `confidences` to the examples by question text.
## Outputs

- **Per-run JSON**: `results/<task>/<dataset>/run_<run>/<dataset>_<task>.json`
- **Aggregated JSON**: `results/<task>/<dataset>/aggregated/<dataset>_<task>.json`
- **Markdown report**: `results/<task>/<dataset>/<dataset>_<task>_report.md`

Replace `<task>` with `auroc`, `ece`, `brier`, or `accuracy`.

## Notes

- All scripts assume higher confidence = more certain (used directly for Brier/ECE). For AUROC, positive class is incorrect predictions (`y_true != y_pred`).
- ECE uses equal-width bins; adjust `n_bins` in config if needed.
