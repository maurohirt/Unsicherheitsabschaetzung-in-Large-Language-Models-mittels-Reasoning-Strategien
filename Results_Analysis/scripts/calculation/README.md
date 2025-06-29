# AUROC Calculation Scripts

This directory contains scripts for calculating Area Under the Receiver Operating Characteristic (AUROC) scores for uncertainty quantification metrics across multiple runs.

## Overview

The main script `calculate_auroc.py` performs the following tasks:

1. Loads data from multiple experimental runs
2. Calculates AUROC scores for specified uncertainty metrics
3. Aggregates results across runs
4. Generates summary statistics and visualizations

## Prerequisites

- Python 3.7+
- Required packages (install with `pip install -r requirements.txt`):
  - numpy
  - scikit-learn
  - scipy
  - tqdm
  - pyyaml
  - jinja2

## Usage

### Command Line Interface

```bash
python calculate_auroc.py --data-dir /path/to/data --output-dir /path/to/output --dataset dataset_name --metrics metric1 metric2 ...
```

### Using Configuration File

1. Edit the configuration file `configs/auroc_config.yaml` to specify:
   - Input/output directories
   - Datasets to process
   - Uncertainty metrics to evaluate
   - Run numbers to include

2. Run the script with the configuration file:

```bash
python calculate_auroc.py --config configs/auroc_config.yaml
```

### Example

To calculate AUROC for entropy and max_prob metrics on the sst2 dataset across runs 0-4:

```bash
python calculate_auroc.py \
  --data-dir ../Data/CoT/raw \
  --output-dir ../results/auroc_scores \
  --dataset sst2 \
  --metrics entropy max_prob \
  --runs 0 1 2 3 4
```

## Output

The script generates the following output files:

- `output_dir/run_X/dataset_auroc.json`: Per-run AUROC results
- `output_dir/dataset_aggregated_auroc.json`: Aggregated results across all runs

Each result file contains:
- AUROC scores
- ROC curve data (FPR, TPR, thresholds)
- Aggregated statistics (mean, std, confidence intervals)

## Customization

To use with your own data, ensure your data files follow this structure:

```
data_dir/
  run_0/
    dataset1.json
    dataset2.json
  run_1/
    dataset1.json
    dataset2.json
  ...
```

Each JSON file should contain:
- `labels`: Ground truth labels
- `predictions`: Model predictions
- `uncertainty_metrics`: Dictionary of uncertainty scores for each metric

## Notes

- The script assumes that higher uncertainty scores indicate more uncertain predictions.
- For binary classification, the positive class for AUROC is defined as incorrect predictions (`y_true != y_pred`).

## Additional Scripts

- `calculate_ece.py`: Calculates Expected Calibration Error (ECE) per run with detailed bin data, aggregates results, and generates JSON and markdown reports.
- `calculate_brier.py`: Computes Brier scores per metric per run, aggregates across runs, and outputs JSON and markdown reports.

## Configuration Files

Each script uses its own configuration file under `configs/`:

- `configs/auroc_config.yaml`
- `configs/ece_config.yaml`
- `configs/brier_config.yaml`

Run the scripts directly; they automatically load their corresponding configuration files:
```bash
python calculate_auroc.py
python calculate_ece.py
python calculate_brier.py
```
