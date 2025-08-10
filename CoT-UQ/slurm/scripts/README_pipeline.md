# CoT-UQ Full Pipeline Runner

This directory contains a unified SBATCH script that runs the complete CoT-UQ pipeline including:
1. Inference generation (`inference_refining.py`)
2. Uncertainty quantification (`stepuq.py`)
3. Results analysis (`analyze_result.py`)

## Usage

### Basic Usage

To run the pipeline with the default configuration:
```bash
sbatch run_full_pipeline.sbatch
```

### Using Custom Configuration

To run with a specific configuration file:
```bash
sbatch run_full_pipeline.sbatch /path/to/your/config.yaml
```

## Configuration File Format

The pipeline reads all parameters from a YAML configuration file. Here's the structure:

```yaml
# Model configuration
model_engine: "llama3-1_8B"  # Options: llama3-1_8B, llama2-13b

# Datasets to process
datasets:
  - "ASDiv"
  - "2WikimhQA"
  - "hotpotQA"
  - "svamp"
  - "gsm8k"

# Output configuration
output_path_base: "/home2/mauro.hirt/CoT-UQ/output"

# Inference parameters
temperature: 1.0
try_times: 20
max_length_cot: 128

# Dataset range (optional)
test_start: "0"      # Starting index (default: "0")
test_end: "full"     # Ending index or "full" for entire dataset (default: "full")

# UQ methods to apply
uq_methods:
  - "probas-mean-bl"
  - "probas-min-bl"
  - "token-sar-bl"
```

### Dataset Range Parameters

- `test_start`: The starting index for dataset processing (string)
  - Default: "0" (start from the beginning)
  - Example: "100" to start from the 101st sample

- `test_end`: The ending index for dataset processing (string)
  - Default: "full" (process all data from test_start)
  - Example: "200" to process samples 0-199 (if test_start="0")
  - Example: "50" with test_start="10" to process samples 10-49

These parameters are useful for:
- Testing with a small subset of data
- Resuming interrupted runs
- Distributing work across multiple jobs

## Available Configuration Files

Several pre-configured files are provided in the `configs/` directory:

- `pipeline_config.yaml` - Default configuration with baseline UQ methods
- `pipeline_config_minimal.yaml` - Minimal configuration for testing (single dataset, single UQ method)
- `pipeline_config_full.yaml` - Full configuration with all datasets and all UQ methods
- `pipeline_config_llama2.yaml` - Configuration for Llama2-13B model

## Environment Requirements

The script requires the following environment variables:
- `HUGGINGFACE_HUB_TOKEN` - For accessing Hugging Face models
- `OPENAI_API_KEY` - Required for analyzing 2WikimhQA and hotpotQA datasets

Set these before submitting the job:
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
export OPENAI_API_KEY="your_api_key_here"
sbatch run_full_pipeline.sbatch
```

## Output Structure

The pipeline creates outputs in the following structure:
```
output/
├── logs/
│   └── pipeline/
│       └── pipeline_<jobid>_<taskid>.out
└── <model_engine>/
    └── <dataset>/
        ├── output_v1.json                 # Inference results
        ├── confidences/
        │   ├── output_v1_<uq_method>.json # UQ results for each method
        │   └── ...
        └── output_v1_w_labels.json        # Analyzed results with labels
```

## Pipeline Behavior

- The pipeline processes each dataset sequentially
- If inference fails for a dataset, the pipeline skips UQ and analysis for that dataset
- If a UQ method fails, the pipeline continues with other UQ methods
- Analysis requires the corresponding UQ output to exist
- For 2WikimhQA and hotpotQA, analysis is skipped if OPENAI_API_KEY is not set

## Example Commands

### Run minimal test
```bash
sbatch run_full_pipeline.sbatch configs/pipeline_config_minimal.yaml
```

### Run full pipeline with all methods
```bash
sbatch run_full_pipeline.sbatch configs/pipeline_config_full.yaml
```

### Run with Llama2 model
```bash
sbatch run_full_pipeline.sbatch configs/pipeline_config_llama2.yaml
```

## Monitoring Progress

The pipeline outputs detailed logs including:
- Configuration parameters
- Progress for each step
- Success/failure status for each component
- Total execution time for each dataset
- Final summary of all operations

Check the log file to monitor progress:
```bash
tail -f output/logs/pipeline/pipeline_<jobid>_<taskid>.out
```
