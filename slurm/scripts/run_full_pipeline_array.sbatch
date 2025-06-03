#!/bin/bash

#SBATCH -p performance
#SBATCH --gres=gpu:rtxA4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=dot_uq_pipeline_array
#SBATCH --output=output/logs/pipeline/pipeline_%A_dataset_%a.out
#SBATCH --error=output/logs/pipeline/pipeline_%A_dataset_%a.err
#SBATCH --export=ALL,HUGGINGFACE_HUB_TOKEN,OPENAI_API_KEY
#SBATCH --array=0-4  # For 5 datasets (ASDiv, 2WikimhQA, hotpotQA, svamp, gsm8k) - adjust based on your config

# Read configuration file path from command line argument
CONFIG_FILE=${1:-"pipeline_config.yaml"}

# Environment setup
PROJECT_ROOT="/home2/mauro.hirt/CoT-UQ"
echo "Project root: $PROJECT_ROOT"

export PYTHONPATH=$PROJECT_ROOT
export SIF_PATH="/home2/mauro.hirt/containers/cot-uq_latest.sif"

# Hugging Face caching
export HOST_HF_CACHE="${PROJECT_ROOT}/hf_cache"
export CONTAINER_HF_CACHE="/root/.cache/huggingface"
mkdir -p ${HOST_HF_CACHE}

# Create log directory
mkdir -p output/logs/pipeline

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Configuration file not found: ${CONFIG_FILE}"
    echo "Usage: sbatch run_full_pipeline_array.sbatch [config_file.yaml]"
    exit 1
fi

echo "Loading configuration from: ${CONFIG_FILE}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

# Create a temporary Python script file
TEMP_SCRIPT=$(mktemp /tmp/pipeline_runner_array_XXXXXX.py)

# Write the Python script to the temporary file
cat > "${TEMP_SCRIPT}" << 'EOF'
import yaml
import subprocess
import os
import sys
import time
from datetime import datetime

# Get config file and array task ID from command line arguments
config_file = sys.argv[1]
array_task_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

# Load configuration
print(f"Loading configuration from: {config_file}")
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration parameters
model_engine = config['model_engine']
datasets = config['datasets']
output_path_base = config['output_path_base']
temperature = config.get('temperature', 1.0)
try_times = config.get('try_times', 20)
max_length_cot = config.get('max_length_cot', 128)
uq_methods = config.get('uq_methods', ['probas-mean-bl', 'probas-min-bl', 'token-sar-bl'])
test_start = config.get('test_start', '0')
test_end = config.get('test_end', 'full')

# If array_task_id is provided, process only that dataset
if array_task_id is not None:
    if array_task_id < 0 or array_task_id >= len(datasets):
        print(f"ERROR: Array task ID {array_task_id} out of range for {len(datasets)} datasets")
        sys.exit(1)
    dataset_to_process = datasets[array_task_id]
    datasets = [dataset_to_process]
    print(f"Array job mode: Processing only dataset {dataset_to_process} (index {array_task_id})")

# Print configuration
print("Configuration loaded:")
print(f"  Model: {model_engine}")
print(f"  Dataset(s) to process: {datasets}")
print(f"  Output path base: {output_path_base}")
print(f"  Temperature: {temperature}")
print(f"  Try times: {try_times}")
print(f"  Max length CoT: {max_length_cot}")
print(f"  UQ methods: {uq_methods}")
print(f"  Test start: {test_start}")
print(f"  Test end: {test_end}")

# Function to run command with proper error handling
def run_command(cmd, step_name):
    print(f"\n{'='*60}")
    print(f"Starting: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Filter output based on the command type
        if 'stepuq.py' in cmd:
            # For stepuq, only show progress bars and important warnings
            filtered_lines = []
            for line in result.stdout.split('\n'):
                # Skip experiment args section
                if line.startswith('---------------experiment args') or line.startswith('-----'):
                    continue
                # Skip lines between experiment args markers
                if any(x in line for x in ['max_length_cot:', 'try_times:', 'temperature:', 
                                          'dataset:', 'datapath:', 'hf_token:', 'api_key:',
                                          'model_engine:', 'uq_engine:', 'model_path:', 
                                          'output_path:', 'test_start:', 'test_end:']):
                    continue
                # Keep progress bars and warnings
                if '%|' in line or 'Loading checkpoint' in line or line.strip():
                    filtered_lines.append(line)
            
            # Print filtered output
            print('\n'.join(filtered_lines))
            
            # Show CrossEncoder warning if present but acknowledge it's expected
            if 'CrossEncoder' in result.stderr:
                print("\nNote: CrossEncoder deprecation warning (expected, can be ignored)")
        
        elif 'analyze_result.py' in cmd:
            # For analyze_result, skip experiment args and "Skipping label generation"
            filtered_lines = []
            skip_section = False
            for line in result.stdout.split('\n'):
                if line.startswith('---------------experiment args'):
                    skip_section = True
                    continue
                elif line.startswith('-----') and skip_section:
                    skip_section = False
                    continue
                elif skip_section:
                    continue
                elif 'Skipping label generation' in line:
                    continue
                elif line.strip():  # Keep non-empty lines
                    filtered_lines.append(line)
            
            print('\n'.join(filtered_lines))
        
        else:
            # For inference, show the output as is
            print(result.stdout)
        
        if result.stderr and 'CrossEncoder' not in result.stderr:
            print(f"\nWarnings/Info: {result.stderr}")
            
        elapsed_time = time.time() - start_time
        print(f"\n✓ {step_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

# Ensure models are cached (only do this once, not for every array job)
if array_task_id is None or array_task_id == 0:
    print("\nEnsuring model is cached...")
    from huggingface_hub import snapshot_download

    repo_map = {
        'llama3-1_8B': 'meta-llama/Llama-3.1-8B',
        'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    }

    repo = repo_map.get(model_engine)
    if repo:
        print(f'Ensuring model {repo} is cached...')
        snapshot_download(repo_id=repo, token=os.getenv('HUGGINGFACE_HUB_TOKEN'), resume_download=True)

# Track overall pipeline status
pipeline_success = True
results_summary = []

# Process each dataset (in array mode, this will be just one dataset)
for dataset in datasets:
    print(f"\n\n{'#'*80}")
    print(f"Processing dataset: {dataset}")
    print(f"{'#'*80}\n")
    
    output_path = f"{output_path_base}/{model_engine}/{dataset}"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/confidences", exist_ok=True)
    
    dataset_start_time = time.time()
    dataset_status = {'dataset': dataset, 'steps': {}}
    
    # Step 1: Inference
    cmd = [
        'python', 'inference_refining.py',
        '--dataset', dataset,
        '--model_engine', model_engine,
        '--model_path', model_engine,
        '--temperature', str(temperature),
        '--output_path', output_path,
        '--try_times', str(try_times),
        '--max_length_cot', str(max_length_cot),
        '--test_start', str(test_start),
        '--test_end', str(test_end)
    ]
    
    success = run_command(cmd, f"Inference for {dataset}")
    dataset_status['steps']['inference'] = 'success' if success else 'failed'
    
    if not success:
        print(f"\nSkipping remaining steps for {dataset} due to inference failure")
        pipeline_success = False
        results_summary.append(dataset_status)
        continue
    
    # Step 2: UQ methods
    uq_success = True
    for uq_method in uq_methods:
        cmd = [
            'python', 'stepuq.py',
            '--dataset', dataset,
            '--model_engine', model_engine,
            '--model_path', model_engine,
            '--uq_engine', uq_method,
            '--output_path', output_path,
            '--temperature', str(temperature),
            '--try_times', '5'
        ]
        
        success = run_command(cmd, f"UQ method {uq_method} for {dataset}")
        dataset_status['steps'][f'uq_{uq_method}'] = 'success' if success else 'failed'
        
        if not success:
            print(f"\nWarning: UQ method {uq_method} failed for {dataset}, continuing with others...")
            uq_success = False
            continue
    
    # Step 3: Analyze results
    if dataset in ['2WikimhQA', 'hotpotQA'] and not os.getenv('OPENAI_API_KEY'):
        print(f"\nSkipping analysis for {dataset} - OpenAI API key required")
        dataset_status['steps']['analysis'] = 'skipped - no API key'
    else:
        analysis_success = True
        for uq_method in uq_methods:
            # Check if UQ output exists
            uq_output = f"{output_path}/confidences/output_v1_{uq_method}.json"
            if not os.path.exists(uq_output):
                print(f"\nSkipping analysis for {uq_method} - UQ output not found")
                continue
            
            cmd = [
                'python', 'analyze_result.py',
                '--dataset', dataset,
                '--model_engine', model_engine,
                '--uq_engine', uq_method,
                '--output_path', output_path
            ]
            
            success = run_command(cmd, f"Analysis for {dataset} with {uq_method}")
            dataset_status['steps'][f'analysis_{uq_method}'] = 'success' if success else 'failed'
            
            if not success:
                analysis_success = False
    
    dataset_elapsed_time = time.time() - dataset_start_time
    dataset_status['total_time'] = f"{dataset_elapsed_time:.2f} seconds"
    results_summary.append(dataset_status)
    
    print(f"\n{'='*60}")
    print(f"Dataset {dataset} completed in {dataset_elapsed_time:.2f} seconds")
    print(f"{'='*60}")

# Print final summary
print(f"\n\n{'#'*80}")
print(f"PIPELINE SUMMARY")
print(f"{'#'*80}\n")

for result in results_summary:
    print(f"\nDataset: {result['dataset']}")
    print(f"Total time: {result.get('total_time', 'N/A')}")
    print("Steps:")
    for step, status in result['steps'].items():
        status_symbol = '✓' if status == 'success' else '✗'
        print(f"  {status_symbol} {step}: {status}")

print(f"\n\nPipeline completed at: {datetime.now()}")

if pipeline_success:
    print("\n✓ All pipeline steps completed successfully!")
    sys.exit(0)
else:
    print("\n✗ Some pipeline steps failed. Check logs for details.")
    sys.exit(1)
EOF

# Check if this is an array job
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Array job mode - process single dataset
    echo "Running as array job, task ID: ${SLURM_ARRAY_TASK_ID}"
    
    # Run the pipeline in Singularity with array task ID
    singularity exec --nv \
      -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
      -B ${PROJECT_ROOT}:${PROJECT_ROOT} \
      "${SIF_PATH}" \
      bash -lc "
        # Navigate to the code directory
        cd ${PROJECT_ROOT}
        
        # Configure Hugging Face env inside container
        export HF_HOME=${CONTAINER_HF_CACHE}
        export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
        export OPENAI_API_KEY=${OPENAI_API_KEY}
        
        # Install PyYAML if not present
        pip install -q PyYAML
        
        # Run the Python script with array task ID
        python ${TEMP_SCRIPT} ${CONFIG_FILE} ${SLURM_ARRAY_TASK_ID}
      "
else
    # Non-array mode - process all datasets sequentially
    echo "Running in sequential mode (no array job)"
    
    # Run the pipeline in Singularity without array task ID
    singularity exec --nv \
      -B ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE} \
      -B ${PROJECT_ROOT}:${PROJECT_ROOT} \
      "${SIF_PATH}" \
      bash -lc "
        # Navigate to the code directory
        cd ${PROJECT_ROOT}
        
        # Configure Hugging Face env inside container
        export HF_HOME=${CONTAINER_HF_CACHE}
        export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
        export OPENAI_API_KEY=${OPENAI_API_KEY}
        
        # Install PyYAML if not present
        pip install -q PyYAML
        
        # Run the Python script without array task ID
        python ${TEMP_SCRIPT} ${CONFIG_FILE}
      "
fi

# Clean up temporary file
rm -f "${TEMP_SCRIPT}"

echo "===== PIPELINE EXECUTION COMPLETED ====="
