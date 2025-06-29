#!/usr/bin/env python3
"""
Script to calculate AUROC scores for uncertainty quantification metrics across multiple runs.
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import itertools
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.metrics.auroc import AUROC
from src.data.loader import DataLoader
from src.utils.config_loader import load_yaml_with_imports, resolve_metrics


def load_run_data(data_loader: DataLoader, run_id: int, dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Load data for a specific run and dataset using the DataLoader.
    
    Args:
        data_loader: Instance of DataLoader
        run_id: ID of the run to load
        dataset_name: Name of the dataset to load
        
    Returns:
        Dictionary containing the loaded data or None if not found
    """
    try:
        return data_loader.load_run_data(run_id, dataset_name)
    except Exception as e:
        print(f"Error loading data for run {run_id}, dataset {dataset_name}: {e}")
        return None


def process_run(data_loader: DataLoader, run_id: int, dataset_name: str, metrics: List[str]) -> Dict[str, Any]:
    """
    Process a single run and calculate AUROC for the specified metrics.
    
    Args:
        data_loader: Instance of DataLoader
        run_id: ID of the run to process
        dataset_name: Name of the dataset to process
        metrics: List of metric names to calculate AUROC for
        
    Returns:
        Dictionary containing AUROC results for all metrics
    """
    # Load the data for this run and dataset
    data = load_run_data(data_loader, run_id, dataset_name)
    if not data or 'examples' not in data:
        print(f"No valid data found for run {run_id}, dataset {dataset_name}")
        return {}
    
    # Initialize the AUROC calculator
    auroc_calculator = AUROC()
    results = {}
    
    # Extract true labels (1 for correct, 0 for incorrect)
    y_true = np.array([1 if item.get('label') is True else 0 for item in data['examples']])
    
    # For binary classification, we'll use (y_true != y_pred) as the positive class
    # Since we're evaluating uncertainty, we want to detect incorrect predictions
    # So we'll set y_pred to all ones (assuming all predictions are correct)
    # and use the actual labels as y_true
    y_pred = np.ones_like(y_true)  # Assume all predictions are correct
    
    # Calculate AUROC for each metric
    for metric in metrics:
        # Check if the metric is in the data (with _confidences suffix)
        metric_key = f"{metric}_confidences"
        if metric_key not in data:
            print(f"Warning: Metric '{metric}' not found in run {run_id}, dataset {dataset_name}")
            continue
            
        # Get confidence scores (higher score = more confident)
        conf_scores = np.array(data[metric_key])
        
        # For AUROC, we want to detect incorrect predictions (y_true == 0)
        binary_labels = y_true  # 1 for correct, 0 for incorrect
        
        try:
            # Calculate AUROC
            metric_result = auroc_calculator.calculate(
                y_true=binary_labels,
                y_pred=y_pred,
                y_prob=conf_scores
            )
            results[metric] = metric_result
        except Exception as e:
            print(f"Error calculating AUROC for {metric} in run {run_id}, dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def aggregate_across_runs(all_run_results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Any]:
    """
    Aggregate AUROC results across multiple runs.
    
    Args:
        all_run_results: List of per-run results
        metrics: List of metric names
        
    Returns:
        Dictionary containing aggregated results
    """
    aggregated = {}
    
    for metric in metrics:
        # Collect all results for this metric across runs
        metric_results = []
        
        for run_result in all_run_results:
            if metric in run_result:
                metric_results.append(run_result[metric])
        
        # Aggregate the results
        if metric_results:
            aggregated[metric] = AUROC.aggregate_results(metric_results)
    
    return aggregated


def load_config(config_path: Path) -> Dict[str, Any]:
    # Load YAML with imports; imported configs contain datasets and uncertainty_metrics

    """
    Load and validate the configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated configuration dictionary
    """
    # Load the config file with imports
    config = load_yaml_with_imports(config_path)
    # Use imported dataset and uncertainty metrics directly
    if 'datasets' in config:
        # Ensure datasets is a list
        if not isinstance(config['datasets'], list):
            raise ValueError(f"Datasets must be a list, got {type(config['datasets'])}")
    else:
        config['datasets'] = []
    # Collect metrics from imported UQ method configs
    metrics = []
    for key in ['baseline_methods', 'cot_methods', 'true_probability_methods', 'self_probing_methods']:
        if key in config:
            metrics.extend([item['name'] for item in config[key]])
    config['metrics'] = metrics
    
    # Set default values if not specified
    config.setdefault('data_dir', Path('Data/CoT/raw'))
    config.setdefault('output_dir', Path('results/auroc_scores'))
    config.setdefault('datasets', [])
    config.setdefault('metrics', [])
    config.setdefault('runs', list(range(5)))  # Default to runs 0-4
    config.setdefault('roc_curve', {'n_points': 100})
    config.setdefault('confidence_interval', {'level': 0.95})
    
    # Convert paths to Path objects
    config['data_dir'] = Path(config['data_dir'])
    config['output_dir'] = Path(config['output_dir'])
    
    # Resolve metric groups if needed
    if 'grouped_metrics' in config:
        # Add the grouped metrics to the config for resolution
        config['_all_metrics'] = {
            **{k: v for k, v in config.items() if k not in ['grouped_metrics']},
            **config['grouped_metrics']
        }
        
        # Resolve any group references in the metrics list
        config['metrics'] = [
            m[1:] if isinstance(m, str) and m.startswith('@') else m
            for m in config['metrics']
        ]
    
    return config


def process_dataset(config: Dict[str, Any], data_loader: DataLoader, dataset: str) -> None:
    """
    Process a single dataset across all runs.
    
    Args:
        config: Configuration dictionary
        data_loader: Instance of DataLoader
        dataset: Name of the dataset to process
    """
    results_dir = Path(config['results_path'])
    metrics = config['metrics']
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each run
    all_run_results = []
    
    for run_num in tqdm(config['runs'], desc=f"Processing {dataset}"):
        # Process this run
        run_result = process_run(data_loader, run_num, dataset, metrics)
        
        if run_result:  # Only save if we got valid results
            all_run_results.append(run_result)
            
            # Save per-run results
            run_output_dir = results_dir / f"run_{run_num}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = run_output_dir / f"{dataset}_auroc.json"
            AUROC.save_results(run_result, output_file)
    
    # Aggregate results across runs if we have any valid results
    if all_run_results:
        aggregated_results = aggregate_across_runs(all_run_results, metrics)
        
        # Save aggregated results
        agg_output_dir = results_dir / "aggregated"
        agg_output_dir.mkdir(parents=True, exist_ok=True)
        
        agg_output_file = agg_output_dir / f"{dataset}_auroc.json"
        AUROC.save_results(aggregated_results, agg_output_file)

        
        print(f"\nAggregated AUROC results for {dataset} saved to {agg_output_file}")
        
        # Print summary
        print_summary(aggregated_results, metrics, config['runs'])
        # Save markdown report for this dataset
        md_content = f"# AUROC Report for {dataset}\n\n"
        md_content += "## Aggregated Results\n\n"
        md_content += "| Metric | Mean | Std Dev | Min | Max | Runs |\n"
        md_content += "|--------|-----:|--------:|----:|----:|-----:|\n"
        for metric_name, stats in aggregated_results.items():
            ci = stats.get('confidence_interval', [float('nan'), float('nan')])
            md_content += (f"| {metric_name} | {stats['mean_auroc']:.4f} | "
                           f"{stats['std_auroc']:.4f} | {stats['min_auroc']:.4f} | "
                           f"{stats['max_auroc']:.4f} | {stats['n_runs']} |\n")
        report_path = results_dir / f"{dataset}_auroc_report.md"
        with open(report_path, 'w') as md_file:
            md_file.write(md_content)

        print(f"Markdown report for {dataset} saved to {report_path}")
    else:
        print(f"No valid results were generated for {dataset}.")


def print_summary(aggregated_results: Dict[str, Any], metrics: List[str], runs: List[int]) -> None:
    """Print a summary of the aggregated results."""
    print("\nSummary of AUROC scores:")
    print("-" * 100)
    print(f"{'Metric':<30} {'Mean':<10} {'Std Dev':<10} {'95% CI':<25} {'Min':<8} {'Max':<8} {'Runs'}")
    print("-" * 100)
    
    for metric in metrics:
        if metric not in aggregated_results:
            continue
            
        stats = aggregated_results[metric]
        ci = stats.get('confidence_interval', [float('nan'), float('nan')])
        
        print(f"{metric:<30} "
              f"{stats.get('mean_auroc', float('nan')):>7.4f}    "
              f"{stats.get('std_auroc', float('nan')):>7.4f}    "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]  "
              f"{stats.get('min_auroc', float('nan')):>7.4f}  "
              f"{stats.get('max_auroc', float('nan')):>7.4f}  "
              f"{stats.get('n_runs', 0)}/{len(runs)}")


def main():
    # Get project root and load config
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / 'configs' / 'auroc_config.yaml'
    config = load_config(config_path)
    # Convert paths to absolute
    for key in ['data_dir', 'output_dir', 'results_path']:
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str(project_root / config[key])
    data_dir = Path(config['data_dir'])
    results_dir = Path(config['results_path'])
    # Initialize DataLoader
    print("Initializing DataLoader...")
    data_loader = DataLoader({'data_path': str(data_dir), 'model': config.get('model_dir', 'llama3-1_8B')})
    print("Starting AUROC calculation...")
    print(f"Runs: {config['runs']}")
    print(f"Datasets: {', '.join(config['datasets'])}")
    print(f"Metrics: {', '.join(config['metrics'])}")
    # Process each dataset
    for dataset in config['datasets']:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset}")
        process_dataset(config, data_loader, dataset)
    print("\nAUROC calculation complete! Results saved to:", results_dir)
    return
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate AUROC scores for uncertainty metrics across multiple runs.')
    parser.add_argument('--config', type=str, default='configs/auroc_config.yaml',
                      help='Path to the configuration file (default: configs/auroc_config.yaml)')
    
    # Parse command line arguments
    args = parser.parse_args()
    config_path = Path(args.config)
    
    # Load and validate configuration
    print(f"Loading configuration from {config_path}...")
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize DataLoader with the correct configuration
    print("Initializing DataLoader...")
    data_loader_config = {
        'data_path': str(config['data_dir']),
        'model': config.get('model_dir', 'llama3-1_8B')  # Default to llama3-1_8B if not specified
    }
    data_loader = DataLoader(data_loader_config)
    
    # Ensure output directory exists
    config['output_dir'] = Path(config['output_dir'])
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"- Data directory: {config['data_dir']}")
    print(f"- Output directory: {config['output_dir']}")
    print(f"- Datasets: {', '.join(config['datasets'])}")
    print(f"- Metrics: {', '.join(config['metrics'])}")
    print(f"- Runs: {config['runs']}\n")
    
    # Process each dataset
    for dataset in config['datasets']:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*80}")
        process_dataset(config, data_loader, dataset)
    
    print("\nAUROC calculation complete!")


if __name__ == "__main__":
    main()
