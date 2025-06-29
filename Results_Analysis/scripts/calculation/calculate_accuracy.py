#!/usr/bin/env python3
# scripts/calculation/calculate_accuracy.py
"""
Main script for calculating accuracy metrics across multiple runs and datasets.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.metrics.accuracy import AccuracyCalculator
from src.utils.config_loader import load_yaml_with_imports

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file with imports support."""
    return load_yaml_with_imports(config_path)

def main():
    # Get the project root directory (Results_Analysis)
    project_root = Path(__file__).parent.parent.parent
    
    # Load configuration using absolute path
    config_path = project_root / 'configs' / 'accuracy_config.yaml'
    config = load_config(config_path)
    
    # Convert all paths to absolute paths
    for path_key in ['data_path', 'output_path', 'results_path']:
        if path_key in config and not Path(config[path_key]).is_absolute():
            config[path_key] = str(project_root / config[path_key])
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    print("Starting accuracy calculation...")
    print(f"Runs: {config['runs']}")
    print(f"Datasets: {', '.join(config['datasets'])}")
    print(f"Output directory: {config['output_path']}")
    
    # Calculate accuracy metrics
    results = AccuracyCalculator.calculate_accuracy_for_all_runs(
        data_loader=data_loader,
        runs=config['runs'],
        datasets=config['datasets']
    )
    
    # Ensure output directory exists (using absolute path)
    output_dir = Path(config['output_path']).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (absolute): {output_dir}")
    
    # Save results using the AccuracyCalculator's method
    results_dir = Path(config['results_path']).absolute()
    AccuracyCalculator.save_accuracy_results(
        results=results,
        output_dir=output_dir,
        save_to_results=True,
        results_path=config['results_path']
    )
    
    # Print summary
    print(f"\nAccuracy calculation complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    
    if results['aggregated']['by_dataset']:
        print("\nAggregated Accuracy by Dataset:")
        for dataset, stats in results['aggregated']['by_dataset'].items():
            print(f"- {dataset}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    
    if results['aggregated']['overall']:
        overall = results['aggregated']['overall']
        print(f"\nOverall Accuracy: {overall['mean_accuracy']:.4f} "
              f"({overall['total_correct']:,}/{overall['total_examples']:,} correct)")

if __name__ == "__main__":
    main()
