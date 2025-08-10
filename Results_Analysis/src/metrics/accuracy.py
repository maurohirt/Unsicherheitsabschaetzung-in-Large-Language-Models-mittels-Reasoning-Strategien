# src/metrics/accuracy.py
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

class AccuracyCalculator:
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return np.mean(y_true == y_pred)

    @classmethod
    def calculate_accuracy_for_run(
        cls,
        run_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate accuracy for a single run."""
        y_true = run_data['y_true']
        y_pred = run_data['y_pred']
        
        accuracy = cls.calculate_accuracy(y_true, y_pred)
        num_examples = len(y_true)
        num_correct = int(accuracy * num_examples)
        
        return {
            'accuracy': accuracy,
            'num_correct': num_correct,
            'num_total': num_examples
        }
    
    @classmethod
    def calculate_accuracy_for_all_runs(
        cls,
        data_loader: Any,
        runs: List[int],
        datasets: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate accuracy for all runs and datasets.
        
        Args:
            data_loader: Instance of DataLoader to load run data
            runs: List of run IDs to process
            datasets: List of dataset names to process
            
        Returns:
            Dictionary containing accuracy results
        """
        results = {'by_run': {}, 'aggregated': {'by_dataset': {}, 'overall': {}}}
        all_accuracies = {dataset: [] for dataset in datasets}
        total_correct = 0
        total_examples = 0
        
        # Calculate accuracy for each run and dataset
        for run_id in tqdm(runs, desc="Processing runs"):
            run_key = f'run_{run_id}'
            results['by_run'][run_key] = {}
            
            for dataset in datasets:
                try:
                    # Load and process data
                    data = data_loader.load_run_data(run_id, dataset)
                    processed = data_loader.get_ground_truth_and_predictions(data)
                    
                    # Calculate accuracy
                    metrics = cls.calculate_accuracy_for_run(processed)
                    results['by_run'][run_key][dataset] = metrics
                    all_accuracies[dataset].append(metrics['accuracy'])
                    
                    # Update totals
                    total_correct += metrics['num_correct']
                    total_examples += metrics['num_total']
                    
                except Exception as e:
                    print(f"Error processing run {run_id}, dataset {dataset}: {e}")
                    all_accuracies[dataset].append(np.nan)
        
        # Calculate aggregated statistics by dataset
        for dataset in datasets:
            accs = [r[dataset]['accuracy'] for r in results['by_run'].values() 
                   if dataset in r and not np.isnan(r[dataset].get('accuracy', np.nan))]
            if not accs:
                continue
                
            results['aggregated']['by_dataset'][dataset] = {
                'mean_accuracy': float(np.nanmean(accs)),
                'std_accuracy': float(np.nanstd(accs, ddof=1) if len(accs) > 1 else 0),
                'min_accuracy': float(np.nanmin(accs)),
                'max_accuracy': float(np.nanmax(accs)),
                'num_runs': len([x for x in accs if not np.isnan(x)])
            }
        
        # Calculate overall statistics
        if total_examples > 0:
            all_accs = []
            for run_data in results['by_run'].values():
                for dataset_data in run_data.values():
                    if 'accuracy' in dataset_data and not np.isnan(dataset_data['accuracy']):
                        all_accs.append(dataset_data['accuracy'])
            
            results['aggregated']['overall'] = {
                'mean_accuracy': total_correct / total_examples,
                'std_accuracy': float(np.nanstd(all_accs, ddof=1) if len(all_accs) > 1 else 0),
                'min_accuracy': float(np.nanmin(all_accs)) if all_accs else 0,
                'max_accuracy': float(np.nanmax(all_accs)) if all_accs else 0,
                'total_correct': total_correct,
                'total_examples': total_examples,
                'num_runs': len(runs),
                'num_datasets': len(datasets)
            }
        
        return results
    
    @staticmethod
    def save_accuracy_results(
        results: Dict,
        output_dir: Path,
        save_to_results: bool = True,
        results_path: str = 'results/accuracy'
    ) -> None:
        """
        Save accuracy results to files.
        
        Args:
            results: Results dictionary from calculate_accuracy_for_all_runs
            output_dir: Base output directory
            save_to_results: If True, also save to results/accuracy/
        """
        # Create results/accuracy directory if needed
        results_dir = Path(results_path).absolute()
        if save_to_results:
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save per-run results
        run_rows = []
        for run, datasets in results['by_run'].items():
            if save_to_results:
                run_dir = results_dir / run
                run_dir.mkdir(exist_ok=True)
                
                # Save per-dataset results for this run
                run_data = {}
                for dataset, metrics in datasets.items():
                    run_data[dataset] = {
                        'accuracy': metrics['accuracy'],
                        'num_correct': metrics['num_correct'],
                        'num_total': metrics['num_total']
                    }
                
                with open(run_dir / 'accuracy.json', 'w') as f:
                    json.dump(run_data, f, indent=2, default=float)
            
            # Add to run rows for combined CSV
            for dataset, metrics in datasets.items():
                run_rows.append({
                    'run': run,
                    'dataset': dataset,
                    'accuracy': metrics['accuracy'],
                    'num_correct': metrics['num_correct'],
                    'num_total': metrics['num_total']
                })
        
        # 2. Save all runs to a single CSV
        if run_rows:
            run_df = pd.DataFrame(run_rows)
            if save_to_results:
                run_df.to_csv(results_dir / 'all_runs_accuracy.csv', index=False, float_format='%.6f')
            run_df.to_csv(output_dir / 'accuracy_by_run.csv', index=False, float_format='%.6f')
        
        # 3. Save aggregated results
        agg_data = {
            'by_dataset': results['aggregated']['by_dataset'],
            'overall': results['aggregated']['overall']
        }
        
        if save_to_results:
            with open(results_dir / 'aggregated_accuracy.json', 'w') as f:
                json.dump(agg_data, f, indent=2, default=float)
        
        with open(output_dir / 'aggregated_accuracy.json', 'w') as f:
            json.dump(agg_data, f, indent=2, default=float)
        
        # 4. Create detailed markdown report
        markdown_content = "# Accuracy Results\n\n"
        
        # Per-run section
        markdown_content += "## Per-Run Accuracy\n\n"
        markdown_content += "| Run | Dataset | Accuracy | Correct/Total |\n"
        markdown_content += "|-----|---------|---------:|---------------:|\n"
        for row in run_rows:
            markdown_content += f"| {row['run']} | {row['dataset']} | {row['accuracy']:.4f} | {row['num_correct']:,}/{row['num_total']:,} |\n"
        
        # Aggregated by dataset
        markdown_content += "\n## Aggregated by Dataset\n\n"
        markdown_content += "| Dataset | Mean | Std Dev | Min | Max | Runs |\n"
        markdown_content += "|---------|-----:|--------:|----:|----:|-----:|\n"
        
        for dataset, stats in results['aggregated']['by_dataset'].items():
            markdown_content += (
                f"| {dataset} | {stats['mean_accuracy']:.4f} | "
                f"{stats['std_accuracy']:.4f} | {stats['min_accuracy']:.4f} | "
                f"{stats['max_accuracy']:.4f} | {stats['num_runs']} |\n"
            )
        
        # Overall summary
        if results['aggregated']['overall']:
            overall = results['aggregated']['overall']
            markdown_content += "\n## Overall Summary\n\n"
            markdown_content += f"- **Mean Accuracy**: {overall['mean_accuracy']:.4f}\n"
            markdown_content += f"- **Std Dev**: {overall['std_accuracy']:.4f}\n"
            markdown_content += f"- **Range**: {overall['min_accuracy']:.4f} - {overall['max_accuracy']:.4f}\n"
            markdown_content += f"- **Total Correct**: {overall['total_correct']:,}\n"
            markdown_content += f"- **Total Examples**: {overall['total_examples']:,}\n"
            markdown_content += f"- **Overall Accuracy**: {overall['mean_accuracy']:.4f} "
            markdown_content += f"({overall['total_correct']:,}/{overall['total_examples']:,})\n"
        
        # Save markdown report
        if save_to_results:
            with open(results_dir / 'accuracy_report.md', 'w') as f:
                f.write(markdown_content)
        
        with open(output_dir / 'accuracy_summary.md', 'w') as f:
            f.write(markdown_content)
