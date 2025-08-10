import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
import scipy.stats as stats
from .base import BaseMetric

class AUROC(BaseMetric):
    """
    Enhanced AUROC metric with support for:
    - Per-run AUROC calculation
    - Aggregation of results across multiple runs
    - Confidence interval calculation
    - Saving/loading results
    """
    
    def __init__(self):
        super().__init__("auroc")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate AUROC score and ROC curve for a single run.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (between 0 and 1)
            
        Returns:
            Dictionary containing:
                - score: AUROC score
                - fpr: False Positive Rate array
                - tpr: True Positive Rate array
                - thresholds: Thresholds array
        """
        try:
            # Ensure inputs are numpy arrays
            y_true = np.asarray(y_true)
            y_prob = np.asarray(y_prob)
            
            # Handle edge cases
            if len(np.unique(y_true)) == 1:
                raise ValueError("Only one class present in y_true. ROC AUC is not defined in that case.")
                
            # Calculate AUROC and ROC curve
            score = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            # Ensure the curve starts at (0,0) and ends at (1,1)
            fpr = np.concatenate([[0], fpr, [1]])
            tpr = np.concatenate([[0], tpr, [1]])
            
            return {
                'score': float(score),  # Convert numpy float to Python float
                'fpr': fpr.tolist(),   # Convert to list for JSON serialization
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'n_samples': len(y_true)
            }
            
        except Exception as e:
            print(f"Error calculating AUROC: {e}")
            return {
                'score': float('nan'),
                'fpr': [],
                'tpr': [],
                'thresholds': [],
                'error': str(e),
                'n_samples': len(y_true) if 'y_true' in locals() else 0
            }
    
    @staticmethod
    def aggregate_results(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate AUROC results across multiple runs.
        
        Args:
            run_results: List of per-run AUROC results
            
        Returns:
            Dictionary containing aggregated statistics:
                - mean_auroc: Mean AUROC across runs
                - std_auroc: Standard deviation of AUROC scores
                - median_auroc: Median AUROC across runs
                - min_auroc: Minimum AUROC score
                - max_auroc: Maximum AUROC score
                - confidence_interval: 95% confidence interval for AUROC
                - n_runs: Number of runs used for aggregation
                - n_samples: Total number of samples across all runs
        """
        if not run_results:
            raise ValueError("No results provided for aggregation")
            
        # Extract valid AUROC scores
        valid_results = [r for r in run_results if not np.isnan(r.get('score', np.nan))]
        
        if not valid_results:
            return {
                'mean_auroc': float('nan'),
                'std_auroc': float('nan'),
                'median_auroc': float('nan'),
                'min_auroc': float('nan'),
                'max_auroc': float('nan'),
                'confidence_interval': [float('nan'), float('nan')],
                'n_runs': 0,
                'n_samples': 0
            }
        
        # Calculate basic statistics
        scores = [r['score'] for r in valid_results]
        n_runs = len(valid_results)
        total_samples = sum(r.get('n_samples', 0) for r in valid_results)
        
        mean_auroc = float(np.mean(scores))
        std_auroc = float(np.std(scores, ddof=1))  # Sample standard deviation
        median_auroc = float(np.median(scores))
        min_auroc = float(np.min(scores))
        max_auroc = float(np.max(scores))
        
        # Calculate 95% confidence interval
        if n_runs > 1:
            ci = stats.t.interval(0.95, n_runs-1, loc=mean_auroc, scale=stats.sem(scores))
            confidence_interval = [float(ci[0]), float(ci[1])]
        else:
            confidence_interval = [mean_auroc, mean_auroc]
        
        # Interpolate ROC curves to common FPR points for mean ROC curve
        all_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        
        for result in valid_results:
            if 'fpr' in result and 'tpr' in result and len(result['fpr']) > 0 and len(result['tpr']) > 0:
                fpr = np.asarray(result['fpr'])
                tpr = np.asarray(result['tpr'])
                interp_tpr = np.interp(all_fpr, fpr, tpr, left=0.0, right=1.0)
                interp_tprs.append(interp_tpr)
        
        # Calculate mean and std of TPRs if we have valid curves
        mean_tpr = np.mean(interp_tprs, axis=0).tolist() if interp_tprs else []
        std_tpr = np.std(interp_tprs, axis=0, ddof=1).tolist() if interp_tprs else []
        
        return {
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'median_auroc': median_auroc,
            'min_auroc': min_auroc,
            'max_auroc': max_auroc,
            'confidence_interval': confidence_interval,
            'individual_scores': scores,
            'mean_fpr': all_fpr.tolist(),
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'n_runs': n_runs,
            'n_samples': total_samples
        }
    
    @classmethod
    def save_results(cls, results: Dict[str, Any], output_path: Path) -> None:
        """
        Save AUROC results to a JSON file.
        
        Args:
            results: Results dictionary to save
            output_path: Path to save the results to (including .json extension)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_serialization(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_serialization(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_serialization(x) for x in obj]
            return obj
        
        serializable_results = convert_for_serialization(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    @classmethod
    def load_results(cls, file_path: Path) -> Dict[str, Any]:
        """
        Load AUROC results from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Deserialized results dictionary
        """
        with open(file_path, 'r') as f:
            return json.load(f)
