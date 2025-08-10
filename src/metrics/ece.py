import numpy as np
from typing import Dict, Any
from .base import BaseMetric

class ECE(BaseMetric):
    """Expected Calibration Error (ECE) metric."""
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize ECE metric.
        
        Args:
            n_bins: Number of bins to use for probability calibration
        """
        super().__init__("ece")
        self.n_bins = n_bins
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between the predicted probability and the 
        actual accuracy in probability bins.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (between 0 and 1)
            
        Returns:
            Dictionary containing:
                - score: ECE score
                - bin_edges: Edges of the probability bins
                - bin_acc: Accuracy in each bin
                - bin_conf: Average confidence in each bin
                - bin_counts: Number of samples in each bin
        """
        try:
            # Create bins
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
            bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=True)
            
            bin_acc = np.zeros(self.n_bins)
            bin_conf = np.zeros(self.n_bins)
            bin_counts = np.zeros(self.n_bins, dtype=np.int32)
            
            # Calculate accuracy and confidence per bin
            for i in range(self.n_bins):
                mask = (bin_indices == i)
                if np.any(mask):
                    bin_acc[i] = np.mean(y_true[mask])
                    bin_conf[i] = np.mean(y_prob[mask])
                    bin_counts[i] = np.sum(mask)
            
            # Calculate ECE
            ece = np.sum(np.abs(bin_acc - bin_conf) * bin_counts) / np.sum(bin_counts)
            
            return {
                'score': ece,
                'bin_edges': bin_edges,
                'bin_acc': bin_acc,
                'bin_conf': bin_conf,
                'bin_counts': bin_counts
            }
            
        except Exception as e:
            print(f"Error calculating ECE: {e}")
            return {
                'score': np.nan,
                'bin_edges': np.array([]),
                'bin_acc': np.array([]),
                'bin_conf': np.array([]),
                'bin_counts': np.array([], dtype=np.int32)
            }
