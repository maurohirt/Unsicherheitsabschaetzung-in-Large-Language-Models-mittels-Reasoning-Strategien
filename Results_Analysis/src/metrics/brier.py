import numpy as np
from typing import Dict, Any
from .base import BaseMetric

class BrierScore(BaseMetric):
    """Brier Score metric."""
    
    def __init__(self):
        super().__init__("brier")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Brier score.
        
        The Brier score is the mean squared difference between the predicted
        probability and the actual outcome, so the best possible score is 0.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (between 0 and 1)
            
        Returns:
            Dictionary containing:
                - score: Brier score
        """
        try:
            # Brier score = 1/N * sum((y_true - y_prob)^2)
            score = np.mean((y_true - y_prob) ** 2)
            return {'score': score}
        except Exception as e:
            print(f"Error calculating Brier score: {e}")
            return {'score': np.nan}
