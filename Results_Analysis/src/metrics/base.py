from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate the metric.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (between 0 and 1)
            
        Returns:
            Dictionary containing the metric value and any additional information
        """
        pass
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Alias for calculate method."""
        return self.calculate(y_true, y_pred, y_prob)
