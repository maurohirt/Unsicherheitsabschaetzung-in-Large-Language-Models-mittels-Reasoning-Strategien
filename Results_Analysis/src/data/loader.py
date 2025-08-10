import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

class DataLoader:
    """Loads and processes experiment data from JSON files."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration.
        
        Args:
            config: Configuration dictionary with data paths and settings
        """
        self.config = config
        self.base_path = Path(config['data_path'])
    
    def load_run_data(self, run_id: int, dataset: str) -> Dict[str, Any]:
        """Load data for a specific run and dataset.
        
        Args:
            run_id: ID of the run to load
            dataset: Name of the dataset
            
        Returns:
            Dictionary containing the loaded data with keys:
            - examples: List of example dicts with questions, answers, and labels
            - {metric}_confidences: Confidence scores for each metric
        """
        print(f"\n{'='*50}")
        print(f"Loading data for run {run_id}, dataset: {dataset}")
        print(f"Base path: {self.base_path}")
        
        run_dir = self.base_path / f"run_{run_id}" / self.config['model'] / dataset
        print(f"Looking for data in: {run_dir}")
        
        if not run_dir.exists():
            print(f"ERROR: Run directory does not exist: {run_dir}")
            return {}
        
        # Initialize data dictionary
        data = {'examples': []}
        
        # Load the main output file with labels (one JSON object per line)
        output_file = run_dir / "output_v1_w_labels.json"
        print(f"Loading main output file: {output_file}")
        
        if not output_file.exists():
            print(f"ERROR: Main output file not found: {output_file}")
            return {}
            
        try:
            with open(output_file, 'r') as f:
                # Read all lines and parse each as JSON
                data['examples'] = [json.loads(line) for line in f if line.strip()]
                print(f"Loaded {len(data['examples'])} examples from {output_file}")
                
                # Print first example for verification
                if data['examples']:
                    print("\nFirst example:")
                    print(f"  Question: {data['examples'][0].get('question', 'N/A')}")
                    print(f"  Correct: {data['examples'][0].get('label', 'N/A')}")
                    print(f"  LLM Answer: {data['examples'][0].get('llm answer', 'N/A')}")
                    print(f"  Correct Answer: {data['examples'][0].get('correct answer', 'N/A')}")
                    
        except json.JSONDecodeError as e:
            print(f"Error parsing {output_file}: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        # Load confidence files for different uncertainty metrics
        conf_dir = run_dir / "confidences"
        print(f"\nLooking for confidence files in: {conf_dir}")
        
        if not conf_dir.exists():
            print(f"WARNING: Confidence directory not found: {conf_dir}")
            return data
            
        conf_files = list(conf_dir.glob("output_v1_*.json"))
        print(f"Found {len(conf_files)} confidence files")
        
        for conf_file in conf_files:
            if conf_file.name == "output_v1_w_labels.json":
                continue  # Skip the main file
                
            metric_name = conf_file.stem.replace("output_v1_", "")
            print(f"\nProcessing confidence file: {conf_file.name}")
            print(f"  Metric name: {metric_name}")
            
            try:
                with open(conf_file, 'r') as f:
                    # Each line is a JSON object with question, answer, and confidence
                    conf_entries = [json.loads(line) for line in f if line.strip()]
                
                print(f"  Found {len(conf_entries)} confidence entries")
                if conf_entries:
                    print(f"  First entry - Question: {conf_entries[0].get('question', 'N/A')}")
                    print(f"                Confidence: {conf_entries[0].get('confidence', 'N/A')}")
                
                # Create a mapping from question to confidence for this metric
                conf_dict = {entry['question']: entry['confidence'] for entry in conf_entries}
                
                # Match confidences with examples by question
                confidences = []
                matched = 0
                for example in data['examples']:
                    question = example['question']
                    conf = conf_dict.get(question, None)
                    if conf is None:
                        print(f"  WARNING: No confidence score found for question: {question[:50]}...")
                        conf = 0.0  # Default to 0.0 if not found
                    else:
                        matched += 1
                    confidences.append(conf)
                
                print(f"  Successfully matched {matched}/{len(data['examples'])} examples")
                data[f"{metric_name}_confidences"] = confidences
                
            except Exception as e:
                print(f"ERROR loading {conf_file}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nFinished loading data for run {run_id}, dataset {dataset}")
        print(f"Available metrics in data: {[k for k in data.keys() if k.endswith('_confidences')]}")
        print(f"{'='*50}\n")
        
        return data
    
    def get_ground_truth_and_predictions(self, data: Dict) -> Dict[str, np.ndarray]:
        """Extract ground truth and predictions from loaded data.
        
        Args:
            data: Loaded data dictionary with 'examples' and confidence scores
            
        Returns:
            Dictionary with:
            - y_true: Ground truth labels (1 for correct, 0 for incorrect)
            - y_pred: Predicted labels (1 for correct, 0 for incorrect)
            - confidences: Dictionary of confidence scores for each metric
        """
        if 'examples' not in data or not data['examples']:
            raise ValueError("No examples found in the loaded data")
            
        # Extract ground truth (1 for correct, 0 for incorrect)
        y_true = np.array([1 if item.get('label') is True else 0 for item in data['examples']])
        
        # For prediction, we'll assume the model's answer is correct if label is True
        # In a real scenario, you might want to compare with ground truth
        y_pred = np.ones_like(y_true)  # Assume all predictions are correct
        
        # Get confidence scores for each uncertainty metric
        confidences = {}
        for key in data.keys():
            if key.endswith('_confidences') and isinstance(data[key], (list, np.ndarray)):
                metric_name = key.replace('_confidences', '')
                conf_data = data[key]
                
                # Convert to numpy array and ensure proper shape
                if isinstance(conf_data, list):
                    conf_data = np.array(conf_data)
                
                # If we have per-token confidences, take mean or max
                if len(conf_data.shape) > 1:
                    # Take mean confidence across tokens for each example
                    conf_data = np.mean(conf_data, axis=1)
                
                # Ensure we have the right number of confidence scores
                if len(conf_data) != len(y_true):
                    print(f"Warning: Mismatch in confidence scores length for {metric_name} "
                          f"({len(conf_data)} vs {len(y_true)} examples)")
                    continue
                    
                confidences[metric_name] = conf_data
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'confidences': confidences
        }
