#!/usr/bin/env python3
# scripts/visualization/plot_accuracy_table.py
"""
Script for generating a clean table of aggregated accuracy metrics.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def load_aggregated_results(results_dir: Path) -> Dict[str, Any]:
    """Load aggregated accuracy results from JSON file."""
    agg_file = results_dir / 'accuracy' / 'aggregated_accuracy.json'
    with open(agg_file, 'r') as f:
        return json.load(f)

def create_accuracy_table(agg_results: Dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame with the aggregated accuracy results."""
    # Extract dataset metrics
    datasets = []
    for dataset, metrics in agg_results['by_dataset'].items():
        row = {
            'Dataset': dataset,
            'Mean Accuracy': metrics['mean_accuracy'],
            'Std Dev': metrics['std_accuracy'],
            'Min': metrics['min_accuracy'],
            'Max': metrics['max_accuracy'],
            'Runs': metrics['num_runs']
        }
        datasets.append(row)
    
    # Create DataFrame and sort by Mean Accuracy (descending)
    df = pd.DataFrame(datasets)
    df = df.sort_values('Mean Accuracy', ascending=False)
    
    # Format the Mean ± Std as a string
    df['Accuracy'] = df.apply(
        lambda x: f"{x['Mean Accuracy']:.4f} ± {x['Std Dev']:.4f}", 
        axis=1
    )
    
    # Select and reorder columns
    df = df[['Dataset', 'Accuracy', 'Min', 'Max', 'Runs']]
    
    return df

def save_table_as_image(df: pd.DataFrame, output_path: Path, title: str = "Accuracy by Dataset"):
    """Save the DataFrame as an image."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f3f3f3'] * len(df.columns)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for (i, col) in enumerate(df.columns):
        cell = table[0, i]
        cell.set_facecolor('#4f81bd')
        cell.set_text_props(weight='bold', color='white')
    
    # Add title
    plt.title(title, fontsize=14, pad=20)
    
    # Save figure
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_table_as_latex(df: pd.DataFrame, output_path: Path):
    """Save the DataFrame as a LaTeX table."""
    # Format numbers for LaTeX
    df_formatted = df.copy()
    for col in ['Min', 'Max']:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}")
    
    # Convert to LaTeX
    latex_str = df_formatted.to_latex(
        index=False,
        column_format='lrrrrr',
        escape=False,
        float_format="%.4f",
        caption="Aggregated accuracy metrics across different datasets.",
        label="tab:accuracy_metrics"
    )
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)

def main():
    # Set paths
    results_dir = Path('results')
    output_dir = results_dir / 'tables' / 'accuracy'
    
    # Load results
    agg_results = load_aggregated_results(results_dir)
    
    # Create table
    df = create_accuracy_table(agg_results)
    
    # Save as CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'accuracy_table.csv'
    df.to_csv(csv_path, index=False)
    
    # Save as image
    img_path = output_dir / 'accuracy_table.png'
    save_table_as_image(df, img_path)
    
    # Save as LaTeX
    latex_path = output_dir / 'accuracy_table.tex'
    save_table_as_latex(df, latex_path)
    
    print(f"Tables saved to: {output_dir.absolute()}")
    print("\nAccuracy Table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
