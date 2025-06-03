import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration: runs to include and methods
runs = ["0", "1", "2"]
methods = [
    "self-probing-bl",
    "self-probing-allsteps",
    "self-probing-keystep",
    "self-probing-allkeywords",
    "self-probing-keykeywords"
]
# Define display names for methods (for x-axis labels)
method_display_names = {
    "self-probing-bl": "Baseline",
    "self-probing-allsteps": "All Steps",
    "self-probing-keystep": "Key Step",
    "self-probing-allkeywords": "All Keywords",
    "self-probing-keykeywords": "Key Keywords"
}
# Order of datasets to display
datasets_order = ["HotpotQA", "2WikimhQA", "GSM8K", "SVAMP", "ASDiv"]

def load_data(base_dir):
    """Load AUROC data from all runs."""
    all_dfs = []
    for run in runs:
        csv_path = base_dir / f"auroc_scores/auroc_scores_run_{run}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping run {run}")
            continue
        df = pd.read_csv(csv_path)
        df = df[df['UQ_method'].isin(methods)].copy()
        df['run'] = run
        all_dfs.append(df)
    if not all_dfs:
        print("No data loaded for any run. Exiting.")
        exit(1)
    return pd.concat(all_dfs, ignore_index=True)

def plot_selfprobing_variance(df, output_path):
    """Create a plot showing variance in self-probing methods across runs."""
    # Determine y-axis limits
    y_min = max(0, df['AUROC'].min() - 0.01)
    y_max = min(1, df['AUROC'].max() + 0.01)

    # Create subplots
    n_datasets = len(datasets_order)
    ncols = 3  # Number of columns in the subplot grid
    nrows = int(np.ceil(n_datasets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), sharey=True)
    
    # Flatten axes for easier iteration
    if n_datasets > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, dataset in enumerate(datasets_order):
        ax = axes[i]
        data_ds = df[df['dataset'].str.lower() == dataset.lower()]
        
        if data_ds.empty:
            ax.text(0.5, 0.5, f"No data for {dataset}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset)
            ax.set_xticks([])
            continue

        # Plot data for each method
        x = np.arange(len(methods))
        for j, method in enumerate(methods):
            # Get values for this method
            vals = data_ds[data_ds['UQ_method'] == method]['AUROC'].values
            if len(vals) == 0:
                continue
                
            # Plot min-max line
            ax.vlines(j, vals.min(), vals.max(), color='lightblue', linewidth=3, alpha=0.7)
            
            # Plot individual points with slight jitter
            jitter = np.random.normal(loc=j, scale=0.05, size=len(vals))
            ax.scatter(jitter, vals, color='blue', edgecolor='black', alpha=0.7, 
                      zorder=3, label='_nolegend_')
            
            # Plot mean as a horizontal line
            ax.hlines(np.mean(vals), j-0.2, j+0.2, color='red', linewidth=2, zorder=4)

        # Set x-ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels([method_display_names.get(m, m) for m in methods], 
                          rotation=45, ha='right')
        ax.set_title(dataset)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('AUROC')
        ax.grid(True, alpha=0.3)

    # Remove any extra axes
    for k in range(n_datasets, nrows * ncols):
        fig.delaxes(axes[k])

    # Add a single legend
    handles = [
        plt.Line2D([0], [0], color='lightblue', lw=3, alpha=0.7, label='Min-Max Range'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Run Values'),
        plt.Line2D([0], [0], color='red', lw=2, label='Mean')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    
    # Adjust layout and save
    fig.suptitle("AUROC Variance in Self-Probing Methods Across Runs", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved self-probing variance plot to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    output_dir = base_dir / "figures/variance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selfprobing_variance.png"

    df_all = load_data(base_dir)
    plot_selfprobing_variance(df_all, output_path)
