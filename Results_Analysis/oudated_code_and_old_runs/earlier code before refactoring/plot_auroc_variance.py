import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

# Configuration: runs to include and methods
runs = ["0", "1", "2", "3", "4"]
methods = ["probas-min-bl", "probas-min", "probas-mean-bl", "probas-mean", "token-sar-bl", "token-sar"]
datasets_order = ["HotpotQA", "2WikimhQA", "GSM8K", "SVAMP", "ASDiv"]


def load_data(base_dir):
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


def load_paper_data(base_dir):
    """Load and process paper AUROC data."""
    paper_path = base_dir / "auroc_scores_paper.csv"
    if not paper_path.exists():
        print(f"Warning: {paper_path} not found. Paper data will not be included.")
        return None
    
    # Mapping from paper method names to our format
    method_mapping = {
        'Probas-mean': 'probas-mean-bl',
        'Probas-mean w/ CoT-UQ': 'probas-mean',
        'Probas-min': 'probas-min-bl',
        'Probas-min w/ CoT-UQ': 'probas-min',
        'TOKENSAR': 'token-sar-bl',
        'TOKENSAR w/ CoT-UQ': 'token-sar'
    }
    
    df = pd.read_csv(paper_path)
    df = df[df['UQ_method'].isin(method_mapping.keys())].copy()
    df['UQ_method'] = df['UQ_method'].map(method_mapping)
    return df

def plot_variance(df, paper_df, output_path):
    # Determine y-axis limits considering both run and paper data
    all_vals = df['AUROC']
    if paper_df is not None:
        all_vals = pd.concat([all_vals, paper_df['AUROC']])
    y_min = max(0, all_vals.min() - 0.01)
    y_max = min(1, all_vals.max() + 0.01)

    n_datasets = len(datasets_order)
    ncols = 3
    nrows = int(np.ceil(n_datasets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), sharey=True)
    axes = axes.flatten()

    for i, dataset in enumerate(datasets_order):
        ax = axes[i]
        data_ds = df[df['dataset'].str.lower() == dataset.lower()]
        if data_ds.empty:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset)
            ax.set_xticks([])
            continue

        x = np.arange(len(methods))
        for j, method in enumerate(methods):
            vals_m = data_ds[data_ds['UQ_method'] == method]['AUROC'].values
            if vals_m.size == 0:
                continue
            # Plot min-max line for run data
            ax.vlines(j, vals_m.min(), vals_m.max(), color='lightblue', linewidth=3, alpha=0.7)
            # Plot individual run points with jitter
            jitter = np.random.normal(loc=j, scale=0.05, size=len(vals_m))
            ax.scatter(jitter, vals_m, color='blue', edgecolor='black', zorder=3, alpha=0.7, label='_nolegend_')
            # Plot mean as a horizontal line
            ax.hlines(np.mean(vals_m), j-0.2, j+0.2, color='red', linewidth=2, zorder=4)
            
            # Plot paper data if available
            if paper_df is not None:
                paper_vals = paper_df[(paper_df['dataset'].str.lower() == dataset.lower()) & 
                                    (paper_df['UQ_method'] == method)]['AUROC'].values
                if len(paper_vals) > 0:
                    ax.scatter(j, paper_vals[0], color='green', marker='X', s=60, zorder=5, label='_nolegend_')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(dataset)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('AUROC')

    # Remove any extra axes
    for k in range(n_datasets, nrows*ncols):
        fig.delaxes(axes[k])

    # Add legend
    handles = [
        Line2D([0], [0], color='lightblue', lw=3, alpha=0.7, label='Min-Max Range'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Run Values'),
        Line2D([0], [0], color='red', lw=2, label='Mean'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markeredgecolor='green', markersize=8, label='Paper Value')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    
    fig.suptitle("AUROC: Run Data vs Paper Data", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Saved variance plot to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    output_dir = base_dir / "figures/variance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "auroc_variance_boxplots.png"

    df_all = load_data(base_dir)
    paper_df = load_paper_data(base_dir)
    plot_variance(df_all, paper_df, output_path)
