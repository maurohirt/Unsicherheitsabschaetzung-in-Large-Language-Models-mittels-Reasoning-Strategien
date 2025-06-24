import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# The main plotting function, now tailored for a single subplot
def plot_auroc_subplot(df, uq_method_type, dataset_name, ax):
    # Filter for the specific UQ method type and dataset
    filtered_df = df[
        (df['UQ_method'].str.contains(uq_method_type, case=False, na=False)) &
        (df['dataset'] == dataset_name)
    ].copy()

    if filtered_df.empty:
        ax.text(0.5, 0.5, f"No data for {dataset_name}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Dataset: {dataset_name}')
        ax.set_xlabel('Model Engine')
        ax.set_ylabel('AUROC')
        ax.tick_params(axis='x', rotation=0)
        return None # Indicate no plot was made, so no handles/labels to return

    # Plotting on the given axes
    sns.barplot(x='model_engine', y='AUROC', hue='UQ_method', data=filtered_df, palette='viridis', ax=ax)

    ax.set_title(f'Dataset: {dataset_name}')
    ax.set_xlabel('Model Engine')
    ax.set_ylabel('AUROC')
    ax.tick_params(axis='x', rotation=0) # No rotation needed

    # Remove legend from individual subplots, we'll create a single one later
    ax.get_legend().remove()

    # Return handles and labels for the combined legend
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


if __name__ == "__main__":
    # Configuration
    run_number = "4"  # Processing run 4
    
    # Setup paths
    base_dir = Path(__file__).parent
    csv_file = base_dir / f"auroc_scores/auroc_scores_run_{run_number}.csv"
    output_dir = base_dir / f"figures/run_{run_number}"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run parse_auroc.py with run_number={run_number} first.")
        exit()

    unique_datasets = df['dataset'].unique()
    if len(unique_datasets) == 0:
        print("No datasets found in the CSV. Cannot generate plots.")
        exit()

    uq_method_types = ["p-true", "self-probing"]

    for uq_type in uq_method_types:
        # Filter data for the current UQ method type across all datasets to determine global y-limits
        data_for_uq_type = df[df['UQ_method'].str.contains(uq_type, case=False, na=False)]

        if data_for_uq_type.empty:
            print(f"No data for {uq_type} UQ methods. Skipping plot generation.")
            continue

        min_auroc = data_for_uq_type['AUROC'].min()
        max_auroc = data_for_uq_type['AUROC'].max()

        # Add a small margin to the min/max values, ensuring limits stay within [0, 1]
        y_lower = max(0.0, min_auroc - 0.0125)
        y_upper = min(1.0, max_auroc + 0.0125)

        # Create a figure with subplots for all datasets
        # Adjust figsize based on the number of datasets
        fig, axes = plt.subplots(1, len(unique_datasets), figsize=(5 * len(unique_datasets), 7), sharey=True)

        # Ensure axes is an array even for a single subplot
        if len(unique_datasets) == 1:
            axes = [axes]

        all_handles = []
        all_labels = []

        for i, dataset_name in enumerate(unique_datasets):
            result = plot_auroc_subplot(df, uq_type, dataset_name, axes[i])
            if result:
                handles, labels = result
                # Collect handles and labels from the first successful plot to create a single legend
                if not all_handles: # Only collect once
                    all_handles = handles
                    all_labels = labels

        # Set the global y-limits for all subplots
        axes[0].set_ylim(y_lower, y_upper)

        # Add an overall title for the figure
        fig.suptitle(f'AUROC Scores for {uq_type.replace("-", " ").title()} UQ Methods', fontsize=16, y=1.02)

        # Place a single legend for the entire figure below all subplots
        if all_handles: # Only add legend if there was data to plot
            fig.legend(all_handles, all_labels, title='UQ Method', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(all_labels))

        plt.tight_layout(rect=[0, 0.15, 1, 0.9]) # Adjust rect to make space for legend and suptitle (increased bottom, decreased top)

        # Save the figure
        output_file = output_dir / f"auroc_{uq_type}_all_datasets.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    print("All plots generated successfully.")
