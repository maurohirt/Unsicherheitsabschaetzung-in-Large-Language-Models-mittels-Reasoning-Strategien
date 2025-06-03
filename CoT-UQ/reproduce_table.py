import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

def create_auroc_table_figure(csv_file_path, output_image_path):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: {csv_file_path} not found. Please ensure auroc_scores.csv exists.")
        return

    # Filter for Llama 3.1-8B model
    df_llama3 = df[df['model_engine'] == 'llama3-1_8B'].copy()

    if df_llama3.empty:
        print("No Llama 3.1-8B data found in the CSV. Cannot reproduce table.")
        return

    print(f"Datasets found in Llama 3.1-8B data: {df_llama3['dataset'].unique()}")

    # Define mappings for AP strategy methods
    ap_mapping = {
        'Probas-mean': 'probas-mean-bl',
        'Probas-mean w/ CoT-UQ': 'probas-mean',
        'Probas-min': 'probas-min-bl',
        'Probas-min w/ CoT-UQ': 'probas-min',
        'TOKENSAR': 'token-sar-bl',
        'TOKENSAR w/ CoT-UQ': 'token-sar'
    }

    # Define groups for SE strategy methods (base and CoT-UQ variants)
    se_base_methods = {
        'P(True)': ['p-true-bl'],
        'Self-Probing': ['self-probing-bl']
    }

    se_full_methods_groups = {
        'P(True) w/ CoT-UQ': {'baseline': 'p-true-bl', 'cot_uq_variants': ['p-true-allsteps', 'p-true-keystep', 'p-true-allkeywords', 'p-true-keykeywords']},
        'Self-Probing w/ CoT-UQ': {'baseline': 'self-probing-bl', 'cot_uq_variants': ['self-probing-allsteps', 'self-probing-keystep', 'self-probing-allkeywords', 'self-probing-keykeywords']}
    }

    # Define the desired order of methods for the table display
    # Each tuple represents a pair of (base_method_name, cot_uq_method_name)
    method_display_order = [
        ('Probas-mean', 'Probas-mean w/ CoT-UQ'),
        ('Probas-min', 'Probas-min w/ CoT-UQ'),
        ('TOKENSAR', 'TOKENSAR w/ CoT-UQ'),
        ('P(True)', 'P(True) w/ CoT-UQ'),
        ('Self-Probing', 'Self-Probing w/ CoT-UQ')
    ]

    # Datasets order for columns (from paper, for consistent ordering)
    paper_datasets_order = ["HotpotQA", "2WikimhQA", "GSM8K", "SVAMP", "ASDiv"]
    # Convert unique datasets from df_llama3 to lowercase for case-insensitive comparison
    unique_datasets_lower = [d.lower() for d in df_llama3['dataset'].unique()]

    # Filter to only include datasets present in your data, maintaining original casing for table headers
    available_datasets = []
    for d_paper in paper_datasets_order:
        if d_paper.lower() in unique_datasets_lower:
            available_datasets.append(d_paper)

    print(f"Available datasets for table columns: {available_datasets}")

    # Initialize table data structure
    # Store (value, is_bold) tuples for each cell
    table_data_for_df = []

    for base_method_name, cot_uq_method_name in method_display_order:
        # Handle AP methods
        if base_method_name in ap_mapping:
            # Base AP method row
            row_base_ap = {'Method': base_method_name}
            # CoT-UQ AP method row
            row_cot_uq_ap = {'Method': cot_uq_method_name}

            for dataset_name in available_datasets:
                val_base = df_llama3[
                    (df_llama3['strategy'] == 'AP') &
                    (df_llama3['UQ_method'] == ap_mapping[base_method_name]) &
                    (df_llama3['dataset'].str.lower() == dataset_name.lower())
                ]['AUROC'].max() # Use max for robustness

                val_cot_uq = df_llama3[
                    (df_llama3['strategy'] == 'AP') &
                    (df_llama3['UQ_method'] == ap_mapping[cot_uq_method_name]) &
                    (df_llama3['dataset'].str.lower() == dataset_name.lower())
                ]['AUROC'].max() # Use max for robustness

                # Determine bolding for AP pair
                bold_base = False
                bold_cot_uq = False
                if pd.notna(val_base) and pd.notna(val_cot_uq):
                    if val_base > val_cot_uq:
                        bold_base = True
                    elif val_cot_uq > val_base:
                        bold_cot_uq = True

                row_base_ap[dataset_name] = (val_base * 100, bold_base, '') if pd.notna(val_base) else (np.nan, False, '')
                row_cot_uq_ap[dataset_name] = (val_cot_uq * 100, bold_cot_uq, '') if pd.notna(val_cot_uq) else (np.nan, False, '')

            table_data_for_df.append(row_base_ap)
            table_data_for_df.append(row_cot_uq_ap)

        # Handle SE methods
        elif base_method_name in se_base_methods:
            # Base SE method row
            row_base_se = {'Method': base_method_name}
            # CoT-UQ SE method row
            row_cot_uq_se = {'Method': cot_uq_method_name}

            for dataset_name in available_datasets:
                # Get value for base SE method
                val_base_se_series = df_llama3[
                    (df_llama3['strategy'] == 'SE') &
                    (df_llama3['UQ_method'].isin(se_base_methods[base_method_name])) &
                    (df_llama3['dataset'].str.lower() == dataset_name.lower())
                ].set_index('UQ_method')['AUROC']

                val_base_se = val_base_se_series.max() if not val_base_se_series.empty else np.nan
                base_se_indicator = val_base_se_series.idxmax().split('-')[-1] if not val_base_se_series.empty else ''

                # Get value for CoT-UQ SE method with exclusion logic
                methods_dict = se_full_methods_groups[cot_uq_method_name]
                all_se_candidates = [methods_dict['baseline']] + methods_dict['cot_uq_variants']

                auroc_series_all = df_llama3[
                    (df_llama3['strategy'] == 'SE') &
                    (df_llama3['UQ_method'].isin(all_se_candidates)) &
                    (df_llama3['dataset'].str.lower() == dataset_name.lower())
                ].set_index('UQ_method')['AUROC']

                val_cot_uq_se = np.nan
                cot_uq_se_indicator = ''

                if not auroc_series_all.empty:
                    overall_max_auroc_uq_method = auroc_series_all.idxmax()

                    if overall_max_auroc_uq_method == methods_dict['baseline']:
                        cot_uq_only_series = auroc_series_all[auroc_series_all.index.isin(methods_dict['cot_uq_variants'])]
                        if not cot_uq_only_series.empty:
                            val_cot_uq_se = cot_uq_only_series.max()
                            cot_uq_se_indicator = cot_uq_only_series.idxmax().split('-')[-1]
                    else:
                        val_cot_uq_se = auroc_series_all.max()
                        cot_uq_se_indicator = overall_max_auroc_uq_method.split('-')[-1]

                # Determine bolding for SE pair
                bold_base_se = False
                bold_cot_uq_se = False
                if pd.notna(val_base_se) and pd.notna(val_cot_uq_se):
                    if val_base_se > val_cot_uq_se:
                        bold_base_se = True
                    elif val_cot_uq_se > val_base_se:
                        bold_cot_uq_se = True

                row_base_se[dataset_name] = (val_base_se * 100, bold_base_se, base_se_indicator) if pd.notna(val_base_se) else (np.nan, False, '')
                row_cot_uq_se[dataset_name] = (val_cot_uq_se * 100, bold_cot_uq_se, cot_uq_se_indicator) if pd.notna(val_cot_uq_se) else (np.nan, False, '')

            table_data_for_df.append(row_base_se)
            table_data_for_df.append(row_cot_uq_se)

    print(f"Table data before DataFrame conversion: {table_data_for_df}")

    final_df = pd.DataFrame(table_data_for_df)

    # Reorder columns and remove 'Model' and 'Strategy' to match the paper's table structure
    final_columns = ['Method'] + available_datasets
    final_df = final_df[final_columns]

    # Prepare cell text and bolding flags for Matplotlib table
    cell_text = []
    cell_bold_flags = []

    # Add column headers
    header_row = list(final_df.columns)
    cell_text.append(header_row)
    cell_bold_flags.append([False] * len(header_row)) # Headers are not bolded by this logic

    # Add data rows
    for index, row in final_df.iterrows():
        current_text_row = []
        current_bold_row = []
        for col_name in final_df.columns:
            if col_name in ['Method']:
                current_text_row.append(row[col_name])
                current_bold_row.append(False)
            else: # Dataset columns contain (value, is_bold, indicator) tuples
                value, is_bold, indicator = row[col_name]
                if pd.notna(value):
                    if indicator:
                        current_text_row.append(f"{value:.2f} ({indicator})")
                    else:
                        current_text_row.append(f"{value:.2f}")
                    current_bold_row.append(is_bold)
                else:
                    current_text_row.append('-')
                    current_bold_row.append(False)
        cell_text.append(current_text_row)
        cell_bold_flags.append(current_bold_row)

    # --- Render DataFrame as a Matplotlib table and save as PNG ---
    fig, ax = plt.subplots(figsize=(12, 8)) # Adjust figure size as needed
    ax.axis('off') # Hide axes

    # Create the table
    table = ax.table(cellText=cell_text,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) # Adjust scaling for better readability

    # Apply styling to headers and bold specific cells
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0: # Header row (index 0 in cell_text)
            cell.set_facecolor('#D3D3D3') # Light gray background
            cell.set_text_props(weight='bold')
        elif j == 0: # The 'Method' column
            cell.set_facecolor('#F0F0F0') # Slightly lighter gray

        # Apply bolding based on cell_bold_flags
        # Note: cell_bold_flags has an extra header row at index 0
        if i > 0 and cell_bold_flags[i][j]: # Skip header row (i=0)
            cell.set_text_props(weight='bold')

    plt.title('Reproduced AUROC Table for Llama 3.1-8B (My Data)', fontsize=14, pad=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title cutoff

    # Create 'figures' directory if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    print(f"Table saved to {output_image_path}")

if __name__ == "__main__":
    # Configuration
    run_number = "2"  # Change this to match the run number you want to process (e.g., "0", "1", "2", etc.)
    
    # Setup paths
    base_dir = Path(__file__).parent
    auroc_csv_path = base_dir / f"auroc_scores/auroc_scores_run_{run_number}.csv"
    output_dir = base_dir / f"figures/run_{run_number}"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
    output_image_file = output_dir / f"reproduced_auroc_table_llama3_run_{run_number}.png"
    
    # Check if input file exists
    if not auroc_csv_path.exists():
        print(f"Error: {auroc_csv_path} not found. Please run parse_auroc.py with run_number={run_number} first.")
        exit()
    
    print(f"Processing AUROC scores from: {auroc_csv_path}")
    print(f"Saving output to: {output_image_file}")
    
    create_auroc_table_figure(auroc_csv_path, output_image_file)
