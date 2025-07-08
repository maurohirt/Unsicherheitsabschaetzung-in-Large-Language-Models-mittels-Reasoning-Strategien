import re
import pandas as pd
import os
from pathlib import Path

def parse_log_file(log_file_path):
    data = []
    with open(log_file_path, 'r') as f:
        content = f.read()

    # Split the content into individual analysis blocks
    analysis_blocks = re.split(r'============================================================\nStarting: Analysis for ', content)

    for block in analysis_blocks:
        if not block.strip():
            continue

        model_engine = None
        dataset = None
        uq_engine = None
        auroc = None

        # Extract model_engine, dataset, and uq_engine from the command line
        command_match = re.search(r'Command: python analyze_result\.py --dataset (\S+) --model_engine (\S+) --uq_engine (\S+)', block)
        if command_match:
            dataset = command_match.group(1)
            model_engine = command_match.group(2)
            uq_engine = command_match.group(3)

        # Extract AUROC
        auroc_match = re.search(r'AUROC: (\S+)', block)
        if auroc_match:
            auroc = float(auroc_match.group(1))

        if all([model_engine, dataset, uq_engine, auroc]):
            strategy = "SE" if "self-probing" in uq_engine or "p-true" in uq_engine else "AP"
            # Determine reasoning_type based on dataset
            if dataset in ["HotpotQA", "2WikimhQA"]:
                reasoning_type = 'logical'
            else:
                reasoning_type = 'mathematical'
            data.append({
                'model_engine': model_engine,
                'dataset': dataset,
                'reasoning_type': reasoning_type,
                'strategy': strategy,
                'UQ_method': uq_engine,
                'AUROC': auroc
            })

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Configuration
    run_number = "4"  # Processing run 4
    
    # Setup paths
    base_dir = Path(__file__).parent
    log_directory = base_dir / f"runs_logs/logs_Lama3_run_{run_number}"
    output_dir = base_dir / "auroc_scores"
    output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
    output_csv = output_dir / f"auroc_scores_run_{run_number}.csv"

    # Delete the existing output file if it exists
    if output_csv.exists():
        output_csv.unlink()
        print(f"Deleted existing {output_csv}")

    all_data_frames = []

    # Check if log directory exists
    if not log_directory.exists():
        print(f"Error: Log directory not found: {log_directory}")
        exit(1)

    # Iterate over all files in the log directory
    for filename in os.listdir(log_directory):
        if filename.endswith(".out"):
            log_file_path = log_directory / filename
            print(f"Processing {log_file_path}...")
            df_from_file = parse_log_file(str(log_file_path))
            if not df_from_file.empty:
                all_data_frames.append(df_from_file)

    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        # Remove duplicates based on all columns across all processed files
        combined_df.drop_duplicates(inplace=True)

        # Save to CSV
        combined_df.to_csv(output_csv, index=False)
        print(f"AUROC scores saved to {output_csv}")
        print(combined_df.head())
        print(f"Total unique entries: {len(combined_df)}")
    else:
        print(f"No AUROC scores found in any .out files in {log_directory}")
