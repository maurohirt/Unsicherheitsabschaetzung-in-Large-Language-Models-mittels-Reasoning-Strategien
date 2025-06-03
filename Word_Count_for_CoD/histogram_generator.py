import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_histograms():
    base_dir = "/Users/maurohirt/Desktop/Bachelorarbeit/Outputs_BA/Word_Count_for_CoD/"
    csv_dir = os.path.join(base_dir, "word_count_csv")
    hist_output_dir = os.path.join(base_dir, "histograms")

    # Create the histograms output directory if it doesn't exist
    os.makedirs(hist_output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(csv_dir, "*_word_counts.csv"))

    if not csv_files:
        print(f"No CSV files found in {csv_dir}. Exiting.")
        return

    all_step_data_for_overall_avg = {} # New: { 'Step 1': [counts], 'Step 2': [counts], ... }

    # --- Generate 5 histograms (one for each dataset) ---
    for csv_file_path in csv_files:
        # Extract dataset name from CSV filename (e.g., "2wikimhQA" from "2wikimhQA_word_counts.csv")
        dataset_name = os.path.basename(csv_file_path).replace("_word_counts.csv", "")
        print(f"\nProcessing {dataset_name} for individual histogram...")

        try:
            df = pd.read_csv(csv_file_path)
            if df.empty:
                print(f"  Warning: {csv_file_path} is empty or contains no data. Skipping.")
                continue
        except pd.errors.EmptyDataError:
            print(f"  Warning: {csv_file_path} is empty. Skipping.")
            continue
        except Exception as e:
            print(f"  Error reading {csv_file_path}: {e}. Skipping.")
            continue

        # Identify columns that represent step word counts
        step_columns = sorted([col for col in df.columns if col.startswith("Step ") and col.endswith(" Words")],
                              key=lambda x: int(x.split(" ")[1])) # Sort like Step 1, Step 2...
        
        if not step_columns:
            print(f"  No step columns (e.g., 'Step X Words') found in {dataset_name}. Skipping histogram generation for this file.")
            continue

        average_words_per_step = {}
        for step_col in step_columns:
            # Ensure column is numeric, coercing errors to NaN, then drop NaNs for mean calculation
            numeric_counts = pd.to_numeric(df[step_col], errors='coerce').dropna()
            
            if not numeric_counts.empty:
                average_words_per_step[step_col.replace(" Words", "")] = numeric_counts.mean()
                
                # Accumulate data for overall average
                step_key_for_overall = step_col.replace(" Words", "") # e.g. "Step 1"
                if step_key_for_overall not in all_step_data_for_overall_avg:
                    all_step_data_for_overall_avg[step_key_for_overall] = []
                all_step_data_for_overall_avg[step_key_for_overall].extend(numeric_counts.astype(int).tolist())
            else:
                # If a step column has no valid numbers, its average is 0 or could be skipped
                average_words_per_step[step_col.replace(" Words", "")] = 0 

        if not average_words_per_step:
            print(f"  No valid step data to plot for {dataset_name}.")
            continue
            
        step_labels_for_plot = list(average_words_per_step.keys())
        average_values_for_plot = list(average_words_per_step.values())

        plt.figure(figsize=(12, 7)) # Adjusted figure size
        bars = plt.bar(step_labels_for_plot, average_values_for_plot, color='skyblue', width=0.6)
        plt.xlabel("Step Label", fontsize=12)
        plt.ylabel("Average Number of Words", fontsize=12)
        plt.title(f"Average Words per Step for {dataset_name}", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        # Add text labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(average_values_for_plot, default=1), f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

        hist_save_path = os.path.join(hist_output_dir, f"{dataset_name}_avg_words_per_step.png")
        plt.savefig(hist_save_path)
        plt.close() 
        print(f"  Saved histogram for {dataset_name} to {hist_save_path}")

    # --- Generate the 6th histogram (overall average words per step number) ---
    if all_step_data_for_overall_avg:
        print("\nGenerating overall average words per step histogram...")
        
        # Calculate overall average for each step number
        overall_avg_words_per_step_label = {}
        for step_label, counts_list in all_step_data_for_overall_avg.items():
            if counts_list: # Ensure there are counts to average
                overall_avg_words_per_step_label[step_label] = sum(counts_list) / len(counts_list)
            else:
                overall_avg_words_per_step_label[step_label] = 0
        
        # Sort by step number for plotting (e.g., Step 1, Step 2, ...)
        sorted_step_labels_overall = sorted(overall_avg_words_per_step_label.keys(), 
                                            key=lambda x: int(x.split(" ")[1]))
        sorted_avg_values_overall = [overall_avg_words_per_step_label[label] for label in sorted_step_labels_overall]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(sorted_step_labels_overall, sorted_avg_values_overall, color='mediumseagreen', width=0.6)
        plt.xlabel("Step Label", fontsize=12)
        plt.ylabel("Overall Average Number of Words (All Datasets)", fontsize=12)
        plt.title("Overall Average Words per Step (All Datasets)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(sorted_avg_values_overall, default=1), f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

        overall_avg_hist_save_path = os.path.join(hist_output_dir, "overall_avg_words_per_step.png") # New filename
        plt.savefig(overall_avg_hist_save_path)
        plt.close()
        print(f"  Saved overall average per step histogram to {overall_avg_hist_save_path}")
    else:
        print("\nNo word count data collected to generate the overall average per step histogram.")

if __name__ == "__main__":
    generate_histograms()
    print("\n--- Histogram generation complete. ---")
