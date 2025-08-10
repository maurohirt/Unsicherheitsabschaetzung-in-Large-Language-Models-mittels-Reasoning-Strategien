import pandas as pd
import json
import os

def find_and_print_gsm8k_sample(target_step_count=9):
    base_dir = "/Users/maurohirt/Desktop/Bachelorarbeit/Outputs_BA/Word_Count_for_CoD/"
    csv_file_path = os.path.join(base_dir, "word_count_csv", "hotpotQA_word_counts.csv")
    json_file_path = os.path.join(base_dir, "data", "hotpotQA_output_v1.json")

    sample_id_to_find = None
    actual_step_column_found = None

    print(f"Attempting to read CSV: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print(f"CSV file {csv_file_path} is empty.")
            return
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV {csv_file_path}: {e}")
        return

    # Try to find a sample with the target_step_count
    # We look for columns from target_step_count down to 1
    for i in range(target_step_count, 0, -1):
        step_column_name = f"Step {i} Words"
        if step_column_name in df.columns:
            # Find rows where this step column is not NaN and greater than 0 (if numeric)
            # pd.to_numeric to handle cases where it might be read as object
            valid_entries = df[pd.to_numeric(df[step_column_name], errors='coerce').notna()]
            if not valid_entries.empty:
                sample_id_to_find = valid_entries.iloc[0]["id"]
                actual_step_column_found = step_column_name
                print(f"Found a sample in CSV with ID '{sample_id_to_find}' that has data in column '{actual_step_column_found}'.")
                break
    
    if not sample_id_to_find:
        print(f"Could not find any sample in {csv_file_path} with at least one step, or target step column '{f'Step {target_step_count} Words'}' (or lower) not found/empty.")
        # Fallback: try to find ANY sample with the maximum number of steps present in the CSV
        step_cols_present = [col for col in df.columns if col.startswith("Step ") and col.endswith(" Words")]
        if step_cols_present:
            max_step_col_in_csv = sorted(step_cols_present, key=lambda x: int(x.split(" ")[1]), reverse=True)[0]
            valid_entries = df[pd.to_numeric(df[max_step_col_in_csv], errors='coerce').notna()]
            if not valid_entries.empty:
                sample_id_to_find = valid_entries.iloc[0]["id"]
                actual_step_column_found = max_step_col_in_csv
                print(f"Fallback: Found a sample in CSV with ID '{sample_id_to_find}' that has data in column '{actual_step_column_found}' (max step column).")
            else:
                print(f"Fallback: Max step column '{max_step_col_in_csv}' has no valid entries.")
                return    
        else:
            print("No step columns found in the CSV at all.")
            return


    if not sample_id_to_find:
        print(f"No suitable sample ID found in {csv_file_path}.")
        return

    print(f"\nSearching for ID '{sample_id_to_find}' in JSON file: {json_file_path}")
    found_json_line = False
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f_json:
            for line_number, line in enumerate(f_json):
                try:
                    data = json.loads(line)
                    if str(data.get("id")) == str(sample_id_to_find): # Ensure ID comparison is robust
                        print(f"\n--- Original JSON Line for ID {sample_id_to_find} (from line {line_number + 1}) ---")
                        print(line.strip())
                        found_json_line = True
                        break 
                except json.JSONDecodeError:
                    # This might happen if there are non-JSON lines, though not expected for these files
                    print(f"Warning: JSON decode error on line {line_number + 1} of {json_file_path}")
                    continue
        if not found_json_line:
            print(f"ID '{sample_id_to_find}' not found in {json_file_path}.")
            
    except FileNotFoundError:
        print(f"JSON file not found: {json_file_path}")
    except Exception as e:
        print(f"Error reading JSON file {json_file_path}: {e}")

if __name__ == "__main__":
    # User specifically mentioned 8 steps for gsm8k
    find_and_print_gsm8k_sample(target_step_count=9)
