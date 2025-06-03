import re
import json
import csv
import os
import glob

def count_words_in_steps(llm_response_content):
    """
    Counts words in each step of an LLM response string.
    Steps are identified by "Step X:" and the counting excludes "Final Answer:".
    Returns a dictionary with step labels as keys and word counts as values.
    """
    if not isinstance(llm_response_content, str):
        return {}

    final_answer_marker = "Final Answer:"
    if final_answer_marker in llm_response_content:
        response_before_final_answer = llm_response_content.split(final_answer_marker, 1)[0].strip()
    else:
        response_before_final_answer = llm_response_content.strip()

    step_pattern = re.compile(r'(Step \d+):\s*')
    matches = list(step_pattern.finditer(response_before_final_answer))
    
    step_word_counts = {}

    for i, match in enumerate(matches):
        step_label = match.group(1) 
        content_start_index = match.end()
        
        if i + 1 < len(matches):
            content_end_index = matches[i + 1].start()
        else:
            content_end_index = len(response_before_final_answer)
            
        step_text = response_before_final_answer[content_start_index:content_end_index].strip()
        words = step_text.split()
        word_count = len(words)
        
        step_word_counts[f"{step_label} Words"] = word_count 
                                                    
    return step_word_counts

# --- Main part of the script ---
if __name__ == "__main__":
    base_dir = "/Users/maurohirt/Desktop/Bachelorarbeit/Outputs_BA/Word_Count_for_CoD/"
    data_dir = os.path.join(base_dir, "data")
    output_csv_dir = os.path.join(base_dir, "word_count_csv") # New output directory

    # Create the output CSV directory if it doesn't exist
    os.makedirs(output_csv_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}. Exiting.")
        exit()

    for json_file_path in json_files:
        file_basename = os.path.basename(json_file_path)
        file_name_stem = os.path.splitext(file_basename)[0]
        # Remove '_output_v1' if present
        cleaned_file_name_stem = file_name_stem.replace("_output_v1", "")
        
        csv_file_path = os.path.join(output_csv_dir, f"{cleaned_file_name_stem}_word_counts.csv")

        all_results_for_file = []
        all_step_headers_for_file = set()

        print(f"\n--- Processing file: {json_file_path} ---")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f_json:
                for line_number, line in enumerate(f_json):
                    try:
                        sample_data = json.loads(line)
                        llm_response = sample_data.get("llm response") 
                        sample_id = sample_data.get("id", f"unknown_id_{line_number+1}")
                        question = sample_data.get("question", "")

                        record = {
                            "id": sample_id,
                            "question": question
                        }

                        if llm_response:
                            step_counts = count_words_in_steps(llm_response)
                            record.update(step_counts)
                            for step_key in step_counts.keys():
                                all_step_headers_for_file.add(step_key)
                        
                        all_results_for_file.append(record)
                            
                    except json.JSONDecodeError:
                        print(f"  Warning: Could not decode JSON on line {line_number+1} in {os.path.basename(json_file_path)}. Skipping line.")
                        all_results_for_file.append({
                            "id": f"decode_error_line_{line_number+1}",
                            "question": "JSON DECODE ERROR",
                        })
                        continue
            
            if not all_results_for_file:
                print(f"  No data processed from {os.path.basename(json_file_path)}.")
                continue

            sorted_step_headers = []
            if all_step_headers_for_file: 
                 sorted_step_headers = sorted(list(all_step_headers_for_file), 
                                             key=lambda x: int(re.search(r'Step (\d+) Words', x).group(1)) if re.search(r'Step (\d+) Words', x) else float('inf'))
            
            fieldnames = ["id", "question"] + sorted_step_headers

            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_results_for_file)
            
            print(f"  Processing complete for {os.path.basename(json_file_path)}.")
            print(f"  Word counts saved to: {csv_file_path}")
            print(f"  Processed {len(all_results_for_file)} records.")
            if all_results_for_file:
                 print(f"  CSV Headers: {fieldnames}")
                 if len(all_results_for_file) >0:
                    print(f"  First record example: {all_results_for_file[0]}")


        except FileNotFoundError:
            print(f"  Error: Input file not found at {json_file_path} (this should not happen if glob found it).")
        except Exception as e:
            print(f"  An unexpected error occurred while processing {os.path.basename(json_file_path)}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- All files processed. ---")
