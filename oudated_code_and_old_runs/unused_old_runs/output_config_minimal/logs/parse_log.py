import re
import pandas as pd

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        log_content = f.readlines()

    data = []
    current_command_line = ""

    for line in log_content:
        if "Command:" in line and "analyze_result.py" in line:
            current_command_line = line.strip()
        elif "AUROC:" in line:
            auroc_score = float(line.split("AUROC:")[1].strip())
            
            # Extract Model Name
            model_match = re.search(r'--model_engine\s+([\w\d\.-]+)', current_command_line)
            model_name = model_match.group(1) if model_match else "N/A"

            # Extract Dataset Name
            dataset_match = re.search(r'--dataset\s+([\w\d\.-]+)', current_command_line)
            dataset_name = dataset_match.group(1) if dataset_match else "N/A"

            # Extract UQ Method Name
            uq_method_match = re.search(r'--uq_engine\s+([\w\d\.-]+)', current_command_line)
            uq_method_name = uq_method_match.group(1) if uq_method_match else "N/A"

            data.append({
                "Model Name": model_name,
                "Dataset Name": dataset_name,
                "UQ Method Name": uq_method_name,
                "AUROC Score": auroc_score
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    log_file = "/Users/maurohirt/Desktop/Bachelorarbeit/Outputs_BA/output_config_minimal/logs/pipeline/pipeline_72652_72652.out"
    df = parse_log_file(log_file)
    print(df.to_markdown(index=False))
    df.to_csv("/Users/maurohirt/Desktop/Bachelorarbeit/Outputs_BA/output_config_minimal/logs/auroc_scores.csv", index=False)
    print("\nAUROC scores saved to auroc_scores.csv")
