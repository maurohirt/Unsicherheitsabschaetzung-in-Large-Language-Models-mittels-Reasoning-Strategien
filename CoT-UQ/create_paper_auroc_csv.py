import pandas as pd
import os

def create_paper_auroc_csv(output_csv_path):
    data = []

    # Llama 3.1-8B data from the paper's table
    paper_data_llama3_1_8B = {
        'Probas-mean': {'HotpotQA': 53.73, '2WikimhQA': 56.80, 'GSM8K': 53.17, 'SVAMP': 53.94, 'ASDiv': 58.34},
        'Probas-mean w/ CoT-UQ': {'HotpotQA': 62.01, '2WikimhQA': 65.22, 'GSM8K': 63.64, 'SVAMP': 59.83, 'ASDiv': 64.52},
        'Probas-min': {'HotpotQA': 58.34, '2WikimhQA': 56.81, 'GSM8K': 54.95, 'SVAMP': 54.79, 'ASDiv': 58.69},
        'Probas-min w/ CoT-UQ': {'HotpotQA': 64.37, '2WikimhQA': 70.02, 'GSM8K': 63.09, 'SVAMP': 60.49, 'ASDiv': 64.84},
        'TOKENSAR': {'HotpotQA': 53.57, '2WikimhQA': 56.92, 'GSM8K': 54.46, 'SVAMP': 55.01, 'ASDiv': 58.71},
        'TOKENSAR w/ CoT-UQ': {'HotpotQA': 61.07, '2WikimhQA': 65.38, 'GSM8K': 65.10, 'SVAMP': 62.11, 'ASDiv': 66.91},
        'P(True)': {'HotpotQA': 62.39, '2WikimhQA': 53.56, 'GSM8K': 48.15, 'SVAMP': 51.58, 'ASDiv': 47.23},
        'P(True) w/ CoT-UQ': {'HotpotQA': 63.10, '2WikimhQA': 57.77, 'GSM8K': 52.60, 'SVAMP': 60.00, 'ASDiv': 53.20},
        'Self-Probing': {'HotpotQA': 54.33, '2WikimhQA': 56.39, 'GSM8K': 49.24, 'SVAMP': 51.63, 'ASDiv': 50.86},
        'Self-Probing w/ CoT-UQ': {'HotpotQA': 57.20, '2WikimhQA': 58.38, 'GSM8K': 51.89, 'SVAMP': 54.26, 'ASDiv': 53.79},
    }

    logical_datasets = ["HotpotQA", "2WikimhQA"]

    for method, datasets_aurocs in paper_data_llama3_1_8B.items():
        strategy = 'AP' if method.startswith(('Probas', 'TOKENSAR')) else 'SE'
        for dataset, auroc_percent in datasets_aurocs.items():
            reasoning_type = 'logical' if dataset in logical_datasets else 'mathematical'
            data.append({
                'model_engine': 'llama3-1_8B',
                'dataset': dataset,
                'reasoning_type': reasoning_type,
                'strategy': strategy,
                'UQ_method': method, # Using paper's method name directly
                'AUROC': auroc_percent / 100.0 # Convert percentage to decimal
            })

    df_paper = pd.DataFrame(data)

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_paper.to_csv(output_csv_path, index=False)
    print(f"Paper AUROC scores saved to {output_csv_path}")
    print(df_paper.head())

if __name__ == "__main__":
    output_file = "/Users/maurohirt/Desktop/Bachelorarbeit/Outputs_BA/auroc_scores_paper.csv"
    create_paper_auroc_csv(output_file)
