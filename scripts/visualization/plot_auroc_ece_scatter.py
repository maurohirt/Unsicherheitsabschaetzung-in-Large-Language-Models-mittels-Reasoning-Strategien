#!/usr/bin/env python3
"""
Plot AUROC vs ECE scatter plots: one for AP (baseline & CoT), one for SE (p-true & self-probing).
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import load_yaml_with_imports


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    project_root = Path(__file__).resolve().parents[2]
    # load configs
    auroc_cfg = load_yaml_with_imports(project_root / 'configs' / 'auroc_config.yaml')
    ece_cfg = load_yaml_with_imports(project_root / 'configs' / 'ece_config.yaml')
    datasets = auroc_cfg.get('datasets', [])

    # group definitions
    group_keys = ['baseline_methods', 'cot_methods', 'true_probability_methods', 'self_probing_methods']
    groups = {g: [m['name'] for m in ece_cfg.get(g, [])] for g in group_keys}

    # define metric subsets
    ap_metrics = groups['baseline_methods'] + groups['cot_methods']
    se_metrics = groups['true_probability_methods'] + groups['self_probing_methods']

    # load and merge aggregated metrics
    rows = []
    # auroc aggregated dir
    auroc_dir = project_root / auroc_cfg.get('results_path', 'results/cot/auroc') / 'aggregated'
    # ece aggregated dir
    ece_dir = project_root / ece_cfg.get('results_path', 'results/cot/ece')

    for ds in datasets:
        auroc_file = auroc_dir / f"{ds}_auroc.json"
        ece_file = ece_dir / ds / 'aggregated' / f"{ds}_ece.json"
        if not auroc_file.exists() or not ece_file.exists():
            print(f"Skipping {ds}: missing aggregated files")
            continue
        auroc_data = load_json(auroc_file)
        ece_data = load_json(ece_file)
        for m in set(ap_metrics + se_metrics):
            if m in auroc_data and m in ece_data:
                mean_auroc = auroc_data[m].get('mean_auroc', np.nan)
                mean_ece = ece_data[m].get('mean_ece', np.nan)
                rows.append({'dataset': ds, 'metric': m,
                             'mean_auroc': mean_auroc, 'mean_ece': mean_ece})
    if not rows:
        print("No data to plot.")
        return
    df = pd.DataFrame(rows)

    # create scatter plots
    subsets = [
        ('ap', ap_metrics, 'Aggregated Probabilities (Baseline & CoT)'),
        ('se', se_metrics, 'P(True) & Self-Probing')
    ]
    for key, metrics_sub, title in subsets:
        df_sub = df[df['metric'].isin(metrics_sub)]
        plt.figure(figsize=(6, 6))
        sns.set(style='whitegrid')
        sns.scatterplot(data=df_sub, x='mean_auroc', y='mean_ece', hue='metric', s=100, palette='tab10')
        plt.xlabel('Mean AUROC')
        plt.ylabel('Mean ECE')
        plt.title(f'{title} AUROC vs ECE')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Method', fontsize=8)
        plt.tight_layout()
        out_dir = project_root / 'results' / 'cot' / 'figures' / 'scatter'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'auroc_vs_ece_{key}.png'
        plt.savefig(out_file, dpi=300)
        print(f"Saved scatter plot: {out_file}")
        plt.close()

if __name__ == '__main__':
    main()
