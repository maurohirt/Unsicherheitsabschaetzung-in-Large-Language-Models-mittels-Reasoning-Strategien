#!/usr/bin/env python3
"""
Custom P(True) AUROC vs ECE scatter: colored by baseline light/dark, shape by dataset, legend includes n=5 runs.
"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

# ensure project root on sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.utils.config_loader import load_yaml_with_imports

def lighten(color, factor=0.5):
    c = np.array(to_rgb(color))
    return tuple(c + (1 - c) * factor)


import argparse

def main():
    parser = argparse.ArgumentParser(description='P(True) AUROC vs ECE scatter')
    parser.add_argument('--auroc_cfg', type=str, default='configs/auroc_config_cod.yaml')
    parser.add_argument('--ece_cfg', type=str, default='configs/ece_config_cod.yaml')
    args = parser.parse_args()

    auroc_cfg = load_yaml_with_imports(project_root / args.auroc_cfg)
    ece_cfg = load_yaml_with_imports(project_root / args.ece_cfg)
    datasets = auroc_cfg.get('datasets', [])
    ptrue_group = ece_cfg.get('true_probability_methods', [])
    metrics_order = [m['name'] for m in ptrue_group]

    # load aggregated data
    rows = []
    auroc_dir = project_root / auroc_cfg.get('results_path', 'results/cod/auroc') / 'aggregated'
    ece_dir = project_root / ece_cfg.get('results_path', 'results/cod/ece')
    for ds in datasets:
        auroc_file = auroc_dir / f"{ds}_auroc.json"
        ece_file = ece_dir / ds / 'aggregated' / f"{ds}_ece.json"
        if not auroc_file.exists() or not ece_file.exists():
            continue
        a = json.loads(auroc_file.read_text())
        e = json.loads(ece_file.read_text())
        for m in metrics_order:
            if m in a and m in e:
                rows.append({
                    'dataset': ds,
                    'metric': m,
                    'mean_auroc': a[m].get('mean_auroc', np.nan),
                    'mean_ece': e[m].get('mean_ece', np.nan)
                })
    df = pd.DataFrame(rows)

    # marker for each dataset
    marker_list = ['o', 's', '^', 'D', 'v']
    marker_map = {ds: marker_list[i % len(marker_list)] for i, ds in enumerate(datasets)}
    # color scheme for P(True) variants and baseline
    variants = [m for m in metrics_order if not m.endswith('-bl')]
    pal = sns.color_palette('tab10', n_colors=len(variants))
    color_map = {}
    # assign distinct colors to variants
    for idx, m in enumerate(variants):
        color_map[m] = pal[idx]
    # baseline in black
    for m in metrics_order:
        if m.endswith('-bl'):
            color_map[m] = 'black'

    # plot
    plt.figure(figsize=(6, 6))
    sns.set(style='whitegrid')
    for m in metrics_order:
        for ds in datasets:
            sub = df[(df['metric'] == m) & (df['dataset'] == ds)]
            if sub.empty:
                continue
            x = sub['mean_ece'].values[0]
            y = sub['mean_auroc'].values[0]
            plt.scatter(x, y, marker=marker_map[ds], color=color_map[m], s=100)
    # metric legend
    metric_handles = []
    for m in metrics_order:
        label = m.replace('p-true-', '').replace('-bl', '') + (' (baseline)' if m.endswith('-bl') else '')
        h = Line2D([0], [0], marker='o', color=color_map[m], linestyle='None', markersize=8, label=label)
        metric_handles.append(h)
    leg1 = plt.legend(handles=metric_handles, title='Method (n=5 runs)', bbox_to_anchor=(1.05, 1), loc='upper left')
    # dataset legend
    dataset_handles = []
    for ds in datasets:
        h = Line2D([0], [0], marker=marker_map[ds], color='black', linestyle='None', markersize=8, label=ds)
        dataset_handles.append(h)
    plt.gca().add_artist(leg1)
    plt.legend(handles=dataset_handles, title='Dataset', bbox_to_anchor=(1.05, 0.6), loc='upper left')

    plt.xlabel('Mean ECE')
    plt.ylabel('Mean AUROC')
    plt.title('P(True) AUROC vs ECE')
    plt.tight_layout()

    out_dir = project_root / 'results' / 'cod' / 'figures' / 'scatter'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'auroc_vs_ece_ptrue_custom.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved custom P(True) scatter to {out_file}")
    plt.close()

if __name__ == '__main__':
    main()
