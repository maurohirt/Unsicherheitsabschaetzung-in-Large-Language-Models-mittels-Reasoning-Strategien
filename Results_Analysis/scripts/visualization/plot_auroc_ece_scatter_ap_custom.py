#!/usr/bin/env python3
"""
Custom AP AUROC vs ECE scatter: colored by metric pairs (baseline light/dark), shape by dataset.
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
    # CLI for custom config paths
    parser = argparse.ArgumentParser(description='AP AUROC vs ECE scatter (custom)')
    parser.add_argument('--auroc_cfg', type=str, default='configs/auroc_config_cod.yaml')
    parser.add_argument('--ece_cfg', type=str, default='configs/ece_config_cod.yaml')
    args = parser.parse_args()

    auroc_cfg = load_yaml_with_imports(project_root / args.auroc_cfg)
    ece_cfg = load_yaml_with_imports(project_root / args.ece_cfg)
    datasets = auroc_cfg.get('datasets', [])
    groups = {g: [m['name'] for m in ece_cfg.get(g, [])] for g in ['baseline_methods', 'cot_methods', 'ap_methods']}
    # include baseline, cot and all-token variants
    ap_metrics = groups.get('baseline_methods', []) + groups.get('cot_methods', []) + groups.get('ap_methods', [])

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
        for m in ap_metrics:
            if m in a and m in e:
                rows.append({
                    'dataset': ds,
                    'metric': m,
                    'mean_auroc': a[m].get('mean_auroc', np.nan),
                    'mean_ece': e[m].get('mean_ece', np.nan)
                })
    df = pd.DataFrame(rows)
    # define stats ordering
    stats = ['probas-mean', 'probas-min', 'token-sar']
    metrics_order = [f"{s}-bl" for s in stats if f"{s}-bl" in ap_metrics] + [s for s in stats if s in ap_metrics]
    # marker map per dataset
    marker_list = ['o', 's', '^', 'D', 'v']
    marker_map = {ds: marker_list[i] for i, ds in enumerate(datasets)}
    # color map
    pal = sns.color_palette('tab10', n_colors=len(stats))
    base_colors = {s: pal[i] for i, s in enumerate(stats)}
    color_map = {}
    for s in stats:
        color_map[f"{s}-bl"] = lighten(base_colors[s], factor=0.5)
        color_map[s] = base_colors[s]

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
        label = m.replace('probas-', '').replace('-bl', '') + (" (baseline)" if m.endswith('-bl') else "")
        h = Line2D([0], [0], marker='o', color=color_map.get(m, 'grey'), linestyle='None', markersize=8, label=label)
        metric_handles.append(h)
    # dataset legend drawn first (top)
    dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='black', linestyle='None', markersize=8, label=ds)
                       for ds in datasets]
    leg_ds = plt.legend(handles=dataset_handles,
                        title='Dataset',
                        bbox_to_anchor=(1.05, 1),
                        loc='upper left')
    plt.gca().add_artist(leg_ds)

    # method legend drawn second (below dataset)
    plt.legend(handles=metric_handles,
               title='Method (n=5 runs)',
               bbox_to_anchor=(1.05, 0.6),
               loc='upper left')

    plt.xlabel('Mean ECE')
    plt.ylabel('Mean AUROC')
    plt.title('Aggregated Probabilities AUROC vs ECE')
    plt.tight_layout()

    out_dir = project_root / 'results' / 'cod' / 'figures' / 'scatter'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'auroc_vs_ece_ap_custom.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved custom AP scatter to {out_file}")
    plt.close()


if __name__ == '__main__':
    main()
