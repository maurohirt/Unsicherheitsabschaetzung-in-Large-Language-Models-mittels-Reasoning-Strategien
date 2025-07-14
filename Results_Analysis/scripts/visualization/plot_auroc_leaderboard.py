#!/usr/bin/env python3
"""
Plot an AUROC “leaderboard” across datasets: violin distributions of run-wise AUROC,
mean diamonds with CIs, and jittered points showing raw variability.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ensure project root on sys.path for src imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_yaml_with_imports


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='AUROC leaderboard (CoD)')
    parser.add_argument('--cfg', type=str, default='configs/auroc_config_cod.yaml')
    args = parser.parse_args()

    cfg = load_yaml_with_imports(project_root / args.cfg)
    datasets = cfg.get('datasets', [])

    metrics = cfg.get('metrics', [])
    # group definitions from imported UQ config
    group_keys = ['baseline_methods', 'cot_methods', 'ap_methods', 'true_probability_methods', 'self_probing_methods']
    groups = {g: [m['name'] for m in cfg.get(g, [])] for g in group_keys}
    # family map
    fam_map = {m: g for g, ms in groups.items() for m in ms}
    families = list(groups.keys())

    # load aggregated AUROC JSONs
    results_dir = project_root / cfg.get('results_path', 'results/cod/auroc') / 'aggregated'
    # derive metrics from aggregated JSON if not set
    if not metrics:
        first_ds = next((d for d in datasets), None)
        if first_ds:
            path = results_dir / f"{first_ds}_auroc.json"
            if path.exists():
                metrics = list(json.loads(path.read_text()).keys())
        # simplified x-axis labels by stripping prefixes
    # simplified x-axis labels: strip prefixes/suffixes
    metric_labels = []
    for m in metrics:
        fam = fam_map.get(m, '')
        if fam in ['baseline_methods', 'cot_methods']:
            # remove 'probas-' prefix
            if m.startswith('probas-'):
                label = m[len('probas-'):]
            else:
                label = m
            # remove '-bl' suffix
            if label.endswith('-bl'):
                label = label[:-3]
        elif fam == 'true_probability_methods':
            # remove 'p-true-' prefix
            if m.startswith('p-true-'):
                label = m[len('p-true-'):]
            else:
                label = m
            # rename 'bl' to 'baseline'
            if label == 'bl':
                label = 'baseline'
        elif fam == 'self_probing_methods':
            # remove 'self-probing-' prefix
            if m.startswith('self-probing-'):
                label = m[len('self-probing-'):]
            else:
                label = m
            # rename 'bl' to 'baseline'
            if label == 'bl':
                label = 'baseline'
        else:
            label = m
        metric_labels.append(label)
    rows = []
    for ds in datasets:
        path = results_dir / f"{ds}_auroc.json"
        if not path.exists():
            print(f"Warning: missing {path}")
            continue
        data = json.loads(path.read_text())
        for metric, stats in data.items():
            indiv = stats.get('individual_scores', [])
            ci = stats.get('confidence_interval', [np.nan, np.nan])
            mean = stats.get('mean_auroc', np.nan)
            fam = fam_map.get(metric, 'other')
            for score in indiv:
                rows.append({'dataset': ds, 'metric': metric,
                             'score': score, 'mean': mean,
                             'ci_low': ci[0], 'ci_high': ci[1],
                             'family': fam})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data to plot.")
        return

    # palette for families
    pal = sns.color_palette('tab10', n_colors=len(families))
    fam_pal = dict(zip(families, pal))
    # colors per metric in order
    col_list = [fam_pal.get(fam_map.get(m), (0.5,0.5,0.5)) for m in metrics]

    sns.set(style='whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(12, 12), sharey=True)
    axes = axes.flatten()
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds]
        # violin of distributions (manual)
        dataset_scores = [sub[sub['metric'] == m]['score'].tolist() for m in metrics]
        parts = ax.violinplot(dataset_scores, positions=list(range(len(metrics))), widths=0.8, showextrema=False)
        # color each violin body
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(col_list[idx])
            pc.set_edgecolor('black')
            pc.set_alpha(0.8)
        # jittered points
        sns.stripplot(x='metric', y='score', data=sub,
                      order=metrics, color='black', size=3,
                      jitter=True, ax=ax, alpha=0.6)
        # mean diamond and CI
        for i, m in enumerate(metrics):
            msub = sub[sub['metric'] == m]
            if msub.empty:
                continue
            mean = msub['mean'].iloc[0]
            low = msub['ci_low'].iloc[0]
            high = msub['ci_high'].iloc[0]
            ax.scatter(i, mean, color='black', marker='D', zorder=10)
            ax.vlines(i, low, high, color='black', linewidth=1)

        # random baseline
        ax.axhline(0.5, linestyle='--', color='grey')
        ax.set_title(ds)
        ax.set_xlabel('')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    # set Y-axis label for dataset panels
    for ax in axes[:-1]:
        ax.set_ylabel('AUROC')
    # legend panel
    legend_ax = axes[-1]
    legend_ax.axis('off')
    # custom legend labels
    label_map = {
        'baseline_methods': 'baseline',
        'cot_methods': 'CoD (std tokens)',
        'ap_methods': 'AP all-tokens',
        'true_probability_methods': 'p(true)',
        'self_probing_methods': 'self-probing'
    }
    handles = [Patch(facecolor=fam_pal[family], label=label_map.get(family, family)) for family in families]

    legend_ax.legend(handles=handles, loc='center')
    plt.tight_layout()
    # save figure under results/cot/figures
    out = project_root / Path(cfg.get('results_path', 'results/cod/auroc')).parent / 'figures' / 'auroc_leaderboard.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"Saved AUROC leaderboard to {out}")
    print(f"Saved AUROC leaderboard to {out}")


if __name__ == '__main__':
    main()
