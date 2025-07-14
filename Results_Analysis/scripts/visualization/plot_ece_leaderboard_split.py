#!/usr/bin/env python3
"""
Plot two ECE "leaderboard" plots: one for baseline & CoT, one for P(True) & Self-Probing.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import load_yaml_with_imports


def main():
    import argparse
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Plot split ECE leaderboards')
    parser.add_argument('--config', type=str, default='configs/ece_config.yaml', help='Path to ECE config YAML')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    cfg = load_yaml_with_imports(config_path)
    datasets = [d for d in cfg.get('datasets', []) if d != '2WikimhQA']
    raw_metrics = cfg.get('metrics', [])
    # resolve metrics groups
    metrics = []
    for item in raw_metrics:
        if isinstance(item, str) and item.startswith('@'):
            grp = item[1:]
            entries = cfg.get(grp, [])
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict) and 'name' in e:
                        metrics.append(e['name'])
                    elif isinstance(e, str):
                        metrics.append(e)
        else:
            metrics.append(item)
    # de-dup
    seen = set()
    metrics = [m for m in metrics if not (m in seen or seen.add(m))]

    # group definitions
    group_keys = ['baseline_methods', 'cot_methods', 'ap_methods', 'true_probability_methods', 'self_probing_methods']
    groups = {g: [e['name'] for e in cfg.get(g, [])] for g in group_keys}
    fam_map = {m: g for g, ms in groups.items() for m in ms}
    families = list(groups.keys())
    runs = cfg.get('runs', [])

    # collect data
    rows = []
    results_dir = project_root / cfg.get('results_path', 'results/cot/ece')
    for ds in datasets:
        ds_dir = results_dir / ds
        agg_path = ds_dir / 'aggregated' / f"{ds}_ece.json"
        if not agg_path.exists():
            print(f"Skipping {ds}: missing {agg_path}")
            continue
        agg_data = json.loads(agg_path.read_text())
        for run in runs:
            run_file = ds_dir / f"run_{run}" / f"{ds}_ece.json"
            if not run_file.exists():
                continue
            run_data = json.loads(run_file.read_text())
            for m in metrics:
                if m not in run_data:
                    continue
                score = run_data[m].get('score', np.nan)
                stats = agg_data.get(m, {})
                mean = stats.get('mean_ece', np.nan)
                ci = stats.get('confidence_interval', [np.nan, np.nan])
                ci_low, ci_high = ci if len(ci)==2 else (np.nan, np.nan)
                fam = fam_map.get(m, 'other')
                rows.append({'dataset': ds, 'metric': m, 'score': score,
                             'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high,
                             'family': fam})
    if not rows:
        print("No data to plot.")
        return
    df = pd.DataFrame(rows)

    # simplified x-labels
    metric_labels = []
    for m in metrics:
        fam = fam_map.get(m, '')
        if fam in ['baseline_methods', 'cot_methods']:
            label = m[len('probas-'): ] if m.startswith('probas-') else m
            if label.endswith('-bl'):
                label = label[:-3]
        elif fam == 'true_probability_methods':
            label = m[len('p-true-'): ] if m.startswith('p-true-') else m
            if label == 'bl':
                label = 'baseline'
        elif fam == 'self_probing_methods':
            label = m[len('self-probing-'): ] if m.startswith('self-probing-') else m
            if label == 'bl':
                label = 'baseline'
        else:
            label = m
        metric_labels.append(label)

    # family palette
    pal = sns.color_palette('tab10', n_colors=len(families))
    fam_pal = dict(zip(families, pal))
    def infer_family(metric: str) -> str:
        if metric.startswith('p-true-'):
            return 'true_probability_methods'
        if metric.startswith('self-probing-'):
            return 'self_probing_methods'
        if metric.endswith('-bl'):
            return fam_map.get(metric, 'baseline_methods')
        if metric.endswith('-alltokens'):
            return 'ap_methods'
        if metric.startswith(('probas-', 'token-sar-')):
            return 'cot_methods'
        return fam_map.get(metric, 'other')

    # subsets: (identifier, families to include, title)
    subsets = [
        ('baseline_cot_alltokens', ['baseline_methods', 'cot_methods', 'ap_methods'], 'Baseline, CoT & AllTokens ECE Leaderboard'),
        ('ptrue_and_selfprobing', ['true_probability_methods', 'self_probing_methods'], 'P(True) & Self-Probing ECE Leaderboard')
    ]

    for name, fams, title in subsets:
        # subset metrics and labels
        metrics_sub = [m for m in metrics if fam_map.get(m) in fams]
        labels_sub = [lab for m, lab in zip(metrics, metric_labels) if m in metrics_sub]
        colors_sub = [fam_pal.get(fam_map.get(m), (0.5,0.5,0.5)) for m in metrics_sub]

        sns.set(style='whitegrid')
        n_datasets = len(datasets)
        n_cols = 3
        n_rows = (n_datasets + 1) // n_cols + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), sharey=False)
        axes = axes.flatten()
        for ax, ds in zip(axes, datasets):
            sub = df[df['dataset'] == ds]
            if sub.empty:
                continue
            present_metrics = [m for m in metrics_sub if not sub[sub['metric'] == m].empty]
            if not present_metrics:
                continue
            present_labels = [lab for m, lab in zip(metrics_sub, labels_sub) if m in present_metrics]
            present_colors = [fam_pal.get(infer_family(m), (0.5,0.5,0.5)) for m in present_metrics]
            scores = [sub[sub['metric'] == m]['score'].tolist() for m in present_metrics]
            parts = ax.violinplot(scores, positions=list(range(len(present_metrics))), widths=0.8, showextrema=False)
            for idx, pc in enumerate(parts['bodies']):
                pc.set_facecolor(present_colors[idx])
                pc.set_edgecolor('black')
                pc.set_alpha(0.8)
            sns.stripplot(x='metric', y='score', data=sub[sub['metric'].isin(present_metrics)],
                          order=present_metrics, color='black', size=3, jitter=True, ax=ax, alpha=0.6)
            for i, m in enumerate(present_metrics):
                msub = sub[sub['metric'] == m]
                if msub.empty:
                    continue
                mean = msub['mean'].iloc[0]
                low = msub['ci_low'].iloc[0]
                high = msub['ci_high'].iloc[0]
                ax.scatter(i, mean, color='black', marker='D', s=30, zorder=10)
                ax.vlines(i, low, high, color='black', linewidth=1)
            ax.axhline(0, linestyle='--', color='grey')
            ax.set_title(ds)
            ax.set_xlabel('')
            ax.set_xticks(range(len(present_metrics)))
            ax.set_xticklabels(present_labels, rotation=45, ha='right')
            ax.set_ylabel('ECE')

        legend_ax = axes[n_datasets]
        legend_ax.axis('off')
        # remove any unused axes after legend
        for ax in axes[n_datasets+1:]:
            ax.remove()
        label_map = {
            'baseline_methods': 'baseline',
            'cot_methods': 'CoT',
            'ap_methods': 'all tokens',
            'true_probability_methods': 'p(true)',
            'self_probing_methods': 'self-probing'
        }
        handles = [Patch(facecolor=fam_pal[f], label=label_map.get(f, f)) for f in fams]
        legend_ax.legend(handles=handles, loc='center')
        plt.tight_layout()

        results_base = Path(cfg.get('results_path', 'results/cot/ece')).parent
        if 'cod' in str(results_base):
            fig_dir = results_base / 'figures'
        else:
            fig_dir = Path('results/cod/figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / f"{name}_ece_leaderboard.png", dpi=300)
        print(f"Saved {title} to {fig_dir / f'{name}_ece_leaderboard.png'}")


if __name__ == '__main__':
    main()
